# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference network used by the frozen DoorMan HOMIE standing policy.

This is the checkpoint-compatible subset of
``gr00t.rl.trl.modules.homie_modules`` needed to load ``model_stand.pt``.
Training-only estimator updates and PPO distribution helpers are intentionally
omitted; the door task always runs this policy in inference mode.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    """Construct the activation used by the original HOMIE checkpoint."""
    activations: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    try:
        return activations[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported HOMIE activation: {name}") from exc


class HIMEstimator(nn.Module):
    """Checkpoint-compatible latent velocity estimator."""

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        num_height_points: int,
        enc_hidden_dims: tuple[int, ...] = (256, 256),
        tar_hidden_dims: tuple[int, ...] = (256, 256),
        latent_dim: int = 32,
        num_prototype: int = 64,
    ) -> None:
        super().__init__()
        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs

        encoder_dims = (
            temporal_steps * num_one_step_obs + num_height_points,
            *enc_hidden_dims,
            3 + latent_dim,
        )
        self.encoder = self._build_mlp(encoder_dims, "elu", final_activation=False)

        target_dims = (num_one_step_obs, *tar_hidden_dims, latent_dim)
        self.target = self._build_mlp(target_dims, "elu", final_activation=False)
        self.proto = nn.Embedding(num_prototype, latent_dim)

    @staticmethod
    def _build_mlp(
        dimensions: tuple[int, ...],
        activation: str,
        *,
        final_activation: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:], strict=True)
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if final_activation or index < len(dimensions) - 2:
                layers.append(_activation(activation))
        return nn.Sequential(*layers)

    def forward(self, observation_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.encoder(observation_history.detach())
        velocity, latent = parts[..., :3], parts[..., 3:]
        return velocity.detach(), F.normalize(latent, dim=-1, p=2).detach()


class HIMActorCritic(nn.Module):
    """Network skeleton matching every parameter stored in ``model_stand.pt``."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_one_step_obs: int,
        num_one_step_critic_obs: int,
        actor_history_length: int,
        critic_history_length: int,
        num_actions: int = 15,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 256),
        critic_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_one_step_obs = num_one_step_obs
        self.actor_history_length = actor_history_length
        self.actor_proprioceptive_obs_length = actor_history_length * num_one_step_obs
        self.critic_proprioceptive_obs_length = (
            critic_history_length * num_one_step_critic_obs
        )
        self.num_height_points = num_actor_obs - self.actor_proprioceptive_obs_length
        self.actor_use_height = self.num_height_points > 0

        dynamic_latent_dim = 32
        terrain_latent_dim = 32
        actor_input_dim = num_one_step_obs + 3 + dynamic_latent_dim
        if self.actor_use_height:
            actor_input_dim += terrain_latent_dim

        self.estimator = HIMEstimator(
            temporal_steps=actor_history_length,
            num_one_step_obs=num_one_step_obs,
            num_height_points=0,
            latent_dim=dynamic_latent_dim,
        )

        if self.actor_use_height:
            self.terrain_encoder = nn.Sequential(
                nn.Linear(num_one_step_obs + self.num_height_points, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, terrain_latent_dim),
            )

        self.actor = self._build_policy_mlp(
            actor_input_dim,
            tuple(actor_hidden_dims),
            num_actions,
            activation,
        )
        self.critic = self._build_policy_mlp(
            num_critic_obs,
            tuple(critic_hidden_dims),
            1,
            activation,
        )
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

    @staticmethod
    def _build_policy_mlp(
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        activation: str,
    ) -> nn.Sequential:
        dimensions = (input_dim, *hidden_dims, output_dim)
        layers: list[nn.Module] = []
        for index, (layer_input, layer_output) in enumerate(
            zip(dimensions[:-1], dimensions[1:], strict=True)
        ):
            layers.append(nn.Linear(layer_input, layer_output))
            if index < len(dimensions) - 2:
                layers.append(_activation(activation))
        return nn.Sequential(*layers)


class HomieActorModule(nn.Module):
    """Deterministic inference view of the actor and estimator."""

    def __init__(self, full_model: HIMActorCritic) -> None:
        super().__init__()
        self.actor = full_model.actor
        self.estimator = full_model.estimator
        self.terrain_encoder = (
            full_model.terrain_encoder if full_model.actor_use_height else None
        )
        self.actor_use_height = full_model.actor_use_height
        self.num_one_step_obs = full_model.num_one_step_obs
        self.num_height_points = full_model.num_height_points
        self.std = full_model.std

    def forward(self, observation_history: torch.Tensor) -> dict[str, torch.Tensor]:
        velocity, dynamic_latent = self.estimator(
            observation_history[..., : self.num_one_step_obs * 6]
        )
        if self.actor_use_height and self.terrain_encoder is not None:
            terrain_input = observation_history[
                ..., -(self.num_one_step_obs + self.num_height_points) :
            ]
            terrain_latent = self.terrain_encoder(terrain_input)
            current_observation = observation_history[
                ...,
                -(self.num_one_step_obs + self.num_height_points) : -self.num_height_points,
            ]
            actor_input = torch.cat(
                (current_observation, velocity, dynamic_latent, terrain_latent), dim=-1
            )
        else:
            actor_input = torch.cat(
                (
                    observation_history[..., -self.num_one_step_obs :],
                    velocity,
                    dynamic_latent,
                ),
                dim=-1,
            )

        action_mean = self.actor(actor_input)
        return {
            "actions": action_mean,
            "action_mean": action_mean,
            "action_sigma": self.std.expand_as(action_mean),
        }


init_actor_critic_dict = {
    "num_actor_obs": 516,
    "num_critic_obs": 89,
    "num_one_step_obs": 86,
    "num_one_step_critic_obs": 89,
    "actor_history_length": 6,
    "critic_history_length": 1,
    "num_actions": 15,
    "actor_hidden_dims": [512, 256, 256],
    "critic_hidden_dims": [512, 256, 256],
    "activation": "elu",
    "init_noise_std": 0.0,
}
