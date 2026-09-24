"""Frozen HOMIE controller used for free-base standing."""

from pathlib import Path
from typing import Sequence

import torch

from gr00t.rl.trl.modules.homie_modules import (
    HIMActorCritic,
    HomieActorModule,
    init_actor_critic_dict,
)

from .homie_cfg import (
    HOMIE_DEFAULT_JOINT_POS,
    HOMIE_OBS_JOINT_ORDER,
    HomieControllerCfg,
)


class HomieController:
    """Maintain HOMIE history and produce lower-body joint targets."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        cfg: HomieControllerCfg | None = None,
    ) -> None:
        self.cfg = cfg or HomieControllerCfg()
        self.num_envs = num_envs
        self.device = torch.device(device)

        checkpoint_path = Path(self.cfg.checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"HOMIE checkpoint not found: {checkpoint_path}"
            )

        if init_actor_critic_dict["num_one_step_obs"] != self.cfg.frame_dim:
            raise RuntimeError(
                "HOMIE frame dimension does not match checkpoint metadata."
            )

        if init_actor_critic_dict["actor_history_length"] != self.cfg.history_length:
            raise RuntimeError(
                "HOMIE history length does not match checkpoint metadata."
            )

        if init_actor_critic_dict["num_actions"] != self.cfg.num_actions:
            raise RuntimeError(
                "HOMIE action dimension does not match checkpoint metadata."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                "HOMIE checkpoint has no model_state_dict."
            )

        # Legacy HOMIE assigns Normal.set_default_validate_args = False, which
        # destroys the callable needed by RSL-RL. Isolate that constructor's
        # global side effect without changing upstream code or policy weights.
        normal_type = torch.distributions.Normal
        previous_descriptor = normal_type.__dict__.get("set_default_validate_args")
        try:
            full_model = HIMActorCritic(**init_actor_critic_dict)
        finally:
            if previous_descriptor is None:
                if "set_default_validate_args" in normal_type.__dict__:
                    delattr(normal_type, "set_default_validate_args")
            else:
                normal_type.set_default_validate_args = previous_descriptor
        full_model.load_state_dict(state_dict)

        self.policy = HomieActorModule(full_model).to(self.device)
        self.policy.eval()

        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

        default_joint_pos = torch.tensor(
            HOMIE_DEFAULT_JOINT_POS,
            dtype=torch.float32,
            device=self.device,
        )

        self._default_joint_pos = default_joint_pos.unsqueeze(0).repeat(
            self.num_envs,
            1,
        )

        self._command = torch.zeros(
            self.num_envs,
            7,
            dtype=torch.float32,
            device=self.device,
        )
        self._command[:, 3] = self.cfg.default_height

        self._history = torch.zeros(
            self.num_envs,
            self.cfg.history_length,
            self.cfg.frame_dim,
            dtype=torch.float32,
            device=self.device,
        )

        self._last_action = torch.zeros(
            self.num_envs,
            self.cfg.num_actions,
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def default_joint_pos(self) -> torch.Tensor:
        """Default positions in HOMIE's 29-joint observation order."""

        return self._default_joint_pos

    @property
    def last_action(self) -> torch.Tensor:
        """Most recent raw 15-dimensional HOMIE action."""

        return self._last_action

    @property
    def observation_history(self) -> torch.Tensor:
        """Flattened six-frame HOMIE observation."""

        return self._history.reshape(self.num_envs, -1)

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Clear history and previous actions for selected environments."""

        if env_ids is None:
            self._history.zero_()
            self._last_action.zero_()
            self._command.zero_()
            self._command[:, 3] = self.cfg.default_height
            return

        env_ids_tensor = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._history[env_ids_tensor] = 0.0
        self._last_action[env_ids_tensor] = 0.0
        self._command[env_ids_tensor] = 0.0
        self._command[env_ids_tensor, 3] = self.cfg.default_height

    def _check_input(
        self,
        name: str,
        value: torch.Tensor,
        width: int,
    ) -> None:
        expected_shape = (self.num_envs, width)

        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {tuple(value.shape)}."
            )

        if value.device != self.device:
            raise ValueError(
                f"{name} is on {value.device}, expected {self.device}."
            )

    def build_frame(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        base_ang_vel: torch.Tensor,
        projected_gravity: torch.Tensor,
    ) -> torch.Tensor:
        """Build one 86-dimensional observation frame."""

        self._check_input(
            "joint_pos",
            joint_pos,
            len(HOMIE_OBS_JOINT_ORDER),
        )
        self._check_input(
            "joint_vel",
            joint_vel,
            len(HOMIE_OBS_JOINT_ORDER),
        )
        self._check_input("base_ang_vel", base_ang_vel, 3)
        self._check_input("projected_gravity", projected_gravity, 3)

        frame = torch.cat(
            (
                self._command,
                base_ang_vel * 0.5,
                projected_gravity,
                joint_pos - self._default_joint_pos,
                joint_vel * 0.05,
                self._last_action,
            ),
            dim=-1,
        )

        if frame.shape != (self.num_envs, self.cfg.frame_dim):
            raise RuntimeError(
                f"Constructed HOMIE frame has shape {tuple(frame.shape)}."
            )

        if not torch.isfinite(frame).all():
            raise RuntimeError("HOMIE observation contains NaN or Inf.")

        return frame

    def compute_joint_targets(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        base_ang_vel: torch.Tensor,
        projected_gravity: torch.Tensor,
    ) -> torch.Tensor:
        """Run HOMIE and return targets for the 15 leg/waist joints."""

        frame = self.build_frame(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
        )

        self._history[:, :-1].copy_(
            self._history[:, 1:].clone()
        )
        self._history[:, -1].copy_(frame)

        with torch.inference_mode():
            output = self.policy(self.observation_history)
            action = output["actions"]

        if action.shape != (self.num_envs, self.cfg.num_actions):
            raise RuntimeError(
                f"HOMIE returned action shape {tuple(action.shape)}."
            )

        if not torch.isfinite(action).all():
            raise RuntimeError("HOMIE action contains NaN or Inf.")

        self._last_action.copy_(action)

        return (
            self._default_joint_pos[:, : self.cfg.num_actions]
            + self.cfg.action_scale * action
        )
