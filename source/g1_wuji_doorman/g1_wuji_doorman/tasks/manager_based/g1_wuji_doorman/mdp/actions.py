# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action terms for the G1+Wuji DoorMan task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from g1_wuji_doorman.assets.robots import (
    DOORMAN_ALL_DOF_NAMES,
    DOORMAN_LEFT_ARM_DOF_NAMES,
    DOORMAN_LEFT_HAND_DOF_NAMES,
    DOORMAN_LOWER_BODY_DOF_NAMES,
    DOORMAN_RIGHT_ARM_DOF_NAMES,
    DOORMAN_RIGHT_HAND_DOF_NAMES,
)
from g1_wuji_doorman.controllers.hand import (
    HandController,
    PrimitiveHandController,
    WUJI_HAND_REST_POSE,
)
from g1_wuji_doorman.controllers.standing import (
    HOMIE_OBS_JOINT_ORDER,
    HomieController,
    HomieControllerCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_LOWER_BODY_END = len(DOORMAN_LOWER_BODY_DOF_NAMES)
_LEFT_ARM_END = _LOWER_BODY_END + len(DOORMAN_LEFT_ARM_DOF_NAMES)
_RIGHT_ARM_END = _LEFT_ARM_END + len(DOORMAN_RIGHT_ARM_DOF_NAMES)
_LEFT_HAND_END = _RIGHT_ARM_END + len(DOORMAN_LEFT_HAND_DOF_NAMES)
_RIGHT_HAND_END = _LEFT_HAND_END + len(DOORMAN_RIGHT_HAND_DOF_NAMES)

LOWER_BODY_SLICE = slice(0, _LOWER_BODY_END)
LEFT_ARM_SLICE = slice(_LOWER_BODY_END, _LEFT_ARM_END)
LEFT_HAND_SLICE = slice(_RIGHT_ARM_END, _LEFT_HAND_END)
RIGHT_HAND_SLICE = slice(_LEFT_HAND_END, _RIGHT_HAND_END)


def compose_doorman_joint_targets(
    default_joint_targets: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    left_arm_actions: torch.Tensor,
    left_hand_targets: torch.Tensor,
    right_hand_rest_pose: torch.Tensor,
    action_scale: float,
    lower_body_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge controller outputs into ordered, limit-safe 69D targets."""
    num_envs = default_joint_targets.shape[0]
    expected_shapes = {
        "default_joint_targets": (num_envs, len(DOORMAN_ALL_DOF_NAMES)),
        "soft_joint_pos_limits": (
            num_envs,
            len(DOORMAN_ALL_DOF_NAMES),
            2,
        ),
        "left_arm_actions": (num_envs, len(DOORMAN_LEFT_ARM_DOF_NAMES)),
        "left_hand_targets": (num_envs, len(DOORMAN_LEFT_HAND_DOF_NAMES)),
    }
    values = {
        "default_joint_targets": default_joint_targets,
        "soft_joint_pos_limits": soft_joint_pos_limits,
        "left_arm_actions": left_arm_actions,
        "left_hand_targets": left_hand_targets,
    }
    for name, expected_shape in expected_shapes.items():
        if values[name].shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {tuple(values[name].shape)}."
            )

    if right_hand_rest_pose.shape not in (
        (1, len(DOORMAN_RIGHT_HAND_DOF_NAMES)),
        (num_envs, len(DOORMAN_RIGHT_HAND_DOF_NAMES)),
    ):
        raise ValueError(
            "right_hand_rest_pose must have shape "
            f"(1, {len(DOORMAN_RIGHT_HAND_DOF_NAMES)}) or "
            f"({num_envs}, {len(DOORMAN_RIGHT_HAND_DOF_NAMES)}), "
            f"got {tuple(right_hand_rest_pose.shape)}."
        )

    if lower_body_targets is not None and lower_body_targets.shape != (
        num_envs,
        len(DOORMAN_LOWER_BODY_DOF_NAMES),
    ):
        raise ValueError(
            "lower_body_targets must have shape "
            f"({num_envs}, {len(DOORMAN_LOWER_BODY_DOF_NAMES)}), "
            f"got {tuple(lower_body_targets.shape)}."
        )

    targets = default_joint_targets.clone()
    if lower_body_targets is not None:
        targets[:, LOWER_BODY_SLICE] = lower_body_targets
    targets[:, LEFT_ARM_SLICE] = (
        default_joint_targets[:, LEFT_ARM_SLICE]
        + action_scale * left_arm_actions
    )
    targets[:, LEFT_HAND_SLICE] = left_hand_targets
    targets[:, RIGHT_HAND_SLICE] = right_hand_rest_pose

    return torch.clamp(
        targets,
        min=soft_joint_pos_limits[..., 0],
        max=soft_joint_pos_limits[..., 1],
    )


class G1WujiDoormanAction(ActionTerm):
    """Convert policy actions into ordered G1+Wuji joint-position targets."""

    cfg: "G1WujiDoormanActionCfg"

    def __init__(
        self,
        cfg: "G1WujiDoormanActionCfg",
        env: ManagerBasedEnv,
    ) -> None:
        super().__init__(cfg, env)

        if not isinstance(self._asset, Articulation):
            raise TypeError(
                f"Asset '{cfg.asset_name}' must be an Articulation."
            )

        joint_ids, joint_names = self._asset.find_joints(
            list(DOORMAN_ALL_DOF_NAMES),
            preserve_order=True,
        )
        if tuple(joint_names) != DOORMAN_ALL_DOF_NAMES:
            raise RuntimeError(
                "Resolved G1+Wuji joint order does not match "
                "DOORMAN_ALL_DOF_NAMES."
            )
        self._joint_ids = joint_ids

        homie_joint_count = len(HOMIE_OBS_JOINT_ORDER)
        if tuple(joint_names[:homie_joint_count]) != HOMIE_OBS_JOINT_ORDER:
            raise RuntimeError(
                "The first 29 DoorMan joints do not match HOMIE's "
                "observation order."
            )
        self._homie_joint_ids = joint_ids[:homie_joint_count]

        if _RIGHT_HAND_END != len(DOORMAN_ALL_DOF_NAMES):
            raise RuntimeError(
                "Control-group dimensions do not cover all robot joints."
            )

        self._left_arm_slice = LEFT_ARM_SLICE
        self._left_hand_slice = LEFT_HAND_SLICE
        self._right_hand_slice = RIGHT_HAND_SLICE

        self._default_joint_targets = (
            self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        )
        self._soft_joint_pos_limits = (
            self._asset.data.soft_joint_pos_limits[
                :, self._joint_ids
            ].clone()
        )

        controller_type = cfg.hand_controller_type
        if not isinstance(controller_type, type) or not issubclass(
            controller_type,
            HandController,
        ):
            raise TypeError(
                "hand_controller_type must inherit from HandController."
            )

        self._hand_controller = controller_type(
            num_envs=self.num_envs,
            device=self.device,
        )
        if self._hand_controller.target_dim != len(
            DOORMAN_LEFT_HAND_DOF_NAMES
        ):
            raise RuntimeError(
                "Hand controller target dimension does not match the "
                "left-hand joint group."
            )

        homie_controller_type = cfg.homie_controller_type
        if cfg.homie_decimation < 1:
            raise ValueError("homie_decimation must be at least 1.")
        if homie_controller_type is not None:
            if not isinstance(homie_controller_type, type) or not issubclass(
                homie_controller_type,
                HomieController,
            ):
                raise TypeError(
                    "homie_controller_type must be HomieController, a "
                    "subclass, or None."
                )
            self._homie_controller = homie_controller_type(
                num_envs=self.num_envs,
                device=self.device,
                cfg=(replace(HomieControllerCfg(), checkpoint_path=cfg.homie_checkpoint_path)
                     if cfg.homie_checkpoint_path else None),
            )
        else:
            self._homie_controller = None

        self._action_dim = (
            len(DOORMAN_LEFT_ARM_DOF_NAMES)
            + self._hand_controller.command_dim
        )

        if cfg.delta_action_scale <= 0.0:
            raise ValueError("delta_action_scale must be positive.")
        if cfg.delta_action_clip <= 0.0:
            raise ValueError("delta_action_clip must be positive.")
        if cfg.action_scale <= 0.0:
            raise ValueError("action_scale must be positive.")

        self._right_hand_rest_pose = torch.tensor(
            WUJI_HAND_REST_POSE,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self._lower_body_targets = self._default_joint_targets[
            :, LOWER_BODY_SLICE
        ].clone()
        self._left_arm_actions = torch.zeros(
            self.num_envs,
            len(DOORMAN_LEFT_ARM_DOF_NAMES),
            device=self.device,
        )
        self._left_hand_targets = self._default_joint_targets[
            :, LEFT_HAND_SLICE
        ].clone()
        self._homie_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.invalid_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._raw_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
        )
        self._delta_actions = torch.zeros_like(self._raw_actions)
        self._last_delta_actions = torch.zeros_like(self._raw_actions)
        self._processed_actions = self._default_joint_targets.clone()
        self._set_right_hand_rest_pose()
        self._clamp_joint_targets()

    @property
    def action_dim(self) -> int:
        """Number of policy actions consumed by the selected backend."""
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Unprocessed policy actions."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Final 69-dimensional joint-position targets."""
        return self._processed_actions

    @property
    def delta_actions(self) -> torch.Tensor:
        """DoorMan-style cumulative policy actions."""
        return self._delta_actions

    @property
    def last_delta_actions(self) -> torch.Tensor:
        """Most recent unaccumulated policy increments."""
        return self._last_delta_actions

    @property
    def hand_controller(self) -> HandController:
        """Selected interchangeable hand-control backend."""
        return self._hand_controller

    @property
    def homie_controller(self) -> HomieController | None:
        """Frozen standing controller, or ``None`` in mapping-only tests."""
        return self._homie_controller

    @property
    def controller_actions(self) -> torch.Tensor:
        """DoorMan `actions`: unscaled HOMIE outputs plus accumulated arm/hand commands.

        Inactive right arm/hand and locomotion channels are omitted. The primitive
        command is represented before its physical interpolation, as in DoorMan.
        """
        lower = (self._homie_controller.last_action if self._homie_controller is not None
                 else torch.zeros(self.num_envs, len(DOORMAN_LOWER_BODY_DOF_NAMES), device=self.device))
        return torch.cat((lower, self._delta_actions), dim=-1)

    def _refresh_processed_actions(self) -> None:
        self._processed_actions.copy_(
            compose_doorman_joint_targets(
                default_joint_targets=self._default_joint_targets,
                soft_joint_pos_limits=self._soft_joint_pos_limits,
                left_arm_actions=self._left_arm_actions,
                left_hand_targets=self._left_hand_targets,
                right_hand_rest_pose=self._right_hand_rest_pose,
                action_scale=self.cfg.action_scale,
                lower_body_targets=self._lower_body_targets,
            )
        )

    def _update_homie_targets(self, env_ids: torch.Tensor) -> None:
        if self._homie_controller is None:
            return

        self._lower_body_targets[env_ids] = (
            self._homie_controller.compute_joint_targets(
                joint_pos=self._asset.data.joint_pos[
                    env_ids[:, None], self._homie_joint_ids
                ],
                joint_vel=self._asset.data.joint_vel[
                    env_ids[:, None], self._homie_joint_ids
                ],
                base_ang_vel=self._asset.data.root_ang_vel_b[env_ids],
                projected_gravity=self._asset.data.projected_gravity_b[env_ids],
                env_ids=env_ids,
            )
        )

        self.invalid_state |= self._homie_controller.invalid_state

    def _set_right_hand_rest_pose(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._processed_actions[
            env_ids,
            self._right_hand_slice,
        ] = self._right_hand_rest_pose

    def _clamp_joint_targets(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)
        limits = self._soft_joint_pos_limits[env_ids]
        self._processed_actions[env_ids] = torch.clamp(
            self._processed_actions[env_ids],
            min=limits[..., 0],
            max=limits[..., 1],
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Accumulate DoorMan delta actions and build physical targets."""
        expected_shape = (self.num_envs, self.action_dim)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Policy actions must have shape {expected_shape}, "
                f"got {tuple(actions.shape)}."
            )

        invalid = ~torch.isfinite(actions).all(dim=-1)
        self.invalid_state |= invalid
        # Keep bad policy values out of the physics engine; terminate the affected env.
        actions = torch.where(invalid[:, None], 0.0, actions)
        self._raw_actions.copy_(actions)
        self._last_delta_actions.copy_(actions)
        self._delta_actions.add_(
            actions,
            alpha=self.cfg.delta_action_scale,
        )
        self._delta_actions.clamp_(
            min=-self.cfg.delta_action_clip,
            max=self.cfg.delta_action_clip,
        )

        arm_dim = len(DOORMAN_LEFT_ARM_DOF_NAMES)
        self._left_arm_actions.copy_(self._delta_actions[:, :arm_dim])
        hand_command = (
            self._delta_actions[:, arm_dim:] * self.cfg.action_scale
        )
        self._left_hand_targets.copy_(
            self._hand_controller.compute_joint_targets(hand_command)
        )
        self._refresh_processed_actions()

    def apply_actions(self) -> None:
        """Apply the most recently processed position targets."""
        env_ids = (self._homie_step_counter == 0).nonzero(as_tuple=False).flatten()
        if env_ids.numel():
            self._update_homie_targets(env_ids)
            self._refresh_processed_actions()
        self._homie_step_counter = (
            self._homie_step_counter + 1
        ) % self.cfg.homie_decimation

        self._asset.set_joint_position_target(
            self._processed_actions,
            joint_ids=self._joint_ids,
        )

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Reset action buffers and the selected hand backend."""
        if env_ids is None:
            env_ids = slice(None)

        controller_env_ids = env_ids
        if isinstance(controller_env_ids, slice):
            controller_env_ids = torch.arange(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
            )[controller_env_ids]

        self._raw_actions[env_ids] = 0.0
        self._delta_actions[env_ids] = 0.0
        self._last_delta_actions[env_ids] = 0.0
        self._left_arm_actions[env_ids] = 0.0
        self._left_hand_targets[env_ids] = self._default_joint_targets[
            env_ids,
            LEFT_HAND_SLICE,
        ]
        self._lower_body_targets[env_ids] = self._default_joint_targets[
            env_ids,
            LOWER_BODY_SLICE,
        ]
        self._processed_actions[env_ids] = (
            self._default_joint_targets[env_ids]
        )
        self._set_right_hand_rest_pose(env_ids)
        self._clamp_joint_targets(env_ids)
        self._hand_controller.reset(controller_env_ids)
        if self._homie_controller is not None:
            self._homie_controller.reset(controller_env_ids)
        self._homie_step_counter[env_ids] = 0
        self.invalid_state[env_ids] = False


@configclass
class G1WujiDoormanActionCfg(ActionTermCfg):
    """Configuration for the composite G1+Wuji action term."""

    class_type: type[ActionTerm] = G1WujiDoormanAction
    hand_controller_type: type[HandController] = PrimitiveHandController
    homie_controller_type: type[HomieController] | None = HomieController
    homie_checkpoint_path: str | None = None
    homie_decimation: int = 4
    delta_action_scale: float = 0.3
    delta_action_clip: float = 15.0
    action_scale: float = 0.25
