# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms used by the G1 Wuji Doorman environment."""

from isaaclab.envs.mdp.actions.actions_cfg import (
    JointEffortActionCfg,
    JointPositionActionCfg,
)

from isaaclab.envs.mdp.events import (
    reset_joints_by_offset,
    reset_scene_to_default,
)
from isaaclab.envs.mdp.observations import joint_pos_rel, joint_vel_rel
from isaaclab.envs.mdp.rewards import is_alive, is_terminated, joint_vel_l1
from isaaclab.envs.mdp.terminations import (
    joint_pos_out_of_manual_limit,
    time_out,
)

from .actions import (
    G1WujiDoormanAction,
    G1WujiDoormanActionCfg,
    compose_doorman_joint_targets,
)
from .rewards import joint_pos_target_l2

__all__ = [
    "JointEffortActionCfg",
    "reset_joints_by_offset",
    "joint_pos_rel",
    "joint_vel_rel",
    "is_alive",
    "is_terminated",
    "joint_vel_l1",
    "joint_pos_out_of_manual_limit",
    "time_out",
    "joint_pos_target_l2",
    "JointPositionActionCfg",
    "reset_scene_to_default",
    "G1WujiDoormanAction",
    "G1WujiDoormanActionCfg",
    "compose_doorman_joint_targets",
]

from .commands import DoorTaskStateCfg, FINGER_LINKS
from .observations import door_task_observation
from .task_rewards import door_task_reward
from .terminations import task_invalid_state, task_fall, task_success, task_stage_timeout

__all__ += [
    "DoorTaskStateCfg", "FINGER_LINKS", "door_task_observation", "door_task_reward",
    "task_invalid_state", "task_fall", "task_success", "task_stage_timeout",
]
