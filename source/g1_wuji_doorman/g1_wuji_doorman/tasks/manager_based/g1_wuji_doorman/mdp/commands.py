"""Central task state for the fixed, right-hinged DoorMan baseline.

Isaac Lab computes termination/reward BEFORE CommandManager.compute. Every task
termination and reward requests idempotent advance(), so their declaration order
is immaterial. compute() refreshes geometry after automatic resets.
"""

import math
import omni.usd
from pathlib import PurePosixPath

import torch
import torch.nn.functional as F

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_mul, subtract_frame_transforms
from g1_wuji_doorman.assets.robots import (
    DOORMAN_LEFT_ARM_DOF_NAMES, DOORMAN_LEFT_HAND_DOF_NAMES, DOORMAN_RIGHT_ARM_DOF_NAMES,
)
from g1_wuji_doorman.controllers.hand import WUJI_HAND_OPEN_POSE, WUJI_HAND_GRASP_POSE
from .contracts import contact_filter_indices, door_info_from_metadata

FINGER_LINKS = tuple(
    f"left_finger{finger}_{link}"
    for finger in range(1, 6)
    for link in ("link1", "link2", "link3", "link4", "tip_link")
)


def pose6d(position, quaternion):
    """Position followed by first two rotation-matrix columns (column-major)."""
    rotation = matrix_from_quat(quaternion)
    return torch.cat((position, rotation[:, :, 0], rotation[:, :, 1]), dim=-1)


class DoorTaskState(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot, self.door = env.scene["robot"], env.scene["door"]
        self.palm_id = self.robot.find_bodies("left_palm_link")[0][0]
        self.target_id = self.door.find_bodies("grasp_target")[0][0]
        self.arm_ids = self.robot.find_joints(list(DOORMAN_LEFT_ARM_DOF_NAMES), preserve_order=True)[0]
        self.hand_ids = self.robot.find_joints(list(DOORMAN_LEFT_HAND_DOF_NAMES), preserve_order=True)[0]
        self.right_arm_ids = self.robot.find_joints(list(DOORMAN_RIGHT_ARM_DOF_NAMES), preserve_order=True)[0]
        self.door_ids = self.door.find_joints(["hinge_joint", "handle_joint", "latch_joint"], preserve_order=True)[0]
        self.open_pose = torch.tensor(WUJI_HAND_OPEN_POSE, device=self.device)
        self.pose_delta = torch.tensor(WUJI_HAND_GRASP_POSE, device=self.device) - self.open_pose
        self.alignment_offset = torch.tensor(cfg.target_palm_quat, device=self.device).expand(self.num_envs, -1)
        self.contact_sensor = env.scene["handle_contacts"]
        view = self.contact_sensor.contact_physx_view
        self.contact_indices = torch.tensor(
            contact_filter_indices(view.sensor_paths, view.filter_paths, FINGER_LINKS),
            dtype=torch.long, device=self.device,
        )
        stage = omni.usd.get_context().get_stage()
        self.privileged_door_info = door_info_from_metadata([
            stage.GetPrimAtPath(str(PurePosixPath(path).parent)).GetCustomData()
            for path in view.sensor_paths
        ], self.device)
        self.foot_ids = self.robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0]
        limits = self.door.data.joint_pos_limits[:, self.door_ids]
        if not torch.allclose(limits[..., 0], torch.zeros_like(limits[..., 0]), atol=1e-5):
            raise ValueError("Expected positive hinge/handle/latch travel starting at zero")
        self.handle_travel = limits[:, 1, 1].clone()
        self.latch_travel = limits[:, 2, 1].clone()
        if not ((self.handle_travel > 0) & (self.latch_travel > 0)).all():
            raise ValueError("Handle and latch must have positive travel")
        self.stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reward_stage = self.stage.clone()
        self.hold = self.stage.clone()
        self.success_hold = self.stage.clone()
        self.time_in_stage = torch.zeros(self.num_envs, device=self.device)
        self.previous_angles = torch.zeros(self.num_envs, 2, device=self.device)
        self.progress = torch.zeros_like(self.previous_angles)
        self.transition = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success = self.transition.clone()
        self.stage_timeout = self.transition.clone()
        self.initial_root_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._last_step = -1
        self._timeouts = torch.tensor(cfg.stage_timeouts_s, device=self.device)
        for name in ("distance", "door_angle", "handle_angle", "contact_count", "max_stage", "success", "base_drift", "foot_speed", "invalid_state"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        for stage in range(4):
            self.metrics[f"stage_{stage}_timeout"] = torch.zeros(self.num_envs, device=self.device)
        self.refresh()

    @property
    def command(self):
        return F.one_hot(self.stage, num_classes=4).float()

    def refresh(self):
        robot, door = self.robot.data, self.door.data
        self.relative_pos, self.relative_quat = subtract_frame_transforms(
            robot.body_pos_w[:, self.palm_id], robot.body_quat_w[:, self.palm_id],
            door.body_pos_w[:, self.target_id], door.body_quat_w[:, self.target_id],
        )
        self.target_in_palm = pose6d(self.relative_pos, self.relative_quat)
        root_pos, root_quat = subtract_frame_transforms(
            door.root_pos_w, door.root_quat_w, robot.root_pos_w, robot.root_quat_w,
        )
        self.root_in_door = pose6d(root_pos, root_quat)
        q = door.joint_pos[:, self.door_ids]
        dq = door.joint_vel[:, self.door_ids]

        
        # This generated asset uses positive hinge/handle rotation and latch translation.
        self.door_state = torch.stack((q[:, 0], dq[:, 0], q[:, 1], dq[:, 1], q[:, 2]), dim=-1)
        self.distance = self.relative_pos.norm(dim=-1)
        error_quat = quat_mul(self.relative_quat, self.alignment_offset)
        self.orientation_error = 2 * torch.atan2(error_quat[:, 1:].norm(dim=-1), error_quat[:, 0].abs())
        matrix = self.contact_sensor.data.force_matrix_w
        if matrix is None or matrix.shape[1:] != (1, 25, 3):
            raise RuntimeError(f"Expected handle-to-25-links contact matrix, got {None if matrix is None else matrix.shape}")
        self.contact_forces_w = torch.gather(
            matrix[:, 0], 1, self.contact_indices[..., None].expand(-1, -1, 3)
        )
        self.link_forces = self.contact_forces_w.norm(dim=-1).reshape(self.num_envs, 5, 5)
        self.tip_forces = self.link_forces[:, :, -1]
        self.finger_forces = self.link_forces.amax(dim=-1)
        self.contact_count = (self.finger_forces > self.cfg.contact_threshold).sum(dim=-1)
        self.grasp_contact = (self.contact_count >= 3) & (self.finger_forces[:, 0] > self.cfg.contact_threshold)


        # Actual finger posture; independent of the interchangeable action backend.
        self.closure = (((robot.joint_pos[:, self.hand_ids] - self.open_pose) * self.pose_delta).sum(-1)
                        / self.pose_delta.square().sum().clamp_min(1e-6)).clamp(0, 1)
        self.fallen = (robot.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2] < self.cfg.min_root_height)
        self.fallen |= -robot.projected_gravity_b[:, 2] < self.cfg.min_upright
        finite_state = torch.cat((robot.root_state_w, robot.joint_pos, robot.joint_vel,
                                  door.root_state_w, door.joint_pos, door.joint_vel,
                                  self.door_state, self.target_in_palm, self.root_in_door,
                                  self.contact_forces_w.flatten(1)), dim=-1)
        self.invalid = ~torch.isfinite(finite_state).all(dim=-1)
        self.invalid |= robot.joint_vel.abs().amax(dim=-1) > self.cfg.max_joint_speed
        # CommandManager is constructed before ActionManager in Isaac Lab.
        if hasattr(self._env, "action_manager"):
            self.invalid |= self._env.action_manager.get_term("doorman").invalid_state

    def advance(self):
        """Safe from any termination/reward; advance at most once per high-level step."""
        if self._last_step == self._env.common_step_counter:
            return
        self._last_step = self._env.common_step_counter
        self.refresh()
        self.reward_stage.copy_(self.stage)
        angles = self.door_state[:, [0, 2]]
        self.progress.copy_(angles - self.previous_angles)
        self.previous_angles.copy_(angles)
        self.time_in_stage += self._env.step_dt
        self.stage_timeout.copy_(self.time_in_stage >= self._timeouts[self.stage])
        healthy = ~(self.fallen | self.invalid | self.stage_timeout)
        pregrasp = ((self.distance < self.cfg.reach_distance)
                    & (self.orientation_error < self.cfg.reach_angle)
                    & (self.closure < 0.6))
        grasp = self.grasp_contact & (self.distance < 0.12)
        unlatch = grasp & (self.door_state[:, 4] >= self.cfg.latch_release_fraction * self.latch_travel)
        condition = torch.where(self.stage == 0, pregrasp, torch.where(self.stage == 1, grasp, unlatch))
        condition &= (self.stage < 3) & healthy
        self.hold.copy_(torch.where(condition, self.hold + 1, 0))
        self.transition.copy_(self.hold >= self.cfg.transition_hold_steps)
        self.stage += self.transition.long()
        self.time_in_stage[self.transition] = 0
        self.hold[self.transition] = 0
        opened = (self.stage == 3) & (self.door_state[:, 0] >= self.cfg.success_angle) & healthy
        self.success_hold.copy_(torch.where(opened, self.success_hold + 1, 0))
        self.success.copy_(self.success_hold >= math.ceil(self.cfg.success_hold_s / self._env.step_dt))
        self._update_metrics()

    def _update_metrics(self):
        self.metrics["distance"].copy_(torch.nan_to_num(self.distance))
        self.metrics["door_angle"].copy_(torch.nan_to_num(self.door_state[:, 0]))
        self.metrics["handle_angle"].copy_(torch.nan_to_num(self.door_state[:, 2]))
        self.metrics["contact_count"].copy_(self.contact_count)
        self.metrics["max_stage"].copy_(self.stage)
        self.metrics["success"].copy_(self.success)
        self.metrics["base_drift"].copy_(torch.nan_to_num((self.robot.data.root_pos_w[:, :2] - self.initial_root_xy).norm(dim=-1)))
        self.metrics["foot_speed"].copy_(torch.nan_to_num(self.robot.data.body_lin_vel_w[:, self.foot_ids].norm(dim=-1).amax(dim=-1)))
        self.metrics["invalid_state"].copy_(self.invalid)
        for stage in range(4):
            self.metrics[f"stage_{stage}_timeout"].copy_(self.stage_timeout & (self.reward_stage == stage))

    def _resample_command(self, env_ids):
        self.stage[env_ids] = 0
        self.reward_stage[env_ids] = 0
        self.hold[env_ids] = 0
        self.success_hold[env_ids] = 0
        self.time_in_stage[env_ids] = 0
        self.progress[env_ids] = 0
        self.transition[env_ids] = False
        self.success[env_ids] = False
        self.stage_timeout[env_ids] = False
        self.initial_root_xy[env_ids] = self.robot.data.root_pos_w[env_ids, :2]
        self.refresh()
        self.previous_angles[env_ids] = self.door_state[env_ids][:, [0, 2]]

    def _update_command(self):
        self.refresh()

    def compute(self, dt):
        # No periodic resampling and no stage advancement after reward/reset.
        self._update_command()


@configclass
class DoorTaskStateCfg(CommandTermCfg):
    class_type: type = DoorTaskState
    resampling_time_range: tuple = (1.0e9, 1.0e9)
    reach_distance: float = 0.08
    reach_angle: float = math.radians(30)
    contact_threshold: float = 1.0
    transition_hold_steps: int = 5
    latch_release_fraction: float = 0.8
    max_joint_speed: float = 200.0
    success_angle: float = math.radians(30)
    success_hold_s: float = 0.5
    stage_timeouts_s: tuple = (8.0, 6.0, 6.0, 10.0)
    min_root_height: float = 0.45
    min_upright: float = 0.5
    # DoorMan left-hand grasp orientation: target frame rotated +90 deg about X.
    target_palm_quat: tuple = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
