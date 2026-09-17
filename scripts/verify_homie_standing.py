"""Verify G1+Wuji standing with the validated DoorMan HOMIE setup."""

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--report_interval", type=float, default=2.0)
parser.add_argument(
    "--with_door",
    action="store_true",
    help="Run in the full door scene at the original x=0.04 m stand-off.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Lab imports must come after AppLauncher.
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

from g1_wuji_doorman.assets.robots.g1_wuji import (
    DOORMAN_ALL_DOF_NAMES,
    DOORMAN_BODY_DOF_NAMES,
    G1_WUJI_FREE_BASE_CFG,
)
from g1_wuji_doorman.controllers.standing import HomieController
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman.g1_wuji_doorman_env_cfg import (
    G1WujiDoormanSceneCfg,
)


PHYSICS_DT = 0.005
CONTROL_DECIMATION = 4
CONTROL_DT = PHYSICS_DT * CONTROL_DECIMATION

MIN_HEIGHT = 0.50
MAX_TILT_RADIANS = math.radians(45.0)
MAX_ROOT_DRIFT = 0.25
MAX_FOOT_SPEED = 0.50
DOOR_SCENE_ROBOT_POS = (0.04, -0.20, 0.75)


def require_order(asset, requested_names, find_method, label):
    """Resolve names and fail immediately if the requested order changed."""

    indices, resolved_names = find_method(list(requested_names), preserve_order=True)
    if tuple(resolved_names) != tuple(requested_names):
        raise RuntimeError(
            f"{label} order mismatch.\n"
            f"Expected: {tuple(requested_names)}\n"
            f"Resolved: {tuple(resolved_names)}"
        )
    return indices, resolved_names


def main() -> int:
    """Run the exact standalone control path validated in DoorMan."""

    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=PHYSICS_DT,
            device=args_cli.device,
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                max_position_iteration_count=8,
                max_velocity_iteration_count=4,
            ),
        )
    )

    if args_cli.with_door:
        scene_cfg = G1WujiDoormanSceneCfg(num_envs=1, env_spacing=4.0)
        scene_cfg.robot = G1_WUJI_FREE_BASE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        scene_cfg.robot.init_state.pos = DOOR_SCENE_ROBOT_POS
        scene = InteractiveScene(scene_cfg)
        sim.set_camera_view(
            eye=(-2.4, 2.8, 1.7),
            target=(0.0, -0.10, 0.85),
        )
        sim.reset()
        scene.update(PHYSICS_DT)
        robot = scene["robot"]

        # InteractiveScene does not perform the environment reset for us.
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos,
            robot.data.default_joint_vel,
        )
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        write_data_to_sim = scene.write_data_to_sim
        update_assets = scene.update
    else:
        ground_cfg = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        )
        ground_cfg.func("/World/GroundPlane", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

        robot = Articulation(
            G1_WUJI_FREE_BASE_CFG.replace(prim_path="/World/Robot")
        )
        sim.set_camera_view(
            eye=(2.6, 2.0, 1.65),
            target=(0.0, 0.0, 0.75),
        )
        sim.reset()
        robot.update(PHYSICS_DT)
        write_data_to_sim = robot.write_data_to_sim
        update_assets = robot.update

    body_ids, body_names = require_order(
        robot,
        DOORMAN_BODY_DOF_NAMES,
        robot.find_joints,
        "DoorMan body joint",
    )
    all_ids, _ = require_order(
        robot,
        DOORMAN_ALL_DOF_NAMES,
        robot.find_joints,
        "DoorMan full joint",
    )
    foot_ids, foot_names = require_order(
        robot,
        ("left_ankle_roll_link", "right_ankle_roll_link"),
        robot.find_bodies,
        "DoorMan foot body",
    )

    default_all_pos = robot.data.default_joint_pos[:, all_ids].clone()
    controller = HomieController(num_envs=1, device=sim.device)
    initial_root_xy = robot.data.root_pos_w[:, :2].clone()

    control_steps = max(1, round(args_cli.seconds / CONTROL_DT))
    report_steps = max(1, round(args_cli.report_interval / CONTROL_DT))
    warmup_steps = max(1, round(2.0 / CONTROL_DT))

    min_height = float("inf")
    max_tilt = 0.0
    max_root_drift = 0.0
    max_foot_speed = 0.0
    finite = True
    first_fall_step = None

    print("HOMIE_STANDING_START", flush=True)
    print(f"physics_frequency={1.0 / PHYSICS_DT:.1f} Hz", flush=True)
    print(f"control_frequency={1.0 / CONTROL_DT:.1f} Hz", flush=True)
    print(f"joints={robot.num_joints} bodies={robot.num_bodies}", flush=True)
    print(f"body_mapping={body_names}", flush=True)
    print(f"foot_bodies={foot_names}", flush=True)
    print(
        f"scene={'door' if args_cli.with_door else 'standalone'} "
        f"robot_start={DOOR_SCENE_ROBOT_POS if args_cli.with_door else (0.0, 0.0, 0.75)}",
        flush=True,
    )

    for control_step in range(control_steps):
        root_quat_w = robot.data.root_quat_w
        base_ang_vel = quat_apply_inverse(
            root_quat_w,
            robot.data.root_ang_vel_w,
        )
        gravity_w = torch.zeros_like(robot.data.root_pos_w)
        gravity_w[:, 2] = -1.0
        projected_gravity = quat_apply_inverse(root_quat_w, gravity_w)

        with torch.inference_mode():
            lower_targets = controller.compute_joint_targets(
                joint_pos=robot.data.joint_pos[:, body_ids],
                joint_vel=robot.data.joint_vel[:, body_ids],
                base_ang_vel=base_ang_vel,
                projected_gravity=projected_gravity,
            )

        full_targets = default_all_pos.clone()
        full_targets[:, :15] = lower_targets
        robot.set_joint_position_target(full_targets, joint_ids=all_ids)

        for _ in range(CONTROL_DECIMATION):
            write_data_to_sim()
            sim.step()
            update_assets(PHYSICS_DT)

        height = robot.data.root_pos_w[:, 2]
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        tilt = torch.maximum(roll.abs(), pitch.abs())
        root_drift = torch.linalg.vector_norm(
            robot.data.root_pos_w[:, :2] - initial_root_xy,
            dim=-1,
        )
        foot_speed = torch.linalg.vector_norm(
            robot.data.body_lin_vel_w[:, foot_ids, :2],
            dim=-1,
        )

        state_is_finite = all(
            torch.isfinite(value).all().item()
            for value in (
                robot.data.root_state_w,
                robot.data.joint_pos,
                robot.data.joint_vel,
                controller.last_action,
            )
        )
        finite = finite and state_is_finite
        min_height = min(min_height, height.min().item())
        max_tilt = max(max_tilt, tilt.max().item())
        max_root_drift = max(max_root_drift, root_drift.max().item())
        if control_step >= warmup_steps:
            max_foot_speed = max(max_foot_speed, foot_speed.max().item())

        if first_fall_step is None and (
            height.min().item() < MIN_HEIGHT
            or tilt.max().item() > MAX_TILT_RADIANS
            or not state_is_finite
        ):
            first_fall_step = control_step
            break

        if (control_step + 1) % report_steps == 0:
            print(
                "HOMIE_STANDING_METRIC "
                f"time={(control_step + 1) * CONTROL_DT:.2f} "
                f"height={height.min().item():.4f} "
                f"tilt_deg={math.degrees(tilt.max().item()):.2f} "
                f"root_drift={root_drift.max().item():.4f} "
                f"foot_speed={foot_speed.max().item():.4f}",
                flush=True,
            )

    passed = (
        first_fall_step is None
        and finite
        and min_height >= MIN_HEIGHT
        and max_tilt <= MAX_TILT_RADIANS
        and max_root_drift <= MAX_ROOT_DRIFT
        and max_foot_speed <= MAX_FOOT_SPEED
    )

    print(
        "HOMIE_STANDING_RESULT "
        f"min_height={min_height:.4f} "
        f"max_tilt_deg={math.degrees(max_tilt):.2f} "
        f"max_root_drift={max_root_drift:.4f} "
        f"max_foot_speed={max_foot_speed:.4f} "
        f"finite={finite} first_fall_step={first_fall_step}",
        flush=True,
    )
    print(
        "HOMIE_STANDING_PASS" if passed else "HOMIE_STANDING_FAIL",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
