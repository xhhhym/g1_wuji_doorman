"""Visualize a smooth Wuji left-hand open/grasp/open command."""

import argparse
from types import SimpleNamespace

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cycles", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Lab imports must come after AppLauncher.
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionManager
from isaaclab.sim import SimulationContext

from g1_wuji_doorman.assets.robots import (
    DOORMAN_ALL_DOF_NAMES,
    DOORMAN_LEFT_HAND_DOF_NAMES,
    G1_WUJI_FREE_BASE_CFG,
)
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman import mdp


PHYSICS_DT = 0.005
HIGH_LEVEL_DECIMATION = 8

# (label, duration in seconds, command at start, command at end)
PHASES = (
    ("open_hold", 1.0, -1.0, -1.0),
    ("smooth_grasp", 2.0, -1.0, 1.0),
    ("grasp_hold", 2.0, 1.0, 1.0),
    ("smooth_open", 2.0, 1.0, -1.0),
)


def _spawn_robot(sim: SimulationContext) -> Articulation:
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
        eye=(1.35, 1.45, 1.15),
        target=(0.0, 0.28, 0.86),
    )
    return robot


def main() -> None:
    if args_cli.cycles < 1:
        raise ValueError("--cycles must be at least 1.")

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
    robot = _spawn_robot(sim)
    sim.reset()
    robot.update(PHYSICS_DT)

    action_cfg = mdp.G1WujiDoormanActionCfg(
        asset_name="robot",
        arm_action_scale=0.25,
        homie_decimation=4,
    )
    env_proxy = SimpleNamespace(
        scene={"robot": robot},
        num_envs=1,
        device=sim.device,
        sim=sim,
    )
    action_manager = ActionManager(
        cfg=SimpleNamespace(doorman=action_cfg),
        env=env_proxy,
    )
    if action_manager.total_action_dim != 8:
        raise RuntimeError(
            f"Expected primitive action dimension 8, "
            f"got {action_manager.total_action_dim}."
        )

    all_joint_ids, all_joint_names = robot.find_joints(
        list(DOORMAN_ALL_DOF_NAMES),
        preserve_order=True,
    )
    if tuple(all_joint_names) != DOORMAN_ALL_DOF_NAMES:
        raise RuntimeError("Full joint order mismatch.")
    hand_joint_ids, hand_joint_names = robot.find_joints(
        list(DOORMAN_LEFT_HAND_DOF_NAMES),
        preserve_order=True,
    )
    if tuple(hand_joint_names) != DOORMAN_LEFT_HAND_DOF_NAMES:
        raise RuntimeError("Left-hand joint order mismatch.")

    action = torch.zeros(1, 8, device=sim.device)
    max_abs_joint_velocity = 0.0
    max_abs_hand_velocity = 0.0

    for cycle in range(args_cli.cycles):
        for label, duration, command_start, command_end in PHASES:
            print(
                f"HAND_VISUAL_PHASE cycle={cycle + 1} "
                f"phase={label}",
                flush=True,
            )
            phase_steps = max(1, round(duration / PHYSICS_DT))

            for step in range(phase_steps):
                if not simulation_app.is_running():
                    return

                if step % HIGH_LEVEL_DECIMATION == 0:
                    progress = step / max(1, phase_steps - 1)
                    hand_command = (
                        command_start
                        + (command_end - command_start) * progress
                    )
                    action[:, -1] = hand_command
                    action_manager.process_action(action)

                action_manager.apply_action()
                robot.write_data_to_sim()
                sim.step()
                robot.update(PHYSICS_DT)

                state_tensors = (
                    robot.data.root_state_w,
                    robot.data.joint_pos[:, all_joint_ids],
                    robot.data.joint_vel[:, all_joint_ids],
                )
                if not all(
                    torch.isfinite(value).all().item()
                    for value in state_tensors
                ):
                    raise RuntimeError(
                        f"Non-finite robot state during phase '{label}'."
                    )

                max_abs_joint_velocity = max(
                    max_abs_joint_velocity,
                    robot.data.joint_vel[:, all_joint_ids]
                    .abs()
                    .max()
                    .item(),
                )
                max_abs_hand_velocity = max(
                    max_abs_hand_velocity,
                    robot.data.joint_vel[:, hand_joint_ids]
                    .abs()
                    .max()
                    .item(),
                )

    print(
        "HAND_VISUAL_PASS "
        f"cycles={args_cli.cycles} "
        f"max_abs_joint_velocity={max_abs_joint_velocity:.4f} "
        f"max_abs_hand_velocity={max_abs_hand_velocity:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
