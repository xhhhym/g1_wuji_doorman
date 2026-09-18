"""Show G1+Wuji and the door while the left-hand primitive moves 0 to 1.

Primitive value 0 means the verified open pose and 1 means the verified grasp
pose. The motion is executed through the production DoorMan delta ActionTerm;
the script does not write finger targets directly.
"""

import argparse
from types import SimpleNamespace

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--ramp_seconds",
    type=float,
    default=4.0,
    help="Duration of the primitive 0-to-1 ramp. Default: 4.0.",
)
parser.add_argument(
    "--hold_seconds",
    type=float,
    default=2.0,
    help="Open and grasp hold duration. Default: 2.0.",
)
parser.add_argument(
    "--exit_after_test",
    action="store_true",
    help="Close Isaac Sim after the primitive reaches 1.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Lab imports must come after AppLauncher.
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import ActionManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from g1_wuji_doorman.assets.robots import (
    DOORMAN_ALL_DOF_NAMES,
    DOORMAN_LEFT_HAND_DOF_NAMES,
)
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman import mdp
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman.g1_wuji_doorman_env_cfg import (
    G1WujiDoormanSceneCfg,
)


PHYSICS_DT = 0.005
HIGH_LEVEL_DECIMATION = 8
HIGH_LEVEL_DT = PHYSICS_DT * HIGH_LEVEL_DECIMATION
DELTA_ACTION_SCALE = 0.3
DELTA_ACTION_CLIP = 15.0
ACTION_SCALE = 0.25


def _require_joint_order(asset, requested_names: tuple[str, ...], label: str):
    joint_ids, joint_names = asset.find_joints(
        list(requested_names),
        preserve_order=True,
    )
    if tuple(joint_names) != requested_names:
        raise RuntimeError(
            f"{label} order mismatch. Expected {requested_names}, "
            f"resolved {tuple(joint_names)}."
        )
    return joint_ids


def main() -> None:
    """Run the combined door scene and one smooth primitive 0-to-1 ramp."""
    if args_cli.ramp_seconds <= 0.0:
        raise ValueError("--ramp_seconds must be positive.")
    if args_cli.hold_seconds < 0.0:
        raise ValueError("--hold_seconds must be non-negative.")

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
    sim.set_camera_view(
        eye=(2.3, -2.4, 1.55),
        target=(0.30, -0.05, 0.90),
    )

    scene = InteractiveScene(
        G1WujiDoormanSceneCfg(num_envs=1, env_spacing=4.0)
    )
    sim.reset()
    scene.update(PHYSICS_DT)

    robot = scene["robot"]
    door = scene["door"]

    # InteractiveScene does not perform the environment reset for this
    # standalone script, so explicitly restore both articulations.
    for asset in (robot, door):
        root_state = asset.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        asset.write_root_pose_to_sim(root_state[:, :7])
        asset.write_root_velocity_to_sim(root_state[:, 7:])
        asset.write_joint_state_to_sim(
            asset.data.default_joint_pos,
            asset.data.default_joint_vel,
        )
    scene.write_data_to_sim()
    scene.update(PHYSICS_DT)

    action_cfg = mdp.G1WujiDoormanActionCfg(
        asset_name="robot",
        homie_decimation=4,
        delta_action_scale=DELTA_ACTION_SCALE,
        delta_action_clip=DELTA_ACTION_CLIP,
        action_scale=ACTION_SCALE,
    )
    env_proxy = SimpleNamespace(
        scene=scene,
        num_envs=scene.num_envs,
        device=sim.device,
        sim=sim,
    )
    action_manager = ActionManager(
        cfg=SimpleNamespace(doorman=action_cfg),
        env=env_proxy,
    )
    action_term = action_manager.get_term("doorman")
    if action_manager.total_action_dim != 8:
        raise RuntimeError(
            f"Expected primitive action dimension 8, "
            f"got {action_manager.total_action_dim}."
        )

    all_joint_ids = _require_joint_order(
        robot,
        DOORMAN_ALL_DOF_NAMES,
        "Full robot joint",
    )
    hand_joint_ids = _require_joint_order(
        robot,
        DOORMAN_LEFT_HAND_DOF_NAMES,
        "Left-hand joint",
    )

    action = torch.zeros(1, 8, device=sim.device)

    # DoorMan maps effective command -1/1 to primitive alpha 0/1. Initialize
    # the accumulator at -1 and align the physical hand before stepping, so
    # the visualization begins open without an artificial velocity spike.
    action[:, -1] = -1.0 / (DELTA_ACTION_SCALE * ACTION_SCALE)
    action_manager.process_action(action)
    initial_targets = action_term.processed_actions.clone()
    robot.write_joint_state_to_sim(
        initial_targets,
        torch.zeros_like(initial_targets),
        joint_ids=all_joint_ids,
    )
    robot.set_joint_position_target(
        initial_targets,
        joint_ids=all_joint_ids,
    )
    scene.write_data_to_sim()
    scene.update(PHYSICS_DT)

    max_abs_joint_velocity = 0.0
    max_abs_hand_velocity = 0.0
    previous_alpha = 0.0

    def primitive_alpha() -> float:
        effective_command = torch.clamp(
            action_term.delta_actions[0, -1] * ACTION_SCALE,
            min=-1.0,
            max=1.0,
        )
        return ((effective_command + 1.0) * 0.5).item()

    def advance(hand_delta: float) -> float:
        nonlocal max_abs_joint_velocity, max_abs_hand_velocity

        action.zero_()
        action[:, -1] = hand_delta
        action_manager.process_action(action)

        for _ in range(HIGH_LEVEL_DECIMATION):
            if not simulation_app.is_running():
                return primitive_alpha()

            action_manager.apply_action()
            scene.write_data_to_sim()
            sim.step()
            scene.update(PHYSICS_DT)

            state_tensors = (
                robot.data.root_state_w,
                robot.data.joint_pos[:, all_joint_ids],
                robot.data.joint_vel[:, all_joint_ids],
                door.data.root_state_w,
                door.data.joint_pos,
                door.data.joint_vel,
            )
            if not all(
                torch.isfinite(value).all().item()
                for value in state_tensors
            ):
                raise RuntimeError("Robot or door state contains NaN/Inf.")

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

        return primitive_alpha()

    print(
        "DOOR_HAND_PRIMITIVE_START "
        f"robot_joints={robot.num_joints} door_joints={door.num_joints} "
        f"primitive={primitive_alpha():.3f}",
        flush=True,
    )

    hold_updates = max(0, round(args_cli.hold_seconds / HIGH_LEVEL_DT))
    print("DOOR_HAND_PRIMITIVE_PHASE phase=open_hold primitive=0.000", flush=True)
    for _ in range(hold_updates):
        advance(0.0)

    ramp_updates = max(1, round(args_cli.ramp_seconds / HIGH_LEVEL_DT))
    # Cumulative action must move from -4 to +4 because action_scale=0.25.
    ramp_delta = 8.0 / (DELTA_ACTION_SCALE * ramp_updates)
    report_updates = max(1, ramp_updates // 10)
    print(
        "DOOR_HAND_PRIMITIVE_PHASE "
        f"phase=ramp_0_to_1 updates={ramp_updates}",
        flush=True,
    )
    for update in range(ramp_updates):
        current_alpha = advance(ramp_delta)
        if current_alpha + 1.0e-6 < previous_alpha:
            raise RuntimeError(
                "Primitive alpha decreased during the 0-to-1 ramp."
            )
        previous_alpha = current_alpha
        if (update + 1) % report_updates == 0 or update + 1 == ramp_updates:
            print(
                "DOOR_HAND_PRIMITIVE_PROGRESS "
                f"step={update + 1}/{ramp_updates} "
                f"primitive={current_alpha:.3f}",
                flush=True,
            )

    final_alpha = primitive_alpha()
    print("DOOR_HAND_PRIMITIVE_PHASE phase=grasp_hold primitive=1.000", flush=True)
    for _ in range(hold_updates):
        final_alpha = advance(0.0)

    if abs(final_alpha - 1.0) > 1.0e-5:
        raise RuntimeError(
            f"Primitive did not reach 1.0; final value is {final_alpha:.6f}."
        )

    print(
        "DOOR_HAND_PRIMITIVE_PASS "
        f"primitive_start=0.000 primitive_end={final_alpha:.3f} "
        f"max_abs_joint_velocity={max_abs_joint_velocity:.4f} "
        f"max_abs_hand_velocity={max_abs_hand_velocity:.4f}",
        flush=True,
    )

    if args_cli.exit_after_test:
        return

    print(
        "[INFO] Primitive reached 1. The window remains open; "
        "close Isaac Sim to exit.",
        flush=True,
    )
    while simulation_app.is_running():
        advance(0.0)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
