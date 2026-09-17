"""Verify the G1+Wuji free-base articulation configuration."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Lab imports must come after AppLauncher.
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from g1_wuji_doorman.assets.robots import (
    G1_WUJI_CFG,
    G1_WUJI_FREE_BASE_CFG,
)
DT = 1.0 / 200.0


def joint_indices_to_list(indices, num_joints: int) -> list[int]:
    """Convert an actuator's joint indices to a normal Python list."""

    if isinstance(indices, slice):
        return list(range(num_joints))[indices]
    if isinstance(indices, torch.Tensor):
        return indices.detach().cpu().tolist()
    return list(indices)


def main() -> None:
    """Load the articulation and verify actuator coverage."""

    # First make sure copy() did not modify the original inspection config.
    assert G1_WUJI_CFG.spawn.rigid_props.disable_gravity is True
    assert G1_WUJI_CFG.spawn.articulation_props.fix_root_link is True

    assert G1_WUJI_FREE_BASE_CFG.spawn.rigid_props.disable_gravity is False
    assert G1_WUJI_FREE_BASE_CFG.spawn.articulation_props.fix_root_link is False

    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=DT,
            device=args_cli.device,
        )
    )

    robot = Articulation(
        G1_WUJI_FREE_BASE_CFG.replace(prim_path="/World/Robot")
    )
    sim.reset()
    robot.update(DT)

    print(f"robot_joints={robot.num_joints}")
    print(f"robot_bodies={robot.num_bodies}")
    print(f"actuator_groups={list(robot.actuators)}")

    owners: dict[int, str] = {}
    duplicate_joints: list[str] = []

    for actuator_name, actuator in robot.actuators.items():
        joint_ids = joint_indices_to_list(
            actuator.joint_indices,
            robot.num_joints,
        )
        joint_names = [robot.joint_names[index] for index in joint_ids]

        print(
            f"actuator={actuator_name} "
            f"count={len(joint_ids)} "
            f"first={joint_names[0]} "
            f"last={joint_names[-1]}"
        )

        for joint_id, joint_name in zip(joint_ids, joint_names):
            if joint_id in owners:
                duplicate_joints.append(
                    f"{joint_name}: {owners[joint_id]} and {actuator_name}"
                )
            owners[joint_id] = actuator_name

    missing_joints = [
        robot.joint_names[index]
        for index in range(robot.num_joints)
        if index not in owners
    ]

    if robot.num_joints != 69:
        raise RuntimeError(
            f"Expected 69 joints, found {robot.num_joints}"
        )

    all_actuator_ids = joint_indices_to_list(
        robot.actuators["all"].joint_indices,
        robot.num_joints,
    )
    if len(all_actuator_ids) != 69:
        raise RuntimeError("DoorMan actuator must contain exactly 69 joints.")

    if duplicate_joints:
        raise RuntimeError(
            f"Duplicate actuator assignments: {duplicate_joints}"
        )

    if missing_joints:
        raise RuntimeError(
            f"Joints without actuators: {missing_joints}"
        )

    expected_default_positions = {
        "left_hip_pitch_joint": -0.1,
        "right_hip_pitch_joint": -0.1,
        "left_knee_joint": 0.3,
        "right_knee_joint": 0.3,
        "left_ankle_pitch_joint": -0.2,
        "right_ankle_pitch_joint": -0.2,
    }

    for joint_name, expected_position in expected_default_positions.items():
        joint_ids, _ = robot.find_joints(joint_name)

        if isinstance(joint_ids, slice) or len(joint_ids) != 1:
            raise RuntimeError(
                f"Expected one joint named {joint_name}, found {joint_ids}"
            )

        actual_position = robot.data.default_joint_pos[
            0, joint_ids[0]
        ].item()

        print(
            f"default_position {joint_name}="
            f"{actual_position:.6f}"
        )

        if abs(actual_position - expected_position) > 1.0e-6:
            raise RuntimeError(
                f"{joint_name}: expected {expected_position}, "
                f"found {actual_position}"
            )

    print("FREE_BASE_CONFIG_PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
