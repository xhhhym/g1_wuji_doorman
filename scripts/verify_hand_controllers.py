"""Verify interchangeable Wuji hand-control backends."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Lab project imports must come after AppLauncher.
import torch

from g1_wuji_doorman.controllers.hand import (
    JointTargetHandController,
    PrimitiveHandController,
    WUJI_HAND_GRASP_POSE,
    WUJI_HAND_OPEN_POSE,
)


def main() -> None:
    """Verify every hand backend against the shared controller contract."""
    device = torch.device(args_cli.device)

    primitive = PrimitiveHandController(
        num_envs=3,
        device=device,
    )

    primitive_command = torch.tensor(
        [[-1.0], [0.0], [1.0]],
        dtype=torch.float32,
        device=device,
    )
    primitive_targets = primitive.compute_joint_targets(
        primitive_command
    )

    open_pose = torch.tensor(
        WUJI_HAND_OPEN_POSE,
        dtype=torch.float32,
        device=device,
    )
    grasp_pose = torch.tensor(
        WUJI_HAND_GRASP_POSE,
        dtype=torch.float32,
        device=device,
    )
    expected_primitive_targets = torch.stack(
        (
            open_pose,
            0.5 * (open_pose + grasp_pose),
            grasp_pose,
        )
    )

    torch.testing.assert_close(
        primitive_targets,
        expected_primitive_targets,
    )

    if primitive.command_dim != 1:
        raise RuntimeError(
            f"Primitive command_dim must be 1, "
            f"got {primitive.command_dim}."
        )

    joint_target = JointTargetHandController(
        num_envs=3,
        device=device,
    )
    joint_command = expected_primitive_targets.clone()
    joint_targets = joint_target.compute_joint_targets(joint_command)

    torch.testing.assert_close(joint_targets, joint_command)

    if joint_target.command_dim != 20:
        raise RuntimeError(
            f"Joint-target command_dim must be 20, "
            f"got {joint_target.command_dim}."
        )

    if primitive_targets.shape != (3, 20):
        raise RuntimeError(
            f"Expected primitive output shape (3, 20), "
            f"got {tuple(primitive_targets.shape)}."
        )

    try:
        primitive.compute_joint_targets(
            torch.zeros(3, 2, device=device)
        )
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "Primitive controller accepted an invalid command shape."
        )

    primitive.reset()
    joint_target.reset()

    print(
        "HAND_CONTROLLER_VERIFY_PASS "
        f"primitive_action_dim={7 + primitive.command_dim} "
        f"joint_target_action_dim={7 + joint_target.command_dim} "
        f"target_dim={primitive.target_dim}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
