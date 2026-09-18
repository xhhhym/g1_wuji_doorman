"""Verify DoorMan action composition without creating a simulator scene."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Lab project imports must come after AppLauncher.
import torch

from g1_wuji_doorman.assets.robots import (
    DOORMAN_ALL_DOF_NAMES,
    DOORMAN_LEFT_ARM_DOF_NAMES,
    DOORMAN_LEFT_HAND_DOF_NAMES,
    DOORMAN_LOWER_BODY_DOF_NAMES,
    DOORMAN_RIGHT_ARM_DOF_NAMES,
)
from g1_wuji_doorman.controllers.hand import (
    JointTargetHandController,
    PrimitiveHandController,
    WUJI_HAND_REST_POSE,
)
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman.mdp.actions import (
    LEFT_ARM_SLICE,
    LEFT_HAND_SLICE,
    LOWER_BODY_SLICE,
    RIGHT_HAND_SLICE,
    compose_doorman_joint_targets,
)


NUM_ENVS = 2
ARM_SCALE = 0.25


def _compose(
    controller,
    policy_actions: torch.Tensor,
    default_targets: torch.Tensor,
    limits: torch.Tensor,
    right_hand_rest: torch.Tensor,
    lower_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    arm_dim = len(DOORMAN_LEFT_ARM_DOF_NAMES)
    hand_targets = controller.compute_joint_targets(
        policy_actions[:, arm_dim:]
    )
    return compose_doorman_joint_targets(
        default_joint_targets=default_targets,
        soft_joint_pos_limits=limits,
        left_arm_actions=policy_actions[:, :arm_dim],
        left_hand_targets=hand_targets,
        right_hand_rest_pose=right_hand_rest,
        arm_action_scale=ARM_SCALE,
        lower_body_targets=lower_targets,
    )


def _changed_columns(
    actual: torch.Tensor,
    baseline: torch.Tensor,
) -> set[int]:
    changed = (actual - baseline).abs().amax(dim=0) > 1.0e-6
    return set(torch.nonzero(changed, as_tuple=False).flatten().tolist())


def _verify_backend(
    controller_type: type[PrimitiveHandController]
    | type[JointTargetHandController],
    default_targets: torch.Tensor,
    limits: torch.Tensor,
    right_hand_rest: torch.Tensor,
) -> None:
    controller = controller_type(num_envs=NUM_ENVS, device=args_cli.device)
    action_dim = len(DOORMAN_LEFT_ARM_DOF_NAMES) + controller.command_dim
    baseline_action = torch.zeros(
        NUM_ENVS,
        action_dim,
        device=args_cli.device,
    )
    baseline = _compose(
        controller,
        baseline_action,
        default_targets,
        limits,
        right_hand_rest,
    )

    if baseline.shape != (NUM_ENVS, len(DOORMAN_ALL_DOF_NAMES)):
        raise RuntimeError(f"Unexpected target shape: {baseline.shape}.")

    # Every arm action dimension must move only its matching left-arm joint.
    for action_index in range(len(DOORMAN_LEFT_ARM_DOF_NAMES)):
        action = baseline_action.clone()
        action[:, action_index] = 0.5
        targets = _compose(
            controller,
            action,
            default_targets,
            limits,
            right_hand_rest,
        )
        expected = {LEFT_ARM_SLICE.start + action_index}
        changed = _changed_columns(targets, baseline)
        if changed != expected:
            raise RuntimeError(
                f"Arm action {action_index} changed {changed}, "
                f"expected {expected}."
            )

    if controller.command_dim == 1:
        action = baseline_action.clone()
        action[:, -1] = 1.0
        targets = _compose(
            controller,
            action,
            default_targets,
            limits,
            right_hand_rest,
        )
        changed = _changed_columns(targets, baseline)
        expected = set(range(LEFT_HAND_SLICE.start, LEFT_HAND_SLICE.stop))
        if not changed or not changed.issubset(expected):
            raise RuntimeError(
                f"Primitive command changed unexpected joints: {changed}."
            )
    else:
        for command_index in range(controller.command_dim):
            action = baseline_action.clone()
            action[:, len(DOORMAN_LEFT_ARM_DOF_NAMES) + command_index] = 0.5
            targets = _compose(
                controller,
                action,
                default_targets,
                limits,
                right_hand_rest,
            )
            expected = {LEFT_HAND_SLICE.start + command_index}
            changed = _changed_columns(targets, baseline)
            if changed != expected:
                raise RuntimeError(
                    f"Hand command {command_index} changed {changed}, "
                    f"expected {expected}."
                )

    if not torch.equal(
        baseline[:, RIGHT_HAND_SLICE],
        right_hand_rest.expand(NUM_ENVS, -1),
    ):
        raise RuntimeError("Right hand is not held at the rest pose.")

    print(
        "ACTION_MAPPING_BACKEND_PASS "
        f"backend={controller_type.__name__} "
        f"action_dim={action_dim}",
        flush=True,
    )


def main() -> None:
    device = torch.device(args_cli.device)
    default_targets = torch.zeros(
        NUM_ENVS,
        len(DOORMAN_ALL_DOF_NAMES),
        device=device,
    )
    # Non-zero right-arm defaults prove that the merge preserves this group.
    right_arm_start = LEFT_ARM_SLICE.stop
    right_arm_end = right_arm_start + len(DOORMAN_RIGHT_ARM_DOF_NAMES)
    default_targets[:, right_arm_start:right_arm_end] = 0.1

    limits = torch.empty(
        NUM_ENVS,
        len(DOORMAN_ALL_DOF_NAMES),
        2,
        device=device,
    )
    limits[..., 0] = -2.0
    limits[..., 1] = 2.0
    right_hand_rest = torch.tensor(
        WUJI_HAND_REST_POSE,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    for controller_type in (
        PrimitiveHandController,
        JointTargetHandController,
    ):
        _verify_backend(
            controller_type,
            default_targets,
            limits,
            right_hand_rest,
        )

    lower_targets = torch.full(
        (NUM_ENVS, len(DOORMAN_LOWER_BODY_DOF_NAMES)),
        0.2,
        device=device,
    )
    primitive = PrimitiveHandController(NUM_ENVS, device)
    action = torch.zeros(NUM_ENVS, 8, device=device)
    targets = _compose(
        primitive,
        action,
        default_targets,
        limits,
        right_hand_rest,
        lower_targets=lower_targets,
    )
    torch.testing.assert_close(targets[:, LOWER_BODY_SLICE], lower_targets)

    tight_limits = limits.clone()
    tight_limits[:, LEFT_ARM_SLICE, 1] = 0.05
    action[:, : len(DOORMAN_LEFT_ARM_DOF_NAMES)] = 100.0
    targets = _compose(
        primitive,
        action,
        default_targets,
        tight_limits,
        right_hand_rest,
    )
    if not torch.all(targets[:, LEFT_ARM_SLICE] == 0.05):
        raise RuntimeError("Soft joint limits did not clamp arm targets.")

    print(
        "ACTION_MAPPING_VERIFY_PASS "
        "backends=2 arm_sweep=7 hand_sweep=21 processed_dim=69",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
