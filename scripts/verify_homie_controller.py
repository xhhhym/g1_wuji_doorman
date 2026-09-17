"""Load the HOMIE checkpoint and verify its observation/action contract."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

from g1_wuji_doorman.controllers.standing import HomieController


def main() -> None:
    num_envs = 2
    controller = HomieController(
        num_envs=num_envs,
        device=args_cli.device,
    )

    joint_pos = controller.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    base_ang_vel = torch.zeros(
        num_envs,
        3,
        device=args_cli.device,
    )
    projected_gravity = torch.zeros(
        num_envs,
        3,
        device=args_cli.device,
    )
    projected_gravity[:, 2] = -1.0

    targets = None

    for _ in range(10):
        targets = controller.compute_joint_targets(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
        )

    assert targets is not None
    assert targets.shape == (num_envs, 15)
    assert controller.observation_history.shape == (num_envs, 516)
    assert torch.isfinite(targets).all()
    assert torch.isfinite(controller.last_action).all()

    print(f"history_shape={tuple(controller.observation_history.shape)}")
    print(f"action_shape={tuple(controller.last_action.shape)}")
    print(f"target_shape={tuple(targets.shape)}")
    print(
        "action_range="
        f"[{controller.last_action.min().item():.6f}, "
        f"{controller.last_action.max().item():.6f}]"
    )
    print(
        "target_range="
        f"[{targets.min().item():.6f}, "
        f"{targets.max().item():.6f}]"
    )

    controller.reset()

    if torch.count_nonzero(controller.observation_history) != 0:
        raise RuntimeError("HOMIE history was not cleared by reset().")

    if torch.count_nonzero(controller.last_action) != 0:
        raise RuntimeError("HOMIE last action was not cleared by reset().")

    print("HOMIE_CONTROLLER_PASS")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()