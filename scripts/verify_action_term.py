"""Verify the composite G1+Wuji action term with selectable hand backends."""

import argparse
from types import SimpleNamespace

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--hand_backend",
    choices=("primitive", "joint_target"),
    default="primitive",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Lab project imports must come after AppLauncher.
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionManager
from isaaclab.sim import SimulationContext

from g1_wuji_doorman.assets.robots import (
    DOORMAN_ALL_DOF_NAMES,
    DOORMAN_LEFT_ARM_DOF_NAMES,
    DOORMAN_LEFT_HAND_DOF_NAMES,
    DOORMAN_LOWER_BODY_DOF_NAMES,
    DOORMAN_RIGHT_ARM_DOF_NAMES,
    G1_WUJI_FREE_BASE_CFG,
)
from g1_wuji_doorman.controllers.hand import (
    JointTargetHandController,
    PrimitiveHandController,
    WUJI_HAND_GRASP_POSE,
    WUJI_HAND_OPEN_POSE,
)
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman import mdp


PHYSICS_DT = 0.005
DELTA_ACTION_SCALE = 0.3
DELTA_ACTION_CLIP = 15.0
ACTION_SCALE = 0.25


def _hand_test_targets(device: torch.device | str) -> torch.Tensor:
    """Return open/mid/grasp physical targets."""
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
    return torch.stack(
        (
            open_pose,
            0.5 * (open_pose + grasp_pose),
            grasp_pose,
        )
    )


def _make_action(
    backend: str,
    test_index: int,
    hand_targets: torch.Tensor,
) -> torch.Tensor:
    """Create one delta action that reaches a requested hand test state."""
    delta_to_target_scale = DELTA_ACTION_SCALE * ACTION_SCALE
    if backend == "primitive":
        actions = torch.zeros(1, 8, device=hand_targets.device)
        actions[:, -1] = (
            (-1.0, 0.0, 1.0)[test_index] / delta_to_target_scale
        )
        return actions

    actions = torch.zeros(1, 27, device=hand_targets.device)
    actions[:, len(DOORMAN_LEFT_ARM_DOF_NAMES):] = (
        hand_targets[test_index] / delta_to_target_scale
    )
    return actions


def main() -> None:
    """Instantiate and validate one configured action backend."""
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
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    robot = Articulation(
        G1_WUJI_FREE_BASE_CFG.replace(prim_path="/World/Robot")
    )
    sim.reset()
    robot.update(PHYSICS_DT)

    backend_types = {
        "primitive": PrimitiveHandController,
        "joint_target": JointTargetHandController,
    }
    action_cfg = mdp.G1WujiDoormanActionCfg(
        asset_name="robot",
        hand_controller_type=backend_types[args_cli.hand_backend],
        homie_decimation=4,
        delta_action_scale=DELTA_ACTION_SCALE,
        delta_action_clip=DELTA_ACTION_CLIP,
        action_scale=ACTION_SCALE,
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
    action_term = action_manager.get_term("doorman")

    expected_action_dim = 8 if args_cli.hand_backend == "primitive" else 27
    if action_manager.total_action_dim != expected_action_dim:
        raise RuntimeError(
            f"Expected action dimension {expected_action_dim}, "
            f"got {action_manager.total_action_dim}."
        )

    joint_ids, joint_names = robot.find_joints(
        list(DOORMAN_ALL_DOF_NAMES),
        preserve_order=True,
    )
    if tuple(joint_names) != DOORMAN_ALL_DOF_NAMES:
        raise RuntimeError("Verification joint order mismatch.")
    limits = robot.data.soft_joint_pos_limits[:, joint_ids]

    left_hand_start = (
        len(DOORMAN_LOWER_BODY_DOF_NAMES)
        + len(DOORMAN_LEFT_ARM_DOF_NAMES)
        + len(DOORMAN_RIGHT_ARM_DOF_NAMES)
    )
    left_hand_end = left_hand_start + len(
        DOORMAN_LEFT_HAND_DOF_NAMES
    )
    hand_test_targets = _hand_test_targets(sim.device)
    expected_hand_targets = torch.clamp(
        hand_test_targets,
        min=limits[0, left_hand_start:left_hand_end, 0],
        max=limits[0, left_hand_start:left_hand_end, 1],
    )

    actual_hand_targets = []
    for test_index in range(3):
        action_manager.reset()
        actions = _make_action(
            args_cli.hand_backend,
            test_index,
            hand_test_targets,
        )
        action_manager.process_action(actions)
        targets = action_term.processed_actions

        torch.testing.assert_close(
            action_term.last_delta_actions,
            actions,
        )
        torch.testing.assert_close(
            action_term.delta_actions,
            torch.clamp(
                actions * DELTA_ACTION_SCALE,
                min=-DELTA_ACTION_CLIP,
                max=DELTA_ACTION_CLIP,
            ),
        )

        if targets.shape != (1, 69):
            raise RuntimeError(
                f"Expected target shape (1, 69), "
                f"got {tuple(targets.shape)}."
            )
        if not torch.isfinite(targets).all():
            raise RuntimeError(
                "Processed joint targets contain NaN or Inf."
            )
        if not torch.all(targets >= limits[..., 0]):
            raise RuntimeError(
                "A target is below its soft joint limit."
            )
        if not torch.all(targets <= limits[..., 1]):
            raise RuntimeError(
                "A target is above its soft joint limit."
            )

        actual_hand_targets.append(
            targets[0, left_hand_start:left_hand_end].clone()
        )
        for _ in range(4):
            action_manager.apply_action()
            robot.write_data_to_sim()
            sim.step()
            robot.update(PHYSICS_DT)

    torch.testing.assert_close(
        torch.stack(actual_hand_targets),
        expected_hand_targets,
    )

    action_manager.reset()
    unit_increment = torch.zeros(
        1,
        action_term.action_dim,
        device=sim.device,
    )
    unit_increment[:, 0] = 1.0
    action_manager.process_action(unit_increment)
    torch.testing.assert_close(
        action_term.delta_actions[:, 0],
        torch.tensor([0.3], device=sim.device),
    )
    action_manager.process_action(unit_increment)
    torch.testing.assert_close(
        action_term.delta_actions[:, 0],
        torch.tensor([0.6], device=sim.device),
    )
    clipping_increment = unit_increment.clone()
    clipping_increment[:, 0] = 100.0
    action_manager.process_action(clipping_increment)
    torch.testing.assert_close(
        action_term.delta_actions[:, 0],
        torch.tensor([DELTA_ACTION_CLIP], device=sim.device),
    )

    action_manager.reset()
    if not torch.all(action_term.raw_actions == 0.0):
        raise RuntimeError("Raw action buffer was not cleared on reset.")
    if not torch.all(action_term.delta_actions == 0.0):
        raise RuntimeError("Cumulative delta buffer was not cleared on reset.")
    if not torch.all(action_term.last_delta_actions == 0.0):
        raise RuntimeError("Last delta buffer was not cleared on reset.")
    if action_term.homie_controller is None:
        raise RuntimeError("HOMIE was not connected to the action term.")

    print(
        "ACTION_TERM_VERIFY_PASS "
        f"backend={args_cli.hand_backend} "
        f"action_dim={action_term.action_dim} "
        f"processed_dim={action_term.processed_actions.shape[-1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
