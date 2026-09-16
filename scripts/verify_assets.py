"""Visualize and dynamically verify the G1+Wuji articulation."""

import argparse
import math

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Static and dynamic verification for the G1+Wuji robot."
)

parser.add_argument(
    "--mode",
    type=str,
    default="static",
    choices=("static", "joint_motion", "floating_base"),
    help=(
        "Verification mode: "
        "'static' keeps the base fixed with gravity disabled; "
        "'joint_motion' moves selected joints with the base fixed; "
        "'floating_base' releases the base and enables gravity."
    ),
)

parser.add_argument(
    "--joint_pattern",
    action="append",
    default=None,
    help=(
        "Regular expression selecting joints in joint_motion mode. "
        "May be supplied multiple times. "
        "Default: '.*_shoulder_pitch_joint'."
    ),
)

parser.add_argument(
    "--amplitude",
    type=float,
    default=0.15,
    help="Joint-motion amplitude in radians. Default: 0.15.",
)

parser.add_argument(
    "--frequency",
    type=float,
    default=0.25,
    help="Joint-motion frequency in Hz. Default: 0.25.",
)

parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help=(
        "Maximum number of simulation steps. "
        "Use 0 to keep running until the window is closed."
    ),
)

parser.add_argument(
    "--print_joint_names",
    action="store_true",
    help="Print every articulation joint name after initialization.",
)

parser.add_argument(
    "--print_interval",
    type=int,
    default=120,
    help="Number of simulation steps between diagnostic messages.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Isaac Lab imports must come after AppLauncher
# -----------------------------------------------------------------------------

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from g1_wuji_doorman.assets.robots import G1_WUJI_CFG


# -----------------------------------------------------------------------------
# Scene creation
# -----------------------------------------------------------------------------

def create_robot() -> Articulation:
    """Create a mode-specific copy of the G1+Wuji configuration."""

    robot_cfg = G1_WUJI_CFG.replace(
        prim_path="/World/Robot",
    )

    if args_cli.mode == "floating_base":
        # Dynamic whole-body physics:
        # release the root and enable gravity.
        robot_cfg.spawn.articulation_props.fix_root_link = False
        robot_cfg.spawn.rigid_props.disable_gravity = False
    else:
        # Static inspection and isolated joint-motion testing:
        # keep the root fixed and disable gravity.
        robot_cfg.spawn.articulation_props.fix_root_link = True
        robot_cfg.spawn.rigid_props.disable_gravity = True

    return Articulation(cfg=robot_cfg)


def reset_robot(robot: Articulation) -> None:
    """Reset the robot to the default state from G1_WUJI_CFG."""

    root_state = robot.data.default_root_state.clone()

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # Clear cached states and actuator buffers.
    robot.reset()


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def check_finite_state(robot: Articulation) -> None:
    """Stop immediately if the simulation generates NaN or Inf."""

    state_tensors = {
        "root position": robot.data.root_pos_w,
        "root orientation": robot.data.root_quat_w,
        "root linear velocity": robot.data.root_lin_vel_w,
        "root angular velocity": robot.data.root_ang_vel_w,
        "joint position": robot.data.joint_pos,
        "joint velocity": robot.data.joint_vel,
    }

    for state_name, state_tensor in state_tensors.items():
        if not torch.isfinite(state_tensor).all():
            raise RuntimeError(
                f"Non-finite value detected in {state_name}. "
                "The articulation may be numerically unstable."
            )


def print_diagnostics(
    robot: Articulation,
    joint_targets: torch.Tensor,
    step_count: int,
) -> None:
    """Print basic physical-state diagnostics."""

    root_height = robot.data.root_pos_w[0, 2].item()
    max_joint_velocity = robot.data.joint_vel.abs().max().item()
    max_tracking_error = (
        joint_targets - robot.data.joint_pos
    ).abs().max().item()

    print(
        f"[STEP {step_count:06d}] "
        f"root_z={root_height:.4f} m | "
        f"max_joint_vel={max_joint_velocity:.4f} rad/s | "
        f"max_position_error={max_tracking_error:.4f} rad"
    )


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def run_simulator(
    sim: SimulationContext,
    robot: Articulation,
) -> None:
    """Run the selected verification mode."""

    sim_dt = sim.get_physics_dt()

    default_joint_pos = robot.data.default_joint_pos.clone()
    joint_targets = default_joint_pos.clone()

    selected_joint_ids: list[int] = []
    selected_joint_names: list[str] = []

    if args_cli.mode == "joint_motion":
        joint_patterns = args_cli.joint_pattern

        if joint_patterns is None:
            joint_patterns = [".*_shoulder_pitch_joint"]

        selected_joint_ids, selected_joint_names = robot.find_joints(
            joint_patterns
        )

        if len(selected_joint_ids) == 0:
            raise ValueError(
                "No joints matched --joint_pattern. "
                "Run with --print_joint_names to inspect available names."
            )

        print("[INFO] Moving the following joints:")

        for joint_id, joint_name in zip(
            selected_joint_ids,
            selected_joint_names,
        ):
            print(f"  {joint_id:02d}: {joint_name}")

        print(
            f"[INFO] Motion amplitude: {args_cli.amplitude:.4f} rad"
        )
        print(
            f"[INFO] Motion frequency: {args_cli.frequency:.4f} Hz"
        )

    if args_cli.mode == "static":
        print("[INFO] Static mode:")
        print("       root fixed, gravity disabled, default pose held.")

    elif args_cli.mode == "joint_motion":
        print("[INFO] Joint-motion mode:")
        print("       root fixed, gravity disabled, selected joints moving.")

    elif args_cli.mode == "floating_base":
        print("[INFO] Floating-base mode:")
        print("       root released, gravity enabled, default joint pose held.")
        print(
            "[INFO] The robot may fall because no balance controller "
            "is running. Falling alone is not an asset failure."
        )

    step_count = 0

    while simulation_app.is_running():
        if args_cli.steps > 0 and step_count >= args_cli.steps:
            print(
                f"[INFO] Reached requested step count: {args_cli.steps}"
            )
            break

        # Start every step from the default pose target.
        joint_targets[:] = default_joint_pos

        if args_cli.mode == "joint_motion":
            elapsed_time = step_count * sim_dt

            position_offset = (
                args_cli.amplitude
                * math.sin(
                    2.0
                    * math.pi
                    * args_cli.frequency
                    * elapsed_time
                )
            )

            joint_targets[:, selected_joint_ids] += position_offset

        # Keep every commanded position inside the soft joint limits.
        lower_limits = robot.data.soft_joint_pos_limits[..., 0]
        upper_limits = robot.data.soft_joint_pos_limits[..., 1]

        joint_targets.clamp_(
            min=lower_limits,
            max=upper_limits,
        )

        # Send position targets to the implicit actuators.
        robot.set_joint_position_target(joint_targets)

        # Transfer actuator commands to PhysX.
        robot.write_data_to_sim()

        # Advance physics.
        sim.step()

        # Refresh Isaac Lab state buffers.
        robot.update(sim_dt)

        check_finite_state(robot)

        if (
            args_cli.print_interval > 0
            and step_count % args_cli.print_interval == 0
        ):
            print_diagnostics(
                robot=robot,
                joint_targets=joint_targets,
                step_count=step_count,
            )

        step_count += 1


def main() -> None:
    """Create the scene and run the selected verification mode."""

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=(3.0, 3.0, 2.0),
        target=(0.0, 0.0, 1.0),
    )

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    # Lighting
    light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/Light", light_cfg)

    robot = create_robot()

    # Initialize PhysX handles and Isaac Lab buffers.
    sim.reset()

    print("[INFO] G1+Wuji initialized successfully.")
    print(f"[INFO] Verification mode: {args_cli.mode}")
    print(f"[INFO] Number of joints: {robot.num_joints}")
    print(f"[INFO] Number of bodies: {robot.num_bodies}")

    if args_cli.print_joint_names:
        print("[INFO] Joint names:")

        for joint_index, joint_name in enumerate(robot.joint_names):
            print(f"  {joint_index:02d}: {joint_name}")

    reset_robot(robot)

    run_simulator(
        sim=sim,
        robot=robot,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()