"""Load G1+Wuji and the Doorman door, then demonstrate the door motion.

The robot is held at its configured default pose.  The door first settles with
the handle released, then the handle is pressed to retract the latch, the door
is pushed open with a physical torque, and finally the handle is released.
"""

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Visual G1+Wuji and Doorman handle/open-door verification."
)
parser.add_argument(
    "--push_torque",
    type=float,
    default=20.0,
    help="World-Z torque applied to the door panel in N m. Default: 20.0.",
)
parser.add_argument(
    "--phase_seconds",
    type=float,
    default=3.0,
    help="Duration of the press, open, and release phases. Default: 3.0.",
)
parser.add_argument(
    "--exit_after_test",
    action="store_true",
    help="Close Isaac Sim when the demonstration finishes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Lab imports must come after AppLauncher.
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse

from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman.g1_wuji_doorman_env_cfg import (
    G1WujiDoormanSceneCfg,
)


DT = 1.0 / 120.0
REST_HANDLE_DEG = -15.0
PRESSED_HANDLE_DEG = 45.0


def single_id(ids: list[int] | slice, names: list[str], label: str) -> int:
    """Require exactly one articulation item for a verification target."""

    if isinstance(ids, slice) or len(ids) != 1:
        raise RuntimeError(f"Expected one {label}, found: {names}")
    return ids[0]


def main() -> None:
    """Run the combined robot-and-door visual demonstration."""

    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args_cli.device))
    sim.set_camera_view(eye=(2.3, -2.4, 1.55), target=(0.3, 0.0, 0.9))

    scene_cfg = G1WujiDoormanSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(DT)

    robot = scene["robot"]
    door = scene["door"]

    hinge_ids, hinge_names = door.find_joints(".*hinge.*")
    handle_ids, handle_names = door.find_joints(".*handle.*")
    panel_ids, panel_names = door.find_bodies("door_panel")
    latch_ids, latch_names = door.find_bodies("latch_link")
    handle_body_ids, handle_body_names = door.find_bodies("door_handle")
    left_palm_ids, left_palm_names = robot.find_bodies("left_palm_link")

    hinge_id = single_id(hinge_ids, hinge_names, "hinge joint")
    handle_id = single_id(handle_ids, handle_names, "handle joint")
    panel_id = single_id(panel_ids, panel_names, "door panel body")
    latch_id = single_id(latch_ids, latch_names, "latch body")
    handle_body_id = single_id(handle_body_ids, handle_body_names, "door handle body")
    left_palm_id = single_id(left_palm_ids, left_palm_names, "left palm body")

    device = door.device
    robot_target = robot.data.default_joint_pos.clone()
    zero_force = torch.zeros((scene.num_envs, 1, 3), device=device)
    panel_torque = torch.zeros((scene.num_envs, 1, 3), device=device)

    def latch_position_in_panel() -> torch.Tensor:
        panel_pos = door.data.body_pos_w[0, panel_id]
        panel_quat = door.data.body_quat_w[0, panel_id]
        latch_pos = door.data.body_pos_w[0, latch_id]
        return quat_apply_inverse(panel_quat, latch_pos - panel_pos)

    def advance(seconds: float, handle_deg: float, torque_z: float = 0.0) -> None:
        steps = max(1, round(seconds / DT))
        handle_target = torch.full(
            (scene.num_envs, 1),
            math.radians(handle_deg),
            dtype=torch.float32,
            device=device,
        )
        panel_torque.zero_()
        panel_torque[..., 2] = torque_z

        for _ in range(steps):
            # Keep G1+Wuji fixed in its configured default pose.  This script
            # verifies co-loading and door mechanics, not a learned policy.
            robot.set_joint_position_target(robot_target)
            door.set_joint_position_target(handle_target, joint_ids=[handle_id])
            door.set_external_force_and_torque(
                zero_force,
                panel_torque,
                body_ids=[panel_id],
                is_global=True,
            )
            scene.write_data_to_sim()
            sim.step()
            scene.update(DT)

    print("\n[PHASE 1/4] Robot and door loaded; settling with handle released.")
    advance(2.0, REST_HANDLE_DEG)
    latch_rest = latch_position_in_panel().clone()
    robot_root_x = robot.data.root_pos_w[0, 0].item()
    door_root_x = door.data.root_pos_w[0, 0].item()
    panel_closed_x = door.data.body_pos_w[0, panel_id, 0].item()
    left_palm_y = robot.data.body_pos_w[0, left_palm_id, 1].item()
    handle_y = door.data.body_pos_w[0, handle_body_id, 1].item()
    palm_handle_y_error = handle_y - left_palm_y

    print("[PHASE 2/4] Pressing the door handle to 45 degrees.")
    advance(args_cli.phase_seconds, PRESSED_HANDLE_DEG)
    pressed_handle_deg = math.degrees(door.data.joint_pos[0, handle_id].item())
    latch_pressed = latch_position_in_panel().clone()
    latch_travel = torch.linalg.vector_norm(latch_pressed - latch_rest).item()

    print("[PHASE 3/4] Holding the handle down and pushing the door open.")
    advance(args_cli.phase_seconds, PRESSED_HANDLE_DEG, args_cli.push_torque)
    opened_hinge_deg = math.degrees(door.data.joint_pos[0, hinge_id].item())
    panel_open_delta_x = door.data.body_pos_w[0, panel_id, 0].item() - panel_closed_x

    print("[PHASE 4/4] Removing the push and releasing the handle.")
    advance(args_cli.phase_seconds, REST_HANDLE_DEG)
    returned_handle_deg = math.degrees(door.data.joint_pos[0, handle_id].item())

    handle_pass = abs(pressed_handle_deg - PRESSED_HANDLE_DEG) < 2.0
    latch_pass = latch_travel > 0.025
    door_pass = opened_hinge_deg > 20.0
    return_pass = abs(returned_handle_deg) < 2.0
    station_pass = door_root_x > robot_root_x
    opens_away_pass = panel_open_delta_x > 0.05
    overall_pass = (
        station_pass
        and handle_pass
        and latch_pass
        and door_pass
        and opens_away_pass
        and return_pass
    )

    print("\n============== ROBOT + DOOR MOTION RESULT ==============")
    print(f"Robot loaded:    bodies={robot.num_bodies}, joints={robot.num_joints}")
    print(
        f"Default station: robot_x={robot_root_x:6.3f} m, "
        f"door_x={door_root_x:6.3f} m  "
        f"[{'PASS' if station_pass else 'FAIL'}]"
    )
    print(
        f"Left-palm Y:     {left_palm_y:7.3f} m, "
        f"handle_y={handle_y:7.3f} m, error={palm_handle_y_error:+7.3f} m"
    )
    print(
        f"Handle pressed:  {pressed_handle_deg:7.3f} deg  "
        f"[{'PASS' if handle_pass else 'FAIL'}]"
    )
    print(
        f"Latch travel:    {latch_travel * 1000.0:7.3f} mm  "
        f"[{'PASS' if latch_pass else 'FAIL'}]"
    )
    print(
        f"Door opened:     {opened_hinge_deg:7.3f} deg  "
        f"[{'PASS' if door_pass else 'FAIL'}]"
    )
    print(
        f"Panel delta X:   {panel_open_delta_x:7.3f} m  "
        f"[{'PASS' if opens_away_pass else 'FAIL'}]"
    )
    print(
        f"Handle returned: {returned_handle_deg:7.3f} deg  "
        f"[{'PASS' if return_pass else 'FAIL'}]"
    )
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("========================================================\n")

    if args_cli.exit_after_test:
        return

    print("[INFO] Demonstration finished; the Isaac Sim window will remain open.")
    while simulation_app.is_running():
        advance(DT, REST_HANDLE_DEG)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
