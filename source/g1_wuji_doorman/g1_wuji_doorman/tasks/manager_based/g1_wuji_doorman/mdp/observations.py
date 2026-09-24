"""DoorMan-style asymmetric teacher observations."""
import torch


def door_task_observation(env, name: str):
    task = env.command_manager.get_term("door_task")
    robot = env.scene["robot"].data
    if name == "stage":
        value = task.command
    elif name == "gravity":
        value = robot.projected_gravity_b
    elif name == "base_ang_vel":
        value = robot.root_ang_vel_b * 0.25
    elif name == "base_lin_vel":
        value = robot.root_lin_vel_b
    elif name == "root_in_door":
        value = task.root_in_door
    elif name in ("arm_pos", "hand_pos"):
        ids = task.arm_ids if name == "arm_pos" else task.hand_ids
        value = robot.joint_pos[:, ids] - robot.default_joint_pos[:, ids]
    elif name in ("arm_vel", "hand_vel"):
        ids = task.arm_ids if name == "arm_vel" else task.hand_ids
        value = robot.joint_vel[:, ids] * 0.05
    elif name == "target_in_palm":
        value = task.target_in_palm
    elif name == "door_state":
        value = task.door_state.clone()
        value[:, [1, 3]] *= 0.1
        value[:, 4] /= task.latch_travel
    elif name == "tip_forces":
        value = task.tip_forces.clamp(0, 50) / 50
    elif name == "finger_forces":
        value = task.finger_forces.clamp(0, 50) / 50
    elif name == "actions":
        # Original DoorMan: controller-space commands, before primitive expansion.
        value = env.action_manager.get_term("doorman").controller_actions
    elif name == "delta_actions":
        # Original _get_obs_delta_actions returns the raw increment, not its integral.
        value = env.action_manager.get_term("doorman").last_delta_actions
    elif name == "privileged_door_info":
        value = task.privileged_door_info
    elif name == "full_handle_contact":
        # Exact filtered world-frame forces for all 25 left-hand links.
        value = task.contact_forces_w.flatten(1) * 0.01
    elif name == "teacher_task_state":
        stage_timeout = task._timeouts[task.stage]
        value = torch.stack(
            (
                task.transition.float(),
                task.success.float(),
                task.time_in_stage / stage_timeout,
                env.episode_length_buf.float() * env.step_dt / env.max_episode_length_s,
                task.success_hold.float() * env.step_dt / task.cfg.success_hold_s,
            ),
            dim=-1,
        )
    else:
        raise ValueError(name)
    # Invalid physical states terminate before PPO receives the next observation.
    # DoorMan action observations use scale=1; do not erase cumulative values 10..15.
    bound = 100 if name in ("actions", "delta_actions") else 10
    return torch.nan_to_num(value, nan=0, posinf=bound, neginf=-bound).clamp(-bound, bound)
