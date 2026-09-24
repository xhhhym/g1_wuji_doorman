"""Stage-conditioned rewards. Rates integrate with step_dt; bonuses cancel it."""
import torch


def door_task_reward(env, name: str):
    t = env.command_manager.get_term("door_task")
    r = t.robot.data
    stage = t.reward_stage
    contact = t.grasp_contact.float()
    near = torch.exp(-t.distance.square() / 0.2**2)
    if name == "reach":
        value = near * torch.where(stage == 0, 1.0, 0.25)
    elif name == "align":
        value = near * torch.exp(-t.orientation_error.square() / 0.6**2)
    elif name == "open_hand":
        value = (stage == 0) * near * (1 - t.closure)
    elif name == "contact":
        value = (stage >= 1) * t.contact_count / 5
    elif name == "closure":
        value = (stage >= 1) * (t.contact_count >= 2) * near * t.closure
    elif name == "handle_amount":
        value = (stage == 2) * contact * (t.door_state[:, 2] / 0.785398).clamp(0, 1)
    elif name == "handle_progress":
        value = (stage == 2) * contact * t.progress[:, 1].clamp_min(0) / env.step_dt
    elif name == "handle_regression":
        value = (stage == 2) * (-t.progress[:, 1]).clamp_min(0) / env.step_dt
    elif name == "latch":
        value = (stage == 2) * contact * (t.door_state[:, 4] / 0.03).clamp(0, 1)
    elif name == "door_progress":
        value = (stage == 3) * contact * t.progress[:, 0].clamp_min(0) / env.step_dt
    elif name == "door_amount":
        value = (stage == 3) * (t.door_state[:, 0] / t.cfg.success_angle).clamp(0, 1)
    elif name == "door_regression":
        value = (stage == 3) * (-t.progress[:, 0]).clamp_min(0) / env.step_dt
    elif name == "held_open":
        value = (stage == 3) * (t.success_hold * env.step_dt / t.cfg.success_hold_s).clamp(0, 1)
    elif name == "transition":
        value = t.transition.float() / env.step_dt
    elif name == "success":
        value = t.success.float() / env.step_dt
    elif name == "failure":
        value = (t.fallen | t.invalid).float() / env.step_dt
    elif name == "upright":
        value = r.projected_gravity_b[:, :2].square().sum(-1)
    elif name == "base_drift":
        value = (r.root_pos_w[:, :2] - t.initial_root_xy).square().sum(-1)
    elif name == "right_arm_rest":
        value = (r.joint_pos[:, t.right_arm_ids] - r.default_joint_pos[:, t.right_arm_ids]).square().sum(-1)
    elif name == "action_rate":
        value = (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)
    elif name == "joint_velocity":
        value = r.joint_vel[:, t.arm_ids + t.hand_ids].square().sum(-1)
    elif name == "joint_limits":
        value = ((r.soft_joint_pos_limits[..., 0] - r.joint_pos).clamp_min(0)
                 + (r.joint_pos - r.soft_joint_pos_limits[..., 1]).clamp_min(0)).sum(-1)
    elif name == "excess_force":
        value = (t.link_forces - 30).clamp_min(0).square().sum((1, 2)) / 900
    elif name == "lost_contact":
        value = (stage >= 2) * (1 - contact)
    else:
        raise ValueError(name)
    value = torch.nan_to_num(value, nan=0, posinf=0, neginf=0)
    return value if name == "failure" else torch.where(t.invalid, 0.0, value)
