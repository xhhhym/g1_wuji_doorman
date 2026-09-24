"""Verify real two-env observations/resets plus controlled stage/reward semantics."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, quat_apply, quat_mul
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman.g1_wuji_doorman_env_cfg import G1WujiDoormanEnvCfg
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman.mdp.task_rewards import door_task_reward
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman.mdp.observations import door_task_observation
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman.mdp.terminations import task_success, task_fall, task_stage_timeout, task_invalid_state
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman.mdp.contracts import contact_filter_indices
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman.mdp.commands import FINGER_LINKS


def main():
    cfg = G1WujiDoormanEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device
    cfg.seed = 42
    # Verify teacher metadata follows the spawned door, not a hard-coded vector.
    cfg.scene.door.spawn.rand_door_width = 0.9
    cfg.scene.door.spawn.rand_door_weight = 12.0
    env = ManagerBasedRLEnv(cfg=cfg)
    assert callable(torch.distributions.Normal.set_default_validate_args)
    obs, _ = env.reset()
    t = env.command_manager.get_term("door_task")
    assert obs["policy"].shape == (2, 139)
    assert obs["critic"].shape == (2, 219)
    expected_door_info = torch.tensor(
        [0.9, 2.0, 0.8, 0.23, 0.12, -1.0, 2.0, -1.0], device=env.device
    )
    torch.testing.assert_close(t.privileged_door_info[0], expected_door_info)
    assert env.action_manager.total_action_dim == 8
    assert env.scene["handle_contacts"].data.force_matrix_w.shape == (2, 1, 25, 3)
    view = env.scene["handle_contacts"].contact_physx_view
    reversed_rows = [list(reversed(row)) for row in view.filter_paths]
    assert contact_filter_indices(view.sensor_paths, reversed_rows, FINGER_LINKS) == [list(range(24, -1, -1))] * 2
    bad_rows = [row.copy() for row in reversed_rows]
    bad_rows[0][0] = bad_rows[0][1]
    try:
        contact_filter_indices(view.sensor_paths, bad_rows, FINGER_LINKS)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Duplicate contact filters must fail")
    action_term = env.action_manager.get_term("doorman")
    default_targets = action_term.processed_actions.clone()
    env.action_manager.process_action(torch.zeros(2, 8, device=env.device))
    # Zero action after reset must not jump the hand target.
    torch.testing.assert_close(action_term.processed_actions, default_targets)
    increment = torch.zeros(2, 8, device=env.device)
    increment[:, 0] = 1
    env.action_manager.process_action(increment)
    torch.testing.assert_close(door_task_observation(env, "delta_actions"), increment)
    torch.testing.assert_close(door_task_observation(env, "actions")[:, 15:], 0.3 * increment)
    # Preserve cumulative values beyond 10; the action observation clip is DoorMan's 100.
    action_term._delta_actions[:, 0] = 12.0
    assert (door_task_observation(env, "actions")[:, 15] == 12).all()
    env.reset()
    # Relative transforms must be invariant to arbitrary common world translations.
    r, d = t.robot.data, t.door.data
    p, q = r.body_pos_w[:, t.palm_id], r.body_quat_w[:, t.palm_id]
    target, target_q = d.body_pos_w[:, t.target_id], d.body_quat_w[:, t.target_id]
    offset = torch.tensor([[12., -7., 3.], [-4., 8., 0.]], device=env.device)
    base = subtract_frame_transforms(p, q, target, target_q)
    moved = subtract_frame_transforms(p + offset, q, target + offset, target_q)
    for a, b in zip(base, moved):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    # A common world rotation must preserve both relative position and quaternion.
    z = torch.zeros(2, device=env.device)
    rotation = quat_from_euler_xyz(z, z, z + 0.7)
    rotated = subtract_frame_transforms(quat_apply(rotation, p), quat_mul(rotation, q),
                                       quat_apply(rotation, target), quat_mul(rotation, target_q))
    for a, b in zip(base, rotated):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    # Controlled signal tests exercise the actual state machine, not physical grasping.
    refresh = t.refresh
    t.refresh = lambda: None
    try:
        t.distance[:] = torch.tensor([0.01, 1.0], device=env.device)
        t.orientation_error[:] = 0
        t.closure[:] = 0
        t.fallen[:] = False
        t.invalid[:] = False
        t.grasp_contact[:] = False
        t.door_state[:] = 0
        for expected in (1, 2, 3):
            if expected >= 2:
                t.grasp_contact[0] = True
                t.contact_count[0] = 3
            if expected == 3:
                t.door_state[0, 4] = 0.025
            for _ in range(t.cfg.transition_hold_steps):
                env.common_step_counter += 1
                # Deliberately query success before invalid_state; all consumers agree.
                task_success(env)
                hold = t.hold.clone()
                stage = t.stage.clone()
                task_fall(env)
                task_stage_timeout(env)
                task_invalid_state(env)
                assert torch.equal(t.hold, hold) and torch.equal(t.stage, stage)
            assert t.stage.tolist() == [expected, 0], t.stage
            assert t.transition.tolist() == [True, False]
        t.door_state[0, 0] = t.cfg.success_angle + 0.01
        for _ in range(13):
            env.common_step_counter += 1
            t.advance()
        assert t.success.tolist() == [True, False]
        assert door_task_reward(env, "success")[0] * env.step_dt == 1
        # Timeout cannot grant a stage transition or success on that same step.
        t.time_in_stage[1] = t.cfg.stage_timeouts_s[0]
        env.common_step_counter += 1
        t.advance()
        assert t.stage_timeout[1] and not t.success[1]
        # Reward direction: approach, alignment, closure gating, signed progress.
        t.reward_stage[:] = 0
        t.distance[:] = torch.tensor([0.01, 0.3], device=env.device)
        assert door_task_reward(env, "reach")[0] > door_task_reward(env, "reach")[1]
        t.distance[:] = 0.01
        t.orientation_error[:] = torch.tensor([0.0, 1.0], device=env.device)
        assert door_task_reward(env, "align")[0] > door_task_reward(env, "align")[1]
        t.closure[:] = torch.tensor([0., 1.], device=env.device)
        assert door_task_reward(env, "open_hand")[0] > door_task_reward(env, "open_hand")[1]
        t.reward_stage[:] = 1
        t.closure[:] = 1
        t.contact_count[:] = torch.tensor([3, 0], device=env.device)
        assert door_task_reward(env, "closure")[0] > 0
        assert door_task_reward(env, "closure")[1] == 0
        t.grasp_contact[:] = True
        for stage, prefix, col in ((2, "handle", 1), (3, "door", 0)):
            t.reward_stage[:] = stage
            t.progress[:, col] = torch.tensor([0.01, -0.01], device=env.device)
            assert door_task_reward(env, prefix + "_progress")[0] > 0
            assert door_task_reward(env, prefix + "_progress")[1] == 0
            assert door_task_reward(env, prefix + "_regression")[1] > 0
            assert door_task_reward(env, prefix + "_regression")[0] == 0
    finally:
        t.refresh = refresh
    env.reset()
    # Partial reset must not wipe the other environment's task/action state.
    env.action_manager.process_action(torch.ones(2, 8, device=env.device))
    t.stage[:] = 2
    t.hold[:] = 3
    t.progress[:] = 0.1
    action_term._homie_step_counter[:] = 3
    homie = action_term.homie_controller
    homie._history[1] = 0.125
    env._reset_idx(torch.tensor([0], device=env.device))
    assert action_term._homie_step_counter.tolist() == [0, 3]
    assert not homie._history[0].any() and (homie._history[1] == 0.125).all()
    # Only env 0 updates on this substep; env 1's history is untouched.
    action_term.apply_actions()
    assert (homie._history[1] == 0.125).all()
    assert t.stage.tolist() == [0, 2] and t.hold.tolist() == [0, 3]
    assert not t.progress[0].any() and t.progress[1].all()
    action_term = env.action_manager.get_term("doorman")
    assert not action_term.delta_actions[0].any() and action_term.delta_actions[1].all()
    env.reset()
    # Non-finite policy output is isolated and marked for termination, not sent to PhysX.
    bad_actions = torch.zeros(2, 8, device=env.device)
    bad_actions[0, 0] = float("nan")
    env.action_manager.process_action(bad_actions)
    assert action_term.invalid_state.tolist() == [True, False]
    assert torch.isfinite(action_term.processed_actions).all()
    t.refresh()
    assert t.invalid.tolist() == [True, False]
    env.reset()
    # HOMIE non-finite input is isolated per env; a valid peer must remain usable.
    positions = homie.default_joint_pos.clone()
    positions[0, 0] = float("nan")
    targets = homie.compute_joint_targets(positions, torch.zeros_like(positions),
                                         torch.zeros(2, 3, device=env.device),
                                         torch.tensor([[0., 0., -1.]] * 2, device=env.device))
    assert homie.invalid_state.tolist() == [True, False] and torch.isfinite(targets).all()
    env.reset()
    # Real physics: finite observation/rewards, auto-reset, stationary HOMIE baseline.
    resets = 0
    for _ in range(210):
        obs, reward, terminated, truncated, _ = env.step(torch.zeros(2, 8, device=env.device))
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
        assert torch.isfinite(reward).all()
        resets += int((terminated | truncated).sum())
    assert resets >= 2, "8-second PREGRASP timeout should reset both environments"
    print(
        f"TASK_MDP_VERIFY_PASS envs=2 actor_obs=139 critic_obs=219 action=8 "
        f"door_privileged=8 contacts=25 transitions=3 success_hold=13 resets={resets}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Also clean up a partially constructed environment after an assertion/error.
        from isaaclab.sim import SimulationContext
        sim = SimulationContext.instance()
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
        app.close()
