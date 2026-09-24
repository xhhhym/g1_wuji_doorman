"""Verify the real RSL-RL LSTM interface, memory/reset semantics, and a PPO update."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.algorithms import PPO
from g1_wuji_doorman.tasks.manager_based.g1_wuji_doorman.agents.rsl_rl_ppo_cfg import PPORunnerCfg


def main():
    cfg = PPORunnerCfg()
    torch.manual_seed(42)
    obs = TensorDict({"policy": torch.randn(2, 139), "critic": torch.randn(2, 219)}, batch_size=[2]).to(args.device)
    policy_cfg = cfg.policy.to_dict()
    assert policy_cfg.pop("class_name") == "ActorCriticRecurrent"
    policy = ActorCriticRecurrent(obs, cfg.obs_groups, 8, **policy_cfg).to(args.device)
    assert policy.memory_a.rnn.num_layers == policy.memory_c.rnn.num_layers == 2
    assert policy.memory_a.rnn.hidden_size == policy.memory_c.rnn.hidden_size == 256
    with torch.inference_mode():
        first = policy.act_inference(obs).clone()
        policy.evaluate(obs)
        first_memory = tuple(x.clone() for x in policy.memory_a.hidden_states)
        policy.act_inference(obs)
        policy.evaluate(obs)
        assert any(not torch.equal(a, b) for a, b in zip(first_memory, policy.memory_a.hidden_states))
        peer_a = tuple(x[:, 1].clone() for x in policy.memory_a.hidden_states)
        peer_c = tuple(x[:, 1].clone() for x in policy.memory_c.hidden_states)
        policy.reset(torch.tensor([True, False], device=args.device))
        for state, peer in zip(policy.memory_a.hidden_states, peer_a):
            assert not state[:, 0].any()
            torch.testing.assert_close(state[:, 1], peer)
        for state, peer in zip(policy.memory_c.hidden_states, peer_c):
            assert not state[:, 0].any()
            torch.testing.assert_close(state[:, 1], peer)
        replay = policy.act_inference(obs)
        torch.testing.assert_close(replay[0], first[0])
    # Fresh memories outside inference_mode; two environments, two sequence minibatches.
    policy = ActorCriticRecurrent(obs, cfg.obs_groups, 8, **policy_cfg).to(args.device)
    algo_cfg = cfg.algorithm.to_dict()
    algo_cfg.pop("class_name")
    algo_cfg.update(num_mini_batches=2, num_learning_epochs=1)
    algo = PPO(policy, device=args.device, **algo_cfg)
    algo.init_storage("rl", 2, 8, obs, [8])
    before = policy.memory_a.rnn.weight_ih_l0.detach().clone()
    for step in range(8):
        with torch.no_grad():
            algo.act(obs)
            dones = torch.tensor([step == 3, step == 5], device=args.device)
            algo.process_env_step(obs, torch.ones(2, device=args.device), dones, {})
    with torch.no_grad():
        algo.compute_returns(obs)
    loss = algo.update()
    assert all(torch.isfinite(p).all() for p in policy.parameters())
    assert not torch.equal(before, policy.memory_a.rnn.weight_ih_l0)
    print(f"RECURRENT_POLICY_VERIFY_PASS actor=139 critic=219 lstm=2x256 partial_reset=True ppo_update={loss}", flush=True)


try:
    main()
finally:
    app.close()
