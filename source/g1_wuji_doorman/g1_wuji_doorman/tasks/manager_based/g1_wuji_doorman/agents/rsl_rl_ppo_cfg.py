# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # This task accumulates policy outputs as joint-target increments. Keep one
    # sampled increment bounded even if the Gaussian mean becomes unstable.
    clip_actions = 1.0
    # Match the DoorMan recurrent teacher's temporal batch length.
    num_steps_per_env = 64
    max_iterations = 150
    save_interval = 50
    experiment_name = "g1_wuji_doorman"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticRecurrentCfg(
        # Lower than DoorMan's 0.8 because this policy controls cumulative arm
        # increments and upstream RSL-RL does not expose DoorMan's max-std clamp.
        init_noise_std=0.3,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        # DoorMan teacher: separate two-layer, 256-unit LSTMs before the MLPs.
        # RSL-RL calls PyTorch SiLU "swish".
        activation="swish",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Two-env diagnostics showed std growth and exploding action penalties.
        # Large cloud batches provide exploration without a strong entropy push.
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.9975,
        lam=0.985,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
