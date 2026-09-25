# Doorman 时序与控制观测对齐

2026-09-25，基于提交 `9f90ebb85952ac9a4db4c7e2195c9b3eaad7cf87` 后的修改。原版对照为本机 `GR00T-VisualSim2Real-original` 的 `016c70c1e4e76f521963c36691ee69a6ab3ac9cd`。本记录描述实现和接线验证，不代表策略已成功开门。

## 当前接口

- 任务仍是自由基座站在门前、左手操作；PREGRASP → GRASP → UNLATCH → OPEN_DOOR，无走路/穿门阶段；HOMIE 速度命令为零。
- `PPORunnerCfg.policy` 使用 `ActorCriticRecurrent`：actor/critic 各自 LSTM 2×256，后接 MLP [512,256,128]、SiLU（RSL-RL 名称 swish）。对应原版 `door_open_homie_lstm.yaml` 的网络结构。
- 高层 25 Hz、物理 200 Hz；高层用 LSTM 隐状态表达历史，不额外堆叠当前帧。HOMIE 保留独立 50 Hz、6×86 历史。
- primitive action=8；actor 当前帧139维、critic219维。旧111/191维 MLP checkpoint 不兼容，需要新训。

观测 `actions` 为 `[HOMIE raw15, cumulative_left_arm7, cumulative_left_hand1]`，共23维，尚未展开手primitive，也未裁剪到关节软限位。`delta_actions` 为刚执行的原始8维高层增量。命名对应原版 `_get_obs_actions` / `_get_obs_delta_actions`：后者**不是累计量**。两项scale=1、clip=100，避免抹平累计10–15的区别。固定右臂/右手和行走速度通道不纳入当前动作观测。

`action_rate` 已按原版 `_reward_penalty_delta_action_rate` 改为原始增量平方和；权重仍−.01，不改其他24项权重。actor 新增5维 `finger_forces`，与阶段判据读取同一整指最大接触力。JointTarget 后端 action=27，两项动作观测各增加19维，推导 actor177、critic257；本次验证了该后端目标映射，未跑其完整PPO训练。

## 修正入口

路径均相对仓库根目录；任务目录为 `source/g1_wuji_doorman/g1_wuji_doorman/tasks/manager_based/g1_wuji_doorman`。

| 文件 / 符号 | 修改 |
|---|---|
| `mdp/terminations.py::_task`、`mdp/task_rewards.py::door_task_reward` | 每个消费者均请求幂等 `DoorTaskState.advance`，移除done声明顺序依赖 |
| `mdp/contracts.py::contact_filter_indices` | 核对真实PhysX sensor/filter路径，按25link规范顺序重排；重复/缺失直接报错 |
| `mdp/contracts.py::door_info_from_metadata` | 从每env门USD customData读取8D参数；未标定的非right/out方向直接拒绝 |
| `mdp/commands.py::DoorTaskState` | 实际关节行程用于归一化及80%解锁阈值；拓宽非有限检查，增加base_drift/foot_speed/invalid_state指标 |
| `mdp/actions.py::G1WujiDoormanAction` | 非有限action隔离；每env HOMIE子步时钟；subset计算/局部reset不会改变peer状态；公开controller_actions |
| `controllers/standing/homie.py::HomieController` | 子集历史更新，异常输入/输出按env标记到reset，提供有限回退目标 |
| `controllers/standing/homie_cfg.py::default_homie_checkpoint` | 支持G1_HOMIE_CHECKPOINT；默认加载包内models/model_stand.pt；ActionCfg也可给homie_checkpoint_path |
| `controllers/standing/homie_modules.py` | 从DoorMan/GR00T迁入checkpoint兼容的HOMIE推理网络；运行时不再依赖gr00t Python包；新旧网络固定输入输出逐元素一致 |
| `assets/robots/g1_wuji.py` | 自由基座默认左手与primitive零命令中点一致、右手与rest一致 |
| `assets/door/doorman_door.py::DoorSpawnerCfg/build_frame` | 增加cover/keyhole/子面板显式覆盖字段；任务配置固定残留几何随机项 |
| `scripts/rsl_rl/train.py::main` | recurrent minibatch按env划分，选择不大于请求数且整除env数的batch数，警告并保存实际配置 |
| `scripts/rsl_rl/cli_args.py::update_rsl_rl_cfg` | --experiment_name真正应用 |
| `scripts/rsl_rl/play.py::main` | 新增--max_steps、--skip_export；有限步回放检查action/obs/reward有限，记录reset次数；保留done清理LSTM |
| `scripts/verify_action_term.py`、`scripts/verify_task_mdp.py` | 关闭前清理SimulationContext回调，避免本地Isaac Lab停止回调陷入渲染循环 |

新增固定几何字段：`rand_door_cover_width=.04`、`rand_spawn_keyhole=False`、`rand_keyhole_offset=.075`、`rand_num_subpanels=0`、`rand_subpanel_frame_width=.125`、`rand_subpanel_bottom=0`，另固定总高2.7与hook_length=.05。生成器None值仍支持随机化；@clone和replicate_physics仍不意味着每env有独立门几何。

## 实测验证

证据目录：`logs/verification/2026-09-25_doorman_alignment/`。

| 日志 | 实测结果 |
|---|---|
| `task_mdp.log` | 2env，139/219维、真实metadata、路径映射、已知共同SE(3)变换、阶段幂等更新、HOMIE异步相位reset、异常输入、210真实物理步；PASS、2次reset |
| `recurrent_policy.log` | actor/critic hidden和cell按env清零、peer不变、重复输入推进记忆、reset后重放一致、真实PPO更新改变LSTM权重；PASS |
| `action_primitive.log`、`action_joint_target.log` | 两种手后端8/27维输入、69维输出；PASS，清理后正常退出 |
| `train.log` | 2env×32步×10次更新=640 transitions；batch4自动调整为2；训练正常完成 |
| `checkpoint.log` | 参数均有限；model_0→model_9的actor/critic第一层LSTM权重最大变化约.00603/.00560 |
| `play.log` | 新checkpoint回放210步、2次reset；action/obs/reward均有限，正常退出 |

短训末轮stage0 timeout=1，success=0；没有把受控阶段测试当作真实抓握成功。短训checkpoint在 `logs/rsl_rl/g1_wuji_doorman_recurrent/2026-09-25_00-03-24_alignment_smoke/model_9.pt`，最终配置在同目录 `params/`。静态AST和git diff --check通过；环境未安装ruff，未声称完整pre-commit通过。ONNX/JIT导出未测。

### 复验命令

```bash
cd /home/yimin/CUHK/g1_wuji_doorman
source /home/yimin/miniconda3/etc/profile.d/conda.sh
conda activate doorman
DISPLAY= python scripts/verify_task_mdp.py --headless
DISPLAY= python scripts/verify_recurrent_policy.py --headless
DISPLAY= python scripts/verify_action_term.py --headless --hand_backend primitive
DISPLAY= python scripts/verify_action_term.py --headless --hand_backend joint_target
DISPLAY= python scripts/rsl_rl/train.py --headless \
  --task Template-G1-Wuji-Doorman-v0 --num_envs 2 --seed 42 \
  --max_iterations 10 --experiment_name g1_wuji_doorman_recurrent --run_name alignment_smoke
```

推理使用 `scripts/rsl_rl/play.py --headless --task Template-G1-Wuji-Doorman-v0 --num_envs 2 --checkpoint <新LSTM模型路径> --max_steps 210 --skip_export`。长期训练应重新设置环境数与迭代预算；10次更新仅验证接线。

## 保留的设计边界

阶段状态库是保存“已抓住把手”等物理状态，以便从后期阶段开始训练，和走路阶段无关。本次未实现，reset仍从门前PREGRASP开始。若未来加入，不能只改stage编号，还要处理完整物理状态、累计命令、HOMIE和高层记忆。

没有依据新增任意抓握力闭合判据、焊死base、强制脚固定或改变timeout的bootstrap语义；原积分/primitive饱和语义保留。仍需物理验证锁舌-门框真实闭锁、手接触驱动开门、持续抓握和漂移/脚滑边界。当前指标有助于观察，不等于硬约束。

**门驱动单位：**生成器USD写入hinge D=3、handle K=50/D=.5；本机PhysX初始化输出分别约171.887、2864.79/28.648，即乘180/π后的运行时值。`ImplicitActuatorCfg(stiffness=None,damping=None)`继承这些有效值，不能把USD数值直接当每rad增益。本次不改变它们；以后直接覆盖执行器配置要核对单位。实际初始化表保存在 `physics_initialization.log`。

完整架构地图、24项奖励公式、观测切片与修改手册见研究目录 `/home/yimin/Documents/ChatGPT/IsaacLab学习/代码仓库架构.md`。
