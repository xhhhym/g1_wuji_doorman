# G1 + Wuji DoorMan：观测与奖励冒烟基线

> 2026-09-25 更新：本文记录旧111/191维MLP冒烟基线。当前已改为139/219维LSTM；实现、修正及验证见 [DOORMAN_RECURRENT_ALIGNMENT.md](DOORMAN_RECURRENT_ALIGNMENT.md)。


2026-09-24：本地 RTX 4070、`doorman` Conda 环境、2 个并行环境完成 10 次 PPO 更新。
当前配置是 DoorMan 风格 privileged teacher policy 的训练链路验证，策略尚未学会抓握或开门。
当前固定从 PREGRASP 初始状态 reset。

## 复现

```bash
cd /home/yimin/CUHK/g1_wuji_doorman
source /home/yimin/miniconda3/etc/profile.d/conda.sh
conda activate doorman

# 首次配置训练环境时需要；本机已安装。
python -m pip install rsl-rl-lib==3.0.1 GitPython

# MDP 确定性验证 + 2 env 真实物理运行。
DISPLAY= python -u scripts/verify_task_mdp.py --headless

# 与本次成功运行相同的训练命令。
DISPLAY= python -u scripts/rsl_rl/train.py \
  --task Template-G1-Wuji-Doorman-v0 \
  --num_envs 2 --headless --max_iterations 10 \
  --run_name smoke_2env --seed 42
```

必须激活 `doorman`，激活脚本提供 Isaac Sim 自带的 Python 包与 CUDA 动态库路径。
无窗口运行清空该进程的 DISPLAY，避开本机出现过的 X11 探测原生崩溃。

## 接口

- 保留 free-base、重力、冻结 HOMIE、7D 左臂累计 delta + 1D 手部 primitive。
- physics/HOMIE/policy 频率保持 200/50/25 Hz。
- teacher actor 使用 111D 当前帧状态，其中包含 DoorMan 风格的 8D 精确门参数；
  asymmetric critic 使用 191D，在 actor 输入外增加 25 个手部 link 的 3D 精确把手接触力和 5D 内部任务状态。
- 网络为两层 128 单元 ELU，观测归一化，初始动作标准差 0.3；rollout 32 步，学习率 3e-4。
- episode 上限 30 秒；四阶段各自 timeout 为 8/6/6/10 秒。

| 观测（顺序固定） | 维度 | 处理 |
|---|---:|---|
| stage one-hot | 4 | PREGRASP/GRASP/UNLATCH/OPEN_DOOR |
| projected gravity | 3 | base frame |
| base angular velocity | 3 | base frame，乘 0.25 |
| base linear velocity | 3 | base frame |
| root in door frame | 9 | 相对位置 + 旋转矩阵前两列 |
| left arm position / velocity | 7+7 | 相对默认关节位置；速度乘 0.05 |
| left hand position / velocity | 20+20 | 相对默认关节位置；速度乘 0.05 |
| grasp target in palm frame | 9 | 真实 grasp_target，位置 + 6D 旋转 |
| hinge q/dq、handle q/dq、latch q | 5 | 速度乘 0.1；latch 除以 0.03 m |
| fingertip filtered force | 5 | 每个 tip 的力模长，裁剪到 50 N 后除以 50 |
| privileged door info | 8 | 门宽/高、把手高/偏移、质量和开门方向；teacher 专用 |
| last action | 8 | 刚执行的高层 policy action |

teacher actor 最终为 111D，critic 为 191D，观测裁剪至 ±10。critic 额外 80D 是完整过滤接触力
（25×3）和 transition/success/stage time/episode time/success hold（5D）。这些都来自仿真真值，
部署前需要训练 student policy 蒸馏掉特权输入。primitive 更换成其他后端时 last_action 维度随动作变化。

## 状态与接触

`mdp/commands.py` 的 `DoorTaskState` 集中持有每个环境的 stage、计时、连续满足计数、
角度历史与当步进度。局部 reset 只清空指定环境；HOMIE history 和累计 action 由原 ActionTerm reset。

本地 Isaac Lab 的 step 顺序为 termination → reward → reset → command → observation。
因此 `TerminationsCfg.invalid_state` **必须排在第一项**：它先调用 `advance()`，按
`common_step_counter` 去重，然后其他 termination/reward 只读状态。常规 command.compute
仅刷新几何，不推进 stage。reward 使用 transition 前的 stage，下一帧 observation 使用更新后的 stage。

单一 `door_handle` ContactSensor 过滤左手 25 个刚体（每指 link1–4 + tip），遵循 one-to-many API。
状态转换统计不同手指，而非同一手指的多个 link；至少 3 指且包含拇指，力阈值 1 N。
actor 只取 5 个 tip 通道，抓握判定使用整根手指各 link 的最大力模长。

PREGRASP 要求距离 <8 cm、角误差 <30°、实际手姿闭合比例 <0.6；GRASP 要求多指接触且距离
<12 cm；UNLATCH 要求接触保持且锁舌缩回 ≥24 mm。每次推进要求连续 5 步，单向推进。
门角 ≥30° 且直立保持 13 步（0.52 秒）后成功终止。跌倒或异常状态不会获得成功奖励。
门关节成功方向使用当前生成资产的正 hinge/handle/latch 方向；换开门方向或资产时需重新校准。
左掌期望方向沿用 DoorMan 的目标坐标系绕 X 轴 +90°，仍需后续 GUI 抓握校准。

## 奖励

24 项初始权重定义在 `g1_wuji_doorman_env_cfg.py`，实现位于 `mdp/task_rewards.py`：

- 接近、方向对齐、张手预形；
- 多指接触、接触门控闭合；
- 把手下压量/正向进度/回弹、锁舌缩回；
- 门角进度/打开量/回退、保持打开；
- 一次性阶段 bonus 和 success bonus；
- 跌倒/数值异常、倾斜、base 漂移、右臂参考姿态、动作变化、关节速度/限位、过大接触力和接触丢失。

Isaac Lab 会给原始奖励乘 `step_dt`。正负角度增量先除以 `step_dt`，故加权后是实际每步角度收益；
阶段/成功/失败事件同样抵消 `step_dt`，实际奖励分别为 +2/+10/−5。其余为按时间积分的奖励率。
手指闭合量从实际 20D 关节姿态投影到 open→grasp 方向，避免奖励仅握紧控制命令。

## 实测结果

- `TASK_MDP_VERIFY_PASS envs=2 actor_obs=111 critic_obs=191 action=8 door_privileged=8 contacts=25 transitions=3 success_hold=13 resets=2`
- 验证相对位姿平移不变性、重复调用不推进状态、受控信号触发全部阶段和 success/timeout、
  接近/对齐/张手/闭合门控/正负进度的奖励方向，以及局部 reset。
- 真实物理零动作运行 210 步，观测与奖励有限，两个环境自动 reset。
- teacher PPO：10 次更新，640 transitions，19.06 秒，退出码 0；网络实际输入为 actor 111D、critic 191D。
- checkpoint `model_0.pt` 与 `model_9.pt` 全部模型张量有限；actor 参数最大变化约 0.004897。
- TensorBoard 的 51 个 scalar tag 全部有限；末次 value loss 0.4488，surrogate loss −0.0493。
- 本次训练没有跌倒或数值异常；终止来自 PREGRASP timeout，stage 仍为 0，接触数与成功率为 0。

本次产物：

```text
logs/rsl_rl/g1_wuji_doorman/2026-09-24_16-22-11_teacher_privileged_smoke_2env/
  model_0.pt
  model_9.pt
  events.out.tfevents.*
  params/env.yaml
  params/agent.yaml
  smoke_train.log
  verify_task_mdp.log
```

HOMIE 推理网络现已迁入本项目的 `controllers/standing/homie_modules.py`，并删除上游构造器
覆盖 `Normal.set_default_validate_args` 的全局副作用；checkpoint参数和推理输出保持不变。

## 当前验证范围

阶段完整推进使用受控信号验证，**不代表真实物理抓握/解锁/开门已经成功**。真实手指与把手接触时
过滤力非零、碰门板不计入抓握，仍需专门的接触实验与 GUI 校准。尚未做 staged reset/state bank、
curriculum、domain randomization、feet-slip 或身体/门框非期望接触惩罚。当前足以继续小规模训练调试，
不是设计文档全部 staged MDP 验收项完成。
