# G1 + Wuji + Doorman 资产验证记录

记录日期：2026-09-16

本文记录当前项目的常用运行命令、G1/Wuji 与 Doorman 门资产的验证结果，以及今天确认的重要参数和注意事项。

## 1. 项目与环境

项目目录：

```text
/home/yimin/CUHK/g1_wuji_doorman
```

每次打开新终端后，先执行：

```bash
cd /home/yimin/CUHK/g1_wuji_doorman
source /home/yimin/miniconda3/etc/profile.d/conda.sh
conda activate doorman
```

## 2. 同时加载机器人和门

运行：

```bash
python scripts/zero_agent.py --task Template-G1-Wuji-Doorman-v0
```

必须使用当前真正注册的任务名称：

```text
Template-G1-Wuji-Doorman-v0
```

下面这个名称当前没有注册，使用它会触发 `gymnasium.error.NameNotFound`：

```text
Isaac-G1-Wuji-Doorman-v0
```

任务注册位置：

```text
source/g1_wuji_doorman/g1_wuji_doorman/tasks/manager_based/g1_wuji_doorman/__init__.py
```

机器人和门的场景配置位置：

```text
source/g1_wuji_doorman/g1_wuji_doorman/tasks/manager_based/g1_wuji_doorman/g1_wuji_doorman_env_cfg.py
```

### 联合加载并自动演示把手下压和开门

可视化运行：

```bash
python scripts/verify_robot_door_motion.py
```

无界面运行并在验证后退出：

```bash
python scripts/verify_robot_door_motion.py --headless --exit_after_test
```

该脚本使用正式场景配置同时加载 G1 + Wuji 和 Doorman 门。机器人保持默认固定姿态，脚本依次执行把手下压、锁舌缩回、门板打开和把手释放。实测结果：

```text
Robot loaded:    bodies=91, joints=69
Default station: robot_x=-0.000 m, door_x=0.600 m  PASS
Left-palm Y:      -0.198 m, handle_y=-0.195 m, error=+0.003 m
Handle pressed:   45.000 deg  PASS
Latch travel:     30.000 mm   PASS
Door opened:     120.000 deg  PASS
Panel delta X:     0.398 m    PASS
Handle returned:   0.000 deg  PASS

OVERALL: PASS
```

机器人采用单位四元数朝向并面向 `+X`。门位于机器人前方 `0.6 m`，门板打开后继续向 `+X` 移动 `0.398 m`，即远离机器人打开。机器人在开门任务中的初始位置为 `(0.0, -0.35, 0.75)`；该 Y 偏移使默认左手掌与门把手只相差约 `3 mm`。该脚本验证的是“机器人和门同时存在时，默认站位和门机构均正常”，不是机器人手臂主动完成开门动作。

## 3. 当前门资产结构

当前保留了 Doorman 的核心机构：

```text
door_panel  -- revolute joint --> hinge_joint
door_handle -- revolute joint --> handle_joint
latch_link  -- prismatic joint --> latch_joint
handle_joint -- PhysX mimic --> latch_joint
```

关键参数：

- 把手范围：`0° ~ 45°`
- 锁舌范围：`0 ~ 0.03 m`
- mimic gearing：`-0.03 / 45`
- 把手完整转动 `45°` 时，锁舌缩回 `30 mm`
- 把手 Drive 目标：`-15°`
- 把手刚度：`50.0`
- 把手阻尼：`0.5`

`target_position=-15°` 位于把手的 `0°` 下限之外，因此会形成预紧力，将把手持续压在 `0°` 限位上。它用于模拟真实门把手的回位弹簧，而不是让把手实际转到 `-15°`。

## 4. 当前几何与质量配置

正式场景中的固定参数位于 `g1_wuji_doorman_env_cfg.py`：

```text
门宽：          0.85 m
门高：          2.00 m
把手高度：      0.80 m
把手横向位置：  0.23 m
门质量：        10.0 kg
轴长度：        0.20 m
把手长度：      0.20 m
把手半径：      0.02 m
```

这些参数用于让程序化 Doorman 门接近此前 XML 门的大小、质量和把手位置。

## 5. 与原版 Doorman 的差异

核心关节类型和 mimic 联动没有改变，但当前迁移版并非只修改了质量。已确认的代码差异包括：

```text
轴质量：              0.2  -> 0.3
门铰链角度上限：      150° -> 120°
门铰链目标位置：      -10° -> 0°
门铰链阻尼：          50.0 -> 3.0
锁舌高度：            door_height - 0.1 -> door_handle_height
```

把手参数目前已经恢复为原版 Doorman 数值：

```python
handle_drive.GetTargetPositionAttr().Set(-15.0)
handle_drive.GetDampingAttr().Set(0.5)
handle_drive.GetStiffnessAttr().Set(50.0)
```

目前的自动物理测试已经在这些当前参数下通过，因此后续应先停止继续改门资产，进入机器人操作验证。如果以后需要严格复现原版 Doorman，再逐项恢复并分别测试，避免一次修改多个动力学参数。

## 6. 常见信息与报错

### `Environment ... doesn't exist`

原因是任务名称错误。使用：

```bash
python scripts/zero_agent.py --task Template-G1-Wuji-Doorman-v0
```

### `Not all actuators are configured: 2 != 3`

门共有三个关节，但只直接配置 hinge 和 handle 两个 actuator。`latch_joint` 是 mimic joint，不应再增加独立 actuator。因此在门机构验证脚本中看到这条 warning 是预期现象。

### GPU、IOMMU、CPU powersave 等 warning

这些信息通常不是任务退出原因。判断问题时，应查看日志末尾的 Python traceback，尤其是最后一个异常类型和报错行。

## 7. 当前阶段结论

目前已经完成：

- G1 + Wuji 资产能够加载。
- Doorman 门、门框、把手和锁舌能够加载。
- 门板、把手、锁舌的关节结构已经确认。
- 把手与锁舌的 mimic 联动通过物理测试。
- 门的锁定、解锁、打开和把手回位均通过测试。
- 机器人和门能够在同一个 Isaac Lab 场景中加载。

下一阶段是固定 G1 根节点，让机器人站在原地完成接近、抓握、下压把手和推门动作，而不是继续修改门资产。

## 8. 版权说明

迁移 NVIDIA/Doorman 源代码时，保留原文件中的版权声明和 Apache-2.0 许可证头。新增的验证脚本与项目代码应避免删除或覆盖原始来源信息。
