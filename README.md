# UAV Reinforcement Learning Baseline

## 项目简介

这是一个面向无人系统控制与强化学习方向的最小单 UAV Q-learning baseline。项目目标是理解并跑通环境建模、智能体训练、策略评估、指标记录和结果可视化的完整实验流程，为后续研究 DQN、多 UAV、集中训练分散执行（CTDE）及 GNN/GAT 奠定可复现的基线。

## 问题定义

- 起点为 `(0, 0)`，目标为 `(5, 5)`，状态表示为 `[x, y, x_g, y_g]`。
- 动作为 `up / down / left / right`，每步在二维整数网格中移动一格。
- 默认奖励为当前位置到目标的负欧氏距离：$r=-\sqrt{(x-x_g)^2+(y-y_g)^2}$；到达目标时奖励改为 `100`。
- 每个 episode 最多执行 `100` 步。环境使用固定起点和固定目标，没有空间边界，也不包含真实 UAV 动力学。

## Q-learning

智能体使用表格型 Q-learning，更新规则为：

$$
Q(s,a) \leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]
$$

超参数为 `alpha=0.1`、`gamma=0.9`、`epsilon=0.3`。训练阶段采用 ε-greedy 策略，即以 30% 概率随机探索；测试阶段使用纯 greedy policy，并关闭学习更新。

## 实验设计

主实验固定 `seed=0`，分别从新智能体开始独立训练 `100` 和 `500` episodes；两次运行使用相同环境、奖励和超参数。CSV 记录验证，500-episode 运行的前 100 episodes 与独立 100-episode 运行逐条完全一致，因此可以将其视为沿相同训练轨迹再学习 400 episodes。

Training 与 Evaluation 相互分离：训练阶段更新 Q-table；评估阶段使用已训练的 Q-table，在新环境中关闭探索，以 greedy policy 运行一个 episode，且不更新 Q-table。

## 实验结果

| 训练轮数 | Success | Steps | Total reward | Final distance | Greedy policy 表现 |
| ---: | :---: | ---: | ---: | ---: | --- |
| 100 | False | 100 | -626.84 | 6.32 | 进入 `(-1,3) ↔ (-1,4)` 周期振荡 |
| 500 | True | 10 | 67.11 | 0 | 到达 `(5,5)` |

从 `(0,0)` 到 `(5,5)`，四方向单步移动至少需要 $|5|+|5|=10$ 步，因此 500-episode 策略在当前动作模型下达到最短路径。

根目录的旧版 `q_table.pkl` 同样来自 100 episodes 训练，但当时没有固定随机种子。当前 greedy 评估中，该策略在 `(0,0) ↔ (1,0)` 之间振荡并最终失败。两次 100-episode 实验形成不同失败策略，说明训练轮数有限时，结果仍会受到 ε-greedy 随机探索历史的影响。

## Training Metrics

以下指标来自 500-episode 训练记录；它们包含训练时的随机探索，与上表的单次 greedy 评估含义不同。

| 区间 | 平均 reward | 平均 steps | Success rate |
| --- | ---: | ---: | ---: |
| 前 50 episodes | -582.19 | 64.92 | 58% |
| 后 50 episodes | 45.03 | 15.00 | 100% |

训练后期呈现 `reward ↑`、`steps ↓`、`success rate ↑` 的整体趋势。

## 项目结构

- `env/uav_env.py`：定义二维导航环境、状态转移、奖励与终止条件。
- `agent/q_learning.py`：实现 Q-table、ε-greedy 动作选择和 Q-learning 更新。
- `train.py`：执行训练并保存逐 episode 指标和 Q-table。
- `test_policy.py`：在不探索、不学习的条件下评估 greedy policy。
- `compare_training.py`：运行 100 vs 500 episodes 受控对比并保存评估结果与轨迹。
- `visualize_training.py`：从训练 CSV 生成 reward、steps 和 success rate 曲线。
- `animate_trajectories.py`：从轨迹 CSV 生成 100 vs 500 episodes 对比动画及终态图。
- `results/`：保存 CSV、Q-table、PNG 曲线、轨迹 GIF 和终态图。

## 可视化

- [Reward curve](results/reward_curve_500_episodes_seed_0.png)：展示单回合 reward 与 50-episode 移动平均。
- [Steps curve](results/steps_curve_500_episodes_seed_0.png)：展示单回合步数与 50-episode 移动平均。
- [Success rate curve](results/success_rate_curve_500_episodes_seed_0.png)：展示 50-episode 滚动成功率。
- [100 vs 500 trajectory GIF](results/trajectory_comparison_seed_0.gif)：直观对比 100 episodes 的振荡失败与 500 episodes 的 10 步成功轨迹。

## 当前结论与局限

在 `seed=0` 的受控实验中，独立 100-episode 运行尚未形成成功的 greedy policy；同配置的独立 500-episode 运行训练到第 500 回合后，策略能够用最短的 10 步到达目标。

当前 baseline 只描述二维离散位置，不包含速度、加速度、航向角或真实飞行动力学；只研究单 UAV，主要分析一个随机种子，且起点与目标固定。后续可开展 multi-seed 统计、随机起点/目标、DQN、UAV 动力学、多智能体协同、CTDE 以及 GNN/GAT；这些均为后续方向，尚未在当前版本中实现。
