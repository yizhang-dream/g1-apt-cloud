# 文献综述：冻结解码器 + 位置控制 RL 的地形泛化极限与解法

> 生成日期：2026-08-14。由 6 路并行文献调研代理合并去重而成。
> 背景：APT-RL × SONIC × G1 复现。我们的管道 = RL 输出 z(16) → 冻结 token VAE →
> 冻结 SONIC 解码器 → 29 关节位置目标；rough 0.06 通过、0.08 硬悬崖（〔归因修正
> 2026-08-14，见 MQ09〕：此悬崖是"蒸馏路径（无 planner 10Hz 重规划）"的边界，非
> 解码器本身——同一解码器 + planner 重规划在 MuJoCo rough 0.08 walk≈flat）；解码器微调
> 打转崩溃（E44）；aux/elevation/gate 均无正价值（E15–E21b）。

## 一句话结论

**没有人在做"把一个冻结解码器当作硬性动作瓶颈、同时指望它泛化到训练分布外地形"这件事——因为这本来就不成立。** 六路文献独立收敛到同一个根因，也收敛到同一组解法。

## 根因（六路互证）

| 角度 | 结论 |
|---|---|
| ①冻结潜空间先验 RL | 冻结解码器 = **硬动作瓶颈** = 只能输出它见过的平地步态；文献里无人这样拿它做地形泛化。 |
| ②人形 RL 地形配方 | 同一位置控制（关节目标+PD）的主流配方，**4096 envs + 地形课程 + foot-clearance 奖励**，盲走 0.10m 噪声/0.23m 台阶——远超我们 0.06。 |
| ③感知 | 感知只有当**动作通道能表达新步态/换技能**时才有价值；我们冻结解码器 = "看得见动不了"。 |
| ④越出冻结解码器 | 别在关节空间优化（E44 打转的根）；用潜空间瓶颈 / LoRA / SPAR 整流受控改。 |
| ⑤力矩 vs 位置 | 位置控制**不是**障碍（RMA/ANYmal/Unitree 全是位置+PD）；力矩只是高冲击技能的助力。 |
| ⑥离散步态选择 | 选择只有当**技能库真的有不同技能**才有用；我们 SONIC 的 {walk/slow/jump/stealth} 全是平地模式，无地形技能。 |

**合并成一句**：我们的 0.06 悬崖 = **①把先验做成了硬瓶颈（政策只能输出解码器已有的平地步态）** + **②RL 太弱（64 envs、无课程、无抬脚奖励）**；与位置控制、感知、力矩、步态选择本身都无关。

## 最关键的再框定（对项目核心问题）

论文 APT-RL **不是**把解码器当硬瓶颈——它有一条**真正能越出流形的表达通道**：
`aux(12 维，加在 PD 目标里) + 步态选择(trot/bound) + 感知`，且 RL 规模是千级 envs + 地形课程。

我们的复现把这个**逃逸通道全做没了**：aux 弱（12 维下体、scale 0.2、64 envs）、无 bound、盲走。
所以"蒸馏先验是否必要"这个问题，我们目前的回答是**被混杂的**：我们测的是"先验当硬瓶颈"，
不是论文的"先验 + 逃逸通道"。要公平回答，得先给我们的政策补上同样的逃逸通道。

## 解法阶梯（按落地价值 × 成本 × 与项目问题的契合排序）

### 1. （决定性对照）跑 stock Isaac Lab G1-rough 配方
- [G1 rough env cfg](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py)、[terrain cfg](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/terrains/config/rough.py)
- 配方：4096 envs、关节位置目标+PD、地形难度课程（噪声 0.02–0.10m/方块 0.20m/台阶 0.23m/坡 22°）、foot-clearance+gait-phase 奖励、网络 [512,256,128]。
- **价值**：这是"去掉冻结解码器"的干净基线，预期直接清过 0.06–0.1m，把"解码器是瓶颈"钉死。成本 = 一次训练。

### 2. （直接攻 0.06 悬崖）冻结解码器 + 完整关节空间残差（RuN/ReSkill 式）
- [RuN（Unitree G1 上的人形残差策略，位置+PD，与我们同构）](https://arxiv.org/abs/2509.20696)、[ReSkill](https://arxiv.org/abs/2211.02231)、[MoRE](https://ar5iv.labs.arxiv.org/html/2506.08840)
- 机制：`q_target = q_decoder + a_residual`，解码器保持冻结、只提供"风格"，残差提供"越界稳定性"。
- **关键**：我们的 aux（E15–E21b）之所以无价值，是因为它是**弱残差**（12 维下体、scale 0.2、64 envs）；文献明确说残差要**全关节 + 地形输入 + 足量 envs**才有效。这是把我们已判死的 aux 线**用正确配方重开**。
- ReSkill 原话：不完整的技能空间"会削弱 RL 的学习能力"，残差"放宽了对穷尽数据集的需求"。

### 3. （若还要改解码器）LoRA / SPAR 受控微调
- [SLowRL（RL 循环内 LoRA 微调，防遗忘）](https://www.semanticscholar.org/paper/SLowRL%3A-Safe-Low-Rank-Adaptation-Reinforcement-for-Daneshmand-Omar/8e7b3944390a962887b8d00554287fc3cb222d4d)、[SPAR（支撑保持动作整流）](https://icml.cc/virtual/2026/poster/63368)
- 我们 E44v2 用"全权重+L2 锚定"太糙；LoRA 把参数改动锁在低秩子空间，SPAR 把输出投影回先验支撑集——两个都精确打我们的"打转崩溃"。

### 4. （再框定 C 臂）先验当"软参考"而非硬瓶颈
- [AMP](https://arxiv.org/abs/2104.02180)、[GMP](https://arxiv.org/abs/2503.09015)、[ZPRL/Beyond Action Residuals](https://arxiv.org/abs/2605.19919)
- 机制：解码器输出只作**风格奖励/参考**，政策直接输出全关节动作——对 0.06 悬崖和打转**双重免疫**。
- 价值：若项目的 claim 是"先验作为 prior"而非"先验作为硬约束"，这条 C 臂能隔离"是约束的锅还是内容的锅"。

### 5. 感知的替代：RMA 式本体自适应
- [RMA](https://ar5iv.labs.arxiv.org/html/2107.04034)：用 proprioceptive adaptation latent 替代显式感知，粗糙地形扰动靠本体消化。
- 价值：若不想做视觉，RMA 是"盲走也能泛化"的最强证据。

## 优先级来源清单（去重）

1. [RuN — Unitree G1 人形残差策略（位置+PD，直接同构）](https://arxiv.org/abs/2509.20696)
2. [ReSkill — Residual Skill Policies（CoRL 2022）](https://arxiv.org/abs/2211.02231)
3. [Beyond Action Residuals / ZPRL — 瓶颈潜空间 RL](https://arxiv.org/abs/2605.19919)
4. [SLowRL — 安全的低秩适配 RL](https://www.semanticscholar.org/paper/SLowRL%3A-Safe-Low-Rank-Adaptation-Reinforcement-for-Daneshmand-Omar/8e7b3944390a962887b8d00554287fc3cb222d4d)
5. [SPAR — 支撑保持动作整流（ICML 2026）](https://icml.cc/virtual/2026/poster/63368)
6. [Motion Priors Reimagined — 平地技能不够，需增补原语](https://proceedings.mlr.press/v305/zhang25j.html)
7. [RMA — 本体自适应替代感知](https://ar5iv.labs.arxiv.org/html/2107.04034)
8. [Isaac Lab G1 rough（决定性对照配方）](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py)
9. [APT-RL（Science Robotics adz7397，参考对象）](https://www.science.org/doi/full/10.1126/scirobotics.adz7397)
10. [MoRE — 混合残差专家，人形复杂地形](https://ar5iv.labs.arxiv.org/html/2506.08840)
11. [CMoE — 真机 G1 地形/台阶](https://ar5iv.labs.arxiv.org/html/2603.03067)
12. [Discovery of Skill-Switching Criteria（APT-RL 直接续作）](https://arxiv.org/abs/2502.06676)
13. [Hoeller 2022 — elevation-in-obs 事实标准](https://ar5iv.labs.arxiv.org/html/2201.08117)
14. [Action Space Design（CoRL 2024）— 动作空间才是杠杆](https://openreview.net/forum?id=GGuNkjQSrk)
15. [Learning Torque Control for Quadrupedal Locomotion](https://ar5iv.labs.arxiv.org/html/2203.05194)

> 注：多条 arXiv 标号（2509/2605/2606…）为 2025–2026 预印本，机制结论稳健，具体数字以正式版为准。
