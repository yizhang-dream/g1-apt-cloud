# DS 线调研：HIL 混合模仿学习评估（B4/T1 设计储备）

> 【层位 L2 侧轴｜调研（2026-09-05c；触发点 = owner 指令「重点关注 HIL:
> Hybrid Imitation Learning for Dynamic Athletic Control，感觉很大概率能用在
> 我们这个身上，或者可以参考」；同日 owner 认可本文定位）】↑
> `refine-logs/README.md`（扇出树根）｜上游：`DS_TERRAIN_ADAPTER_CHARTER.md`
> §3/§4（近邻表第五行由本文增补）、`DS_OFFICIAL_DATA_PLAN.md`（B4/T1 阶段
> 归属）｜状态：**活跃（设计储备，未立项；不改变 B2-s 第一闸与预算表）**。

**一句话结论**：HIL 的「tracking + style 混合奖励」配方与本项目「自然度 ×
任务性能」核心矛盾完全同构，其三个机制可转为 B4/T1 阶段的最小判别实验
候选；但它是**纯仿真 SMPL 角色、无 sim-to-real、不用 token 解码器**——
定位 = 设计储备，不是底座或管线替换。

## 1. 论文档案

- **题目**：HIL: Hybrid Imitation Learning for Dynamic Athletic Control。
- **作者/单位**：Jiashun Wang, Yifeng Jiang, Haotian Zhang, Chen Tessler,
  Davis Rempe, Jessica Hodgins, Xue Bin Peng（NVIDIA DAIR 等）。
- **版本**：[arXiv 2505.12619](https://arxiv.org/abs/2505.12619)（v1 2025-05
  曾名 *HIL: Hybrid Imitation Learning of Diverse Parkour Skills from
  Videos*，v2 泛化为 athletic control）；**ACM TOG 2026** 正式发表
  （[DOI 10.1145/3829364](https://dl.acm.org/doi/10.1145/3829364)，2026-07；
  [NVIDIA DAIR 项目页](https://research.nvidia.com/labs/dair/publication/hil2026/)；
  项目站 jiashunwang.github.io/HIL/）。**代码/数据截至 09-05 未开源**。
- **平台**：Isaac Gym，SMPL 仿真人体（120 Hz 仿真 / 30 Hz 策略，4096
  并行环境，4×V100）；**纯仿真，无真机**，论文自述 SMPL 本体「actuation
  不真实」，sim-to-real 列为 future work。

## 2. 方法拆解（"Hybrid"到底是什么）

**混合的两种模仿学习信号**（模仿学习 = 从参考动作/示教数据学策略）：

1. **逐帧 motion tracking 奖励**（式 2：位置/旋转/线/角速度/根高指数项 +
   能量惩罚）；
2. **AMP 式对抗风格奖励**（AMP = Adversarial Motion Priors，对抗运动
   先验；判别器区分「策略轨迹 vs 参考轨迹」，判别器输出转风格奖励）。
   判别器输入 = 10 步状态历史 + **场景点云**，二分类 + gradient penalty，
   r_style = −log(1−D)。

**组合方式**：r = 0.5·r_task + 0.5·r_style（tracking 模式内同样叠加 style
项）；**两模式** = tracking 模式（逐帧参考可用）与 AIL 模式（goal-
conditioned，无参考），critic 额外接收二值任务指示 k。训练调度：先纯
tracking 4B samples 掌握单技能，再两模式各半环境并行 2B samples 泛化。

**最关键的设计决策——goal-conditioned 化**：观测 = 场景点云（每物体 15
点、N=60、180 维）+ 目标位置，**明确不用 phase / 目标姿态**输入，原文
理由 = "we cannot rely on these inputs, as they are unavailable in novel
environments"（新环境里拿不到逐帧参考，所以控制器必须学会不依赖它）。
策略 = PointNet + transformer，输出 PD 目标关节角（固定 σ=0.055）；
PPO + GAE。另配 PSI（对初始状态加高斯噪声，提升扰动恢复）。

**数据**：parkour 任务 = 19 段 YouTube 视频（各 30 s，15 种技能）→ TRAM
位姿估计 → 手动标注 box 障碍几何 → MaskedMimic 物理精修去穿模/滑步；
heading 任务 = 7 分钟剑盾动捕（Peng et al. 2022）。

## 3. 结果与消融（关键数字）

**Parkour 表 1**（技能准确率 / 跟踪误差 / 完成率）：

| 方法 | 准确率 | 跟踪误差 | 完成率 |
|---|---|---|---|
| **HIL** | **0.66** | **0.31** | **0.74** |
| AMP w/ ws（带权重） | 0.54 | 0.37 | 0.85 |
| Task Reward w/ ws | 0.15 | 0.54 | 0.86 |
| MaskedMimic | 0.50 | 0.41 | 0.00（20 障碍泛化崩） |
| AMP（原版） | 0.06 | 1.49 | 0.11 |
| ASE | 0.03 | 1.63 | 0.00 |

结论：AMP 系自然但任务崩、纯 task 能完成但动作丑——**HIL 表 1 = 「自然
度 × 任务性能」权衡的最强定量外证**，与本项目 0.08 悬崖（蒸馏路径边界）
的叙事同构，写相关工作时与 T-GMP 并列引用。

**消融表 2**：去判别器 → 0.53/0.36/0.62；去 PSI → 0.50/0.37/0.52；
**判别器去场景信息 → 技能准确率 0.66→0.38（单项最大跌幅）**；critic 去
任务指示 k → loss 大 5×。鲁棒性：动作噪声 σ=0.05 完成率 >70%、σ=0.1
>50%；5 障碍训练 → 20 障碍序列 40%。泛化：SAMP 坐椅动作（未见技能）
可迁移。

**论文自述 limitation**：绊障恢复仍有伪影；box 简化几何限制泛化；动作-
场景对依赖人工标注（自述难规模化）；大扰动仍失败；sim-to-real 未解决。

## 4. 对 DS 线的适用性评估（owner 09-05c 认可）

### 4.1 三个可借机制（→ 最小判别实验候选）

1. **goal-conditioned tracker（去相位/目标姿态依赖）**——我们做未见动作/
   未见组合评测（纲领 §5 主指标）正缺「不依赖逐帧参考的控制器」；HIL 是
   该设计的成品先例。与 D031 相位彩票线索相关联（过栏结果随到达相位变
   化），**注意按 gate≠机制 纪律：这是设计线索，不是相位机制的证明**。
2. **判别器地形条件化**——HIL 消融单项最大（0.66→0.38）；我方对应改动 =
   风格判别器吃地形观测。改动小、可预注册消融，**三候选中优先级最高**。
3. **两模式并行 + critic 任务指示**——T1 过渡衔接（tracking 模式 ↔ 受控
   评测模式切换）的现成结构参考。

### 4.2 四个不能直接搬的点

1. **纯仿真 SMPL、actuation 不真实、无 sim-to-real**——其成功率数字不能
   外推到 G1 真机口径（对照 `DS_S2R_EVIDENCE.md`：gap 主体是执行器动力
   学，SMPL 仿真连这一层都没有）。
2. **预算差三个量级**：4096 env × 4B samples × 4×V100 vs 本地 3060 12G
   单卡（E 系 2000it ≈ 27min 量级）；照搬必缩水，效果未知。
3. **数据管线不可移植**：YouTube 视频管线对官方数据独走的我们无价值；
   反向启示 = heading 任务 7 分钟动捕即可起效，**支持官方 SMPL 镜像
   131,455 段做阶段一数据底座是够用的**。
4. **不替代 SONIC/token 底座**：HIL 的先验 = 逐帧参考 + AMP 判别器，与
   冻结 decoder 结构不同维度；它是 B4/T1 阶段 RL 配方参考，不是管线。

### 4.3 纲领触点（需裁决项）

HIL 式 style reward 需要把动捕语料放进 RL 判别器——这会扩大 09-05b
退役影响面裁定中「**动捕只喂 VAE 与评测**」的口径，**须先走 owner 裁决**。
不触口径的替代路径：风格语料改用 **token → 冻结 decoder 解码出的动作**
（仍留在「官方数据 → VAE/decoder」通道内），本身可作预注册对照臂。

## 5. 落法建议（已获 owner 认可）

- **定位**：B4/T1 阶段设计储备；纲领 §3 近邻表增补为五个近邻（本文 09-05c
  修订）；**不改变主线与预算表，B2-s（smpl obs 逆向）仍是第一闸**。
- **候选判别实验优先级**：① 判别器地形条件化；② goal-conditioned 化
  tracker；③ 两模式并行调度。均按「最小判别实验 + 预注册」模式立项，
  立项时另立 `_PLAN.md` 挂树（纲领 §10 纪律）。
- **写作用途**：§3 表 1 与 T-GMP 并列作为「自然度 × 地形/技能权衡」近邻
  证据链；引用时注明纯仿真边界，不得当作真机 SOTA 对照。
