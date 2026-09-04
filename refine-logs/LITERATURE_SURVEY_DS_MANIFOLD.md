# 文献综述：DS 步态流形计划近邻地图与创新点评估

> 【层位 L2 侧轴｜调研（2026-09-04；owner 指令"搜索相关论文，判断创新点是否足够，
> 找可借鉴论文"；支撑 `DS_GAIT_MANIFOLD_PLAN.md`，检索渠道 = arXiv（researchclaw）
> + web 搜索，2026-09 时点）】↑
> `refine-logs/README.md`（扇出树根）｜上游：`DS_GAIT_MANIFOLD_PLAN.md`（被评估
> 的执行计划）、`LITERATURE_SURVEY_FROZEN_DECODER.md`（前次调研：冻结解码器
> 地形泛化，结论已入 MQ09/ROOT_CAUSE 证据链）｜事实源：`tracker/D.md`

---

## 一句话结论

**创新点按"移植复现 + 扩展 + 定界"定位足够，按"全新方法"定位不够。**
"冻结官方部署级 token 解码器 + token VAE 多步态流形 + held-out 族流形连续性
检验 + 预注册切换对照"这一组合在 G1 上**无同路线先例**；但每个组件单独看都有
很近的先例——**多步态统一控制、latent 多步态、学习型切换各自都已被占位**
（人形五步态 AMP 2026-04、四足 skill latent space TIE 2025、HugWBC G1 步态
参数化、APT 自带 selector）。对外写作若把主创新写成"人形多步态 RL"或
"skill latent space multigait"会被审稿人直接指近邻；主创新表述必须锚定在
**冻结底座 + 流形检验协议**这个组合上。

## 近邻地图（三层）

### 第一层：同题（多步态统一控制 + 切换，含人形）

| 论文 | 载体 | 方法 | 与我们的关系 |
|---|---|---|---|
| **APT-RL**（Science Robotics adz7397 / arXiv 2607.13579，KAIST） | 四足，wild 地形 ~6 m/s | 动作预训练 transformer + RL + 感知；步态 selector = 单 logit 2Hz + 0.5s 锁 | 我们复现的源头论文。我们 Phase 5 = 其 selector 做成 4-way + 冻结对照 + 预注册停止规则 |
| **Multi-Gait Learning for Humanoid Robots**（arXiv 2604.19102，2026-04） | 12-DoF 人形，5 步态（行走/正步/跑/爬楼/跳），零样本真机 | **选择性 AMP**：稳定步态用 AMP 正则、跑步/跳跃故意不加；PPO+DR | **最需警惕的近邻**（人形+多步态+真机）。但路线 = 对抗判别器先验，非 token 预训练/VAE/latent；摘要无学习型切换 |
| **HugWBC**（arXiv 2502.03206） | G1，行走/跳跃/站立/单脚跳，真机 | 单策略 + **细粒度步态参数条件化**（频率/摆高/duty/躯干姿态等 8 命令） | 与 G3"全速度段连续可调"直接同类；条件化对象是手工步态参数而非 token 流形 |
| **Gait-Conditioned RL with Multi-Phase**（UCL, Humanoids 2025） | 人形 | 步态条件化 RL（站立/走/跑） | 条件化先例，规模较小 |
| **Learning Free Gait Transition via Phase**（IEEE T-IE 2025） | 四足 | 相位驱动的自由步态过渡 | 切换机制对照（非学习型 selector） |

### 第二层：同方法（latent 空间承载多步态/技能）

| 论文 | 载体 | 方法 | 与我们的关系 |
|---|---|---|---|
| **Skill Latent Space Based Multigait Learning**（IEEE TIE 72(2):1743–1752, 2025, Xin Liu et al.） | 四足 | 训练中**同步构建技能潜空间**；步态/腿摆高/体高可在潜空间微调 | "skill latent space + multigait"一词已被占（四足版）；必引，我们的 delta = G1 + 官方 token + held-out 检验 |
| **Learning Multiple Gaits within Latent Space**（Wu, Xue et al.） | 四足 | 显式步态参数化奖励 + 条件对抗运动先验（latent） | latent 多步态又一四足先例 |
| **DiffuseLoco**（arXiv 2404.19264，CoRL 2025） | 腿式（quadruped/G1 级） | **离线数据集 → 扩散策略**，单一策略 walk/run/jump 多技能，实时控制，不设显式 gate | 与我们"离线官方数据 → 流形 → 策略"最近的可操作借鉴；切换交给模型而非显式选择头 |
| **UniTracker**（arXiv 2507.07356） | **G1**，真机 | 三阶段：特权教师 → **CVAE 学生**通用跟踪策略 → 快速适配 | G1 + 潜空间最近邻（跟踪任务非步态流形）；必引 |
| **Motion VAEs / MVAE**（Ling & van de Panne, SIGGRAPH 2020） | 动画角色 | 自回归 CVAE，**latent 即动作空间**，可规划/控制 | 思想源头，related work 必引 |
| **MaskedMimic**（Tessler et al., SIGGRAPH 2024）/ ASE / CALM | 动画/物理角色 | 掩码修复统一多控制模式 / 非结构化数据的潜空间技能 | "单策略多模式 + 条件化"思想源头；G5 感知条件化的引用锚 |
| **Radosavovic et al.**（arXiv 2410.03654） | 人形，野外 4 英里 | transformer 平地预训练 → 不平地 RL **微调**（非冻结） | pretrain→RL 先例；反衬我们"冻结解码器"是差异点 |

### 第三层：同模式（冻结底座 + RL 在潜空间上引导）与 sim2real 校准

| 论文 | 载体 | 方法 | 与我们的关系 |
|---|---|---|---|
| **ASAP**（arXiv 2502.01143，RSS 2025，CMU LeCAR） | G1，真机 | **delta action model** 事后校正 sim2real 物理差（-52.7% 跟踪误差） | Phase 0 执行校准门的对照：ASAP 事后 delta 校正 vs 我们**事前 oracle 回放实现率门**；其"执行器动力学非均匀"结论与 D033/D034 互相印证 |
| **RL Token**（arXiv 2604.23073）/ **Bottleneck latent RL steering**（arXiv 2605.19919）/ **Post-Training as Latent Control**（arXiv 2412.02125） | VLA/操作 | 冻结预训练底座，RL 只在潜 token/bottleneck 上学引导 | 证明"冻结底座 + RL 潜空间引导"模式在**操作域**成立；我们是 loco 域版本，可引为模式合法性证据 |
| **Transferable Latent-to-Latent Locomotion Policy**（arXiv 2503.17626） | 腿式 | 预训练 latent-to-latent 策略 + 多任务 encoders/decoders | locomotion 侧 latent 迁移先例 |
| **Humanoid-LLA**（arXiv 2511.22963）/ **UniT**（XPENG） | 人形 | 语言→统一 motion token 潜空间；token 化动作接口 | token 化动作表示趋势的背景引用 |
| ExBody / ExBody2 / HumanPlus | G1/H1 | 动捕重定向 + 教师-学生跟踪 | 同构管线背景（项目前置史已对照）；与我们数据源（官方回路 token）不同 |

## 创新点逐条评估（对照 DS_GAIT_MANIFOLD_PLAN §0 目标）

| # | 我们的主张 | 最近先例 | delta 是否成立 | 判定 |
|---|---|---|---|---|
| 1 | 冻结官方 token 解码器底座上的 G1 多步态流形 + 学习切换 | TIE 2025（四足 latent，自训）；2604.19102（人形 AMP，无 latent）；DiffuseLoco（扩散，非冻结 token 底座） | 同一技术路线（部署级冻结 token 底座）无先例；VLA 侧模式存在但非 loco | **成立**（组合新颖性） |
| 2 | held-out 步态族流形连续性检验（{18,19,2} 近邻重建 ≤1.5× + 插值 lattice 合法率 ≥90%） | 上述近邻均"训什么测什么"，无 held-out 技能族检验协议 | 验证协议层面新 | **成立**（协议增量，非概念新点） |
| 3 | 学习型 4 族切换 vs frozen best-fixed 两臂对照 + 预注册停止规则 | APT 自带 selector（单 logit）；T-IE 2025 free gait transition | 严谨性增量（预注册 + 冻结对照），概念不新 | **部分成立** |
| 4 | Phase 0 执行保真度前置校准门（oracle token 回放实现率）+ 执行层三层衰减排序（Isaac 79.8% > 官方 WBC 48.7% > harness 17.5%） | ASAP 事后 delta 校正 | "事前门 + 分层归因"是新组织方式；D034"Isaac 比官方快 61%，两套 realized 互不外推"本身是可写的发现 | **成立**（方法论副产物） |
| 5 | 官方 deploy 回路数据卫生学（物理检验 + 自回归 context，禁开环提取） | DiffuseLoco 等用普通离线数据 | 数据来源论证独有（源于 D033 定界），但属实验设计而非方法 | **弱成立** |
| 6 | 负结果与定界（0.08 悬崖 = 蒸馏路径边界、相位彩票 38%、realized 不随 cmd） | 无直接对应 | 项目实际贡献主张（HANDOFF/README §3 口径） | **成立**（复现研究型贡献） |

**总判定**：创新点**足够支撑立得住的研究叙事**，前提是叙事定位为
"把 APT-RL 范式经官方 token 底座移植到 G1 + 步态流形扩展 + 执行保真度定界"，
并把上表先例全部主动引用。**不够支撑**"我们提出了多步态统一控制/潜空间
步态学习"式的方法主张——那两个位置已分别被 2604.19102 和 TIE 2025 占住。

## 可借鉴清单（怎么用）

1. **DiffuseLoco**（最值得动手借鉴）：①离线混合技能数据直接学单策略多技能
   的配方与消融设计；②其"不设显式 gate、切换由条件隐式承载"可作 Phase 5
   讨论节的天然对照（显式 4-way gate vs 隐式条件化）；③离线数据多模态
   （步态分布重叠段）的处理经验，对应我们过渡段数据。
2. **2604.19102 选择性 AMP**：**"动态步态不宜过强正则"**有外部实证——直接
   支持我们 Phase 3 的 per-族加权设计（HAPPY/JUMP 重建/正则权重应低于
   SLOW/RUN；现有 transition ×2 加权是其补充而非替代）。
3. **HugWBC**：G3 连续速度段（0.2–0.8 可调）的同类设计参照——它证明
   G1 上频率/摆高/速度连续参数条件化 + 真机成立；也是我们对外写作时
   "参数化步态条件化"先例的必须引用项。
4. **TIE 2025 skill latent space**：四足 latent multigait 的参数微调方式
   （腿摆高/体高在潜空间调）与我们 vb 连续轴类比；引言/related work 布局参照。
5. **ASAP**：Phase 0 校准门的引用锚（执行器动力学非均匀 → 必须分层校准）；
   未来若上真机（G5 后），delta action model 是现成的下一步路线。
6. **UniTracker**：G1 + CVAE 潜空间的最近邻写法参照（三阶段 vs 我们的
   冻结底座）；其"MLP 学生在部分观测下漂移、CVAE 学生注入全局意图"论证
   可借用于论证 latent 底板的必要性。
7. **MVAE / MaskedMimic**：思想源头引用；MaskedMimic 的掩码条件化是 G5
   （感知条件化/高度图→相位对齐）未来叙事的引用锚。
8. **Radosavovic 2024 / VLA 侧 RL Token、bottleneck latent**：一句话级
   引用——前者证明 pretrain→RL 在人形成立（但微调非冻结），后两者证明
   冻结底座 + RL 潜空间引导模式在操作域成立。

## 表述雷区（对外写作/立项书）

- ❌ 主创新写成"人形多步态统一 RL"→ 被 2604.19102、HugWBC 指近邻。
- ❌ 使用"skill latent space multigait"作为我们的术语主张 → TIE 2025 已占
  （四足）。
- ❌ "learned gait switching"单拎当新点 → APT 自带 selector，T-IE 2025 有
  free gait transition。
- ✅ 安全表述：**"首个在冻结官方部署级 token 解码器上构建的 G1 步态动作
  流形，配 held-out 族流形连续性检验协议与预注册的切换对照"**——所有
  限定词（冻结/官方/token/检验协议）都是防撞车护栏。
- 引用 realized 数字必须注明执行栈（Isaac / 官方 WBC / harness 两套互不
  外推，D034 caveat）；论文对照与对外材料只按 TO 线证据（不混线纪律）。

## 地形维度补遗（09-04 owner 追问"创新点难道不是地形解决吗"）

**不是，也不能是。** 按 HANDOFF/README §1.1 的项目自己的口径：官方组件
（planner/encoder/decoder）= 承重墙不是贡献；属于我们的是①跨先验迁移问题
本身（论文 RL 机制搬到第三方冻结先验上是否成立——论文和 SONIC 都不问）、
②蒸馏层自研架构、③双解耦潜空间（E37/E39）、④**产出是发现不是能力**。
地形在项目里是**考卷/测量域**，不是答案。

且地形解决是所有维度里**被占得最满**的：

- 源论文 APT-RL 标题即 "in the wild"：四足 wild 地形 + 感知 + 多技能 +
  ~6 m/s + 学习切换——"地形解决"恰是它已交付的东西（我们复现它）。
- 人形侧 [Radosavovic 2024（arXiv 2410.03654）](https://arxiv.org/abs/2410.03654)
  **盲走**（无感知）4 英里徒步 + 旧金山陡坡；感知系人形地形线更多。
- 本仓 08-14 前次调研（`LITERATURE_SURVEY_FROZEN_DECODER.md` ②）已确认：
  标准人形地形配方（4096 envs + 地形课程 + foot-clearance 奖励）盲走
  0.10m 噪声/0.23m 台阶，**远超我们的 0.06**。

地形维度里真正属于我们的，是"**冻结先验约束下地形能到哪**"的覆盖性与
定界发现，而非通过能力本身：Gate0 形状依赖边界（只凸 0.06 过/有坑 ±0.06
全倒/只凸 0.08 全倒）、0.08 悬崖归因修正（蒸馏路径边界非解码器，MQ09）、
D030 最小步态族 set-cover、D031 相位彩票 38%、D032 stones 过渡带死点。

**DS 计划 G2 的正确读法**随之明确：它是**流形技能多样性的覆盖性证据**
（4 族 + 学习切换 ≥ 各地形 best-fixed 单族，相对主张），不是"我们解决了
地形"的绝对主张。对外若把创新锚在地形通过率，反而是最弱表述——那正是
源论文与整条地形 RL 文献的腹地；安全锚仍是"冻结底座 + 流形 + 检验协议"。
（G5 感知条件化同理：地形感知通行能力已被占，我们的位置是"高度图→相位
对齐条件化 token 流形"这一特定组合，38%→~100% 是对自家缺陷的修复靶。）

## 检索留痕

- arXiv（researchclaw）：motion token VAE locomotion humanoid；multi-gait
  switching RL legged；latent skill interpolation legged；motion latent
  physics character control VAE；action tokenizer pretraining RL locomotion。
- web：APT-RL Science Robotics；ASAP delta action model；DiffuseLoco；
  HugWBC；skill latent space multigait TIE；frozen action decoder RL
  fine-tune humanoid；multi-gait humanoid VAE latent switching 2025。
- 未检索到同路线工作（冻结部署级 token 解码器 + VAE 流形 + held-out 族
  检验）；未发现 DS 计划必须搁置的撞车证据。
