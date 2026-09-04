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

## 地形处理路线谱系与 SONIC 定位补遗（09-04 owner 三问）

### 崎岖地形的三条处理路线

- **路线 A（主流）：从零 RL + 程序生成地形，不采任何真地形数据。**
  Isaac 4096+ 并行环境里参数化地形程序生成、难度课程递增；策略被奖励驱动
  "长出"涌现步态；teacher（特权高程图）→student（本体感知历史隐式地形估计，
  RMA 2021 / Miki 2022 Science Robotics wild）→零样本上真机；数据量级一夜
  数亿步仿真交互。代表：Rudin 2021（2101.01328）、Hwangbo 2019（actuator
  net）、Lee 2020（盲走）、extreme parkour（2309.14341）、Radosavovic 2024。
  **与我们"官方回路 1:1 实时采集 2.5h 拿 6–8 万步"是两种数据经济学。**
- **路线 B（动作先验，我们的近亲）**：参考动作（动捕库或自产教师轨迹）+
  判别器（AMP 系 2104.02180）或全身跟踪（ExBody/HumanPlus）或 token 预训练
  （APT）。强在自然性与全身技能，地形靠 APT 的感知+aux 才进 wild。
- **路线 C（模型基）**：MPC+WBC，学习只在低层/估计器；我们的 run_sim_loop
  WBC 是此系对照栈。

### APT 预训练数据与我们同构（重要）

APT 的 transformer 预训练数据也是**采样自已训练好的策略**（自家 TO 教师在
多技能上跑出的轨迹）——与我们采样 SONIC 官方回路结构同构。区别在先验的
**所有权**：他们的教师是自己的（可无限重采、可带地形/感知重训）；SONIC 是
第三方冻结部署系统，只能拿它吐的数据，且其模式集无地形技能。"被限制"的
精确含义 = 不是范式被限制，是先验的所有权与覆盖域被限制。

### "基于已训练动作泛化"的创新点判定

概念层是整个 motion-prior 领域的立身前提，**不能当概念创新卖**。有位置
的具体版本：①第三方冻结先验的**泛化边界测量**（0.06/0.08 定界、相位彩票、
执行衰减三层都是其产出，无人做过系统版）；②先验范式内扩容的**三角选择**：
加残差越出流形（APT aux；我方 E48 三配置全负）/ 解冻微调（E44 打转）/
**不动底座重采数据扩流形（DS 计划 = 此臂）**。卖点句式："第三方冻结先验
的泛化边界学 + 流形内扩容"，不是"基于预训练动作泛化"。

### SONIC 的谱系定位（⚠️ 本小节首句判断已勘误，见下节精读）

SONIC = NVIDIA GEAR **GR00T-WholeBodyControl**：64 维 latent motion token、
50Hz 解码到全身 29 关节、planner ONNX 10Hz 重规划 + C++ deploy 栈 + 500Hz
命令流。~~仓内从未引用配套论文——它是工程化开源资产，非研究首发~~
**【勘误 09-04】SONIC 有正式配套论文且发表于 Science Robotics**（见下节），
"工程资产非研究首发"判断错误；仓内文档确未引用它（本项目文档缺口，本次补）。
"先训动作大模型→冻结/复用→下游适配"的更早正源仍成立：Radosavovic
sensorimotor pre-training（2023，2306.10007）、动画线 MVAE→VQ-token；VLA 线
token 接口 + 冻结底座已是标配。SONIC 对我们的特殊性：第三方可得但**不给
端到端重训通道**——制造了"第三方冻结先验"这个研究对象。

### SONIC 配套论文精读（09-04 补；owner 指令"仔细研读引用"）

**Luo et al., "SONIC: Supersizing Motion Tracking for Natural Humanoid
Whole-Body Control"，Science Robotics 11(117) eaed4592 (2026) = arXiv
2511.07820**（v1 2025-11-11；NVIDIA GEAR，28 作者）。**与 APT-RL
（Science Robotics 11(116) eadz7397）是同刊相邻两卷的姊妹篇**——本项目
"跨先验迁移"的两侧因此都是 SR 2026 正式论文：APT-RL 的 RL 机制 × SONIC
的动作先验。

方法要点（带原文数字）：
- **架构**：三个 MLP encoder（robot=关节 pos+vel / human=SMPL 3D 关节 /
  hybrid=上身稀疏关键点+下身机器人运动）→ **FSQ 量化**（明确弃用 VQ-VAE
  防 codebook collapse；FSQ 比 VQ-VAE MPJPE 好 8.7mm）→ **2 token ×
  FSQ-32-32 = 64 维 universal token**；10 未来帧；输出 29 维关节目标 + PD。
  多速率：策略 50Hz / 命令流 500Hz / 运动规划 10Hz（与我方工程事实逐一吻合：
  planner ONNX 10Hz 重规划、真机 LowCmd 500Hz、decoder 50Hz）。
- **训练**：PPO + 非对称 actor-critic，总损失 = L_ppo+L_recon+L_token+L_cycle，
  straight-through 穿 FSQ；Isaac Lab **每 GPU 4096 envs**，16/32/128 GPU
  扩展实验 = 2K/9K/**21K GPU 小时**，50k iters，1.2M→42M 参数。
- **数据**：700h 原始动捕 → GMR+PyRoki 重定向 G1 → 611h / **100M+ 帧
  @50Hz** / 317,189 段 / 8,447 子类 / 33 大类；公开子集 **BONES-SEED
  142,220 条 288h（522 演员，HuggingFace）**；外部测试 PHUMA 68k 动作；
  自适应采样按 1s 分 bin（β=200, α=0.1）。
- **kinematic planner** = 潜空间自回归 masked token 预测，0.8–2.4s 片段、
  最快 100ms 重规划、25+ 技能/风格、命令速度 **0–6.0 m/s**、蹲/跪骨盆
  0.3–0.8m、爬行 0–0.5 m/s。（README 侧另有 MotionBricks=VQVAE 实时潜空间
  合成子项目；论文正文用 FSQ+GEM，无 MotionBricks。）
- **结果**：test-content 99.6% 成功 / MPJPE-L 23.8mm（vs BeyondMimic 81.6%）；
  vs OpenHomie 0–5 m/s 存活 98.5% vs 43.0%，**稳定跟踪至 ~4 m/s**；真机 G1
  123 条 99.2% 成功、25.7mm。**全文无崎岖地形评测**（"In-the-Wild
  Navigation"是导航不是地形通行）；limitation 自认极端/高动态下可能失稳。
- **下游复用是其设计意图**：VLA（GR00T N1.5）预测 78 维 = 64 维 token +
  14 维手关节；FSQ token 接口 vs 直接回归 SMPL = 68% vs 27% 成功率；
  5 项 loco-manip 平均 75%。L_token+L_cycle 保证三编码器在共享潜空间对齐
  （去掉散度 ×8）。

对我方主张的影响：
1. **定位升级**：我方"第三方冻结先验"不是无名资产，是 SR 2026 正式论文的
   42M 参数行为基础模型。HANDOFF §1 问题①（论文 RL 机制搬到第三方冻结
   先验上是否成立）因此更锐利——他们自己展示了 VLA→token 下游，**无人展示
   任务 RL→token（带流形检验）下游**；我方占的正是这个空位。
2. **地形空隙确认**：SONIC 论文无地形评测/课程，且训练数据是纯动捕（无
   物理地形交互）——D030 正障碍零通过、Gate0 0.06/0.08 边界是**先验数据域
   的必然**，且该先验的地形覆盖域**官方论文从未声明也从未测量**。我方 G2
   覆盖性主张填补的是官方空白，非与官方竞争。
3. **速度声明对照**：论文命令域 0–6 m/s、跟踪稳至 ~4 m/s，远高于我方
   D033 planner 2.12 / 回路 1.0 / harness 0.37——支持"RUN 族材料在先验中
   为真，衰减在执行栈"（D034 已证 Isaac 79.8%）。G3 1.5–3.0 目标有论文级
   上界背书。
4. **流形最近邻其实在官方栈内**：他们的 FSQ token 空间 + 潜空间 planner
   （masked token 预测）+ MotionBricks（VQVAE 合成）已经是"动作流形"三件套
   ——我方 token VAE 的 delta 必须对它们表述：连续速度/regime 条件化 +
   RL 消费 + held-out 族检验协议，官方三件套均无。
5. **⚠️ 新数据选项（未立项，owner 裁定）**：BONES-SEED 公开子集（142k 条
   G1 重定向动捕）+ 官方 encoder（ONNX 已发布）→ **可批量离线编码出 token
   数据**，绕开我方 2.5h 官方回路采集瓶颈；物理有效性有代理背书（decoder
   本就是在物理仿真中 PPO 训出来跟踪这些 token 的，真机 99.2%）。与 D033
   数据卫生学（防 planner 开环漂移）不冲突——那是"未检验的命令→token 路径"
   问题，而"编码器输出 + Phase 0 oracle 回放抽检门"可复用现成质检机制。
   风险：动捕重定向非动态可行解的比例未知，需 Phase 2 门先行抽检。

## 检索留痕

- arXiv（researchclaw）：motion token VAE locomotion humanoid；multi-gait
  switching RL legged；latent skill interpolation legged；motion latent
  physics character control VAE；action tokenizer pretraining RL locomotion。
- web：APT-RL Science Robotics；ASAP delta action model；DiffuseLoco；
  HugWBC；skill latent space multigait TIE；frozen action decoder RL
  fine-tune humanoid；multi-gait humanoid VAE latent switching 2025。
- 未检索到同路线工作（冻结部署级 token 解码器 + VAE 流形 + held-out 族
  检验）；未发现 DS 计划必须搁置的撞车证据。
- SONIC 配套论文（09-04 补）：GR00T-WholeBodyControl README Citation 块 +
  [arXiv 2511.07820](https://arxiv.org/abs/2511.07820) 摘要页 + v3 HTML
  全文精读；正文 FSQ/GEM，MotionBricks 仅 README 侧子项目。
