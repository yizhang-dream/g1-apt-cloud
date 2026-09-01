# TO40-C 设计定稿：力矩前馈通道门控实验（Rung 0）

> 【层位 L2 侧轴｜设计定稿：TO40-C（2026-09-01 定稿开跑）】↑
> `refine-logs/README.md`（扇出树根地图）｜上游：`TO_TORQUE_MAINLINE.md`
> （力矩主线宣言 §3 阶梯路线图，本轮 = Rung 0）、`TO38_PLAN.md`（评测协议与
> 决策表骨架，本轮平移复用）、`LEG_LEVEL_TO_REPORT.md`（TO36 三门收束）、
> `tracker/TO.md`（Run 行唯一事实源）｜状态：**活跃**

**这篇讲什么**：把论文（APT-RL, Science Robotics 2026）混合控制律里的"力矩前馈"
那一项，第一次接到我们的 RL 基座上，问一个是非题——**这条通道会不会像 E48 那样
把基座毁掉**。读完你会知道：三条训练臂怎么配、判据怎么定、结果落到哪个分支就
该往哪走。

---

## 0. 一句话问题与背景

论文的控制律是

```text
τ = τ_dec(潜空间/相位/速度)  +  kp·(q_d − q) − kd·q̇
     └── 力矩前馈（本轮的主角）      └── 位置 PD 闭环（我们已有）
```

我们至今**从未把 τ_dec 那一项接进闭环**（`TO_TORQUE_MAINLINE.md` §1 判定：
"混合控制 = 未开工"）。邻域已有两条判死史：

- **TO36-C 门**：TO 解开环回放（只有前馈、没有稳定层）1.84 s 必倒；
- **E48**：给策略开"动作侧自由残差通道"（29 维可学残差）→ 基座被噪声梯度占据，
  全地形劣化（负结果，见 `stages/stage11_e48_residual.md`）。

τ_ff 与 E48 的关键差别：**它是固定前馈，不含可学自由度**，策略学不到"用它作弊"，
失败模式理论上更温和——但**这只是推理，从未验证**。本轮就是验证这一步，它是整条
力矩主线的门控（gate）：通道不可用，Rung 1–5 全部无意义。

## 1. 注入口径（怎么把 τ 加进去）

**数据源**：`to38_ref.npz`（TO38 已导出的 LUT，M=120 × 一个 stride）里的
`tau_ref6`——来自 TO36 F11b 平地膝可行周期解（v_avg 0.277 m/s、T 2.4 s、审计
PASS），列序 SONIC 矢状 6 关节 [Lhip, Lknee, Lankle, Rhip, Rknee, Rankle]，B 门
符号映射已乘。峰值 |τ| ≈ 15–16 N·m（髋/膝）、4–6 N·m（踝）。

**注入方式**：Isaac Lab 的腿部执行器是**隐式 PD**（位置目标进、力矩出），所以

```text
给位置目标加偏移 dq  ⇔  给关节加力矩 kp·dq
τ_total = kp·(q_d + dq − q) − kd·q̇ = [kp·(q_d − q) − kd·q̇] + kp·dq
```

取 `dq = w_gate · w_τ · τ_ref(ψ) / kp` 即精确得到 `τ_ff = w_gate · w_τ · τ_ref`。
这条等价关系**只有 kp 用对才成立**（见 §2 的口径修正）。

**门控 w_gate**（复用 TO38 的 cmd 门）：`w_gate = exp(−(cmd_vx − 0.277)² / 0.0036)`
——即命令速度离 TO 解速度越远、前馈越弱（±0.06 m/s 内基本全开）。理由：单速度
TO 解只在它自己的速度点上有物理意义（TO37b 已证跨速度泛化 FAIL 86% std），门外
硬灌前馈等于灌噪声。

**时钟 ψ**：与 TO38 完全一致——自由跑（周期 2.4 s，reset 随机初相），**不与解码器
walk 时钟共钟**（解码器步频 ≈0.49 s/周期，与 TO 的 2.4 s/stride 差一个量级，共钟会
把步态压出 VAE 流形）。这带来一条**本轮已知设计限制**：τ_ff 与真实步态相位不同步，
论文形态是同步的；本轮问的是"通道是否可用/是否毁基座"，**相位同步版留作 Rung 0b**
（仅在本轮 PASS 时才有必要）。

**obs 不变**：三臂都带 TO38b 那 12 维零块（LUT 照载、时钟照跑、obs 置零），
观测维度完全一致 → 三臂差异**唯一**归因于 τ 通道，且与 TO38/39 的对照臂同口径。

## 2. 开跑前的口径修正（预注册的代码改动）

设计复核时发现 `apt_flat_env.py` 的 `to_tau_kp` 第 3/6 列（踝 pitch）写成
**40.17924**——那是 **hip_yaw** 的刚度（SONIC_KP 每腿排序为 [hip_pitch, hip_roll,
hip_yaw, knee, ankle_pitch, ankle_roll]，取前三个当 [髋, 膝, 踝] 是错位）。
踝 pitch 的真值是 `2×STIFFNESS_5020 = 28.50125`。

后果：不修则踝的实际前馈只有名义的 28.50/40.18 = **71%**，§1 的"dq 等价 τ_ff"
口径不成立，实验测的就不是它声称的东西。

**修法（本轮开跑前做，属实验前置而非结果）**：
1. 运行时**直接从 articulation 读实际 stiffness**（`robot.data.default_joint_stiffness`）
   取 6 个矢状关节的 kp，作为唯一真值；
2. cfg 默认值改为正确的 (99.09843, 99.09843, 28.50125) ×2，仅作读取失败时的回退；
3. 启动时打印 sim 读到的 kp 与 |dq| 峰值，进日志作为可追溯证据。

修正后 |dq| 峰值：髋 0.09–0.17 rad、膝 0.15 rad、踝 0.14–0.22 rad（≈5–13°）。

## 3. 三臂定义（lab-ts，同 commit / 同 seed / 同机顺序运行）

基线配方 = E47 精确配方（与 TO38/39 逐字一致）：
`--latent-mode --latent-vae-path <e39>/vae.pt --latent-speed-bins --latent-dir-bins
--latent-kl-prior zero --progress-scale 1.0 --heading-scale 0.4`，
128 envs × 2000 iters、seed 0、平地、cmd 采样 U(0, 0.8)。

| 臂 | 附加参数 | 角色 |
|---|---|---|
| **TO40c-ctrl** | `--to-ref --to-ref-npz to38_ref.npz --to-ref-obs-zero --to-ref-w 0` | 对照臂（= TO38b 配方在本 commit 下重跑；不复用旧 ckpt，保证同 commit 配对） |
| **TO40c-t10** | ctrl + `--to-tau --to-tau-w 1.0` | 主臂：满幅力矩前馈 |
| **TO40c-t05** | ctrl + `--to-tau --to-tau-w 0.5` | 剂量臂：半幅（剂量-响应，TO39 范式） |

> 为什么重训对照臂而不复用 TO38b：TO38b 的 ckpt 产自旧 commit，本轮改了 kp 口径与
> 注入代码路径；配对纪律（TO38_PLAN §2）要求"同一 commit、同一 seed、同机"。
> TO38b 旧结果仍作**跨轮一致性旁证**（同配方两次训练是否落在同一形态）。

## 4. 评测协议（TO38 §3 骨架 + TO39 路径效率主指标）

- **A 60s 存活**：每次 eval 内含 seed {0,1,2} × 60 s。命令点分三类：
  - **门开带（主指标）**：cmd {0.25, 0.277, 0.30}（w_gate ≥ 0.5，0.277 处 = 1.0）；
  - **TO38 锚点**：cmd 0.2（w_gate 0.19，复现低速带口径）；
  - **门外控制点**：cmd 0.5（w_gate ≈ 1e−6，注入实质关闭）——两臂**必须等效**，
    否则说明"门外无副作用"的前提破了（可证伪点）。
- **完整 battery**（A@0.8 + B 推扰 + C 切换 + D 跳跃）：只对 ctrl 与 t10 的 best
  ckpt 跑（门控判定所需），t05 只跑门开带 + 0.2（剂量证据）。
- **ckpt 规则（对称、非手挑）**：各臂 train_log reward 的 50-iter 窗口最优段，
  与 TO38/TO39 同规则；另跑 final ckpt 作稳健性对照。
- **2×2 交叉注入诊断（eval-only，不重训，cmd 0.277 × 3 seed）**：

  | | eval τ ON | eval τ OFF |
  |---|---|---|
  | **ctrl 策略** | 无适应时的通道扰动强度（E48 式破坏测试） | 基线 |
  | **t10 策略** | 训练/评测一致（主结果） | 策略对前馈的依赖度（撤走是否塌） |

## 5. 判据与决策表（开跑前预注册）

- **floor（每臂各自，失败即报失败、不进分支）**：A 60s `completed`、
  h_min ≥ 0.6、disp > 0.5 m（TO36-C 门口径）。
- **主指标 1**：门开带 |vx − cmd| 的 60 s 均值（seed×cmd 聚合），**配对差分**
  （同 seed 同 cmd 逐对作差），等效边界 δ = 0.03 m/s。
- **主指标 2（TO39 教训）**：路径效率 `disp / (v_speed · 60 s)`，< 0.5 判绕圈。
  弧线走在 TO38 曾漏网，本轮列主指标。
- **次指标**：h_min 分布、B 推扰完成率、C 切换位移、cmd 0.5 控制点差分。

**门控三分支（Rung 0 出口）**

| 分支 | 触发条件 | 结论与下一步 |
|---|---|---|
| **① 通道可用** | t10 floor 全过 且 主指标 1 差分 ≤ +δ（等效或更好）且 效率 ≥ 0.5 | τ_ff 消费通道成立 → 主线推进 Rung 1（速度网格加密 + 条件化 τ_dec）；若差分 < −δ 或效率显著提升，记"通道正向" |
| **② 通道无效** | floor 过、但三臂在全部指标上等效，且交叉诊断中 ctrl+τ ON 也无变化 | 注入信噪比不足（幅度太小/门太窄）→ 正路 = 提高 w_τ 或换真力矩叠加口径（effort superposition），而非放弃通道 |
| **③ 通道毁基座** | t10 floor 失败，或差分 > +δ 且/或 效率 < 0.5、h_min < 0.6 | 与 E48 同族的负结果 → **力矩前馈消费形态关闭**，主线回落 TO38 obs 注入形态（`TO_TORQUE_MAINLINE.md` §0 已预留此降级） |

**剂量-响应**：t05 落在 ctrl 与 t10 之间 = 机制证据（效应由 τ 幅度驱动）；
若 t05 与 t10 无差、或非单调，则效应可能来自训练噪声，判定降级为"不可判定"，
需补 seed。

**统计口径（诚实声明）**：每臂 n=1 训练 seed（与 TO38/39 同预算约束），
评测 3 eval seed；因此**跨臂差异的置信度由剂量单调性与交叉诊断背书**，
不宣称统计显著性。若主判定落在 δ 边界附近（|差分| ∈ [0.02, 0.04]），
按"不可判定"处理并补第二训练 seed。

## 6. 本轮不主张什么（边界）

1. **不主张力矩数值正确**：LUT 的 τ 出自平面 6-DOF 模型 × 符号映射，B 门数字口径
   FAIL、支撑链有 −5 N·m 的 URDF↔MJCF CoM 系统差（`TO_TORQUE_MAINLINE.md` §2.4）。
   本轮只测**通道**，不测力矩保真度。
2. **不主张论文形态达成**：ψ 自由跑 ≠ 论文的相位同步 τ_dec；且只有单速度。
3. **不主张跨速度能力**：门外 τ 基本关闭，0.5 控制点只用来证明"无副作用"。

## 7. 产物清单

commit sha ／ 三臂完整命令行 + seed ／ `train_log.json` ×3 ／ ckpt 目录 ×3 ／
评测 JSON（门开带 + 锚点 + 控制点 + full battery + 交叉诊断）／
`to40c_analyze.py` 汇总输出 ／ Run 行（`tracker/TO.md` TO40 节）。
canonical 在 lab-ts `~/ros2_data/apt_g1/outputs/`（`to40c_*`），
小产物（eval JSON / train_log / 汇总）同步进仓 `apt_g1/outputs/sync/to40c/`；
`EXPERIMENT_TRACKER.md` 行数同步、`apt_g1/SCRIPT_MAP.md` 登记新脚本、
`python refine-logs/tools/tree_check.py` 三项全绿后提交。

---

## 8. 开跑前落地的实现（2026-09-01，已进代码）

1. **kp 口径修正落地**：`apt_flat_env.py` 的 `to_tau_kp` 默认值改为
   (99.09843, 99.09843, 28.50125)×2；**运行时优先从 `robot.data.default_joint_stiffness`
   按关节名（不按 SONIC 顺序）重排取 6 个矢状关节真值**，仅不可用时回退 cfg。
   冒烟实测打印 `kp from sim (sagittal6) = [99.0984, 99.0984, 28.5012, 99.0984,
   99.0984, 28.5012]`（与 §2 预算一致，踝 28.50 而非 40.18）。
2. **冒烟 2 iters 通过**：rew 2.007→2.357、fall 0、kp 打印正确、无 NaN，ckpt/train_log
   正常落盘。
3. **注入口径复述**：`dq = tau_ref6 * (w_gate * w_tau / kp)`，只加在 6 个矢状关节的
   位置目标上；obs 三臂都是 12 维零块（`--to-ref-obs-zero`），仅 τ 通道不同。

---

## 9. 开跑后评审修正（2026-09-01，预注册准则修订并留痕）

> 设计经 rubric 评审（autoscirub 口径）后并入以下修订；**训练协议（三臂/配方/
> 注射）不变**，修订集中在**结果判定规则**与**未决情形的处置**，均为开跑前
> 预注册准则的明确化，不是对已执行协议的马后炮。

1. **分层互斥判定（替补 §5 的 δ 规则冲突）**：原 §5 分支表「diff > +δ → 分支③」
   与 §5 统计段「|diff|∈[0.02,0.04] → 不可判定」在 (0.03,0.04] 重叠（正负两侧都
   冲突）。改为分三层、互斥：
   - `|diff| ≤ 0.02` 且 PM2(效率) ≥ 0.5 → **分支①：通道可用**；
   - `0.02 < |diff| ≤ 0.04` → **未决窗**：不进三分支，**先补第二训练 seed**；
   - `|diff| > 0.04` → 明确分支：`< −0.04` = **通道正向（明显优）**、
     `> +0.04` = **分支③：通道毁基座（明显劣）**。
   未决窗与三分支**互斥**，不再有同一差值同时落入两支的歧义。
2. **主指标逐 cmd 达标**：门开带 0.25/0.277/0.30 **各自**须满足分层判定，合并配对
   均值只作报告、不替代逐 cmd。「0.277（gate=1.0，注入最强）」单列重点。合并均值
   若被 0.25/0.30 平掉导致 0.277 单独劣化，按「逐 cmd 判定」识别为局部退化而非整体等效。
3. **路径效率定义钉死**：PM2 = `disp / (v_speed · 60s)`，其中 `v_speed` = 60s 窗
   **实测平均速度模长**（`v_speed` 字段），`disp` = 基座**净位移欧氏模长**
   （2D）。该式 ≡ **直线度**：直行 ≈1.0、绕圈 <0.5。`to40c_analyze.py` 已按此实现。
4. **分支②（通道无效）的第一假设 = 相位异步抵消**：ψ 自由跑（≠论文相位同步 τ_dec）
   使 τ_ff 在步态相位上 assist/resist **平均抵消**，导致「信噪比不足」——这与「注入
   太小」在效应上不可区分。分支②触发时**先按相位异步抵消处置**（修法 = Rung 0b
   相位同步注入，或相位对齐），再考虑提 w_τ；避免朝错误方向调参。
5. **剂量-响应子判（t05 过、t10 不过）**：若 t05 floor 过而 t10 floor 失败，则
   剂量-响应已显示「效应由幅度驱动、半幅可用」——**不直接判分支③关闭形态**，
   而是先降幅到 t05 附近重试（剂量调优），仅 t05 也失败才判「通道毁基座」。
6. **ctrl floor 失败的专门处置**：ctrl 是基座对照，若其 floor 失败则整轮配对失效
   ——预注册：**整轮判「不可判定/重训」，不进三分支**；先排查 kp 读取/代码路径问题。
7. **力矩饱和核查（本轮已定量排除）**：施加的 |τ_ff| ≤ ~16 N·m（髋/膝）/4–6 N·m（踝），
   远低于执行器 effort limit（髋/膝 139 N·m、踝 50 N·m）——**饱和概率极低**。
   故「dq 偏移 ⇔ 加法力矩」等价在本轮成立，无需真力矩叠加口径。
8. **跨口径澄清**：§1「B 门符号已乘」指**方向正确性**已在 LUT 按 B 门校验；
   §6「B 门数字口径 FAIL」指 **τ 幅值保真度**（含 −5 N·m CoM 系统差）未验证。
   两者不矛盾：本轮只测通道、不测保真度。

---

## 10. Rung 1 预注册约束（2026-09-01 外部评审锁定，开跑前生效）

> 评审裁定：TO40C 收束质量合格（falsification 干净），但 PASS 语义与 Rung 1
> 科学目标须开跑前压紧，防止「τ_ff 有效 → 已实现目标速度 → 只需扩网格」的
> 叙事跃迁。本节为 Rung 1 开跑前置条件；**服务器空闲不构成开跑理由**。

### 10.1 分支① PASS 语义限定（通道级，非机制级）

**PASS-AS-CHANNEL，不是 PASS-AS-MECHANISM**：

> 分支① 通过 = τ_ff 通道已获得足够证据支持其作为**可用学习/控制信号**
> （非伪信号/数值假象，非 E48 式破坏），且存在**可重复的低速正向效应**；
> **尚不能声称已识别出目标速度 0.277 的真实 τ_ff 生产机制**。

Rung 1 核心 caveat（原封继承 tracker TO40C 收束 #4）：
**「τ_ff 把可达速度向低端扩展（填空洞），但未在本速度点生产真 0.277 步态」**。
Rung 1 及后续文档引用 TO40C 结论时，不得写成「τ_ff 有效 / τ_ff 是目标机制」
这类机制级表述。

### 10.2 科学目标重述：识别，不是更多速度点

Rung 1 的问题不是「τ_ff 能不能在更多速度上工作」，而是：

> **在控制 τ_dec mismatch 后，τ_ff 的真实效应曲线是什么，其速度支持区间
> 和边界在哪里？**

观测效应的 `observed effect = τ_ff 效应 + decoder mismatch + interaction`
只是**概念框架**，不是三个仅凭实验即可唯一识别的因果量。Rung 1 的交付物是
一组**预注册 contrast family**（用于区分 τ_ff 主效应、decoder mismatch 贡献
与二者 interaction 的可估计对比，定义见 §10.3），不是单点 pass/fail，
也不承诺唯一因果分解。

### 10.3 开跑前锁死四项 + contrast family

| 项目 | 预注册内容（标注【提案】的值级待批，其余为硬约束） |
|---|---|
| 速度网格 | **已批准（四轮评审落盘；七轮定名）**：canonical 定义 = **6-point regular grid（0.20–0.325 步长 0.025）+ 0.277 anchor = 7 target-speed records**——禁止写成「0.20–0.325 共 7 个均匀点」；0.5 门外控制点保留。primary = support/boundary 定位，曲线精拟合为非 primary。**边界依据 = 既有 robot/task operating envelope，非 TO40C 结果**：0.2–0.25 为已知解码器低速空洞带（TO38/39）、0.277 为 TO 参考解速度点（§4 门开带 gate=1.0 处）、≥0.3 为解码器健康带起点（TO40C ctrl 实测 vx 0.59），全部落在基座 cmd 支撑 U(0, 0.8)（E47 配方）之内；TO40C 观测只决定「在此区间做定位」这一决策。grid bounds are defined by the pre-existing robot/task operating envelope, while TO40C observations determine only the decision to investigate this interval. **定位网格声明**：Rung 1 speed grid is a pre-registered localization grid derived from the already-closed TO40C observation; it is not itself treated as confirmatory evidence for the existence of the effect. |
| τ_dec 条件化 | 条件变量 = cmd 速度。**pre-registered treatment mapping artifact**——mapping 改变 = treatment condition 改变，故属**实验设计 / treatment specification**，不只是实现知识（实现 invariance 只管 checkpoint/arch/preprocess/normalization 哈希，见 §10.7）。开跑前入仓一个不可变映射表（【提案】`apt_g1/configs/rung1_tau_dec_mapping.yaml`），字段：`target_speed / decoder_condition_id / command_regime / decoder_checkpoint_hash / decoder_architecture_hash / preprocessing_hash / normalization_hash / mapping_rule_version / mapping_provenance`（provenance 非自由文本；五轮增补见下方补充纪律 4–8）。**不设判定性 expected-mismatch 字段**——先验预期（如有）只能作非判定性设计注释，不得参与 stopping / 筛选 / 解释门槛（防预写「0.25 mismatch low、0.3 mismatch high」污染后验解释的 confirmation bias）。冻结后任何改动须新 commit + tracker 留痕——杜绝「某点效果差 → 回头换 decoder condition 再跑」 |
| 主指标 | primary = **逐 cmd 配对差分效应曲线 + 不确定度**（3 eval seed × 2 训练 seed）；合并均值只作 gate、不作效应量解释；衰减结构 −0.09→−0.009 本身是 Rung 1 要解释的对象。**禁筛选顺序**：−0.0402（TO40C gate）不得用作 Rung 1 逐点筛选条件；全部预注册速度点先全部取得 effect + uncertainty，**再**作 gate/boundary/disappearance 判定——不得先筛点再画曲线 |
| 停止规则 | 见 §10.4 |

**anti-selection invariant**：No speed point may be removed, added, reweighted,
or locally refined based on its observed effect before the primary Rung 1
analysis is frozen. 主分析冻结后若需局部加密（如边界夹在 0.25–0.275 时补
0.2625 级点），只能开 **Rung 1b extension**（独立预注册），不得回改 Rung 1
网格。

**treatment mapping artifact 补充纪律（四轮评审）**：

1. **provenance 非自由文本**：须含 `source_artifact_id / source_commit /
   generation_procedure`，可追溯到具体冻结 artifact / calibration 流程 /
   commit——「0.275 为何映射到 condition X」靠 artifact 可答，不靠作者记忆。
2. **变量身份唯一**：`target_speed`（= cmd）是 **treatment variable**；
   achieved speed（实测基座速度）是 **measured outcome / diagnostic
   variable**——两者不得互换（防 treatment/outcome 混淆）。
3. **冻结后改动 mapping = treatment specification 变更 = Rung 1 实验身份
   失效**：不得边跑边改，须重开预注册再 freeze。
4. **pre-run-only（五轮）**：YAML 禁一切 observed / expected 结果字段
   （observed_mismatch、expected_vx、posthoc_quality 连辅助性注释也不入）
   ——文件身份 = treatment specification，不是 specification + results。
5. **确定性可复现（五轮）**：同一 source artifact + 同一 mapping rule
   version + 同一 input → **byte-identical** 输出；mapping 是 reproducible
   transformation，非人工挑选表（杜绝「condition 是按当前经验选的」重开
   post-hoc selection）。
6. **身份块与双版本号（五轮）**：YAML 顶部含 `artifact_id / schema_version /
   mapping_rule_version / created_from / freeze_status`；**schema_version ≠
   mapping_rule_version**（schema 演进与规则变更的实验意义不同，不得混记）。
7. **command_regime = cmd 的 operational definition（五轮，实物锚定；六轮修正）**：
   `target_speed` = 每 episode **恒定**的指令前向速度（env `cmd_vx`；TO40C
   eval 实物即此口径——单值贯穿 60s 全程，无 episode 均值/初值/plateau 之辨）。
   decoder 条件选择**沿用冻结代码既有确定性函数**：
   `vb = bucketize(cmd_v, linspace(0, vx_max, n+1)[1:-1])`
   （`apt_flat_env.py`；n = latent_vae_n_bins = 3、vx_max = 0.8，生成时读
   冻结 cfg、不硬编码；bucketize right=False——恰在边界的值归上侧 bin，
   本网格无点触界）。**结构 = 7 target-speed records（6 均匀点 + 0.277 锚点）
   × 2 个 decoder conditions**：{0.20, 0.225, 0.25} → vb0（slow 带）、
   {0.275, 0.277, 0.30, 0.325} → vb1（mid 带）；forward cmd（y=0）→ dir bin 4
   （条件 id vb0_db4 / vb1_db4）。YAML = 该既有函数的**物化查表**，无人工挑选
   自由度（deterministic ≠ 科学充分：只保证无挑选自由度，不证明两态
   conditioning 足以控制 mismatch——后者正是 Rung 1 的识别问题）。
   slow/mid 边界 0.2667 与 TO40C 的 0.25→0.277 行为分裂带对齐：**post hoc
   consistency observation（诊断事实），不是网格设计依据，不得反向用作
   grid validity 论证**。
8. **source_commit 语义（五轮）**：指向**生成 mapping 的输入材料**（冻结
   cfg / ckpt / calibration artifact）所在 commit，不是「生成 YAML 时的
   HEAD」。

**contrast family（每预注册速度点 v 上的可估计对比，替代笼统「三分解」）**：

- **τ_ff 主效应**：`Δ_ff(v) = Y(τ ON, v) − Y(τ OFF, v)`（同一 τ_dec 状态内配对）；
- **decoder-conditioning contrast**：`Δ_cond(v) = Y(τ OFF, 条件化, v) −
  Y(τ OFF, 非条件化(0.6 步态混合), v)`。**estimand 用中性命名，不在名称里
  预设因果**：两臂间除 conditioning 外若还存在任何操作差异（mapping 选择、
  normalization、latent selection、implementation artifact、interaction），
  该对比测到的就不是 mismatch 本身；「诊断 decoder mismatch 贡献」只出现在
  **解释层**，不作 estimand 名称；
- **DiD interaction contrast**（命名纪律：始终称 contrast，不称 interaction
  effect）：`= Δ_ff(v)|条件化 − Δ_ff(v)|非条件化`。**解释前提**：
  DiD is interpreted as an interaction contrast only under the pre-registered
  comparability/invariance conditions of the two τ_dec states——两 τ_dec 状态
  不可比时，DiD 只是「两个不同 decoder 状态下 τ_ff 效应不同」；可比性条件
  满足后解释等级可提升，但仍不得写成「证明 interaction」或 τ_ff×τ_dec
  机制交互结论。

诚实声明：这是 contrast family（该对比基下可估计的量），主效应/conditioning
contrast 的数值依赖对比基选择；不宣称唯一因果分解。

### 10.4 停止规则（两条早停，触发即收束、不烧算力）

1. **消失规则**：低带效应在 τ_dec 条件化后消失 → 收束为「TO40C 低带效应由
   decoder mismatch 主导，τ_ff 无独立速度生产贡献」——**停止加密网格**，
   不逐 0.01 找显著点。**「消失」的数学定义（不显著 ≠ 归零）**：条件化后
   逐 cmd 效应的 seed 配对分布**全部落入 practical-null 区间 |Δ| ≤ 0.02 m/s**
   （复用 §9.1 分层判定的等效层界；落入 (0.02, 0.04] 未决窗则补第二训练
   seed，既不宣布消失也不宣布存在）；不得以 p > 0.05 或「看起来接近 0」替代。
   **0.02 的身份**：0.02 m/s is the pre-registered practical-equivalence
   boundary inherited from §9.1——它是**决策边界（decision boundary）**，
   不是估计误差界，更不是「效应为零」的自然常数/运动学阈值；「全部落入」是
   预注册的安全性/收束规则（operational stopping criterion），
   **不构成总体效应必然为零的 population-level inference**。
2. **边界规则**：连续两个网格点差分 > +0.04 或 floor 失败 → 停止向该方向
   外推，该速度记为机制边界。**「连续两点」= 预注册网格上的相邻点；无效点
   （run failure / decoder condition failure / 质量门失败）打断序列而非被
   跳过**——`0.25 PASS → 0.275 无效 → 0.30 PASS` 不构成「连续两点」。
   **无效点不可替代（no imputation）**：不得以插值、邻点均值或任何估计方式
   补出无效点的效应值用于停止规则判定。

仅当效应在条件化后保留且随速度连续衰减时，才允许外扩网格预算
（扩边 = 机制识别成功后的覆盖扩展，不是搜索显著点）。

### 10.5 排序确认：Rung 1 > Rung 0b

维持 tracker 收束 #5 排序：Rung 0b（相位同步注入）为**增量改进非前置**。
理由：TO40C 已证现有通道足以产生可检测低速效应，应先画效应 landscape
（识别实验），再优化机制实现（相位同步）；不让实现复杂度阻塞识别实验。

### 10.6 falsification 与 identification 的阶段边界

TO40C 完成的是 **falsification**（排除「E48 式基座破坏」这一竞争解释——
2×2 交叉 + 全电池 + 门外 eq 三处证据）；它**不自动推出**「τ_ff → 目标行为」
的因果。Rung 1 做的是 **identification**（机制定位）。两阶段结论不得合并表述。

### 10.7 implementation-invariance gate（冻结身份核查，Rung 1 启动 entry gate）

decoder 是冻结基座，本项目的科学问题是「decoder 冻结后 τ_ff 通道是否仍解释
行为变化」——故 decoder 的任何变化都是**实验身份变化**。核查内容：Rung 1 全部
速度条件下所用 decoder 的**权重 / 架构 / 预处理 / normalization / checkpoint
与冻结基线逐项一致**（哈希四件套，与 §10.3 mapping artifact 字段同源），
唯一变化来自预注册的 τ_ff / conditioning 操作。核查在 dry-run 阶段执行，但
**地位是 entry gate：decoder invariance FAIL → Rung 1 不得启动**——不是先跑、
发现哈希不一致再事后解释修正。否则「0.20 有效、0.325 无效」无法排除 decoder
artifact（实验身份问题，非统计问题）。核查记录（哈希 + 一致性 diff = 空）
进 §7 产物清单。

### 10.8 设计冻结审查清单（design-freeze review，七/八轮评审落盘）

> **状态板（八轮评审，9.9/10；两层 freeze 语义——Specification FROZEN ≠
> execution frozen，不得混写）**：
> Specification **FROZEN\***（\* = frozen pending incompatibility resolution;
> no implementation may reinterpret it——protocol frozen against
> implementation drift，仅 owner 可因 genuine incompatibility 决定 reopen；
> grid APPROVED、estimands/stopping/anti-selection
> LOCKED、invariance requirements LOCKED）｜ **Incompatibility #1
> RESOLVED（十二轮 owner 裁定 Decision 1 = YES / B，附推翻条款：
> 同速双 condition 无法科学构造则推翻 B）**｜ mapping **v2 GENERATED**
> （schema 2 全交叉 14 rows × τ ON/OFF = 28 cells，generated-not-frozen；
> v1 退役为自然 assignment 参照；**A PASS / B PASS**，C 待冻结复核，
> D pending）｜ Implementation **STARTED**（Unambiguous parts PROCEEDING，
> Conditioning realization = eval-time 条件覆写，随 v2 spec）｜
> **B 链位置（十四轮更新）**：① mapping reopen ✅ → ② 同速双-condition
> assignment spec ✅（v2 全交叉，A/B PASS）→ ③ **Decision 2′ = Mode A
> 裁定**（τ(v,C) = τ(v)：同 v 同 τ 材料、只变 decoder condition；R1/R2
> 消解；Mode B 永久关闭）→ ④ **material specification 草案完成**
> （`TO41_RUNG1_IMPL.md` §5：G(v)=TO36 dircol 逐网格点、机械验收八判据、
> infeasibility→material-unavailable→cells invalid 规则；**待 owner
> 冻结**）→ 材料生成（单独算力授权已附估算 ≈ ≤1 天墙钟）→ ⑤ 机械验收 →
> ⑥ A/B/C/D 重走（＋ τ material identity / conformance 检查）→ ⑦
> execution freeze → experiment。**Conditioning runtime BLOCKED**
> （override 仅限 condition-selection plumbing；plumbing 之外的内容变化 →
> D1 重定义——预声明 ≠ invariance）。**Δ_cond v2 恒等式 =
> Δ_ff(v,C1) − Δ_ff(v,C2)**（Mode A 下即「同 v 同 τ、只变 decoder
> condition」的受控对比；reopen delta，随 v2 freeze 批准）。
> D1/D2/D3 **PENDING**
> （最小样本 execution integrity test：7 speeds × τ_ff ON/OFF × 最小
> episode/seed；**非小型实验、效应估计为零目的**；D3 逐点对照表入产物）｜
> Execution freeze **PENDING**（owner decision）｜ **Compute BLOCKED**
> （D + execution freeze 全过前不得开跑；**并覆盖 per-condition τ
> material generation**——material-generation compute 事后单独授权，
> 仍非 experimental compute）。
>
> **D 三层判定（七轮定稿；三层全 PASS 才算 D PASS，禁用「dry-run
> successful」式模糊表述）**：
> **D1** decoder invariance——checkpoint / architecture / preprocessing /
> normalization 四哈希与冻结基线一致，**且 decoder 参数/state 不因
> τ_dec treatment 改变（conditioning 只选择、不修改）**；
> **D2** assignment invariance——同 target_speed 下 τ_ff ON/OFF → 同
> decoder condition（bin 仅依赖 cmd_vx、构造上正交；违反则 DiD 解释等级
> 下降）；
> **D3** artifact-runtime conformance——在运行环境中以冻结实现对同 7 个
> cmd 求值，runtime assignment 与 YAML records **逐项相等**（比哈希更强的
> 端到端证明，封堵 generator 与冻结实现两条执行路径的 divergence risk）。
>
> **implementation 纪律（八轮）**：实现只实例化冻结规格、不得重新解释——
> 禁改 grid / bin mapping / 默认值 / τ_ff ON-OFF 定义 / outcome / pairing，
> 禁条件筛选、禁按 dry-run 结果调 treatment、禁为跑通改 decoder、禁合并
> 条件或速度点（0.275≠0.277）；规格无法一致实现 → **report incompatibility
> → stop → owner 决定是否 reopen**（章程/代码面测绘/incompatibility 报告：
> `TO41_RUNG1_IMPL.md`）。变量身份进代码命名：target_speed = treatment ｜
> decoder_condition = conditioning state ｜ τ_ff = intervention ｜
> achieved speed = outcome/diagnostic；禁止含义模糊的裸 `speed` 命名。
>
> **冻结纪律（七轮）**：
> ① freeze 动作 = **纯状态迁移**——freeze commit 只改 freeze_status，
> 不得触碰任何内容字段；D 要求改 YAML → 回退 generated-not-frozen、
> 修正、重走 A/B/C/D。
> ② **D 先于 freeze**：D 全 PASS → freeze commit（同时记录 execution
> environment commit，与 source rule commit = 187f2fb 分列）→ final
> hash → EXECUTION-READY → 停止设计讨论、进入 compute。
> ③ artifact 100% conformant 与科学结果「两态 conditioning 不足」可同时
> 成立——后者不是 artifact 失败。
> ④ freeze decision 由项目 owner 执行；评审方裁定的是
> eligible-for-freeze。
> ⑤ 网格批准 = admissible + 有定位意义，非「已证各点为最优信息点」
> （admissible ≠ informative），不得追加 optimal-grid 类措辞。
> operator test = conditional pass（Q1/Q2 冻结后复核）。五轮 artifact 硬
> 约束见 §10.3 补充纪律 4–8。

| # | 锁点 | 状态 |
|---|---|---|
| 1 | 网格正式批准 | **已批准**（canonical 定义 = 6-point regular grid + 0.277 anchor = 7 target-speed records；边界依据 = operating envelope，见 §10.3） |
| 2 | τ_dec mapping artifact | **已产出**（468a1e7，generated-not-frozen）；A PASS / B PASS / C conditional / **D pending**；冻结 = 纯状态迁移 commit，D 全 PASS 后由 owner 执行 |
| 3 | 三类效应的确切 statistical contrast | **已锁**（§10.3：Δ_ff / decoder-conditioning contrast / DiD interaction contrast，附可比性前提） |
| 4 | practical-null /「消失」的数学定义 | **已锁**（§10.4：\|Δ\| ≤ 0.02 = 继承 §9.1 的 practical-equivalence decision boundary；未决窗补 seed；保守规则不 post-hoc 放宽） |
| 5 | 「连续两点」相邻点规则 | **已锁**（§10.4：无效点打断序列 + no imputation） |
| 6 | frozen decoder implementation-invariance gate | **已锁为 entry gate**（§10.7：FAIL → 不启动） |

执行顺序（评审确认）：scientific specification → pre-registration freeze →
implementation → dry-run / integrity checks（含 §10.7 entry gate）→ compute。
**compute availability ≠ experimental readiness**。
