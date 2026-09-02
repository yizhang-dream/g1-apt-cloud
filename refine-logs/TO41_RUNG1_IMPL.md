# TO41 Rung 1 implementation 章程：运行时实现与 integrity test

> 【层位 L3 实施章程】↑ `refine-logs/README.md`（扇出树根地图）｜
> 上游：`TO40C_PLAN.md` §10（**frozen specification**，规格侧一切判据以该节
> 为准，本文不得与其冲突）｜ 状态：**Implementation STARTED（八轮授权），
> Compute BLOCKED**（D + execution freeze 全过前不得开跑，并覆盖
> per-condition τ material generation——Decision 3）。分支状态：
> Unambiguous parts PROCEEDING（仅章程/harness 骨架）｜ Conditioning
> impl **BLOCKED** ｜ Incompatibility #1 已裁定为 Mode A（§3.3，十四轮）｜
> Estimands **LOCKED\*** ｜ **owner freeze review 完成（三十三轮，
> 2026-09-02，§8）：G↓ CLOSED/PASS + material 7/7 + 批准进入 D1/D2/D3 +
> 不为 0.277/0.300/0.325 重解；Main Rung 1 experimental compute 仍不
> 批准，待 D1/D2/D3 全 PASS + execution freeze**。
> **FROZEN\*/LOCKED\* 语义（十一轮）**：除 genuine incompatibility 外不得
> 修改；发现 incompatibility 时由 owner 决定是否 reopen——protocol frozen
> against implementation drift, but reopenable by owner decision。

## 0. 授权与两层 freeze

- 八轮裁定：implementation AUTHORIZED——**"只实现已冻结的 specification，
  不再修改 specification"**；protocol 侧（§10）自本章程起不再修改。
- 两层 freeze 语义：**specification freeze 已完成**（§10）；**execution
  freeze 未完成**（待 D 全 PASS 后由 owner 纯状态迁移）。任何文档不得把
  二者混写为「Protocol FROZEN」。

## 1. 实现纪律（不可协商，全文约束）

1. **instantiate, not reinterpret**。九不：不改 grid / 不改 bin mapping /
   不新增默认值 / 不改 τ_ff ON-OFF 定义 / 不改 outcome / 不改 pairing /
   不加条件筛选 / 不按 dry-run 结果调 treatment / 不为跑通改 decoder。
2. 工程便利不得合并条件、合并速度点（0.275 ≠ 0.277）、跳点；invalid point
   = break sequence + no imputation（§10.4）。
3. 规格无法与实际代码一致实现 → **report incompatibility（本文 §3）→
   stop → owner 决定是否 reopen**。禁止「修协议使其能跑」。
4. 变量身份进代码命名：`target_speed` = treatment ｜ `decoder_condition` =
   conditioning state ｜ `tau_ff` = intervention ｜ achieved speed（`vx` /
   `v_speed`）= outcome/diagnostic。禁止含义模糊的裸 `speed` 命名。

## 2. 代码面测绘（2026-09-01，冻结源 apt_flat_env.py@187f2fb）

- **τ 注入路径**（`apt_flat_env.py:765-771`）：`dq = tau * (to_tau_w *
  w).unsqueeze(1) / kp`，作用于 6 矢状关节位置目标；`w` = cmd-proximity
  Gaussian gate（`to_ref_gate2` = 0.0036，σ = 0.06，围绕 LUT 参考速度）。
- **τ 来源**：`to38_ref.npz::tau_ref6`——**单速 0.277** 相位表；
  `_to_ref_lookup()`（:776-785）按步态相位线性插值。
- **decoder 条件**：cmd→bin 的 bucketize（:583-591）已在基座栈内，
  与 mapping YAML 同一冻结函数（D3 的对照对象）。
- **可直接实例化部分**：τ_ff ON/OFF 两臂（ctrl / t10 配方）× 7 target-speed
  eval 网格 + 2 训练 seed（主指标要求 3 eval × 2 train seed；TO40C 仅
  seed 0，故两臂均需重训）= 现有 CLI 配方纯配置组合，无新协议面。
- **不可直接实例化部分**：条件化臂——见 §3。

## 3. incompatibility report #1（OPEN；十轮定稿——Decision 1 压缩为单一科学问题）

**estimand-family 提示（十轮）**：读法 A/B 不是对称的实现选项，而是两个
不同的 estimand family。

- **读法 A（自然 regime）**：C = C(v) 是 target-speed 的确定性函数，本身
  纳入 treatment regime——估计的是 **Y(v, z, C(v))**，即「自然 decoder
  regime 下的 τ_ff effect」。**A 不是「把 conditioning 去掉」**；不得事后
  指控「Rung 1 没有控制 decoder condition」——A 的效应定义就是自然条件
  效应。后果链（NO）：① 放弃 Δ_cond / conditioning DiD；② estimand
  收缩；③ §10 reopen；④ runtime 实现自然 C(v)；⑤ 现行 mapping artifact
  保留为自然 assignment artifact。

- **读法 B（interventional）**：设计变为**两个 treatment axes (v, z, C)**，
  最小设计矩阵 = v × z × C。**关键判据（十轮锁死，十一轮维持为 hard
  validity criterion、不降级为实现建议）：condition assignment 必须在同一
  target speed 内存在可操作变化**（同一 v 下 C₁ vs C₂）——否则
  effect(C₁)/effect(C₂) 仍与 v 混杂、interaction contrast 无身份；
  「runtime 只有 target_speed → condition」不构成 B。即 C = T(v, c)，
  c 为预注册 condition assignment。**有效性链（十一轮）**：可操作 →
  assignment invariant → decoder invariant → material valid → effect
  identifiable——**能指定两个 condition ≠ conditioning 有效**（可操作性
  ≠ 解释性）。**positivity / cell coverage gate（十一轮唯一新增统计护栏）**：
  B 的每个被比较 cell (v, C₁) 与 (v, C₂) 必须**实际存在且可运行**——
  implementation/D 阶段以 cell coverage gate 显式检查，而非仅验证
  assignment function；否则 interaction contrast 只是形式存在。后果链
  （YES）：① mapping reopen；② condition assignment 重新预注册；
  ③ representative-speed rule 预注册；④ τ material generation；
  ⑤ material acceptance；⑥ A/B/C/D 全部重走；⑦ execution freeze。

**Decision 2 — τ material rule（仅当 B；九轮基础 + 十轮硬约束）**：τ
reference 由**预注册、确定性的代表点规则**生成：`τ_c = G(v_c)`
（condition → representative speed → dircol → τ material），禁止人工挑
「看起来合适」的 τ。两条身份链分离：**condition identity 来自冻结 binning
函数；τ reference identity 来自单独冻结的 material-generation rule**——
不得做 bin = gait、bin → τ 的隐含推论（slow bin ≠ slow gait 是未经检验的
语义跃迁）。Δ_cond 措辞保持中性：其观测差异可能含 τ-reference construction
difference，不得写回「decoder mismatch effect」。**Compute 边界
（Decision 3，九轮）**：material generation 在 Decision 1/2 冻结前不授权
（跑 dircol = 执行未批准的设计选择）；冻结后**单独授权**，且仍不是
Rung 1 experimental compute。

**Decision 2 追加硬约束（十轮）**：v_c 必须由 treatment specification 本身
确定——**不得由 TO40C outcome、Rung 1 dry-run outcome 或任何 achieved-speed
measurement 反推**（防「0.25 的 τ 解得最好所以选 0.25」式 post-hoc
treatment construction）。

**material acceptance test（若 B 批准，先于一切实验使用）**：顺序 =
material specification → freeze material-generation rule → dircol →
**mechanical acceptance** → material frozen → experiment。acceptance 仅限
工程/数值判据：solver convergence、constraint validity、required fields
present、deterministic reproduction、hash、expected dimensionality、
no NaN/Inf。**禁止**以「跑出来的机器人效果很好」作为 acceptance
criterion——否则 material generation 本身变成实验优化。

**D3 解释上限（重申并加严）**：D3 PASS 只能说「implementation conforms to
treatment specification」，**不能说「conditioning is valid」**，也不能说
C₁/C₂ 是两个真实 gait regimes。

**评审方有条件倾向（留痕）**：若项目核心问题确为「控制 decoder-condition
mismatch 后识别 τ_ff 真实效应」，则读法 A 使其不可识别（C(v) 与 v 完全
绑定），倾向 B；该倾向以 owner 确认科学问题为前提，**不得为让现有 §10
能继续跑而选 B**。

**Decision 界面（单一科学问题，owner 裁定）**：

> Rung 1 是否需要在**相同 target speed** 下人为改变 decoder condition，
> 以识别 condition-dependent τ_ff effect？

YES → B 链七步（上）；NO → A 链五步（上）。裁定前：Conditioning impl
BLOCKED、τ material BLOCKED、Compute BLOCKED（含材料生成）。

### 3.1 裁决与 reopen 执行（十二轮）

**裁定（owner，十二轮）：Decision 1 = YES / B**——Rung 1 在相同 target
speed 下主动改变 decoder condition，识别 condition-dependent τ_ff effect。
理由 = 项目原始科学目标（真 0.277 步态 / 全域 τ_dec 精确解 / 为每个速度
生产匹配 τ）本身指向 controlled condition。**随附推翻条款**：若同速双
condition 最终无法科学构造，推翻 B 而非硬造 condition。

**阶段命名约束**：本阶段只叫 **controlled decoder-condition contrast**；
不得称 gait-condition effect——C₁/C₂ 是 assignment 槽位，「bin = gait」
是后续解释、非本阶段身份。

**mapping reopen 已执行（v2，generated-not-frozen）**：生成器升级 v2——
全交叉 `T(v, c) = cond_c`，14 rows（7 speeds × {C1, C2}）× τ ON/OFF =
**28 eval cells**；`natural_condition` 列标出非自然配对（0.275/0.277/0.30/
0.325 × C1 与 0.20/0.225/0.25 × C2 为干预所在）。v1 退役为自然 assignment
参照（schema v1，git 468a1e7）。**realization 语义**：条件覆写 = eval 时
作用于冻结 decode 路径的干预；训练臂不变（τ ON/OFF × 2 seeds）；
**Δ_cond v2 恒等式 = Δ_ff(v,C1) − Δ_ff(v,C2)**，Δ_ff(v,C) = Y(ON,C) −
Y(OFF,C)。

**设计矩阵（28 eval cells；矩阵本身 = positivity / cell coverage 证据）**：

| target_speed | C1·OFF | C1·ON | C2·OFF | C2·ON |
|---|---|---|---|---|
| 0.200 | RUN | RUN | RUN | RUN |
| 0.225 | RUN | RUN | RUN | RUN |
| 0.250 | RUN | RUN | RUN | RUN |
| 0.275 | RUN | RUN | RUN | RUN |
| 0.277 | RUN | RUN | RUN | RUN |
| 0.300 | RUN | RUN | RUN | RUN |
| 0.325 | RUN | RUN | RUN | RUN |

（C1 = vb0_db4、C2 = vb1_db4；任一 cell 无法运行 = invalid——不插值/
不跳过/不合并，与 §10.4 纪律一致。）

**代表性速度规则（B 链第③步，【提案】待批——批前 τ material 生成维持
BLOCKED）**：
- **R1（提案）**：v_c = 各 bin 内距边界最远的已批准网格点（slow→0.20，
  mid→0.325）。理由：bin 归属最纯（离 0.2667 边界最远）、均为已批准
  网格点、完全确定性。
- R2（备选）：v_c = bin 中点最近网格点——对当前几何与 R1 重合
  （slow 中点 0.1333→0.20；mid 中点 0.40→0.325）。
- 十轮独立性硬约束适用：v_c 不得以任何 outcome / achieved-speed 调整。
  v_c 批准 + material spec 冻结后，material-generation compute 单独授权
  （dircol@0.20 与 @0.325 两解，TO36 管线，服务器 CPU）。

**当前状态**：Specification FROZEN*（§10.3 的 Δ_cond v2 恒等式为 reopen
delta，随 v2 freeze 一并批准）｜ mapping **v2 GENERATED**（A PASS / B
PASS / C 待冻结复核 / D pending）｜ τ material BLOCKED（待 v_c 规则
批准）｜ Compute BLOCKED。

### 3.2 十三轮：R1 暂不批准——新切割问题（τ ∈ C？）

**批准维持**：YES/B、v2 全交叉 mapping、28-cell 设计、positivity gate、
controlled decoder-condition contrast 命名、推翻条款。**暂不批准**：R1
（0.20/0.325）、per-condition τ dircol、conditioning runtime 实现、D
dry-run。

**原因（比 incompatibility #1 更深的识别风险）**：v2 路径
C → v_c → τ_c 把 **decoder condition 与 τ material 绑定**——改变 C 将
同时改变两者，Δ_cond 便不再是 controlled decoder-condition contrast，而是
**joint condition/material contrast**。R1 的「距边界最远」优化的是
classification margin，不是 τ material 的科学代表性——工程 heuristic，
非 treatment rule，故不批准。这不是 B 失败，而是检查 B 的 treatment
是否被错误实现成另一个 treatment。

**新切割问题（唯一待裁，owner）**：

> **τ 是 C 的组成部分，还是在同一 target speed 上保持一致的控制量？**
> 即 τ(v,C) = τ(v)（Mode A）还是 τ(v,C) = τ_C（Mode B）？

- **Mode A（condition-only intervention）**：同一 target speed 的 C1/C2
  使用**同一份** τ material → Δ_cond(v) = Δ_ff(v,C1) − Δ_ff(v,C2) 接近
  **纯 condition contrast**。事实注记：Mode A 与原始目标「为每个速度生产
  匹配的 τ」一致（τ = τ(v)，speed-matched），且**消解代表速度问题**
  （v_c = v，R1/R2 不再需要）；成本 = 最多 7 个 dircol 解（每网格点一个）。
- **Mode B（condition-specific material）**：合法，但 estimand 必须改名
  **joint condition/material contrast**，不得再称纯 decoder-condition
  contrast；成本 = 2 个 dircol 解（代表规则届时重新讨论）。

**eval-time override 实现约束（十三轮加严）**：override 只得改变
**condition selection**（plumbing），不得触及 decoder weights、input
transform、normalization semantics、latent dimensionality、checkpoint；
若 preprocessing/normalization 文件发生 plumbing 之外的内容变化 → D1
须重新定义甚至重新审查——**预声明不把 invariance violation 变成
invariance**。

**状态板（十三轮）**：B decision APPROVED ｜ v2 mapping GENERATED /
A-B PASS ｜ cell coverage SPECIFIED ｜ conditioning runtime BLOCKED ｜
representative-speed DEFERRED（Mode A 下消解、Mode B 下重启）｜ τ
material BLOCKED ｜ material-generation BLOCKED ｜ D1/D2/D3 BLOCKED ｜
execution freeze BLOCKED ｜ compute BLOCKED ｜ **Decision 2′ OPEN
（owner）**。

### 3.3 裁决与 Mode A 落地（十四轮）

**裁定（owner，十四轮）：Decision 2′ = Mode A**——`τ(v,C) = τ(v)`。理由 =
estimand identification 干净：同 v、同 τ 材料、只变 decoder condition，
Δ_cond 才回答「**相同目标速度、相同 τ_ff 材料下，decoder condition 本身
是否改变 τ_ff 的效应**」。三好处：① 消除 C→τ 混杂（C1/C2 不再携带不同
τ material）；② R1/R2 失效且无需替代（representative speed 不再是
treatment specification 的组成部分，v_c = v）；③ 与原始目标「为每个速度
生产匹配 τ」同构。**Mode B 整条路线永久关闭**（除非未来另开实验回答
joint condition/material question）。

**实现契约**：(v, z, C1, τ(v)) vs (v, z, C2, τ(v))——同一 v 下逐 cell 使用
**完全相同的 τ material identity**，唯一预定改变 = decoder condition
selection。**D2/D3 扩展检查（十四轮）**：same target speed → same τ
material identity across C1/C2（防 Mode A 被偷偷做成 Mode B）；每速度另过
material conformance check：`τ_runtime(v) == τ_frozen-material(v)`。

**科学边界（保留）**：Mode A 成功也不得写「证明了 decoder condition 是
τ_ff effect 的机制」——识别的是 **controlled decoder-condition contrast**；
C1/C2 是人为 assignment，不自动等同两个自然 gait/ecological regimes。

## 4. D dry-run 设计（execution integrity test，非小型实验）

- 最小样本：7 target speeds × τ_ff ON/OFF × 最小 episode/seed；
  **效应估计为零目的**，dry-run 结果不得用于调整 treatment。
- **D1**：四哈希对冻结基线 + decoder state_dict 前后全等（conditioning
  只选择、不修改参数）。
- **D2**：两臂同 target_speed 的 decoder condition 逐项一致；**＋ τ
  material identity 一致（Mode A，十四轮）：同一 v 下 C1/C2 cells 使用
  同一 τ(v) 身份——防 Mode A 被偷偷做成 Mode B**。
- **D3**：运行环境冻结实现对 7 cmd 求值 vs YAML——**逐点对照表**入产物
  （`0.200 yaml=vb0_db4 runtime=vb0_db4 PASS` / …），不只报总 PASS。
  **D3 范围限制（九轮）**：只证明 decoder condition assignment 正确；
  若 Decision 2 落地，τ material 另需自己的 **material conformance check**
  （`τ_runtime(v) == τ_frozen-material(v)` 逐速度对照——**十四轮起适用**，
  Mode A 下按 v 逐点），不得并入 D3 解释。
- D 全 PASS → owner freeze（纯状态迁移 commit，只改 freeze_status）→
  EXECUTION-READY → compute。
- **三十三轮增补（owner freeze review，§8 详；execution sanity，非新
  protocol rule）**：最终 D artifact 须机械生成 **28-cell table**
  （7 v × {C1,C2} × {τ ON,τ OFF}，per-cell τ material hash）——逐格验证
  「same v → same τ identity」且 7 个 target speed 在最终 runtime 真正
  被 launch（非仅存在于 mapping.yaml），机械对应 §6 审计问 2；telemetry
  **预保存诊断量**：`target_speed` / `material_realized_speed` /
  `abs_error` / `material_hash` / `source_lineage` / `decoder_condition`
  / `tau_ff`（诊断用途，不改 estimand、不加 stopping rule）。

## 5. τ(v) material specification（Mode A；十五轮 spec FROZEN）

### 5.1 生成规则 G（确定性，无代表速度）

`τ(v) = G(v)`：对每个已批准网格点 v ∈ {0.200, 0.225, 0.250, 0.275, 0.277,
0.300, 0.325}，用 TO36 dircol 管线（`to36_hybrid_dircol.py`，Drake/IPOPT，
腿级周期解）生成**速度匹配的** τ 参考表（`tau_ref6` 相位表，格式对齐
`to38_ref.npz`）。**v_c = v**——代表速度概念消解；十轮硬约束（不得由
outcome/achieved-speed 反推）自动满足。产物 = 7 份材料 + 一份 manifest
（每份 sha256 + 生成参数 + 审计记录）。

### 5.2 mechanical acceptance（仅工程/数值判据；禁「机器人跑得好」）

① IPOPT 收敛（solver status）；② TO36 物理审计全过（能量漂移/冲击残差/
关节限位，逐项，沿用 TO36 验收制）；③ 周期闭合残差 ≤ 预注册阈值；④ 必填
字段齐全（6 矢状关节 × 相位网格）；⑤ 维数/格式对齐 to38_ref.npz；⑥ 无
NaN/Inf；⑦ 确定性复现（同参数重跑 LUT hash 一致）；⑧ 每份 sha256 入
manifest。

### 5.3 infeasibility 规则、stop rule 与搜索预算（十五轮定稿）

**搜索预算（两层，防止「预算超支但 protocol 没写」的争议）**：
- **Hard resource cap**：每速度最多 **8 个预注册 deterministic starts**；
  全战役最多 **7 × 8 = 56 starts**；服务器同时最多 **8 个 dircol 进程**
  （8-way concurrency）。
- **Planning estimate（非上限、非承诺）**：预计每速度 1–2 starts 收敛，
  单 start 40–60 min；实际墙钟取决于并行调度与提前成功停止（纯求解量级
  2–7 h + overhead），**不做单点「≤X 天」承诺**。

**per-speed stop rule（canonical material identity 闭环）**：每速度按
**预注册 deterministic seed order** 顺序尝试；**首个完整通过 §5.2 机械
验收的候选即冻结为该速度 canonical material，并停止该速度后续 starts**
——不因预算未用完而继续跑。**禁止按 outcome 选候选**：「最平滑」「残差
最小」「机器人跑得最好看」均属 post-hoc material optimization。同一 v
多解合格时，canonical = 预注册 seed order 中**首个**全过验收者——material
identity 由规则唯一确定，零挑选自由度。

**seed tuple（完整生成输入身份）**：manifest 必须记录完整 start 身份——
solver seed / initial-state seed / initial-guess source / continuation /
warm-start policy——只记 RNG seed 不保证 same optimization problem。预注册
seed order 的具体 tuple 表在 campaign manifest 固化（launch preparation
产物）；**规则本身**（顺序尝试、首过即停、完整 tuple 记录）由本节冻结。

**infeasibility**：某速度 8 starts 全部失败 → **material-unavailable** →
对应 cells 判 invalid（不插值/不跳过/不合并/不换代表速度）。低速点
0.200/0.225/0.250 风险真实（TO36 F 线最好 0.318 m/s）。

**报告纪律（十五轮）**：**material-unavailable ≠ experimental null**——
最终分析必须区分「treatment material 生成失败 / 不支持的工作点」与「实验
测得效应 ≈ 0」；低速 cells 因材料缺席消失时，报告不得被解读为「τ_ff 低速
无效应」。

### 5.4 compute 授权（十五轮：授权预算，不授权结果）

授权语义 = **authorize the pre-registered search budget**：最多 56 starts、
8-way concurrency——**不是**「保证 7 个速度全部产出可用 τ」（低速
material-unavailable 是本 spec 明示的可能结局）。范围仅 material
generation（**非** experimental compute）；Rung 1 experimental compute 仍须
execution freeze 后另行启动。

### 5.5 freeze 程序与当前状态（十五轮）

**本节（§5 material specification）已获 owner 批准并 FROZEN（十五轮）；
material-generation compute 已授权（campaign budget：≤56 starts / 8-way
concurrency）**。执行序：campaign manifest（完整 seed tuple 表）固化 →
dircol 生成（按 5.3 stop rule 顺序尝试、首过即停）→ §5.2 机械验收 →
manifest + 材料冻结（纯状态迁移）→ D1/D2/D3 dry-run（含 τ material
identity / conformance 检查）→ execution freeze → experiment。

### 5.6 campaign 执行不变量（十六轮；launch 前逐条确认）

1. campaign manifest = **只读预注册 planned search sequence**：planned
   start order / actual_start / accepted_start 三账分离——manifest 只含
   planned；actual 入 campaign log（runtime），accepted 入 material
   manifest（最终）。每速度 8 starts 是**允许的确定性搜索序列，不是必须
   执行的 workload**（N_actual(v) ≤ 8）。
2. manifest freeze → launch 后不得改 seed order；基础设施失败记
   `status = infrastructure_failure`，不删除、不前移。
3. **campaign accounting 三态 + 例外**：`solver_failed` /
   `solver_converged_but_mechanical_invalid` / `mechanically_valid`（+
   `infrastructure_failure`）——solver convergence ≠ material validity，
   事后读者必须能区分「8 次全败」败在哪一层。
4. **首过即停由脚本机械执行**（验收结果自动推进 freeze/stop），禁止
   人工看日志判断「这个解不错」再手动停。
5. **两个 manifest 身份分离**：campaign manifest（允许尝试什么）vs
   material manifest（最终采用了什么：v → accepted_start#k → tau_sha256）。
   审计链 = planned → attempted → accepted → frozen。
6. **campaign 期间禁止查看机器人 experiment outcome 来选择/淘汰 τ**——
   验收仅依据 §5.2 机械判据；此边界写进 launch 脚本输出/operator 说明。
7. 低速材料缺席 = **structural missingness due to unavailable material**，
   最终统计输入单独归类，不得混入普通 missing observation。
8. **并发语义**：`C_global ≤ 8`（全局），非 per-speed ≤ 8；**同一速度内
   seed 严格顺序执行**（wave 调度：wave w = 各未收速度的第 w 个 start），
   canonical = 预注册顺序中首个通过者（顺序模式）。
9. campaign manifest 文件 = `apt_g1/configs/rung1_tau_campaign_manifest.yaml`
   （planned sequence 已产出）；**Decision 2″ 未决（速度定位机制 + 容差带）
   前 manifest 不 freeze、campaign 不 launch**。

### 5.7 热启动 reopen（十九轮裁定落地；范围 = material-generation procedure，不涉科学协议）

**裁定**：(i) 热启动输入 reopen **批准**；(ii) `--v-target` solver 扩展
**不批准**；剩余 4 个 cold starts **不执行**（hard cap 是上限不是 spend
target，未消耗预算不自动结转为 hot-start 预算）。**范围控制**：仅
generation procedure reopen——B / Mode A / v2 mapping / 7 speeds /
τ(v,C)=τ(v) / 容差 0.02 / 机械验收 全部不变。

**accounting 修正（两字段制）**：`solver_terminal_status`（k=0..3 均为
solver_failed）与 `material_status`（k=3 = mechanically_invalid，因回退解
A门=FAIL + 容差门 FAIL）分离记录；k=3 的旧 "PASS" 事件标 superseded 保留
（driver_buggy_classification → manual review → corrected 审计链，不覆盖）。

**driver 全门集修复（campaign 恢复前置，已完成）**：
`apt_g1/to41_material_driver.py`——PASS = G1 字段 ∧ G2 solver terminal
success（log 末级 success=True）∧ G345 审计（gate_a∧gate_b、drift<2.0、
ke_drop≥−1e-6，镜像管线 `_audit_pass`/AUDIT_THRESH）∧ G6 无 NaN/Inf ∧
G7 速度容差 ∧ G8 配置身份；validate 子命令输出逐门 JSON。

**热启动源要求（freeze 前置检查单）**：`source_artifact_id / source_commit /
sha256 / contains(v_aux, lam, XM) / solver+config provenance / state
dimensions / model compatibility`；**to38_ref.npz 仍禁止作 --guess-npz**
（不因 reopen 放宽）。服务器候选已在：`to36_hybrid_gait_F9.npz`(0.318)、
`F11b_flat`、`to37_v0.08/v0.16`、`to37_fast56_v0435`、`to37_fast48b_v0678`。

**source-selection rule（二十轮定稿；二十一轮形式化 + APPROVED）**：

> `s(v) = argmax{ s ∈ R_valid : v_dump(s) ≤ v }`，其中 **R_valid = { s：
> provenance / schema / hash / compatibility / material-validity 全通过的
> 冻结 canonical sources }**——机械审计失败的 dump 即使 v_dump 落在区间内
> 也不进 R_valid。选**不超过 target 的最高已验证速度** source；continuation
> 只沿「已验证低速 → 目标」方向前进，**禁止用高于 target 的 source 反向
> 求 target**。反向 continuation 仅在有冻结且验证过的操作证据时另立规则。
> 无合格 source → source-unavailable → material-unavailable 链，禁止
> fallback。

tie-break = 冻结 registry 的 **canonical source identity / 固定 artifact-ID
排序**——不用 source_commit 新旧。`v_dump` 一律取 artifact 内**实测
realized speed**（v_avg 字段），文件名名义标签不作依据。G(v) 身份不变：
热启动只改求解路径，材料 identity 仍是 τ(v)。

**registry 纪律（二十一轮）**：① 预期 assignment 在 registry freeze 前一律
标注 **expected / pending verification**，文件名速度不是事实；② freeze 后
registry **immutable**——campaign/smoke 中发现的新 dump 不得动态加入（新
source = 新 registry version + 新 generation decision）；③ registry freeze
后生成 **source-selection audit table**（target | eligible | selected |
reason 逐行机械生成，0.300 行须显示 F9 因 v_dump>target 被排除）。

**按规则的 expected assignment（PENDING registry verification——非事实）**：
0.200–0.300 → to37_v0.16（expected）；0.325 → F9（expected）。smoke 证据
分层（二十一轮）：historical feasibility（F 线先验，仅工程依据）→ 本次
smoke reachability → 本次 material validity——**三层不互相替代**；smoke
判定拆分报告：`reachability PASS/FAIL` 与 `material validity PASS/FAIL`
分开（stage 5 抵达但审计 FAIL = reachability PASS + material validity
FAIL，不记 smoke FAIL）。

**执行序（最小）**：driver 全门集 ✅ → 冻结 hot-start source artifact +
selection rule → 更新 campaign manifest（hot-start ladder）→ 一个 smoke
continuation（判据 = 推过 stage-4 barrier 并抵达目标阶段）→ 机械 + 速度
门 → canonical material。smoke 仍在 stage-4 附近失败 → 才升级讨论 TO36
reachability boundary / solver 扩展。

**状态板（十九轮）**：Decision 2″ REOPENED → HOT-START PATH ｜ B / Mode A /
v2 mapping LOCKED ｜ τ(v) spec + acceptance FROZEN ｜ campaign budget
AUTHORIZED ｜ cold k=0..3 CLOSED/AUDITED ｜ cold k=4..7 NOT EXECUTED ｜
hot-start source + rule PENDING FREEZE ｜ material compute BLOCKED ｜
D1/D2/D3 PENDING ｜ execution freeze PENDING ｜ Rung 1 experiment BLOCKED。

### 5.8 Decision 2″ 裁决与 smoke 协议（十七轮；十八轮 smoke 执行日志附后）

**裁定（owner，十七轮）：接受提案**——机制 = `--stages` 末段
`v_min = requested_v`（现有旗标实例化，**暂不扩 solver**；`--v-target`
仅在 smoke 失败时作为 reopen 议题评估）。容差判据⑨：
`|v_realized − v| ≤ 0.02 m/s` 入 §5.2 机械验收。

**解释纪律（三条，不得突破）**：
1. 0.02 是 **material validity gate**，不是相邻 target speed 的统计可
   区分界（0.250/0.275/0.277 间距 < 0.02）——它只回答「解是否足够接近
   被要求的目标」，不回答「realized speed 可代表哪个 target」。
2. material manifest 必存 `requested_v / v_realized / abs_error`；
   canonical identity = **requested_v + canonical seed/order**——不得按
   v_realized 重命名、合并或替换材料。`v_min = v` 是 target specification
   mechanism，`v_realized` 是 material-generation outcome，二者不得互换。
3. **campaign log 固定语句**：passing v_realized 只是材料生成容差内命中
   目标的证据，**不是**冻结解码器或 τ 机制科学有效的证据。

**smoke 协议（manifest freeze 的前置）**：v = 0.277，ladder k=0 tuple
（knots=14 / max-iter=4000 / retries=0 / cold）+ 末段 v_min=0.277。验收
五条：① CLI 解析末段 v_min 符合预期；② v_realized 可获得；③
|v_realized − 0.277| ≤ 0.02；④ 同一完整 tuple 重跑具有确定性；⑤
`_audit_pass` 正常接管机械验收。**smoke 失败 → 不许调 knobs 凑 ±0.02 →
Decision 2″ reopen → 评估 --v-target solver 扩展**。smoke 通过则 run A
追认为 campaign start k=0@0.277（配置与 ladder 完全一致），run B 为
确定性见证（预声明：见证不占 56-start 预算）。

**smoke 执行日志（实时追加）**：
- **k=0（run A/B，十七轮）**：双双于 IPOPT stage 1/5
  `EXIT: Error in step computation` **确定性失败**（同位置同状态 → 判据④
  在失败语义下成立）。分类 = `solver_failed`（冷启动同伦第一阶段数值
  失败，**先于** Decision 2″ 定位语义被测试——末段 v_min=0.277 尚未触及，
  故 --v-target reopen 不触发：失败与定位机制无关，reopen 属范畴错误）。
  处置 = 按冻结 ladder 续跑 k=1..7（§5.3 stop rule 脚本机械执行，首过
  即停）。证据：服务器 `outputs/to41_smoke_smoke.log`（k=0）+
  `outputs/to41_smoke_k1to7.log`（续跑）；脚本
  `/home/cvgluser/to41_smoke.sh`、`/home/cvgluser/to41_smoke_k1to7.sh`
  （md5 已核对）；执行 commit = 服务器 5b1f413（管线代码与本地一致）。
- **k=1..3 与 ladder 暂停（十八轮追加，含勘误）**：
  - k=1（retries=1）/ k=2（knots=18）：`solver_failed`（快速失败）。
  - k=3（knots=24/t_max=2.0）：同伦爬升 **stage 4（v_min=0.16）`EXIT: Error
    in step computation` 崩溃**；管线 dump stage-3 水平回退解，dump 行自记
    `步速=0.140m/s A门=FAIL`。v_avg=0.140 → 容差门⑨
    |0.140−0.277|=0.137 ≫ 0.02 **FAIL** → 改分类
    `solver_converged_but_mechanical_invalid`（续跑脚本只 grep 能量审计
    "OK"，PASS 判定不完整——**campaign driver 必须实现全门集**：A 门 +
    速度容差 + 机械验收，此为 execution 修正非 spec 变更）。
  - **ladder 于 k=3 暂停，k=4..7 未运行**（预算消耗 **4/8**；勘误：k=1..7
    为 7 个 start，累计 ≤8，无 k=8——十七轮汇报笔误已 corrected）。暂停
    理由：k=4..7 的 t_max ≥ 2.4（更长 T → 更慢解）对「到达 0.277」这一
    目的被 k=3 的失败支配，先解决定位机制再议；owner 可改判预算耗尽优先。
  - **Decision 2″ REOPENED（按十七轮预定触发生效）**：证据 = 末段
    v_min=0.277 **从未被到达**（stage 4 即崩）+ 回退解 v_avg=0.140 远离
    目标——v_min 阶梯**无法把 realized speed 定位到目标**。smoke 五条中
    ②③ FAIL。十八轮措辞锁定：**「预注册 k=0/k=3 配置分别在 stage 1/4
    确定性失败，尚未测试到目标速度阶段」，不写「0.277 不可达」**；k=0
    失败仅确立该 solver 配置失败；确定性复现的正式验收仍在成功 candidate
    上做（k=0 的重现只是工程诊断事实）。
- **reopen 后的两个候选路径（owner 裁定）**：
  - **(i) 热启动输入 reopen**：manifest 的 `guess: cold` 改为允许
    `--guess-npz` 挂接 F 线既有解 dump（含 v_aux/lam/XM，服务器定位 +
    sha256 冻结为 material-generation 输入）。F 线历史结论「平地冷/热启动
    均进不了 v_td 盆」提示冷启动路线先天受限，而 F9/F11 本身就是热启动
    continuation 产物——**推荐**，且零 solver 代码变更。
  - **(ii) `--v-target` solver 扩展**：给 solve 加显式速度约束（十七轮
    预案）。变更面更大，且同样要面对 stage-4 攀爬失败（约束不解决
    reachability）。
  - 两案均不改变 §5 的机械验收与预算框架；裁定前 τ material / campaign
    launch 维持 BLOCKED。

### 5.9 downward-continuation sub-campaign（二十轮裁决；独立实验身份）

**未选①/②理由（留痕）**：① 会把 Rung 1 缩成中高速 contrast，而主问题恰是
**0.20–0.25 解码器空洞带 + τ_ff 正向效应**——不能让「无法生成 τ」伪装成
「低速不重要」；② 向下续延（0.2768→0.20）是新 generation operator，
当前无冻结、验证过的操作证据，动用即违反上一轮锁定的 continuation
discipline。

**③ 身份**：不是主 protocol 的补丁，而是对「低速材料可生成性」这一新问题的
**独立 sub-campaign / 独立实验身份**，单独预注册 `G↓(v)`（source
selection / continuation direction / solver settings / seed budget /
acceptance / infeasibility / canonical material rule 全部独立冻结，不得从
主 campaign 借临时规则）。

**状态板**：Main registry FROZEN ｜ Upstream selection rule LOCKED ｜
Low-speed material UNAVAILABLE（0.200/0.225/0.250/0.275）｜ Downward
continuation = NEW SUB-CAMPAIGN ｜ Main Rung 1 compute BLOCKED ｜ Main
campaign manifest NOT YET UPDATED。

**Sub-campaign spec（下一份独立文档，起草中）**：targets {0.200, 0.225,
0.250, 0.275}；continuation direction = downward；source registry 沿用
R_valid（v_dump ≥ 0.2764 上游；downward 起点 = 经独立验证的向下投影）；
acceptance/infeasibility/canonical 与主 campaign 同构但**独立编号**；
预注册问题 =「是否存在机械有效、可复现的 downward continuation，使
0.20–0.275 获得合法 τ(v)？」。**主链位置**：冻结主 spec → 低速 sub-campaign
→ 最终 material availability map → Rung 1 执行——sub-campaign 若证
0.20–0.275 仍 material-unavailable，structural missingness 即成为**先验
确定的输入事实**，非执行意外缺口。

**二十一轮 registry 结论**：monotone upstream 下 0.200/0.225/0.250/0.275
= SOURCE-UNAVAILABLE（R_valid 最低 0.2764）。文件名速度标签系统性不可信
（to37_v0.08/v0.12/v0.16/seed 实测均 ≈0.2764 同 sha 族）——已通过
`rung1_source_registry.yaml`（v_avg 实测身份）隔离此污染；旧 F 线命名不作
任何 continuation selection 依据。

## 6. implementation audit 八问（每步自检）

1 YAML 是否被原样消费？ 2 7 个 cmd 是否全部 launch？ 3 τ_ff ON/OFF 是否
只改变预注册 treatment？ 4 decoder 是否保持 frozen？ 5 assignment 是否
一致？ 6 outcome logging 是否完整？ 7 seed pairing 是否保持？ 8 failure
是否按预注册规则处理？

## 7. 产物清单

本章程 + 实现 commit sha + launch 配置（7 cmd × 臂）+ τ(v) manifest
（7 份材料哈希 + 审计记录）+ D1/D2/D3 报告（含 D3 逐点对照表 + τ material
conformance 对照）+ Run 行（开跑后入 `tracker/TO.md` TO41xx）。

## 8. owner freeze review（三十三轮，2026-09-02）：G↓ 结案 + D 阶段授权 + 解释纪律

**裁定**（记录全文 = `TO41_G_DOWN_SPEC.md` §10）：G↓ = CLOSED / PASS；
material availability **7/7 COMPLETE**；determinism **PASS**；**不为
0.277/0.300/0.325 重新求解**；**批准进入 D1/D2/D3**；**主 Rung 1
experimental compute 仍不批准**——待 D1/D2/D3 全 PASS + execution
freeze。设计审查到此收益已低：D 三层全 PASS 后即做 execution freeze，
然后真正开跑 Rung 1。

**解释纪律（freeze 前落盘，D/Rung 1 全阶段有效）**：

> **target_speed defines treatment identity; v_realized records material
> realization quality. Passing the pre-registered realization tolerance
> does not convert the realized speed into a new treatment level.**

操作化三条：

1. 主分析 treatment identity = `target_speed`（冻结 grid
   {0.200, 0.225, 0.250, 0.275, 0.277, 0.300, 0.325} 不动）；
   `v_realized`（0.2000 / 0.2250 / 0.2500 / 0.2750 / 0.2768 / 0.2925 /
   0.3179）= material realization diagnostic / covariate。
2. **禁止**把 target 改标 realized（如 0.300 → 0.2925，破坏已冻结
   treatment grid）；**禁止**分析阶段把 `v_realized` 当连续 treatment
   重跑主分析——那会把 treatment assignment 从 **pre-registered
   target** 换成 **post-generation realized variable**，重新引入
   endogenous treatment（可做诊断，不可替代 primary treatment
   definition）。
3. 容差门⑨使材料合法，但不使 τ(v_material) 与 τ(v_target) 数学同一：
   实际观测严格是 Y(v_target, C, τ(v_material))，不自动等于
   Y(v_target, C, τ(v_target))，主结果解释时保持区分。

**预保存诊断量（telemetry/log 必须可机械取出；诊断用途，不改 estimand、
不加 stopping rule）**：`target_speed` / `material_realized_speed` /
`abs_error` / `material_hash` / `source_lineage` / `decoder_condition` /
`tau_ff`——事后检查 effect 是否随 material realization error 系统变化。
已知 realization error 非均匀（低速 1e-08 量级 vs 0.300/0.325 的
7.5e-03/7.1e-03），若主效应与该梯度共变，解释时必须引用本诊断。

**28-cell runtime launch table**：见 §4 三十三轮增补——7 v × {C1,C2} ×
{τ ON,τ OFF}，per-cell `τ_hash`，验证 7 个 target speed 真正 launch 且
same v → same τ identity。

**0.2667 分裂带降级**：仅作 post-hoc consistency observation（历史诊断
信息）；不进入 target definition / source selection / material
generation / cell assignment / stopping rule（详见 G_DOWN_SPEC §10）。

**D 三层工作对象（本轮点名）**：

- **D1 decoder invariance**：C1/C2 只改 condition selection；VAE
  weights / checkpoint / input transform / normalization semantics /
  latent dimension 全不变。
- **D2 treatment consistency**：∀v ∈ 7-grid，C1 + same τ(v) 与 C2 +
  same τ(v) 同时成立——**逐 cell 验证 C1/C2 的 τ material hash 完全
  相同**（Mode A 最关键的新检查）。
- **D3 artifact/runtime conformance**：同时证明 T_runtime(v,C) =
  T_mapping(v,C) **且** τ_runtime(v) = τ_frozen(v)——否则 mapping 虽
  正确，material plumbing 仍可能把 Mode A 偷换回 Mode B。

**项目语义迁移（本轮定性）**：问题已从「我们能不能生成 τ(v)?」（G↓ 回答：
能，CLOSED）转为 **「冻结 decoder 后，在严格控制 τ(v) material identity
的情况下，C1/C2 是否改变 τ_ff effect?」**——这才是 B / Mode A 真正要回答
的问题。
