# TO41 Rung 1 implementation 章程：运行时实现与 integrity test

> 【层位 L3 实施章程】↑ `refine-logs/README.md`（扇出树根地图）｜
> 上游：`TO40C_PLAN.md` §10（**frozen specification**，规格侧一切判据以该节
> 为准，本文不得与其冲突）｜ 状态：**Implementation STARTED（八轮授权），
> Compute BLOCKED**（D + execution freeze 全过前不得开跑，并覆盖
> per-condition τ material generation——Decision 3）。分支状态：
> Unambiguous parts PROCEEDING（仅章程/harness 骨架）｜ Conditioning
> impl **BLOCKED** ｜ **Incompatibility #1 OPEN**（十轮定稿为单一科学
> 问题 + A/B 两条后果链，见 §3；裁定权在 owner）｜ Estimands
> **LOCKED\*** ｜ **Owner decision REQUIRED**。
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

## 4. D dry-run 设计（execution integrity test，非小型实验）

- 最小样本：7 target speeds × τ_ff ON/OFF × 最小 episode/seed；
  **效应估计为零目的**，dry-run 结果不得用于调整 treatment。
- **D1**：四哈希对冻结基线 + decoder state_dict 前后全等（conditioning
  只选择、不修改参数）。
- **D2**：两臂同 target_speed 的 decoder condition 逐项一致。
- **D3**：运行环境冻结实现对 7 cmd 求值 vs YAML——**逐点对照表**入产物
  （`0.200 yaml=vb0_db4 runtime=vb0_db4 PASS` / …），不只报总 PASS。
  **D3 范围限制（九轮）**：只证明 decoder condition assignment 正确；
  若 Decision 2 落地，τ material 另需自己的 **material conformance check**
  （`τ_runtime(C) == τ_frozen-material(C)` 逐项对照），不得并入 D3 解释。
- D 全 PASS → owner freeze（纯状态迁移 commit，只改 freeze_status）→
  EXECUTION-READY → compute。

## 5. implementation audit 八问（每步自检）

1 YAML 是否被原样消费？ 2 7 个 cmd 是否全部 launch？ 3 τ_ff ON/OFF 是否
只改变预注册 treatment？ 4 decoder 是否保持 frozen？ 5 assignment 是否
一致？ 6 outcome logging 是否完整？ 7 seed pairing 是否保持？ 8 failure
是否按预注册规则处理？

## 6. 产物清单

本章程 + 实现 commit sha + launch 配置（7 cmd × 臂）+ D1/D2/D3 报告
（含 D3 逐点对照表）+ Run 行（开跑后入 `tracker/TO.md` TO41xx）。
