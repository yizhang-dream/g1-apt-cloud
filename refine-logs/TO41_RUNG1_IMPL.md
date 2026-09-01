# TO41 Rung 1 implementation 章程：运行时实现与 integrity test

> 【层位 L3 实施章程】↑ `refine-logs/README.md`（扇出树根地图）｜
> 上游：`TO40C_PLAN.md` §10（**frozen specification**，规格侧一切判据以该节
> 为准，本文不得与其冲突）｜ 状态：**Implementation STARTED（八轮授权），
> Compute BLOCKED**（D + execution freeze 全过前不得开跑，并覆盖
> per-condition τ material generation——Decision 3）。分支状态：
> Unambiguous parts PROCEEDING（仅章程/harness 骨架）｜ Conditioning
> impl **BLOCKED** ｜ **Incompatibility #1 OPEN**（九轮细化为
> Decision 1–3，见 §3；裁定权在 owner）。

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

## 3. incompatibility report #1（OPEN；九轮细化——上游 fork 先于八轮的 (a)/(b)）

冻结规格定义 Δ_cond = Y(τ OFF, 条件化) − Y(τ OFF, 非条件化(0.6 步态混合))，
但其 runtime semantics 取决于一个更上游的语义 fork——**conditioning
identity**。八轮的 (a)/(b) 是本 fork 的下游子问题，owner 决策压缩为三项：

### Decision 1 — conditioning identity（一句话 fork）

> Rung 1 要识别的是「**自然 decoder condition 下** τ_ff 的 effect」，
> 还是「**人为指定 decoder condition 后** τ_ff 的 effect」？

- **读法 A（observational / 分层变量）**：C = C(v) 由冻结 bucketize 确定，
  估计 Y(v, τ_ff, C(v))。此时**不存在 conditioning intervention**——§10.3 的
  Δ_cond / DiD 没有 treatment contrast，estimand 集收缩为 Δ_ff(v) 族。
  这是**规格变更**：须 owner 确认后按 reopen 程序修订 §10.3，本文不得代改。
- **读法 B（interventional treatment）**：C 可干预（force C = slow / mid）→
  需要非恒等的 condition assignment → 现行 mapping artifact（自然函数的
  恒等物化）不敷使用 → **mapping reopen**。八轮读法 (b)（钉扎）仅在此
  读法下非退化。
- 八轮读法 (a)（per-condition τ LUT）在两种读法下均可成为 τ_ff 的材料
  形态，但它**不回答 Decision 1**。

### Decision 2 — τ material rule（仅当 Decision 1 = B）

每个 decoder condition 的 τ reference 必须由**预注册、确定性的代表点规则**
生成：`τ_c = G(v_c)`（condition → representative speed → dircol → τ
material），禁止人工挑「看起来合适」的 τ。两条身份链分离：
**condition identity 来自冻结 binning 函数；τ reference identity 来自单独
冻结的 material-generation rule**——不得做 bin = gait、bin → τ 的隐含推论
（slow bin ≠ slow gait 是未经检验的语义跃迁）。Δ_cond 措辞保持中性：
其观测差异可能含 τ-reference construction difference，不得写回
「decoder mismatch effect」。

### Decision 3 — compute 授权边界

**Compute BLOCKED 继续覆盖 per-condition τ material generation**：
representative speed 未冻结前跑 dircol，不是「准备实验」，而是执行尚未
批准的设计选择。Decision 1/2 冻结后，material-generation compute **单独
授权**，且即使获授权也仍不是 Rung 1 experimental compute。

**纪律重申**：未决期间不写任何猜测性 conditioning code；实现仅限无歧义
部分（章程/harness 骨架）。现有 0.277 单速 LUT **不得**硬套七点
（τ(0.20)=τ(0.277)=τ(0.325) 再声称已 conditioning = 最危险的工程冲动）。

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
