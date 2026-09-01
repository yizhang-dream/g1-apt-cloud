# TO41-G↓ spec：低速 downward-continuation material 子战役

> 【层位：L3 实施层·独立实验身份】↑ `refine-logs/README.md`（扇出树根地图）｜
> 上游：`TO41_RUNG1_IMPL.md` §5.9（二十轮裁定 ③，本 spec 是其落点）｜
> 状态：**P↓-S APPROVED（三十轮）；R_valid↓ / schedule rule / manifest 待
> 冻结；无任何 material-generation compute 授权，Main Rung 1 compute
> BLOCKED 不变**。

## 0. 身份与边界

- 本战役回答的唯一问题：**是否存在机械有效、可复现的 downward
  continuation，使 v = {0.200, 0.225, 0.250, 0.275} 获得合法 τ(v)
  material？**
- **独立实验身份**：source selection / P↓ / continuation schedule /
  solver / acceptance / infeasibility / canonical rule 全部独立定义，
  **不借用主 campaign 的任何隐含常识**（尤其不继承其 8-start budget、
  不默认其 projection 语义、不直接复用其 source-selection rule）。
- 可能结局（两种都是有效结果，预注册）：
  - 全成功 → 解决主 campaign 最大 coverage obstacle；
  - 部分/全失败 → **material-generation support boundary**（如
    `0.275 valid / 0.250 valid / 0.225 unavailable / 0.200 unavailable`），
    成为主 Rung 1 的先验输入事实，**不是失败实验**。
- 材料 identity = τ(v)（requested v）；P↓ 只改求解路径，不改变 identity。

## 1. Decision 1 —— P↓（三十轮裁定：P↓-S 严格版）

**裁定**：`X_guess = X_source`（**source state 原样保留**——不做轨迹/
相位/时间缩放，无人为重构低速轨迹）；P↓-T / P↓-R 均不选。理由 =
信息增益最高：把问题干净定义为「冻结 source state 不变，通过预注册
downward continuation schedule 检验已有解支能否向目标速度延拓」。

**严格化（三十轮）**：**「仅改 schedule」必须不含人工自由度**——`v_min =
v_target` 以及 `w-time` 等**所有** schedule 参数均由**冻结、确定性规则
从 v_source 与 v_target 计算**；smoke 中禁止任何人工调 schedule（发现
"时间尺度不好"→ 不允许微调，属 reopen）。**解释边界**：P↓-S 成功 ≠
同伦动力学数学可逆（只说「给定 source/solver/schedule 存在一条成功的
downward continuation path」）；失败 ≠ 「不存在低速解」（只说「该预注册
downward procedure 未生成 material」）。

## 1a. deterministic schedule rule【提案待批】

- `v_min(v) = v_target`（末段同伦 v_min 直接取目标速度）。
- `w-time(v) = w_time0 · (v_source / v_target)`（从冻结基值 w_time0 =
  0.0 派生——当前管线默认 w_time=0，且 F 线按 `--w-step` 拉速度而不靠
  w-time；本规则保 P↓-S 状态不变，仅速度定位由 v_min 承担）。
- knobs（knots / t_max / max_iter / retries）：沿用主 pipeline 的
  P↓-S-compatible 冷/热参面，**每个参数有 freeze 来源**（末段 v_min 由
  v_target 计算；节点/时长/迭代上限从 ↓-ladder 固定表读取，见 §3）。
- **禁 smoke 调参**：↓-smoke 只验证 schedule 被接受与抵达 target stage，
  不得据 smoke 结果修改任意 schedule 参数（违者 reopen，不当"工程便利"）。

## 2. source selection（R_valid↓；三十轮：静态定义，非 smoke 产物）

- `s↓(v) = argmin{ s ∈ R_valid↓ : v_dump(s) ≥ v } v_dump(s)`，tie-break =
  冻结 registry 的 **canonical artifact-ID 序**（不用 commit 新旧）。
- **R_valid↓（冻结数据，三十轮实测）**：to37_v016（v_dump=0.2764，
  mode=foot/knots=40，guess schema 齐全 v_aux 6/lam 4/XM 40×12 两组、
  无 NaN、drift (1.303,1.333) < 2.0、ke_drop OK、sha16 50d95baf）——
  **静态定义**（schema/model/provenance/material-valid/source-generation
  identity 全过），**非 smoke 产物**（不得"先试 downward 成功才宣布
  valid↓"）；F11b_flat（0.2768，同族）为次选。
- 按当前 registry：four targets s↓ 均为 **to37_v016 (0.2764)**（argmin_{≥v}；
  F11b_flat 0.2768 次选）——**预期值基于冻结实测，非人工挑**。
- 每个 target **独立**从该 canonical source 生成（`G↓(v) =
  Solve(P↓(X_s), v)`），**禁止 chain**（0.225 不得喂 0.250 的结果——
  材料链式依赖、失败污染后续、provenance 混乱）。**parent/source 身份固定
  = source artifact hash**（"这次给 0.275 用的 source 副本"不改变身份）；
  生成的 child material 有独立 material identity（lineage = parent →
  projection → solver → child）。

## 3. continuation schedule 与预算（独立规定，不继承 7×8）

- 同主 campaign：decreasing effort 阶梯（per-target 每 start 一组
  knots/t_max/max_iter/retries），**子战役独立编号**（↓-k0..↓-kN）。
- `hard cap`：每 target ≤ **6** starts（【提案】）、全战役 ≤ **24**
  starts（【提案】；= 4 targets × 6）、服务器 **⊆ 8-way** 全局并发
  （同主 campaign 纪律：速度内严格串行，canonical = 预注册 order 首过
  即停，**非 wall-clock**）。
- `planning estimate`（非承诺）：单 start 40–90 min（下降向预计不慢于
  上升向），实际墙钟取决于提前停止。
- **不把 8-start 写成"合理默认"**——本战役的数值 regime（downward
  continuation）不同于主 campaign，预算独立、仅参考。

## 4. acceptance 两层（分离，不得混淆）

- **G↓-reachability acceptance**：solver 是否**成功到达 requested
  target-speed stage**（逐级 stage 抵达 + v_realized 可得）。
- **G↓-material validity acceptance**：candidate 是否通过全部
  机械/物理/字段/速度（|v_realized − v| ≤ 0.02，沿用⑨但独立编号
  ⑨↓）/复现门（G1/G2/G345/G6/G7 全门集，driver 复用但独立命名）。
- **禁止**：未到 target 直接记为 mechanically_invalid（两 failure mode
  不同）——reachability FAIL 认"operator/schedule 失败"；
  material validity FAIL 认"已到 target 但机械不过"。
- 完整结果对照表（per target：eligible source → selected → P↓ 选项 →
  stages reached → v_realized → gates PASS/FAIL 明细）随 material
  availability map 输出。

## 5. infeasibility 与报告

- 某 target 预算耗尽仍无可过 validity 门解 → **↓-material-unavailable**
  → 该 target 的 cells 判 invalid（不插值/不跳/不换源/不用"最近的下游"
  fallback——本战役的 source 级 fallback 同样禁）。
- 全部失败（或部分）→ **material-generation support boundary** 入最终
  availability map；措辞纪律沿用主 campaign：boundary ≠ τ_ff 低速无效应
  （工程 reachability 不被读成机制 null）。
- 结果两种都算成功结案：全过 = coverage 解决；边界 = 先验输入事实。

## 6. 执行序与 gate

```
Decision 1 (P↓) owner 裁定 → P↓-S 则：
  compatibility 核对 R_valid↓ → source-selection audit table
  → manifest（G↓ 独立）→ ↓-smoke (source→最近 target, 0.275,
    判据 = 推过 downward stage barrier 并抵达 v=0.275)
  → per-target 独立生成（4 分支，无 chain）→ 两层 acceptance
  → material availability map → 主 Rung 1 预注册输入
```

↓-smoke 失败 → **reopen Decision 1**（评估 P↓-T/-R），不 pilot 不调参；
本 spec 不授权任何实现代码（driver/launcher）——只有裁定后可写。

## 7. 产物清单

本 spec + P↓ 裁定 commit + R_valid↓ 核对表 + G↓ manifest + ↓-smoke 报告
+ 4-target 独立生成 + 两层 acceptance 明细 + **material availability
map**（主 Rung 1 的预注册输入）。
