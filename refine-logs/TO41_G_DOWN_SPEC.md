# TO41-G↓ spec：低速 downward-continuation material 子战役

> 【层位：L3 实施层·独立实验身份】↑ `refine-logs/README.md`（扇出树根地图）｜
> 上游：`TO41_RUNG1_IMPL.md` §5.9（二十轮裁定 ③，本 spec 是其落点）｜
> 状态：**DRAFT — 待 owner 裁定 Decision 1（P↓）；未获裁前无任何
> material-generation compute 授权，Main Rung 1 compute BLOCKED 不变**。

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

## 1. Decision 1 —— P↓ 的数学身份（唯一待裁上游对象）

`X_guess = P↓(X_source)`，三种 operator 语义（**互斥三选一；禁止
"先试试哪个投影最好"的分叉 pilot**——失败即 reopen，不 pilot）：

| 选项 | 定义 | 论证 | 风险 |
|---|---|---|---|
| **P↓-S（推荐）** State-preserving parameter shift | `X_guess = X_source`（完整解 dump 原样），仅改 continuation schedule：`--stages` 末段 `v_min = v_target`（+ w-time 按目标速度调整）。solver 在同伦中把解从 v_source 拉向 v_target | 与主 campaign 已验证可行的 hot-start **完全同构**（仅方向相反：F9/F11 本身就是同机制产物）；**零新近似自由度**；失败归因最干净（同伦不可逆性 → support boundary 证据，或 schedule 问题） | 依赖同伦链路可逆（未验证；正是本 spec 要测的） |
| P↓-T | 轨迹时间/空间缩放：相位按 T_target/T_source 缩放、step 按 v_target/v_source 缩放 | 直观 | 破坏物理约束（限位/足端高度/冲击残差）；引入近似回调自由度，验收框变脏 |
| P↓-R | 丢弃轨迹、仅用源解结构参数（周期数/支持拓扑）重建低速初猜再交 solver | 结构先验 | 与已失败的冷启动近邻问题高度重叠；信息增益低 |

**理由**：选 P↓-S 的理由不取决于"它大概能行"，而取决于它是唯一不引入
新近似自由度的 operator；同伦可逆性这一前提本身正是 sub-campaign 要
检验的问题。owner 裁定 P↓ 后，本 spec 其余部分按所选 operator 细化
（P↓-T/-R 的 schedule/acceptance 含义届时重写 Decision 2+，属 reopen）。

## 2. source selection（指向 R_valid↓，不复制主 rule）

- `s↓(v) = argmin{ s ∈ R_valid↓ : v_dump(s) ≥ v } v_dump(s)`，tie-break =
  冻结 registry 的 **canonical artifact-ID 序**（不用 commit 新旧）。
- **R_valid↓ = { s : 主 registry 已验证 v_dump 实测 ∧ 具备
  downward-projection compatibility }**——compatibility 在 P↓ 裁定后单独
  核对（至少：guess schema 齐全 = v_aux/lam/XM、mode=knots 面与 solve
  的 `--stages` 参数面兼容、无 NaN）；数值上合格但无法稳定向下
  continuation 的材料 **不进 R_valid↓**。
- 按当前主 registry 实测（0.2764 最低已验证源），预期 four targets 的
  s↓ 均为 **to37_v016 (v_dump=0.2764)**（argmin_{≥v}；0.2768 F11b_flat
  次选）——**预期值标注 pending compatibility 核对**，非事实。
- 每个 target **独立**从该 canonical source 生成（`G↓(v) =
  Solve(P↓(X_s), v)`），**禁止 chain**（0.225 不得喂 0.250 的结果——
  τ(0.225) 依赖 τ(0.250) 会造成材料链式依赖、失败污染后续、provenance 混乱）。

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
