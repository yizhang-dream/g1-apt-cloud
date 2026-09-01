# TO41 Rung 1 implementation 章程：运行时实现与 integrity test

> 【层位 L3 实施章程】↑ `refine-logs/README.md`（扇出树根地图）｜
> 上游：`TO40C_PLAN.md` §10（**frozen specification**，规格侧一切判据以该节
> 为准，本文不得与其冲突）｜ 状态：**Implementation STARTED（八轮授权），
> Compute BLOCKED**（D + execution freeze 全过前不得开跑）。

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

## 3. incompatibility report #1（owner 决策项；未决前条件化臂不得实现）

冻结规格定义 Δ_cond = Y(τ OFF, 条件化) − Y(τ OFF, 非条件化(0.6 步态混合))，
但「条件化」的运行时实现在冻结源与 mapping artifact 下存在**非等价读法**：

- **读法 (a)：τ 参考按 decoder condition 索引**——per-condition τ LUT
  （slow/mid 各一条参考，target_speed→condition 的 YAML 决定用哪条）。
  含义：需要**新的 τ 材料**（现仅 0.277 单速表；TO40a 插值 FAIL 即此数据
  缺口，正是 Rung 1 立项动因）。物料生成方式（dircol 逐 bin 解？各 bin 内
  代表速度取值？）= treatment 物料决策 + 服务器 dircol 算力（CPU 单解
  40–60 min 量级）。
- **读法 (b)：decoder condition 钉扎/覆写**——按 mapping 强制 decode 条件。
  但当前 mapping 是**自然 bucketize 函数的物化（恒等映射）**，钉扎 = 基线，
  Δ_cond 恒 0，非可行设计；若 owner 本意是覆写为「速度匹配 gait 条件」，
  则 mapping artifact 需为非恒等 → 触发 **mapping/material reopen**。

**按 §10.8 implementation 纪律上报，stop 于本项**：owner 需决策
（读法选择；若 (a) 需一并给出 per-condition τ 材料的生成规格与其算力
授权）。未决期间实现仅推进无歧义部分（§2 可直接实例化项的配置准备与
D harness 骨架），不写条件化臂代码。

## 4. D dry-run 设计（execution integrity test，非小型实验）

- 最小样本：7 target speeds × τ_ff ON/OFF × 最小 episode/seed；
  **效应估计为零目的**，dry-run 结果不得用于调整 treatment。
- **D1**：四哈希对冻结基线 + decoder state_dict 前后全等（conditioning
  只选择、不修改参数）。
- **D2**：两臂同 target_speed 的 decoder condition 逐项一致。
- **D3**：运行环境冻结实现对 7 cmd 求值 vs YAML——**逐点对照表**入产物
  （`0.200 yaml=vb0_db4 runtime=vb0_db4 PASS` / …），不只报总 PASS。
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
