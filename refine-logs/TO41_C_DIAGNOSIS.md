# TO41 Rung 1 (c) 诊断：56-cell 现有产物的 variance / natural-vs-interventional / comparability 分解

> 【层位 L2 专题分析（判读诊断）】↑ `refine-logs/README.md`（扇出树根地图）｜
> 上游：`tracker/TO.md` 四十轮三连判读行（数据唯一事实源）＋ 四十一轮 owner
> 裁定（2026-09-03：选 (c) 先补维度再判，不新增训练/speed/protocol，第三
> 训练 seed 不授权）｜ 下游：owner 对分支 (b) 的最终裁定 → 收束报告或
> 第三 seed 授权重述 ｜ 状态：**活跃**（诊断完成，待 owner 裁定分支）
>
> 执行产物：`apt_g1/rung1/eval_diagnose.py`（SCRIPT_MAP §8c 已登记）→
> `apt_g1/outputs/sync/to41_eval/diagnosis_v1.{json,txt}`。

## 0. 本轮裁定与措辞纪律

owner 四十一轮裁定要点（全文见会话记录，tracker/TO.md 有 Run 行）：

1. **选 (c)**：先做现有 56-cell 数据的 variance + natural/interventions +
   C1/C2 comparability 诊断；**不新增训练，不新增 speed，不修改 protocol**。
2. **第三训练 seed 暂不授权**——当前更基础的问题是「正在比较的两个
   condition 到底是不是可解释的两个 treatment regimes」。
3. 裁定特别质疑：**3 eval seeds 近似逐位相同**可能意味着 eval randomness
   根本没有进入关键执行路径，第一优先级是查 eval seed identity 是否真的
   改变 rollout 的随机源/初始状态。
4. 措辞纪律修正（owner 原话要求）：不再说「执行层面无可挑剔——下面的读数
   可以放心当作系统真实行为」；改为 **「execution conformance 已通过
   （G1–G10 全 PASS）；因此这些读数可信地反映当前 frozen system 的实际
   行为」**——「实际行为」是事实，**不因此已有科学解释**。本文全篇遵守
   该口径。

分叉树（owner 预注册）：C1/C2 可比 + train-seed variance 主导 → (a) 第三
seed；C1/C2 明显不可比 → (b) 收束为 condition-specific contrast；其他结构
→ 重新解释但不擅改 protocol。

## 1. 数据与口径（先说清能算什么、不能算什么）

- 输入 = 已入仓产物：56 份 formal receipt
  （`sync/to41_eval/receipts/`）＋ `effect_table_v1.txt`（28 行 = 7 v ×
  2 C × 2 train seed）。**零新增执行、零新增数据**。
- 主指标口径：err60s = per-step |vx−cmd| 的 3-eval-seed 均值，由服务器端
  原始 rollout 计算；**per-step 原始行未入仓**，receipt 只存 vx_mean 等
  聚合字段。两者**不可互推**（例：v0.2 C1 off s0，|vx_mean−cmd|=0.119 vs
  err60s=0.028——蠕行格速度双向波动，聚合口径系统性偏大）。故：
  - **V 块（方差分解）全部取自 effect_table_v1 本身**：其 per-seed pairs
    列就是逐 eval-seed 的配对差 ON−OFF，正是主指标的原生粒度，无需重构。
  - **S/D 块用 receipt 聚合字段时仅作形态/几何证据**，不冒充主指标。
- eval seed 的操作身份（receipt `eval_seed_note` ＋ 代码）：
  `jitter_and_reset(env, seed=eval_seed)`，`rng(1000+seed)`。

## 2. V 块：方差分解——train-seed 方差压倒性主导

| 量 | 中位 | 备注 |
|---|---|---|
| eval-seed sd（同 cell 3 个配对差，df=2） | 0.0018（max 0.0157） | 56 cell |
| eval-seed 极差 | 0.0035（max 0.0310） | 56 cell |
| train-seed \|Δdff(s0−s1)\| | **0.0518**（min 0.031, max 0.110） | 28 (v,C) |
| 超 0.02 决策边界的 train-seed 差 | 14/28 | — |
| F_like = \|Δdff\|²/(2σ²_eval/3) | **133**（min 43, max 547） | df≈(1,2) 仅作量级参照 |
| 若无 train 效应时 \|Δdff\| 的噪声期望 | 0.0047 | 实测中位 0.0518 ≈ 11× |

**结论**：Δ_ff 的不确定性几乎全部来自训练 seed（最保守的单格比值也有
43×）。这与四十轮肉眼判读一致，现在有了量化版本。

## 3. S 块：eval seed identity 审计——「强镇定」而非「随机未生效」

**代码事实**（`eval_apt_isaac.py:101` / `eval_cell.py:158-171`）：

- seed 唯一入口 = `np.random.default_rng(1000+seed)`，驱动三处初始扰动：
  root z ±0.005 m、29 个 body joint pos ±0.01 rad、全关节速度 ±0.02 rad/s；
  obs history 从扰动后状态 refill；router state reset。
- policy 推理 `deterministic=True`（无采样噪声）；`disturbance_prob=0`，
  cmd 恒定每步重申——eval 随机性**仅** = 初始条件扰动。
- 既有 harness 语义：jitter 后不刷新 `_last_obs`，首步沿用上轮末 obs
  （三 seed 一致继承，不影响有效性，但意味着随机性从 step 2 起才进入观测）。

**数据事实**：56/56 格的 3 个 eval seed 读数**均非逐位相同**（逐位全同格数
= 0）。spread 中位：vx_mean 0.0019、h_min 0.0095、disp 0.41 m。

**结论**：owner 的质疑被数据否定——随机性确实进入了执行路径（若是死 RNG
应观察到 56/56 逐位全同）。真实图景是**系统对初始扰动强镇定**：60 s 内
deterministic 策略把 1e-2 量级的初始差异衰减到 1e-3 量级的行为差异。
「eval 噪声可忽略」这一表述继续成立，且现在知道**为什么**。附带一提：
disp 的相对波动并不小（蠕行格 ~0.5 m 路径上 spread 0.4 m），说明终点位置
对初始条件敏感，但主指标（速度误差幅度）被强镇定。

## 4. N 块：natural vs interventional——Δ_cond 是双段拼接的 estimand

receipt 的 `condition_override` 字段逐格记录了自然 assignment 与实际
decode 输入。56/56 格验证，且与 `bucketize` 复算（edges =
linspace(0, 0.8, 4)[1:-1] = [0.267, 0.533]）全符：

| 速度段 | natural decode 输入 | C1 臂 | C2 臂 |
|---|---|---|---|
| v < 0.267（0.200/0.225/0.250） | vb0 | **natural no-op**（changed 0/9000） | forced vb0→vb1（9000/9000） |
| v ≥ 0.267（0.275/0.277/0.300/0.325） | vb1 | forced vb1→vb0（9000/9000） | **natural no-op**（changed 0/9000） |

三个直接后果：

1. **每个速度点上 C1/C2 恰好一臂 natural、一臂跨 bin 强制**。「C1 =
   natural 臂」的说法只在低 v 段成立。C1 臂全程 decode 输入恒为 vb0、
   C2 臂恒为 vb1（natural 或强制）——**C1 ≡ vb0-regime，C2 ≡
   vb1-regime**（全 v 轴），matched-ness 是 (v,C) 的派生属性。
2. **Δ_cond(v) = Δ_ff(v,C1) − Δ_ff(v,C2) 在 bin 边界两侧 contrast 内容
   互换**：低 v 段 = (matched − forced)，高 v 段 = (forced − matched)。
   数值沿 v 平滑（四十轮已记），但**整条 v 轴不是一个 homogeneous
   estimand**——这是对「Δ_cond 大且结构化」的重要降级。
3. **τ_ff 效应的调制主要由 decode-regime × seed 承载，matched-ness 调制
   ≈ 0**。用 effect_table 逐格均值验证（m/s）：

   | 分组（7 格均值） | s0 | s1 |
   |---|---|---|
   | matched 格的 Δ_ff | +0.042 | +0.019 |
   | mismatched 格的 Δ_ff | +0.041 | +0.002 |
   | **matched−mismatched** | **+0.001** | **+0.017** |
   | C1 (vb0-regime) 的 Δ_ff | +0.075 | −0.022 |
   | C2 (vb1-regime) 的 Δ_ff | +0.008 | +0.042 |
   | **regime 差（C1−C2）** | **+0.068** | **−0.064** |

   regime 差是 matched-ness 差的 4–50 倍且**符号随训练 seed 翻转**；
   matched-ness 差本身弱且 seed 不稳。即：Δ_ff 的 seed 依赖结构 =
   「τ_ff 有害/有益取决于 decode 输入是 vb0 还是 vb1，而哪个 regime 更
   敏感随训练 seed 换向」，与该 regime 是否为自然 assignment 无关。
   （以上为观察性归纳，非机制结论；PASS-AS-CHANNEL 纪律继续有效。）

natural vs interventional 的正面回答：C2 在低 v 段是真正的 off-natural-
support 干预（natural 下 vb1 在低 v 从不出现），但 vb1-regime 在高 v 段有
natural 观测——两个 decode regime 都在自然日内出现过，**只是在同一
commanded speed 下永远只有一个 regime 是自然的**。Δ_cond 因此永远混淆
「decode regime 差异」与「speed-bin 语义错配」，除非引入新的设计维度
（本轮明确不做）。

## 5. D 块：C1/C2 comparability——7/7 速度点 OFF 基线分离

treatment-free 层（τ_ff OFF 臂）的 support 诊断（err60s 跨 train-seed
区间；几何字段为 receipt 聚合，s,e 级 range）：

| v | C1_off err | C2_off err | 间隙 | C2/C1 比 | vx_mean C1 / C2 | disp C1 / C2 (m) |
|---|---|---|---|---|---|---|
| 0.200 | [0.028, 0.043] | [0.382, 0.386] | 0.339 | 10.8× | [0.08,0.17] / [0.59,0.63] | [0.1,8.3] / [30.8,31.8] |
| 0.225 | [0.053, 0.065] | [0.359, 0.362] | 0.294 | 6.1× | [0.09,0.19] / [0.59,0.63] | [0.5,9.5] / [30.9,31.6] |
| 0.250 | [0.080, 0.090] | [0.333, 0.339] | 0.243 | 4.0× | [0.08,0.18] / [0.59,0.63] | [0.6,10.2] / [30.9,31.3] |
| 0.275 | [0.104, 0.115] | [0.308, 0.313] | 0.193 | 2.8× | [0.08,0.19] / [0.59,0.63] | [0.4,9.4] / [30.9,31.5] |
| 0.277 | [0.109, 0.119] | [0.306, 0.309] | 0.187 | 2.7× | [0.08,0.18] / [0.59,0.63] | [0.4,9.2] / [30.7,32.0] |
| 0.300 | [0.138, 0.150] | [0.275, 0.283] | 0.125 | 1.9× | [0.09,0.19] / [0.59,0.63] | [0.4,10.4] / [30.9,31.3] |
| 0.325 | [0.198, 0.223] | [0.263, 0.270] | 0.040 | 1.3× | [0.08,0.18] / [0.59,0.63] | [0.4,9.8] / [30.8,31.2] |

- **7/7 速度点 err 区间分离**（判定规则：跨 seed 区间无重叠且间隙 >
  0.02；诊断性规则，非协议硬点）。最低间隙在 v=0.325 处（0.040），仍超
  决策边界 2 倍。
- 几何图景一目了然：**两 condition 的步态速度由 decode regime 决定而几乎
  不由 cmd 决定**——C2 OFF 全速 ~0.61 m/s 前进（disp ~31 m/60s），
  C1 OFF 蠕行 ~0.13 m/s（disp 数米以内）；h_min 两臂均 ~0.71–0.76 m
  （零倒地，与 audit 一致）。err 随 v 的收敛趋势只是 cmd 向 C2 固有速度
  靠近的几何效应。
- **判定：C1 与 C2 在 OFF 层不可比**——不存在可用于 pooled interaction
  contrast 的共同 support。Δ_cond 只能读作 condition-specific contrast
  （与四十轮判读的谨慎表述一致，现在有了 support 证据）。

## 6. W 块：综合裁决与分叉

| 问题 | 数据回答 |
|---|---|
| eval variance ≪ train variance？ | **是**（sd 中位 0.0018 vs 0.052；F_like 中位 133） |
| eval seed 是否真的进入随机路径？ | **是**（0/56 逐位全同；jitter 三处初始扰动 + deterministic 推理）——「可忽略」的正确表述是强镇定 |
| C1/C2 可比（OFF 共同 support）？ | **否**（7/7 分离，最低间隙 0.040 > 0.02） |
| Δ_cond estimand 身份 | bin 边界两侧 contrast 内容互换；regime(vb0/vb1)×seed 是主结构，matched-ness 调制 ≈ 0 |

**按 owner 预注册分叉树 → 落在分支 (b)：收束为 condition-specific
contrast。** C1/C2 明显不可比时，第三训练 seed 无法解决 C1 vs C2 的解释
问题（第三 seed 改善的是 Δ_ff 的 seed 稳定性估计，而不可比性是设计维度
的属性，与 seed 数无关）。

**若 owner 仍考虑 (a)**，本诊断把第三 seed 的信息增益重述为：检验
「decode-regime(vb0/vb1) × τ_ff 交互符号」的 seed 稳定性（当前 s0/s1 恰
好换向），而**不是**笼统的「Δ_ff 是否稳定」；且其结论只能落在各 regime
内部，永远升格不到跨 condition 的 interaction 机制结论。

三不做重申（owner 裁定）：不把 s0/s1 符号翻转解释成「随机」（现在知道它
结构化为 regime×seed 交互，但机制仍未知）；checkpoint selection 保持冻结；
不按本轮效应曲线增加 speed points（anti-selection 锁死）。

## 7. 产物与本轮零新增执行声明

- 分析脚本：`apt_g1/rung1/eval_diagnose.py`（纯 stdlib，本机可跑；输入 =
  receipts + effect_table，输出 = diagnosis_v1）。
- 诊断产物：`apt_g1/outputs/sync/to41_eval/diagnosis_v1.json` /
  `diagnosis_v1.txt`（本次入仓）。
- 零新增训练、零新增评测、零 protocol 变更；云端算力本轮消耗 = 0。
- Run 行已追加 `tracker/TO.md`（TO41 系列连续编号）。
