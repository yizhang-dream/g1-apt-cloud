# TO41 Rung 1 科学收束：regime-specific τ_ff contrast（分支 (b) ACCEPT）

> 【层位 L2 收束报告（Rung 1 关线）】↑ `refine-logs/README.md`（扇出树根地图）｜
> 上游：`tracker/TO.md` 四十/四十一轮判读行（数据唯一事实源）＋
> `TO41_C_DIAGNOSIS.md`（V/S/N/D 四块诊断）＋ 四十二轮 owner 终裁
> （2026-09-03：分支 (b) ACCEPT，六不做清单）｜ 下游：无（本线关闭；
> 第三训练 seed 降级为未来独立 robustness 实验，非本线延续）｜
> 状态：**收束**（Rung 1 关线；零新增执行，云端算力消耗 0）
>
> 数据产物：`apt_g1/outputs/sync/to41_eval/`（56 receipts + effect_table_v1 +
> eval_audit + ckpt_selection + diagnosis_v1）。主指标 = err60s（per-step
> |vx−cmd| 的 3 eval-seed 均值）；Δ_ff = err(ON)−err(OFF)，负 = ON 更好。

## 0. owner 终裁要点（四十二轮，2026-09-03）

1. **分支 (b) ACCEPT**：收束为 condition-stratified / regime-specific
   τ_ff effects。诊断落点（C1/C2 OFF 层 7/7 速度点不可比）被接受为
   Rung 1 的合法科学产出，不再是待修复的障碍。
2. **不做清单（全冻结）**：不加第三训练 seed；不加新 eval seed；不扩
   speed grid；不改 Mode A；不改 decoder binning；不重新求 τ；不重新
   设计 C1/C2。
3. **措辞纪律**：`Δ_cond = Δ_ff(v,C1) − Δ_ff(v,C2)` 保留计算与数据，
   但只作 descriptive contrast；**不得称 interaction effect / causal
   decoder-condition interaction**（共同 support 不成立）。主报告身份 =
   `Δ_ff(v,C1)` 与 `Δ_ff(v,C2)` 各自作为 regime 内 condition-specific
   contrast。
4. **核心重定位**：本 Rung 1 最有价值的产出不是「seed 方差大」，而是
   更上游的结构事实——**C1/C2 不是同一 locomotion support 上两个可交换
   的 decoder conditions，而是两个不同的 decode regimes**。

## 1. 术语与身份重构（conceptual cleanup）

诊断 N 块确立的身份（56/56 receipt 验证）：decode 输入由
`bucketize(cmd_v, edges=[0.267, 0.533])` 决定，每速度点恰好一臂 natural、
一臂跨 bin 强制。因此：

| 身份 | 正确表述 | 作废表述 |
|---|---|---|
| C1 | **treatment-assigned vb0 decode regime**（全程 decode 输入恒 vb0） | 「natural condition」（只在 v<0.267 段凑巧成立） |
| C2 | **treatment-assigned vb1 decode regime**（全程 decode 输入恒 vb1） | 「intervention condition」 |
| natural / matched | **(v, C) 的派生属性**：该 regime 在当前 cmd 下是否恰为自然 bucket | 固定实验臂身份 |

Δ_cond(v) 在 bin 边界两侧 contrast 内容互换（低 v 段 = matched−forced，
高 v 段 = forced−matched），整条 v 轴不是 homogeneous estimand——这使
pooled interaction 从 estimand 层面就不可辨识，与 seed 数无关。

## 2. 主结果：7 v × 2 C × 2 s_train regime-specific τ_ff contrast 曲线

PRIMARY 表（数据 = `effect_table_v1.txt` 逐字；err60s，m/s）：

| v | C (regime) | s_train | ON err | OFF err | Δ_ff | per-eval-seed pairs |
|---|---|---|---|---|---|---|
| 0.200 | C1 (vb0) | s0 | 0.117 | 0.028 | **+0.090** | +0.090/+0.089/+0.090 |
| 0.200 | C1 (vb0) | s1 | 0.035 | 0.043 | **−0.008** | −0.011/−0.002/−0.012 |
| 0.200 | C2 (vb1) | s0 | 0.390 | 0.382 | **+0.009** | +0.010/+0.007/+0.009 |
| 0.200 | C2 (vb1) | s1 | 0.429 | 0.386 | **+0.044** | +0.043/+0.045/+0.042 |
| 0.225 | C1 (vb0) | s0 | 0.138 | 0.053 | **+0.086** | +0.085/+0.085/+0.086 |
| 0.225 | C1 (vb0) | s1 | 0.047 | 0.065 | **−0.017** | −0.026/−0.008/−0.019 |
| 0.225 | C2 (vb1) | s0 | 0.366 | 0.359 | **+0.006** | +0.005/+0.007/+0.006 |
| 0.225 | C2 (vb1) | s1 | 0.405 | 0.362 | **+0.042** | +0.042/+0.045/+0.040 |
| 0.250 | C1 (vb0) | s0 | 0.165 | 0.080 | **+0.086** | +0.087/+0.089/+0.081 |
| 0.250 | C1 (vb0) | s1 | 0.080 | 0.090 | **−0.010** | −0.013/−0.024/+0.007 |
| 0.250 | C2 (vb1) | s0 | 0.340 | 0.333 | **+0.007** | +0.006/+0.006/+0.008 |
| 0.250 | C2 (vb1) | s1 | 0.381 | 0.339 | **+0.043** | +0.043/+0.042/+0.043 |
| 0.275 | C1 (vb0) | s0 | 0.193 | 0.104 | **+0.088** | +0.087/+0.090/+0.088 |
| 0.275 | C1 (vb0) | s1 | 0.100 | 0.115 | **−0.015** | −0.006/−0.014/−0.025 |
| 0.275 | C2 (vb1) | s0 | 0.316 | 0.308 | **+0.007** | +0.007/+0.009/+0.006 |
| 0.275 | C2 (vb1) | s1 | 0.355 | 0.313 | **+0.042** | +0.040/+0.041/+0.044 |
| 0.277 | C1 (vb0) | s0 | 0.191 | 0.109 | **+0.082** | +0.087/+0.077/+0.080 |
| 0.277 | C1 (vb0) | s1 | 0.098 | 0.119 | **−0.021** | −0.024/−0.006/−0.032 |
| 0.277 | C2 (vb1) | s0 | 0.313 | 0.306 | **+0.008** | +0.004/+0.005/+0.014 |
| 0.277 | C2 (vb1) | s1 | 0.353 | 0.309 | **+0.044** | +0.044/+0.044/+0.044 |
| 0.300 | C1 (vb0) | s0 | 0.215 | 0.138 | **+0.077** | +0.076/+0.078/+0.077 |
| 0.300 | C1 (vb0) | s1 | 0.117 | 0.150 | **−0.033** | −0.025/−0.043/−0.030 |
| 0.300 | C2 (vb1) | s0 | 0.290 | 0.275 | **+0.015** | +0.015/+0.016/+0.015 |
| 0.300 | C2 (vb1) | s1 | 0.330 | 0.283 | **+0.046** | +0.047/+0.045/+0.046 |
| 0.325 | C1 (vb0) | s0 | 0.241 | 0.223 | **+0.018** | +0.013/+0.022/+0.020 |
| 0.325 | C1 (vb0) | s1 | 0.149 | 0.198 | **−0.049** | −0.052/−0.039/−0.056 |
| 0.325 | C2 (vb1) | s0 | 0.265 | 0.263 | **+0.001** | +0.003/+0.001/+0.000 |
| 0.325 | C2 (vb1) | s1 | 0.305 | 0.270 | **+0.036** | +0.035/+0.038/+0.033 |

读法（三条，全部 seed 内自洽）：

- **C1 (vb0 regime)**：Δ_ff 符号随训练 seed 翻转——s0 = +0.018…+0.090
  （低速最大、随 v 单调衰减），s1 = −0.008…−0.049。τ_ff 在 vb0 regime
  内**无害/有益的判定完全取决于训练 seed**。
- **C2 (vb1 regime)**：两 seed 同号（+0.001…+0.046，ON 略差），s1 的
  效应幅度约为 s0 的 5 倍且随 v 平稳。τ_ff 在 vb1 regime 内一致地
  轻微有害。
- **per-eval-seed pairs 三点几乎恒定**（如 +0.090/+0.089/+0.090）——
  每格内部不确定性远小于跨 seed 结构差，整张表的主导变异轴是
  (regime × train seed)。

SECONDARY / DESCRIPTIVE（Δ_cond^raw(v) = Δ_ff(v,C1) − Δ_ff(v,C2)，保留
数据不升格）：s0 = +0.017…+0.081（全正），s1 = −0.052…−0.084（全负）。
仅可描述为「两个 regime 的 τ_ff contrast 数值不同且方向随 seed 换向」，
**不得解释为 pooled interaction**。

| 统计身份 | 量 |
|---|---|
| PRIMARY | regime-specific Δ_ff（上表 28 行） |
| SECONDARY / DESCRIPTIVE | Δ_cond^raw(v)（7 v × 2 seed） |
| NOT INTERPRETABLE AS | pooled causal decoder-condition interaction |

## 3. 配套五件（收束要求的完整曲线附件）

### 3a. eval-seed variance——可忽略，且已知为什么

同 cell 3 个配对差的 sd 中位 **0.0018**（max 0.0157）；极差中位 0.0035。
56/56 格三 eval seed 读数均非逐位相同（0 格逐位全同）——eval seed 确实
进入执行路径（`jitter_and_reset`，rng(1000+seed)：root z ±5mm / 29 joint
pos ±0.01rad / joint vel ±0.02rad/s），但 deterministic 策略对初始扰动
**强镇定**（60 s 内把 1e-2 量级初始差异衰减到 1e-3 量级行为差异）。
「eval 噪声可忽略」成立且有机制性说明；**再加 eval seed 的价值 = 低，
本问题关闭**。

### 3b. train-seed variance——压倒性主导

|train-seed Δdff(s0−s1)| 中位 **0.0518**（min 0.031, max 0.110），
14/28 格超 0.02 决策边界；F_like = |Δdff|²/(2σ²_eval/3) 中位 **133**
（min 43）；若无 train 效应时该统计量的噪声期望仅 0.0047。**Δ_ff 的
不确定性几乎全部来自训练 seed**（最保守单格也有 43×）。

### 3c. OFF baseline separation——7/7 速度点分离（不可比性证据）

treatment-free 层（τ OFF）跨 train-seed err 区间（详表 =
`TO41_C_DIAGNOSIS.md` §5；判定规则 = 区间无重叠且间隙 > 0.02）：

| v | C1_off err | C2_off err | 间隙 |
|---|---|---|---|
| 0.200 | [0.028, 0.043] | [0.382, 0.386] | 0.339 |
| 0.225 | [0.053, 0.065] | [0.359, 0.362] | 0.294 |
| 0.250 | [0.080, 0.090] | [0.333, 0.339] | 0.243 |
| 0.275 | [0.104, 0.115] | [0.308, 0.313] | 0.193 |
| 0.277 | [0.109, 0.119] | [0.306, 0.309] | 0.187 |
| 0.300 | [0.138, 0.150] | [0.275, 0.283] | 0.125 |
| 0.325 | [0.198, 0.223] | [0.263, 0.270] | **0.040**（最小，仍 2× 边界） |

7/7 分离 → **两 regime 无共同 support**，pooled interaction contrast
的预注册可比性前提不成立。

### 3d. target vs realized speed——步态速度由 decode regime 决定，几乎不由 cmd 决定

从 56 份 receipt `episodes[].vx_mean`（描述性几何口径，非主指标；蠕行格
双向波动使其与 err60s 不可互推）提取，OFF/ON 两臂各 3 eval-seed 范围：

| regime | OFF vx_mean (m/s) | ON vx_mean (m/s) | disp OFF |
|---|---|---|---|
| C1 (vb0) | **0.081–0.193**（蠕行；s0 ~0.085，s1 ~0.16–0.19） | 0.101–0.176 | 0.1–10.4 m |
| C2 (vb1) | **0.589–0.632**（s0 ~0.59，s1 ~0.63） | 0.574–0.597 | 30.7–32.0 m |

- cmd 从 0.200 扫到 0.325（±63%），两 regime 的 realized speed 几乎不动
  ——**C1 恒蠕行 ~0.13 m/s 量级、C2 恒全速 ~0.61 m/s 量级**；err 随 v
  的收敛趋势只是 cmd 向 C2 固有速度靠近的几何效应。
- 训练 seed 连 regime 内部的蠕行速度都改变（C1 OFF s0 ~0.085 vs
  s1 ~0.17）——train-seed 方差在 treatment-free 层同样可见。
- h_min 全格 0.71–0.76 m，168/168 episodes 零倒地（与 audit 一致）。

### 3e. material lineage（全链哈希锚）

| 环节 | 身份 |
|---|---|
| freeze HEAD / runtime | sync clone@`8f6ba1e`（训练）/ commit `1548cfe`（eval） |
| 冻结解码器 | token_vae_e39 `vae.pt` ckpt sha `f6adfc50…`（state_dict `01e4d382…`，before/after 一致）；SONIC decoder onnx（frozen） |
| τ 材料 | G↓ ↓-k0 canonical `gdown_v{v}_k0.npz`（k=0 首过，确定性复现逐位一致）→ derived LUT `to41_lut_{v}.npz` → env buffer pre/post sha 一致 |
| 训练 | `to41r1-{ctrl,t10}-{s0,s1}` 4 runs，E47 精确配方（TO40C_PLAN §3 逐字），128 envs × 2000 it，串行 lab-ts，09-03 02:14–04:50，4/4 rc=0，fall ≤0.8% |
| ckpt 选择 | 预注册 50-iter 窗口 argmax（对称非手挑），manifest sha `8fab587f…`：ctrl-s0 it350 / t10-s0 it200 / ctrl-s1 it1250 / t10-s1 it50（早期窗机械接受，非 treatment effect） |
| eval | 56 receipts（28 cells × 2 train seed），168/168 completed，`eval_checker` G1–G10 全 PASS（audit sha `f9f08753…`），每 ckpt 供 14 cells 共用（G2） |
| eval 随机源 | 唯一入口 jitter rng(1000+seed)；policy deterministic；disturbance_prob=0 |

## 4. 效应量级对照——conditioning 通道 ≥ τ_ff 效应

两个尺度的「conditioning 效应」都大于或等于 τ_ff 效应：

1. **treatment-free 层**：OFF 基线两 regime err 区间全 v 分离，间隙
   0.040–0.339（C2/C1 均值比 1.3×–10.8×；realized speed 0.61 vs 0.13
   m/s ≈ 4.7×）。
2. **τ_ff 响应的 regime 结构**：regime 差 |Δdff(C1−C2)| = +0.068(s0) /
   −0.064(s1)，与 τ_ff condition-specific effect 本身（0.01–0.09）同
   量级甚至更大，且 matched-ness 调制 ≈ 0（+0.001/+0.017）。

结论：**τ_ff 不是在一个近似相同的 locomotion substrate 上做微调，而是在
两个本身已非常不同的 control regimes 上产生不同影响**。这比原问题
「τ_ff 是否有效」更有研究价值，且是本 Rung 1 真正的正面产出。

## 5. Scientific closure

### 5a. What Rung 1 established

1. τ_ff 注入通道在 28-cell 全矩阵内 **execution 干净**（G1–G10 全 PASS、
   override 900/900 生效、OFF 臂零泄漏、168/168 completed）。
2. conditioning override（vb0↔vb1 decode 干预）是**强 behavioral
   treatment**（两 seed 一致）：单方面改变步态速度 4.7×、err 基线最高
   10.8×。PASS-AS-CHANNEL 纪律下的「通道可用」口径在此再次成立，且
   本轮是 decode 侧而非 τ 侧。
3. eval 管线随机性 = 初始条件扰动且系统强镇定；eval 方差相对 train-seed
   方差可忽略（F_like 中位 133）。
4. C1/C2 构成两个无共同 support 的 decode regimes（7/7 OFF 分离）；
   natural/matched 是 (v,C) 派生属性。
5. τ_ff 的 condition-specific effect **强依赖训练 seed 与 decode
   regime**：C1 regime 内符号随 seed 翻转；C2 regime 内两 seed 同号
   但幅度差 ~5×。

### 5b. What Rung 1 failed to identify

1. **稳定的 τ_ff 主效应**：不存在跨 seed 一致的 Δ_ff（C1 regime 符号
   翻转；无一格两 seed 同过 0.02 边界同向）。
2. **pooled causal decoder-condition interaction**：共同 support 不成立
   （3c），Δ_cond 在 bin 边界两侧 estimand 内容互换（§1）——即便数值
   大且结构化，该 contrast 不可辨识。
3. **「τ_ff 填补低速 decoder 空洞」叙事**：TO40C 低速正向（−0.09）与
   本轮 C1-s0 正带（+0.09 ON 更差）方向相反，且该正带被 s1 完全翻转
   ——低速效应不跨训练 seed 稳定。若论文要写「填空洞」，当前证据不足；
   严谨叙事 = **「τ_ff interacts with a strongly regime-dependent frozen
   decoder substrate, and its low-speed effect is not stable across
   training seeds」**。

### 5c. What remains open（均为未来独立实验，不属本线）

1. 第三训练 seed：只回答 regime×τ_ff 交互符号的 seed 稳定性（当前 s0/s1
   恰换向），**不能**把 C1=0.13 / C2=0.61 m/s 变成共同 support——是
   future robustness experiment，不是本 Rung 1 的修复。
2. regime 可比化设计（如 speed-bin 语义与 cmd 解耦的新设计维度）——本线
   明确不做。
3. decode regime × 训练相位互作假说（t10-s1 it50 早期窗、已知 KL 漂移
   形态）——仅记录，未检验。
4. 机制解释（为何 vb0 regime 对 τ_ff 的响应符号随 seed 换向）——本轮
   无证据，保持未知。

## 6. 核心结论（论文结果段骨架）

**中文**：Rung 1 未能在训练 seed 间建立稳定的 τ_ff 主效应。观测到的
τ_ff contrast 强依赖训练 seed，且在两个 OFF 基线按预注册可比性判据
无共同 support 的 decoder regimes 之间不同。因此 pooled 跨 regime
interaction 不可辨识；合法的科学产出是 regime-specific τ_ff contrast。
conditioning 通道本身是强行为 treatment（两 seed 一致），但其与 τ_ff 的
跨 regime 交互不可辨识。

**English（canonical）**：

> Rung 1 did not establish a stable τ_ff main effect across training
> seeds. The observed τ_ff contrast is strongly training-seed dependent
> and differs between two decoder regimes whose OFF baselines have no
> common support under the pre-registered comparability criterion.
> Therefore the pooled cross-regime interaction is not identified; the
> valid scientific output is regime-specific τ_ff contrast.

## 7. 关闭声明

- **关闭**：Rung 1 主矩阵（{ctrl,t10} × 2 seeds × 7 v × 2 C × 3 eval
  seeds = 56 cells / 168 episodes）全部判读；eval seed 增补问题；第三
  训练 seed 对本线的授权问题（永久不授权为本线修复，仅可能为独立后续）。
- **冻结**：ckpt manifest（8fab587f）、τ 材料（G↓ k=0）、Mode A runtime、
  decoder binning、speed grid、effect_table_v1 / diagnosis_v1 判读口径。
- **零新增执行声明**：本文所有数字来自已入仓产物
  （effect_table_v1.txt / receipts / diagnosis_v1 / ckpt_selection.json /
  eval_audit.json），零新增训练、评测、protocol 变更；云端算力消耗 0。
- Run 行见 `tracker/TO.md` TO41 系列（TO41-R1-\* 四行已补 DONE 终态；
  本轮收束为叙事块，无新 Run 行——无新执行）。
