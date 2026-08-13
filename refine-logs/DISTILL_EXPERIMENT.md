# 蒸馏方案可行性实验（2026-08-11）

## 一句话结论

**可行，但有前提**：用「相位路由器」结构（command + proprio → 步态相位 → 专家原型 token）蒸馏官方闭环数据，可以在 MuJoCo 无弹性带条件下实现稳定的站立、慢走和行走（3/3 随机种子 × 20 秒，行走 vx≈0.81–0.83 m/s，与官方一致）；而朴素行为克隆（BC 回归直接拟合 token）在闭环中全部失败。蒸馏方案的「上限」不是数据或信息不足，而是模型结构能否保留步态相位。

## 实验设置

### 数据采集（官方闭环，无弹性带）

- 链路：`run_sim_loop.py`（MuJoCo sim，`ENABLE_ELASTIC_BAND: False`）+ C++ `g1_deploy_onnx_ref`（planner → encoder → decoder）。
- 时长约 7 分钟，50 Hz 控制，共 **20,838 步**。
- 命令覆盖：IDLE、SLOW_WALK、WALK、FORWARD_JUMP（各含 idle 间隔与急停过渡）。
- 每步记录：
  - token（64 维，policy-input logfile 的 token 块，与 state logger `token_state.csv` 完全一致，最大差 0.0）；
  - proprio 历史（930 维：10 帧 ang_vel + joint pos/vel + last_actions + gravity，与 apt_g1 环境 `_get_sonic_history()` 语义一致）；
  - 命令特征（13 维：模式 one-hot + 方向 + 速度 + 高度 + planner 标志，来自 `--record-input-file`）。
- 验证：encoder 输出已是 k/16 网格（25 个取值）；官方链路把 encoder 输出**直接**喂给 decoder（不再做 wrapper 的 FSQ 二次量化，二次量化会改变 token，例如 0.375→0.3125）。

### 评测

- `MujocoG1FlatEnv`（无带、aux=0、token 直通 decoder）。
- **Oracle 控制组**：官方 token 按相位顺序回放 → 各场景全部稳定（walk vx≈0.8）→ 证明评测链路正确，失败不是 harness 问题。
- 最终实验：1000 控制步（20 s）× 3 个随机种子（初始高度/关节/速度加小扰动）。

## 关键发现

1. **oracle 可行**：官方 token 回放即可稳定站立/慢走/行走（跳跃在 500 步级偶尔边缘）。蒸馏的上限 = 官方闭环行为。
2. **token 序列本身高度可预测**：用上一帧 token 预测下一帧，val 精度 93.8%（MSE 0.00025）。
3. **state→token 的标签是多值的**：planner 的参考相位不锁机器人身体相位（每次 1 Hz 重规划都有独立的帧相位），同一 proprio 可能对应不同参考 token。回归模型学的是条件均值 → 相位被抹平。
4. **闭环复合误差**：开环 val MSE ≈0.0012，闭环中 token 误差涨到 0.028–0.039（约 20–30 倍）；oracle 注入实验显示每 5–10 步必须注入一次正确 token 才能维持稳定。
5. **解码器对随机误差不敏感**：oracle token 上随机扰动 8/64 维 ±1 level 仍稳定 → 失败不是 decoder 敏感，而是预测误差的**系统性**。
6. **kNN 记忆蒸馏可行**：每步用 proprio 在训练集中找最近邻（同命令组），输出其官方 token → 4 种步态 600 步全部存活 → 证明 state→token 映射在原则上可学，此前神经网络失败是结构/学习问题。

## 模型对比

| 模型 | 开环 per-dim 精度 | 闭环表现 |
|---|---|---|
| MLP / Deep / GRU / Transformer（非 AR） | 60–72% | 全部跌倒（约 3–10 s） |
| AR（上一帧 token 作输入） | 72% | 跌倒（约 3–5 s） |
| AR-delta（teacher forcing） | ~100% | 更差（约 1–2 s，exposure bias：推理时 prev 是自己预测，分布漂移） |
| AR-delta + free-run 微调 | – | 仍跌倒（约 1–2 s） |
| 相位分类路由器（固定周期分箱） | 相位精度 idle 5%/slow 95%/walk 15–28%/jump 100% | idle ✓ slow ✓；walk/jump ✗（固定周期与 1 Hz 重规划不符） |
| **相位回归路由器（最终）** | 相位角误差 slow≈0.001/walk≈0.005/jump≈0.016 | **idle 3/3、slow 3/3、walk 3/3、jump 1/3** |
| kNN 记忆蒸馏（上限参照） | – | 4 种步态 600 步全部存活 |

## 最终方案：相位回归路由器

每个命令组（mode, speed, direction）独立一个模型：

1. 对该组官方 token 轨迹做 PCA（64→2），得到圆形步态相位 φ = atan2(c2, c1)；
2. 将 φ 分成 40 个相位 bin，原型 token = bin 内官方 token 的均值（量化到 k/16）；
3. 小 MLP（930+13 → 2）回归 (sin φ, cos φ)，MSE 训练；
4. 推理：MLP → 相位 → bin → 原型 token；对相位 (sin, cos) 做 EMA(0.3) 平滑。

关键设计：**不回归 token 本身，而是回归相位 + 查表**，保留步态的周期/离散结构；这也是它与 BC 回归的本质区别。

## 最终结果（1000 步 = 20 s，3 seeds，初始扰动）

| 场景 | 完成数 | h_min | vx | 位移 |
|---|---|---|---|---|
| IDLE 站立 | 3/3 | 0.759–0.760 | 0.003 | 0.06 m |
| SLOW_WALK | 3/3 | 0.689–0.727 | 0.13–0.56 | 2.5–11.2 m |
| WALK | 3/3 | 0.729–0.732 | 0.81–0.83 | 16.2–16.6 m |
| FORWARD_JUMP | 1/3（另 2 个 ~6 s 跌倒） | ~0.20 | 0.21–0.84 | ~4.5 m |

命令切换测试（idle 5s → walk 10s → idle 5s → slow 10s → jump 5s → idle 5s，共 40 s）：**完整通过**，h_min 0.74。

## 上限与局限

- 跳跃是当前边缘项：oracle 本身在该 env 中 500 步级也会偶发跌倒；需要更多跳跃数据、更细原型、相位平滑或 DAgger 提升。
- 倒走被留出（未进训练数据），未覆盖；速度连续变化（官方命令空间是离散模式）未覆盖。
- 20 s 以上长时鲁棒性、真实硬件未验证。
- 每命令组一个路由器（命令空间小，可接受）；跨模式连续切换已验证。
- 数据分布只在官方闭环流形内；离流形状态需要 DAgger 或 RL 补充。

## 对原计划的回答

1. **蒸馏是必要的吗？** 对当前目标（低成本、单卡、无 Isaac Lab 的 no-band 控制器）而言，蒸馏是**充分且最低成本**的路线：它直接产出了可工作的无带站立/行走，不需要并行 RL。
2. **之前的 BC 回归为什么失败？** 不是数据量或模型容量问题，而是「回归 token 均值」抹掉了相位多值标签；必须用「相位 + 原型」或检索式结构。
3. **下一步**：
   - 补数据（跳跃/倒走/变速/转弯段）重训相位路由器；
   - DAgger：用当前路由器驱动闭环，收集状态并用官方 encoder/相位标签重标注，迭代提升；
   - 若追求地形/高速等更高上限：转 Isaac Lab 并行 RL，蒸馏结果可作为初始化或先验。

## 产物

- 数据：`apt_g1/data/exp1/{proprio,cmd,token,mode,speed,meta_modes}.npy`
- 脚本：`apt_g1/{train_distill*.py, train_router.py, train_phase_router.py, train_knn_mlp.py, eval_distill.py, knn_eval*.py, perturb_eval.py, eval_final*.py}`
- 结果：`apt_g1/outputs/distill/{eval_phase_final.json, eval_switch.json, eval_knn*.json, perturb.json, eval_ar*.json, eval_mlp.json, model_*.pt, norm_*.npz, phase_*.pt/npz/json}`

---

## 第二期：补命令覆盖（exp2）与路由 v2-v5（2026-08-11 续）

### 补采数据 exp2

- 新增 32,675 步官方无带闭环数据（**0 次跌倒**），命令覆盖：
  - 慢走/行走的**倒走**（mode 1/2 + 's'）、**转向**（e/q，多角度）、**横移**（','/'.'）、更多跳跃（3,756 步）、STEALTH_WALK（2,255 步）。
- 合并后数据集 `apt_g1/data/exp_all/`：**53,513 步**，5 种模式（0/1/2/17/18）。
- 路由分组改为 (mode, speed, 方向角 8-bin)，每个命令组独立相位路由器。

### 迭代过程中发现并修复的问题

1. **v2 回归**：合并后 idle 变差（倒走跌倒）→ 原因：每段 idle 开头有 1–2 s 由上一步态衰减来的过渡 token（段首 token 距离 p90=1.83 vs 段中 0.11），污染 PCA 相位与原型。修复：密度离群过滤（每行到组内最近邻 token 距离 ≤2.5×中位数）。
2. **评测 bug**：只测 vx 会把横移/转向误判为「没动」；且转向类命令应 fdir=mdir（训练里 mdir≈fdir，评测若 fdir=[1,0,0] 是 OOD 输入）。修复：2D 速度/路径指标 + fdir=mdir。
3. **慢走速度退化**：合并后慢走原型几乎站住（vx≈0.01）。定位：exp2 慢走段与 exp1 慢走是**不同节律**的步态（oracle 回放 exp1 vx 0.32 vs exp2 vx 0.09），混在一起 PCA 相位失效；且 exp1 末段慢走 token 本身在 env 里不稳。修复（v5）：慢走组只用 exp1 第一段（4,504 步）→ vx 0.07–0.16，2/3 种子完整。

### v5 最终结果（20 s × 3 随机种子，无带，2D 指标）

| 场景 | 完成数 | h_min | vx / vy | 路径 |
|---|---|---|---|---|
| IDLE | 3/3 | 0.76 | 0.003 / 0.0 | 0.1 m |
| SLOW_WALK 前进 | 2/3 | 0.72–0.76 | 0.07–0.25 | 2.2–4.5 m |
| SLOW_WALK 后退 | 3/3 | 0.76 | ≈0 | 0.9 m（官方后退慢走本身几乎不动） |
| WALK 前进 | 3/3 | 0.68–0.71 | 0.83 / 0.04 | 16.8–17.0 m |
| **WALK 后退** | 3/3 | 0.73 | −0.78 | 15.9–16.0 m |
| FORWARD_JUMP | 1/3 | ~0.20 | 0.2–0.8 | 3–5 m（oracle 本身边缘） |
| 右转 60° | 0/3（~4 m 后倒） | 0.20–0.22 | 0.22–0.30 / 0.07–0.18 | ~4 m |
| 左转 60° | 3/3（几乎不移动） | 0.76 | ≈0 | 0.1–0.2 m |
| 右横移 | 3/3 | 0.76 | 0.14 / 0.0 | 3.4–3.6 m |
| 左横移 | 3/3（几乎不移动） | 0.76 | ≈0 | 0.1–0.2 m |
| STEALTH_WALK | 0/3（~7–9 m 后倒） | 0.21–0.22 | 0.96 / −0.12 | ~7.5–8.7 m |
| 58 s 命令切换（含 jump） | 完整通过 | 0.74 | 0.48 / 0.02 | — |

### 第二期结论（上限更完整）

- **可行能力进一步扩展**：除站立/前进行走外，新增 **WALK 倒走（3/3，−16 m）**、右横移、命令切换（58 s 含跳跃）。
- **上限的构成**：
  1. **oracle 本身就是上限**：在本 MuJoCo env 中，官方 turn/strafe/jump token 回放也边缘（部分 ~200 步跌倒），蒸馏控制器不可能超过老师；
  2. **相位路由器对数据同质性敏感**：把不同节律的同类步态混进同一组会破坏 PCA 相位（慢走案例），需要按数据段清洗或分开建模；
  3. **BC 回归的硬上限**仍是闭环复合误差（每 5–10 步需纠正），与第二期结论一致。
- **仍待提升**：慢走速度（0.07–0.16 vs 官方 0.3）、跳跃、转向与 stealth 的鲁棒性；方案：按干净段建模 + DAgger + 更多跳跃/转向数据（或转 Isaac Lab RL）。

### 第二期产物

- 数据：`apt_g1/data/exp_all/`（53,513 步）、`apt_g1/data/exp2_raw/`（原始 CSV+logs）
- 路由 v2–v5：`apt_g1/outputs/distill_v2..v5/{phase_meta.json, phase_g*.pt, proto_g*.npy, phase_norm.npz, eval_battery_v*.json, eval_switch_v*.json}`
- 脚本：`apt_g1/{train_phase_router_v2,v21,v23,v4,v5}.py`, `apt_g1/{eval_battery_v2,v21,v23,v4,v5}.py`

---

## 第三期：原型调优 + DAgger 尝试（v6/v7，2026-08-11 续）

### 原型调优（v6，最终版）

对边缘分组做原型构造方式/相位 bin 数扫描（mean/median/nearest × B=40/64）：

| 分组 | 最佳配置 | 效果 |
|---|---|---|
| 跳跃 (17,-1,bin4) | median, B=40 | **1/3 → 3/3 完成**（20s×3 seeds） |
| 右转 60° (1,0.2,bin5) | mean, B=64 | **2/3 → 3/3 完成**（h_min 0.67–0.75） |
| 左转/左横移 (1,0.2,bin2) | nearest, B=40 | 从「站立不动」→ **3/3 且真正移动**（路径 0.3–10 m） |

其余分组保持 mean/B=40。最终 v6（= distill_final）完整结果：

| 场景 | 完成数（20s×3 seeds） | 速度/路径 |
|---|---|---|
| 站立 | 3/3 | vx 0.003 |
| 慢走前进 | 2/3（其余种子 vx 0.08–0.23） | 2.3–5.1 m |
| 慢走后退 | 3/3 | ~0.9 m |
| 前进行走 | 3/3 | 0.83 m/s，16.9 m |
| 倒走行走 | 3/3 | −0.78 m/s，16.0 m |
| 跳跃 | 3/3 | 3.7–6.3 m |
| 右转 60° | 3/3 | 3.2–4.9 m（vy 0.11–0.14） |
| 左转 60° | 3/3 | 0.3–10.1 m |
| 右/左横移 | 3/3 | 3.4–3.5 / 0.4–8.4 m |
| STEALTH | 0/3 | **与 oracle 持平**（官方 stealth token 回放也在第 361 步跌倒） |
| 58s 命令切换（含跳跃） | 完整通过 | h_min 0.74 |

### DAgger 尝试（v7，未采用）

对慢走做了 1 轮 DAgger-lite：用 v6 慢走路由器闭环 rollout 收集 2,543 个学生状态，用 kNN（最近官方慢走状态）的 PCA 相位作标签重训相位网络。结果：慢走从「2/3 完成、vx 0.08–0.23」退化为「3/3 完成但 vx≈0.01（站住不动）」——kNN 相位标签在慢走这种弱节律步态上偏向保守均值，收益为负。**结论：慢走这类弱节律步态需要更干净的参考标签（如真实 planner 相位）或 RL 辅助，单纯加闭环状态重标注不够。**

### 第三期结论（可行性 + 上限终版）

1. **蒸馏方案可行性：证明**。一个小型神经网络（14 个命令组相位路由器，每组 ~0.5M 参数）从 53,513 步官方无带闭环数据蒸馏而来，可在 MuJoCo 无带条件下稳定完成：站立、慢走（2/3）、前进/倒走行走（3/3，速度与官方一致）、跳跃（3/3）、转向/横移（3/3）、58s 命令切换。
2. **上限构成（已量化）**：
   - **教师上限**：stealth 官方 token 在本 env 本身不稳（361 步倒），蒸馏控制器无法超越老师 → 0/3 属于上限内；
   - **数据同质性**：相位路由器对「同一命令下不同节律的混合数据」敏感（慢走案例），需要按干净数据段建模或原型调优（median/nearest/B 扫描可显著改变结果）；
   - **BC 回归硬上限**：直接 token 回归的闭环复合误差（每 5–10 步需纠正）依旧无法绕过；
   - **弱节律步态**（慢走）是当前唯一明显低于教师水平的项目（vx 0.1–0.2 vs 官方 0.3），1 轮 DAgger-lite 未改善，需要更精确的相位监督或 RL。
3. 20 s 以上长时鲁棒性、真实硬件未验证（留给后续）。

### 第三期产物

- 最终路由：`apt_g1/outputs/distill_final/`（= distill_v6：`phase_meta.json, phase_g*.pt, proto_g*.npy, phase_norm.npz, eval_battery_v6.json, eval_switch_v6.json`）
- DAgger 尝试：`apt_g1/outputs/distill_v7/`
- 脚本：`apt_g1/{proto_variants.py, build_v6.py, eval_battery_v6.py, eval_battery_v7.py, train_dagger_slow.py}`

---

## 第四期：单一 encoder 模块 + 60s+ 压力测试（2026-08-12）

### 单一 encoder 模块

`apt_g1/encoder/phase_router_encoder.py` 提供 `PhaseRouterEncoder`：

- 统一接口：`encode(Command, proprio_history) -> 64-d token`，内部自动完成组选择（mode+speed+方向角 bin，含回退）、PCA 相位推理、原型查表、EMA 相位平滑，组切换时自动重置 EMA；
- 高层命令映射：`Command.from_vxvy(vx, vy, yaw)` 把线速度命令映射到官方命令空间（IDLE / SLOW_WALK / WALK / 倒走 / 转向）；
- 单目录加载（distill_final 全部权重+原型+归一化），可替换官方 planner+encoder 链路。
- 压测全程通过该模块驱动（与内联评测一致：vx 0.83、60s 50.9–52.0 m），验证了模块与评测实现等价。

### 60s+ 压力测试结果（无带 MuJoCo，`stress_test.py`）

**A. 连续直线行走 60 s × 3 seeds：3/3 完成**（vx 0.83–0.85，位移 50.9–52.0 m）

**B. 行走中扰动冲击（45 s，t=10s/25s 各一次，200/500 N，4 方向 × 3 seeds = 24 组）：21/24 完成**

- 绝大多数冲击后 0.02–2.6 s 内恢复（|vx−0.83|<0.15 持续 0.5 s）；
- 3 组失败为种子依赖的边缘情况（fwd-500 seed0、left-200 seed1/2），且冲击后高度仍 ≥0.68（冲击本身被吸收，跌倒发生在更晚时刻）→ 说明单次扰动不是问题，复合应力下的边缘状态才是。

**C. 68 s 命令切换马拉松 × 3 seeds：0/3 完成** —— 均在中后期跌倒：

- seed0/seed2 在 jump 阶段（约 3–4 s 内）跌倒；seed1 在 walk_back 阶段跌倒；
- 隔离验证：walk_back 单独 60 s 3/3 完成（−0.80 m/s，48.7 m）；walk 40 s→idle 3 s→jump 5 s 为 2/3（幸存时 h_min 也仅 0.21–0.23）→ **长时间连续运行后进入跳跃是当前最脆弱的环节**。

**重要修正**：此前「58 s 命令切换通过」是 env 默认 episode_length=20 s 的假象（1000 步即终止并复位）；本次用正确的 episode 长度重测后，多命令连续运行的真实上限约为 35–40 s，瓶颈在跳跃（及个别 walk_back 转换）。

### 压力测试结论

1. 单命令连续运行（前进/倒走）60 s+ 稳定，位移约 50 m；
2. 行走中 200–500 N 冲击基本可吸收并恢复；
3. **多命令连续切换是当前真实上限**：35–40 s 后进入跳跃会跌；这是比「20 s 单场景」更可信的控制器上限表述；
4. 修复方向：跳跃的长时间上下文鲁棒性需要真实 planner 相位监督的 DAgger，或 RL 微调（此前 kNN 近似标签的 DAgger 已证无效）。

### 第四期产物

- 模块：`apt_g1/encoder/{__init__.py, phase_router_encoder.py}`
- 压测：`apt_g1/stress_test.py`（A/B/C 三块）、`apt_g1/stress_isolate.py`（隔离验证）
- 结果：`apt_g1/outputs/distill_final/{stress_test_results.json, stress_isolate.json}`
