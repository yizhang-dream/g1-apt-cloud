# G1 人形「腿级/全身轨迹优化（TO）」技术路线调研报告

> 背景：复现 Science Robotics 2026 APT-RL 管线。论文原版 TO 是 SRBD（2D 单刚体），
> 对四足成立，搬到人形 G1 后（TO06）失败：SRBD 力矩量级（hip 峰值 25–76 N·m）
> 远小于 G1 行走所需（~100–300 N·m），短暂移动即倒。
> 本报告回答：标准人形腿级/全身 TO 有哪些做法、从 SRBD 升级的最小可行第一步是什么、
> 关键参考与代码、风险与陷阱。
> 产出日期：2026-08-15。仅分析报告，不写代码文件。

---

## 0. 先对齐：为什么 TO06 失败（这不是「调参」问题，是「模型"类"错了」）

TO06 的反馈链条很关键：`eval_torque_srb.py` 把 SRBD 解出的力矩当作**前馈** τ_SRB
加到 G1 的力矩闭环 τ = τ_SRB + kp·(q_des − q) − kd·q̇ 上。SRBD 解出的 hip 力矩用
**静态力矩臂** τ_hip = Fz·(foot_x − x) 算，它假设「腿无质量、力只由 GRF 提供」。
对人形 G1：

- 缺少**腿段惯性**（大腿 L1≈0.34m、小腿 L2≈0.34m 各自有质量与转动惯量）；
- 缺少**摆动腿动力学**（swing leg 的真实 hip/knee 力矩，SRBD 里根本不存在 swing leg）；
- 缺少**躯干角动量控制**（SRBD 只有一个 pitch 惯量 I=3.981，代表整机，不分裂出躯干）；
- 缺**完整逆动力学** τ = M(q)q̈ + C(q,q̇)q̇ + g(q) + Jᵀf_c（现在只剩 Jᵀf 一个静态项）。

所以 TO06「量级差一个数量级、且结构上少了摆动相」是**模型选择错误**，不是 solver 收敛
或成本函数调不好的问题。正确方向=把优化模型从「点质量单刚体」升到「有多段腿的
多体模型」，让力矩由**完整多体动力学**自洽产出。

**术语**：SRBD = Single Rigid Body Dynamics（单刚体动力学，把整机当作 6DOF 刚体、
腿只贡献足底接触力）；leg-level/full-body TO = 腿级/全身轨迹优化，把腿的关节当作
显式优化变量。

---

## 1. 标准人形/双足「腿级/全身 TO」的主流做法（对照 + 适用性评估）

下面 5 类是领域里真实覆盖「点到嘴、能产出周期性行走」的做法。每类给
模型复杂度 / 是否要 hybrid dynamics / 典型求解器 / 代码成熟度 / 对 G1 场景的评价。

### (a) Centroidal Dynamics TO（质心动力学轨迹优化）

- **模型复杂度**：中。整机用一个质心点 + 一个质心角动量矢量（Centroidal Angular
  Momentum，CAM）描述，质心动力学方程：
  `m(p̈_c − g) = Σ f_i`（力的质心方程）以及
  `Ḣ_c = Σ (r_i − p_c) × f_i`（角动量平衡，这里把"躯干+腿"的角动量都并进 H_c）。
  这是 SRBD 的**直接提升版**：SRBD 其实丢掉了 H_c（假设角动量守恒/为 0），
  centroidal 版把 H_c 显式当状态变量，**能抓住躯干角动量控制**。腿往往仍简化为
  「点到点接触」或加一个**腿延伸动力学 (limb dynamics)** 的简化模型。
- **Hybrid dynamics**：不是必须显式建（可以用「质心轨迹 + 预置足序」的
  相分离表达，或在一个规划层里用 timetable 解决）。
- **典型求解器**：IPOPT（CasADi/MATLAB 里很多实现）、或 MIQP/QP 序列化
  （如针对规划的 centroidal dynamics 二次规划）。
- **代码成熟度**：高。This is the workhorse of humanoid TO+WBC stacks
  （ATLAS / TORO / HRP / Talos 的很多 WBC 上都先用 centroidal 规划）。

**对我们的适用性**：**高，且是最符合「从 SRBD 增量升级」的一条**。因为 centroidal
正是「SRBD 少了什么」的答案——它加了角动量 H_c（躯干控制）和 limb 质量效应。
但它**还不够**：单纯 centroidal TO 仍不产出「摆动腿的 hip/knee 力矩」，也没法保证
「腿段惯量在摆动相被驱动」的物理量级。所以它做**规划层**很合适，但做大闭环
**前馈力矩**时仍缺 leg-ID 那一段（见 §2 的推荐，正好 =「centroidal 规划 + full-ID」）。

### (b) 平面双足直接配点法（Direct Collocation，5/7 连杆平面双足）

- **模型复杂度**：高但结构规整。直接用**拉格朗日多体动力学**（5 连杆 = 躯干+
  大腿×2+小腿×2；7 连杆再加足/或加脚踝自由度），每个连杆有质量/惯量。动力学是
  `M(q)q̈ + C(q,q̇)q̇ + g(q) = B·τ + Jᵀ·f`，其中 q 是平面关节向量、
  f 是接触力（用互补条件/直接配点把接触建模为路径约束或 impact 约束）。
- **Hybrid dynamics**：**是，几乎是必需的**。平面双足一个步态循环 = 支撑相
  （单腿触地）+ 摆动相 + 落地冲击映射（foot strike 时速度突跳，用角动量守恒的
  impact map 更新）。经典实现（MIT Underactuated 的 compass gait 的 dircol 版本、
  Drake 的 walking 示例）都用**显式 hybrid 节点**把一个周期切成长度/相分离的
  子问题再串起来。
- **典型求解器**：IPOPT（via CasADi / Drake）/ SNOPT / 内点法。直接配点把连续
  ODE 转成稀疏非线性等式约束，配点法（Hermite-Simpson / trapezoidal）稀疏性好，
  IPOPT 能解。
- **代码成熟度**：高（MIT Underactuated Python 笔记、Drake 的 `examples/`、
  大量教科书）。

**对我们的适用性**：**模型最「接地气」，但实现成本最高**。它能把「腿段惯性 +
摆动相 + 冲击映射」一网打尽，力矩天然是 ~100 N·m 量级。缺点：5/7 连杆是**平面
（2D sagittal）**模型，只能解 sagittal 步态（不含侧向/横滚，G1 的 3D 双腿还得再
补横滚/偏航），且 hybrid + 冲击 remap 对 IPOPT 初值和缩放敏感，调试周期长。
**作为「第二条路线」很值，作为「第一步」过重。**

### (c) LIPM / 倒立摆 + 全身逆动力学（RNEA）

- **模型复杂度**：低→中。**LIPM = Linear Inverted Pendulum Model（线性倒立摆，
  把 CoM 当摆锤、支撑足当支点、腿无质量、CoM 高度锁定的简化模型）**。LIPM 只解决
  「CoM 轨迹 + 关键点 ZMP/CoP」（ZMP = Zero Moment Point，零力矩点，地面反力合力
  穿过支撑多边形的点），**不解决腿几何**。
- 腿的关节轨迹/力矩由**第二层**——全身逆动力学（RNEA = Recursive Newton-Euler
  Algorithm，递归牛顿-欧拉，从末端（足）向基座递归求各关节力/力矩的 O(n) 算法，
  MuJoCo 的 `mj_inverse` 正实现了它）——从给定的 CoM + 腿 FK 反解出来。
- **Hybrid dynamics**：LIPM 层用「timetable + 足序」规避，不显式建 hybrid；leg-ID
  层本身不需要 impact（只是正向运动学 + 逆动力学）。
- **典型求解器**：LIPM=直线/解析或小 NLP；leg 可用闭式解或二次规划（加接触约束）。
- **代码成熟度**：**极高**。这是工业人形控制最主流的「规划器 + WBC（whole-body
  control，全身控制）」分层范式（ATLAS 早期、A1/Go1 用的轨迹生成 + WBC 都是这条）。
  MuJoCo 生态、DeepMind 的很多双足栈都直接建在这之上。

**对我们的适用性**：**高，且它是"工程最稳健、落地上最快"的一条**。因为「轨迹生成
（LIPM/质心）→ RNEA 逆动力力矩」正好把「谁产生力矩」分给了成熟的 RNEA，力矩量级
天然自洽（RNEA 输入的就是真实 43-DOF G1 的 M,C,g,J，力矩必然落在 100–300 N·m）。
**注意**：这就是我们 `recover_id_torque.py` 已经用 `mj_inverse` 做过的方向——但
我们当时是**回放带位置跟踪控制器的轨迹**取 ID 力矩，缺的是「**自洽规划出的
CoM/腿轨迹**」。所以这条路线真正的增量不是「学会用 mj_inverse」，而是「**先解决
LIPM/质心规划**这一层」。这正是推荐的第一步的雏形。

### (d) 混合零动力学（HZD, Hybrid Zero Dynamics）

- **模型复杂度**：高（且理论性强）。HZD 系（Ames / AMBER 机器人）把步态定义为
  **虚拟约束（virtual constraints）**——控制把某些状态量（通常是髋/膝角）束缚在
   相变量（相位)的光滑函数上，然后用受控 ZD 流形让维数下降，最后只需保证
  hybrid impact 映射把状态映回不变流形（Zero Dynamics，零动力学）即可稳定。
- **Hybrid dynamics**：**核心就是它**——不变性条件必须在「摆动→支撑」冲击映射下
  也成立，这是 HZD 的成立条件。
- **典型求解器**：参数优化（把虚拟约束的系数当决策变量 + 周期性/不变性约束），
  可用 FMINCON/SNOPT/IPOPT；也有少变量下的 NLP。
- **代码成熟度**：中（纯研究）+ 学术代码多（AMBER 实验室、Atrias/Cassie 系）。

**对我们的适用性**：**低→中（不推荐做第一步）**。HZD 是最优雅、但工程门槛最高、
且对「3D + 双足 + 加速度积分」的 G1 场景，把「不变流形」硬造出来很费。它更适合
四足/专用双足的正式稳定性理论，不适合我们「尽快产出自洽 TO 前馈力矩」的目标。
可作为远期优雅方法笔记，不进入本路线。

### (e) 运动学步态合成 + 全身 ID

- **模型复杂度**：低（运动学）/ 中（ID 用真实多体）。步态合成=纯几何（走几步、
  每步足的高低、髋/膝/踝角度由脚的位置 FK+IK 解出），不含动力学优化；
  ID 再用 RNEA 给出产生该轨迹所需的力矩。
- **Hybrid dynamics**：不需要（运动学直接分段，装配/交班用几何保证）。
- **典型求解器**：闭式解/Kinematic IK（如 MuJoCo 自带的 inverse kinematics、
  或 CasADi 里的小 NLP）；ID 用 `mj_inverse`/`mujoco.forward_dynamics`。
- **代码成熟度**：极高（工业 gait 发生器几乎都是这条：轨迹→腿部 IK→力矩）。

**对我们的适用性**：**高，且是「第一步"骨架"」最现实的实现载体**。它与 (c) 很接近，
差别是 (c) 先 LIPM 再 leg，这一条直接几何合成腿轨迹再 ID。**运动学合成 + 完整
RNEA-ID** 正是我们 `srb_to_torque.py`（静态力矩臂）的**正确升级**：把「静态力矩臂
τ=Jᵀf」换成「真实多体 ID 力矩」，量级立刻对上。这条配上 (c) 的 CoM 规划，
就是推荐的第一步（见 §2）。

**五类汇总表**：

| 做法 | 模型复杂度 | 需要 hybrid | 典型求解器 | 代码成熟度 | G1 场景评价 |
|---|---|---|---|---|---|
| (a) Centroidal dynamics TO | 中 | 可不显式 | IPOPT/QP | 高 | 高（做规划层，抓角动量） |
| (b) 平面 5/7 连杆 direct collocation | 高 | 基本必须 | IPOPT/SNOPT | 高 | 模型最实但实现最重（2D） |
| (c) LIPM + 全身 RNEA-ID | 低→中 | 规避（timetable） | 解析/小 NLP + QP | 极高 | 高（工程最稳，力矩自洽） |
| (d) HZD | 高 | 核心 | 参数优化 NLP | 中 | 低（太重，理论门槛高） |
| (e) 运动学步态合成 + 全身 ID | 低 | 不需 | IK + RNEA (`mj_inverse`) | 极高 | 高（第一步骨架） |

> 关键洞察：**这五种不是互斥，而是「层次」**(hierarchy)。工业人形正确姿势是
> (a)/(c) 做**上层规划**（CoM/角动量），(e) 做**腿部运动学合成**，最底下用**完整
> RNEA-ID（或 WBC）**承接关节力矩。我们推荐的第一步恰恰就是「按这个层次搭，但
> 每层用最省的一版」。

---

## 2. 从 SRBD 到多体模型：推荐【最小可行的第一步】

### 结论先行

> **推荐第一步 =「3 段平面双腿（大腿-小腿，不含脚/或含简化脚）+ 躯干」的**
> **单刚体质心支撑 + 摆动腿运动学合成 +【完整逆动力学 RNEA/拉格朗日】力矩**，
> 用 CasADi direct collocation 解一个周期步态 NMP。
即：把 TO06 的「SRBD 点质量 + 静态力矩臂」升级成**「模型"多了一段腿 + 腿有质量
惯量 + 力矩由完整动力学算出"」**，但**不一步上到 5/7 连杆的平面 hybrid dircol**。

为什么这是最小可行：
1. **张拉出我们要的量级**：只要腿段有质量/惯量、且力矩由完整 `M,C,g,Jᵀf` 给出，
   hip/knee 力矩就会落在 G1 真实的 ~100–300 N·m 区间（这正是 TO06 缺的）。这是
   我们能否复现论文 aux 正向价值的**前置硬条件**。
2. **周期行走**：用「周期边界条件」（一个步态循环状态回到自身）+ 一个预先固定的足
   序（左右交替、stance/swing 分开处理），CasADi 配点可解出周期步态。
3. **CasADi 可解**：direct collocation 对规模可控（一个周期 ~20–40 个配点 ×
   每点状态 5–7 个关节 + 接触力），IPOPT 数秒到分钟级可解，GPU 也不需要。

### 具体模型

**关心平面（sagittal, x–z）2D**，这是第一步（3D 见 §3 风险）。

- **段**：躯干（torso，含髋，pitch 惯量 I_t）+ 左/右各：大腿（thigh，长 L1≈0.34m）+ 小腿
  （shin，长 L2≈0.34m）。足（foot）第一步**做刚体足底几何**（两点/support 多边形），
  或干脆当小腿末端的「接触点」，暂不给质量（给也简单，先不给更能聚焦动力学）。
- **关节（2D）**：每个腿 hip_pitch + knee（knee 弯曲），共 4 个主动关节 + 躯干 pitch
  自由度。状态 q = [x_c, z_c, θ_t, θ_hipL, θ_kneeL, θ_hipR, θ_kneeR]（7 维），
  q̈ 对应 7 维，Leg 关节对应 4 个主动力矩 + 躯干被控。
- **质量/惯量怎么定**：直接从 G1 的 MJCF（`scene_43dof.xml`，**在服务器
  `GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/`**）读每段的
  `mass` 与 `fullinertia`。若只想先建标量模型：大腿/小腿按 G1 实测各约 1–3 kg
  （整机 36 kg，两条腿+手臂占比），躯干含负重约 25–30 kg。**宁可用真实 XML 抽参**，
  别手估——因为「量级对不上」正是 TO06 的病根，真实惯量才能把力矩量级拉到正确区间。
- **COMs**：每段各自的局部重心，动力学里用复合 (composite) 公式把各段惯量/质量
  合成到整机 — 这自动包含「腿段惯性」。注意**腿在摆动时，整机质心和惯量张量会变**
  （不同于 SRBD 锁死），这正是要捕获的效果。

**动力学形式**（拉格朗日/RNEA 都行，CasADi 里推荐用 RNEA 符号或 `casadi` 的
动态方程，或用 MuJoCo 的 `mj_inverse` 当黑箱算子在配点里查表）：

```
M(q) q̈ + C(q,q̇) q̇ + g(q) = B·τ  +  J_s^T · f_s        (支撑腿触地力 f_s，姿态约束)
```

- 摆动腿（swing leg）**没有**接触力项，其 hip/knee 力矩由 M,C,g 里的腿段质量/惯量
  提供 → 这就是「摆动腿需要真实力矩」被建模出来。
- `J_s` = 支撑腿足底接触点雅可比，`f_s` = 地面反力（垂直 ≥0 但 ≤摩擦锥，水平 ≤ μ·垂直）。

**周期边界** + **phase（支撑/摆动）** 的 hybrid 表达，**第一步用「timetable」而非
显式 impact map 上量**：
- 一个周期 = 左支撑（右腿 swing）→ 右支撑（左腿 swing）→ 回到起点状态。
- 两相位长度 T/2 由决策变量给，配点分别做。
- **落地冲击（impact map）第一步先不做强化**：用「支撑切换点两腿位置连续、速度连续」
  的弱连续约束近似（对平面平坦地面，这就是"跳过 explicit impact"的妥协；见 §3 为何
  可接受）。这让我们避开 impact remap 的数值爆炸，先把「量级/周期行走」跑通。

**CasADi direct collocation 公式化草图**：

```
决策变量:
  per-node state   s_k = [q_k, dq_k]                      (q 7 维, dq 7 维)
  per-node input   u_k = [τ_hipL, τ_kneeL, τ_hipR, τ_kneeR, f_sz, f_sx]
  (f_s 只在支撑腿求, swing 腿 f_s = 0 或小)
  global: T (周期时长), 相位长度, (可选) 步长 S

直接配点 (trapezoidal / Hermite-Simpson) 在 [0, T] 上, 相邻节点满足:
  q_{k+1} - q_k - (dt/2)(dq_k + dq_{k+1})            = 0        (位移一致)
  M(q_k) dqdot_k + C dq_k + g − B u_k − J_s^T f_k    = 0        (动力学, 每节点)
  (配点法额外满足中点 collocation eq)

约束:
  周期:   s_N = s_0   (q, dq 回到起点 → 周期性行走)
  触地:   支撑脚 z=0 且速度=0 (无滑动); 摆动脚 z>0 (离地, 或 foot clearance 弧)
  摩擦锥: 0 ≤ f_sz;  |f_sx| ≤ μ·f_sz                 (μ≈0.6–0.8 干摩擦)
  力矩限: |τ_i| ≤ 300 N·m (G1 hip/knee effort limit, 标定后收紧)
  关节限: |q_i| ≤ G1 joint limits
  ZMP 落在支撑脚多边形: 这步在 f_sx/f_sz 下隐含, 显式加更稳

成本:
  J = α·Σ ||τ||²  (最小化峰值/均值力矩, 平稳)
    + β·Σ ||(z_c − h_ref)||²  (躯干高度参考, 防塌陷)
    + γ·(推进: 让 CoM 前进期望速度, 或直接用步长/速度 in 目标)
    + 周期 cost (SRBD Eq.2 的类推, 已经是周期约束, 可去掉或保留松弛)
```

**与现有代码的衔接**：`srb_to.py` 可留作「热启动初值源」（先用 SRBD 解出的
z0/th0/周期时长喂给新的多体 NMP 当初值，帮助 IPOPT 收敛），`srb_to_torque.py`
的 IK 骨架可复用（但把静态力矩臂换成完整 RNEA/拉格朗日力矩，或直接对
数据节点用 `mj_inverse` 黑箱算子算真 τ）。`recover_id_torque.py` 已证明
`mj_inverse` 可用——区别在**轨迹来源**：从「回放带位置跟踪器」→ 换成「NMP 自洽
规划的轨迹」。

---

## 3. 关键参考文献与开源代码（每个标注对本场景的适用性）

1. **MIT Underactuated Robotics（Russ Tedrake）——compass gait / 5-link walker**
   - 内容：教科书级「简单腿 + 直接配点 / 打靶法」Python 笔记，含 compass-gait 极限环、
     落地冲击映射、用 `dircol` 配点找周期步态。
   - 代码：[github.com/RussTedrake/underactuated](https://github.com/RussTedrake/underactuated/blob/kneed_compass_gait/README.md)
     （`exercises/contact/compass_gait_limit_cycle.ipynb`）。
   - 适用性：**最适合我们**——它演示了「用极小模型（compass，2 连杆 no massless）
     在 CasADi/Drake 直接配点下获得周期行走 + impact map」，正是我们「从 SRBD 升级 +
     用配点」的教科书入口。**直接抄它的配点骨架，把模型换成 G1 3 段双腿**。

2. **Drake 的 biped / compass gait 示例（RobotLocomotion/drake）**
   - 内容：Robocop（complextorical planning for constrained dynamics）与 compass gait
     biped 示例，展示如何把接触写成路径约束（`TimeSteppingRigidBodyManipulator`）、
     配点 + impact 一体化优化。
   - 代码：[drake github](https://github.com/RobotLocomotion/drake/blob/03407f7e30a1a05d05c6492cd717f272c8156a76/systems/plants/TimeSteppingRigidBodyManipulator.m#L3)。
   - 适用性：**高**，尤其「接触约束建模成 path constraint」这部分，是我们在 CasADi
     里怎么写摩擦锥/无滑动/足离地的**最佳参考**。Drake 本身重（不引入），只借思路。
     另：Posa 的 `Optimization and stabilization of trajectories for constrained
     dynamical systems`（dair.seas.upenn）是接触配点优化的理论底座，值得速读。

3. **CasADi 官方 optimal control examples（direct multiple shooting / collocation）**
   - 内容：CasADi 自带 `dae_multiple_shooting.py`、pendulum 最优控制示例，模板级
     展示「多段/直接打靶 vs 直接配点 + IPOPT」怎么写。
   - 代码：[raw.githubusercontent.com/casadi/casadi/.../docs/examples/python/](https://raw.githubusercontent.com/casadi/casadi/refs/tags/nightly-temp/docs/examples/python/dae_multiple_shooting.py)
     也可以看 [bioptim](https://app4.secure.forcepoint.com/pyomeca/bioptim#7)（CasADi+IPOPT 的
     最优控制框架，把 biorbd 动力学接给 IPOPT，示例丰富）。
   - 适用性：**高（工程模板）**。直接套 CasADi 的 `nlpsol('ipopt')` + RK4/配点
     (我们 `srb_to.py` 已经这么写了)，把模型段换多体。

4. **Centroidal dynamics TO（Orin 等）**
   - 内容：ORIN/Nakamura 2013「Centroidal dynamics of a humanoid robot」建立质心角动量
     模型；后续 Wensing-Orin 把它普遍用于 WBC 规划。也见 「Reaction Mass Pendulum (RMP)」
     等质心角动量感知规划。[IROS14 WBC slides](https://walk-man.eu/WBC-SI/wensing-orin%20-%20IROS14_WS-WBC.pdf)
     、[Centroidal dynamics of a humanoid robot](https://dl.acm.org/doi/10.1007/s10514-013-9341-4)。
   - 适用性：**适合第二/三步**（当我们需要正式「躯干角动量控制」时，把角动量 H_c
     加进质心 TO）。第一步可以先不管纯质心版，直接多体 RNEA 就已经含躯干惯量了。

5. **MuJoCo inverse dynamics（`mj_inverse`）——我们已用的 RNEA 黑箱**
   - 内容：MuJoCo 里 `mujoco.mj_inverse()` 从状态+加速度算「产生该运动所需各关节
     力矩」，等价于完整 RNEA。**`recover_id_torque.py` 已在用**。
   - 适用性：**关键部件** — 它就是我们「完整 ID 力矩」的最省实现：配点解出的
     (q, dq, dq̈, 接触) 直接喂给它得到真 τ，量级必然对（因为用的是 G1 真 M,C,g,J）。

---

## 4. 风险与陷阱

### 4.1 Hybrid dynamics 的落地冲击（impact map）怎么稳
- **风险**：落地瞬间速度突跳，若直接配点把整个周期当连续光滑问题解，冲击处不可微，
  IPOPT 在支撑切换点会卡、收敛差或输出物理上不可能的「穿地」轨迹。
- **健壮做法（按先易后难排）**：
  1. **第一步（推荐）**：用「timetable 双相位 + 节点连续性弱约束」**绕开显式 impact**——
     支撑切换点要求两腿位置/速度连续，先不施加角动量守恒 remap。平坦地面 + 足底几何
     时这是可接受的初版（我们目标是先得到量级正确的自洽力矩 + 周期行走，不是严格
     impact 物理）。**在 G1 MuJoCo 上用 position/velocity PD 吸收**掉一点点冲击，足够
     跑闭环冒烟。
  2. 若必须显式 impact：用**角动量守恒**（`H(q⁻)·q̇⁻ = H(q⁺)·q̇⁺`，H 是关节空间惯性
     矩阵）作为冲击映射，把 `q̇⁻ → q̇⁺` 当配点里一个「跳跃约束」——MIT Underactuated
     的 compass gait 就这么写的，照抄。
  3. 仍不稳：换**直接配置在「冲击点两侧各开一段」**，即把周期切成
     「支撑-摆动-支撑」三段、每段内部光滑、段间接 impact 约束。这是教科书标准做法，
     但节点变多、较慢。

### 4.2 接触约束（摩擦锥 / 无滑移）
- **风险**：漏了摩擦锥/无滑动，解出来可能「脚在地上滑」或「拉拽地面」，量级和物理
  都不对。
- **做法**：每条支撑腿加 `f_sz ≥ 0`（不能拉）且 `|f_sx| ≤ μ·f_sz`（μ 干摩擦系数，
  取 0.6–0.8），以及支撑脚点速度=0（无滑动）。配点法里这些是**逐节点的 path
  约束**，稀疏可解。μ 给太大允许横滑（让步态懒）、太小过度约束（难收敛）——
  起步 μ=0.7 附近调。

### 4.3 摆动腿足部离地（foot clearance）
- **风险**：不约束「摆动脚离地高度」，解可能让 swing 脚蹭地/穿地，闭环里脚撞地。
- **做法**：给摆动脚加**最小离地高度**（如 `z_swingFoot ≥ 0.05m`，或摆动脚沿一段抬弧线
  的最低点约束）+ 整周期无穿地约束（所有脚 `z ≥ 0`）。配点法里就是逐节点线性/非线性
  不等式约束，简单。

### 4.4 平面 (2D sagittal) → 3D G1 的迁移缺口
- **横滚/偏航 (roll/yaw)**：2D 模型没有侧向。G1 双腿本质是 3D——需要侧向（横滚）
  腿内/外转、踝滚转、以及偏航（转向）。**第一步别碰 3D**：先在 2D sagittal 拿到
  「量级正确 + 周期走」的力矩与轨迹，验证「单腿支撑 / 摆腿 / 5 连杆动力学能产出
  100–300 N·m」这一核心结论。
- **缺失的肩膀/手臂**：G1 有 14 个上半身 DOF（双肩+肘等）。2D 模型忽略它们 = 力矩只
  覆盖 hip/knee(踝)。**闭环时**躯干/手臂惯性对步态稳定性有影响（尤其大步快走），
  所以 2D 解必须先**在 G1 全模型 MuJoCo 上闭环冒烟**（把 2D 力矩投影到对应的 sagittal
  关节，其余关节 PD 稳住），确认不因缺肩臂补偿而倒——这是 2D→3D 的**最小验收门**。
- **踝/lateral 踝**：真正的 3D 需要把踝滚/踝偏的力矩也补充，那是升 3D 模型的第二步，
  不在第一步。

### 4.5 其他工程陷阱
- **初值与缩放**：多体 NMP 对比 SRBD 更容易局部最优/发散。用 `srb_to.py` 的周期解
  当热启动 + 对关节角单位（rad）与力矩单位（N·m）做归一化，ipopt 设 `mu_strategy`
  `adaptive`。不收敛就减小配点密度起步。
- **腿段惯量别手估**：从服务器 `scene_43dof.xml` 抽真实 mass/fullinertia，否则
  "量级对不上"的 TO06 病根会复现。
- **别一步上 (b) 平面 dircol + 显式 impact**：那是第二条路线，第一步用它会让
  「量级/周期行走」这个核心问题被 impact remap 的天坑拖住。

---

## 5. 推荐的第一步（一句话 + 难度 + 风险）

> **一句话：用 CasADi direct collocation 做一个「躯干 + 两条各 2 连杆腿（含真实
> 段惯量）+ 支撑脚接触（摩擦锥/无滑动）+ 摆动脚离地」的平面 2D 周期步态 NMP，
> 得到自洽的 (q,q̇,τ) 轨迹，再用 MuJoCo `mj_inverse`（RNEA）对节点算真力矩，把
> 2D 力矩投影到 G1 的 hip_pitch/knee/ankle_pitch 关节在 43-DOF 模型上闭环冒烟，
> 验证力矩量级落在 100–300 N·m 且能短暂行走。**
> **为什么**：它正好是「从 SRBD 升级」的**最小增量**——把「无质量腿 + 静态力矩臂」
> 换成「有质量腿 + 完整 RNEA 力矩」，而**不一步上 5/7 连杆 + 显式 impact 的天坑**；
> 它直接回答「G1 人形腿级 TO 做出来、力矩量级对不对」这个核心问题，是所有后续
> （力矩级 aux 正向、全 3D TO、WBC 解码器）的**前置门**。

- **实现难度**：**约 3–5 天（人/周）**。其中：配点建模 + IPOPT 收敛 ~1–2 天；
  G1 场景 XML 抽惯量 + 2D 模型 ~0.5 天；mj_inverse 力矩 + 2D→G1 投影闭环冒烟
  ~1 天；调摩擦锥/离地/初值收敛 + 第一次能稳定跑通 ~1–1.5 天。
- **主要风险点**：
  1. **IPOPT 收敛 / 初值敏感**（多体 NMP 局部最优）——用 SRBD 热启动 + 减少配点 +
     归一化缓解，**高**。
  2. **2D→3D 时缺肩臂/横滚导致闭环倒**——第一步用「仅 sagittal 关节受力矩、
     其余 PD 稳住」的冒烟口，可能发现必须补踝横滚，**中**。
  3. **显式 impact 引发的物理/数值问题**——第一步用「timetable + 连续性弱约束」
     规避，**低**（先不碰）。
  4. **摩擦锥/离地约束过度或不足**——μ 和 foot clearance 参数要调，**中**。
- **验收门（DoD）**：(1) NMP 解出一个周期闭环返回初始状态的周期解；(2) 完整
  RNEA 力矩 hip/knee 落在 ~100–300 N·m 区间（对比 TO06 的 25–76 N·m）；(3) 该轨迹
  在 G1 43-DOF MuJoCo 上闭环能持续站/短暂移动不倒（不强求远距走，先"不倒 + 量级对"）。

---

## 术语表（按出现顺序）

| 术语 | 中文一句话定义 | 检索关键词 |
|---|---|---|
| SRBD | 单刚体动力学：整机当 6DOF 刚体、腿只贡献足底接触力 | single rigid body dynamics |
| TO | 轨迹优化：求解一条满足动力学/约束、最小化成本的状态-输入轨迹 | trajectory optimization |
| centroidal dynamics | 质心动力学：用质心点 + 质心角动量描述整机动量 | centroidal dynamics |
| CAM | 质心角动量：整机相对质心的总角动量 | centroidal angular momentum |
| direct collocation | 直接配点法：把连续 ODE 离散到节点、用稀疏等式约束求 NLP | direct collocation direct transcription |
| hybrid dynamics | 混合动力学：支撑相 + 摆动相 + 落地冲击映射 的组合 | hybrid dynamics impact map |
| impact map | 冲击映射：落地瞬间用角动量守恒更新速度 | impact map foot strike |
| LIPM | 线性倒立摆：CoM 当摆锤、支撑足当支点的简化行走模型 | linear inverted pendulum |
| ZMP / CoP | 零力矩点/压力中心：地面反力合力穿过支撑多边形的点 | zero moment point |
| RNEA | 递归牛顿-欧拉：从足向基座递归求各关节力和力矩的 O(n) 算法 | recursive newton euler |
| WBC | 全身控制：用整机动力学 + 优化分配关节力矩满足多个任务 | whole body control |
| HZD | 混合零动力学：用虚拟约束把状态约束到不变流形上稳定步态 | hybrid zero dynamics |
| virtual constraints | 虚拟约束：把某些角约束成相位变量的光滑函数 | virtual constraint |
| foot clearance | 足部离地净空：摆动脚最低抬离高度 | foot clearance swing |

---

## 与项目定位的衔接（为什么这条路不重蹈已判死路线）

- **不重蹈「PD 力矩解码器」**：我们要的是**自洽规划力矩**（不含位置跟踪器的
  完整动力力矩），不是 PD 跟踪误差标签。多体 TO + RNEA 产出的是规划前馈。
- **不重蹈「SRBD」**：模型从点质量升到多体，力矩量级才能对上。
- **不重蹈「从零直出关节」**：我们不训练策略，只产出轨迹/力矩数据，作为
  力矩级解码器 + aux 残差的正向前置——这正是「复现论文 aux 正向价值」唯一
  缺的「自洽 TO/ID 规划数据」（HANDOFF §5 未决事项）。
- **回答主线问题**：这条路的产出（G1 腿级 TO 力矩）→ 训练力矩级解码器 →
  在「冻结解码器 + aux」管道里测 aux 正向价值，直接回答论文「aux 正向」机制
  是否成立。所以第一步是整个「力矩级 aux 正向」论证链的地基。
