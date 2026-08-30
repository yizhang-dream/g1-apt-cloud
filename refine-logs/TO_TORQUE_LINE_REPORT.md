# TO 力矩线收束报告（TO01–TO22，2026-08-14 ~ 08-16）

> 定位：回答"论文的 aux 正向价值需要什么前提"这条线的**前半程**（数据管线
> 成功 + 力矩闭环负结果 + 八假设系统消除）。后半程 QP-WBC bring-up（TO23–35）
> 见 `WBC_BRINGUP_REPORT.md`。逐 Run 数据行见 `EXPERIMENT_TRACKER.md` TO 段，
> 本文是机制层收束。

## 1. 一句话结论

自洽 TO 力矩数据**可以自建**（SRB TO→IK→力矩，解码器 MAE 0.57–1.1 N·m），
但任何"合成步态 + 关节空间 PD/前馈"的力矩闭环都在 ~3.5s 倒——八个便宜解释
全部排除后，根因是**缺浮基反应式稳定层**（QP-WBC / 落脚点自适应），而不是
力矩算错、PD 不够硬或模型不够好。

## 2. 数据管线成就（TO01–TO05，正结果）

| 环节 | 结果 |
|---|---|
| SRB TO（CasADi 复刻论文 Impulse-scale TO：2D 单刚体 + 半正弦 GRF + 动量守恒 Eq.1 + 周期成本 Eq.2 + IPOPT） | walk/run 全速度收敛，周期 cost ~1e-17（完美周期） |
| duty factor 扩展（walk d=1.0 / run d=0.5 腾空相 + 垂直速度周期项） | run cost 9.4e-2 → 1.7e-17（F_x 正弦破坏周期性、腾空相未积分这两个物理 bug 修复后） |
| 2 连杆 IK + 静态力矩臂 + ankle CoP 杠杆臂 | (state,torque) 数据集；IK 往返 err≈0 |
| 力矩解码器 MLP(sinφ,cosφ,v,d)→(hip,knee,ankle) | **MAE 0.879/1.100/0.572 N·m**（vs PD 标签 18.76、ID 力矩 4.13）→ 自洽规划力矩可学性 ~60× 优于 PD 标签 |
| 跨步态泛化 | train walk→test run MAE 32.9–84.1 N·m → **步态身份（duty）是必要输入**，与论文"策略输出 gait logit"设计一致 |

已知简化（记录在案）：水平推进力 F_x=0（滑行+弹跳模型）；ankle 用 CoP 杠杆臂
而非完整 3 连杆 IK；"crawl"与多速度同轴冗余（τ=S·f(φ,d), S=v·T）。

## 3. 力矩闭环失败阶梯（TO06–TO17，负结果链）

| 试什么 | 存活 | 教训 |
|---|---|---|
| TO06：SRB 力矩前馈 τ_SRB+PD | 1.6–9.5s | SRB（2D 单刚体 36kg）与 43-DOF G1 差太远；**τ=Jᵀf 是接触力不是电机力矩**（TO09 定量） |
| 方向 A-ID：ID 力矩前馈 | 2.5s（站住不前进） | ID 来自带位置控制器的重放，非自洽规划力矩 |
| TO11：正确 τ_clean=M q̈+C q̇+g−Jᵀf 前馈 + 轻 PD | **3.58s**（0.82m） | 正确力矩方向对、优于 SRB，但运动学合成的运动物理上不能自洽维持 |
| TO15：2D NMP 力矩（动力学自洽的 2D 步态） | 3.1s | 2D→3D 缺口（无踝/3D/臂）使自洽性只对 2D 模型成立 → **自洽规划力矩须从接近 43-DOF 的模型算** |
| TO16/17：固定偏移 aux / 力矩级 RL aux | 中性（0.81 vs 0.82m） | 论文 aux 正向的前提是前馈本身稳定走（四足 SRBD）；G1 管道不满足 → **方向 ① 完整收束** |

## 4. 八假设系统消除（TO18–TO22，2026-08-16）

统一失败签名：开局跟踪良好（jerr 0.005）→ 第 1 秒 CoM 滞后（vx 0.02 vs 计划
0.32）→ 膝/踝静差 0.2–0.5 rad（力矩从不饱和）→ 永双支撑蹲蹭 ~0.7–0.8m →
第 ~7 步后仰倒塌。

| # | 假设 | 实验 | 判定 |
|---|---|---|---|
| TO18 | PD 目标拉站姿与前馈互搏 | PD 跟踪规划轨迹 q_gait(φ)+速度前馈 | 排除（3.5–3.9s） |
| TO19 | 运动质量（动力学一致性） | NMP 轨迹跟踪（±DC 锚定） | 排除且更差（1.8–2.7s） |
| TO20 | 缺 CoM 稳定反馈 | stance 踝 CoM PD 扫参 | 排除 |
| TO21 | CoM 轨迹无倒立摆动力学 | LIPM 周期轨道（解析初值 ξ̇0≈v 验证） | 排除（3.6s，CoM 第 1 秒即滞后） |
| TO22 | 踝自由度从未规划 | **平脚踝公式 ankle_pitch=−hip_pitch−knee** | 修好 0.47 rad/周期踝误差但同样 3.4s 倒 → 排除 |
| TO22b | 前馈力矩本身错 | ff_scale 0 / −1 | 排除（纯 PD 同样蹲塌） |
| TO22c | PD 增益不足/踝太弱 | kp_scale 2–3×；踝 boost 3–5× | 部分归因但不解（踝 boost 存活 4.9s 但位移崩） |

**根因定性**：合成步态是运动学/简化动力学参考，关节空间 PD + 前馈不构成浮基
系统的反应式稳定——缺任务空间 WBC/QP 层（CoM-ZMP + 躯干姿态 + 摆动脚任务 +
接触/力矩约束）或落脚点自适应。**对照**：SONIC 解码器步态数据被同增益 PD 稳定
跟踪（数据自带平衡极限环）——**数据步态自带稳定器，合成步态不带**；这也再次
解释论文 SRBD+PD 只在四足成立（支撑裕度大）。

## 5. 可复用发现清单（比负结果更值钱的部分）

1. **正确自洽力矩公式：τ_clean = qfrc_inverse − qfrc_constraint**（完整逆
   动力学减接触 Jᵀλ）——调和 SRB（τ=Jᵀf=只接触→塌）与 A-ID（缺−接触→站住
   不前进）两个负结果。
2. **`mj_inverse` 对浮基 G1 双腿同驱返回错误力矩**（hip +154 vs 手动 −3.0，
   50× 误差且左右不对称）→ 正确路由 = 手动 `M@qacc + qfrc_bias −
   qfrc_constraint`（M=mj_fullM）。
3. **平脚踝规划公式 ankle_pitch = −hip_pitch − knee**（残差 <0.01 rad）——
   此前所有合成步态数据的踝通道恒定，是系统级遗漏。
4. 真实 G1 腿几何/惯量定量：髋 0.657m、大腿 0.3406、小腿 0.30、踝 0.056；
   腿惯性 hip 力矩 ~40–46 N·m 与 SRB 静态 stance 接触（25–76）同量级——
   "无质量腿"SRB 漏掉的主要分量。
5. IK 符号约定：hip_pitch=−θ_h、knee=θ_k（FK 往返 ankle 误差 0）。
6. LIPM 周期轨道解析初值法（2×2 线性系统，ξ̇0≈v 验证通过）。

## 6. 产物索引

- 脚本（全部现役，见 SCRIPT_MAP §4）：`srb_to.py`、`srb_to_torque.py`、
  `train_torque_decoder_srb.py`、`eval_torque_srb.py`、`kinematic_gait_id.py`、
  `foot_gait_id.py`、`train_torque_decoder_gait.py`、`eval_torque_gait.py`、
  `eval_torque_nmp.py`、`planar_biped_model.py`、`nmp_biped.py`、
  `train_aux_rl.py`、`probe_full_id_torque.py`、`lipm_gait_id.py`。
- 数据（服务器 `apt_g1/outputs/`）：`srb_to_torque_v1.npz`、
  `torque_gait_data.npz`（603 样本）、`nmp_biped_gait.npz`。
- 全身 TO 侦察报告：`docs/g1_fullbody_trajectory_optimization_roadmap.md`。

## 7. 若重启

- 正路 A：QP-WBC 反应式稳定层（TO23 已证方向正确，后续见 WBC 报告——
  诚实 QP 下加宽脚掌 8.52s、真实窄脚 1.96s，横向欠权限是物理边界）。
- 正路 B：**接受"RL 当稳定器 + TO 步态作参考轨迹/前馈叠加"**（E45–47 已证
  冻结解码器 + 从零 RL 可行；TO 数据可充当更丰富的参考源）。
- 全身/腿级 TO（非 SRB）仍是"力矩级 aux 正向"的必要前置（roadmap §1c）。
