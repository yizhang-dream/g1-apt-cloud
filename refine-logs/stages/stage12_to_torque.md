## 阶段 12：TO 力矩线（TO01–TO35，2026-08-14 ~ 08-17）

> 【层位 L2 叶子｜阶段史明细（1 阶段 = 1 文件）】↑ `HANDOFF/02_EXPERIMENT_HISTORY.md`（阶段索引）｜↓ `refine-logs/tracker/` 系列文件（L3 Run 台账）｜≈ 同层：`refine-logs/` 专题日志。

> 详细收束见 `refine-logs/TO_TORQUE_LINE_REPORT.md`（TO01–22）与
> `refine-logs/WBC_BRINGUP_REPORT.md`（TO23–35）。此处为阶段速览。

| 段 | 做了什么 | 结论 |
|---|---|---|
| TO01–05 | 自建 SRB TO（CasADi，复刻论文 Impulse-scale TO）→ 2 连杆 IK → 力矩，duty factor + ankle CoP 扩展 | 数据管线成功：周期 cost ~1e-17，力矩解码器 MAE 0.57–1.1 N·m（vs PD 18.76、ID 4.13）；**步态身份（duty）是必要输入**（跨步态 MAE 32–84） |
| TO06 | SRB 力矩前馈闭环（τ_SRB+PD） | 负结果：短暂移动 1.6–9.5s 即倒——SRB 是 2D 单刚体，论文 SRB TO 四足专用，人形需全身/腿级 TO |
| TO07–10 | 自洽力矩探针：τ_clean = qfrc_inverse − qfrc_constraint；发现 `mj_inverse` 浮基 bug（手动 M@qacc+qfrc_bias 路由）；真实 G1 腿几何/惯量定量（hip 惯性力矩 ~46 N·m） | 正确执行器力矩 ≈±51 N·m；SRB 的 τ=Jᵀf 是接触力不是电机力矩——TO06 根因 |
| TO11 | 正确 τ_clean 前馈闭环 | 3.58s 倒（优于 SRB 2.0s）——正确力矩也救不了运动学合成的运动 |
| TO12–15 | 平面 5 连杆模型 + 直接配点 NMP（动力学自洽步态）→ G1 闭环 | NMP 跑通（里程碑）但 2D→3D 缺口使它反而不如 kinematic τ_clean（3.1 vs 3.58s） |
| TO16–17 | 固定偏移 aux 探针 + 力矩级 RL aux | 双双中性——论文 aux 正向的前提是 SRBD 前馈本身稳定（四足），G1 替代管道不满足，方向 ① 完整收束 |
| TO18–22 | 逐假设消除"~3.5s 倒"的八个便宜解释（PD 目标/运动质量/CoM 反馈/LIPM/踝规划/前馈符号/PD 增益/踝刚度） | 全部排除 → 根因 = **合成步态 + 关节空间 PD 不构成浮基反应式稳定，缺 QP-WBC/落脚点自适应层**（SONIC 数据步态自带平衡极限环是对照）；可复用发现：平脚踝公式 ankle_pitch=−hip_pitch−knee |
| TO23–28 | QP-WBC v1→v2（`wbc_gait.py`）：完整浮基动力学含 qfrc_passive、统一双接触、双支撑窗、姿态正则、真实 CoP 盒、捕捉落脚、梯形侧摆、LIPM-MPC 参考层 | 机制端到端验证（站立完美 6s com_err±0.00）；存活从 3.5s 墙推到 5.9–8.3s；**15 个真 bug 清单**（决定性：动力学等式漏 qfrc_passive，1.1s→8.3s）；踏步/行走未收敛，负结果报告收束 |
| TO29–35 | 重启清单执行：质心角动量 MPC（同配置存活 ×2 至 4.24s）、H 列空间任务（无净增益）、仿真加宽脚掌——**第 15 bug：WBC2 锥约束 16 行上下界 ±1e10 完全空洞** | 修复后全表重排：诚实 QP 真实窄脚 1.96s；加宽脚 f=2.0+匹配 CoP 8.52s（首破 8s 但为踉跄孤点，h_mean 0.563）；TO34/35 邻域 ~15 组全退化 → **"横向欠权限是物理边界"在诚实 QP 下证实；无稳定盆地，行走未启动** |

**TO 线一句话**：自洽 TO 力矩数据可以自建且高度可学，但"合成步态 + PD/WBC"
在窄脚 G1 上到 8.52s 踉跄存活为止——与潜空间线（E45–47：RL 当稳定器 + 冻结
解码器提供稳定流形）互为印证：**稳定流形要么来自数据（SONIC），要么用 RL 学，
不能靠简化模型合成**。

