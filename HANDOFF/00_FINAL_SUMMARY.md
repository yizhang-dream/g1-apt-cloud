# 项目最终交接总结（2026-08-14 定稿；2026-08-27 增量补遗见 §9）

> 【层位 L0｜结论卡，1 条结论 = 1 句话】↓ `HANDOFF/README.md`（L1 总览与
> 钻取导航）｜证据终点：`refine-logs/tracker/` 系列文件（L3 事实源；
> `EXPERIMENT_TRACKER.md` 为总索引）。
> 一页纸版。详细证据链见 `refine-logs/FINAL_REPORT.md`（**收束定稿**：两个核心问题
> 最终回答 + 对照阶梯 + 负结果 + 剩余方向）、
> `refine-logs/STAGE_SUMMARY_2026-08-13.md`、`refine-logs/EXPERIMENT_TRACKER.md`。

## 1. 项目目标

在 Unitree G1 上，用 NVIDIA GEAR-SONIC 官方 token 数据 + 冻结解码器，复现
Science Robotics 2026 APT-RL 管线（TO 数据 → 潜空间 → RL 增强 → 地形泛化），
回答两个问题：**① 蒸馏先验是否必要？② 论文机制在替代管道里是否成立？**

## 2. 两个核心问题的最终答案

**① 蒸馏（路由器）先验是否必要？—— 管道内不必要但显著提速；解码器才是关键先验。**
- E27（有解码器、无路由器）：19m 可走；冻结先验（有路由器）：47m → 路由器不必要但 ≈2.5× 提速。
- E9/E11（直出关节、无解码器）：0/3 立即倒 → SONIC 解码器/token 流形才是管道内关键先验。
- mjlab 从零（无任何 SONIC 组件，官方配方 + 官方算力）：原生 sim 60s 直行 44.5–48.0m、
  地形 0.08 无悬崖（1/3 走满 60s）→ 足够算力+正确配方下，蒸馏非普适必要；蒸馏的价值
  在当前设备（3060、64 envs）与数据条件下是实打实的可用下限。

**② 论文机制在替代管道是否成立？—— 部分成立。**
- ✅ 成立：2Hz 门控（E13）、可学习步态选择（E20c gate-only）、感知复原（P1/P2-lite corr 0.965）、
  对抗解耦（E37/E39，自研机制，方向+速度双解耦是潜空间线最优动作表征）。
- ⚠️ 中性：连续潜空间（E26 无损不增益）。
- ❌ 不成立/受限：aux 在位置 token 管道下永远是破坏源（E2–E22b，根因缺力矩级解码器 +
  自洽 TO/ID 数据）；地形 0.08 在蒸馏路径（相位 VAE/路由器，无 planner 10Hz 重规划）
  全方案不可破（〔归因修正 2026-08-14，见 MQ09〕：悬崖是"蒸馏路径边界"非"冻结解码器
  边界"——同一解码器 + planner 重规划在 MuJoCo rough 0.08 walk≈flat；mjlab 对照也证明
  至少部分是 SONIC/Isaac 特有，非 G1 通病）；潜空间线速度上限 ≈0.46 m/s。
  **〔补 2026-08-14，TO06〕** 力矩级前馈（SRB TO 力矩）也不成立：SRB 是 2D 单刚体、
  与 43-DOF G1 差太远，τ_SRB+PD 短暂移动即倒（对比 ID 力矩至少站住）→ 论文 SRB TO
  四足专用，人形需全身/腿级 TO。

## 3. 对照阶梯（潜空间线核心数字，A 60s 直行）

| 方案 | 位移 | 说明 |
|---|---|---|
| 冻结先验（路由器+解码器） | 47m | 满速，管道最优 |
| E26 相位偏移 | 46m | 无损但中性 |
| E27 latent→VAE→SONIC | 19m | 无行为先验可走 |
| E31 速度条件化 | ~10m（vx 0.535） | 破速度上限但系统性左转 |
| E35 方向条件化 | 16.3m | 修漂移、速度回落 |
| E37 方向解耦 | 21.55m（直行 0.97） | 快且直（0.37 档） |
| E38 方向解耦+高激励 | 19.4m（直行 0.81） | 方向解耦不是 0.535 的钥匙 |
| **E39 双解耦（方向+速度）** | **24.65m（vx 0.417，直行 0.98）** | **潜空间线历史最佳（甜点）** |
| E40 双解耦+更高激励 | 24.40m（vx 0.456） | Pareto 前沿另一端 |
| E43 快区加权解耦 | 13.5m（直行 0.65） | 过度挤压损伤流形；E40 漂移主因在解码器侧 |
| **E47 从零+E39 VAE+heading** | **23.8m（vx 0.42，直行 0.944）** | **从零（无 warm start）逼近 E39 walk 先验；A/B/C/D 全过** |
| E9/E11 vanilla | 0m | 无 SONIC 失败 |
| mjlab 从零（原生 sim） | 44.5–48.0m | 非 SONIC 对照锚点 |

## 4. 地形结论

- **〔2026-08-15 G0 修订〕"0.06 通过"是形状依赖的**：论文形状 rough（对称 ±0.06、
  0.2m 粗格）上 **E47 与 E39 双双 0/12 全倒**（5–31s）；对称 ±0.06、0.1m 格对照
  同样 0/12 → **坑（负障碍）是唯一必要难点变量，格子大小无关**。蒸馏路径地形边界
  按形状重排：只凸 0.06 过 / **有坑 ±0.06 全倒** / 只凸 0.08 全倒（E48 残差关曾
  1/3 存活 26.8m 但 E48c 未复现，合计 1/9 为边缘事件——0.08 悬崖仍成立）。
- SONIC 蒸馏路径（Isaac harness，盲走）：只凸 0.06 全过；**只凸 0.08 全方案
  0/9–0/12 全倒**（冻结先验、E39、E42 地形训练、E47 从零最优 0.06 12/12 但
  0.08 0/12——均不可破；训练地形匹配无效）。〔归因修正 2026-08-14，
  见 MQ09〕0.08 悬崖是"蒸馏路径（无 planner 10Hz 重规划）"的边界，非解码器本身——
  同一解码器 + planner 重规划在 MuJoCo rough 0.08 上 walk≈flat（3.38m≈3.39m/6s）。
- mjlab 从零（原生 sim，盲走）：0.06 平滑退化（27–49m）；0.08 无悬崖（15–45m，
  1/3 走满 60s）→ **0.08 悬崖是蒸馏 SONIC 路径（+Isaac 地形配方）特有，非 G1 通病**。

## 5. 视频库（本地 `apt_g1/outputs/`，共 13 个）

早期线：`e29/e31/e35_mujoco.mp4` ｜ 双解耦平地：`e39/e40_mujoco.mp4`
地形：`e39t_mujoco.mp4`（0.06）、`e39t08_mujoco.mp4`（0.08 绊倒）、
`e42t_mujoco.mp4` ｜ 三连对比：`e39_cmp_mujoco.mp4` ｜
从零线：`e47_mujoco.mp4`（2026-08-27 补渲染，E47 从零+E39 VAE+heading）｜
非 SONIC 对照：`mjlab_fromscratch.mp4`（平地）、`mjlab_fromscratch_r06/r08.mp4`（地形）｜
早期 MuJoCo：`rough_v9/v6*.mp4`

数据图表（2026-08-27 补生成，`apt_g1/outputs/figs/`，本地与服务器各一份）：
`fig1_latent_ladder.png`（对照阶梯）、`fig2_speed_straight_pareto.png`
（速度-直行 Pareto）、`fig3_terrain_boundary.png`（地形形状边界矩阵）、
`fig4_planner_line.png`（MQ08/10/11 规划器线）、`fig5_to_battle.png`
（TO 线存活战役）、`fig6_e48_residual.png`（残差通道）；另有早期
`latent_cmp.png`（E27–E30）、`terrain_summary.png`。

## 6. 产物位置

- 服务器：checkpoint `GR00T-WholeBodyControl/outputs/isaac_e{37,38,39,40,42,43}_*`；
  E44 系列 `isaac_e44{a,b,c,p1,p1b,p2,p2b,v2a,v3a}_*`；从零线
  `isaac_from0_01`（vanilla 对照）/ `isaac_from0_dec_01`（E45）/
  `isaac_e45_e39_from0`（E46）/ `isaac_e47_heading`（E47，ckpt it_500）；
  E48 `isaac_e48_residual` / `isaac_e48c_resfreeze`；
  VAE `apt_g1/outputs/token_vae_e{27,31,35,37,39,43}/`；
  TO 数据 `apt_g1/outputs/{srb_to_torque_v1,torque_gait_data,nmp_biped_gait}.npz`；
  图表 `apt_g1/outputs/figs/`；评测 JSON `apt_g1/outputs/isaac_eval_e*.json`
  + `terr_e*.json`；mjlab `unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-08-14_00-52-58/`
  （model_6499.pt + policy.onnx）。
- 本地：`apt_g1/outputs/` 视频 13 个 + figs/ 图表 7 张 + 评测 JSON 已 scp 回。
- 文档：`HANDOFF/README.md`、`FINAL_REPORT.md`（含 §9 阶段 2 全史）、
  `STAGE_SUMMARY_2026-08-13.md`（方向 D E28–E43 全链条）、
  `TO_TORQUE_LINE_REPORT.md`（TO01–22 收束）、`WBC_BRINGUP_REPORT.md`
  （TO23–35 收束）、`EXPERIMENT_TRACKER.md`（台账）。

## 7. 未做/受限（诚实清单）

- 论文 aux 正向价值：**〔2026-08-14 更新，TO01–TO06〕** 自洽 SRB TO 力矩已能自建
  （跨 walk+run × 3 关节可学 MAE 0.57–1.1 N·m），但 **TO06 证明 SRB（2D 单刚体）力矩
  太简化、不驱动 43-DOF G1 走路**（短暂移动即倒）→ 论文 SRB TO 是四足专用简化，
  人形 aux 正路仍需 G1 全身/腿级 TO（非 SRB）。
- 论文级千级并行复现：3060 上限 1024–2048 envs（mjlab 已验证可行），4096 需 24GB GPU。
- 真机部署/sim-to-real：未做（无真机条件）。
- 地形 0.08（SONIC 管道）：冻结解码器假设下不可破，除非解冻/替换解码器。

## 8. 交接指引

读 `refine-logs/FINAL_REPORT.md`（收束定稿，10 分钟掌握全貌）→ 需要数字查
`EXPERIMENT_TRACKER.md` → 需要视觉看 §5 视频。全部实验已闭环，无 RUNNING 任务。

## 9. 2026-08-15 ~ 08-17 增量补遗（E44 / G0 / E48 / MQ07–12 / TO 线收束）

> 本文主体定稿于 08-14；本节按其后三天的增量补齐（详见
> `HANDOFF/02_EXPERIMENT_HISTORY.md` 阶段 8–12 与 `HANDOFF/README.md` §6）。

1. **MQ07–MQ12（规划器复刻线）**：官方三模型全栈闭环复刻成功（8 模式
   fall=None）；**MQ09 归因修正——0.08 悬崖是"蒸馏路径（无 planner 10Hz
   重规划）"的边界，不是冻结解码器本身**（同一解码器 + 重规划 walk≈flat）；
   但 MQ10 跨 seed 后盲重规划真实边界仅 ≈0.12–0.14 且对地形实现高度敏感；
   MQ11/12：步态模式（crawl 3/3 @0.14）是真杠杆，切换 transition 脆弱，
   复刻 ADAPTING 状态机是"感知→选步态"的剩余前置。
2. **E44（解码器微调，稳健负结果）**：v1/v2/v3/两阶段全变体失败——PPO 梯度
   确实能移动解码器，但任何程度微调都把直行破坏成"打转+2s 倒"（SONIC 快走
   固有偏置被重新激活）；冻结解码器对照组 3/3 走 12.75–13m 证明 harness 无
   bug。**冻结解码器是承重墙**。
3. **E45–E47（从零 + 冻结解码器，已入 §3 阶梯）**：vanilla 从零蹲蹭作弊
   （假 vx、disp=0）vs E45 真走路 14m；E46 换 E39 双解耦 VAE 从零破速度
   天花板（0.418）；**E47 +轻 heading = 从零线最优（0.42/直行 0.944/23.8m，
   A/B/C/D 全过）**，无须 warm start、无须 walk 先验。
4. **G0（地形结论修订，已入 §4）**：论文形状（对称 ±0.06 有坑、0.2m 格）
   上 E47/E39 双双 0/12 全倒——**坑（负障碍）是唯一必要难点变量**，
   "0.06 达论文上限"仅对只凸形状成立。
5. **E48/E48c（全关节残差，跨 3 配置稳健负结果）**：残差开 = 全地形破坏
   （it_200 抽查证明从早期即坏）；先冻后放 = 放开即被"冲刺+摔倒计速度"漏洞
   占据。128-envs/3060 规模下文献解法 2 关闭；E48-noaux 那次 0.08 存活
   （26.8m）未复现，合计 1/9 为边缘事件，0.08 悬崖仍成立。
6. **TO 线收束（TO01–TO35）**：数据管线成功（自洽 SRB TO 力矩可学 MAE
   0.57–1.1 N·m）但力矩闭环全灭于 ~3.5s；TO18–22 八假设消除定根因"缺浮基
   反应式稳定层"；TO23–28 QP-WBC 把存活推到 8.3s（机制验证 + 15 bug 清单）；
   TO31–32 发现并修复锥约束空洞 bug 后诚实重排：真实窄脚 1.96s、加宽脚
   8.52s（踉跄孤点非稳态）——**横向欠权限是物理边界；合成步态无稳定流形，
   稳定流形要么来自数据（SONIC）要么用 RL 学（E45–47）**。报告：
   `TO_TORQUE_LINE_REPORT.md` + `WBC_BRINGUP_REPORT.md`。
7. **图表补全（2026-08-27）**：`figs/fig1–fig6` 六张汇总图（对照阶梯 /
   速度-直行 Pareto / 地形形状边界 / 规划器线 / TO 战役 / E48 残差），
   `plot_paper_figures.py` 生成，数据优先读服务器评测 JSON。
8. **腿级 TO 线收束（TO36，2026-08-29~30，Drake dircol，5 人日时间盒）**：
   三门 DoD——**A 门达成**（F11b：平地+刚性冲击+膝限位可行周期解
   0.277 m/s，审计验收制采纳）；**B 门双验证执行**（43-DOF MuJoCo 逆
   动力学基座行消去法复核：摆动链一致 PASS，支撑链 −5 N·m 归因 =
   URDF↔MJCF 整机 CoM x 差 1.4 cm，已定量；100–300 N·m 数字口径系
   TO06 时代估计，慢拖步诚实量级 6.7–33 N·m）；**C 门负结果**（开环
   回放 15 稳层配置穷尽最优 1.84 s——TO 的 2D 相位解不是 43-DOF 真接触
   系统的不变流形，capture 落足反馈 +40% 是正向线索但达标需重规划级
   工程）。自洽 TO 力矩数据 (q,v,u,接触) 已可用于力矩解码器主线；
   闭环行走须 RL 稳定器（E45–47 已证可行）叠加 TO 参考。报告：
   `LEG_LEVEL_TO_REPORT.md`（真 bug #16 接口假缝/#17 混叠伪解）。
