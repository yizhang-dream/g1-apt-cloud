# APT-RL × SONIC × G1 复现实验综合报告（2026-08-12）

## 1. 任务与结论速览

- 目标：在 MuJoCo 上按 APT-RL 论文补齐"控制器线 + 数据线"，之后转 Isaac Lab
  继续探索论文结合方向，回答"是否需要更好的数据/网络保证泛化性"，并验证论文
  "先给特权地图、后蒸馏感知"的顺序。
- **一句话结论**：蒸馏先验（冻结相位路由器 + SONIC 解码器）是当前管道最优
  控制器（平坦地面全项通过、粗糙地形盲走也有鲁棒性）；论文的 aux / elevation /
  步态选择机制在逐一补齐后均未带来"前进 + 地形存活"兼得的正向价值，根因是
  替代管道缺少力矩级解码器、结构化连续潜空间、千级并行与强制任务奖励——这些
  正是论文与主流开源实现（unitree_rl_mjlab）的生效条件。

## 2. 论文方法对照表（论文用了什么，我们怎么处理）

| 论文阶段 | 论文方法 | 我们的实现 | 结论 |
|---|---|---|---|
| 数据 | 2D SRBD 轨迹优化，动量守恒周期轨道，含力矩 | SONIC 官方闭环数据（68,093 步），无力矩、非周期 | 替换；周期与力矩是两个实质差异 |
| 表示学习 | TVAE 统一潜空间（z∈ℝ16，KL 0.1）| 相位路由器（PCA 相位 + 40-bin 原型）；VAE-lite 试过 | VAE-lite 后验坍缩/闭环失败；显式相位+原型最稳 |
| RL | latent 动作 + aux(12) + gait logit，Isaac Gym 千级 env | phase/aux/gate + elevation，64 envs | aux/gate/elevation 均无正向价值；相位锚定可存活但倒走 |
| 混合控制 | τ=τ_dec + PD(q_default−q+a·a_aux)，kp=80 | 位置目标 q_des + Isaac Lab PD | 无前馈力矩，物理语义不同 |
| 感知 | 教师特权地图 → 策略；学生 depth+LIDAR→GRU 蒸馏 | 特权 9×9 elevation patch（E16）；学生复原地图（P1） | 地图无 latent/gait 通道时不产生价值；蒸馏机制验证成立 |
| 奖励 | velocity tracking（Rudin 2022）+ style terms | 5 项最小奖励 + progress/anti-stop | 最小奖励允许 idle/倒走坍缩 |
| 并行 | 数千 env | 64 envs（RTX 3060） | 规模差 ~64 倍 |

## 3. MuJoCo 两条线（已完成，见 MUJOCO_APT_LOG.md）

- 控制器线：7 个 RL 变体全部劣于 aux=0 基线 → 判定单进程 MuJoCo PPO 基础设施
  不足，终止。
- 数据线：闭合周期 token 数据集构建（闭合误差 0.00000 达成）→ 重训路由器复测
  无益 → 漂移根源在控制器层缺闭环修正，终止。

## 4. Isaac Lab 阶段

### 4.1 机制矩阵（E1–E14，平坦地面）

- 蒸馏先验必要：vanilla RL 800/2000 iters 均 0/3 立即倒；先验版全项通过。
- aux 在平坦地面无正向收益（超速 + 偏航漂移）；修正版 2Hz 门控（E13）显著
  减少漂移（位移 3–11m → 34–37m），是论文门控机制的正向验证。
- 离散 token 相位选择学不出振荡器（E3 系列）；TVAE 平滑流形缺失是根因。

### 4.2 地形/感知/动作通道（E15–E20）

| 变体 | rough 0.06 | rough 0.08 | 前进质量 |
|---|---|---|---|
| 冻结先验 noaux | 3/3 | 0–1/3 | 42–46m |
| E15 盲 aux | 2/3 | 0/3 | 1.7–9.7m |
| E16 aux+elevation | 0/3 | 0/3 | - |
| E17 gate+aux+elevation | 0/3（或站住） | 0/3 | - |
| E17b +progress | 0/3 | 0/3 | 能移动 |
| E18 phase 直控 | 0/3 | 0/3 | 能移动 |
| E19 phase 锚定 | **3/3** | **3/3** | 倒走/站住 |
| E19c +aux 正则 | 1/3 | 0/3 | 平坦前进 8s |
| E20 +anti-stop | 0/3、0/3、2/3@0.08（爬行） | 平坦 3/3（快但画圈） | 速度/存活冲突 |
| E20c gate+anti-stop | 0/3（倒走） | **0.06 3/3 前进 38–43m；0.08 一 seed 59s** | gate 学对（选先验组），aux 破坏 |

**E20 系列最终结论**：
1. anti-stop 方向压力能让 gate 头收敛到"选择先验自己的 walk_fwd 组"（aux=0
   时行为与冻结先验一致）；这是唯一"学习不破坏先验"的组件。
2. aux 通道从 E2 到 E20c 始终是破坏源：任何学习到的 12 维关节偏移修正都让先验
   退化（超速/偏航/倒走/跌倒）。论文 aux 的正向价值依赖力矩级解码器（PD 稳定
   下的力矩修正），SONIC 位置目标 + 关节偏移的替代无法复现。
3. 结论收敛：**现有管道最优 = 冻结先验 + （可选）anti-stop 训练的 gate 选择
   先验组；继续提升必须换力矩级数据/解码器、千级并行或真机部署**。

**E20c gate-only 全量验证**（2026-08-12）：平坦 A 3/3（vx 0.71–0.77）、B 12/12、
C 3/3、D 3/3 —— 与冻结先验全项持平；rough 0.06 A 3/3（38–43m）、B 10/12、
C 3/3、D 3/3。可学习步态选择（论文 gait logit 类比）在正确奖励压力下被验证为
"无损组件"；aux 是唯一破坏源。结果文件：
`outputs/e20c_gateonly_{flat,rough06}_BCD.json`。

**跨地形扩展**：rough 0.06 三 terrain seed **9/9 全部前进存活**（38–45m）；
rough 0.08 三 seed 0/9 完成但存活 21–59s（= 冻结先验边缘区，无增益无损失）。
结果文件：`outputs/e20c_gateonly_n{0.06,0.08}_s{1,2}.json`。

### 4.3 数据/网络泛化（v8/v8c/v9/VAE-lite）

- v8 共享相位回归：失败（相位误差 1.6–2.3 rad）。
- v8c 共享回归解码器：walk_fwd 3/3，其余差于离散原型。
- exp3 补采 walk 6 方向 + v9 重建：walk_fwd/back 3/3；新方向受教师上限限制
  （官方 token 回放 95–270 步内倒）。
- VAE-lite：后验坍缩；最佳 checkpoint 闭环 walk 0/3。
- **最优结构**：显式 PCA 相位 + 离散原型查表（原型是流形精确点，闭环不漂移）。

### 4.4 感知（P1）

- 学生（粗 3×3+噪声 感知代理 + 基座状态）→ 特权 9×9 地图：corr 0.954。
- 机制验证：感知能复原地图（论文 stage-4 同构）；真实 depth/LIDAR 需渲染环境。

## 5. 对用户问题的最终回答

1. **是否需要更好的数据？** 是，但覆盖只是必要不充分。补全 walk 方向后仍受
   教师（SONIC 官方 token + 本 MuJoCo env）上限约束；要突破需换带力矩标注的
   数据（论文 TO 方案）或换解码器/环境。
2. **是否需要更好的网络？** 是，但必须是"带正则的结构化潜空间"，朴素条件回归
   （v8c）与 VAE-lite 都会离开流形；显式相位 + 原型是当前数据下的最优结构。
3. **先给地图、后蒸馏感知？** 顺序正确且已验证：特权地图已接入（E16），感知
   复原机制已验证（P1）；但地图要产生价值，必须先有能消费它的 latent/gait
   通道 + 强制任务奖励（E19 的相位锚定是第一步，anti-stop 正在补第二步）。

## 6. 开源参考（佐证方向）

- unitree_rl_mjlab（Apache-2.0）：地形 raycast 高度扫描直接进策略 + 显式
  phase 观测 + 丰富 task/style 奖励 + 4096 envs。
- Gaitor/VAE-Loco：连续可解释统一步态潜空间（与论文 TVAE 同思路）。
- Isaac-Velocity-Flat-G1-v0/v1：Isaac Gym 版 G1 速度跟踪，v1 做过 sim-to-real。
- CALM/ASE：latent skill + 对抗先验。

## 7. 遗留与建议

- **硬件**：RTX 3060 12GB 限制 64 envs；要复现论文级结果需 ≥4096 envs
  （如 4090/A6000/多卡）。
- **数据**：力矩标注数据（TO 或系统辨识）是解锁力矩级解码器的前提。
- **解码器**：SONIC 位置 token 解码器对 off-manifold 输入敏感；力矩级解码器
  （PD 稳定）才能容忍潜空间插值。
- **渲染/视频**：无显示服务器无法出 3D 视频；本地有显示环境后可补。
- **文档 "G 200 T"**：未找到，待用户提供路径。

## 7b. 视频产出（2026-08-12 本地渲染）

服务器无显示无法渲染，改在本地 Windows（Python 3.13 + mujoco 3.11 + CPU 推理）
渲染成功，每个 42 秒（2100 帧 @50fps）：

- v9（exp_all3 新方向数据）高光：`apt_g1/outputs/distill_v9/v9_reel_local.mp4`
- v6（旧路由器，新方向 fallback）对照：`apt_g1/outputs/distill_final/v6_reel_local.mp4`

两者场景一致（idle/walk_fwd/walk_back/walk±45°/walk+135°/jump/slow_walk），
可直接对比"新数据路由器 vs fallback 行为"。脚本：
`apt_g1/render_reel_local.py`、`render_reel_local_v6.py`。

## 7c. MuJoCo 粗糙地形 + 力矩级解码器（2026-08-12 夜，详见 DATA_GENERALIZATION_LOG.md）

### MuJoCo 本地 hfield 鲁棒性曲线（修复 MuJoCo 3.11 hfield 碰撞语义后）

- 关键修复：`elevation` 是内联数字非文件；`base_z` 不参与碰撞（碰撞面 =
  geom.pos.z + size[2]×data）。修复前机器人穿地（ncon=0），早期"全倒"为伪结果。
- 本地地形（0.4m 粗格、σ=1.2、无坡度上限）比 Isaac 同标称更陡：0.02/0.03/0.04
  的 p99 坡度 ≈ 0.08/0.12/0.16，max ≈ 0.11/0.16/0.22。
- 曲线（walk 前进，3 seeds）：0.00–0.02 3/3（15.8–17.1m）→ 0.03 2/3（v9）→
  0.04 0/3 → 0.06 0/3（v6 1/3 完成）→ 0.08/0.10 0/3。按坡度对齐后与 Isaac
  阈值（0.06 3/3）一致 → 差异是生成参数不是控制器。
- **平台对照闭合**：平滑地形（coarse 1.0m、σ=3.0，p99 坡度≈Isaac）下
  0.06 **3/3 完成**（14.5–17.0m）、0.08 0/3——与 Isaac 完全一致。
- 视频：陡地形 `rough_v9.mp4`（12.5s 倒）、`rough_v6.mp4`（30.9s 倒）；
  平滑地形 `rough_v9_smooth.mp4` / `rough_v6_smooth.mp4`（走满 20s）；
  结果 `outputs/rough_mujoco_sweep.json` / `rough_mujoco_smooth.json`。
- 慢走组：0.06 上靠站住刷存活（vx≈0.01），无实际通过能力
  （`rough_mujoco_slow.json`）。

### 力矩级解码器（论文混合控制的最直接测试，结论为负面）

1. 从 exp_all3 离线重算 PD 力矩标签（14,633 行）；phase+cmd→力矩 MLP
   val RMSE 18.76 N·m（≈预测零 = 不可约反馈误差）。
2. 论文式闭环（tau_dec + PD(q_default)，aux=0）：平坦 3/3 存活 20s 但
   **vx≈0.03 不前进**；粗糙 0.06 62–82 步倒。混合式（token PD + tau_dec）
   平坦 63–73 步倒（双倍反馈失稳）。
3. **结论**：闭环 PD 力矩标签不是论文 TO 那种"规划前馈"，无法替代 token
   位置路径；力矩通道无法用 PD 标签廉价补上。这从机制上解释了 E2–E20c
   aux 全负面的根源。

### E21a（进行中）：gate + anti-stop + 特权地图，rough 0.06 训练

把地图给 gate 通道（而非 E16 的 aux 通道），检验"地图经步态选择通道产生
价值"（论文/unitree_rl_mjlab 的主流接法）。若 0.08 上比 E20c（0/9）有提升，
则地图+gate 机制成立。

**E21a 结果**：gate-only 3/3（37–43m，= 先验）；gate+aux 2/3 慢走（vx 0.14）；
0.08 上 gate-only 0/3（同 E20c）——**特权地图不能突破 token 流形上限**。
另发现 E20c 训练本就带地图观测（checkpoint obs=172），E21a 与 E20c 的差异
主要在 anti-stop 参数；E21a gate 在未见台阶上迁移更差（2/3 慢爬）。

### E21b：先验×离散地形（Isaac 内置 4 类）

| 地形 | 冻结先验 A60s | E20c gate-only | E20c gate+aux |
|---|---|---|---|
| stairs（4–8cm） | 3/3，18–22m | — | — |
| stairs_hi（8–14cm） | 3/3，14–20m | 3/3，9–17m | 3/3 但 vx≈0.04 站住 |
| discrete（5–10cm ×10） | 3/3，48m | — | — |
| stones（0.3–0.5m 间距） | 0/3（2 步内倒） | 0/3 | — |

**结论**：先验自带小台阶/小障碍跨越能力；垫脚石是明确盲区（需精确落脚）；
gate-only 保持先验行为，aux 依旧破坏（台阶上站住不爬）。

### P2-lite：深度图像 → 特权高程补丁（MuJoCo 本地，第 4 阶段机制验证）

单目深度（128×96）→ 9×9 高程补丁回归：CNN 单帧 corr 0.74 → +GRU/BPTT
0.87 → +跨地形数据（3 档振幅、3077 帧）**0.965（MAE 0.0265m）**，与几何
反投影上界（0.0275m）持平，接近 P1 语义代理（0.0085m）。结论：感知复原
地图完全可行，数据量+时序结构是关键；真实单目深度弱于语义代理，印证论文
加 2D LIDAR 的动机。图：`outputs/depth_student_ladder.png`、
`depth_student_p2lite.png`。

## 8. 文件索引

- 日志：`refine-logs/{MUJOCO_APT_LOG, ISAAC_APT_LOG, DISTILL_EXPERIMENT,
  DATA_GENERALIZATION_LOG, EXPERIMENT_TRACKER, APT_PROJECT_SUMMARY}.md`
- 代码：`apt_g1/`（isaac/、train_phase_router_v8/v8c/v9.py、train_vae_lite.py、
  train_perception_distill.py、drive_exp3.py、eval_*.py）
- 数据：`apt_g1/data/{exp_all, exp_all3, exp1_raw, exp2_raw, exp3_raw}/`
- 结果：`apt_g1/outputs/{terr_*.json, eval_battery_v8/v9.json,
  eval_vae_lite_ema0.3.json, percept_meta.json, terrain_summary.png}`
