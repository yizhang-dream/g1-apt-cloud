# 产出索引（本地 + 服务器）

> 【层位 L4｜原始产物索引，1 条目 = 1 个文件】↑ `refine-logs/tracker/`
> 系列文件（L3 Run 台账·事实源）｜≈同层：服务器 `outputs/`（eval JSON / figs / mp4，即本索引指向的实体）。
> 约定：本地盘符 `C:\Users\zyz\Documents\gr00t` 简写为 `C:\...\gr00t`；
> 服务器根 `D:\GR00T-WholeBodyControl` 是仓库宿主（只读参考）；
> 服务器 `/home/cvgluser/ros2_data` 简写为 `~/ros2_data`。

## 1. 本地（C:\Users\zyz\Documents\gr00t）

| 路径 | 内容 |
|---|---|
| `tmp\pdfs\paper.txt` / `scirobotics.adz7397.pdf` | APT-RL 论文全文 |
| `refine-logs\` | 全部实验日志与总结（见 01 文档第 5 节） |
| `apt_g1\data\exp_all3\` | 主数据集 68,093 步（token/mode/speed/angle_bin/proprio/cmd） |
| `apt_g1\data\torque_id\` | ID 力矩数据（27k 行，方向 A） |
| `apt_g1\isaac\` | Isaac 代码：env、PPO、train/eval、terrain_cfg、token_window_vae |
| `apt_g1\sonic\` | SONIC 封装与 VAE 原型（token_vae.py、token_seq_vae.py） |
| `apt_g1\outputs\distill_v9\` | **v9 蒸馏路由器**（19 命令组 phase/proto g0..g18 + phase_norm + phase_meta） |
| `apt_g1\outputs\distill_final\`（v6 旧物，仅 14 组 + 混 v6 产物）、`distill_v8\` 等 | 路由器历代版本 |
| `apt_g1\outputs\torque_decoder_id\` | ID 力矩解码器（方向 A） |
| `apt_g1\outputs\flat_battery_fallback_v9.json` | 回退表 24/24 命令评测 |
| `apt_g1\outputs\interp_router_flat.json` | 相位插值 MuJoCo 评测 |
| `apt_g1\outputs\isaac_eval_e23/e24/e25*/e26/e27.json` | E 系列评测 JSON（scp 回本地） |
| `apt_g1\outputs\train_log_e23.json`、`e23_train.log` 等 | 训练日志副本 |
| `apt_g1\train_token_vae_e27.py` | E27 VAE 训练脚本 |

## 2. 服务器（`ssh lab-ts`，与旧地址 cvgluser@10.16.52.225 同一台机器，~/ros2_data）

| 路径 | 内容 |
|---|---|
| `apt_g1\` | 主实验代码（训练/评测/数据/蒸馏脚本，与本地 apt_g1 对应） |
| `apt_g1\data\exp_all3\` | 主数据集（68,093 步） |
| `apt_g1\isaac\` | Isaac 代码（env/ppo/train/eval/token_window_vae） |
| `apt_g1\outputs\` | 评测 JSON、训练日志（e2x_*.log、terr_e*.json、isaac_eval_e*.json） |
| `apt_g1\outputs\token_vae_e27\` | E27 VAE（vae.pt/pca.npz/z_walk.npy/meta.json） |
| `GR00T-WholeBodyControl\outputs\` | **Isaac checkpoint 目录**：isaac_e22a/e22b/e23/e24/e25/e26/e27、isaac_stress_128env、token_vae_e27 |
| `unitree_rl_mjlab\` | 官方从零配方仓库（对照基线） |
| `unitree_rl_mjlab\logs\rsl_rl\g1_velocity\2026-08-13_11-03-40\` | mjlab 训练日志 + checkpoint（model_*.pt 至 900，policy.onnx） |
| `.venv_isaac\` | Isaac Lab 2.1.0 + torch 2.5.1 训练环境 |
| `.venv_mjlab\` | mjlab 1.2.0 + mujoco-warp 3.5.0 + torch 2.13 环境 |
| `xr_teleoperate\`、`Humanoid\`、`groot_transfer_bundle_20260722\` | 官方资产/工具 |
| `proj2605.md`、`cmd\` | 用户笔记/命令记录 |
| `/tmp/run_apt_isaac.sh` | Isaac 运行包装脚本（source venv + PYTHONPATH + cwd） |

## 3. 关键 checkpoint 映射（服务器 GR00T-WholeBodyControl\outputs\）

| 实验 | checkpoint 目录 | 评测 |
|---|---|---|
| E22a/E22b | `isaac_e22a_aux` / `isaac_e22b_aux_reg` | `apt_g1/outputs/isaac_eval_e22a.json` 等 |
| E23 | `isaac_e23_phase_interp` | `isaac_eval_e23.json` |
| E24 | `isaac_e24_phase_antistop` | `isaac_eval_e24.json` |
| E25 | `isaac_e25_phase_anchor` | `isaac_eval_e25*.json`（含消融） |
| E26 | `isaac_e26_phase_only` | `isaac_eval_e26.json`、`terr_e26_*.json` |
| E27 | `isaac_e27_latent` | `isaac_eval_e27.json` |
| E28–E30 | `isaac_e29_klwalk/isaac_e30_klwalk_h1` | `isaac_eval_e28-30*.json`（含 KL 扫描） |
| E31/E32/E34 | `isaac_e31_speedvae` / `_speedvae_heading` / `_speedvae_yawdr` | `isaac_eval_e31/e32/e34.json` |
| E35 | `isaac_e35_dirvae` | `isaac_eval_e35.json` |
| E36 | `isaac_e36_fastdir` | `isaac_eval_e36.json` |
| E37 | `isaac_e37_disentangle` | `isaac_eval_e37.json` |
| E38 | `isaac_e38_fastdir` | `isaac_eval_e38.json` |
| E39 | `isaac_e39_dualdecouple` | `isaac_eval_e39.json` |
| E40 | `isaac_e40_dualdecouple_h` | `isaac_eval_e40.json` |
| E41/E42 地形 | `isaac_e39_dualdecouple` / `isaac_e42_terrtrain` | `terr_e41_n*.json`、`terr_e42_n*.json` |
| E43 快区加权 | `isaac_e43_fastclean` | `isaac_eval_e43.json` |
| E44 解码器微调（v1/v2/v3/p 两阶段/ctrl） | `isaac_e44{a,b,c,p1,p1b,p2,p2b,v2a,v3a}_*` | `isaac_eval_e44*.json` 系列 |
| E45 从零 latent | `isaac_from0_dec_01`（**注意：`isaac_from0_01` 是 vanilla 蹲蹭对照**） | `eval_from0_dec_01.json`（E45）/ `eval_from0_01.json`（vanilla） |
| E46 从零+E39 VAE | `isaac_e45_e39_from0`（目录名带 e45 但实为 E46） | `eval_e45_e39_from0.json` |
| E47 从零+heading | `isaac_e47_heading`（最优 ckpt it_500） | `eval_e47_heading.json`、`eval_e47_BCD.json`、`eval_e47_terrain_*.json`、`eval_e47_{paper,sym}006_s*.json`（G0） |
| G0 论文形状地形（E39 侧） | 复用 `isaac_e39_dualdecouple` it_800 | `eval_e39_paper006_s{0,1}.json` |
| E48/E48c 残差 | `isaac_e48_residual` / `isaac_e48c_resfreeze` | `eval_e48*.json` 系列（见 §6） |
| mjlab M-FROM0 | `unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-08-14_00-52-58/` | `model_6499.pt` + `policy.onnx`（总 6500 iters） |
| 128 envs | `isaac_stress_128env` | `e23_stress_128.log` |

## 3b. 视频产出（本地 `apt_g1\outputs\`，服务器 EGL 渲染后 scp 回）

- `e29/e31/e35_mujoco.mp4`：早期线（慢稳 / 快但漂移 / 方向条件化直行）
- `e39_mujoco.mp4`、`e40_mujoco.mp4`：双解耦策略平地行走（无跌倒）
- `e39t_mujoco.mp4`：**E39 在 rough 0.06 地形行走**（真 hfield 地形渲染）
- `e39t08_mujoco.mp4`：**E39 在 rough 0.08 行走 15s 后绊倒**（蒸馏路径能力边界，非解码器本身〔归因修正见 MQ09〕）
- `e42t_mujoco.mp4`：E42（地形训练策略）在 0.06 行走
- 渲染管线：`rollout_log_joints.py`（--terrain + 高度窗口导出）→
  `replay_render_mujoco.py`（hfield 注入 + 接触对齐自检）

## 4. 数据资产格式（exp_all3）

- `token.npy`：(68093, 64) float32，值域 [-0.875, 0.8125]（SONIC FSQ token）
- `mode.npy`：0=idle，1=slow，2=walk，17=jump，18=stealth
- `speed.npy`：-1/0/0.2/0.6
- `angle_bin.npy`：8-bin 方向索引
- `proprio.npy`：(68093, 930) 10 帧历史；`cmd.npy`：(68093, 14)

## 5. 参考代码位置（D:\GR00T-WholeBodyControl）

- ElasticBand：`gear_sonic\utils\mujoco_sim\unitree_sdk2py_bridge.py`
- SONIC 模拟栈：`gear_sonic\utils\mujoco_sim\`（base_sim.py、sensor_server.py）
- WBC 配置：`gear_sonic\utils\mujoco_sim\wbc_configs\*.yaml`（g1_29dof）
- 部署栈（C++）：`gear_sonic_deploy\`
- 文档约定：`docs\source\references\conventions.md`（坐标系/四元数 wxyz）


## 6. 2026-08-15 新增（Gate 0 论文形状地形 + E48/E48c 残差，均在服务器 apt_g1/outputs/）

- Gate 0 评测 JSON：`eval_e47_paper006_s{0,1}.json`、`eval_e39_paper006_s{0,1}.json`、
  `eval_e47_sym006_s{0,1}.json`（结论：论文形状/对称 ±0.06 全倒，坑是唯一难点变量）。
- E48 训练：`GR00T-WholeBodyControl/outputs/isaac_e48_residual/`（ckpt it_50–2000 +
  final + train_log.json；最优 it_700）。
- E48 评测：`eval_e48_{flat,r006_s0,r008_s0,paper006_s0,sym006_s0}.json`、
  `eval_e48b_it200_{flat,r006_s0,r008_s0}.json`。
- E48c 训练：`GR00T-WholeBodyControl/outputs/isaac_e48c_resfreeze/`（关键 ckpt
  it_800=放开残差前的健康基座、it_1000=放开后早期）。
- E48c 评测：`eval_e48c_{it800,it1000}_{flat,r006_s0,r008_s0}.json`。
- 训练日志：`e48_train.log`、`e48c_train.log`；评测链日志同名 `*.log`。

## 7. MQ 规划器复刻线产出（2026-08-14，结果在 TRACKER MQ 段 / STAGE_SUMMARY §10）

| 脚本 | 产出 |
|---|---|
| `planner_sonic.py` | 关键 8 模式全栈闭环（终端输出，无独立 JSON；结论在 TRACKER MQ07） |
| `terrain_generalize_test.py` | 8 模式 × 平地/rough 0.08 对比表（TRACKER MQ08 结论表） |
| `planner_closed_loop.py` | 10Hz 重规划 + 模式切换 + walk2crawl 触发（TRACKER MQ09/MQ12） |
| `closed_loop_sweep.py` | 振幅扫描 0.08–0.20 × 3 种子（TRACKER MQ10 表） |
| `closed_loop_levers.py` | height 探查 + walk/stealth/crawl @0.14 + crawl 高振幅（TRACKER MQ11） |

## 8. TO / WBC 力矩线产出（2026-08-14 ~ 08-17）

| 产物 | 位置 | 说明 |
|---|---|---|
| `srb_to_torque_v1.npz` | 服务器 `apt_g1/outputs/` | SRB TO→IK→力矩数据（TO01–05） |
| `torque_gait_data.npz` | 同上 | 足-空间步态 τ_clean 数据 603 样本（TO10） |
| `nmp_biped_gait.npz` | 同上 | 平面 5 连杆 NMP 步态 q/qd/tau/f（TO13） |
| `to27_sweep.log` … `to30_sweep.log` | 同上 | QP-WBC 扫参日志（TO27–30） |
| `g1_29dof_with_hand_wf{15,20}.xml` | 服务器（模型变体） | 脚掌加宽 f=1.5/2.0 碰撞球变体（TO31） |
| 收束报告 | 本地 `refine-logs/` | `TO_TORQUE_LINE_REPORT.md`（TO01–22）、`WBC_BRINGUP_REPORT.md`（TO23–35） |
| 全身 TO 侦察 | 本地 `docs/` | `g1_fullbody_trajectory_optimization_roadmap.md` |

## 8b. TO36 腿级 Drake dircol 线产出（2026-08-29 ~，服务器 `apt_g1/outputs/`）

| 产物 | 位置 | 说明 |
|---|---|---|
| `to36_hybrid_gait.npz` + `_diag.json` | 服务器 `apt_g1/outputs/` | hybrid foot 现役解（F11 膝限位修正版起；A 门 v 0.318 原版见 F9 备份） |
| `to36_hybrid_gait_F9.npz` | 同上 | F9 解备份（A 门达成版，膝轨迹超真实限位——D5 定性，被 F11 取代） |
| `to36_hybrid_gait_v1_softlanding.npz` | 同上 | 假缝时代软着陆解（bug #16 证据，已作废） |
| `to36_foot_F1…F11.log` | 同上 | F 系列求解日志（F9=A 门、F10=审计拦截伪解、F11=膝盒修正重解） |
| `to36_world_knots.npz` | 同上 | Stage A 世界系 81 样本/相（B 门 verify / C 门 closedloop 共用，骨盆解析加度+FK 目标） |
| `to36_verify_b.json` + `to36_verify_b_samples.npz` + `to36_verify_b.log` | 同上 | B 门双验证判定（逐关节/逐相分解 + 逐样本时序；D5） |
| `to36_closedloop.json` | 同上 | C 门闭环判定与归因诊断（D5） |
| `to37_seed.npz` / `to37_fast56_v0435.npz` / `to37_fast48b_v0678.npz` | 同上 | TO37 速度网格三点解族（0.277/0.435/0.678，全审计 PASS） |
| `to37_decoder.json` / `to37_decoder.pt` | 同上 | TO37b 条件化解码器（单速度压缩 PASS / 跨速度泛化 FAIL） |
| `to38_ref.npz` | 同上 | TO38 RL 注入 LUT（M=120，角色→SONIC 重排 + B 门符号 + wrap_gap=0 meta） |
| `to38a/` / `to38b/` | 同上 | TO38 双臂 ckpt + train_log（主臂 obs 注入 w=0.3 / 对照臂 obs 零块 w=0，128 envs×2000 iters） |
| `to38{a,b}_eval_*.json`（14 组） | 同上 | TO38 配对评测：两臂 {best,final}×低速带 A + best 全 battery；配对差分结论见 tracker/TO.md TO38-eval 行 |
| `to39c/` | 同上 | TO39c（obs-only）关键 ckpt it150/it200/it2000（自云平台 checkpoint 系统迁出，it2000 尾部退化仅存档） |
| `to39_lowband/` | 同上 | TO39 三臂原始 rollout 行（各 31 行）+ c 臂采样训练曲线；云上 /personal 另有 15 个评测 JSON summary（平台内） |

## 9. 汇总图表（2026-08-27 补生成，服务器与本地 `apt_g1/outputs/figs/`）

生成脚本 `apt_g1/plot_paper_figures.py`（数据优先读服务器评测 JSON，缺失项嵌
TRACKER 台账数字并在图内标注 J/T 来源）。

| 图 | 内容 | 支撑结论 |
|---|---|---|
| `fig1_latent_ladder.png` | 潜空间线对照阶梯（E26→E47 + 冻结先验/vanilla/mjlab，A 60s 位移 + vx 标注） | §3 阶梯可视化：E39/E47 双甜点 |
| `fig2_speed_straight_pareto.png` | 速度-直行 Pareto 散点（E27–E47，气泡=位移） | E31 快而漂 vs E39/E47 快且直 |
| `fig3_terrain_boundary.png` | 地形存活矩阵（只凸 0.06/0.08 × 有坑 ±0.06 两种格子 × 6 方案） | G0 修订：坑是唯一必要难点变量；mjlab 无悬崖 |
| `fig4_planner_line.png` | (a) MQ08 步态激进程度退化 (b) MQ10 振幅扫描 (c) MQ11 步态杠杆 | 盲重规划边界 0.12–0.14；crawl 是真杠杆 |
| `fig5_to_battle.png` | TO 线存活战役（A-ID→TO06→TO11/15→TO18–22→TO23–32，含关键 bug 修复标注） | 3.5s 关节 PD 墙 → QP-WBC 8.52s；锥 bug 翻案 |
| `fig6_e48_residual.png` | E48/E48c 残差开/关 × 地形位移对比 | 残差开全地形破坏；0.08 边缘事件未复现 |

早期图表：`latent_cmp.png`（E27–E30，`plot_latent_cmp.py`）、
`terrain_summary.png`（早期地形，`make_terrain_fig.py`）。
视频：`e47_mujoco.mp4` + `e47_rollout.npz`（2026-08-27 补渲染，从零线最优
控制器 E47 平地行走；rollout 用 `isaac_e47_heading/policy_it_500.pt` +
token_vae_e39 双 bin 旗标）。
