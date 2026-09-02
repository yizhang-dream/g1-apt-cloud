# apt_g1 脚本索引（SCRIPT_MAP）

> 【层位：代码侧轴入口，1 脚本 = 1 行】主轴同粒度层：`refine-logs/tracker/`
> 系列文件（L3 Run 台账·事实源，`EXPERIMENT_TRACKER.md` 为总索引）｜
> ↓更细：脚本源码与文件头注释（含 `_archive/` 归档）。
> 生成于仓库整理阶段（2026-08-13）。逐文件标注每个脚本的角色、用途与对应实验。
> 实验号对照见 `refine-logs/EXPERIMENT_TRACKER.md` 与 `HANDOFF/02_EXPERIMENT_HISTORY.md`。

## 分类标记

| 标记 | 含义 |
|---|---|
| **CANONICAL** | 现行/最终版，复现核心结果所需，保留在顶层 |
| **ARCHIVE** | 被取代的旧版本或已判死路的探索性脚本，已移入 `_archive/`（可恢复） |
| **FORK** | `server_*` 服务端分叉版本，与非 server 版同源但独立演进，原位保留 |
| **DEV** | 诊断/冒烟/数值校验小工具，原位保留 |
| **MODULE** | 被其它脚本 import 的库模块（非入口） |

---

## 1. `apt_g1/` 顶层 —— CANONICAL（保留在顶层）

| 脚本 | 角色 | 用途 | 对应实验 |
|---|---|---|---|
| `train.py` | 入口 | MuJoCo 平坦地 APT-RL 训练主入口 | R 系列 |
| `evaluate.py` | 入口 | 对保存的 APT policy 做确定性 MuJoCo rollout | 通用 |
| `export_reference_tokens.py` | 工具 | 用官方 encoder mode-0 导出参考运动 SONIC token（**2026-08-14 修复 anchor 朝向 offset：584→601，漏了 root_z/anchor_single 的 17 维零填**） | 数据基础 |
| `replay_reference_track.py` | 评测 | **官方参考跟踪闭环复刻**：每步读实时基座四元数在线编码 encoder 观测→token→冻结解码器→关节目标，稳定追踪参考动作（squat/kick/lunge 等，全程不倒） | 动作库扩展 |
| `planner_sonic.py` | 评测 | **官方三模型全栈复刻**（规划器→encoder→decoder）：规划器 ONNX 输入 4 帧 qpos+模式命令→输出未来 qpos 轨迹→FK 转 motion→encoder→token→decoder；27 种 LocomotionMode 已解锁关键 8 模式（idle/slow_walk/walk/run/stealth/squat/kneel/crawl 全栈闭环 fall=None） | 动作库扩展 |
| `terrain_generalize_test.py` | 评测 | **地形泛化测试（goal 第 ④ 部分）**：8 模式 × {平地, rough 0.08} 对比，输出 fall / adv / h_rms / jp_rms（姿态跟踪误差），直接度量冻结解码器"平地→粗糙"泛化 | MQ08 |
| `planner_closed_loop.py` | 评测 | **官方 10Hz 闭环规划器复刻**：每 5 步用 live qpos 重出轨迹（context=4×当前 live 状态），支持运行中切换步态模式 + 粗糙度触发(walk→stealth)；证明重规划是地形泛化机制 | MQ09 |
| `closed_loop_sweep.py` | 评测 | **闭环盲重规划的地形振幅扫描**：均匀凸包地形(无平坦中心) × 振幅 0.08–0.20 × 3 种子，量化官方 planner 盲重规划的真实泛化边界(~0.12–0.14)，发现对地形 seed 高度敏感 | MQ10 |
| `closed_loop_levers.py` | 评测 | **闭环"杠杆"测试**：height 命令(弱) + 步态模式 walk/stealth/crawl 在边界 amp 0.14 × 3 种子；发现 crawl 3/3 不倒、walk 0/3 → 步态模式是真杠杆 | MQ11 |
| `srb_to.py` | 数据生成 | **G1 双足 SRB 轨迹优化器（CasADi）**：复刻 APT 论文"Impulse scale-based TO"（2D 单刚体 q=(x,z,θ) + 半正弦 GRF + 动量守恒 Eq.1 + 周期成本 Eq.2 + IPOPT 求解），产出周期性 CoM 轨迹 + GRF；**TO04 起支持 duty factor d**（walk d=1.0 / run d=0.5 带腾空相）+ 垂直速度 zd0 周期项（`grf_amp(d)`、`integrate_step`） | TO 数据 |
| `srb_to_torque.py` | 数据生成 | **SRB TO → 2 连杆腿 IK → 关节力矩**：把 SRB 的 CoM+足底反力映射成 hip/knee 角 + 力矩（τ=Jᵀf 静态力矩臂），产出 (state, torque) 数据（IK 往返自检 err≈0）；TO04 起 roll_out 支持 (v,T,d) 并积分腾空相；TO05 加 ankle 力矩（CoP 杠杆臂 L_HEEL/L_TOE） | TO 数据 |
| `train_torque_decoder_srb.py` | 训练 | **SRB TO 力矩解码器可学性**：MLP(sinφ,cosφ,v,d)→(hip,knee,ankle 力矩)，跨 walk+run 两步态 + 6 速度；报告 per-gait MAE + 跨步态泛化（train walk→test run）；ankle 用 CoP 杠杆臂模型 | TO 数据 |
| `eval_torque_srb.py` | 评测 | **SRB TO 力矩开环回放（G1 MuJoCo 力矩闭环）**：τ = sign·τ_SRB(2Hz 相位钟) + kp·(zero-token q_des − q) − kd·q̇ 施加到 sagittal 关节（act 0/3/4 与 6/9/10）；TO06 负结果——SRB 力矩太简化不驱动 G1 走路 | TO06 |
| `drive_exp3.py` | 数据采集 | 脚本化官方闭环采集缺失的 walk 方向 | D 系列 / exp3 |
| `build_exp3_dataset.py` | 数据构建 | 合并 exp1+exp2+exp3 raw → `exp_all3`（68,093 步） | D026 |
| `recover_torque_data.py` | 工具 | 为已录 SONIC 闭环数据恢复 PD 力矩标签 | 方向 A |
| `recover_id_torque.py` | 工具 | 用 mj_inverse 重放相位路由器恢复逆动力学力矩 | 方向 A |
| `train_torque_decoder.py` | 训练 | 训练论文式力矩解码器 (phase+cmd → 12-d 腿力矩) | 方向 A |
| `eval_torque_paper.py` | 评测 | MuJoCo 闭环评测论文式力矩控制 | 方向 A |
| `train_phase_router_v9.py` | 训练 | **v9 相位路由器**：从 exp_all3 重建（19 命令组） | D-蒸馏最终 |
| `train_token_vae_e27.py` | 训练 | **E27 相位条件化 token VAE** | E27 |
| `train_token_vae_e31.py` | 训练 | **E31 训练 SpeedPhaseTokenVAE**（速度条件化，3 档速度 bin） | E31 |
| `train_token_vae_e35.py` | 训练 | **E35 训练 DirSpeedPhaseTokenVAE**（方向条件化，8 档方向 bin） | E35 |
| `train_token_vae_e37.py` | 训练 | **E37 方向解耦 VAE**（对抗 dir_head z→8，adv 3.0 + 类平衡 CE） | E37 |
| `train_token_vae_e39.py` | 训练 | **E39 双解耦 VAE**（dir_head z→8 + speed_head z→3 双对抗头） | E39 |
| `train_token_vae_e43.py` | 训练 | **E43 快区加权方向解耦**（fast_extra 2.0，快 bin 行 3× 挤压） | E43 |
| `probe_vae_disentangle.py` | 分析 | E37 解耦探针：fresh 线性分类器 z→8 方向 vs 多数类 | E37 |
| `probe_vae_e39.py` | 分析 | E39 双探针：fresh 分类器 z→8 方向 + z→3 速度 vs 多数类 | E39 |
| `probe_vae_e39_bins.py` | 分析 | **per-bin 方向泄漏探针**（按速度 bin 分组，`[out_dir]` 参数可探 e43） | E40 归因 |
| `probe_fastbin_data.py` | 分析 | **数据侧探针**：快 bin 数据的方向分布（直行占比/左右平衡） | E40 归因 |
| `eval_mjlab_fwd.py` | 评测 | **mjlab 从零策略原生任务评测**（自家 sim、60s 直行命令、[seed steps noise video]） | M-FROM0 |
| `replay_render_mujoco.py` | 渲染 | **MuJoCo 3D 真实模型 offscreen 渲染**（回放 Isaac rollout npz → mp4，MUJOCO_GL=egl） | 渲染管道升级 |
| `plot_latent_cmp.py` | 分析 | **E27–E30 对比图**（产出 `outputs/latent_cmp.png`） | E30 |
| `plot_paper_figures.py` | 分析 | **汇总图表六件套 fig1–fig6**（对照阶梯 / 速度-直行 Pareto / 地形形状边界矩阵 / 规划器线 / TO 战役 / E48 残差），数据优先读服务器评测 JSON、缺失项嵌 TRACKER 台账数字并标 J/T 来源，产出 `outputs/figs/` | 文档补全轮（2026-08-27） |
| `animate_skeleton.py` | 渲染 | **2D 骨架动画**（已降级为调试辅助，被 `replay_render_mujoco.py` 取代） | 渲染 |
| `router_fallback.py` | 评测 | 相位路由器的稳定性门控命令解析（回退表） | 优先级 2 |
| `flat_battery_fallback.py` | 评测 | 带 StableResolver 的全命令空间平坦地 battery | 优先级 2 |
| `switch_marathon_fallback.py` | 评测 | 60s+ 命令切换马拉松（经 StableResolver） | 优先级 2 |
| `interp_router_test.py` | 评测 | 连续潜空间：相位插值原型读取测试（v9 路由器） | 方向 B |
| `oracle_walk_bins.py` | 评测 | 新 walk 方向数据的 oracle 上限检查 | D028 |
| `eval_battery_v9.py` | 评测 | **v9 battery**：per-group 相位路由器目录闭环评测 | D-蒸馏最终 |
| `make_depth_dataset.py` | 数据 | 生成本地 depth → 特权 elevation 数据集（P2-lite） | 感知 |
| `train_depth_student_gru.py` | 训练 | P2-lite v2 深度学生（CNN+GRU+BPTT，最新版） | 感知 |
| `train_perception_distill.py` | 训练 | 感知蒸馏 demo（论文 stage 4 机制） | 感知 |
| `make_rough_xml.py` | 工具 | 构建本地粗糙地形 MJCF（heightfield） | 地形 |
| `make_terrain_fig.py` | 工具 | 地形实验汇总图（survival vs noise） | 地形 |
| `rough_render.py` | 渲染 | MuJoCo 粗糙地形路由器评测 + 视频（本地） | 地形 |
| `render_reel_v9.py` | 渲染 | v9 路由器 highlight reel（含新 walk 方向） | 渲染 |
| `rough_sweep_smooth.py` | 评测 | v9 walk 在平滑本地 hfield 的鲁棒性扫描 | 地形 |
| `stress_test.py` | 评测 | 60s+ 压力测试（蒸馏 PhaseRouterEncoder，无弹力带） | 优先级 2 |
| `perturb_eval.py` | 分析 | 扰动上限：oracle token + k 维偏 1 级的闭环存活 | 分析 |

## 2. `apt_g1/` 顶层 —— ARCHIVE（已移入 `_archive/`）

| 脚本 | 归档原因 |
|---|---|
| `train_phase_router.py` | 被 `_v9` 取代（相位路由器系列初版） |
| `train_phase_router_v2.py` / `_v21` / `_v23` / `_v4` / `_v5` / `_v8` / `_v8c` | 相位路由器演进中间版，最终采用 `_v9` |
| `eval_battery_v2.py` / `_v21` / `_v23` / `_v4` / `_v5` / `_v6` / `_v7` / `_v8` | battery 演进中间版，最终采用 `_v9` |
| `train_distill.py` / `train_distill2` / `train_distill3` / `train_distill4` | BC token 回归（D003），闭环 20-30x 复合误差，被相位路由器取代 |
| `eval_final.py` / `eval_final_v2.py` / `eval_final3.py` | 旧原型 battery，被 `eval_battery_v9` 取代 |
| `knn_eval.py` | kNN 记忆蒸馏初版，保留最新 `knn_eval2.py`（motion-matching） |
| `render_reel_local.py` / `render_reel_local_v6.py` | 旧版/v6 版 reel，保留 `render_reel_v9.py` |
| `rough_sweep.py` / `rough_sweep_slow.py` | 旧粗糙扫描，保留 `rough_sweep_smooth.py` |
| `flat_battery.py` | 无 fallback 的平坦命令审计，保留 `flat_battery_fallback.py` |
| `train_token_vae.py` | 早期 token VAE，被 `train_token_vae_e27.py` 取代 |
| `train_vae_lite.py` / `eval_vae_lite.py` / `train_token_seq_vae.py` | TVAE-lite / 序列 VAE 尝试，均未成功 |
| `train_depth_student.py` | 深度学生初版，保留 GRU 版 |
| `build_v6.py` | v6 专用构建，已过时 |
| `train_phase_ar.py` / `train_apt_phase.py` / `train_router.py` | 被取代的训练尝试（相位自回归 / APT-phase / 路由器蒸馏 v2） |
| `train_knn_mlp.py` | kNN 重标签 + MLP，死路 |
| `train_dagger_slow.py` | slow_fwd 的 DAgger-lite，死路 |
| `proto_variants.py` | 边缘组原型变体调参，探索性 |
| `stress_isolate.py` | walk_back 隔离测试，保留 `stress_test.py` |
| `build_closed_cycles.py` | 闭合周期数据（D 系列），闭合误差 0.00000 但无益 → 死路 |
| `eval_distill.py` | BC 蒸馏闭环评测（已判失败） |
| `eval_closed_router.py` | 闭合周期路由器重评测（死路） |
| `eval_apt_aux.py` | MuJoCo 端 APT aux 闭环评测（R 系列已终止）；Isaac 端等价物见 `isaac/eval_apt_isaac.py` |
| `motion_dataset.py` | （原 `data/`）孤儿源码，无 importer；移入归档以纳入版本控制 |

> `knn_eval2.py` 保留在顶层（kNN 最新版），但其结论（kNN 记忆蒸馏原理可行）已被相位路由器超越，仅供对照。

## 3. `apt_g1/isaac/` —— Isaac Lab 训练栈（全部原位保留）

| 脚本 | 角色 | 用途 |
|---|---|---|
| `__init__.py` | MODULE | 包初始化 |
| `apt_flat_env.py` | MODULE | Isaac Lab DirectRLEnv 平坦地 APT 环境（G1） |
| `apt_flat_env_vanilla.py` | MODULE | Vanilla RL 基线环境（无 SONIC 先验，E9/E11 对照） |
| `batched_router.py` | MODULE | 向量化相位路由器 encoder |
| `elevation_map.py` | MODULE | 特权局部 elevation-map 观测 |
| `sonic_decoder_torch.py` | MODULE | 纯 torch 重实现的 SONIC 解码器（ONNX→torch） |
| `sonic_decoder_isaac.py` | MODULE | Isaac Lab 批量化 SONIC 解码器（ONNX Runtime） |
| `terrain_cfg.py` | MODULE | 地形配置辅助（plane/rough/stairs/stones/discrete + **2026-08-15 G0 新增 `rough_paper`（论文形状：对称±噪声、0.2m 粗格）/ `rough_sym`（对称±噪声、0.1m 格，坑 vs 格子解耦对照）**） |
| `token_window_vae.py` | MODULE | token VAE 三件：**E27 PhaseTokenVAE + E31 SpeedPhaseTokenVAE（+速度条件）+ E35 DirSpeedPhaseTokenVAE（+方向条件）**，冻结解码器供 RL |
| `decft_policy.py` | MODULE | **E44 解码器微调策略**：E39 z头 → 冻结 VAE → token → **可训练 SONIC 解码器** → 29-d 关节目标动作（PPO 评分梯度直达解码器 + 官方解码器漂移正则） |
| `ppo_core.py` | MODULE | 向量化 PPO（含论文式训练附加项；E44 增加 `decoder_ft` 分支与 `decoder_reg_coef`） |
| `train_apt_isaac.py` | 入口 | 训练 APT（相位路由器先验 + aux）策略 |
| `eval_apt_isaac.py` | 入口 | A/B/C/D 评测 |
| `rollout_log_joints.py` | 入口 | **无相机 rollout → npz**（base 位姿 + 29 关节角，SONIC order，供 `replay_render_mujoco.py` 渲染） |
| `eval_fast.py` | 入口 | 守护式评测（只跑请求的 A/B/C/D 段） |
| `render_walk.py` | 渲染 | 从 APT Isaac 环境渲染短行走视频 |
| `inspect_decoder.py` | DEV | 检查发布版 SONIC 解码器 ONNX 图 |
| `parity_decoder.py` / `parity_layers.py` / `parity_onnx2torch.py` | DEV | ONNX↔torch 解码器数值一致性校验 |
| `check_isaac.py` / `smoke_isaac.py` | DEV | Isaac venv 导入检查 / 环境冒烟 |
| `dbg_path.py` | DEV | 诊断 sys.path / PYTHONPATH |
| `server_apt_flat_env.py` | **FORK** | `apt_flat_env.py` 的服务端分叉（同源，body 已分叉） |
| `server_train_apt_isaac.py` | **FORK** | `train_apt_isaac.py` 的服务端分叉 |
| `server_eval_apt_isaac.py` | **FORK** | `eval_apt_isaac.py` 的服务端分叉 |

> **`server_*` 三件**与非 server 版 docstring 逐字相同但 body 差异显著
> （125/86/160 行 diff），是部署到服务器后独立演进的副本。交接包的 run-command
> 引用的是非 server 版（`train_apt_isaac.py` 等）。**哪套为"正统"未判定**，
> 待用户确认；在此之前原位保留两套。

## 4. 库模块（`sonic/` `encoder/` `envs/` `policies/`）

| 模块 | 内容 |
|---|---|
| `sonic/` | `sonic_wrapper.py`（SONIC 封装）、`token_vae.py`、`token_seq_vae.py`、`apt_manager_env.py` |
| `encoder/` | `phase_router_encoder.py`（蒸馏相位路由器）、`phase_ar_encoder.py`；导出 `PhaseRouterEncoder` |
| `envs/` | `g1_flat_env.py`、`mujoco_g1_flat_env.py`（MuJoCo G1 平坦环境） |
| `policies/` | `apt_policy.py`、`phase_aux_policy.py` |

## 5. 配置（`configs/`，23 个 yaml）

平坦地主线的历代配置。CANONICAL 为 `flat_g1_walk_noband.yaml`（最佳冻结零 token 行走）
与 `flat_g1_reference_aux*.yaml`（参考 token + aux 系列）。其余为各种尝试
（jointvae / seqvae / skill / vae16 / residual / ref_band_anneal 等），
保留供历史复现，不单独归档。

## 4. TO 力矩线脚本（TO01–TO22，2026-08-14/16 增补登记）

| 脚本 | 角色 | 用途 | 对应实验 |
|---|---|---|---|
| `srb_to.py` / `srb_to_torque.py` | 数据生成 | SRB TO（CasADi）→ 2 连杆 IK → 力矩 | TO01–TO05 |
| `kinematic_gait_id.py` / `foot_gait_id.py` | 探针 | 关节空间/足空间运动学步态 + 手动全模型 ID 力矩（`M@qacc+qfrc_bias−qfrc_constraint`） | TO08/TO09 |
| `train_torque_decoder_gait.py` | 训练 | phase→τ_clean 解码器；**2026-08-16 起新暴露 `compute_gait_full`（含规划轨迹 Q/Qd）** | TO10 |
| `train_torque_decoder_to36.py` | 训练 | **TO37b（2026-08-30）**：TO36 dircol 解族→条件化解码器 MLP(sinφ,cosφ,v,phase)→6D τ（MJCF 符号，B 门 sign 映射内嵌）；留一速度泛化 + 全量 MAE，对照 TO10 基准 | TO37 |
| `eval_torque_gait.py` / `eval_torque_nmp.py` | 评测 | 力矩闭环冒烟；**2026-08-16 起支持 `--track-gait`（PD 跟踪规划轨迹）与 CoM 踝反馈** | TO11/TO15/TO18–TO20 |
| `eval_torque_srb.py` | 评测 | SRB 力矩前馈闭环（负结果） | TO06 |
| `train_aux_rl.py` / `probe_full_id_torque.py` | 训练/探针 | 力矩级 aux RL（负结果）/ τ_clean 符号探针 | TO17/TO07 |
| `planar_biped_model.py` / `nmp_biped.py` | 模块 | 平面 5 连杆模型 / 直接配点 NMP | TO12–TO14 |
| `wbc_gait.py` | 评测 | **TO23–TO33（现役主线）**：QP-WBC + LIPM/质心 MPC 参考层 + 全机制（梯形侧摆/捕捉落脚/加宽支撑/姿态正则/CoP 盒/knee-guard/H 任务）+ 诊断套件；TO32 修复锥约束空洞 bug 后加宽脚掌首破 8s，`--foot-halfy`+`APT_SCENE`（`foot_gait_id.py`）支持脚宽参数化 | TO23–TO33 |
| `lipm_gait_id.py` | 评测 | **TO21/TO22（新）**：LIPM 周期轨道（解析初值）→ IK（含平脚踝规划 `ankle=−hip−knee`）→ 全模型 ID → 闭环（`--ff-scale/--ankle-kp-boost/--stab-*` 扫参 + 逐关节诊断） | TO21/TO22 |
| `to36_leg_to_drake.py` | DEV | **TO36 v1 solve + B/C 门宿主**：v1 dircol（load/solve，D2 判死留作负结果对照）+ **D5 起 B/C 门现役**：`world`（Stage A，.venv_drake：hybrid foot 解→世界系 81 样本/相，骨盆解析加度+FK 目标）、`verify`（Stage B，.venv_mjlab：B 门双验证——基座行消去法 ID 解跟/尖 λ，绕 TO08 mj_inverse bug；符号映射 FK 搜索；D5 判定：effort PASS/数字口径 FAIL/支撑链 −5 N·m 归因 = URDF↔MJCF CoM 差 1.4 cm）、`closedloop`（C 门：矢状 τ 前馈+gait PD+bias/stab 归因参数）。两 venv 分段运行，过程坑见 tracker TO36-D5 | TO36 |
| `to36_common.py` | 模块 | **TO36 共享件（2026-08-29 决策 C 案抽出）**：v1 原样迁移的 resolve_model/build_plant/KnotKinematics 等模型构造与工具（D1 行为基线），v1 与 hybrid 共用 | TO36 |
| `to36_hybrid_dircol.py` | DEV | **TO36 现役主线（D3 起）**：hybrid 双相位 dircol 周期步态——pointe（5 体 pin）/foot（7 体全掌 weld+踝，真脚几何 URDF 解析）双模式；分离式 Hermite–Simpson（defect+插值）+ 相位接口 P 映射（foot 图映射含倒装踝双翻转，只在双脚平贴交集精确）+ 刚体冲击（辅助变量 λ）。**2026-08-30 修复两真 bug：#16 接口切片错位（假缝，此前全部收敛解作废）、#17 压缩式 HS 缺中点插值约束（混叠伪解：配点自洽但 knot 间能量漂移 45 J）**。配套**审计验收制**（IPOPT 证书仅供参考：每级过 HS/冲击/接口<1e-6 + 相内能量漂移<2 J + 冲击不产能）+ 重力斜坡同伦（slope_deg）+ --guess-npz 链式热启动 + --retries 混合重启 + --v-cap。**foot 平地刚性 A 门达成**（v 0.318 m/s，drift 1.7 J，审计采纳；F9 解备份 to36_hybrid_gait_F9.npz）；pointe 平地解（v 1.701）产自带洞转录、复核未过被 foot 取代。**D5 膝盒修正（F11）**：原对称 ±2.0 放行膝反屈（F9 映射后超真实限位 [−0.087,2.88] 至 46°，C 门根因）→ 相位感知盒（支撑 [−2.88,+0.087]/摆动 [−0.087,2.88]+踝收紧），初值支撑膝负弯曲 | TO36 |
| `to36_setup_drake_env.sh` | DEV | 服务器 `.venv_drake` 建 env 脚本（无 sudo：`--without-pip`+get-pip 引导 + `pip install drake`） | TO36 |
| `to38_analyze.py` | 评测 | **TO38（2026-08-31）**：双臂配对差分分析——读 `to38{a,b}_eval_*.json`，floor 检查 + 低速带 vx 跟踪误差配对差分 + 三分支判定（决策表在 TO38_PLAN §0/§4）；支持多 ckpt 配对（每臂 best=各自窗口最优） | TO38 |
| `to38_export_ref.py` | 模块 | **TO38（2026-08-31）**：`to36_world_knots.npz` → RL 注入用紧凑 LUT `to38_ref.npz`（M=120：q_ref6/tau_ref6/pitch/z/heel_rel + meta）。**关键：world npz q6 列序按角色排**（[支撑 A/K/H, 摆动 H/K/A]，两相互为镜像、符号按角色）——本脚本做逐相角色→SONIC 重排 + B 门符号 + 周期闭合检查（wrap_gap）。配套注入代码：`isaac/apt_flat_env.py` 的 `to_ref*` cfg（12 维 obs 块 + cmd 门控矢状跟踪 reward + 独立 ψ 时钟）、`train_apt_isaac.py`/`eval_apt_isaac.py` 的 `--to-ref*` CLI | TO38 |
| `to40c_analyze.py` | 评测 | **TO40C（2026-09-01）**：三臂（ctrl/t10/t05）配对差分分析——读 `{arm}_eval_{ck}*_a{cmd}.json`，floor 检查（completed/h_min≥0.6/disp>0.5）+ 门开带 vx 跟踪误差配对差分（δ=0.03 等效边界）+ 路径效率（disp/(v_speed·60s)，<0.5 判绕圈）+ 2×2 交叉注入诊断（{arm}_eval_{ck}_x{on|off}_a0277.json）；判定逻辑见 `refine-logs/TO40C_PLAN.md` §5。配套注入代码：`isaac/apt_flat_env.py` 的 `to_tau*`（kp 从 sim 读取） | TO40C |

## 6. `refine-logs/tools/` —— 文档树工具（2026-08-29 增补登记，非实验代码）

| 脚本 | 角色 | 用途 | 对应实验 |
|---|---|---|---|
| `tree_check.py` | DEV | 实验记录扇出树完整性闸门（仿 mini Biosphere `doc_tree_check.mjs`）：挂树/实存/链接三项检查，任一失败 exit 1；树根与规则见 `refine-logs/README.md` | 文档基建（无实验号） |

## 7. `apt_g1/` artifact 工具（2026-09-01 增补登记，非实验代码）

| 脚本 | 角色 | 用途 | 对应实验 |
|---|---|---|---|
| `gen_tau_dec_mapping.py` | DEV | 生成 Rung 1 treatment mapping artifact（`configs/rung1_tau_dec_mapping.yaml`，**schema v2 全交叉**：7 speeds × {C1,C2} = 14 rows，十二轮 B reopen 后 supersedes v1 恒等物化）：cfg 默认值从冻结 `apt_flat_env.py` 正则提取，bin 算子与冻结代码逐字同源（torch bucketize right=False），同输入重跑 byte-identical；产物 pre-run-only（无 observed/expected 字段），内建 gate A schema 自检；验收 gate 见 `refine-logs/TO40C_PLAN.md` §10.8、B 链状态见 `TO41_RUNG1_IMPL.md` | TO40C→Rung 1 |
| `to41_material_driver.py` | DEV | τ(v) material campaign driver：`validate` 子命令做全门集判定（G1 字段/G2 solver terminal success/G345 审计镜像 `_audit_pass`/G6 NaN-Inf/G7 速度容差 0.02/G8 配置身份），输出两字段 accounting JSON（solver_terminal_status ⊥ material_status）；`run` 子命令待 hot-start source 冻结后启用。规格与状态见 `refine-logs/TO41_RUNG1_IMPL.md` §5 | TO41（Rung 1） |

## 8. `apt_g1/rung1/` —— Mode A conditioning runtime + D independent checker（2026-09-02 登记）

D 阶段执行协议（`refine-logs/TO41_D_DRYRUN_PROTOCOL.md`，FROZEN）的可执行化。
**角色口径（协议 §10.2）**：runtime = **state-changing execution code**；checker =
**read-only audit**（只读取+独立计算，禁改 material/mapping/任何实验状态，
发现问题的唯一出口 = report FAIL → 协议 §7 保险丝）。checker 不 import runtime
（双解析器独立实现，selftest 交叉验证）。产物目录 `apt_g1/outputs/sync/to41_d/`。

| 脚本 | 角色 | 用途 | 对应实验 |
|---|---|---|---|
| `rung1/mode_a_runtime.py` | 入口（**state-changing**） | Mode A 契约 `τ(v,C)=τ(v)` / `C=T_mapping(v,C)` 的执行器：mapping lookup（(v,arm)→condition_id）与 material lookup（v→τ material）**两个独立函数**，先落 immutable execution record 再 decode；CLI `--mode static`（28-cell 配置层覆盖核对，本机可跑）/`--mode execute`（decode-only dry-run receipt×28，**仅 lab-ts**，`--env-tag` 强制口径）。receipt 只含 record 字段，无任何 verdict/PASS/ok 字段（协议 §9） | TO41 D（Rung 1） |
| `rung1/d_checker.py` | 审计（**read-only**） | D independent checker：自带独立解析器读冻结 mapping v2 YAML / source registry / G_DOWN_SPEC §9 availability map，重算哈希（材料文件独立 sha256）、重算 D1（七字段+mode/layout/shapes）/D2（same-τ fingerprint+lineage）/D3A/D3B verdict；receipt schema 封闭 + 全域封禁自报 verdict 字段；禁收 performance 字段（协议 §4）；本机运行恒标 `--env-tag local`，lab-ts D 报告必须 `--materials-root` | TO41 D（Rung 1） |
| `rung1/rung1_selftest.py` | 工具（自测） | checker 逻辑自测（synthetic receipt，永不作为 D artifact）：T0 双解析器交叉一致 / T1 lookup 单元（Mode A 恒等式）/ T2 静态覆盖 28/28 / T3 正例控制 / **Negative A–E**（τ 换 hash→D2+D3B FAIL；condition 错 arm→D3A FAIL；decoder 超参/shape 篡改→D1 FAIL；自报 PASS+实际不一致→仍 FAIL；lineage 缺失→schema FAIL）。dry-run 前必须全绿 | TO41 D（Rung 1） |

## 8b. `apt_g1/rung1/` —— launch sanity（真实 env 接线验证，2026-09-02 三十七轮登记）

L1–L4 纯执行 gate（判据唯一事实源 = `refine-logs/TO41_LAUNCH_SANITY.md`；
D protocol FROZEN 不变）的可执行化：把冻结 Mode A runtime 接入**未改动的**
真实 `apt_flat_env.py` τ 注入/控制路径，28 cell 只测接线、不测性能。
**接线纪律**：零 env 文件改动（env sha256 = mapping `preprocessing_hash`
冻结锚，checker 机械校验）——condition 轴 = 实例级 shadowing `env._vae.decode`
（冻结 bucketize 每步照跑，wrapper 记 natural/applied 双记录）；τ 轴 =
`cfg.to_ref_npz` 既有通道 + 实例级探针 `_to_ref_lookup`（:768 唯一消费点）。
l_checker 不 import launch_sanity/env_wiring（audit 独立性同 D §9）。
产物目录 `apt_g1/outputs/sync/to41_sanity/`。

| 脚本 | 角色 | 用途 | 对应实验 |
|---|---|---|---|
| `rung1/env_wiring.py` | 模块（**state-changing**） | Mode A→env 接线 shim：`ConditionOverrideHandle`（decode 调用点拦截，per-call natural/overridden 双记录）/`TauConsumptionProbe`（τ 消费点拦截，per-call digest+buffer 快照）/`canonical_array_sha256`（冻结数组身份规范化，checker 侧独立实现交叉验证）。只产 record，无 verdict 字段；Rung 1 compute 的 eval 侧接线届时复用同一实现（conformance 随之继承） | TO41 L（Rung 1） |
| `rung1/launch_sanity.py` | 入口（**state-changing**） | 28-cell 真实 env 接线驱动：`--mode static`（配置层覆盖核对，本机）/`--mode execute --cell-index N`（**仅 lab-ts，per-cell 进程模型**——每 cell 独立 python 进程 + AppLauncher，receipt 落盘后 os._exit 硬退出，服务器 bash 循环 28 次；新建真实 AptFlatG1Env（TO40C ctrl/t10 配方 × τ(v) 材料 × e39 vae，两臂唯一 cfg 差={to_tau}），jitter_and_reset + 恒定 cmd_vx 每步重申 + 中段强制 episode boundary + 零动作驱动真实 step 循环，不调 env.close()（sim.clear_instance/挂死 gotcha））。冻结锚 preflight（env/vae/arch 三源哈希 + 7 材料可达，Isaac 启动前 fail-fast，每 cell 进程都跑） | TO41 L（Rung 1） |
| `rung1/l_checker.py` | 审计（**read-only**） | L1–L4 independent checker：L1 τ material consumption（buffer 哈希 = checker 独立 np.load 重算 + 4 cell/v Mode A fingerprint + ON 消费/OFF 零注入）/ L2 override persistence（逐 call natural vs applied + checker 重算自然 bucketize + boundary 后 persistence）/ L3 ON/OFF isolation（两臂 cfg diff == {to_tau}）/ L4 28-cell receipt（冻结枚举 + 冻结 env 源哈希锚 + decoder 签名表）。schema 封闭 + 封禁自报 verdict 字段 + 禁收 performance 字段；本机恒 `--env-tag local`（报告文件名显式 not_L_artifact） | TO41 L（Rung 1） |
| `rung1/l_selftest.py` | 工具（自测） | sanity 自测（synthetic receipt + mock env，永不作为 L artifact）：T-L0 双解析器交叉一致 / T-L1 wiring handle mock 单测（override 逐调用/契约 fail-fast/digest 确定性）/ T-L2b canonical hash 双实现交叉 / T-L3 合成正例全 PASS / **Negative A–E**（τ buffer 换 hash→L1+L3 FAIL；boundary 后 applied 回退→L2 FAIL；两臂混入额外 cfg 差→L3 FAIL；receipt 缺件→全级联 FAIL；自报 verdict→schema FAIL）。execute 前必须全绿 | TO41 L（Rung 1） |
