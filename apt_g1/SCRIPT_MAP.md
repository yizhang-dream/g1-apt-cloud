# apt_g1 脚本索引（SCRIPT_MAP）

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
| `export_reference_tokens.py` | 工具 | 用官方 encoder mode-0 导出参考运动 SONIC token | 数据基础 |
| `drive_exp3.py` | 数据采集 | 脚本化官方闭环采集缺失的 walk 方向 | D 系列 / exp3 |
| `build_exp3_dataset.py` | 数据构建 | 合并 exp1+exp2+exp3 raw → `exp_all3`（68,093 步） | D026 |
| `recover_torque_data.py` | 工具 | 为已录 SONIC 闭环数据恢复 PD 力矩标签 | 方向 A |
| `recover_id_torque.py` | 工具 | 用 mj_inverse 重放相位路由器恢复逆动力学力矩 | 方向 A |
| `train_torque_decoder.py` | 训练 | 训练论文式力矩解码器 (phase+cmd → 12-d 腿力矩) | 方向 A |
| `eval_torque_paper.py` | 评测 | MuJoCo 闭环评测论文式力矩控制 | 方向 A |
| `train_phase_router_v9.py` | 训练 | **v9 相位路由器**：从 exp_all3 重建（19 命令组） | D-蒸馏最终 |
| `train_token_vae_e27.py` | 训练 | **E27 相位条件化 token VAE** | E27 |
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
| `terrain_cfg.py` | MODULE | 地形配置辅助 |
| `token_window_vae.py` | MODULE | E27 相位条件化 token VAE（冻结解码器供 RL） |
| `ppo_core.py` | MODULE | 向量化 PPO（含论文式训练附加项） |
| `train_apt_isaac.py` | 入口 | 训练 APT（相位路由器先验 + aux）策略 |
| `eval_apt_isaac.py` | 入口 | A/B/C/D 评测 |
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
