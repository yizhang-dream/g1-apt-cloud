# 产出索引（本地 + 服务器）

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

## 2. 服务器（cvgluser@10.16.52.225，~/ros2_data）

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
| 128 envs | `isaac_stress_128env` | `e23_stress_128.log` |

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
