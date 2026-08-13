# apt_g1/outputs —— 产物说明

> **本目录被 `.gitignore` 排除**：日志、视频、`.npy/.npz/.pt/.onnx` 等重型产物
> 保留在磁盘但不进版本控制。clone 后此目录除本文件外为空。
> canonical 产物在服务器 `~/ros2_data/apt_g1/outputs/` 与
> `~/ros2_data/GR00T-WholeBodyControl/outputs/`，完整索引见
> `HANDOFF/03_OUTPUTS_INDEX.md`。

## canonical 产物（复现核心结果所需）

| 产物 | 含义 | 对应实验 |
|---|---|---|
| `distill_v9/` | **真 v9 相位路由器**（19 命令组 phase_g0..g18 + proto + phase_norm + phase_meta） | D-蒸馏最终 |
| `flat_battery_fallback_v9.json` | 回退表 24/24 命令 3/3×20s 评测 | 优先级 2 |
| `eval_battery_v9.json` | v9 battery 闭环评测 | D-蒸馏最终 |
| `isaac_eval_e*.json` | Isaac E2–E27 的 A/B/C/D 评测（e2/e6/e8/e13/e23/e24/e25*/e26/e27 等） | E 系列 |
| `isaac_e*_train_log.json` | Isaac 各实验训练日志 | E 系列 |
| `terr_*.json` | 地形评测（e15–e20c，n0.04/0.06/0.08/0.10） | 阶段 3 地形 |
| `torque_decoder_id/` | ID 力矩解码器（方向 A，val MAE 4.13） | 方向 A |
| `token_vae_e27/`（服务器） | E27 VAE：vae.pt / pca.npz / z_walk.npy / meta.json | E27 |
| `interp_router_flat.json` | 相位插值 MuJoCo 评测 | 方向 B |
| `id_hybrid_scale.json` | 混合力矩 ×0.2/0.3 平坦评测 | 方向 A |
| `train_log_e23.json` / `e23_stress_128.log` | E23 训练日志 / 128 envs 压力测试 | 方向 C |

## ⚠️ 注意：`distill_final/` 不是 v9

交接包 `03_OUTPUTS_INDEX.md` 曾写"v9 路由器在 `distill_final/`"，这是**笔误**：

- **`distill_v9/`** = 真 v9 路由器（**19** 个命令组，phase/proto g0..g18）。
- **`distill_final/`** = v6 时代旧物（仅 **14** 个组 phase_g0..g13，且混有
  `eval_battery_v6.json`、`v6_reel_local.mp4`）。

已在 `HANDOFF/03_OUTPUTS_INDEX.md` 订正。

## legacy 产物（可忽略，未删）

`distill/`、`distill_v3~v8/`、`distill_ar/`、`distill_closed/`、`depth_data*/`、
`depth_student*/`、`flat_g1/`、`flat_g1_noband/`、`torque_decoder_v9/` 等为历代
中间产物，对应 `apt_g1/_archive/` 中的脚本。
