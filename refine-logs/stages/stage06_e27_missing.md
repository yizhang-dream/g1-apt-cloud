## 阶段 6：E27 缺失实验（2026-08-13，用户指定优先）

> 【层位 L2 叶子｜阶段史明细（1 阶段 = 1 文件）】↑ `HANDOFF/02_EXPERIMENT_HISTORY.md`（阶段索引）｜↓ `refine-logs/tracker/` 系列文件（L3 Run 台账）｜≈ 同层：`refine-logs/` 专题日志。

- 问题："自主学习、不通过 Sonic 学行为、但让 Sonic 动"从未干净做过。
- 实现：相位条件化 token VAE（因果窗口 10×64 → z∈ℝ16；D(z,sinφ,cosφ)→
  token；φ=walk 2-PC PCA 相位，时钟 0.121 rad/步）+ Isaac latent 模式 +
  z_walk warm-start 200 iters。
- 结果：

| 测试 | E27（latent 无行为先验） | E1 冻结先验 | vanilla E9/E11 |
|---|---|---|---|
| A 60s | 3/3，19.1m（vx 0.32） | 3/3，47m | 0/3 立即倒 |
| B/C/D | 12/12 / 3/3 / 3/3 | 全过 | - |

- 结论：**Sonic token 流形本身就是关键先验**；无行为先验时从零 RL 可学
  行走但速度减半（样本效率代价）。
- 产出：`apt_g1/train_token_vae_e27.py`、`apt_g1/isaac/token_window_vae.py`、
  `outputs/token_vae_e27/`、`outputs/isaac_e27_latent/`、
  `outputs/isaac_eval_e27.json`。

