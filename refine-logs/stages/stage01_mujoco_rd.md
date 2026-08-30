## 阶段 1：MuJoCo 两条线（R/D 系列，2026-08-11 ~ 08-12）

> 【层位 L2 叶子｜阶段史明细（1 阶段 = 1 文件）】↑ `HANDOFF/02_EXPERIMENT_HISTORY.md`（阶段索引）｜↓ `refine-logs/tracker/` 系列文件（L3 Run 台账）｜≈ 同层：`refine-logs/` 专题日志。

| 块 | 做了什么 | 结论 | 产出 |
|---|---|---|---|
| 数据采集 D001 | 官方闭环数据（无 band，4 modes）20,838 步 | token 与官方一致 | `apt_g1/data/exp1/` |
| 合并 D026 | exp_all3 = 68,093 步 | 含 idle/slow/walk/jump/stealth | `apt_g1/data/exp_all3/` |
| BC D003 | MLP/GRU/Transformer/AR 回归 token | 开环 val MSE 0.0012，闭环 20-30x 复合误差 | `outputs/distill_*` |
| 闭合周期 | 闭合误差 0.00000 | A 持平、B/C 变差 → 死路 | `distill_closed/` |
| oracle D028 | 官方 token 开环回放 walk 6 新方向 | 95-270 步内全倒 | `oracle_walk_bins.py` |
| 控制器 R 系列 | 7 个 RL 变体（token/VAE/skill/aux） | 全部劣于 aux=0；单进程 PPO 基础设施不足 → 终止 | `EXPERIMENT_TRACKER.md` R 行 |

