# 实验历史与结论（按阶段）

> 完整逐条记录见 `refine-logs\EXPERIMENT_TRACKER.md` 与对应日志。下表是
> 压缩后的阶段史 + 关键数字 + 结论。

## 阶段 1：MuJoCo 两条线（R/D 系列，2026-08-11 ~ 08-12）

| 块 | 做了什么 | 结论 | 产出 |
|---|---|---|---|
| 数据采集 D001 | 官方闭环数据（无 band，4 modes）20,838 步 | token 与官方一致 | `apt_g1/data/exp1/` |
| 合并 D026 | exp_all3 = 68,093 步 | 含 idle/slow/walk/jump/stealth | `apt_g1/data/exp_all3/` |
| BC D003 | MLP/GRU/Transformer/AR 回归 token | 开环 val MSE 0.0012，闭环 20-30x 复合误差 | `outputs/distill_*` |
| 闭合周期 | 闭合误差 0.00000 | A 持平、B/C 变差 → 死路 | `distill_closed/` |
| oracle D028 | 官方 token 开环回放 walk 6 新方向 | 95-270 步内全倒 | `oracle_walk_bins.py` |
| 控制器 R 系列 | 7 个 RL 变体（token/VAE/skill/aux） | 全部劣于 aux=0；单进程 PPO 基础设施不足 → 终止 | `EXPERIMENT_TRACKER.md` R 行 |

## 阶段 2：Isaac 机制矩阵 E1-E14（平坦，2026-08-12）

| 变体 | A 60s | 结论 |
|---|---|---|
| 冻结先验 noaux（E1 基线） | 3/3，47m | 最优 |
| aux gate-on/off（E2/E6） | 3-32m | aux 无正向价值 |
| E13 修正门控 2Hz | 34-37m | 唯一正向机制 |
| E3 phase+aux 联合 | 不走路 | 离散相位学不出振荡器 |
| vanilla E9/E11（直出关节） | **0/3 立即倒** | 蒸馏先验必要 |

## 阶段 3：地形/感知/动作通道 E15-E21（2026-08-12）

| 变体 | rough 0.06 / 0.08 | 结论 |
|---|---|---|
| 冻结先验 | 3/3 / 0-1/3 | 0.08 是悬崖 |
| E16 aux+elevation | 0/3 / 0/3 | 地图无 latent/gait 通道时不产生价值 |
| E19 phase 锚定 | 3/3 存活 | 存活但不前进 |
| E20c gate+anti-stop | 0.06: 3/3 前进 38-43m | gate 学对选先验组；aux 仍是破坏源 |
| E21b 离散地形 | stairs/障碍 3/3，stones 0/3 | 小台阶在能力内，垫脚石是盲区 |

## 阶段 4：优先级链（2026-08-12 夜 ~ 08-13 凌晨）

1. **优先级 1**：官方长效数据只能来自 planner 闭环重规划或真 TO/ID 力矩
   数据；官方 token 开环扩充判死。
2. **优先级 2**：`router_fallback.py` 显式回退表 → 24/24 命令 3/3×20s 无
   跌倒（`flat_battery_fallback_v9.json`）；60s+ 马拉松 2/3。
3. **优先级 3**：E22a（A 34-36m）、E22b（A 0.9-4.3m）均 < 阈值 42.9m →
   aux 判据未达成，定论 aux 无正向价值。

## 阶段 5：三方向 A/B/C（2026-08-13）

### 方向 A：力矩级数据/解码器
- `recover_id_torque.py` 用 mj_inverse 算 ID 力矩（27k 行）→ phase→ID 力矩
  解码器 val MAE 4.13（PD 标签 ~9.4）。
- 混合 ×0.2/0.3 平坦 3/3（15.2-15.4m）；论文式纯 ID 前馈 124-126 步倒；
  粗糙 0.06 0/3。
- 结论：ID 改善前馈，单独不解锁论文式控制；需自洽 TO/ID 规划数据。

### 方向 B：连续潜空间
| 实验 | 设计 | A 60s | B/C/D | 结论 |
|---|---|---|---|---|
| 插值读取 | 相邻原型线性插值（MuJoCo） | 3/3，15.6-15.8m | - | 机制无损 |
| E23 | 自由相位 RL | 0.8-2.0m | C 0/3 | 无压力 → 原地振荡 |
| E24 | +anti-stop+progress | 9-12m | B/C/D 全过 | 鲁棒但慢（学不会时钟） |
| E25 | 相位锚定 + 消融 | aux 关 49m | 全过 | **aux 是唯一破坏源** |
| E26 | 纯相位偏移（aux_scale=0） | 45.7-46.1m | 全过 | 无损但机制中性 |
| E26-T | rough 0.06/0.08 | 0.06: 3/3；0.08: 偏移 0/6、纯时钟 2/6 | - | 0.08 是共同能力边界 |

### 方向 C：千级并行
- Isaac 128 envs：dt 0.90s（64 envs 0.72s），吞吐 +61%，显存 2.8GB 未用满 →
  瓶颈在更新/同步。
- mjlab：1024 envs 冒烟 1.03s/iter、0.84GB、69%；官方默认 4096 envs 面向
  24GB GPU。

## 阶段 6：E27 缺失实验（2026-08-13，用户指定优先）

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

## 关键对照阶梯（本文最核心的一张表）

| 方案 | A 60s 位移 | 说明 |
|---|---|---|
| 冻结先验（路由器+SONIC 解码器） | 47m | 满速，最优 |
| E26 相位偏移（aux 关） | 46m | 无损但中性 |
| E25 相位锚定（aux 归零） | 49m | 同上 |
| E27 latent→VAE→SONIC（无行为先验） | 19m | 可行但半速 |
| E9/E11 vanilla 直出关节 | 0m（立即倒） | 无 Sonic 则失败 |
| mjlab 从零（1024 envs） | 训练中（~900 iters，跌倒率 0.08-0.17） | 外部对照 |
