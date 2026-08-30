## 阶段 3：地形/感知/动作通道 E15-E21（2026-08-12）

> 【层位 L2 叶子｜阶段史明细（1 阶段 = 1 文件）】↑ `HANDOFF/02_EXPERIMENT_HISTORY.md`（阶段索引）｜↓ `refine-logs/tracker/` 系列文件（L3 Run 台账）｜≈ 同层：`refine-logs/` 专题日志。

| 变体 | rough 0.06 / 0.08 | 结论 |
|---|---|---|
| 冻结先验 | 3/3 / 0-1/3 | 0.08 是悬崖 |
| E16 aux+elevation | 0/3 / 0/3 | 地图无 latent/gait 通道时不产生价值 |
| E19 phase 锚定 | 3/3 存活 | 存活但不前进 |
| E20c gate+anti-stop | 0.06: 3/3 前进 38-43m | gate 学对选先验组；aux 仍是破坏源 |
| E21b 离散地形 | stairs/障碍 3/3，stones 0/3 | 小台阶在能力内，垫脚石是盲区 |

