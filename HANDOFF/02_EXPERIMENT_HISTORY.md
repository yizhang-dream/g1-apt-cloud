# 实验历史与结论（按阶段）——阶段索引

> 【层位 L2｜阶段史索引，1 阶段 = 1 行 → 1 个叶子文件】↑ `HANDOFF/README.md`
> （L1 总览与钻取导航）｜↓ `refine-logs/stages/` 13 个阶段叶子（stage01~12 +
> gate0，本表每行的下钻目标）｜再往下：`refine-logs/tracker/` 系列文件
> （L3 Run 台账·事实源）｜≈侧轴：`refine-logs/` 专题日志
> （FINAL_REPORT / TO_TORQUE_LINE_REPORT / WBC_BRINGUP_REPORT…）。
> 2026-08-28 起本文件只做索引；阶段明细整节搬入 stages/ 叶子，内容零改动。

| 阶段 | 时间 | 一句话结论 | 明细 |
|---|---|---|---|
| 1. MuJoCo 两条线（R/D） | 08-11~12 | BC 闭环 20–30× 复合误差、oracle 教师做不了 walk+turn、7 个 RL 变体全劣于 aux=0 → 单进程 PPO 基础设施不足，转 Isaac | [stage01_mujoco_rd.md](../refine-logs/stages/stage01_mujoco_rd.md) |
| 2. Isaac 机制矩阵 E1–E14（平坦） | 08-12 | 冻结先验 noaux 最优（47m）；vanilla 直出关节 0/3 立倒 → 蒸馏先验必要；E13 修正门控 2Hz 是唯一正向机制 | [stage02_isaac_e01_e14.md](../refine-logs/stages/stage02_isaac_e01_e14.md) |
| 3. 地形/感知/动作通道 E15–E21 | 08-12 | rough 0.08 是悬崖；elevation/phase 等通道在无表达通道时不产生价值；gate+anti-stop 学会按命令选先验 | [stage03_terrain_perception_e15_e21.md](../refine-logs/stages/stage03_terrain_perception_e15_e21.md) |
| 4. 优先级链 | 08-12 夜~13 | 官方 token 开环扩充判死；fallback 表 24/24 命令全过；aux 判据未达成 → 定论 aux 无正向价值 | [stage04_priority_chain.md](../refine-logs/stages/stage04_priority_chain.md) |
| 5. 三方向 A/B/C | 08-13 | ID 力矩改善前馈但单独不解锁论文式控制；E25/E26 证明 aux 是唯一破坏源；千级并行瓶颈在更新/同步而非 env 数 | [stage05_directions_abc.md](../refine-logs/stages/stage05_directions_abc.md) |
| 6. E27 缺失实验 | 08-13 | 无行为先验的 latent RL 从零可走但半速（19m vs 47m）→ Sonic token 流形本身就是关键先验 | [stage06_e27_missing.md](../refine-logs/stages/stage06_e27_missing.md) |
| 7. 速度/方向条件化潜空间 E28–E47 | 08-13~14 | 关键对照阶梯：E39 双解耦历史最佳（0.417 m/s / 直行 0.98），E47 从零线最优（0.42 / 0.944），E44 微调稳健负结果；0.08 悬崖归因修正为蒸馏路径边界（见 MQ09） | [stage07_latent_e28_e47.md](../refine-logs/stages/stage07_latent_e28_e47.md) |
| 8. 官方规划器复刻 MQ07–MQ12 | 08-14 | 官方 planner 是运动学盲重规划非地形自适应；0.08 悬崖 = 无重规划蒸馏路径的边界（非解码器本身），重规划真实边界 ≈0.12–0.14；步态模式（crawl）是现成稳健杠杆 | [stage08_planner_mq07_mq12.md](../refine-logs/stages/stage08_planner_mq07_mq12.md) |
| 9. 解码器微调 E44 | 08-14/15 | 稳健负结果：任何程度的微调都重新激活 SONIC 快走固有转向偏置（打转即倒含平地）→ 冻结解码器是承重墙 | [stage09_decoder_finetune_e44.md](../refine-logs/stages/stage09_decoder_finetune_e44.md) |
| 10. 从零 + 冻结解码器 E45–E47 | 08-14 | vanilla 从零靠蹲蹭作弊（h_min 0.20）；冻结解码器防作弊；双解耦 VAE + 轻 heading 让从零策略快且直（0.42 m/s / 0.944），逼近 walk 先验版 | [stage10_from0_e45_e47.md](../refine-logs/stages/stage10_from0_e45_e47.md) |
| Gate 0. 论文形状 rough 地形 | 08-15 | 实测推翻"0.06≈论文上限"推断：坑（负障碍）是唯一必要难点变量；只凸 0.06 过 / 有坑 ±0.06 全倒 / 只凸 0.08 全倒 | [gate0_rough_terrain.md](../refine-logs/stages/gate0_rough_terrain.md) |
| 11. E48/E48c 全关节残差 | 08-15 | 跨 3 配置稳健负结果：基座没立住时额外自由度通道只会被噪声梯度占据；0.08 存活 1/9 为边缘事件，悬崖仍成立 | [stage11_e48_residual.md](../refine-logs/stages/stage11_e48_residual.md) |
| 12. TO 力矩线 TO01–TO35 | 08-14~17 | 自洽 TO 力矩数据可自建且高度可学，但"合成步态 + PD/WBC"到 8.52s 踉跄存活为止 → 稳定流形要么来自数据（SONIC）要么用 RL 学，不能靠简化模型合成 | [stage12_to_torque.md](../refine-logs/stages/stage12_to_torque.md) |
| 13. 腿级 TO 线 TO36（Drake dircol） | 08-29~30 | 三门 DoD：A 达成（平地+刚性+膝限位可行 0.277 m/s，审计验收制）；B 双验证执行（摆动链一致 PASS、支撑链 −5 N·m=URDF↔MJCF CoM 差 1.4 cm 归因）；C 负结果（开环回放 15 稳层配置最优 1.84 s——2D 相位解非 43-DOF 不变流形；capture 落足反馈 +40% 正向线索）→ 自洽 TO 力矩数据就绪可喂解码器，闭环须 RL 稳定器叠加 | [LEG_LEVEL_TO_REPORT.md](../refine-logs/LEG_LEVEL_TO_REPORT.md) |

## 2026-08-27 补产出（文档补全轮）

- 数据图表：`apt_g1/outputs/figs/fig1–fig6`（对照阶梯 / 速度-直行 Pareto /
  地形形状边界矩阵 / 规划器线 / TO 战役 / E48 残差），由
  `apt_g1/plot_paper_figures.py` 生成（数据优先读服务器评测 JSON，缺失项嵌
  TRACKER 台账数字并标注来源）。
- 视频：`e47_mujoco.mp4`（从零线最优控制器 E47 平地行走，补齐视频库从零线空缺）。
