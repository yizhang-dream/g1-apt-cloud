# refine-logs 实验记录扇出树（根地图）

> 【层位：扇出树根地图｜树深度 = 根（此"层"是挂树深度，**不是** HANDOFF/README.md §0
> 的内容粒度层位 L0–L4，两套编号不互换）】↓ 全部子域见下方扇出树。
> 本页仿照 mini Biosphere 项目的文档扇出树纪律建立（2026-08-29）：项目里每一篇
> 实验记录文档都必须挂在这棵树上，由 `refine-logs/tools/tree_check.py` 强制检查。
> 想知道实验记录全貌，从本页出发沿树走；改了任何一篇，先回来更新本页。

## 扇出规则（写文档前必读）

1. **挂树先行**：`refine-logs/` 下每一篇 `.md` 必须出现在本页扇出树里；**新文档先挂树、再写内容**。`python refine-logs/tools/tree_check.py` 强制检查（挂树 / 实存 / 链接三项，全绿才算完成）。
2. **层级与域**：根地图（本页）→ 域（索引 / 台账 / 阶段 / 专题 / 域外路由）→ 正史与叶子。域是扇出单位，新增一级域须在本页挂树并在 `HANDOFF/` 留痕。
3. **单域一正史**：台账域每系列 `tracker/<系列>.md` 是该系列 Run 数据的**唯一权威**（数据唯一事实源）；跨域结论口径的权威是 `HANDOFF/README.md` §3。数据冲突以 tracker 为准，口径冲突以 HANDOFF §3 为准。
4. **回链**：父文档必须列全其子文档（树完整性，脚本强制）；子文档头部【层位】导航条须回链父域（仓库既有惯例）。**存量历史文档豁免回链**，不为其回填改动；2026-08-29 起新建文档必须遵守。
5. **命名**：Run 行只追加进 `tracker/<系列>.md`，不新开文件；阶段叶子 `stages/stageNN_<主题>.md`；专题日志 `<主题>_LOG.md`；收束报告 `<主题>_REPORT.md`；调研 `<主题>_SURVEY.md`；论文/规格对照 `*_SPEC.md`；时点快照加 `_YYYY-MM-DD` 后缀。
6. **状态**：长寿命新文档头部必须标 `活跃 / 冻结 / 归档` 三态横幅；存量历史文档不回填文件头，状态以本页树内标注为准（未标注者默认冻结存档）。
7. **数据纪律不受影响**：实验（含负结果）必入 tracker 系列文件、同步 `EXPERIMENT_TRACKER.md` 行数等流程照 `AGENTS.md` 执行；本树只管**文档挂载与完整性**，不改变数据事实源流程。
8. **语言**：中文，术语首现白话解释（仓库既有规则）。

## 扇出树

```text
refine-logs/README.md……………………………………………………… 根地图（本页）
├── EXPERIMENT_TRACKER.md…………………………………… 索引域：总索引（五系列行数统计与还原
│   …………………………………………………………………… 区间；不存 Run 行）。活跃
├── tracker/……………………………………………………………… 台账域：Run 台账（数据唯一事实源，
│   …………………………………………………………………… append-only；每系列一正史）
│   ├── tracker/R.md……………………………………… R 系列：R001–R020 MuJoCo RL 线
│   │   ……………………………………………………………（token/VAE/skill/aux 变体全劣于 aux=0）。活跃
│   ├── tracker/D.md……………………………………… D 系列：蒸馏线（Distillation Exp /
│   │   …………………………………………………………… Phase2/3 + Stress Test）。活跃
│   ├── tracker/E.md……………………………………… E 系列：Isaac APT 主线 E01–E48
│   │   ……………………………………………………………（含 FB/I21/T1-T2 辅助行与混合节）。活跃
│   ├── tracker/MQ.md…………………………………… MQ 系列：官方规划器复刻 MQ07–MQ12
│   │   …………………………………………………………… + Gate0 论文形状地形评测行。活跃
│   └── tracker/TO.md…………………………………… TO 系列：TO 数据管线 + TO18–TO35
│       ……………………………………………………………（力矩/WBC 线）。活跃
├── stages/……………………………………………………………… 阶段域：阶段史叶子（1 阶段 = 1 文件，
│   …………………………………………………………………… 写完即冻结；索引在 HANDOFF/02）
│   ├── stages/stage01_mujoco_rd.md………… 阶段 1：MuJoCo R/D 两线（基础设施不足判定）
│   ├── stages/stage02_isaac_e01_e14.md…… 阶段 2：Isaac 机制矩阵 E01–E14（平坦地）
│   ├── stages/stage03_terrain_perception_e15_e21.md… 阶段 3：地形/感知 E15–E21
│   ├── stages/stage04_priority_chain.md…… 阶段 4：优先级链收尾
│   ├── stages/stage05_directions_abc.md…… 阶段 5：方向 A/B/C（力矩级/连续潜空间/千级并行）
│   ├── stages/stage06_e27_missing.md………… 阶段 6：E27 缺失实验（相位条件化 token VAE）
│   ├── stages/stage07_latent_e28_e47.md…… 阶段 7：latent 线 E28–E47
│   ├── stages/stage08_planner_mq07_mq12.md 阶段 8：planner 复刻 MQ07–MQ12
│   ├── stages/stage09_decoder_finetune_e44.md… 阶段 9：解码器微调 E44
│   ├── stages/stage10_from0_e45_e47.md…… 阶段 10：从零 + 冻结解码器 E45–E47
│   ├── stages/stage11_e48_residual.md……… 阶段 11：E48/E48c 全关节残差（实证关闭）
│   ├── stages/stage12_to_torque.md………… 阶段 12：TO 力矩线
│   └── stages/gate0_rough_terrain.md……… Gate0：论文形状 rough 地形（地形结论修订）
├── 专题日志域（顶层散件；均为历史冻结存档，按性质分组）
│   ├── MUJOCO_APT_LOG.md…………………………… 运行日志：MuJoCo 两线全程留档
│   ├── ISAAC_APT_LOG.md………………………………… 运行日志：Isaac 阶段运行记录
│   ├── DISTILL_EXPERIMENT.md…………………… 机制专题：蒸馏可行性（相位路由器可、朴素 BC 不可）
│   ├── DATA_GENERALIZATION_LOG.md………… 数据专题：数据/网络泛化实验留档
│   ├── ROOT_CAUSE.md…………………………………… 根因专题：token/VAE/skill RL 不稳的证据链
│   ├── FINAL_REPORT.md………………………………… 收束：最终定稿（两个核心问题的最终回答）
│   ├── TO_TORQUE_LINE_REPORT.md…………… 收束：TO01–TO22 力矩线（前半程）
│   ├── WBC_BRINGUP_REPORT.md…………………… 收束：TO23–TO28 QP-WBC（含 §6 翻案 TO31–32）
│   ├── STAGE_SUMMARY_2026-08-13.md……… 收束：08-13 时点阶段总结
│   ├── APT_PROJECT_SUMMARY.md………………… 收束：08-13 时点总结（最终口径以 FINAL_REPORT 为准）
│   ├── HUMAN_READABLE_COMPLETE_REPORT.md 收束：08-12 人话版全时间线复盘
│   ├── LITERATURE_SURVEY_FROZEN_DECODER.md 调研：冻结解码器 + 位置 RL 地形泛化综述
│   ├── RESEARCHCLAWBENCH_SURVEY.md……… 调研：RCBench agent 选型（08-20）
│   ├── PAPER_TERRAIN_SPEC.md…………………… 对照：论文地形定义（结论已被 gate0 修订）
│   ├── LEG_LEVEL_TO_PLAN.md…………………… 计划：TO36+ 腿级 TO（Drake dircol）
│   │   ……………………………………………………………设计定稿（08-29 grill-me 访谈五项决策）。收束
│   ├── LEG_LEVEL_TO_REPORT.md……………… 收束：TO36 腿级 TO 线（A 门膝可行达成/
│   │   ……………………………………………………………B 门双验证/C 门负结果+归因，08-30）
│   ├── TO_TORQUE_MAINLINE.md………… 主线宣言：TO 力矩路线 × APT 论文实现度
│   │   ……………………………………………………………对照 + 重新判定 + 阶梯路线图（08-31 主路线化）
│   ├── TO38_PLAN.md………………………………… 设计定稿：TO38 RL 稳定器叠加 TO 参考
│   │   ……………………………………………………………（08-31 rubric 审后定稿：b+c 注入方案/
│   │   ……………………………………………………………配对 A/B 决策表/评测协议；双臂开跑）
│   ├── TO40C_PLAN.md………………………………… 设计定稿：TO40-C 力矩前馈通道门控
│   │   ……………………………………………………………（Rung 0：τ_ff+PD+RL 三臂配对 + 2×2
│   │   ……………………………………………………………交叉注入诊断 + 门控三分支）。活跃
│   ├── TO41_RUNG1_IMPL.md………………………… 实施章程：Rung 1 运行时实现 + D1/D2/D3
│   │   ……………………………………………………………integrity test + incompatibility 报告
│   │   ……………………………………………………………（mapping 已产出未冻结；compute BLOCKED）。活跃
│   └── EXPERIMENT_PLAN.md………………………… 计划：早期英文实验计划（被阶段史收束）。归档
└── 域外路由（上游/下游权威，不属本树强制范围，仅作导航）
    ├── ../HANDOFF/README.md……………………… 层位总图（§0）+ 结论口径（§3，跨域权威）
    ├── ../HANDOFF/00_FINAL_SUMMARY.md…… L0 结论卡（5 分钟版）
    ├── ../HANDOFF/02_EXPERIMENT_HISTORY.md L2 阶段史索引（stages/ 的父文档）
    ├── ../HANDOFF/03_OUTPUTS_INDEX.md…… L4 产物索引（→ 服务器 outputs/）
    └── ../apt_g1/SCRIPT_MAP.md………………… 代码轴（1 脚本 = 1 行）
```

⏳ = 规划中节点（允许暂不存在于磁盘，tree_check 跳过存在性检查；挂上真实文件后去掉标记）。

## 正史路由（按问题找权威）

| 你要找什么 | 权威文档 | 状态 |
|---|---|---|
| 某 Run 的数据行 | [tracker/](tracker/E.md) 五系列文件 | 活跃（append-only） |
| 全局行数统计 / 原表还原区间 | [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) | 活跃 |
| 某阶段的来龙去脉 | [HANDOFF/02_EXPERIMENT_HISTORY.md](../HANDOFF/02_EXPERIMENT_HISTORY.md) → [stages/](stages/stage01_mujoco_rd.md) | 冻结 |
| 项目最终结论 / 口径 | [HANDOFF/00_FINAL_SUMMARY.md](../HANDOFF/00_FINAL_SUMMARY.md)、HANDOFF/README.md §3 | 冻结（口径） |
| 某专题全程（蒸馏/泛化/根因/力矩/WBC） | 专题日志域（见树） | 冻结 |
| 产物文件在哪 | [HANDOFF/03_OUTPUTS_INDEX.md](../HANDOFF/03_OUTPUTS_INDEX.md) | 活跃 |
| 某脚本干什么 | [apt_g1/SCRIPT_MAP.md](../apt_g1/SCRIPT_MAP.md) | 活跃 |

## 域速览

- **索引域**：`EXPERIMENT_TRACKER.md` 只做索引与统计；Run 行一律在台账域。
- **台账域 [tracker/](tracker/E.md)**：五个系列文件共同构成数据唯一事实源（R/D/E/MQ/TO）。
- **阶段域 [stages/](stages/stage01_mujoco_rd.md)**：13 篇阶段史叶子；阶段索引在 `HANDOFF/02_EXPERIMENT_HISTORY.md`。
- **专题日志域**：顶层 16 篇历史专题（运行日志 / 机制与数据专题 / 收束报告 / 调研 / 计划），均为冻结或归档存档。
- **域外路由**：HANDOFF 交接包与 SCRIPT_MAP 代码轴，实验记录的上游（口径）与下游（产物/代码）。
