# refine-logs 实验记录索引

本页 `README.md` 是 refine-logs 的索引：每篇文档必须在这里登记；
`python refine-logs/tools/tree_check.py` 检查登记 / 实存 / 链接三项，全绿才算完成。
**新文档先登记、再写内容。**

写文档的规矩（就这四条，其余照 `../AGENTS.md`）：

1. Run 行只追加进 `tracker/<系列>.md`，不新开文件。
2. 长寿命文档头部标 `活跃 / 冻结 / 归档`；存量历史文档不回填。
3. 新文档头部回链父域；`refine-logs/` 下每一篇都要出现在本页。
4. 只写结论与判断，不写过程叙事。索引里一行说清一篇，详细内容留在文档内部。

## 按问题找权威

| 要找什么 | 去哪 | 状态 |
|---|---|---|
| 某 Run 的数据行 | `tracker/` 五系列（下表） | 活跃，append-only |
| 全局行数统计 / 原表还原区间 | `EXPERIMENT_TRACKER.md` | 活跃 |
| 项目最终结论与口径 | `../HANDOFF/00_FINAL_SUMMARY.md`、`../HANDOFF/README.md` §3 | 冻结（口径） |
| 某阶段来龙去脉 | `stages/`（索引在 `../HANDOFF/02_EXPERIMENT_HISTORY.md`） | 冻结 |
| 当前在做什么 | `ds/`、`to/` 里标"活跃"的几篇 | — |
| 产物文件在哪 | `../HANDOFF/03_OUTPUTS_INDEX.md` | 活跃 |
| 某脚本干什么 | `../apt_g1/SCRIPT_MAP.md` | 活跃 |

## 台账（数据唯一事实源）

| 文件 | 范围 |
|---|---|
| `tracker/R.md` | R001–R020，MuJoCo RL 线 |
| `tracker/D.md` | 蒸馏线（Distillation Exp / Phase2-3 / Stress Test） |
| `tracker/E.md` | Isaac APT 主线 E01–E48 |
| `tracker/MQ.md` | 官方规划器复刻 MQ07–MQ12 + Gate0 地形评测 |
| `tracker/TO.md` | TO 数据管线与 TO18–TO35 力矩/WBC 线 |

## 阶段史（`stages/`）

写完即冻结；阶段索引在 `../HANDOFF/02_EXPERIMENT_HISTORY.md`。

| 文件 | 阶段 |
|---|---|
| `stages/stage01_mujoco_rd.md` | 1：MuJoCo R/D 两线 |
| `stages/stage02_isaac_e01_e14.md` | 2：Isaac 机制矩阵 E01–E14 |
| `stages/stage03_terrain_perception_e15_e21.md` | 3：地形/感知 E15–E21 |
| `stages/stage04_priority_chain.md` | 4：优先级链收尾 |
| `stages/stage05_directions_abc.md` | 5：方向 A/B/C |
| `stages/stage06_e27_missing.md` | 6：E27 缺失实验 |
| `stages/stage07_latent_e28_e47.md` | 7：latent 线 E28–E47 |
| `stages/stage08_planner_mq07_mq12.md` | 8：planner 复刻 MQ07–MQ12 |
| `stages/stage09_decoder_finetune_e44.md` | 9：解码器微调 E44 |
| `stages/stage10_from0_e45_e47.md` | 10：从零 + 冻结解码器 E45–E47 |
| `stages/stage11_e48_residual.md` | 11：E48 全关节残差（实证关闭） |
| `stages/stage12_to_torque.md` | 12：TO 力矩线 |
| `stages/gate0_rough_terrain.md` | Gate0：论文形状 rough 地形 |

## DS 线（`ds/`）—— 当前主线

| 文件 | 是什么 | 状态 |
|---|---|---|
| `ds/DS_TERRAIN_ADAPTER_CHARTER.md` | 纲领：冻结底座 × 地形适配 × 未见动作迁移；冲突以它为准 | 活跃 |
| `ds/DS_TERRAIN_AUTHOR_PLAN.md` | 地形作者计划：数据发动机 + token 层作者（G0=D059 生死判别，D060-D062 承接） | 活跃 |
| `ds/DS_TRACKING_STATUS_2026-09-16.md` | 交接快照：D058 PAUSED/D059 地形线立项+G0-① 甲（tokenizer 四代先例）+G0-② 选料键锁定；**当前接手入口** | 活跃 |
| `ds/DS_TRACKING_STATUS_2026-09-15.md` | 交接快照：D057 G3 核验→G4 构建→训练完成；只剩 G5 init 档位图（命令在 §4 仍有效） | 冻结 |
| `ds/DS_TRACKING_STATUS_2026-09-14.md` | 交接快照：D057 WBT 语料接入（§4 接手指令已执行完前四步） | 冻结 |
| `ds/DS_TRACKING_STATUS_2026-09-12.md` | 交接快照：平地跟踪三瓶颈 | 冻结 |
| `ds/DS_CONTINUOUS_EXECUTION_PLAN.md` | 平地命令跟踪持续执行计划（D048 系） | 执行中 |
| `ds/DS_OFFICIAL_DATA_PLAN.md` | 官方数据独走执行计划 | 活跃 |
| `ds/DS_B4LITE_ACTION_SPLIT_PLAN.md` | B4-lite 动作清单与划分协议 | 活跃 |
| `ds/DS_D045_B4LITE_BUILD_LOG.md` | B4-lite 首版构建日志 | 活跃 |
| `ds/DS_GAIT_MANIFOLD_PLAN.md` | 步态流形双源架构 | 执行中 |
| `ds/DS_SONIC_OFFICIAL_DATA.md` | SONIC 官方数据资产定位与离线编码选项 | 活跃 |
| `ds/DS_RECOLLECT_PLAN.md` | 重采线（09-05b 自采退役，设计归档不删档） | 冻结 |
| `ds/DS_S2R_EVIDENCE.md` | Isaac→真机 gap 外部证据 | 活跃 |
| `ds/DS_HIL_SURVEY.md` | HIL 混合模仿学习评估调研 | 活跃 |

## TO 线（`to/`）

| 文件 | 是什么 | 状态 |
|---|---|---|
| `to/TO_TORQUE_MAINLINE.md` | 主线宣言：TO 力矩路线 × APT 论文实现度对照 | 活跃 |
| `to/TO42_PLAN.md` | TO42 学习型 regime 选择，协议冻结候选 | 活跃 |
| `to/TO40C_PLAN.md` | TO40-C 力矩前馈通道门控（Rung 0） | 活跃 |
| `to/TO41_RUNG1_IMPL.md` | Rung 1 实施章程 + D1/D2/D3 integrity test | 活跃 |
| `to/TO41_D_DRYRUN_PROTOCOL.md` | D1/D2/D3 28-cell conformance dry-run 协议 | 活跃 |
| `to/TO41_D_IMPL.md` | Mode A runtime + independent checker 实现日志 | 活跃 |
| `to/TO41_LAUNCH_SANITY.md` | L1–L4 真实 env 接线验证（freeze 前最后一道门） | 活跃 |
| `to/TO41_C_DIAGNOSIS.md` | 56-cell 现有产物诊断；终裁 (b) ACCEPT | 收束 |
| `to/TO41_G_DOWN_SPEC.md` | 低速 downward-continuation 材料生成 | 收束 |
| `to/TO41_RUNG1_CLOSURE.md` | Rung 1 科学关线收束 | 收束 |
| `to/TO38_PLAN.md` | TO38 RL 稳定器叠加 TO 参考 | 冻结 |
| `to/TO_TORQUE_LINE_REPORT.md` | TO01–TO22 力矩线收束（前半程） | 冻结 |
| `to/LEG_LEVEL_TO_PLAN.md` | 腿级 TO（Drake dircol）设计定稿 | 收束 |
| `to/LEG_LEVEL_TO_REPORT.md` | 腿级 TO 线收束 | 冻结 |

## E 线（`e/`）

| 文件 | 是什么 | 状态 |
|---|---|---|
| `e/E49_KL_GUARD_PLAN.md` | E49 退化调研结论 + KL 信任域守卫预注册 | 活跃 |
| `e/E49_STATUS_2026-09-05.md` | E49-A 执行状态交接 | 冻结 |

## 调研（`surveys/`）

| 文件 | 是什么 | 状态 |
|---|---|---|
| `surveys/EMBODIED_AI_FIELD_SURVEY.md` | 具身智能全领域谱系地图（领域背景，不承载本仓口径） | 活跃 |
| `surveys/LITERATURE_SURVEY_FROZEN_DECODER.md` | 冻结解码器 + 位置 RL 地形泛化综述 | 冻结 |
| `surveys/LITERATURE_SURVEY_DS_MANIFOLD.md` | DS 步态流形近邻地图与创新点评估 | 冻结 |
| `surveys/RESEARCHCLAWBENCH_SURVEY.md` | RCBench agent 选型 | 冻结 |

## 运行日志与机制专题（`logs/`）

| 文件 | 是什么 | 状态 |
|---|---|---|
| `logs/TRAIN_SPEEDUP_LOG.md` | 训练栈提速 | 活跃 |
| `logs/MUJOCO_APT_LOG.md` | MuJoCo 两线全程留档 | 冻结 |
| `logs/ISAAC_APT_LOG.md` | Isaac 阶段运行记录 | 冻结 |
| `logs/DISTILL_EXPERIMENT.md` | 蒸馏可行性：相位路由器可、朴素 BC 不可 | 冻结 |
| `logs/DATA_GENERALIZATION_LOG.md` | 数据/网络泛化实验留档 | 冻结 |
| `logs/ROOT_CAUSE.md` | token/VAE/skill RL 不稳的证据链 | 冻结 |

## 收束报告（`reports/`）

| 文件 | 是什么 | 状态 |
|---|---|---|
| `reports/FINAL_REPORT.md` | 最终定稿（两个核心问题的最终回答） | 定稿 |
| `reports/HUMAN_READABLE_COMPLETE_REPORT.md` | 08-12 人话版全时间线复盘 | 冻结 |
| `reports/APT_PROJECT_SUMMARY.md` | 08-13 时点总结（口径以 FINAL_REPORT 为准） | 冻结 |
| `reports/STAGE_SUMMARY_2026-08-13.md` | 08-13 时点阶段总结 | 冻结 |
| `reports/WBC_BRINGUP_REPORT.md` | TO23–TO28 QP-WBC（含 §6 翻案 TO31–32） | 冻结 |

## 早期计划与规格（`plans/`）

| 文件 | 是什么 | 状态 |
|---|---|---|
| `plans/APT_GAP_ANALYSIS.md` | APT 论文差距台账 + decoder 性能画像卡 | 活跃 |
| `plans/PAPER_TERRAIN_SPEC.md` | 论文地形定义（结论已被 Gate0 修订） | 冻结 |
| `plans/EXPERIMENT_PLAN.md` | 早期英文实验计划（被阶段史收束） | 归档 |

## 域外（不属于本树强制范围，仅作导航）

`../HANDOFF/README.md`（层位总图 §0 + 结论口径 §3）、`../HANDOFF/00_FINAL_SUMMARY.md`、
`../HANDOFF/02_EXPERIMENT_HISTORY.md`、`../HANDOFF/03_OUTPUTS_INDEX.md`、`../apt_g1/SCRIPT_MAP.md`
