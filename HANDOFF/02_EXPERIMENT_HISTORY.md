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

## 2026-09-09/10 平地命令跟踪阶段1（D048f–D048k）负结果收官（D 系列）

- D 系列平地命令跟踪阶段1 负结果收官（2026-09-09/10）：三对照（progress 偏置否证 / 更新约束正效应 / 奖励封顶无改善）+ 独立种子复证 + 0.6 固定命令边界补评（外部门 0/18 全败，按 §5e 判读边界冻结收官）。
- 关键正结果：步级解析 KL 看门（KL≤0.05/步）消除晚期崩坏与残差饱和（sat 0.433→0）且跨种子复现 = D048i 稳定训练基线（受约束稳定不崩）。
- 关键负结果：0.4 直走未攻克（task_success 0/162）——速度双向不跟踪（0.4 过冲 0.62 / 0.6 欠速 0.45）、横漂 0.6 全域 5.8–6.4m（归因勘误 09-10：非命令上界——vx_max=0.8 未到、0.4→0.6 跨档界 0.5333 混杂）、早期跟踪增益被守卫同压（it50 vx_rmse 0.118→0.246）。
- 两处口径资产：带符号残差统计 + ckpt 身份信封必填硬化。
- 阶段1 正式关闭：交付 = 稳定训练基线 + 受控负结果三边界（晚期崩坏=更新过大已证已修 / 速度=双向不跟踪 / 横漂=0.6 域全域化、归因未定〔D048l 档位诊断中〕）；B5-gait 另行裁定、阶段2 不进入。
- 09-10 深夜 owner 核验三勘误：D048k 终末航向实为 82.7°~107.2°（yaw_err 弧度误读为度，航向门全败，非保持项）；稳定性复现为两训练种子（D048j 复用 s0）；「命令上界失稳」归因撤回。后续唯一推进=D048l 命令→速度档映射诊断（DS_CONTINUOUS_EXECUTION_PLAN.md §5f，零训练预算）——已完成当日闭环：36 局联合门 0/36，机制事实=速度全由档位中介+档位映射与名义反向+速度-方向反相耦合（D048k 横漂/航向归因收口=档位切换效应），映射修补线按退出条件关闭、转表示层对照待裁定（详 D.md D048l 行）。
- 详 `refine-logs/tracker/D.md` D048f–D048l 行与 `refine-logs/DS_CONTINUOUS_EXECUTION_PLAN.md` §4–§5f。
- **09-10 深夜→09-11 表示层证据链 + APT 差距台账 + 上限卡 + 选择权首验（D048m→D049，owner 授权「逐渐补齐 APT 论文差距+吃透 decoder 性能」）**：①D048m 三角证据（vb 语义错码=表示层本征）→D048n 速度锚定标签恢复单调（因果链闭合，量程 b2 1.31>0.8 失手）→D048p edges 校准负结果（三档塌缩=校准破坏标签分布均衡 6.3:1:2.6，edges 反标定路线关闭）；②D048q 条件轴画像：Q1 vb 软混=**速度轴连续可调**（四组 vx(α) 单调近似线性插值，b1/b2 软混覆盖 0.71-1.20 带）+Q2 db 方向轴粗编码且随重训不稳定（方向三层不可控证据链闭合）；③D048r 首张受控上限卡（owner 批评「未测上限」直答）：接口-主动上限 vx*=1.164（CEM z 净前向）/本体路径 1.71（直线高速段无材料=语料缺口）/方向回收 11×（init b2 机体 1.31 vs 净前向 0.10 口径修正）；④`refine-logs/APT_GAP_ANALYSIS.md` 差距台账建卡（8 项：KL 2.5e-6 与 2Hz 门控已对齐、τ_ff 混控结构性不可比、gait 选择权归策略=最重可补）；⑤D049 vb 选择权归策略两臂（连续软权重 48 维）：联合门 8 格全败（24 局完整维度），但 owner 核验发现 **P1 动作-概率契约缺陷**（采样索引进 log_prob 而 env 执行 softmax(logits)⇒ vb 头无合法策略梯度，E[A∇logp]=0 精确枚举证实）→ **b 臂判「执行失败、策略选择机制验收不成立」**，原机制结论全部撤回重开（force 谱对齐 CEM 系机体/净前向口径混用；a-c06-aux 两局过航向门）；修复路线 owner 冻结=连续采样贯穿+消费者测试门+受控复验，方向标签重训/课程/预算暂停。详 D.md D049 行与 DS_CONTINUOUS_EXECUTION_PLAN §5j。

