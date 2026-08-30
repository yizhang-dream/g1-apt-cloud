## 阶段 8：官方规划器复刻与闭环地形边界（MQ07–MQ12，2026-08-14）

> 【层位 L2 叶子｜阶段史明细（1 阶段 = 1 文件）】↑ `HANDOFF/02_EXPERIMENT_HISTORY.md`（阶段索引）｜↓ `refine-logs/tracker/` 系列文件（L3 Run 台账）｜≈ 同层：`refine-logs/` 专题日志。

> 目标是 goal ①③④：复刻官方 SONIC 三模型全栈，回答"冻结解码器的地形泛化到底
> 差在哪"。全部脚本现役（见 SCRIPT_MAP §1），数字细节见
> `EXPERIMENT_TRACKER.md` MQ 段与 `STAGE_SUMMARY_2026-08-13.md` §10。

| Run | 做了什么 | 结论 |
|---|---|---|
| MQ07 | `planner_sonic.py` 三模型全栈复刻（ONNX 规划器→encoder→decoder），修复速度单位 ×50、encoder anchor offset 584→601 | 关键 8 模式（idle/slow/walk/run/stealth/squat/kneel/crawl）全栈闭环 fall=None |
| MQ08 | 8 模式 × {平地, rough 0.08} 开环对比（`terrain_generalize_test.py`） | 泛化退化与**步态激进程度单调相关**而非 CoM 高度：run 塌陷（根高 0.76→0.25）、walk 停摆（adv −66%）、stealth/crawl/squat 鲁棒（−13~−14%）；"低重心更稳"修正为"慢而稳 > 快而猛"。注意此表是开环视角，MQ09 后需重读 |
| MQ09 | 官方 10Hz 闭环重规划（`planner_closed_loop.py`，每 5 步 live qpos 重出轨迹 + 粗糙度触发切模式） | **闭环重规划让 walk 在 rough ≈ flat**（adv 3.38 vs 3.39m）→ **0.08 悬崖的归因修正：边界在"蒸馏路径缺重规划"，不在冻结解码器本身**（E41/E42 的"解码器边界"口径由此收回） |
| MQ10 | 闭环盲重规划振幅扫描 0.08–0.20 × 3 种子（`closed_loop_sweep.py`） | 修正 MQ09 单种子乐观：跨 seed 后 0.08 仅 1/3 存活、0.14 起基本全灭 → 盲重规划对地形实现高度敏感，真实边界 ≈0.12–0.14 |
| MQ11 | 杠杆测试：height 命令 + 步态模式 @0.14（`closed_loop_levers.py`） | height 是弱杠杆；**步态模式是真杠杆**：walk 0/3、stealth 1/3、**crawl 3/3**，crawl 不倒边界 ≥0.28 但 adv 塌到 ~0 → 缺的是"按地形选模式"的选择器 |
| MQ12 | 粗糙度触发 walk→crawl 自动切换 | 触发正常（25–38 步）但**切换 transition 脆弱**（切换后仅 1/3 存活 vs 纯 crawl 3/3）→ 复刻官方 ADAPTING 状态机是"感知→选步态"落地的唯一剩余前置 |

**MQ 线一句话**：官方 planner 是运动学盲重规划，不是地形自适应；0.08 悬崖是
"无重规划蒸馏路径"特有（对照 mjlab 无悬崖），重规划边界 ≈0.12–0.14，模式选择
（crawl）是官方资产里现成的稳健杠杆。

