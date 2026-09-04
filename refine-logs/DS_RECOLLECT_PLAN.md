# DS 重采线设计定稿：SONIC 全动作组 + 速度轴数据重采集（2026-09-04 owner 方向转向）

> 【层位 L2 侧轴｜设计定稿 + 首冒烟已跑（2026-09-04 owner 指令「完全停掉其他所有，
> 完全主攻数据重采」；TO 线冻结封存，本线为唯一活跃主线）】↑
> `refine-logs/README.md`（扇出树根地图）｜上游：`tracker/TO.md`（TO42 v7 云端
> 余额杀封存）、`TO42_PLAN.md`（被转向取代前的最后 TO 协议）、
> `DATA_GENERALIZATION_LOG.md`（exp_all3 = 68,093 步 / 5 mode 的原始采集）｜
> 数据事实源：`tracker/D.md` D029+（Run 行唯一事实源）｜状态：**活跃**

**这篇讲什么**：SONIC 官方 deploy 源码核查发现动作组全集 27 个（我们只采过 5 个）
且 movement_speed 速度轴从未扫描；D029 冒烟证实 RUN 模式（mode 3，target_vel
1.5–2.7 m/s）在 MuJoCo 仿真闭环零摔倒。本线目标：重采全动作组 + 速度轴网格数据
→ 重训 token VAE → 速度条件轴从「WALK 内部相位三档」升级为「跨 mode 连续速度族」，
攻「快且直」天花板（E39 0.42–0.46）与全速度段材料问题。

---

## 0. 方向转向记录与授权链

- **2026-09-04 owner 指令**：「完全停掉其他所有，完全主攻这个数据重采集」。
  TO42 R1 就地封存（见 §4）；maintenance baseline（TO41 九项冻结清单）不受影响。
- **转向依据链**（本轮会话核查，全部留档于本节与 D029 行）：
  1. owner 质疑 E44 时代「微调必败」结论的三个候选原因（并行太少 / 代码 bug /
     采样遗漏）；
  2. 核查排除 bug 主因（E44-ctrl + E44p 已有对照），确认采样遗漏为实；
  3. deploy 源码核查（`lab-ts:~/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/
     src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner.hpp:78`）坐实遗漏规模。

## 1. 核查发现（SONIC 官方 deploy，权威枚举）

**LocomotionMode 全集 27 个**（节选关键项；`localmotion_kplanner.hpp:78`–105）：

| mode | 名 | 速度注释 | 我们是否采过 |
|---|---|---|---|
| 0 | IDLE | - | ✅ exp_all |
| 1 | SLOW_WALK | 0.1–0.8（键盘钳 0.2–0.8） | ✅ 仅 0.2 默认 + 0.6 两点 |
| 2 | WALK | 0.8–2.5 | ✅ **仅 speed=-1 默认速度** |
| **3** | **RUN** | **2.5–7.5（键盘钳 1.5–3.0）** | ❌ **从未按过 "3" 键** |
| 17 | FORWARD_JUMP | - | ✅ exp_all |
| 18 | STEALTH_WALK | - | ✅ exp_all |
| 8/19/20/22/24 等 | CRAWLING / INJURED / LEDGE / STEALTH_2 / ZOMBIE… | - | ❌ 未采 |

关键结构性事实：

1. **三个 locomotion mode 速度段首尾相接**：SLOW_WALK 0.1–0.8 → WALK 0.8–2.5 →
   RUN 2.5–7.5（注释值）。结构上即论文 trot/bound 拼全速度段的同构物。
2. **planner ONNX（V2，`planner/target_vel/V2/planner_sonic.onnx`）v1 层只支持
   4 mode：IDLE / SLOW_WALK / WALK / RUN**——RUN 是 planner 模型一级公民。
3. **键盘路径下 WALK 速度不可调**（无钳制分支 → 恒 -1 默认）；SLOW_WALK 与 RUN
   的 speed 由 '9'/'0' 键 ±0.1 连续可调（D029 实测生效）。
4. **单解码器是官方架构**：`policy/release/` 单一 encoder+decoder 服务全部 27
   mode；多 gait 结构内嵌于 planner/token 空间。RUN token 的解码质量由 D029
   零摔倒间接实证（闭环可用）。
5. **采集回路全资产在位**：MuJoCo sim（`run_sim_loop.py --interface sim
   --wbc-version sonic_model12_inspire`，`.venv_sim`）+ deploy（`g1_deploy_onnx_ref`
   + planner V2）+ pty 键盘驱动（exp1/2/3 模式）。
6. **步态×地形维度有既有实证但只测过 3/27**（MQ08/11/12）：开环下 stealth
   在 rough 0.08 存活而 walk 停摆（MQ08）；amp 0.14 上 crawl 3/3 不倒 vs
   walk 0/3 全塌（MQ11，不倒边界 ≥0.28）；触发式运行中切换 transition
   脆弱（MQ12，指向 ADAPTING 状态机 / 学习型选择）。**LEDGE_WALKING(20)
   等其余 mode 的地形表现从未测过**——harness（`planner_closed_loop.py`
   + rough heightfield）现成可扩。

**对既有结论的修正**：第一层边界（此前表述「SONIC 数据分布 vx 0–1.0」）作废——
那是采集子集的事实。材料上限修正为 planner 支持的 0.1–7.5 m/s（注释值，RUN 上
端待实测；键盘可达 1.5–3.0）。

## 2. D029 冒烟（2026-09-04，详见 `tracker/D.md`）

9 段键序全程 **0 falls**（MuJoCo 闭环）：WALK 基线 60s ✅；**RUN 默认 60s ✅
（target_vel 1.5）**；RUN 速度阶梯 '0'×2/×4/×6 → **1.7 / 2.1 / 2.7 m/s 全部站住**；
SLOW_WALK 阶梯贴 0.8 上限。数据落 `lab-ts:~/ros2_data/apt_g1/data/ds_smoke/`
（commands 42k 行 / policy_input 21k 行 / target_motion 21k 行 + deploy.log +
events.json）。**结论：RUN 模式与速度轴链路完全可用，当年只是没采。**

## 3. 下一步：采集网格设计（待 owner 过目后执行；09-04 修订：步态×地形轴升级为主轴）

- **轴 1（速度主轴，最高优先）**：SLOW_WALK {0.2…0.8 步长 0.1} × 前进；RUN
  {1.5…3.0 步长 0.15–0.3} × 前进。每点 ≥60s，段间 idle 10s + fall 计数。
- **轴 2（WALK 默认速度点加密）**：WALK 速度不可调，但「WALK 默认」本身是
  C1/C2 regime 的材料来源；按 exp3 方向网格（8 bin）在默认速度下补方向覆盖。
- **轴 3（步态×地形交叉探针，09-04 owner 输入后升级为主轴）**：不是「顺带
  探针」而是地形材料筛选主实验。**依据 = MQ 线既有证据链**：MQ11 在 walk
  盲重规划全灭的 amp 0.14 上 crawl 3/3 不倒（不倒边界 ≥0.28）、stealth 1/3——
  步态本身改变地形边界已被实证；当年结论即「缺按地形选模式的 gait selector
  （= APT 论文机制）」；MQ12 证明触发检测可行但**运行中切换 transition 脆弱**
  （切换后 1/3 存活 vs 从头 3/3）→ 学习型选择器（训练期拥有接口、从始适应
  切换）是 transition 问题的候选解——与 TO42 H1 可塑性论证同构。**未测空白
  = 27 mode 只测过 {walk, stealth, crawl} 三个**；LEDGE_WALKING(20)（名字
  即台阶行走）、STEALTH_WALK_2(22)、INJURED_WALK(19)、FORWARD_JUMP(17)
  的地形档全未扫。执行 = `planner_closed_loop.py`（MODE dict 扩键 + amp
  扫档 {0.08/0.12/0.14/0.20} × seed 3）+ 台阶地形（make_rough_xml 换
  heightfield 档位）× mode {crawl/ledge/stealth/stealth2/injured/jump}，
  产出 = per-(mode, terrain) 存活/前进矩阵 → 地形档位上的最优步态族即为
  VAE 步态条件轴的材料清单。
- **预算估计**：单机 MuJoCo 闭环实时比 1:1（D029 全程 ~10 分钟）；轴 1+2 全网格
  约 2–3 小时机时；轴 3 交叉矩阵（6 mode × 4–5 地形档 × 3 seed × 60s）约
  1–1.5 小时。
- **VAE 重训管线**：`train_token_vae_e39.py` 模式（数据目录换新采集合并集），
  vb 速度轴改为跨 mode 派生（SLOW_WALK+RUN 连续速度 bin），db 方向轴沿用；
  **若轴 3 筛出地形步态族，增设 mode/步态条件轴（结构待轴 3 结果定）**。
- **判读门**：新 VAE 训成后先复跑 E46 口径从零 RL（`--latent-mode` 128 envs）
  对照 E39 底板（vx 0.418 / 直行 0.86），再谈「快且直」天花板是否被推移；
  地形侧对照 = MQ11 per-mode 存活矩阵。

## 4. TO42 R1 封存记录（2026-09-04）

- 云端 wave 演进：v5（TASK_20260904_025，余额杀 signature）→ v6（032，9 分钟即
  死）→ **v7（TASK_20260904_035，on_done 修复 + 4 并发；11:43–13:19 跑 1.6h 后
  再次 archiving.sh signature 终止 = 余额杀；storage 空、仅 smoke-init ckpt，
  无可保全产物）**。R1 判读材料不足，selection-interface contrast 问题保持
  OPEN，不销案、不再投入。
- **后续候选原因 1（高并行微调重试）与 TO42 fbkt 臂大规模基线**随本转向一并
  冻结；若数据重采线产出新 VAE，learned-selection 议题可在新底板上重启（届时
  按 TO42_PLAN 配方移植，不重开旧账）。

## 5. 纪律与边界

- Run 行只进 `tracker/D.md`（D029 起）；本文档只做设计与叙事，不记数据行。
- 新脚本登记 `apt_g1/SCRIPT_MAP.md`（首件：`drive_ds_smoke.py`，D029）。
- 数据集 builder（ds_smoke csv → exp 格式合并）实现时另立脚本登记。
- mid-band（0.275–0.325）夹缝问题不受新材料直接影响（该段在 SLOW_WALK 段内）；
  本线的直接受益者是快段材料与「快且直」天花板——结论表述时不得混同。
