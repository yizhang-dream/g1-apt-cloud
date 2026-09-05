# Tracker 系列：E（E 系列（Isaac APT 主线 E01–E48，含 FB/I21/T1-T2 等辅助行与『地形/数据泛化/感知』混合节））

> 【层位 L3｜Run 台账·系列文件（数据唯一事实源）】↑ `refine-logs/EXPERIMENT_TRACKER.md`（总索引）与 `HANDOFF/02_EXPERIMENT_HISTORY.md`（L2 阶段史）｜↓ `HANDOFF/03_OUTPUTS_INDEX.md` → 服务器 `outputs/`（L4）｜≈ `apt_g1/SCRIPT_MAP.md`（代码轴）。
## 2026-08-12 夜（MuJoCo 收尾 + Isaac E21 + 力矩 + P2-lite）

| Run ID | Purpose | Variant | Data | Metric | Status | Result |
|--------|---------|---------|------|--------|--------|--------|
| MQ01 | MuJoCo 粗糙地形鲁棒性 | 本地 hfield ±0.02–0.10，v9/v6 walk+idle | 3 seeds × 8 amps | 1000 步存活/位移 | DONE | 0.00–0.02 3/3 → 0.03 2/3(v9) → 0.04 0/3 → 0.06 0/3(v6 1/3) → 0.08/0.10 0/3；按坡度对齐≈Isaac 阈值；见 `rough_mujoco_sweep.json` |
| MQ02 | hfield 碰撞修复 | MuJoCo 3.11 elevation/base_z 语义 | - | ncon | DONE | base_z 不参与碰撞，geom.pos=hmin；修复前穿地伪结果 |
| MQ03 | 力矩标签恢复 | exp_all3 → PD 力矩（12 下肢关节） | 14,633 行 | - | DONE | `data/torque_data/` |
| MQ04 | 力矩解码器 | phase+cmd→tau MLP | 14,633 | val RMSE | DONE | 18.76 N·m（≈反馈分量不可约） |
| MQ05 | 论文式力矩闭环 | tau_dec+PD(q_default) | 3 seeds | 存活/位移 | DONE | 平坦 3/3 存活但 vx≈0.03 不前进；rough 0.06 62–82 步倒 |
| MQ06 | 混合力矩闭环 | token PD + tau_dec | 3 seeds | 存活 | DONE | 平坦/rough 63–82 步倒（双倍反馈）→ PD 标签非规划力矩，路线死 |
| I21a | gate+map+anti-stop | rough 0.06 训练，it_400/500 | 64 envs | A60s | DONE | noaux 3/3（37–43m，同先验）；aux 2/3 慢走 vx0.14；0.08 上 0/3（同 E20c）→ 地图不能突破流形 |
| I21b | 先验×离散地形 | stairs/stairs_hi/stones/discrete（Isaac 内置） | - | A60s | DONE | stairs 3/3（18–22m）、stairs_hi 3/3（14–20m）、discrete 3/3（48m）、stones 0/3；E20c gate-only 同先验，gate+aux 台阶站住 |
| P21 | P2-lite 深度复原地图 | 单目深度→9×9 高程补丁 | 700–3077 帧 | val MAE/corr | DONE | CNN 0.047/0.74 → +GRU 0.041/0.87 → +跨地形 0.0265/0.965（=几何上界 0.0275） |

## 2026-08-12 深夜（优先级链：先平坦 → 再闭环）

| Run ID | Purpose | Variant | Data | Metric | Status | Result |
|--------|---------|---------|------|--------|--------|--------|
| FB01 | 平坦命令审计 | v9 全部 gmap 组 ×3 seeds ×20s | 本地 MuJoCo | 完成率/vx/disp | DONE | 稳定=idle、slow 0.2 bin0/1/2/6、slow 0.6 bin4、walk bin1/4、jump；不稳定=walk bin0/2/3/5/6/7、slow bin4/5、stealth；见 `flat_battery_v9.json` |
| FB02 | 显式回退固化 | StableResolver 数据驱动锚点（walk bin1/bin4、slow 0.6 bin4） | - | - | DONE | `router_fallback.py`；bin5 降级形式化为回退表 |
| FB03 | 回退全命令 battery | 24 命令 ×3 seeds | - | 无跌倒 | DONE | **24/24 3/3×20s 无跌倒**；slow_fwd→0.6 组 vx 0.56；walk_back 方向降级（bin1 锚点，bin0 原组 2/3） |
| FB04 | 60s+ 切换马拉松 | S1 无跳跃 65s / S2 含跳跃 50s | - | 完成率 | DONE | S1 **2/3**、S2 2/3；脆弱点在长运行后 walk_back（bin1 锚点 ~29.7s 倒）→ 多命令长跑残余闭环复合误差 |
| I22a | aux 判据实验（E13 风格） | 修正门控 + aux 600it | 64 envs | A/B/C/D | DONE | A aux 34–36m < 阈值 42.9m；B 持平；C/D aux 更差 → **判据不达标** |
| I22b | aux + 正则判据实验 | aux L2/rate + 紧 yaw | 64 envs | A/B/C/D | DONE | A aux 0.9–4.3m（原地振荡）；B/C/D 质量全差 → **判据不达标（更差）**；结论：aux 在该管道无正向价值，需力矩级解码器+TO 数据 |

## Isaac Lab APT 系列（2026-08-12，详见 ISAAC_APT_LOG.md）

| Run ID | Purpose | Variant | Config | A 60s | B 500N | C 68s | D 跳跃 | Status |
|--------|---------|---------|--------|-------|--------|-------|--------|--------|
| E1 | Isaac 基线 | 冻结路由器 aux=0 | num_envs=64 评测单env | 3/3，47m | 12/12 | 3/3 | 3/3 | DONE |
| E2 | aux PPO | gate ON，无扰动 | 500it，latent-KL+探索衰减 | 3/3，3–11m | 12/12 | 3/3 | 3/3（vx 0.57） | DONE |
| E4 | aux+扰动 | 旧语义（每步推）作废 | 800it | it_600：2–4m | - | - | - | DONE(作废) |
| E6 | aux 门控消融 | gate OFF | 600it | 3/3，16–32m | 12/12 | 3/3 | 3/3（vx 0.76） | DONE |
| E7b | aux+扰动+正则 | 修正扰动语义 | 800it | it_600：0/3 全倒 | - | - | - | DONE |
| E10 | aux 轻正则 | gate OFF，aux-l2/rate | 600it | 3/3，3–7m | - | - | - | DONE |
| E3 | phase+aux 联合 RL | 无 EMA | 800it | 不走路 | - | - | - | DONE |
| E3b | phase+aux | +EMA 0.3 | 800it | vx≈0.04 | - | - | - | DONE |
| E3c | phase+aux | +紧速度奖励 | 800it | vx≈0.07 | - | - | - | DONE |
| E8 | phase warm-start | 路由器相位监督→RL | 800it | it_300：3/3，vx 0.99，7–19m | B/C/D 评测中 | | | DONE |
| E9 | vanilla RL | 无先验 29d | 800it | 0/3 立即倒 | - | - | - | DONE |
| E11 | vanilla RL 长训 | 无先验 29d | 2000it | 0/3 立即倒 | - | - | - | DONE |

## 地形 / 数据泛化 / 感知（2026-08-12 续）

| Run ID | Purpose | Variant | Config | A 60s | 说明 | Status |
|--------|---------|---------|--------|-------|------|--------|
| T1 | 先验盲走鲁棒性曲线 | noaux，固定 terrain-seed 0 | rough 0.04/0.06/0.08/0.10 | 3/3、3/3、1/3、0/3 | 悬崖在 0.06→0.08 | DONE |
| T2 | 地形 seed 可复现修复 | 全局 np.random.seed 绑定 terrain seed | eval/train 脚本 | 0.06 两次 3/3 | 此前高度场用全局 np.random，seed 无效 | DONE |
| E15 | gate-fixed aux + 地形课程 | 0.04→0.06→0.08×300/300/400 | rough，无感知 | 0.08: 0/3；0.06: 2/3 | aux 无正向价值（纯 proprio） | DONE |
| D021+ | exp3 补采 walk 方向 | drive_exp3.py 官方闭环 | 14,580 步，0 跌倒 | - | walk bin1-3/5-7 全部补上 | DONE |
| D026 | 合并数据集 | build_exp3_dataset.py | exp_all3=68,093 步 | - | 含 slow 0.6 组 | DONE |
| D027 | v9 路由器重建 | train_phase_router_v9.py | exp_all3 | - | walk_fwd/back 3/3；新方向见 oracle | DONE |
| D028 | walk 方向 oracle 上限 | oracle_walk_bins.py 官方 token 回放 | 6 方向 | 全部 95–270 步内倒 | 教师本身做不了 walk+turn | DONE |
| E16 | aux + 特权 elevation map | 0.04→0.06→0.08 课程 | rough，9×9@0.15m 局部地形 | 0.06: 0/3；0.08: 0/3（负结果，见下方 E16 完成节） | elevation 无 aux 表达通道 | DONE |

### E16 完成（2026-08-12）

| Run ID | Purpose | Variant | Config | A 60s | 说明 | Status |
|--------|---------|---------|--------|-------|------|--------|
| E16 | aux + elevation map | 0.04→0.06→0.08×300/600/1000 | rough seed0 | 0.06: 0/3；0.08: 0/3 | elevation 无 aux 表达通道 → 负面结果 | DONE |
| E16b | 同场景 noaux | 冻结路由器 | 同上 | 0.06: 3/3；0.08: 1/3 | 对照 | DONE |

E16 结论：特权地图只进 aux 通道无正向价值（0.06 反而 2/3→0/3）；论文的
elevation→latent/gait 选择通道是缺失的关键机制。详见 `ISAAC_APT_LOG.md`。

| Run ID | Purpose | Variant | Config | A 60s | 说明 | Status |
|--------|---------|---------|--------|-------|------|--------|
| E17 | 策略学习组选择（gate logit）+ aux + elevation | 0.04→0.08 课程 | rough seed0 | aux 0/3；noaux 3/3 但站住 | idle 坍缩（奖励允许） | DONE |
| E17b | E17 + 前进进度奖励 | progress_scale=0.3 | rough seed0 | aux 0/3；noaux 站住 | 能移动但无法稳定 | DONE |

E17/E17b 结论：给策略组选择通道后，最小化奖励下学会站住规避摔倒；加进度奖励
后恢复移动但粗糙地形仍 0/3。论文的步态选择机制依赖"必须前进"的任务/奖励设计。

| Run ID | Purpose | Variant | Config | A 60s | 说明 | Status |
|--------|---------|---------|--------|-------|------|--------|
| E18 | phase 直控 + aux + elevation + progress | phase-mode warm150 | rough seed0 | 0.06/0.08 均 0/3（~2s 倒） | 相位偏离路由器先验 | DONE |
| E19 | 全程相位锚定 + aux + elevation + progress | warm1000 coef2.0 | rough seed0 | 3/3、3/3 存活 | 先验流形内微调（倒走） | DONE |
| P1 | 感知蒸馏机制演示 | 学生复原特权地图 | 粗3x3+噪声 -> 细9x9 | - | 论文 stage4 机制（corr 0.954） | DONE |

### E19/E19c/P1 完成（2026-08-12）

| Run ID | Purpose | Variant | Config | A 60s | 说明 | Status |
|--------|---------|---------|--------|-------|------|--------|
| E19 | phase 锚定 + aux + elevation | warm1000 coef2.0 + progress | rough 0.06/0.08 | **3/3、3/3 存活** | 首个存活达标的 learned 策略，但倒走/站住 | DONE |
| E19b | E19 phase-only 对照 | aux=0 | rough | 3/3 存活（站住） | 存活来自相位锚定而非 aux | DONE |
| E19c | E19 + aux 正则 | aux-l2 0.01/rate 0.005 | rough 0.06/0.08 | 1/3、0/3 | 正则破坏存活；平坦前进 8s | DONE |
| P1 | 感知蒸馏机制演示 | 学生复原特权地图 | 粗3x3+噪声 → 细9x9 | - | corr 0.954，MAE 0.0085m | DONE |

E19 系列结论：先验流形约束（相位锚定）是 learned 组件存活的关键；"存活 vs
前进任务"权衡下最小奖励选择不倒；完整论文奖励 + 地形感知 + 千级并行是继续
提升所需的条件。详见 `ISAAC_APT_LOG.md`。

| Run ID | Purpose | Variant | Config | A 60s | 说明 | Status |
|--------|---------|---------|--------|-------|------|--------|
| E20 | phase 锚定 + anti-stop | anti-stop 1.0（thresh 0.3） | rough + flat | 0.08: 2/3 爬行；flat 快但画圈 | 速度/存活冲突 | DONE |
| E20c | gate + anti-stop | anti-stop 1.0（thresh 0.1） | rough | aux 0/3；**gate-only 0.06 3/3 前进** | gate 学对、aux 破坏 | DONE |

E20/E20c 结论：anti-stop 让 gate 头收敛到选择先验 walk_fwd 组（aux=0 时行为
与冻结先验一致）；aux 通道始终是破坏源。现有管道最优 = 冻结先验 + gate 选择
先验组。详见 `ISAAC_APT_LOG.md` 与 `FINAL_REPORT.md`。

## 方向 A/B/C 收尾（2026-08-13）

| Run ID | Purpose | Variant | Config | 结果 | 说明 | Status |
|--------|---------|---------|--------|------|------|--------|
| A-ID | 逆动力学力矩数据+解码器 | mj_inverse 重放轨迹算 ID 力矩 | 27k 行，1/99 百分位裁剪 | val MAE 4.13 N·m；混合×0.2/0.3 平坦 3/3 与基线持平 | 论文式纯 ID 前馈 2.5s 倒；粗糙无增益；ID 来自带位置控制器的重放，非自洽规划力矩 | DONE |
| E23 | 连续潜空间 RL（相位插值） | 64env×800it warm200 | flat | 3/3 存活但 disp 0.8–2.0m | 原地振荡/画圈，C 切换 0/3；连续读取代平滑不改流形上限 | DONE |
| S128 | Isaac 并行压力测试 | 128 envs×300 iters | flat | dt 0.90s（64env 0.72s），吞吐 +61% | 显存 2.8GB / GPU 62% 未用满，瓶颈在更新/同步开销 | DONE |
| E24 | 相位 + anti-stop + progress | E23 + anti-stop 1.0@0.1 + progress 0.3 | flat | A 3/3 但 disp 9–12m；**B 12/12、C 3/3、D 3/3** | 鲁棒性恢复但慢（vx 0.3）；自由相位学不会步态时钟 | DONE |
| M1024 | mjlab 官方配方冒烟 | unitree_rl_mjlab Unitree-G1-Flat | 1024 envs | 1.03s/iter，显存 0.84GB | 千级并行在 3060 上可行；从零学速度跟踪的独立对照轨道 | DONE |
| E25 | 相位锚定（路由器时钟+有界偏移） | E24 奖励，EMA 时钟 | flat | A 20–28m（aux 开）；**aux 归零 49.2–49.5m，B 12/12、C 3/3、D 3/3** | 首个学习型全项不劣于冻结先验；aux 是唯一破坏源 | DONE |
| E26 | 纯相位偏移 RL | E25 + aux_scale=0 | flat | **A 45.7–46.1m（≈97%），B 12/12、C 3/3、D 3/3** | 方向 B 闭环：连续潜空间无损但机制中性，aux 仍唯一破坏源 | DONE |
| E26-T | E26 地形扩展 | rough 0.06/0.08 × seed 0/1 | rough | 0.06：3/3（22.7–42.5m）；0.08：偏移 0/6、纯时钟 2/6 | 0.08 是蒸馏路径（无 planner 重规划）共同能力边界；偏移地形无增益〔归因修正见 MQ09〕 | DONE |
| E27 | latent→VAE→SONIC（无行为先验） | 相位条件化 VAE + z_walk warmstart | flat | A 3/3 19.1m（vx 0.32）；B 12/12、C 3/3、D 3/3 | 用户指定缺失实验：token 流形本身是关键先验；E9/E11 直出关节 0/3 | DONE |
| M-FROM0 | mjlab 官方从零训练 | Unitree-G1-Flat 1024 envs，1500 iters + resume 5000 iters（共 6500） | flat | **DONE：总 6500 iters**（resume 自 model_1500.pt，新 run dir `logs/rsl_rl/g1_velocity/2026-08-14_00-52-58/`；末迭代 fell_over **0.125** / time_out 1.5833 / error_vel_xy 1.1695 / error_vel_yaw 1.7973；产出 model_6499.pt + policy.onnx）。**原生任务评测**（`eval_mjlab_fwd.py`，自家 sim、60s 直行命令 vx0.8、3 seeds）：flat **3/3 存活 44.5–48.0m**（vx 0.75–0.80）；rough 0.06：27.6/43.6/**49.3m**（1/3 走满 60s，2/3 38–53s 倒）；rough 0.08：15.1/37.1/**44.8m**（1/3 走满 60s，2/3 21–53s 倒）——**地形退化平滑，自家 sim 里无 0.08 悬崖**（vs 我们 Isaac 管道 0.08 全方案 0/9–0/12 全倒，至少部分是 SONIC 解码器/Isaac 地形配方特有） | **非 SONIC 官方从零配方对照**（无 SONIC 解码器/路由器/token）。注意：它不隔离"蒸馏"——同时换了解码器、路由器、训练配方/堆栈；"蒸馏是否必要"由管道内 E27（无路由器 19m）vs 冻结先验（47m）回答 | DONE |

A/B/C 三方向结论（详见 `DATA_GENERALIZATION_LOG.md` 第 13–15 节、
`ISAAC_APT_LOG.md` E23 节）：
1. 力矩级数据能改善前馈质量（ID MAE 4.13 vs PD ~9.4），但单独不能解锁论文式
   控制或地形鲁棒性，需自洽 TO/逆动力学"规划"数据。
2. 连续潜空间（相位插值）机制无损，但 RL 相位调制无前进压力时坍缩为
   "存活不前进"，不改变流形上限。
3. 千级并行正解是 mjlab（MuJoCo-Warp）官方 4096-env 配方，独立于蒸馏管道；
   当前 Isaac 管道在 3060 上 128 envs 仅 +61% 吞吐，显存/算力未用满。

方向 B 续（E24/E25）与 C 冒烟：E24 证明连续相位 + 前进压力能恢复鲁棒性但
速度不达标（自由相位学不会步态时钟）；E25（相位锚定+有界偏移，训练中）是
针对该根因的修复。mjlab 1024 envs 冒烟通过，千级并行验证达成。

方向 B 最终（E26）：纯相位偏移 RL 在 aux 全程关闭下 A 45.7–46.1m（冻结
先验 47.3–47.7m）、B 12/12、C 3/3、D 3/3——首个学习型全项达标。相位偏移
通道机制中性；aux 关节偏移仍是唯一破坏源，论文 aux 正向价值仍需力矩级
解码器 + TO/逆动力学数据。

## 方向 D：速度/方向条件化潜空间（E28–E35，2026-08-13 下午，详见 ISAAC_APT_LOG.md）

| Run ID | Purpose | Variant | Config | 结果 | 说明 | Status |
|--------|---------|---------|--------|------|------|--------|
| E28a | E27 速度天花板归因消融 | +解冻步频（命令条件化相位速率） | latent_cmd_phase_rate，默认保持 E27 | mean vx 0.342 / disp 13.1m（比 E27 19.1m 跌 31%） | vx 微涨 +8% 但 disp 暴跌 → 方向漂移；解冻步频不够 | DONE |
| E28b | +解冻步频 + 奖励重调 | anti_stop_thresh 0.1→0.6、progress 0.3→0.5、stillness_vx_scale 0.05→0 | 同上 + 奖励重调 | mean vx 0.253 / disp 14.9m | 更慢 −20%；速度不是激励不足 → ~0.3 是解码器流形固有速度上限 | DONE |
| E29 | latent KL 先验对齐 walk 流形 | E27 + `--latent-kl-prior walk --latent-kl 1e-2` | 其余同 E27（64 envs×800 it） | mean vx 0.348 / disp 18.3m（KL 55） | **首个正向杠杆**：z 靠近 walk 流形 → 更快更稳；天花板仍 ~0.35 | DONE |
| E30 | KL coef 扫描（更硬钉向流形） | E29 + KL coef 1e-1 | 其余同 E29 | mean vx 0.336 / disp 16.0m（实测 KL 46.7） | KL 扫描非单调，峰值在 E29（coef 1e-2）；过度钉 z 到 z_walk 均值反而略差 | DONE |
| E31 | 速度条件化 VAE 解码器 | `SpeedPhaseTokenVAE`（3 档速度 bin，D(z,φ,v_bin)→token） | E29 配方 + speed-bins | val recon MAE 0.0753；mean vx **0.535**（+54% vs E29）/ disp ~10m | **打破 ~0.35 上限**！但严重左转漂移（yaw −27°@~4°/s，系统性转向偏置） | DONE |
| E32 | yaw×2 + heading 奖励修漂移 | E31 + yaw_scale 0.5→2.0 + heading_scale 0.8 | 其余同 E31 | mean vx 0.354 / disp ~10.3m | **失败**：heading 强化把速度压回 0.35，漂移没修好（disp/vx 仍 0.5）；系统性转向偏置非奖励可修 | DONE |
| E33 | 开环 yaw 补偿 | eval `--yaw-bias-comp`（命令 yaw +0.07 rad/s） | 其余同 E31 | disp 仍 ~9.6m（无变化） | **失败**：policy 观测含命令 yaw → 主动抵消补偿 | DONE |
| E34 | yaw 域随机化 | train `--yaw-min/max`（命令 yaw U(−0.4,0.4)） | 其余同 E31 | mean vx 0.395 / disp ~7.4m（disp/vx ~0.31 甚至更差） | **失败**：转向偏置是该流形技能固有特征，非奖励/补偿/DR 可修 | DONE |
| E35 | 方向条件化 VAE 解码器 | `DirSpeedPhaseTokenVAE`（8 档方向 bin，D(z,φ,v,ψ)→token） | E31 配方 + dir bins | val recon MAE 0.062（优于 E31 0.0753/E27 0.079）；mean vx 0.295 / disp 16.3m / disp-vx 比 0.93 | **漂移修复**（接近直行）但速度回落到 ~0.3；方向是流形显式轴机制正确，快且直仍需速度激励（E36） | DONE |
| E36 | 快且直：速度激励推选快 bin | E35 + `--anti-stop-thresh 0.4 --progress-scale 0.5` | E35 配方 + 提高速度激励（64 envs×800 it，seed0） | A **3/3 vx 0.372 / disp 15.4m / 直行 0.69**；B **12/12**（h_min 0.73–0.76）；C **3/3**（disp ~11.2m）；D 3/3（vx 0.08 站住） | vx +26%（0.295→0.372）但直行 0.93→0.69、未达快 bin 0.535 → 速度-方向权衡是流形固有项（推快 z 重新引入 E31 型转向偏置）；B/C/D 鲁棒性完整 | DONE |
| E37 | 快且直：方向解耦 VAE（干净快 bin） | E36 配方 + 方向解耦 `DirSpeedPhaseTokenVAE`（对抗式方向分类头：adv_weight 3.0 + 类平衡 CE + 3 dir steps） | E36 配方 + token_vae_e37（64 envs×800 it，seed0） | A **6/6 vx 0.370 / disp 21.55m / 直行 0.97**；B **12/12**（h_min 0.73–0.76）；C **3/3**（disp ~9.9m）；D 3/3（vx 0.11 站住） | VAE：val MAE 0.0737（vs E35 0.062）；z→方向 fresh 分类器 acc 0.707≈多数类 0.702（弱解耦版 0.769）。**方向解耦修好漂移**：同 vx 0.37 下 disp 15.4m→21.55m、直行 0.69→0.97 → 快且直达成（0.37 m/s 档，未达 E31 的 0.535） | DONE |
| E38 | 快且直：更高速度激励推满速档 | E37 配方 + `--anti-stop-thresh 0.5 --progress-scale 1.0` | E37 配方 + token_vae_e37（64 envs×800 it，seed0） | A **6/6 vx 0.399 / disp 19.4m / 直行 0.81**（h_min 0.747–0.757）；B **24/24**（h_min 0.735–0.757）；C **6/6**（fall=None，disp ~10.8m）；D **6/6**（vx ~0.13 站住） | **更高激励只换 +8% 提速（0.370→0.399）却让直行 0.97→0.81、净位移 21.55→19.4m 下降**。方向解耦修好 0.37 档漂移，但**不是解锁 0.535 快 bin 的钥匙**：解耦后 z 速度天花板 ≈0.40，低于 E31 未解耦的 0.535 → 速度天花板是独立于方向纠缠的另一约束；B/C/D 鲁棒性完整 | DONE |
| E39 | 快且直：速度+方向双解耦 VAE | E38 配置 + 双对抗头 `token_vae_e39`（dir_head z→8 + speed_head z→3，adv_dir 3.0 + adv_spd 3.0，类平衡 CE） | E38 配置 + token_vae_e39（64 envs×800 it，seed0） | A **6/6 vx 0.417 / disp 24.65m / 直行 0.98**（h_min 0.752–0.756）；B **24/24**（h_min 0.737–0.755）；C **6/6**（fall=None）；D **6/6**（vx ~0.165） | **假设成立：速度确实纠缠在 z 里**——同一高激励下 E38（仅方向解耦）直行掉到 0.81，E39（双解耦）vx 0.417（+5% vs E38）且直行 0.98、disp 24.65m（+27% vs E38 19.4m、+14% vs E37 21.55m）= **潜空间线历史最佳位移**；VAE val MAE 0.0770，探针：方向 0.722≈多数类 0.702、速度 0.575≈多数类 0.573 | DONE |
| E40 | 快且直：双解耦 × 更高激励探速度上限 | E39 配置 + `--anti-stop-thresh 0.6 --progress-scale 1.5` | E39 配方 + token_vae_e39（64 envs×800 it，seed0） | A **6/6 vx 0.456 / disp 24.40m / 直行 0.89**（h_min 0.739–0.754）；B **24/24**；C **6/6**（fall=None 但 disp ~3m vs E39 ~10m）；D **6/6**（vx ~0.08，disp ~0.7m vs E39 ~3.3m） | **E39 仍是甜点**：更高激励只换 +9% 速度（0.417→0.456）却让直行 0.98→0.89、C/D 机动性明显退化（切换马拉松 disp 10m→3m、跳跃站住）→ 双解耦 z 的 Pareto 前沿 ≈(0.42, 0.98)↔(0.46, 0.89)，0.535 仍不可及 | DONE |
| E41 | 地形泛化：E39 甜点策略上 rough 0.06/0.08 | E26-T 同口径（A 60s × terrain-seed 0/1 × noise 0.06/0.08），checkpoint = E39 | 评测 harness 同 E26-T（--tests A --latent-dir-bins） | rough 0.06：**6/6 存活**（vx 0.42–0.50，disp 11.4–26.1m，h_min 0.742–0.752）；rough 0.08：**s0 0/6 全倒**（fall_step 401–632）、**s1 0/6 全倒**（fall_step 645–1195），h_min≈0.20 | **0.08 悬崖对所有方案一致**：E39 平坦"快且直"不迁移到 0.08 地形。**〔归因修正 2026-08-14〕**E39 是蒸馏路径（相位 VAE，无官方 planner 10Hz 重规划），故"0.08=冻结解码器边界"是过度归因——MQ09 证明同一冻结解码器 + planner 重规划在 MuJoCo rough 0.08 上 walk≈flat（3.38m≈3.39m/6s），悬崖更可能来自"蒸馏路径缺重规划"；跨 sim/时长口径未对齐，Isaac 侧需补"蒸馏策略+planner 重规划"对照定论；0.06 通过（与 E26-T 各线持平） | DONE |
| E42 | 地形上训练：双解耦策略在 rough 0.06 训练 | E39 配方（thresh 0.5 + progress 1.0）+ `--terrain rough --terrain-noise 0.06 --terrain-seed 0` | E39 配方 + token_vae_e39（64 envs×800 it，seed0，训练地形 0.06） | 0.06：**6/6 存活**（vx 0.40–0.45，disp 15.1–24.1m，均值 19.1m）；0.08：**s0 0/6、s1 0/6 全倒**（fall_step 358–966） | **训练地形匹配不破 0.08 悬崖**：地形上训练只让 0.06 位移 18.6→19.1m（噪声内），0.08 仍 0/12 全倒 → 训练分布确非主因；**〔归因修正 2026-08-14〕"0.08=冻结解码器边界"收回**，应为"蒸馏路径（无 planner 重规划）边界"（见 MQ09）；地形问题以 MQ09 重规划为准重新开放 | DONE |
| E43 | 快区加权方向解耦：清掉快 bin 残留方向纠缠 | per-bin 探针归因（快区泄漏 +0.069）+ `fast_extra 2.0` 方向对抗加权 | E40 配置（thresh 0.6 + progress 1.5）+ token_vae_e43（64 envs×800 it，seed0） | A **6/6 vx 0.347 / disp 13.5m / 直行 0.65**（h_min 0.746–0.759）；B **24/24**；C **6/6**（fall=None）；D **6/6**（vx ~0.077） | **预测失败（重要负结果）**：快区泄漏 +0.069→+0.018（VAE val MAE 0.0734 无损）后，高激励下反而更慢更歪（vx 0.347 vs E40 0.456、直行 0.65 vs 0.89）→ 过度挤压快区损伤 z 流形表达能力；**E40 漂移主因不在 z 快区残留泄漏**，而在解码器侧（快 bin 嵌入/数据转向偏置）或快速步态本身 → z 级解耦线以 E39 为最优收束 | DONE |
| E44 | 解码器微调（decft）：PPO 直接训练 SONIC 解码器 | 动作空间=29 维关节目标；策略 = E39 z头(16) → 冻结 token_vae_e39 → token → **可训练 SONIC 解码器**(官方权重初始化) → μ(29) + 高斯噪声；PPO 评分梯度经 log N(a;μ,σ) 到达解码器；正则 = MSE(decoder, 官方 decoder)（`--decoder-reg 1.0`）；课程 0.04→0.06→0.08 分三段续训（E44a/b/c） | E39 配方（thresh 0.5 + progress 1.0 + latent-kl-prior walk）+ `--decft`，64 envs×rollout 24，seed0，warm-start 自 isaac_e39_dualdecouple/policy_it_800.pt | **v1 失败（负结果，机制已验证）**：训练 3 段全跑完（a: it800–1100@0.04, b: it1100–1400@0.06, c: it1400–1680@0.08 后 NaN 崩溃），dec_dw 0→46（权重平均偏移>初始化尺度）、dreg≈0.5–2（输出完全离开官方流形）→ 评测 0/9 全倒**含平地**（fall_step 65–68≈1.3s，v_speed 0.75 但 vx≈0.06/disp≈0 = **原地高速打转退化步态**）。根因：σ_a=0.03 使 PPO 梯度尺度放大 ~1/σ²≈1100 倍，λ=1.0 输出正则压不住 → 解码器失约束漂移；训练期 fall_rate 1–2%（平均 ~2s 一倒）即征兆。**结论：梯度机制生效（解码器确实被 RL 移动），但先验约束是承重墙——弱正则=退化**。v2 修复：decoder_lr 1e-4（÷3）+ decft-aux-std −2.0（σ=0.135，梯度÷22）+ decoder-wreg 1e-3（权重空间锚定）+ decoder-reg 2.0，重跑课程，阶段间加平地评测门 | DONE（v1） |
| E44v2 | 解码器微调 v2：约束版 | 同上 + 权重空间锚定 + 小解码器 LR + 大动作噪声 | E44 配方 + `--decoder-lr 1e-4 --decoder-reg 2.0 --decoder-wreg 1e-3 --decft-aux-std -2.0` | **失败**：约束生效（dec_dw 6–7.5 vs v1 16–24、dreg 0.14–0.3 vs v1 0.5–2、fall 0.1–0.7%），但平地评测门仍 0/3（fall ~2.5s，vx 0.6–1.05 但 disp≈0 = 仍高速打转）。约束只推迟了退化，未消除 | DONE |
| E44v3 | 解码器微调 v3：偏航角速度惩罚 | E44v2 配方 + `--yaw-rate-penalty 1.0`（直接惩罚 ωz²，压打转） | E44v2 配方 + yaw 惩罚 | **失败**：惩罚过强 → 奖励崩到 −2、价值损失爆 400+，训练失稳；解码器仍打转但奖励全被惩罚吞掉。yaw 惩罚是 E32–E34 已记录的"奖励再工程"泥潭的复现 | DONE |
| E44-ctrl | 冻结解码器对照（decft 路径 + 官方解码器） | DecFtPolicy + 官方解码器权重 + E39 sd partial-load，平地 A 60s | 平地，3 seeds | **3/3 完成、disp 12.75–13.0m、vx 0.29、h_min 0.74、无倒** | **证明 decft 代码路径无 bug**（harness 正确）；打转/摔倒 100% 来自解码器漂移。**同时暴露暖启动 bug**：E39 encoder 105 维 vs DecFtPolicy encoder 1023 维，partial load 形状不匹配 → E39 z 头未传入（随机 z），故对照组 vx 0.29<0.42、且 v1/v2/v3 是从随机 z 头起步训解码器（混杂因素）。E39 大脑→DecFtPolicy 的忠实迁移需解决 obs 维度差异（prev-z 反馈 16 维 vs 相位 2 维） | DONE |
| E44p | 解码器微调两阶段（先冻结训 z，再解冻课程） | **p1b**：`--freeze-decoder --decft-phase-std -1.0`（大 z 噪声修复 z 头探索）+ E27 基础奖励，平地从零训 z 800 it；**p2**：resume 解冻解码器（v2 约束），0.06→0.08 课程 | p1b 平地 / p2 课程 | **p1b 平地门 3/3 完成、disp 18.3–18.4m、vx 0.36、无倒**（z 头修复成功）；**p2 最终评测 0/18 全倒含平地**（0.08 s0/s1 各 0/3、0.06 0/3、平地 0/3，均 disp≈0.001 打转、fall ~2.2s；训练期 dec_dw 6.4→9.5 受控但输出仍漂移 dreg 0.26–0.36） | **两阶段消除"随机 z"混杂后，解冻解码器仍把直行破坏成打转** → 结论收敛 | DONE |
| E44 总结 | 解码器微调全变体（v1/v2/v3/two-phase） | PPO 直接训 SONIC 解码器（29 维关节目标动作空间） | — | 全变体 eval 均打转摔倒（含平地） | **稳健负结果**：梯度机制生效（解码器确实被 RL 移动），但**任何程度的解码器微调都会重新激活 SONIC 快走技能的固有转向偏置（E31–E34），把直行破坏成原地打转**，且奖励（track_xy/progress 用 body 系 vx）对打转惩罚不足 → RL 找到"打转+2s 倒"的奖励漏洞。**冻结解码器（+router/方向 bin VAE）的直行鲁棒性是承重墙，不是可有可无的初始化**。0.08 悬崖在本蒸馏/RL 堆栈内无法顶开，但它是"蒸馏路径（无 planner 10Hz 重规划）"的上限，非解码器流形固有上限（归因修正见 MQ09） | DONE |
| 渲染管道升级 | MuJoCo 3D 真实模型视频 | `replay_render_mujoco.py`（回放 npz → MuJoCo G1 offscreen，MUJOCO_GL=egl） | 400 帧/25fps/640×360 | e29/e31/e35_mujoco.mp4 | 骨架动画 `animate_skeleton.py` 降级为调试辅助 | DONE |

E28–E35 方向（详见 `ISAAC_APT_LOG.md` E28–E35 节）：
1. E28–E30 归因消融：解冻步频/奖励重调/流形-KL 三轴都无法把 vx 推过 ~0.35 →
   **~0.3 m/s 是冻结 walk 解码器流形的固有速度天花板**（修正 E27 结论 #2：
   半速=流形天花板，不是样本效率）。
2. E31 速度条件化 VAE 打破 ~0.35 上限（vx 0.535）但自带系统性转向偏置；
   E32–E34 证明该偏置非奖励/开环补偿/命令随机化可修。
3. E35 方向条件化 VAE 修复漂移（disp/vx 0.93）但速度回落 ~0.3 → 快且直需
   速度激励（E36）。
4. **E36（E35 + anti-stop-thresh 0.4 + progress-scale 0.5）**：A vx 0.372
   （+26% vs E35 0.295）/ disp 15.4m / 直行 0.69，B 12/12、C 3/3、D 3/3。
   速度-方向权衡是流形固有项：推快 z 重新引入 E31 型转向偏置（直行 0.93→0.69），
   未达快 bin 0.535。**简单奖励速度激励不能实现"快且直"**。
5. **E37（方向解耦 VAE + E36 速度激励）**：A vx 0.370 / disp 21.55m / 直行
   0.97，B 12/12、C 3/3、D 3/3。**方向解耦修好漂移**：同 vx 0.37 下 disp
   15.4m→21.55m、直行 0.69→0.97 → 快且直达成（0.37 m/s 档）。根因确认：
   E36 的漂移来自 z 隐式编码方向；把方向做成显式轴 + 让 z 方向不变（对抗式
   解耦，fresh 分类器 0.707≈多数类 0.702）即可消除。未达 E31 的 0.535，仍需
   更高速度激励（E38）。

## 从零 RL + 冻结解码器（2026-08-14，方向转变：抛弃 planner、务必用解码器）

| Run ID | Purpose | Variant | Metric | Status | Result |
|--------|---------|---------|--------|--------|--------|
| E45 | 从零 latent RL + 冻结 SONIC 解码器（无 warm start） | `--env apt --latent-mode --latent-vae-path e27/vae.pt`（策略→16d z→冻结VAE→token→冻结解码器），128 envs×2000 iters，最优 ckpt it_200 | A 60s ×3 seed | DONE | **3/3 完成、fall=null、h_min 0.755–0.760（直立）、disp 13.9–14.1m、vx 0.27 m/s** |

**E45 结论（"务必用解码器"的实证）**：

| 路线 | 动作层 | A 60s 结果 |
|---|---|---|
| vanilla 从零 RL（已弃） | 29 维关节直出（无解码器） | **蹲蹭漏洞**：h_min 0.20、vx 0.5–0.65 假速度、**disp=0** |
| **E45 从零 latent RL** | 冻结 SONIC 解码器 | **真走路**：h_min 0.76、disp 14m、3/3 不倒 |

**核心发现**：① **冻结解码器防作弊**——把动作锁在正常步态流形里，策略只能选 token、
没法学出蹲蹭（对比 vanilla 的 h_min 0.2/disp 0 vs E45 的 h_min 0.76/disp 14m）。这是
"蒸馏先验必要"的直接实证。② **从零（随机 z、无 warm start）也能学会走路**——disp 14m
约为 E27（z_walk warmstart，19m）的 73%，证明解码器提供步态、策略只需学会"选对 token"。
③ 训练在 iter 450 后**撞上冻结解码器速度天花板**（vx 涨到 0.7 但 rew 从 1.7 掉到 0.7、
kl 56→100，z 漂离 walk 流形），复现 E28–E30 的 ~0.35 m/s 上限——最优 ckpt 在 it_200
（rew 1.73）。

| E46 | 从零 latent RL + E39 速度条件化 VAE（无 warm start，破速度天花板） | `--latent-mode --latent-vae-path e39/vae.pt --latent-speed-bins --latent-dir-bins --latent-kl-prior zero --progress-scale 1.0`，128 envs×2000 iters，最优 ckpt it_200 | A 60s ×6 seed | DONE | **6/6 完成、fall=null、h_min 0.743–0.753（不蹲）、vx 0.418、disp 21.6–21.8m、直行 0.86** |
| E47 | 从零 latent RL + E39 双解耦 VAE + 轻 heading（E46 + heading 0.4，从零线收束最优） | E46 配方 + `heading_scale 0.4`，128 envs×2000 iters，最优 ckpt it_500（服务器 `apt_g1/outputs/isaac_e47_heading/policy_it_500.pt`） | A 60s ×6 seed + B/C/D 电池 | DONE | **A 6/6 完成、fall=null、disp 23.8m、vx 0.42、直行 0.944；B 500N 冲击 24/24（h_min 0.735–0.757）；C 切换马拉松 6/6；D 6/6（vx ~0.165 站住；D「跳跃」为假阳性——E39 VAE 只覆盖 WALK token、模式命令被忽略，见 FINAL_REPORT 剩余方向 #2）**；轻 heading 修直行且不伤速度，从零（无 warm start）逼近 E39 walk 先验版。视频 `e47_mujoco.mp4`（2026-08-27 补渲染） |
| E47-T | E47 × rough 0.06/0.08 × terrain-seed 0/1（从零最优控制器地形泛化） | eval 同 E26-T 口径（`--tests A --latent-dir-bins`） | rough 0.06/0.08 × seed 0/1 | DONE | **0.06：12/12 存活；0.08：0/12 全倒**——0.08 悬崖对从零最优控制器同样成立（〔MQ09 归因修正后〕悬崖 = 蒸馏路径边界，非解码器/从零本身可解） |

## E48：冻结解码器 + 全关节残差（RuN/ReSkill 式，文献综述解法 2 的实证，2026-08-15）

> `LITERATURE_SURVEY_FROZEN_DECODER.md` 解法阶梯第 2 级：我们已判死的 aux 是"弱残差"
> （12 维下体、scale 0.2、64 envs）；文献说残差要**全关节 + 地形输入 + 足量 envs**。
> E48 按正确配方重开：action = [z(16), res(29)]，q_des = q_decoder(z) +
> res_scale·clamp(res)（**全 29 关节**），+ 特权 elevation obs（`--use-elevation 1`），
> 128 envs × 2000 iters，训练地形 rough 0.06（E42 同款地形匹配）。基础配方复刻 E47
> （e39 VAE + 双 bin + progress 1.0 + heading 0.4）。代码：`apt_flat_env.py`
> （`latent_residual`/`res_*`）、`train/eval_apt_isaac.py`（`--latent-residual` 系旗标；
> eval 的 noaux 臂 = 残差清零消融）。

| Run ID | Purpose | Variant | Metric | Status | Result |
|--------|---------|---------|--------|--------|--------|
| E48 | 全关节残差 + elevation + 地形训练（残差 scale 0.4、res_l2 1e-3） | 从零 z + 残差同时学，128 envs×2000 it，最优 ckpt it_700（rew 1.49） | A 60s × {平地, 我方0.06/0.08, 论文0.06, 对称0.06} × 残差开/关 | DONE | **残差开 = 全地形破坏**：平地 3/3 完成但 disp 仅 1.7–2.6m（微蹲 0.69、原地蹭）；其余 4 组 **0/3 全倒**。**残差关 = 先验完好**：平地 3/3 disp 18.7m、我方 0.06 3/3（6.9–17.7m）、**我方 0.08 1/3 存活（seed1 走 26.8m，蒸馏路径首个 0.08 存活样本）**；论文/对称 0.06 0/3（与 G0 一致，坑致命） |
| E48-ck | 早期 ckpt 抽查（it_200，残差尚小时期） | 同 ckpt 电池 {平地, 0.06, 0.08} | 残差开存活 | DONE | **残差开平地也 0/3 全倒**（vx≈0，即刻倒）→ 残差破坏性从训练早期即存在，非后期利用；残差关 it_200：平地 19.0m / 0.06 2/3 / 0.08 0/3 |
| E48c | 对症修正：先冻残差立基座再放开（ReSkill 忠实姿势） | `--res-freeze-steps 19200`（前 800 iters 残差置零）+ res_scale 0.15 + res_l2 2e-2，其余同 E48；评测 ckpt it_800（放开前）/ it_1000（放开后 200 iters） | 同电池 {平地, 0.06, 0.08} × 残差开/关 | DONE | **基座健康**：it_800 平地 aux 3/3 **28.0m**（残差头因冻结未受有效梯度、均值近零反成微扰动）/noaux 19.8m，0.06 noaux 3/3（19.9–23.0m）；**放开即劣化**：训练 rew 2.1→−1.6→−4.8（vx 冲 1.33 同款冲刺漏洞），it_1000 残差开 0.06 开始摔（2/3）；**0.08 全倒**（it_800 0/6、it_1000 0/6） |

**E48 结论（残差通道负结果 + 0.08 首个存活样本）**：

1. **全关节残差（scale 0.4、弱 L2、从零 z）不是逃逸通道，是破坏源**：残差开时全地形
   劣化（平地 24m→2m、0.06 3/3→0/3），且 it_200 抽查证明破坏性从早期就存在——不是
   "训练后期被利用"，而是**从零 z 还没学会走路时，29 维残差头先学出了垃圾动作**，
   PPO 的 z 通道全程在与残差噪声搏斗。训练曲线上 vx 冲 0.8–1.1（超命令 0.8）、
   后期 rew 转负 kl 200+，正是"残差助跑+摔倒瞬间计速度"的奖励黑客迹象
   （eval 里摔倒样本 vx 高达 1.04–1.09 同证）。
2. **与 E44p 互证**：E44p（先冻结训好 z 再解冻解码器）证明"解冻承载通道"必须先有
   能用的基座；E48 证明反过来也成立——**基座（从零 z）没立住时，额外自由度通道
   （残差/解码器）只会被噪声梯度占据**。文献残差配方（RuN/ReSkill）默认基座是
   已工作的技能策略，我们把它误配到从零场景 → E48c（res_freeze 先立基座再放残差）
   是对症修正。
3. **"0.08 首个存活样本"未复现，降级为边缘事件**：E48-noaux（it_700）0.08 1/3
   （26.8m）是蒸馏路径首个 0.08 存活，但 E48c（it_800/it_1000）0.08 **0/6 + 0/6
   全倒**——两 run 合计 1/9，不构成"破 0.08"；最可能是一次幸运的初始条件/地形
   路径。**0.08 悬崖在蒸馏路径仍然成立**（含 E48/E48c 的地形训练 + 残差在场配方）。
   E48c 顺带确认：128 envs×2000 iters 的地形训练 z 基座本身是健康的（平地
   20–28m、只凸 0.06 3/3），与 E47 持平。
4. **残差通道结论收束（跨 3 配置稳健负结果）**：从零同开（E48）、先冻后放 +
   小尺度 + 强 L2（E48c）——**在我们 128-envs/3060 规模下，全关节残差通道不可用**：
   放开即被"冲刺+摔倒计速度"漏洞占据（训练 vx 超 1.0、rew 崩负），与 E44p
   （解冻解码器）同一模式。文献残差配方（RuN/ReSkill）的前提——可用基座 + 千级
   envs——后者在我们预算内不可得，**LITERATURE_SURVEY 解法 2 在当前规模关闭**；
   要么上 mjlab/更大 GPU（解法 1 路线），要么先解"坑"形状（G0 发现：有坑 ±0.06
   已全倒，比 0.08 更基础的边界）。
5. 产物：`isaac_e48_residual/`、`isaac_e48c_resfreeze/`（ckpt + train_log）、
   `eval_e48_{flat,r006_s0,r008_s0,paper006_s0,sym006_s0}.json`、
   `eval_e48b_it200_*.json`、`eval_e48c_{it800,it1000}_*.json`（服务器
   `apt_g1/outputs/`）。



## E49：去 VAE 直接 token RL 对照（owner 2026-09-05d 裁决 + 四轮评审定稿，协议 = `DS_OFFICIAL_DATA_PLAN.md` §3.2）

> 结构：策略随机初始化**直出 64d token**（无自训 VAE）→ 冻结 SONIC decoder → 29 dof 目标；
> `token = mean + α·std⊙a`（α=1，官方 g1-mode tokens 标定，无示范损失）；A 任务 = E45 同款单一前进配方。
> 对照臂 = E45–E47（带 VAE 全套，已有）；归因臂 E49-B（obs+[sinφ,cosφ]）预注册未点火。
> 冒烟全过（`e49_smoke.py`：obs 153/155、映射、反馈槽=原始 a、B 臂 φ 逐位、初始 a std=0.0182≈e⁻⁴）。
> **owner 快速迭代新标准（09-05）**：发现问题立刻停/改，算力优先于协议跑满——本节 s1/tanh 两行即中止打捞产物。

| Run ID | Purpose | Variant | Metric | Status | Result |
|--------|---------|---------|--------|--------|--------|
| E49-A-s0 | 无界直出 token（A 问主 run） | 128 envs × 2000 it，seed 0，`--token-mode` | A 60s ×3（best + final 双评） | DONE | **best it_50：3/3 存活 disp 29.0m vx 0.47 h_min 0.76（超 E45–E47 全线同口径）；final it_1999：0/3 disp 0 后退 shuffle h_min 0.23**。训练 rew 1.78@it50 → 0.62 单调崩，kl med 374 |
| E49-A-s1 | 复现检查（owner 中止@it1720 打捞） | 同上 seed 1 | it_50 打捞 eval | DONE（中止打捞） | **it_50：3/3 disp 15.3–16.1m vx 0.36**——早期步态 2/2 seed 复现；训练 rew 1.74→0.68、vx 冲 1.67 超速 + fall 出现，kl med 654 |
| E49-A-tanh-s0 | 受限范围消融（当即转修复臂） | `--token-bound tanh`（±α·std 硬界），owner 中止@it870 | 曲线诊断 | DONE（中止） | **恒不摔**（fall=0 全程）但 vx 衰向站立（0.36→0.26@440→0.09@770），rew 1.83→1.16；kl med 256（仍 4× latent）——界把崩坏变成安全退化，未保住步态 |

**E49-A 结论（A 问已答 + 新问题 = 训练崩坏，修复调研中）**：

1. **A 问（去掉额外 VAE、随机策略能否学会行走）= 能，且极快**：两 seed 早期快照
   （it_40–60，≈50 iters = 1 分钟级训练）均 3/3 真走路；s0 it_50 29.0m/0.47 超 E45
   （14m/0.27）与 E47（23.8m/0.42）同预算 eval 口径。**初始 token 邻域（官方 token
   均值附近）本身就近可步态区，搜索不是瓶颈。**
2. **新问题 = 继续训练必然崩坏（2/2 seed）**：PPO 每更新 kl med 374–654 = latent 臂
   （64）的 4–10×（动作维度 4× + 无 VAE 缓冲），rew 单调下降——**PPO 在裸 token
   空间不是在优化而是在随机游走**；终态两形态：无界→超速/摔（s1 vx 1.67）或后退
   shuffle（s0），tanh 界→安全站立化。**VAE 的隐藏第四角色 = KL 冲击缓冲器**
   （z 大位移被冻结 VAE 吸收为流形上近邻 token；对照 E45 kl med 64 且 rew 真在上升
   至 2.07）。【注：该对照日志来自 `isaac_e45_e39_from0.log`，与 tracker E45 行
   e27/vae.pt 的记载存在 VAE 版本出入，已挂 owner 待核】
3. **修复调研（owner 09-05「发现问题直接停，调研怎么修复」）候选**：①PPO 降温
   （lr 3e-4→1e-4 / epochs 5→2，直击 kl 过大病灶，首选最小实验）；②KL-to-init
   正则（token 空间锚定，软版 tanh）；③best-snapshot 操作规程（已证可行，配
   early-stop 省算力）；④长 episode（20s→60s，补「训练期看不见 60s 后摔」盲区）。
   判据预注册：修复 run 的 rew 应上升而非单调降、kl med 落回 ≤~150、final eval
   不逊 best eval。
4. 产物：`outputs/e49/`（stats npz + 全部 train/eval 日志与 JSON，服务器）、
   ckpt `GR00T-WholeBodyControl/outputs/isaac_e49a_s{0,1}/`、
   `isaac_e49a_tanh_s0/`；脚本 `e49_smoke.py`（SCRIPT_MAP 已登记）。
