# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | sanity | zero token + trained aux | stand | survival 1000 steps | MUST | DONE | 20 s no-band, reward 2082 |
| R002 | M1 | slow walk | frozen zero token + aux | 0.0-0.5 m/s | survival, tracking | MUST | DONE | policy_150 survives 850-1000 steps at 0.3 m/s |
| R003 | M2 | speed | unfrozen token warm start | 0.0-0.5 m/s | survival, tracking | MUST | DONE | worse than frozen token; keep frozen |
| R004 | M3 | ablation | aux=0 vs trained aux | walk 0.3 m/s | survival | MUST | DONE | both survive; aux adds speed/command conditioning |
| R005 | M4 | qualitative | best walk policy | 0.3 m/s | video | NICE | DONE | rendered g1_walk_noband.gif |
| R006 | M5 | residual latent+aux | reference token seq + residual token | 0.0-0.5 m/s | survival, tracking | MUST | DONE | failed; best survival ~124 |
| R007 | M5 | direct latent+aux | walking token init + full token | 0.0-0.5 m/s | survival, tracking | MUST | DONE | failed; negative returns |
| R008 | M5 | residual-zero latent+aux | aux-stabilized warm start + residual token | 0.0-0.5 m/s | survival, tracking | MUST | DONE | survives but no forward speed |
| R009 | M6 | VAE latent+aux | 8-d VAE over SONIC tokens + aux | 0.0-0.5 m/s | survival, tracking | MUST | DONE | failed; best survival ~146 |
| R010 | M6 | VAE16 latent+aux | 16-d VAE + walking latent warm start | 0.0-0.5 m/s | survival, tracking | MUST | DONE | failed; negative returns |
| R011 | M7 | skill latent+aux | 2-skill token library + aux | 0.0-0.5 m/s | survival, tracking | MUST | DONE | always chooses idle skill; no walk |
| R012 | M8 | seq TVAE latent+aux | 16-d temporal VAE over 10-token windows | 0.0-0.5 m/s | survival, tracking | MUST | DONE | failed; best survival ~77 |
| R013 | M9 | reference+band anneal | ref token seq + aux, band -> 0 | 0.0-0.5 m/s | survival, tracking | MUST | DONE | failed after band removal |
| R014 | M10 | seq TVAE + 2 envs | 16-d temporal VAE, 2 MuJoCo envs | 0.0-0.5 m/s | survival, tracking | MUST | DONE | stopped early; no improvement |
| R015 | M10 | reference aux warm (user) | official ref tokens + aux warm start | 0.0-0.5 m/s | survival, tracking | MUST | DONE | failed; ~40 steps, no forward |
| R016 | M11 | joint TVAE + aux | 16-d TVAE over G1 joint trajectories | 0.0-0.5 m/s | survival, tracking | MUST | DONE | training improves, eval ~100 steps |
| R017 | M11 | joint TVAE + aux cont | continue joint TVAE checkpoint | 0.0-0.5 m/s | survival, tracking | MUST | DONE | no improvement |
| R018 | M11 | joint TVAE + reset warm | joint TVAE + motion start pose | 0.0-0.5 m/s | survival, tracking | MUST | DONE | fails ~60 steps |
| R019 | M11 | joint TVAE + band anneal | joint TVAE + elastic band -> 0 | 0.0-0.5 m/s | survival, tracking | MUST | DONE | fails after band removal |
| R020 | M12 | joint TVAE + 4 envs | 16-d joint TVAE, 4 parallel envs | 0.0-0.5 m/s | survival, tracking | MUST | DONE | all checkpoints ~65 steps, no improvement |
| R013 | M9 | corrected reference + frozen token + aux | official mode-0 tokens + reference sequence | 0.0-0.5 m/s | survival, tracking | MUST | DONE | 300 iters, not stable; falls ~36 steps; see ROOT_CAUSE.md |

## Distillation Experiment (2026-08-11, see DISTILL_EXPERIMENT.md)

| Run ID | Purpose | Variant | Data | Metric | Status | Result |
|--------|---------|---------|------|--------|--------|--------|
| D001 | Expert data collection | official closed loop, no band, 4 modes | 20,838 ctrl steps (50Hz) | - | DONE | tokens == token_state.csv (max diff 0.0); encoder output on k/16 lattice |
| D002 | Harness validation | oracle token replay in apt_g1 env | same | survival 400-600 steps | DONE | idle/slow/walk/jump stable -> harness OK (walk vx~0.8) |
| D003 | BC regression | MLP / GRU / deep / transformer / AR | 20,838 | val per-dim 60-72% | DONE | closed loop all fall (3-10s); compounding: closed-loop token MSE 20-30x open-loop |
| D004 | AR-delta (teacher forcing) | prev token + delta regression | 20,838 | val per-dim ~100% | DONE | exposure bias; closed loop falls faster (~1-2s) |
| D005 | Random-error tolerance | oracle tokens + k dims +/-1 level | - | survival | DONE | up to 8/64 dims no effect -> failure is systematic, not decoder sensitivity |
| D006 | kNN memory distillation | nearest official state -> token | train rows only | survival 600 steps | DONE | idle/slow/walk/jump all stable -> state->token learnable in principle |
| D007 | Phase classification router | fixed-period bins + classifier | 20,838 | phase acc | DONE | idle/slow OK; walk/jump fail (1Hz replan breaks fixed period) |
| D008 | Phase regression router | PCA circular phase + MLP(sin,cos) + 40 prototypes + EMA0.3 | 20,838 | - | DONE | idle 3/3, slow 3/3, walk 3/3 (vx 0.81-0.83, 16.2-16.6m/20s), jump 1/3; 40s switch episode passed |
| D009 | Command-switch episode | idle->walk->idle->slow->jump->idle | - | survival 40s | DONE | complete, h_min 0.74 |

## Distillation Phase 2 (2026-08-11, exp2 + routers v2-v5)

| Run ID | Purpose | Variant | Data | Metric | Status | Result |
|--------|---------|---------|------|--------|--------|--------|
| D010 | exp2 data collection | backward/turns/strafes/more jump/stealth, no band | 32,675 steps, 0 falls | - | DONE | merged exp_all = 53,513 steps, 5 modes |
| D011 | router v2 (merged, angle bins) | (mode,speed,8-dir-bin) groups | 53,513 | survival 20s x3 | DONE | walk_fwd 3/3, walk_back 3/3, idle regressed, turns stand/stumble |
| D012 | router v2.1 (density filter + 2D metrics) | NN-distance outlier filter; fdir=mdir fix | 53,513 | survival 20s x3 | DONE | idle fixed 3/3; walk 3/3 both dirs; slow standing (vx~0.01) |
| D013 | router v3 (density filter per group) | same | 53,513 | survival | DONE | slow prototypes still standing (vx 0.009) -> data heterogeneity issue |
| D014 | router v4 (slow=exp1 only) | slow group restricted to exp1 | 53,513 | survival | DONE | slow vx 0.07-0.16, 2-3/3 |
| D015 | router v5 (slow=exp1 phase1 only) | slow group = first exp1 slow phase | 53,513 | survival | DONE | slow 2/3 (vx 0.07-0.25); walk fwd/back 3/3 (16-17m); switch 58s passes; jump 1/3; turns/strafes/stealth at or near oracle ceiling |
| D016 | Oracle ceiling check | official turn/strafe token replay | - | survival | DONE | turn bins 1/6/2 fall ~200 steps -> distilled cannot beat teacher; env caps curved-motion ceiling |

## Distillation Phase 3 (2026-08-11, proto tuning + DAgger)

| Run ID | Purpose | Variant | Data | Metric | Status | Result |
|--------|---------|---------|------|--------|--------|--------|
| D017 | Prototype tuning sweep | mean/median/nearest x B=40/64 for marginal groups | 53,513 | survival 20s x3 | DONE | jump median/B40 -> 3/3; turn_right mean/B64 -> 3/3; turn_left/strafe_left nearest/B40 -> 3/3 with real motion |
| D018 | Final router (v6) | best per-group config | 53,513 | survival 20s x3 | DONE | idle 3/3, slow 2/3, walk fwd/back 3/3 (0.83/-0.78 m/s), jump 3/3, turns 3/3, strafes 3/3, stealth 0/3 (=oracle), 58s switch passes |
| D019 | DAgger-lite for slow | student states + kNN phase relabel, retrain slow net | 2,543 new samples | survival | DONE | regressed to standing (vx 0.01); not adopted; weak-rhythm gaits need cleaner phase labels or RL |
| D020 | Stealth oracle check | official stealth token replay | - | survival | DONE | oracle falls at step 361 -> stealth 0/3 is teacher-bound, not a distillation gap |

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

## Stress Test (2026-08-12, encoder consolidation)

| Run ID | Purpose | Variant | Horizon | Metric | Status | Result |
|--------|---------|---------|---------|--------|--------|--------|
| D021 | Single encoder module | PhaseRouterEncoder (group select + EMA + Command.from_vxvy) | - | API | DONE | unified encode() validated end-to-end; matches inline eval (vx 0.83) |
| D022 | 60s straight walk (fwd/back) | long single-command runs | 60s x3 | survival, disp | DONE | walk fwd 3/3 (50.9-52.0m), walk back 3/3 (48.7m) |
| D023 | Disturbance grid | 200/500N impulses x 4 dirs x 3 seeds during walk | 45s x24 | survival, recovery | DONE | 21/24 complete; recovery 0.02-2.6s; 3 seed-dependent late falls |
| D024 | Command-switch marathon | 68s mixed schedule x 3 seeds | 68s x3 | survival | DONE | 0/3; falls at jump (2 seeds) / walk_back (1 seed); earlier 58s pass was a 20s episode-length artifact |
| D025 | Isolation | walk_back 60s; walk40->idle->jump | - | survival | DONE | walk_back 3/3 standalone; jump-after-40s 2/3 (h_min~0.21) -> jump under prolonged running is the residual fragility |

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
| E16 | aux + 特权 elevation map | 0.04→0.06→0.08 课程 | rough，9×9@0.15m 局部地形 | 训练中 | 论文教师式路径 | IN PROGRESS |

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
| E26-T | E26 地形扩展 | rough 0.06/0.08 × seed 0/1 | rough | 0.06：3/3（22.7–42.5m）；0.08：偏移 0/6、纯时钟 2/6 | 0.08 是共同能力边界；偏移地形无增益 | DONE |
| E27 | latent→VAE→SONIC（无行为先验） | 相位条件化 VAE + z_walk warmstart | flat | A 3/3 19.1m（vx 0.32）；B 12/12、C 3/3、D 3/3 | 用户指定缺失实验：token 流形本身是关键先验；E9/E11 直出关节 0/3 | DONE |
| M-FROM0 | mjlab 官方从零训练 | Unitree-G1-Flat 1024 envs × 5000 it | flat | 已跑 ~900 iters，跌倒率 0.08–0.17，已暂停留 checkpoint | 蒸馏是否必要的对照基线（可 --agent.resume 恢复） | PAUSED |

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
