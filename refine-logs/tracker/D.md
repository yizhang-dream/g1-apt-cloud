# Tracker 系列：D（D 系列（蒸馏 Distillation 线 + Stress Test；另有 D021–D028 四行混排在 E.md 的『地形/数据泛化/感知』节））

> 【层位 L3｜Run 台账·系列文件（数据唯一事实源）】↑ `refine-logs/EXPERIMENT_TRACKER.md`（总索引）与 `HANDOFF/02_EXPERIMENT_HISTORY.md`（L2 阶段史）｜↓ `HANDOFF/03_OUTPUTS_INDEX.md` → 服务器 `outputs/`（L4）｜≈ `apt_g1/SCRIPT_MAP.md`（代码轴）。
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

## Stress Test (2026-08-12, encoder consolidation)

| Run ID | Purpose | Variant | Horizon | Metric | Status | Result |
|--------|---------|---------|---------|--------|--------|--------|
| D021 | Single encoder module | PhaseRouterEncoder (group select + EMA + Command.from_vxvy) | - | API | DONE | unified encode() validated end-to-end; matches inline eval (vx 0.83) |
| D022 | 60s straight walk (fwd/back) | long single-command runs | 60s x3 | survival, disp | DONE | walk fwd 3/3 (50.9-52.0m), walk back 3/3 (48.7m) |
| D023 | Disturbance grid | 200/500N impulses x 4 dirs x 3 seeds during walk | 45s x24 | survival, recovery | DONE | 21/24 complete; recovery 0.02-2.6s; 3 seed-dependent late falls |
| D024 | Command-switch marathon | 68s mixed schedule x 3 seeds | 68s x3 | survival | DONE | 0/3; falls at jump (2 seeds) / walk_back (1 seed); earlier 58s pass was a 20s episode-length artifact |
| D025 | Isolation | walk_back 60s; walk40->idle->jump | - | survival | DONE | walk_back 3/3 standalone; jump-after-40s 2/3 (h_min~0.21) -> jump under prolonged running is the residual fragility |

