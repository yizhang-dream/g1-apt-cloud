# Tracker 系列：R（R001–R020（MuJoCo RL 线））

> 【层位 L3｜Run 台账·系列文件（数据唯一事实源）】↑ `refine-logs/EXPERIMENT_TRACKER.md`（总索引）与 `HANDOFF/02_EXPERIMENT_HISTORY.md`（L2 阶段史）｜↓ `HANDOFF/03_OUTPUTS_INDEX.md` → 服务器 `outputs/`（L4）｜≈ `apt_g1/SCRIPT_MAP.md`（代码轴）。
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

