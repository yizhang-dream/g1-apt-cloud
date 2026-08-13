# Auto Review Log

**Run**: flat-ground no-band RL milestone
**Date**: 2026-08-11
**Reviewer backend**: unavailable in this desktop session (no MCP reviewer tools); self-review used as interim.

## Round 1

### What works

- MuJoCo G1 env now recomputes PD and elastic-band force every sim substep.
- SONIC decoder interface matches official policy input logs.
- Zero token is confirmed as the SONIC idle/standing token.
- Frozen zero token + 12-d aux RL stands without the band for 1000 steps (~20 s).

### Weaknesses

- No command-conditioned walking yet; all successful runs are `stand_only=True`.
- Single-env PPO is slow and noisy; best checkpoint was at iteration 50, not the
  final one.
- No systematic ablation showing aux is necessary.
- No vectorized env, so scaling to terrain curriculum is blocked.

### Next actions

1. Run no-band slow walking with 0.0-0.5 m/s commands.
2. If frozen zero token cannot walk, unfreeze token with warm start.
3. Run aux=0 ablation and render video.

## Round 2

### New evidence

- Frozen zero token + aux produces a no-band walker (`policy_150`).
- At 0.3 m/s command it survives 850-1000 steps with avg vx 0.36-0.47 m/s.
- Zero-token no-RL baseline walks at a fixed ~0.20 m/s regardless of command;
  the RL policy adds command conditioning.
- Unfreezing the token from the standing checkpoint degraded returns, so the
  frozen-token variant remains the best walker.
- Aux ablation is ambiguous: aux=0 also survives, so the claim "aux is required
  for stability" is not supported; the supported claim is command conditioning.

### Remaining weaknesses

- Command tracking is loose, especially at low commands (0.1 m/s -> ~0.37 m/s).
- Single-env PPO still limits sample efficiency.
- No multi-seed formal table yet.

## Round 3

### Multi-scheme parallel results

| Scheme | Result |
|--------|--------|
| Frozen zero token + aux | Works: no-band movement, displacement ~3.6 m in 500 steps |
| Reference token sequence + residual token | Failed; best survival ~124 steps |
| Walking token init + full token joint | Failed; negative returns |
| Residual token around zero after aux warm start | Survives but no forward speed |
| 8-d VAE over tokens + aux | Failed; best survival ~146 steps |
| 16-d VAE + walking latent warm start | Failed; negative returns |
| 2-skill token library (idle/walk) + aux | Always chooses idle; no walk |

### Conclusion from failures

Direct RL over 64-d FSQ SONIC tokens is not sample-efficient enough with a
single MuJoCo env. A per-token VAE is also not a motion decoder: it reconstructs
static tokens but not temporal gait structure.

### Recommended next approach

1. Train a **temporal TVAE over 10-token windows** (window -> latent -> future
   token sequence), matching APT-RL's temporal representation learning stage.
2. Warm-start RL from the latent mean of walking windows, not from zero or a
   single static token.
3. Add a **motion phase / gait frequency** observation so the latent can
   condition on gait phase.
4. If temporal VAE still fails, switch to the official GR00T
   `UNITREE_G1_SONIC` VLA action interface instead of hand-training SONIC tokens.

## Round 4

### Paper-method attempts

- Temporal TVAE over 10-token windows (16-d latent, warm-started from walking
  window mean): RL training did not converge; best no-band survival ~77 steps.
- Reference walking token sequence + aux with elastic-band annealing from 1.0
  to 0: while the band is present rewards stay high, but as soon as band_scale
  reaches zero the policy collapses (best survival ~93 steps at band 0).

### Conclusion

The current single-env MuJoCo RL stack cannot turn SONIC motion tokens into a
stable no-band walking controller. The bottleneck is no longer the action
representation alone; it is also the training infrastructure (single env,
no physics-parallel RL, no motion phase reset) and the SONIC decoder being a
motion-tracking prior rather than a balance controller.

### Next real options

1. Use official `UNITREE_G1_SONIC` VLA/GR00T workflow, which already runs
   SONIC with a real control stack in MuJoCo.
2. Vectorize MuJoCo with at least 32-128 parallel envs and retry latent+aux.
3. Train a dedicated balance/velocity policy directly on 29-d joint targets
   without SONIC tokens, then use SONIC only as a motion style prior.

## Round 5

### Latest attempts

- Multi-env PPO (2 envs) with temporal TVAE: stopped early after 50 iterations,
  no improvement (best survival ~121 steps, no forward displacement).
- Official reference tokens + aux warm start without phase observation
  (parallel run found on the server): all checkpoints fail around 40 steps
  with backward velocity and no forward displacement.

### Current status

The paper-style latent+aux training has now been tried with:

- direct SONIC tokens
- token residual
- per-token VAE
- temporal TVAE
- skill selection
- reference-token playback with band annealing
- multi-env PPO

None produces a stable no-band walker. The most reliable no-band behavior is
still the frozen zero-token + aux policy, which moves but does not track
commands well.

### Recommendation

Stop trying to make single-MuJoCo PPO learn SONIC tokens. Move to the official
`UNITREE_G1_SONIC` VLA/GR00T stack or train a direct joint-target velocity
policy and treat SONIC as a style prior.

## Round 6

### Joint-level TVAE (closest to paper)

Implemented a 16-d temporal TVAE directly over G1 joint trajectories
(no SONIC tokens). RL training curves improve steadily, but no-band evaluation
still fails:

- joint TVAE + aux: ~100 steps
- continued training: no improvement
- motion start-pose warm start: ~60 steps
- elastic-band annealing to 0: fails after band removal (~60 steps)

### Conclusion

Even with the correct paper-style representation (TVAE over motion trajectories
plus auxiliary action), the bottleneck is the single-env MuJoCo RL stack.
Training returns improve, but the policy never learns the balance required to
keep the humanoid upright without the band. A real GPU-vectorized RL simulator
is required for the next step.

## Round 7

### Multi-env joint TVAE

Ran the joint-level TVAE + aux with 4 parallel MuJoCo envs. Training completed
400 iterations, but every checkpoint survives only ~65 no-band steps. More
parallel envs did not fix the core issue; the motion-prior decoder drives the
robot forward too aggressively without the band, and RL cannot learn enough
balance with the current stack.

### Blocked

This blocker has now repeated across direct tokens, VAE, temporal TVAE, skill
selection, band annealing, and multi-env PPO: stable no-band walking cannot be
learned in the current single-process MuJoCo RL setup. Meaningful progress
requires one of:

1. a GPU-vectorized RL simulator (e.g., Isaac Lab), or
2. the official GR00T / `UNITREE_G1_SONIC` stack.
