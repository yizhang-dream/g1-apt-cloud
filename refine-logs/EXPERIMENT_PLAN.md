# Experiment Plan

**Problem**: The G1 can now stand on flat ground without the elastic band by
RL over a frozen SONIC zero token plus a 12-d auxiliary action. The next
question is whether the same policy can execute command-conditioned slow walking
on flat ground without the band.

**Method Thesis**: A frozen SONIC decoder provides a standing/idle motion prior
(zero token), and a compact RL auxiliary action is sufficient to learn
flat-ground balance and slow walking in MuJoCo.

**Date**: 2026-08-11

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1: RL aux over frozen SONIC zero token achieves no-band flat-ground locomotion | Proves the APT-style latent+aux architecture works on G1 without training the decoder | No-band policy tracks 0.0-0.5 m/s commands for at least 1000 control steps without falling | B1, B2 |
| C2: Auxiliary action, not the band or the SONIC default pose, is what stabilizes | Rules out the "just uses default PD / elastic band" anti-claim | Ablation: aux=0 fails; band=0 + trained aux succeeds | B3 |
| Anti-claim to rule out | "The gains come only from the elastic band or from SONIC's default controller" | Same checkpoints evaluated with band disabled; band never used in final walk config | B1, B3 |

## Paper Storyline

- Main paper must prove: frozen SONIC latent prior + small aux RL policy is a
  sufficient flat-ground locomotion shell for G1.
- Appendix can support: zero-token idle behavior, command-tracking reward curves,
  and ablation of token freezing.
- Experiments intentionally cut: terrain, perception distillation, and GR00T
  integration until flat-ground walking is stable.

## Experiment Blocks

### Block 1: No-band slow walking (0.0-0.5 m/s)

- Claim tested: C1
- Why this block exists: it is the first command-conditioned locomotion result
  that does not rely on the elastic band.
- Dataset / split / task: MuJoCo flat ground; commands sampled uniformly in
  [0.0, 0.5] m/s, y=0, yaw=0.
- Compared systems: (a) frozen zero token + trained aux (ours), (b) same
  architecture with token unfrozen, (c) no-training aux=0 baseline.
- Metrics: survival steps, velocity tracking error, mean total reward, fall rate
  over 5 deterministic rollouts.
- Setup details: `flat_g1_walk_noband.yaml`, PPO 1 env, 256 rollout steps,
  500 iterations, no band.
- Success criterion: mean survival >= 1000 steps and tracking error < 0.15 m/s
  for 0.3 m/s command.
- Failure interpretation: zero token is not a walking prior; unfreeze token or
  switch to a walking reference token.
- Table / figure target: main result table + reward curve.
- Priority: MUST-RUN

### Block 2: Token training after aux stabilization

- Claim tested: C1 (stronger variant)
- Why this block exists: aux may not produce enough forward velocity if the
  latent prior is idle-only.
- Dataset / split / task: same flat-ground command sampling.
- Compared systems: frozen zero token vs fine-tuned token.
- Metrics: max command speed, survival, tracking error.
- Setup details: start from best frozen-token checkpoint, enable token
  gradients with small std, train 300-500 more iterations.
- Success criterion: survives 1000 steps at 0.5 m/s.
- Failure interpretation: SONIC FSQ token space is too coarse for direct RL;
  consider a residual token policy or motion-library warm start.
- Table / figure target: ablation table.
- Priority: MUST-RUN if B1 fails speed tracking.

### Block 3: Auxiliary action ablation

- Claim tested: C2
- Why this block exists: separates RL contribution from default PD/zero-token
  behavior.
- Compared systems: aux trained vs aux forced to zero.
- Metrics: survival, height variance, fall time.
- Setup details: same config, deterministic evaluation.
- Success criterion: aux=0 fails before 200 steps while trained aux survives
  1000 steps.
- Failure interpretation: if aux=0 also survives, the contribution is only the
  SONIC decoder, not RL.
- Table / figure target: ablation table.
- Priority: MUST-RUN

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | No-band standing sanity | zero-token aux training (done) | 1000 steps | low | none |
| M1 | Slow walking, frozen token | B1 | survives 1000 steps at 0.3 m/s | ~10 min GPU | low |
| M2 | Faster/speed tracking | B2 if needed | survives at 0.5 m/s | ~10 min GPU | medium |
| M3 | Ablations and metrics | B3, multi-seed | tables complete | ~10 min GPU | low |
| M4 | Render + qualitative video | best policy | video shows stable walk | low | low |

## Compute and Data Budget

- Total estimated GPU-hours: 1-2 hours for M1-M4 on one RTX 3060.
- Data preparation needs: none; reference tokens already extracted.
- Human evaluation needs: visual check of rendered walk.
- Biggest bottleneck: single-env PPO sample efficiency; consider vectorized
  MuJoCo envs if convergence is too slow.

## Risks and Mitigations

- Risk: zero token is an idle prior, not a walking prior.
  Mitigation: unfreeze token or warm-start from walking reference token.
- Risk: aux action alone is too weak to generate forward velocity.
  Mitigation: raise aux_scale, add velocity-tracking reward weight, then train token.
- Risk: PPO with one env is slow/noisy.
  Mitigation: increase rollout steps, longer training, or vectorized envs.

## Final Checklist

- [x] Main paper tables are covered (C1/C2 blocks defined)
- [x] Novelty is isolated (B3 aux ablation)
- [x] Simplicity is defended (frozen token vs trained token)
- [x] Frontier contribution is justified or explicitly not claimed (SONIC is reused, not trained)
- [x] Nice-to-have runs are separated from must-run runs
