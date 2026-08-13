> ⚠️ **已过时（2026-08-11 叙事）**：本文停在"MuJoCo 受阻、需 Isaac Lab"
> 的状态。该项目后续已在 Isaac Lab 完成 E1–E27 并得出阶段性结论，现行真相以
> [`../README.md`](../README.md)（2026-08-13 交接包）为准。保留于此仅供历史对照。

# Context Summary

## Project

`apt_g1`: APT-RL x SONIC x Unitree G1 flat-ground locomotion experiment in
MuJoCo. The goal is a no-elastic-band policy that can stand and walk on flat
ground using a frozen SONIC decoder plus a low-dimensional learned action.

## Current Working Result

`outputs/flat_g1_walk_noband/policy_best_walk.pt` is the only reliable no-band
walker so far. It uses the frozen SONIC zero token plus a 12-d auxiliary action.
At a 0.3 m/s command it survives about 850-1000 control steps.

## Root Causes Found

1. The earlier `reference_walk_tokens.npy` was generated with a custom encoder
   observation builder, not the official mode-0 layout. Saved tokens exactly
   match that custom builder and differ from a corrected export by a large
   margin.
2. Per-token VAE and static skill tokens cannot represent the temporal phase of
   a walking sequence.
3. The server-side `TokenSeqVAE` decodes a 10-token window but the environment
   only reads the first token at every step.
4. Even with corrected tokens, the SONIC reference prior is not self-stabilizing
   without the elastic band. Direct joint replay also fails without the band.

## Changes Added

- `apt_g1/export_reference_tokens.py`: official encoder mode-0 token export.
- `apt_g1/configs/flat_g1_reference_aux.yaml`
- `apt_g1/configs/flat_g1_reference_aux_anneal.yaml`
- `apt_g1/configs/flat_g1_reference_aux_post.yaml`
- `apt_g1/configs/flat_g1_reference_aux_nophase_warm.yaml`
- `MujocoG1FlatEnv` and training/evaluation support `add_phase_obs`.
- `refine-logs/ROOT_CAUSE.md`: detailed evidence and remaining issues.

## Remaining Problem

Reference-token + aux variants did not converge after 300 iterations in
multiple formulations. The elastic-band anneal also collapsed once the band was
removed. The next step is not more VAE/skill variants; it is to validate tokens
against the official C++ deployment and then train aux/residual with vectorized
environments, better warm starts, and a band/WBC curriculum.

## Latest Attempts (Paper Method)

Direct SONIC token RL, token residual, per-token VAE, temporal TVAE, skill
selection, joint-level TVAE, elastic-band annealing, and 4-env parallel PPO
were all tried. None produced stable no-band walking.

The closest result is the joint-level TVAE over G1 joint trajectories:
training reward improves steadily, but no-band evaluation still collapses in
~60-110 steps. The bottleneck is the single-process MuJoCo RL stack, not the
action representation.

## Status

Blocked. Meaningful progress requires one of:

1. GPU-vectorized RL simulator (e.g., Isaac Lab)
2. Official GR00T / `UNITREE_G1_SONIC` workflow
3. Multi-process/distributed vectorized MuJoCo

## Machines

- Lab Linux server: `ssh cvgluser@10.16.52.225`
- Windows Codex workspace: `C:\Users\zyz\Documents\gr00t`
