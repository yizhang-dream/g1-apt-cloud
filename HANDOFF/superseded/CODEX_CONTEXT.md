> ⚠️ **已过时（2026-08-11 叙事）**：本文描述的是 MuJoCo 单进程 RL 受阻、
> Isaac 工作尚未开始的早期状态。现行真相以 [`../README.md`](../README.md)
> （2026-08-13 交接包）为准。保留于此仅供历史对照。

# Codex Context for apt_g1 / GR00T / G1

This file is the handoff context for a Codex session on the Windows machine.
The goal is to continue the APT-RL x SONIC x Unitree G1 flat-ground locomotion
experiment in MuJoCo.

## Goal

Follow the APT-RL paper method (`latent action + auxiliary action` joint
learning) to train a Unitree G1 policy that can stand and walk on flat ground
without the MuJoCo elastic band.

## Machines

- Linux training server:
  - `ssh cvgluser@10.16.52.225`
  - Repo: `/home/cvgluser/ros2_data/GR00T-WholeBodyControl`
  - Our code: `/home/cvgluser/ros2_data/apt_g1`
  - Venv: `source ~/ros2_data/GR00T-WholeBodyControl/.venv_sim/bin/activate`
- Windows backup/Codex machine:
  - `ssh zyz@10.16.177.186` or Tailscale `ssh zyz@100.100.101.43`
  - Project: `C:\Users\zyz\Documents\gr00t`

## Architecture

- APT policy outputs:
  - 64-d SONIC token (or low-dim latent when using a TVAE)
  - 12-d auxiliary action on legs
  - optional skill logits
- SONIC decoder ONNX maps `64-d token + 10-frame proprioception (994-d)` to
  `29-d joint targets`.
- MuJoCo env: `apt_g1/envs/mujoco_g1_flat_env.py`

## Key Facts Learned

1. SONIC decoder expects IsaacLab joint order and normalized actions:
   `joint_target = default_angle + action * action_scale`.
2. The released decoder expects FSQ-quantized tokens. `SonicOnnxDecoder`
   quantizes before decoding.
3. The SONIC zero token is the idle/standing token. Frozen zero token + trained
   aux is the only reliable no-band policy so far.
4. MuJoCo PD and the elastic band must be recomputed on every sim substep
   (200 Hz), not held for 4 substeps, or high-gain control becomes unstable.
5. Single-process MuJoCo PPO cannot learn stable no-band walking from SONIC
   tokens or joint-level TVAE latents. Many variants were tried and failed.

## Current Best Result

- Checkpoint: `outputs/flat_g1_walk_noband/policy_best_walk.pt`
- No elastic band, command 0.3 m/s
- Survives ~1000 control steps, displacement ~3.5 m
- Uses frozen zero token + 12-d aux

## Experiments Already Tried

1. Direct SONIC token RL: failed
2. Token residual around reference tokens: failed
3. Per-token VAE + aux: failed
4. Temporal TVAE over SONIC tokens: failed
5. Skill selection (idle/walk tokens): failed
6. Reference token + elastic-band annealing: failed after band removal
7. Joint-level TVAE over G1 joint trajectories: training improves, eval fails
8. 4-env parallel PPO: no improvement

## Status (2026-08-11 update)

**Distillation experiment ran and produced a working no-band controller.**

Summary:

- Collected 20,838 expert control steps from the official closed loop (no
  elastic band, 4 command modes) with per-step `(command, proprio, token)`.
- Plain BC regression (MLP/GRU/transformer/AR) fails in closed loop (falls in
  3-10s) despite 60-100% open-loop token accuracy -> compounding systematic
  error; state->token labels are multi-valued (planner phase not locked to
  body phase).
- kNN memory distillation works for all 4 gaits (600 steps) -> mapping is
  learnable in principle.
- Final solution: **phase-regression router** (per command-group PCA circular
  phase + small MLP regressing (sin,cos) + 40 phase-bin prototype tokens +
  EMA 0.3). Results (1000 steps = 20s, 3 seeds with jitter):
  - IDLE 3/3 (vx 0.003, h_min 0.76)
  - SLOW_WALK 3/3 (vx 0.13-0.56)
  - WALK 3/3 (vx 0.81-0.83, 16.2-16.6 m)
  - FORWARD_JUMP 1/3 (marginal, oracle itself marginal at 500+ steps)
  - 40s command-switch episode passes (h_min 0.74)
- Verdict: distillation is feasible and sufficient for stand/slow-walk/walk
  without Isaac Lab RL, but requires the phase-router structure, not token
  regression. Limits: jump marginal, backward held out, >20s and hardware
  untested.

Next steps: more data (jump/backward/speed variety) + retrain phase router;
DAgger rounds with the phase router as expert; Isaac Lab RL only if targeting
terrain/higher speeds.

## Useful Files

- Configs: `apt_g1/configs/`
  - `flat_g1_walk_noband.yaml` (best frozen-zero walk)
  - `flat_g1_jointvae_multi.yaml` (latest joint TVAE attempt)
  - `flat_g1_ref_band_anneal.yaml` (reference + band annealing)
- Environment: `apt_g1/envs/mujoco_g1_flat_env.py`
- SONIC wrapper: `apt_g1/sonic/sonic_wrapper.py`
- Temporal VAE: `apt_g1/sonic/token_seq_vae.py`
- Training: `apt_g1/train.py`
- Eval: `apt_g1/evaluate.py`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Review log: `review-stage/AUTO_REVIEW.md`
- Context summary: `CONTEXT_SUMMARY.md`

## Run Commands (Linux)

Train frozen-zero walk:

```bash
source ~/ros2_data/GR00T-WholeBodyControl/.venv_sim/bin/activate
cd ~/ros2_data
PYTHONPATH=~/ros2_data/apt_g1:~/ros2_data/GR00T-WholeBodyControl \
  python -m apt_g1.train \
    --config apt_g1/configs/flat_g1_walk_noband.yaml \
    --mujoco \
    --repo-root ~/ros2_data/GR00T-WholeBodyControl \
    --onnx-path ~/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx
```

Evaluate without band:

```bash
python -m apt_g1.evaluate \
  --config apt_g1/configs/flat_g1_walk_noband.yaml \
  --policy outputs/flat_g1_walk_noband/policy_best_walk.pt \
  --repo-root ~/ros2_data/GR00T-WholeBodyControl \
  --onnx-path ~/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
  --steps 1000 --no-band
```

## Note for Windows

This Windows copy is for Codex continuation and code inspection. The training
environment (MuJoCo, SONIC ONNX, venv) is on the Linux server, not on Windows.
