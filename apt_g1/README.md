# apt_g1: APT-RL x SONIC x G1 flat-ground experiment

This package implements the flat-ground APT-RL experiment on the Unitree G1
using the GEAR-SONIC decoder as the pretrained low-level action decoder.

## Status

- [x] Project skeleton and config
- [x] Motion dataset interface
- [x] SONIC encoder/decoder wrapper (loads `UniversalTokenModule`)
- [x] SONIC manager-env adapter with APT auxiliary action
- [x] APT policy (SONIC token + auxiliary action + skill selection)
- [x] Minimal PPO training loop
- [x] MuJoCo G1 flat env (`g1_29dof_rev_1_0.xml`)
- [x] Real training smoke test on Linux (`policy_final.pt`)
- [ ] Long training / reward tuning
- [ ] Terrain-aware observation and reward
- [ ] Perception distillation

## Layout

```text
apt_g1/
  configs/flat_g1.yaml
  data/motion_dataset.py
  envs/g1_flat_env.py
  envs/mujoco_g1_flat_env.py
  policies/apt_policy.py
  sonic/sonic_wrapper.py
  sonic/apt_manager_env.py
  train.py
```

## Smoke test

```bash
python -m apt_g1.train --dummy --max-iters 5
```

## Current milestone

Step 2 (MuJoCo + SONIC training loop) is done. Step 3 (stable standing and slow
walking) is in progress.

Run a short real MuJoCo training on Linux:

```bash
MAX_ITERS=20 NUM_STEPS=64 bash apt_g1/run_mujoco_train.sh
```

Evaluate the saved policy:

```bash
cd ~/ros2_data
source GR00T-WholeBodyControl/.venv_sim/bin/activate
PYTHONPATH=~/ros2_data/apt_g1:~/ros2_data/GR00T-WholeBodyControl \
  python -m apt_g1.evaluate \
    --policy ~/ros2_data/apt_g1/outputs/flat_g1/policy_final.pt \
    --repo-root ~/ros2_data/GR00T-WholeBodyControl \
    --onnx-path ~/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
    --steps 500
```

Current result after 20 PPO iterations with 64 rollout steps:

```text
done at control step 25 (~0.50s)
total reward: 5.331
final root height: 0.780 m
```

Updated result after fixing the SONIC decoder interface:

```text
PPO (100 iters, token frozen, elastic band): survives 20.0 s (1000 control steps)
reference walking token sequence (elastic band): completes all 455 steps (~9.1 s)
```

The MuJoCo + SONIC + APT flat-ground loop is now running end to end.

## No-elastic-band result

The SONIC zero token is the idle/standing token. Training the APT policy from
that zero token with only the 12-d auxiliary action (token frozen) produces a
no-band flat-ground policy:

```text
checkpoint: outputs/flat_g1_noband/policy_50.pt
no elastic band: survives 1000 control steps (~20 s)
total reward: 2082.0
```

Render: `apt_g1/outputs/g1_noband.gif`

## No-band slow walking

Freezing the zero token and training only the auxiliary action also produces a
no-band slow walker:

```text
checkpoint: outputs/flat_g1_walk_noband/policy_150.pt
command 0.3 m/s: survives 850-1000 steps, avg vx 0.36-0.47 m/s
zero-token no-RL baseline: fixed ~0.20 m/s regardless of command
```

Render: `apt_g1/outputs/g1_walk_noband.gif`

Token unfreezing was tried from the no-band standing checkpoint but degraded
returns; the frozen zero-token + aux variant is the current best walker.

### Failed joint-latent variants

Direct joint learning of SONIC tokens is unstable with the current single-env
PPO. Parallel attempts and results are logged in
`refine-logs/EXPERIMENT_TRACKER.md`:

- reference token sequence + residual token: fails
- walking token init + full token: fails
- residual token around zero: survives but no speed
- 8-d / 16-d token VAE: fails
- 2-skill token library: always picks idle

Next candidate: temporal TVAE over 10-token windows, then RL on the low-dim
latent with gait phase observation.

## Correct reference-token export

The reference tokens used in earlier residual/VAE experiments were produced by
a custom encoder builder that did not match the official mode-0 observation
layout. Regenerate them with:

```bash
python -m apt_g1.export_reference_tokens \
  --motion-dir \
    $GR00T/gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428 \
  --encoder-onnx \
    $GR00T/gear_sonic_deploy/policy/release/model_encoder.onnx \
  --output outputs/flat_g1/reference_walk_tokens_official.npy
```

Then use `flat_g1_reference_aux.yaml`: it follows the official token sequence,
freezes the token residual, and trains only aux to stabilize the prior.

## Key SONIC decoder interface fixes

- The SONIC decoder expects joint history and outputs normalized actions in
  **IsaacLab joint order**, while MuJoCo uses its own joint order. The env now
  converts with the official `isaaclab_to_mujoco` / `mujoco_to_isaaclab`
  mappings from `policy_parameters.hpp`.
- Decoder output is a normalized action, so the joint target is
  `default_angle + action * action_scale`, not the raw decoder output.
- The released decoder expects **FSQ-quantized tokens**. `SonicOnnxDecoder`
  now quantizes the policy latent before decoding.
- History is filled from the actual reset state instead of zeros, matching the
  C++ state logger behavior.
- The official MuJoCo sim uses `scene_43dof.xml` plus an optional elastic band
  on the pelvis. Both are available in `MujocoG1FlatEnv`.
- PD torques and the elastic band are recomputed on every MuJoCo substep (200
  Hz), matching the official sim loop. Holding a 50 Hz command for four substeps
  made the high-gain controller unstable.

## Next integration steps

1. Download the SONIC training checkpoint and export its config:

   ```bash
   cd GR00T-WholeBodyControl
   python download_from_hf.py --training
   python gear_sonic/eval_agent_trl.py \
     +checkpoint=sonic_release/last.pt \
     +headless=True ++num_envs=1 +export_onnx_only=true
   ```

   This writes `model_config.yaml` next to the checkpoint.

2. Point `apt_g1/configs/flat_g1.yaml` at:

   ```yaml
   sonic:
     model_config_path: /path/to/model_config.yaml
     checkpoint_path: /path/to/last.pt
   ```

3. Use `SonicDecoder` to decode a pre-quantization latent:

   ```python
   from apt_g1.sonic.sonic_wrapper import SonicDecoder

   decoder = SonicDecoder(
       model_config_path="/path/to/model_config.yaml",
       checkpoint_path="/path/to/last.pt",
       device="cuda:0",
   )
   decoder.load()
   body_action = decoder.decode(token, proprioception)
   ```

4. Use `APTManagerEnvWrapper` to add the auxiliary action after the SONIC
   decoder:

   ```python
   from apt_g1.sonic.apt_manager_env import APTManagerEnvWrapper

   env = APTManagerEnvWrapper(base_env, config, aux_dim=12, aux_scale=0.2)
   obs = env.reset()
   next_obs, reward, done, info = env.step({
       "actions": torch.cat([token, hand_actions], dim=-1),
       "apt_aux": aux_action,
       "action_mode": "direct_latent",
   })
   ```

5. Run MuJoCo training on the Linux workstation:

   ```bash
   ssh cvgluser@10.16.52.225
   cd ~/ros2_data
   source GR00T-WholeBodyControl/.venv_sim/bin/activate
   PYTHONPATH=~/ros2_data/apt_g1:~/ros2_data/GR00T-WholeBodyControl \
     python -m apt_g1.train \
       --mujoco \
       --repo-root ~/ros2_data/GR00T-WholeBodyControl \
       --onnx-path ~/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
       --max-iters 5
   ```

The MuJoCo environment loads `g1_29dof_rev_1_0.xml`, builds the SONIC decoder
input from a 10-frame history, and applies the APT auxiliary action on the 12
lower-body joints. It does not require Isaac Lab or Isaac Sim.
