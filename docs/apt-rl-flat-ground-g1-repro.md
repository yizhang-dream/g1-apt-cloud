# Flat-Ground APT-RL Reproduction on Unitree G1

## Scope

Reproduce the core APT-RL pipeline on a Unitree G1 humanoid in simulation, on
flat terrain first:

1. Build a flat-ground locomotion baseline in simulation.
2. Build a G1 motion dataset with state-action pairs.
3. Train a TVAE representation and gait/skill-specific decoders.
4. Train a PPO policy that emits latent actions + auxiliary actions.
5. Evaluate against a vanilla PPO policy and fixed-gait policies.

GR00T integration and perceptual distillation are intentionally out of scope
for this first flat-ground milestone.

## Recommended simulation baseline

Use MuJoCo, not Isaac Lab. The GR00T-WholeBodyControl repo already ships a
working MuJoCo G1 setup:

- MJCF model: `gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml`
- WBC config: `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`
- Python env: `mujoco` 3.10 inside `.venv_sim`
- SONIC decoder: `gear_sonic_deploy/policy/release/model_decoder.onnx`

The `apt_g1.envs.mujoco_g1_flat_env.MujocoG1FlatEnv` in this repo is a direct
MuJoCo training environment that does not require Isaac Lab.

The simulation environment must expose:

```text
reset() -> observation
step(action) -> observation, reward, terminated, truncated
```

and support either joint-target PD control or direct joint torque control.

## Data representation

Define one state-action sample at every control timestep:

```text
state s_t:
  base linear velocity          (3)
  base angular velocity         (3)
  projected gravity             (3)
  joint positions               (n)
  joint velocities              (n)
  foot heights / contacts       (4 or more)
  command velocity / yaw rate   (3)
  history as needed             (k x n)

action a_t:
  joint targets or joint torques
```

For the first milestone, restrict the action to the lower body:

- 2 hips, 2 knees, 2 ankles per leg
- optionally waist and torso stabilization

The full G1 has 29 DOF. The low-level APT decoder should only control the joints
that participate in locomotion; arms can be fixed or controlled by a separate
policy later.

## Pipeline

### Step 0: Flat-ground PPO baseline

Train a vanilla PPO policy on `Isaac-Velocity-Flat-G1-v1` before touching APT-RL.
This validates:

- simulator stability
- reward function
- joint action limits
- command sampling
- telemetry and checkpointing

This policy is also a source of demonstration data for Step 1.

### Step 1: Motion dataset

APT-RL uses trajectory-optimized state-action data. For a humanoid on flat
ground, start with an existing motion library instead of 2D trajectory
optimization:

1. Use G1-retargeted motion data from BONES-SEED (`bones-studio/seed`).
2. Replay motions in simulation under a PD controller.
3. Record the resulting joint torques.
4. Store short trajectory segments as state-action pairs.

If torque labels are not available, recover them by replaying the motion in the
simulator and logging commanded torques.

For flat ground, create at least two skill datasets:

- walk: 0.0 to 1.5 m/s
- run: 1.5 to 3.0 m/s

Optionally add:

- step up / step down
- turning in place

The data is saved as trajectory episodes:

```text
episode:
  states:    (T, state_dim)
  actions:   (T, action_dim)
  gait_id:   int
  command:   (T, 3)
```

### Step 2: TVAE representation learning

Train a state encoder and skill decoders, following the paper:

- Encoder input: 3 consecutive state frames.
- Latent vector: 16-32 dims (start with 32 for a humanoid).
- Encoder output: Gaussian mean and log-variance.
- Reconstruction loss: MSE over the 3 reconstructed state frames.
- KL loss: KL between posterior and standard Gaussian, weight 0.1.
- Decoder: latent vector -> joint torque or joint target vector.

Train one decoder per skill:

- `Decoder_walk`
- `Decoder_run`

Architecture starting point:

```text
Transformer:
  layers: 2
  attention heads: 8
  hidden size: 64-256
  activation: GELU
  batch size: 500
  learning rate: 0.001
  epochs: 500
```

Freeze the encoder before training decoders.

### Step 3: RL with latent + auxiliary actions

Modify the G1 velocity environment so the policy outputs:

```text
action = [
  latent action z,       dim 16-32
  auxiliary action a_aux, dim n_lower_body_joints
  gait selection logit,  dim 1
]
```

The environment computes the final control input:

```text
tau_dec = Decoder_gait(z)

joint_target = q_default - q + aux_scale * a_aux
tau_input = tau_dec + kp * joint_target - kd * q_dot
```

Start with:

- `kp = 80`
- `kd = 2`
- `aux_scale = 0.2`

These values may need tuning for G1.

Training setup:

- Algorithm: PPO
- Command: linear x velocity, linear y velocity, yaw rate
- Flat terrain command ranges:
  - x: -1.0 to 3.0 m/s
  - y: -0.5 to 0.5 m/s
  - yaw: -0.5 to 0.5 rad/s
- Reward:
  - velocity tracking
  - torso upright
  - body height near target
  - action rate penalty
  - foot clearance / contact style
  - energy / torque penalty
- Gait selection update: 2 Hz
- Latent and auxiliary update: 50-100 Hz

Use a small KL regularization on the latent action, similar to the paper
(coefficient around 1e-5 to 1e-6).

### Step 4: Evaluation

Compare:

1. APT-RL with latent + auxiliary actions
2. APT-RL without auxiliary action
3. APT-RL with fixed walk decoder
4. APT-RL with fixed run decoder
5. Vanilla PPO without any motion prior

Metrics:

- velocity tracking reward
- success rate per command bin
- energy / cost of transport
- sample efficiency (reward vs. training samples)
- gait transition success rate

## What to add after flat ground works

- terrain curriculum: slopes, stairs, gaps, stepping stones, high steps
- skill selection: walk / run / step up / step down / jump
- teacher-student perception distillation: privileged height map -> depth + LiDAR
- GR00T integration through the `UNITREE_G1_SONIC` action interface

## Interface contract for a custom simulator

If the simulation environment is not the bundled MuJoCo env, provide:

```text
get_observation()          -> proprioceptive state, optional height map
apply_action(action)       -> decodes APT latent + auxiliary and applies to robot
get_reward()               -> scalar reward
check_termination()        -> fall / timeout / success
reset_episode()            -> reset robot and terrain
```

The APT-RL module should only depend on this interface, so it can be ported
between MuJoCo and other simulators.

## Immediate next step

Use the MuJoCo environment already implemented in `apt_g1`. The next step is:

1. Run the MuJoCo smoke test on the Linux workstation.
2. Collect walk/run motion episodes or reference tokens.
3. Train or reuse the SONIC decoder.
4. Train the APT policy in MuJoCo with PPO.
5. Compare with a fixed-token / vanilla baseline.
