# APT-RL x GR00T x Unitree G1 Integration Roadmap

## Goal

Adapt the four-stage APT-RL pipeline from
[`scirobotics.adz7397.pdf`](../tmp/pdfs/scirobotics.adz7397.pdf) to the Unitree G1
humanoid, and connect it to NVIDIA Isaac GR00T so that a high-level VLA model can
drive a low-level terrain-aware locomotion/skill policy.

## What the three pieces are

| Piece | What it is | Where it fits |
|---|---|---|
| APT-RL | Four-stage pipeline: 2D trajectory optimization (TO) data, TVAE latent skills, PPO over latent + auxiliary actions, perceptual distillation | Low-level skill prior and terrain-aware locomotion policy |
| Isaac GR00T N1.7 | Vision-language-action (VLA) model, 3B params, diffusion action head | High-level task reasoning, language/image conditioning, action chunking |
| GR00T-WholeBodyControl / GEAR-SONIC | Humanoid whole-body controller on Unitree G1; 64-d latent motion token decoded to full-body joints at 50 Hz | Decoder shell and real-robot deployment stack |
| Unitree G1 | 29-DOF humanoid | Physical and simulated target robot |

The official GR00T workflow already uses `UNITREE_G1_SONIC` as an embodiment tag.
Its VLA action space is 78-d:

- 64-d SONIC motion token
- 7-d left hand joints
- 7-d right hand joints

This is structurally the same idea as APT-RL: a high-level policy emits a compact
latent action, and a lower-level controller decodes it into joint commands.

## Recommended architecture

Keep GR00T as the high-level planner and make APT-RL the terrain-aware low-level
locomotion/skill layer.

```
Language + Camera
        |
        v
GR00T N1.7 (high-level VLA)
  - task understanding
  - arm/hand targets
  - navigation / skill intent
        |
        v
APT-RL locomotion policy (G1, 50-100 Hz)
  - proprioception
  - depth + LiDAR (distilled)
  - latent action + auxiliary action + skill selection
        |
        v
G1 joint commands / SONIC decoder
        |
        v
Unitree G1 (sim: MuJoCo, real: Thor + Unitree SDK)
```

## Mapping APT-RL stages to G1

### Stage 1: Motion dataset

APT-RL generates 180k 2D trot/bound trajectories with 2D SRBD trajectory
optimization. A 2D sagittal-plane prior is much weaker for a humanoid because
balance, upper-body dynamics, and lateral motion matter far more.

Recommended data sources for G1:

- `bones-studio/seed` (BONES-SEED): 142k+ human motions, ~288 hours, already
  retargeted to G1 in MuJoCo format.
- Official NVIDIA G1 demonstration datasets from the GR00T workflow.
- VR teleoperation data collected with GR00T-WholeBodyControl.
- Custom 3D trajectory optimization if a specific parkour skill is needed.

The dataset should contain state-action pairs, where the action is either joint
targets/velocities or joint torques. Torque-labeled data is preferable for an
APT-style decoder.

### Stage 2: Representation learning

Train a TVAE-style encoder and skill decoder on the G1 motion dataset:

- Encoder input: short proprioceptive history.
- Latent action: one compact vector per motion skill, e.g. 16-d for walking,
  running, step-up, step-down, jump, crouch.
- Decoder: latent action -> G1 leg/waist joint commands or torques.

Two implementation choices:

1. Train a new APT decoder on G1 data.
2. Fine-tune or reuse the SONIC encoder/decoder as the motion token decoder and
   attach an APT-style selection policy on top.

Option 2 is the fastest baseline because SONIC already provides a 64-d latent
motion token interface and a deployment stack.

### Stage 3: Reinforcement learning

Train the low-level locomotion policy in MuJoCo (not Isaac Lab) on the G1 model:

- Observations: proprioception + privileged height map (teacher).
- Actions: latent skill action + auxiliary action + skill selection logit.
- Skills: walk, run, step over, step up/down, jump, crouch, recover.
- Terrains: stairs, high steps, gaps, stepping stones, rough terrain, discrete
  obstacles, slopes.
- Algorithm: PPO with velocity tracking, balance, and style rewards.
- Curriculum: increase terrain difficulty over training.

This is the stage where the paper's hybrid torque formula maps to G1:

```text
tau_input = tau_decoder + kp * (q_default - q + aux_scale * a_aux) - kd * q_dot
```

For G1, joint-space PD gains and foot placement matter more than for the
quadruped, so reward shaping and balance regularization are critical.

### Stage 4: Perception distillation

Distill the privileged height-map teacher into a student that uses real onboard
sensors:

- Dense depth camera for short range.
- 2D LiDAR for long range (0.6-5 m).
- CNN + GRU student encoder, trained with DAgger and truncated BPTT.
- Deployment is zero-shot after distillation.

This mirrors the paper and matches the G1's available sensing setup.

## Integration options

### Option A: APT-RL commands SONIC (recommended first step)

Keep SONIC as the whole-body decoder. Train APT-RL as a locomotion/skill
selector that outputs navigation commands or skill tokens. SONIC converts them
into stable whole-body motion. GR00T provides task intent and arm/hand targets.

Risk: SONIC may not be optimized for extreme terrain or high-speed parkour.

### Option B: Replace the SONIC decoder with an APT decoder

Train an APT-RL decoder on G1 data and use GR00T's 64-d latent action slot as the
APT latent action. This is the closest analog to the paper: high-level model
emits latent actions, APT decoder produces joint commands.

Risk: hardest path; requires high-quality torque-labeled G1 data and a robust
whole-body balance layer.

### Option C: Hybrid controller router

Use SONIC for everyday locomotion and switch to APT-RL only when terrain
perception detects hard obstacles. A high-level router decides which low-level
controller is active.

Risk: two controllers must be carefully blended to avoid instability.

## Suggested milestones

1. Run the official GR00T + G1 workflow in the MuJoCo sim environment.
2. Run the GEAR-SONIC VLA workflow (`UNITREE_G1_SONIC`) in MuJoCo.
3. Download BONES-SEED G1 data and inspect state/action format.
4. Train a TVAE encoder + decoder on a small G1 motion library.
5. Train a flat-ground PPO locomotion policy using latent actions in MuJoCo.
6. Add terrain curriculum and skill selection.
7. Add depth + LiDAR distillation.
8. Wrap APT-RL behind the GR00T PolicyServer API.
9. Evaluate in MuJoCo, then deploy to real G1 with Thor.

## Key risks and decisions

- GR00T is a VLA task/manipulation model, not a torque controller. It should not
  output joint torques directly.
- Humanoid balance cannot be handled by the paper's 2D SRBD prior alone. Use
  BONES-SEED or 3D motion data.
- The project needs an explicit definition of "skills" and their transitions on
  G1, e.g. walk <-> run <-> step-up <-> jump.
- The exact interface between GR00T, APT-RL, and SONIC must be decided before
  implementation. Option A is the lowest-risk starting point.

## References

- APT-RL paper: <https://doi.org/10.1126/scirobotics.adz7397>
- Project page: <https://skillquadsr.github.io/>
- Isaac GR00T: <https://github.com/NVIDIA/Isaac-GR00T>
- GR00T-WholeBodyControl: <https://github.com/NVlabs/GR00T-WholeBodyControl>
- GEAR-SONIC project: <https://nvlabs.github.io/GEAR-SONIC/>
- BONES-SEED: <https://huggingface.co/datasets/bones-studio/seed>
