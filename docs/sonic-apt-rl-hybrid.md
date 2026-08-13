# SONIC + APT-RL Hybrid for Perceptive Humanoid Locomotion

## Idea

Replace APT-RL's representation learning stage (TVAE + skill decoders) with the
pretrained GEAR-SONIC encoder-decoder stack, then train a perceptive policy on
top of the SONIC latent action space.

This is feasible because SONIC already provides what APT-RL tries to learn:

- A compact latent motion token (64-d FSQ-quantized).
- A single decoder that maps the token to G1 whole-body joint targets at 50 Hz.
- A pretrained whole-body motion prior from BONES-SEED (142k+ motions).

## What SONIC is and is not

SONIC architecture:

```text
[G1 joints / SMPL / teleop / SOMA]
        |
        v
[embodiment-specific encoder]
        |
        v
[FSQ discrete token, 64-d]
        |
        v
[single whole-body decoder]
        |
        v
[G1 joint targets]
```

Important differences from APT-RL's TVAE:

- APT-RL latent is a continuous Gaussian vector learned by VAE.
- SONIC latent is a discrete FSQ token learned by motion-tracking PPO.
- APT-RL encoder encodes proprioceptive state history.
- SONIC encoder encodes motion references, not terrain-aware state.
- APT-RL decoder outputs torques.
- SONIC decoder outputs joint targets; torques are produced by a PD layer.

Therefore, in the hybrid design:

- Use SONIC **decoder** as the APT-RL action decoder.
- Do not rely on the SONIC **encoder** for terrain awareness.
- Use the SONIC encoder only to tokenize reference motions when building skill
  libraries or collecting training data.

## Hybrid architecture

```text
Proprioception + depth + LiDAR
        |
        v
Perceptive RL policy (50-100 Hz)
  - SONIC token action (64-d)
  - auxiliary action (joint-space correction)
  - skill selection logit
        |
        v
SONIC decoder (frozen or fine-tuned)
        |
        v
PD controller -> G1 joint targets
```

This maps to APT-RL as:

| APT-RL stage | Hybrid replacement |
|---|---|
| 2D TO motion dataset | BONES-SEED / G1 motion library |
| TVAE latent + decoders | SONIC encoder-decoder |
| RL over latent actions | RL over SONIC 64-d token + auxiliary actions |
| Perceptual distillation | Still needed: teacher height map -> depth + LiDAR student |

## Two training routes

### Route 1: RL in SONIC latent space (closest to APT-RL)

Train a custom PPO policy in Isaac Lab:

- Observation: proprioception + privileged terrain height map.
- Action: SONIC token + auxiliary action + skill selection.
- Decoder frozen: SONIC decoder converts token to joint targets.
- Auxiliary action: PD-space correction for terrain not covered by SONIC.
- Reward: velocity tracking, upright torso, body height, foot clearance,
  energy, action rate.
- Curriculum: flat ground -> slopes -> stairs -> gaps -> high steps.

After the teacher policy works, distill it to depth + LiDAR.

Strengths:

- No reward-free imitation data collection required for the RL route.
- Can optimize directly for perceptive terrain traversal.
- Matches the paper's PPO + latent + auxiliary formulation.

Risks:

- FSQ token space is discrete/quantized; RL must explore a 64-d discrete-ish
  action space.
- Frozen SONIC decoder may not produce extreme terrain motions (jumps, high
  steps, fast bounding-like transitions).
- SONIC's whole-body behavior may fight the terrain task's leg-specific needs.

### Route 2: Fine-tune GR00T N1.7 on SONIC tokens (official VLA route)

Collect teleoperation demonstrations on the target terrain, convert them to
LeRobot format with `UNITREE_G1_SONIC`, then fine-tune GR00T N1.7:

- GR00T predicts 64-d SONIC tokens from ego camera + language + state.
- SONIC decoder executes the tokens.
- No explicit RL reward design.

Strengths:

- Official workflow already exists end-to-end.
- Fastest path to a first perceptive demonstration.

Weaknesses:

- Performance is bounded by demonstration coverage.
- Hard to achieve high-speed or agile behavior beyond teleop data.
- Less similar to APT-RL's RL + auxiliary action approach.

## Recommended hybrid plan

Use Route 1 for the APT-RL experiment, with Route 2 as a baseline:

1. Run SONIC VLA baseline on a simple terrain task.
2. Confirm SONIC decoder interface in Isaac Lab.
3. Build a G1 terrain environment with privileged height map.
4. Train a policy that outputs SONIC tokens + auxiliary actions with PPO.
5. Compare frozen vs. fine-tuned SONIC decoder.
6. Add skill selection: walk / run / step-up / step-down / jump.
7. Distill to depth + LiDAR.

## Practical interface questions to resolve first

- Does the SONIC decoder accept raw 64-d continuous values, or must the policy
  output pre-quantized FSQ indices?
- Can the SONIC decoder be called at 100 Hz inside the RL environment, or is it
  limited to 50 Hz?
- Does the released checkpoint support fine-tuning in Isaac Lab for a terrain
  task, or only motion tracking?
- Which joints should the auxiliary action control: legs only, or legs + waist?
- Should arms be held fixed during flat-ground locomotion experiments?

## Verdict

Yes. Use SONIC as the pretrained low-level decoder and train a perceptive RL
policy in the SONIC latent action space. This keeps the paper's core idea:

- a compact latent action,
- a reusable pretrained decoder,
- an auxiliary action for terrain adaptation,
- teacher-student perception distillation.

The main work is no longer representation learning; it moves to:

- exploration in the SONIC token space,
- auxiliary-action design,
- terrain-aware RL,
- sensor distillation.
