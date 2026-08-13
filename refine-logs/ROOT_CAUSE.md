# Root Cause: Why Token/VAE/Skill RL Variants Were Unstable

## Evidence

1. `reference_walk_tokens.npy` was generated with a custom encoder observation
   builder, not with the official mode-0 layout.
2. The saved tokens exactly match that custom builder, but differ from a
   corrected mode-0 export by a mean absolute token difference of ~0.47 and a
   max difference of ~1.56.
3. Official mode 0 only fills four encoder observations:

   - `encoder_mode_4` (4d)
   - `motion_joint_positions_10frame_step5` (290d)
   - `motion_joint_velocities_10frame_step5` (290d)
   - `motion_anchor_orientation_10frame_step5` (60d)

   All other 1118 encoder dimensions are explicitly zeroed. The custom builder
   filled root height, lower-body joints, wrists, and other unused inputs.

4. The custom builder also flattened the 6D anchor orientation with
   `R[:, :2].T.flatten()`; the official deployment uses `R[:, :2].flatten()`
   (row-wise) and applies a heading-compensation quaternion.

## Why Per-Frame VAE and Skill Tokens Did Not Work

- The VAE in `apt_g1/sonic/token_vae.py` compresses one 64-d token at a time.
  A single latent therefore represents one instantaneous token, not a gait.
- `walking_latent_mean16.npy` is the mean of the encoded 455-frame sequence.
  Decoding that mean produces an average static token, not a walking trajectory.
- `skill_tokens.npy` contains only zero + one walking token. A fixed token has
  no phase information, so it cannot reproduce a 455-frame walking cycle.
- The server-side `TokenSeqVAE` decodes a window of 10 tokens, but the
  environment only reads `decoded[0, 0]` at every control step. That ignores
  the remaining 9 generated tokens and does not advance through the decoded
  sequence, so the policy still receives a single static token per step.

## What To Do Next

1. Regenerate reference tokens with `export_reference_tokens.py`, which follows
   official mode-0 encoder layout.
2. Use those tokens as a dynamic reference sequence in the environment, with
   `reference_token_sequence` and the phase observation.
3. Train `train_token: false` first: freeze the token residual at zero and let
   aux learn stabilization around the corrected reference prior.
4. Only after that baseline walks, enable a small token residual.
5. If a low-dimensional latent is still desired, implement a temporal VAE over
   token windows (T, 64), not a per-token VAE, and condition it on phase.

## Remaining Unresolved Issue

A 300-iteration `flat_g1_reference_aux_run` with corrected tokens and frozen
token residual did not converge (falls at ~36 control steps). The corrected
reference tokens survive all 455 steps with the elastic band, but not without
it. This confirms a second issue: the SONIC reference prior is not a
self-stabilizing no-band controller by itself. The aux policy needs a better
warm start, a stronger aux scale, aligned initial state, or a different
training formulation before this route can be called solved.
