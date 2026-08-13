#!/bin/bash
# E15: gate-fixed aux trained on rough terrain via a noise curriculum.
# Stages: rough 0.04 -> 0.06 -> 0.08, all with fixed terrain seed 0, resuming
# the policy between stages. Mirrors the paper's terrain-difficulty curriculum
# (proprio-only; no elevation map yet).
set -e
PY=/home/cvgluser/ros2_data/apt_g1/isaac/train_apt_isaac.py
ARGS="--num-envs 64 --rollout 24 --vx-max 0.8 --use-2hz-gate 1 --latent-kl 2.5e-6 --latent-expl 0.01 --entropy 0.001 --terrain rough --terrain-seed 0"

RESUME=""
STAGE_ITERS="0.04:300 0.06:600 0.08:900"
for PAIR in $STAGE_ITERS; do
  STAGE=${PAIR%:*}
  ITERS=${PAIR#*:}
  OUT=/home/cvgluser/ros2_data/apt_g1/outputs/isaac_e15_s${STAGE}
  mkdir -p "$OUT"
  echo "=== E15 stage noise=$STAGE iters=$ITERS -> $OUT ==="
  if [ -n "$RESUME" ]; then
    bash /tmp/run_apt_isaac.sh -u "$PY" $ARGS --terrain-noise "$STAGE" --iters "$ITERS" --resume "$RESUME" --out "$OUT"
  else
    bash /tmp/run_apt_isaac.sh -u "$PY" $ARGS --terrain-noise "$STAGE" --iters "$ITERS" --out "$OUT"
  fi
  RESUME="$OUT/policy_it_${ITERS}.pt"
done
echo E15_DONE
