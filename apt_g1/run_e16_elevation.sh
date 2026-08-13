#!/bin/bash
# E16: gate-fixed aux WITH privileged elevation map, trained on rough terrain
# via a noise curriculum (paper teacher-style: terrain map goes to the policy
# before perception distillation).
set -e
PY=/home/cvgluser/ros2_data/apt_g1/isaac/train_apt_isaac.py
ARGS="--num-envs 64 --rollout 24 --vx-max 0.8 --use-2hz-gate 1 --latent-kl 2.5e-6 --latent-expl 0.01 --entropy 0.001 --terrain rough --terrain-seed 0 --use-elevation 1"

RESUME=""
for PAIR in 0.04:300 0.06:600 0.08:1000; do
  STAGE=${PAIR%:*}
  ITERS=${PAIR#*:}
  OUT=/home/cvgluser/ros2_data/apt_g1/outputs/isaac_e16_s${STAGE}
  mkdir -p "$OUT"
  echo "=== E16 stage noise=$STAGE iters=$ITERS -> $OUT ==="
  if [ -n "$RESUME" ]; then
    bash /tmp/run_apt_isaac.sh -u "$PY" $ARGS --terrain-noise "$STAGE" --iters "$ITERS" --resume "$RESUME" --out "$OUT"
  else
    bash /tmp/run_apt_isaac.sh -u "$PY" $ARGS --terrain-noise "$STAGE" --iters "$ITERS" --out "$OUT"
  fi
  RESUME="$OUT/policy_it_${ITERS}.pt"
done
echo E16_DONE
