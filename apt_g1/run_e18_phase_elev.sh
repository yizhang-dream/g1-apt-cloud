#!/bin/bash
# E18: policy directly controls the gait phase (router warm-start) + aux +
# privileged elevation map + forward-progress reward, rough-terrain curriculum.
# Closest available analog of the paper's "policy modulates latent with map".
set -e
PY=/home/cvgluser/ros2_data/apt_g1/isaac/train_apt_isaac.py
ARGS="--num-envs 64 --rollout 24 --vx-max 0.8 --use-2hz-gate 1 --entropy 0.001 --latent-kl 2.5e-6 --latent-expl 0.01 --phase-mode --phase-warmstart-iters 150 --phase-warmstart-coef 10.0 --terrain rough --terrain-seed 0 --use-elevation 1 --progress-scale 0.3"

RESUME=""
for PAIR in 0.04:300 0.06:600 0.08:1000; do
  STAGE=${PAIR%:*}
  ITERS=${PAIR#*:}
  OUT=/home/cvgluser/ros2_data/apt_g1/outputs/isaac_e18_s${STAGE}
  mkdir -p "$OUT"
  echo "=== E18 stage noise=$STAGE iters=$ITERS -> $OUT ==="
  if [ -n "$RESUME" ]; then
    bash /tmp/run_apt_isaac.sh -u "$PY" $ARGS --terrain-noise "$STAGE" --iters "$ITERS" --resume "$RESUME" --out "$OUT"
  else
    bash /tmp/run_apt_isaac.sh -u "$PY" $ARGS --terrain-noise "$STAGE" --iters "$ITERS" --out "$OUT"
  fi
  RESUME="$OUT/policy_it_${ITERS}.pt"
done
echo E18_DONE
