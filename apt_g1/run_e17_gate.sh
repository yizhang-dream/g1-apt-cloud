#!/bin/bash
# E17: policy-learned gait/group selection (paper gait-logit analog) + aux +
# privileged elevation map, rough-terrain curriculum.
set -e
PY=/home/cvgluser/ros2_data/apt_g1/isaac/train_apt_isaac.py
ARGS="--num-envs 64 --rollout 24 --vx-max 0.8 --use-2hz-gate 1 --entropy 0.01 --terrain rough --terrain-seed 0 --use-elevation 1 --gate-sel 1"

RESUME=""
for PAIR in 0.04:300 0.06:600 0.08:1000; do
  STAGE=${PAIR%:*}
  ITERS=${PAIR#*:}
  OUT=/home/cvgluser/ros2_data/apt_g1/outputs/isaac_e17_s${STAGE}
  mkdir -p "$OUT"
  echo "=== E17 stage noise=$STAGE iters=$ITERS -> $OUT ==="
  if [ -n "$RESUME" ]; then
    bash /tmp/run_apt_isaac.sh -u "$PY" $ARGS --terrain-noise "$STAGE" --iters "$ITERS" --resume "$RESUME" --out "$OUT"
  else
    bash /tmp/run_apt_isaac.sh -u "$PY" $ARGS --terrain-noise "$STAGE" --iters "$ITERS" --out "$OUT"
  fi
  RESUME="$OUT/policy_it_${ITERS}.pt"
done
echo E17_DONE
