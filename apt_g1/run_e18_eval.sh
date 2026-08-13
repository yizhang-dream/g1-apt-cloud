#!/bin/bash
# E18 eval: phase-policy (with aux) vs phase-only (aux=0) on rough terrain.
set -e
CKPT=/home/cvgluser/ros2_data/apt_g1/outputs/isaac_e18_s0.08/policy_it_1000.pt
for NOISE in 0.06 0.08; do
  OUT=/home/cvgluser/ros2_data/apt_g1/outputs/terr_e18_n${NOISE}_s0.json
  echo "=== E18 phase+aux vs phase-only noise=$NOISE seed0 ==="
  bash /tmp/run_apt_isaac.sh -u /home/cvgluser/ros2_data/apt_g1/isaac/eval_fast.py \
    --checkpoint "$CKPT" --tests A --keys aux,noaux --phase-mode --use-elevation 1 \
    --terrain rough --terrain-noise "$NOISE" --terrain-seed 0 --out "$OUT"
done
echo E18_EVAL_DONE
