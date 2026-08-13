#!/bin/bash
# E16 eval: aux (elevation-aware) vs noaux on rough terrain, fixed seed 0.
set -e
CKPT=/home/cvgluser/ros2_data/apt_g1/outputs/isaac_e16_s0.08/policy_it_1000.pt
for NOISE in 0.06 0.08; do
  OUT=/home/cvgluser/ros2_data/apt_g1/outputs/terr_e16_n${NOISE}_s0.json
  echo "=== E16 aux+noaux noise=$NOISE seed0 (elevation) ==="
  bash /tmp/run_apt_isaac.sh -u /home/cvgluser/ros2_data/apt_g1/isaac/eval_fast.py \
    --checkpoint "$CKPT" --tests A --keys aux,noaux --use-elevation 1 \
    --terrain rough --terrain-noise "$NOISE" --terrain-seed 0 --out "$OUT"
done
echo E16_EVAL_DONE
