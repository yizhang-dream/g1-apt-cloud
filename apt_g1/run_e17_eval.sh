#!/bin/bash
# E17 eval: gate+aux (elevation) vs gate+noaux on rough terrain, fixed seed 0.
set -e
CKPT=/home/cvgluser/ros2_data/apt_g1/outputs/isaac_e17_s0.08/policy_it_1000.pt
for NOISE in 0.06 0.08; do
  OUT=/home/cvgluser/ros2_data/apt_g1/outputs/terr_e17_n${NOISE}_s0.json
  echo "=== E17 gate+aux vs gate+noaux noise=$NOISE seed0 ==="
  bash /tmp/run_apt_isaac.sh -u /home/cvgluser/ros2_data/apt_g1/isaac/eval_fast.py \
    --checkpoint "$CKPT" --tests A --keys aux,noaux --use-elevation 1 --gate-sel 1 \
    --terrain rough --terrain-noise "$NOISE" --terrain-seed 0 --out "$OUT"
done
echo E17_EVAL_DONE
