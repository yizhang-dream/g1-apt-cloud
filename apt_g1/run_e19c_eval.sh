#!/bin/bash
# E19c eval: phase-anchored + aux-reg on rough 0.06/0.08 and flat baseline.
set -e
CKPT=/home/cvgluser/ros2_data/apt_g1/outputs/isaac_e19c_s0.08/policy_it_1000.pt
for NOISE in 0.06 0.08; do
  OUT=/home/cvgluser/ros2_data/apt_g1/outputs/terr_e19c_n${NOISE}_s0.json
  echo "=== E19c noise=$NOISE seed0 ==="
  bash /tmp/run_apt_isaac.sh -u /home/cvgluser/ros2_data/apt_g1/isaac/eval_fast.py \
    --checkpoint "$CKPT" --tests A --keys aux,noaux --phase-mode --use-elevation 1 \
    --terrain rough --terrain-noise "$NOISE" --terrain-seed 0 --out "$OUT"
done
OUT=/home/cvgluser/ros2_data/apt_g1/outputs/terr_e19c_flat.json
echo "=== E19c flat baseline ==="
bash /tmp/run_apt_isaac.sh -u /home/cvgluser/ros2_data/apt_g1/isaac/eval_fast.py \
  --checkpoint "$CKPT" --tests A --keys aux,noaux --phase-mode --use-elevation 1 \
  --terrain plane --out "$OUT"
echo E19C_EVAL_DONE
