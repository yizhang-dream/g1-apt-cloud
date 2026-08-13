#!/bin/bash
# E15 eval: aux (trained through 0.08 curriculum) vs noaux on rough terrain.
# Same fixed terrain seeds as the baseline sweep; tests A (60s walk).
set -e
CKPT=/home/cvgluser/ros2_data/apt_g1/outputs/isaac_e15_s0.08/policy_it_900.pt
# seed 0 for all noises (same terrain as the baseline curve); seeds 1-2 only
# at the cliff noises (0.08/0.10) to test terrain-instance generalization.
for NOISE in 0.04 0.06 0.08 0.10; do
  OUT=/home/cvgluser/ros2_data/apt_g1/outputs/terr_e15_n${NOISE}_s0.json
  echo "=== noise=$NOISE terrain_seed=0 ==="
  bash /tmp/run_apt_isaac.sh -u /home/cvgluser/ros2_data/apt_g1/isaac/eval_fast.py \
    --checkpoint "$CKPT" --tests A --keys aux,noaux \
    --terrain rough --terrain-noise "$NOISE" --terrain-seed 0 --out "$OUT"
done
for NOISE in 0.08 0.10; do
  for SEED in 1 2; do
    OUT=/home/cvgluser/ros2_data/apt_g1/outputs/terr_e15_n${NOISE}_s${SEED}.json
    echo "=== noise=$NOISE terrain_seed=$SEED ==="
    bash /tmp/run_apt_isaac.sh -u /home/cvgluser/ros2_data/apt_g1/isaac/eval_fast.py \
      --checkpoint "$CKPT" --tests A --keys aux,noaux \
      --terrain rough --terrain-noise "$NOISE" --terrain-seed "$SEED" --out "$OUT"
  done
done
echo E15_EVAL_DONE
