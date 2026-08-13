#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cvgluser/ros2_data/GR00T-WholeBodyControl}"
APT_ROOT="${APT_ROOT:-/home/cvgluser/ros2_data/apt_g1}"
ONNX="${ONNX:-$REPO_ROOT/gear_sonic_deploy/policy/release/model_decoder.onnx}"
MAX_ITERS="${MAX_ITERS:-2000}"
NUM_STEPS="${NUM_STEPS:-256}"

source "$REPO_ROOT/.venv_sim/bin/activate"

cd "$(dirname "$APT_ROOT")"
export PYTHONPATH="$APT_ROOT:$REPO_ROOT"
python -m apt_g1.train \
  --mujoco \
  --repo-root "$REPO_ROOT" \
  --onnx-path "$ONNX" \
  --max-iters "$MAX_ITERS" \
  --num-steps "$NUM_STEPS"
