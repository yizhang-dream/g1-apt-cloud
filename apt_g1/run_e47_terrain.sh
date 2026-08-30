#!/bin/bash
# E47 terrain generalization: rough 0.06/0.08 x seed 0/1, A 60s straight walk.
set -e
OUT=/home/cvgluser/ros2_data/apt_g1/outputs
CKPT=$OUT/isaac_e47_heading/policy_it_500.pt
VAE=$OUT/token_vae_e39/vae.pt
for noise_seed in "0.06 0" "0.06 1" "0.08 0" "0.08 1"; do
  set -- $noise_seed
  noise=$1; seed=$2
  tag="eval_e47_terrain_$(echo $noise | tr -d '.')_s$seed"
  echo "=== START $tag (noise=$noise seed=$seed) ==="
  bash /tmp/run_apt_isaac.sh /home/cvgluser/ros2_data/apt_g1/isaac/eval_apt_isaac.py \
    --checkpoint $CKPT --tests A --latent-mode \
    --latent-vae-path $VAE --latent-speed-bins --latent-dir-bins --heading-scale 0.4 \
    --terrain rough --terrain-noise $noise --terrain-seed $seed \
    --out $OUT/$tag.json > $OUT/$tag.log 2>&1
  echo "=== DONE $tag ==="
done
echo "ALL_DONE"
