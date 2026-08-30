"""Data-side probe: is the FAST speed-bin gait direction-biased in the data?

E43 showed that cleaning the fast-region z-leak does NOT fix the speed-induced
drift (E40 straightness 0.89); the culprit must be on the decoder/data side.
This probe checks whether the training data's fast-bin rows are direction-
imbalanced (turn-heavy), which would bake a turning bias into the decoder's
fast speed embedding.

Metrics per speed bin (walk rows only, mode==2):
  - direction-bin distribution (8 bins), fraction in forward bin 4
  - entropy vs uniform (log 8)
  - speed.npy command histogram for context
"""
import os
import numpy as np

data_dir = "data/exp_all3"
tok = np.load(os.path.join(data_dir, "token.npy")).astype(np.float32)
mode = np.load(os.path.join(data_dir, "mode.npy"))
angle_bin = np.load(os.path.join(data_dir, "angle_bin.npy")).astype(np.int64)
speed = np.load(os.path.join(data_dir, "speed.npy"))

import sys
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
from train_token_vae_e39 import walk_phase_rate

pmean, V2, rate, phi = walk_phase_rate(tok, mode)
rw = rate[mode == 2]
edges = np.quantile(rw, [1.0 / 3.0, 2.0 / 3.0])
vb = np.clip(np.digitize(rate, edges), 0, 2).astype(np.int64)

walk = mode == 2
print("walk rows:", int(walk.sum()), "of", len(mode))
print("vb counts (walk):", np.bincount(vb[walk], minlength=3))
print("speed.npy values (walk):", dict(zip(*np.unique(speed[walk], return_counts=True))))
print("angle_bin overall (walk):", np.bincount(angle_bin[walk], minlength=8))

for b in range(3):
    m = walk & (vb == b)
    cnt = np.bincount(angle_bin[m], minlength=8)
    frac4 = cnt[4] / cnt.sum()
    p = cnt / cnt.sum()
    ent = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    print(f"\nvb={b} n={int(m.sum())}")
    print("  angle_bin counts:", cnt.tolist())
    print(f"  forward bin4 fraction: {frac4:.3f}")
    print(f"  entropy: {ent:.3f} (uniform = {np.log2(8):.3f}, all-fwd = 0)")
    print("  speed.cmd dist:", dict(zip(*np.unique(speed[m], return_counts=True))))

# direct comparison: fast vs slow forward fraction
for b in (0, 2):
    m = walk & (vb == b)
    cnt = np.bincount(angle_bin[m], minlength=8)
    print(f"vb={b}: fwd/bin4 = {cnt[4]}/{cnt.sum()} ({cnt[4] / cnt.sum():.3f}), "
          f"turn bins (0-3,5-7) = {cnt.sum() - cnt[4]}/{cnt.sum()}")
