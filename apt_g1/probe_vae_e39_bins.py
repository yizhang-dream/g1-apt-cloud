"""Per-speed-bin direction-leakage probe for the E39 dual-disentangled VAE.

Question: E40 (higher incentive) reached vx 0.456 but straightness dropped to
0.89 — WHERE does the residual drift come from?

- If z leaks direction ONLY in the fast speed-bin region, the E37/E39 fresh
  probe (averaged over all data) hid a fast-region residual entanglement.
- If z is direction-clean in EVERY bin, the drift must live in the decoder's
  speed-embedding / training data (fast tokens collected while turning), so
  z-level disentanglement has reached its limit for this line.

Method: encode every token-window to mu, group by speed bin (vb), train a
FRESH linear classifier z -> 8 direction bins per group, report val acc vs
the group's majority baseline.
"""
import os
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
from train_token_vae_e39 import DirSpeedPhaseTokenVAE, build_windows, walk_phase_rate

torch.manual_seed(0)
np.random.seed(0)
data_dir = "data/exp_all3"
out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/token_vae_e39"

tok = np.load(os.path.join(data_dir, "token.npy")).astype(np.float32)
mode = np.load(os.path.join(data_dir, "mode.npy"))
angle_bin = np.load(os.path.join(data_dir, "angle_bin.npy")).astype(np.int64)

pmean, V2, rate, phi = walk_phase_rate(tok, mode)
rw = rate[mode == 2]
edges = np.quantile(rw, [1.0 / 3.0, 2.0 / 3.0])
vb = np.clip(np.digitize(rate, edges), 0, 2).astype(np.int64)

x = build_windows(tok, 10)
model = DirSpeedPhaseTokenVAE().cuda()
model.load_state_dict(torch.load(os.path.join(out_dir, "vae.pt"), map_location="cuda:0"))
model.eval()
with torch.no_grad():
    mu = model.encode(torch.from_numpy(x).cuda())[0].cpu().numpy()
print("z shape", mu.shape, "vb counts", np.bincount(vb, minlength=3))


def fresh_clf(z, y, n_out, lr=1e-2, epochs=40):
    z = torch.from_numpy(z).float()
    y = torch.from_numpy(y).long()
    n = len(y)
    ntr = int(n * 0.9)
    perm = torch.randperm(n)
    ztr, ytr = z[perm[:ntr]].cuda(), y[perm[:ntr]].cuda()
    zva, yva = z[perm[ntr:]].cuda(), y[perm[ntr:]].cuda()
    clf = nn.Sequential(nn.Linear(z.shape[1], n_out)).cuda()
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for ep in range(epochs):
        clf.train()
        loss = nn.functional.cross_entropy(clf(ztr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        acc = (clf(zva).argmax(1) == yva).float().mean().item()
    return acc


print("=== per speed-bin direction leakage (fresh linear z->8) ===")
results = {}
for b in range(3):
    m = vb == b
    zb, yb = mu[m], angle_bin[m]
    maj = np.bincount(yb).max() / len(yb)
    acc = fresh_clf(zb, yb, 8)
    results[b] = {"n": int(len(yb)), "majority": round(float(maj), 3),
                  "fresh_dir_acc": round(float(acc), 3)}
    print(f"vb={b} n={len(yb)} majority={maj:.3f} fresh_dir_acc={acc:.3f} "
          f"leak={acc - maj:+.3f}")

# overall (all bins) for reference
maj_all = np.bincount(angle_bin).max() / len(angle_bin)
acc_all = fresh_clf(mu, angle_bin, 8)
results["all"] = {"n": len(mu), "majority": round(float(maj_all), 3),
                  "fresh_dir_acc": round(float(acc_all), 3)}
print(f"all   n={len(mu)} majority={maj_all:.3f} fresh_dir_acc={acc_all:.3f} "
      f"leak={acc_all - maj_all:+.3f}")

with open(os.path.join(out_dir, "probe_per_bin.json"), "w") as f:
    import json
    json.dump(results, f, indent=1)
print("saved", os.path.join(out_dir, "probe_per_bin.json"))
