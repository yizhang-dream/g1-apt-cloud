"""Probe: does the trained E39 VAE's z still encode direction or speed?

Loads outputs/token_vae_e39/vae.pt, encodes all rows to mu, then trains FRESH
(non-adversarial) linear classifiers:
  - z -> 8 direction bins, val acc vs majority baseline (~0.70, bin4 = 70%)
  - z -> 3 speed bins,  val acc vs majority baseline (~1/3)
If a fresh classifier clearly beats its majority baseline, z still leaks that
attribute (disentanglement failed); at/below majority => invariant.
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
out_dir = "outputs/token_vae_e39"

tok = np.load(os.path.join(data_dir, "token.npy")).astype(np.float32)
mode = np.load(os.path.join(data_dir, "mode.npy"))
angle_bin = np.load(os.path.join(data_dir, "angle_bin.npy")).astype(np.int64)

# recompute speed bins with the SAME deterministic recipe as training
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
print("z shape", mu.shape)

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

maj_dir = np.bincount(angle_bin).max() / len(angle_bin)
maj_spd = np.bincount(vb).max() / len(vb)
print("majority baselines: dir", round(maj_dir, 3), "speed", round(maj_spd, 3))

acc_dir = fresh_clf(mu, angle_bin, 8)
print("fresh linear classifier dir  val acc", round(acc_dir, 3), "(majority", round(maj_dir, 3), ")")
acc_spd = fresh_clf(mu, vb, 3)
print("fresh linear classifier spd  val acc", round(acc_spd, 3), "(majority", round(maj_spd, 3), ")")

with open(os.path.join(out_dir, "probe.json"), "w") as f:
    import json
    json.dump({"majority_dir": maj_dir, "majority_spd": maj_spd,
               "fresh_dir_acc": acc_dir, "fresh_spd_acc": acc_spd}, f, indent=1)
print("saved", os.path.join(out_dir, "probe.json"))
