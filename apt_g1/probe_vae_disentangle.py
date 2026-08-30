"""Probe: does the trained E37 VAE's z still encode direction?

Loads outputs/token_vae_e37/vae.pt, encodes walk rows to z, then trains a FRESH
(non-adversarial) classifier z -> 8-bin direction and reports val accuracy vs
the majority-class baseline (~0.70, since bin 4 = 70% of data). If the fresh
classifier clearly beats majority, z still leaks direction (disentanglement
failed); if it is at/below majority, z is direction-invariant.
"""
import os
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
from train_token_vae_e37 import DirSpeedPhaseTokenVAE, build_windows, walk_phase_rate

torch.manual_seed(0)
np.random.seed(0)
data_dir = "data/exp_all3"
out_dir = "outputs/token_vae_e37"

tok = np.load(os.path.join(data_dir, "token.npy")).astype(np.float32)
mode = np.load(os.path.join(data_dir, "mode.npy"))
angle_bin = np.load(os.path.join(data_dir, "angle_bin.npy")).astype(np.int64)

x = build_windows(tok, 10)
model = DirSpeedPhaseTokenVAE().cuda()
model.load_state_dict(torch.load(os.path.join(out_dir, "vae.pt"), map_location="cuda:0"))
model.eval()

# encode ALL rows to mu (deterministic z)
with torch.no_grad():
    mu = model.encode(torch.from_numpy(x).cuda())[0].cpu().numpy()
print("z shape", mu.shape)
# majority baseline
maj = np.bincount(angle_bin).max() / len(angle_bin)
print("majority baseline", round(maj, 3))

# fresh linear classifier z -> 8
z = torch.from_numpy(mu).float()
y = torch.from_numpy(angle_bin).long()
n = len(y)
ntr = int(n * 0.9)
perm = torch.randperm(n)
ztr, ytr = z[perm[:ntr]].cuda(), y[perm[:ntr]].cuda()
zva, yva = z[perm[ntr:]].cuda(), y[perm[ntr:]].cuda()

clf = nn.Sequential(nn.Linear(16, 8)).cuda()
opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
for ep in range(40):
    clf.train()
    loss = nn.functional.cross_entropy(clf(ztr), ytr)
    opt.zero_grad(); loss.backward(); opt.step()
clf.eval()
with torch.no_grad():
    acc = (clf(zva).argmax(1) == yva).float().mean().item()
print("fresh linear classifier val acc", round(acc, 3), "(majority", round(maj, 3), ")")
