"""TVAE-lite: structured continuous latent for the SONIC token router.

Paper representation-learning analog: encoder maps (proprio, cmd) to a 16-d
latent z; decoder maps (z, cmd) to the 64-d SONIC token. KL(z || N(0,I)) with
beta annealing shapes a smooth, regularized manifold -- the property the
phase-router prototypes have and the plain v8c regression decoder lacked.

Outputs: outputs/distill_vae/{encoder.pt, decoder.pt, vae_meta.json, norm.npz}
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

D = "/home/cvgluser/ros2_data/apt_g1/data/exp_all3"
ODIR = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_vae"
os.makedirs(ODIR, exist_ok=True)

proprio = np.load(D + "/proprio.npy")
cmd = np.load(D + "/cmd.npy")
token = np.load(D + "/token.npy")
print("data", proprio.shape, cmd.shape, token.shape)

pmean = proprio.mean(0, keepdims=True).astype(np.float32)
pstd = proprio.std(0, keepdims=True).astype(np.float32) + 1e-6
P = ((proprio - pmean) / pstd).astype(np.float32)
X = np.concatenate([P, cmd.astype(np.float32)], axis=1).astype(np.float32)
Y = token.astype(np.float32)

Z = 16
BETA_MAX = 0.1
ANNEAL_EPOCHS = 40


class Encoder(nn.Module):
    def __init__(self, d_in, z=Z, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.mu = nn.Linear(hidden, z)
        self.logvar = nn.Linear(hidden, z)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, d_cmd, z=Z, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z + d_cmd, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=-1))


def kl_normal(mu, logvar):
    return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1)


torch.manual_seed(0)
np.random.seed(0)
n = len(X)
idx = np.random.RandomState(0).permutation(n)
ntr = int(n * 0.8)
tr, va = idx[:ntr], idx[ntr:]
ds = TensorDataset(
    torch.from_numpy(X[tr]), torch.from_numpy(Y[tr]), torch.from_numpy(cmd[tr])
)
ld = DataLoader(ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)

enc = Encoder(X.shape[1]).cuda()
dec = Decoder(cmd.shape[1]).cuda()
opt = torch.optim.AdamW(
    list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-5
)
recon_f = nn.MSELoss()

xv = torch.from_numpy(X[va]).cuda()
yv = torch.from_numpy(Y[va]).cuda()
cv = torch.from_numpy(cmd[va]).cuda()

best = 1e9
for ep in range(100):
    beta = BETA_MAX * min(1.0, ep / ANNEAL_EPOCHS)
    enc.train()
    dec.train()
    tl, tk, tb = 0.0, 0.0, 0
    for xb, yb, cb in ld:
        xb = xb.cuda(non_blocking=True)
        yb = yb.cuda(non_blocking=True)
        cb = cb.cuda(non_blocking=True)
        mu, lv = enc(xb)
        z = mu + torch.randn_like(mu) * lv.mul(0.5).exp()
        rec = dec(z, cb)
        loss = recon_f(rec, yb) + beta * kl_normal(mu, lv).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tl += loss.item() * len(yb)
        tk += kl_normal(mu, lv).sum().item()
        tb += len(yb)
    enc.eval()
    dec.eval()
    with torch.no_grad():
        mu, lv = enc(xv)
        rec = dec(mu, cv)
        val_rec = float(recon_f(rec, yv).item())
        val_kl = float(kl_normal(mu, lv).mean().item())
    if val_rec < best:
        best = val_rec
        torch.save(enc.state_dict(), f"{ODIR}/encoder.pt")
        torch.save(dec.state_dict(), f"{ODIR}/decoder.pt")
    if (ep + 1) % 10 == 0:
        print(
            f"ep {ep+1} loss {tl/tb:.5f} kl {tk/tb:.4f} "
            f"val_rec {val_rec:.6f} val_kl {val_kl:.4f} beta {beta:.3f}",
            flush=True,
        )
print("best val_rec", best)

np.savez(f"{ODIR}/norm.npz", pmean=pmean[0], pstd=pstd[0])
json.dump(
    {
        "z_dim": Z,
        "d_cmd": cmd.shape[1],
        "d_proprio": 930,
        "beta_max": BETA_MAX,
        "best_val_rec": best,
        "note": "TVAE-lite on exp_all3 (proprio+cmd -> z16 -> token)",
    },
    open(f"{ODIR}/vae_meta.json", "w"),
    indent=1,
)
print("vae ready:", ODIR)
