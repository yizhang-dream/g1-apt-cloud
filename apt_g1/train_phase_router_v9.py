"""v9: rebuild the phase-router from the merged dataset (exp_all3).

Groups: (mode, round(speed,2), 8-bin direction). Per group: PCA circular phase,
per-group MLP (proprio 930 + cmd 14 -> sin/cos), 40-bin prototypes (mean; v6
BEST overrides for existing marginal groups). Outputs outputs/distill_v9/ in the
same format as distill_final so PhaseRouterEncoder loads it unchanged.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

D = "/home/cvgluser/ros2_data/apt_g1/data/exp_all3"
ODIR = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_v9"
os.makedirs(ODIR, exist_ok=True)

proprio = np.load(D + "/proprio.npy")
cmd = np.load(D + "/cmd.npy")
token = np.load(D + "/token.npy")
mode = np.load(D + "/mode.npy")
speed = np.load(D + "/speed.npy")
ab = np.load(D + "/angle_bin.npy")
modes_list = np.load(D + "/meta_modes.npy")
print("data", proprio.shape, cmd.shape, token.shape)

# v6 per-group prototype config overrides (mean/median/nearest x B)
BEST = {
    (17, -1.0, 4): ("median", 40),
    (1, 0.2, 5): ("mean", 64),
    (1, 0.2, 2): ("nearest", 40),
    (1, 0.2, 6): ("nearest", 40),
}


class PhaseNet(nn.Module):
    def __init__(self, d_in, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def dsign(x):
    return 1 if x > 1e-6 else (-1 if x < -1e-6 else 0)


# ---- groups ---------------------------------------------------------------
groups = {}
for i in range(len(mode)):
    g = (int(mode[i]), round(float(speed[i]), 2), int(ab[i]))
    groups.setdefault(g, []).append(i)
groups = {g: np.array(v) for g, v in groups.items() if len(v) >= 300}
print("groups:", {str(k): len(v) for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))})

tr_idx = np.arange(len(proprio))
pmean = proprio[tr_idx].mean(0, keepdims=True).astype(np.float32)
pstd = proprio[tr_idx].std(0, keepdims=True).astype(np.float32) + 1e-6
Pmat = ((proprio - pmean) / pstd).astype(np.float32)
X = np.concatenate([Pmat, cmd.astype(np.float32)], axis=1).astype(np.float32)

meta = {}
for gi, (g, rows) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
    T = token[rows]
    mu = T.mean(0)
    Tc = T - mu
    _, _, Vt = np.linalg.svd(Tc, full_matrices=False)
    V2 = Vt[:2]
    proj = Tc @ V2.T
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    how, B = BEST.get(g, ("mean", 40))
    bi = np.floor((phi + np.pi) / (2 * np.pi) * B).astype(int) % B
    proto = np.zeros((B, 64), dtype=np.float32)
    centers = np.linspace(-np.pi, np.pi, B, endpoint=False)
    for b in range(B):
        sel = np.where(bi == b)[0]
        if len(sel) == 0:
            proto[b] = proto[b - 1] if b > 0 else T[0]
            continue
        if how == "mean":
            proto[b] = T[sel].mean(0)
        elif how == "median":
            proto[b] = np.median(T[sel], axis=0)
        else:
            d = np.abs((phi[sel] - centers[b] + np.pi) % (2 * np.pi) - np.pi)
            proto[b] = T[sel[int(np.argmin(d))]]
    proto = np.clip(np.round(proto * 16) / 16, -1, 1).astype(np.float32)
    np.save(f"{ODIR}/proto_g{gi}.npy", proto)

    y = np.stack([np.sin(phi), np.cos(phi)], axis=1).astype(np.float32)
    ntr = int(len(rows) * 0.8)
    ds = TensorDataset(torch.from_numpy(X[rows[:ntr]]), torch.from_numpy(y[:ntr]))
    ld = DataLoader(ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    net = PhaseNet(930 + cmd.shape[1]).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    best = 1e9
    xv = torch.from_numpy(X[rows[ntr:]]).cuda()
    yv = torch.from_numpy(y[ntr:]).cuda()
    for ep in range(60):
        net.train()
        tl, tb = 0.0, 0
        for xb, yb in ld:
            xb = xb.cuda(non_blocking=True)
            yb = yb.cuda(non_blocking=True)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            tb += len(yb)
        net.eval()
        with torch.no_grad():
            err = float(lossf(net(xv), yv).item())
        if err < best:
            best = err
            torch.save(net.state_dict(), f"{ODIR}/phase_g{gi}.pt")
    with torch.no_grad():
        pv = net(xv).cpu().numpy()
    a = np.arctan2(pv[:, 0], pv[:, 1])
    b = np.arctan2(yv.cpu().numpy()[:, 0], yv.cpu().numpy()[:, 1])
    ang_err = float(np.mean(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)))
    meta[str(gi)] = dict(
        group=list(g),
        rows_all=int(len(rows)),
        rows=int(len(rows)),
        mu=mu.tolist(),
        V2=V2.tolist(),
        n_bins=B,
        proto_how=how,
        val_mse=round(best, 6),
        val_ang_err=round(ang_err, 4),
    )
    print(
        f"group {g} n={len(rows)} how={how} B={B} val_mse={best:.6f} ang_err={ang_err:.4f}",
        flush=True,
    )

np.savez(f"{ODIR}/phase_norm.npz", pmean=pmean[0], pstd=pstd[0])
json.dump(
    {
        **meta,
        "_meta": {
            "data": "exp_all3",
            "modes_list": modes_list.tolist(),
            "note": "v9 rebuild from merged exp1+exp2+exp3",
        },
    },
    open(f"{ODIR}/phase_meta.json", "w"),
    indent=1,
)
print("v9 ready:", ODIR)
