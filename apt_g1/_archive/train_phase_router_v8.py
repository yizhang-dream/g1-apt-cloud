"""v8: continuous conditional router (network-generalization experiment).

Replaces the per-(mode, speed, angle-bin) group MLPs + discrete 40-bin
prototype lookup with TWO shared networks:

    phase net:   (proprio 930, cmd 14)          -> (sin phi, cos phi)
    token dec:   (sin phi, cos phi, cmd 14)     -> 64-d token (k/16 lattice)

The phase labels are the per-group PCA circular phases from the v6 meta
(same coordinate frames as the working router), so the learned decoder is a
continuous analogue of the prototype table: unseen commands interpolate
instead of falling back to the nearest group.

Runs on the MuJoCo training venv (torch + CUDA). Output: outputs/distill_v8/
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

D = "/home/cvgluser/ros2_data/apt_g1/data/exp_all"
ODIR6 = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_v6"
ODIR8 = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_v8"
os.makedirs(ODIR8, exist_ok=True)

proprio = np.load(D + "/proprio.npy")
cmd = np.load(D + "/cmd.npy")
token = np.load(D + "/token.npy")
mode = np.load(D + "/mode.npy")
speed = np.load(D + "/speed.npy")
ab = np.load(D + "/angle_bin.npy")
modes_list = np.load(D + "/meta_modes.npy")
norm = np.load(ODIR6 + "/phase_norm.npz")
meta = json.load(open(ODIR6 + "/phase_meta.json"))

pmean = norm["pmean"].ravel().astype(np.float32)
pstd = norm["pstd"].ravel().astype(np.float32)
assert pmean.shape[0] == 930 and cmd.shape[1] == 14


def phase_labels_for_group(gi: int) -> np.ndarray:
    md = meta[str(gi)]
    g = tuple(md["group"])
    rows = np.where((mode == g[0]) & (np.abs(speed - g[1]) < 1e-6) & (ab == g[2]))[0]
    T = token[rows]
    mu = np.array(md["mu"], dtype=np.float32)
    V2 = np.array(md["V2"], dtype=np.float32)
    proj = (T - mu) @ V2.T
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    return rows, np.stack([np.sin(phi), np.cos(phi)], axis=1).astype(np.float32)


# ---- labels: per-group PCA phase -------------------------------------------
row_list = []
sc_list = []
group_of_row = np.empty(len(token), dtype=np.int64)
for gi, md in meta.items():
    rows, sc = phase_labels_for_group(int(gi))
    row_list.append(rows)
    sc_list.append(sc)
    group_of_row[rows] = int(gi)
rows_all = np.concatenate(row_list)
sc_all = np.concatenate(sc_list)
print("rows with labels", len(rows_all), "of", len(token))

Xp = np.concatenate(
    [((proprio - pmean) / pstd).astype(np.float32), cmd.astype(np.float32)], axis=1
).astype(np.float32)
Xt = np.concatenate([sc_all, cmd[rows_all].astype(np.float32)], axis=1).astype(
    np.float32
)
yp = sc_all
yt = token[rows_all]


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=512, drop=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def train_net(x, y, d_out, epochs, tag, batch=512, lr=1e-3, wd=1e-5, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(x)
    idx = np.random.RandomState(seed).permutation(n)
    ntr = int(n * 0.8)
    tr, va = idx[:ntr], idx[ntr:]
    ds = TensorDataset(torch.from_numpy(x[tr]), torch.from_numpy(y[tr]))
    ld = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    net = MLP(x.shape[1], d_out).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    xv = torch.from_numpy(x[va]).cuda()
    yv = torch.from_numpy(y[va]).cuda()
    best = 1e9
    for ep in range(epochs):
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
            pv = net(xv)
        err = float(lossf(pv, yv).item())
        if err < best:
            best = err
            torch.save(net.state_dict(), f"{ODIR8}/{tag}.pt")
        if (ep + 1) % 10 == 0:
            print(f"{tag} ep {ep+1} loss {tl/tb:.5f} val {err:.5f}", flush=True)
    print(f"{tag} best val {best:.6f}", flush=True)


print("training phase net: X", Xp.shape, "y", yp.shape)
train_net(Xp, yp, 2, 60, "phase_net")

# per-group phase net val error (for the log)
print("per-group phase val angle error (rad):")
net = MLP(Xp.shape[1], 2).cuda()
net.load_state_dict(torch.load(f"{ODIR8}/phase_net.pt", map_location="cuda"))
net.eval()
with torch.no_grad():
    pred = net(torch.from_numpy(Xp[rows_all]).cuda()).cpu().numpy()
for gi in sorted({int(k) for k in meta}):
    sel = group_of_row[rows_all] == gi
    if sel.sum() == 0:
        continue
    a = np.arctan2(pred[sel, 0], pred[sel, 1])
    b = np.arctan2(yp[sel, 0], yp[sel, 1])
    d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    print(f"  group {gi} n={int(sel.sum())} ang_err={float(d.mean()):.4f}")

print("training token decoder: X", Xt.shape, "y", yt.shape)
train_net(Xt, yt, 64, 80, "token_dec")

json.dump(
    {
        "modes_list": modes_list.tolist(),
        "groups": {gi: md["group"] for gi, md in meta.items()},
        "d_proprio": 930,
        "d_cmd": 14,
        "note": "v8 continuous conditional router: shared phase net + token decoder",
    },
    open(f"{ODIR8}/v8_meta.json", "w"),
    indent=1,
)
np.savez(f"{ODIR8}/phase_norm.npz", pmean=pmean, pstd=pstd)
print("v8 ready:", ODIR8)
