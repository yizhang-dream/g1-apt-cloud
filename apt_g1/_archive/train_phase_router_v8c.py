"""v8c: shared token decoder with per-group phase predictors.

Findings from v8 (see train_phase_router_v8.py): pooling all groups into one
phase regressor destroys phase accuracy (per-group ~0.005 rad -> shared
1.6-2.3 rad), so the proprio->phase map must stay per-group. The prototype
lookup, however, can be replaced by ONE shared decoder:

    token dec: (sin phi, cos phi, cmd 14, frame one-hot 14) -> 64-d token

At inference, an unseen command keeps the fallback group's phase frame
(per-group phase net) and passes the *unseen* cmd + that frame one-hot to the
decoder, which interpolates the token manifold across commands instead of
snapping to the nearest group's discrete prototypes.

Output: outputs/distill_v8c/ (decoder + meta; phase nets reused from v6).
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

D = "/home/cvgluser/ros2_data/apt_g1/data/exp_all"
ODIR6 = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_v6"
ODIR8C = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_v8c"
os.makedirs(ODIR8C, exist_ok=True)

cmd = np.load(D + "/cmd.npy")
token = np.load(D + "/token.npy")
mode = np.load(D + "/mode.npy")
speed = np.load(D + "/speed.npy")
ab = np.load(D + "/angle_bin.npy")
modes_list = np.load(D + "/meta_modes.npy")
meta = json.load(open(ODIR6 + "/phase_meta.json"))

groups = sorted({int(k) for k in meta})
gi_of = {tuple(md["group"]): int(gi) for gi, md in meta.items()}
frame = np.zeros((len(token), len(groups)), dtype=np.float32)
sc_all = np.zeros((len(token), 2), dtype=np.float32)
row_count = 0
for gi, md in meta.items():
    gi = int(gi)
    g = tuple(md["group"])
    rows = np.where((mode == g[0]) & (np.abs(speed - g[1]) < 1e-6) & (ab == g[2]))[0]
    T = token[rows]
    mu = np.array(md["mu"], dtype=np.float32)
    V2 = np.array(md["V2"], dtype=np.float32)
    proj = (T - mu) @ V2.T
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    sc_all[rows, 0] = np.sin(phi)
    sc_all[rows, 1] = np.cos(phi)
    frame[rows, groups.index(gi)] = 1.0
    row_count += len(rows)
print("rows", row_count, "groups", len(groups))

Xt = np.concatenate(
    [
        sc_all,
        cmd.astype(np.float32),
        frame,
    ],
    axis=1,
).astype(np.float32)
yt = token.astype(np.float32)


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=512, drop=0.1):
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


def train(x, y, tag, epochs=80, batch=512, lr=1e-3, wd=1e-5, seed=0):
    torch.manual_seed(seed)
    n = len(x)
    idx = np.random.RandomState(seed).permutation(n)
    ntr = int(n * 0.8)
    tr, va = idx[:ntr], idx[ntr:]
    ds = TensorDataset(torch.from_numpy(x[tr]), torch.from_numpy(y[tr]))
    ld = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    net = MLP(x.shape[1], y.shape[1]).cuda()
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
            err = float(lossf(net(xv), yv).item())
        if err < best:
            best = err
            torch.save(net.state_dict(), f"{ODIR8C}/{tag}.pt")
        if (ep + 1) % 10 == 0:
            print(f"{tag} ep {ep+1} loss {tl/tb:.5f} val {err:.5f}", flush=True)
    print(f"{tag} best val {best:.6f}", flush=True)


print("training shared token decoder:", Xt.shape, yt.shape)
train(Xt, yt, "token_dec")

json.dump(
    {
        "modes_list": modes_list.tolist(),
        "groups": {gi: md["group"] for gi, md in meta.items()},
        "groups_order": groups,
        "d_cmd": 14,
        "d_frame": len(groups),
        "note": "v8c: per-group phase nets (v6) + shared continuous token decoder",
    },
    open(f"{ODIR8C}/v8c_meta.json", "w"),
    indent=1,
)
print("v8c ready:", ODIR8C)
