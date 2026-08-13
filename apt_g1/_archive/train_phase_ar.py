"""Retrain phase nets with prev-phase (sin,cos) as input — phase autoregression.

For each command group: PCA circular phase labels, per-bin mean prototypes
(same as distill_final), and a PhaseNet whose input is
[norm proprio 930, cmd 14, prev sin, prev cos] -> (sin, cos) of the current
phase. Prev is teacher-forced from the previous contiguous row during training;
at eval the encoder feeds back its own last phase.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


D = "/home/cvgluser/ros2_data/apt_g1/data/exp_all"
SRC = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_final"
DST = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_ar"
B = 40


class PhaseNetAR(nn.Module):
    def __init__(self, d_in, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    proprio = np.load(D + "/proprio.npy")
    cmd = np.load(D + "/cmd.npy")
    token = np.load(D + "/token.npy")
    mode = np.load(D + "/mode.npy")
    speed = np.load(D + "/speed.npy")
    ab = np.load(D + "/angle_bin.npy")
    meta = json.load(open(SRC + "/phase_meta.json"))
    norm = np.load(SRC + "/phase_norm.npz")
    pmean = norm["pmean"].ravel().astype(np.float32)
    pstd = norm["pstd"].ravel().astype(np.float32)
    Xp = ((proprio - pmean) / pstd).astype(np.float32)
    os.makedirs(DST, exist_ok=True)

    out_meta = {}
    for gi, md in meta.items():
        gi = int(gi)
        g = tuple(md["group"])
        rows = np.where(
            (mode == g[0]) & (np.abs(speed - g[1]) < 1e-6) & (ab == g[2])
        )[0]
        if g == (1, 0.2, 4):  # v5 slow restriction: exp1 first slow phase only
            rows = rows[(rows >= 1831) & (rows < 8250)]
        if len(rows) < 100:
            print("skip", gi, g, len(rows))
            continue
        T = token[rows]
        mu = T.mean(0)
        Tc = T - mu
        _, _, Vt = np.linalg.svd(Tc, full_matrices=False)
        V2 = Vt[:2]
        proj = Tc @ V2.T
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        bi = np.floor((phi + np.pi) / (2 * np.pi) * B).astype(int) % B
        proto = np.zeros((B, 64), dtype=np.float32)
        cnt = np.zeros(B, dtype=np.float32)
        for k in range(len(rows)):
            proto[bi[k]] += T[k]
            cnt[bi[k]] += 1
        proto = np.clip(np.round((proto / np.maximum(cnt, 1)[:, None]) * 16) / 16, -1, 1)
        np.save(f"{DST}/proto_g{gi}.npy", proto)

        # prev phase: previous contiguous row's (sin, cos)
        prev = np.zeros((len(rows), 2), dtype=np.float32)
        for k in range(1, len(rows)):
            if rows[k] == rows[k - 1] + 1:
                prev[k] = np.array([np.sin(phi[k - 1]), np.cos(phi[k - 1])], dtype=np.float32)
        y = np.stack([np.sin(phi), np.cos(phi)], axis=1).astype(np.float32)
        X = np.concatenate([Xp[rows], cmd[rows], prev], axis=1).astype(np.float32)
        ntr = int(len(rows) * 0.8)
        ds = TensorDataset(torch.from_numpy(X[:ntr]), torch.from_numpy(y[:ntr]))
        ld = DataLoader(ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
        net = PhaseNetAR(X.shape[1]).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.MSELoss()
        best = 1e9
        for ep in range(60):
            net.train()
            tl = 0.0
            tb = 0
            for x, yy in ld:
                x = x.cuda(non_blocking=True)
                yy = yy.cuda(non_blocking=True)
                opt.zero_grad()
                loss = lossf(net(x), yy)
                loss.backward()
                opt.step()
                tl += loss.item() * len(yy)
                tb += len(yy)
            net.eval()
            if len(rows) - ntr >= 20:
                with torch.no_grad():
                    pv = net(torch.from_numpy(X[ntr:]).cuda()).cpu().numpy()
                err = float(np.mean(1 - (pv[:, 0] * y[ntr:, 0] + pv[:, 1] * y[ntr:, 1])))
            else:
                err = float(tl / tb)
            if err < best:
                best = err
                torch.save(net.state_dict(), f"{DST}/phase_g{gi}.pt")
        out_meta[str(gi)] = dict(
            group=list(g), rows=int(len(rows)), mu=mu.tolist(), V2=V2.tolist(),
            n_bins=B, ang_err=best, d_in=int(X.shape[1]),
        )
        print(f"gi {gi} group {g} rows {len(rows)} ang_err {best:.4f}", flush=True)

    np.savez(f"{DST}/phase_norm.npz", pmean=pmean, pstd=pstd)
    json.dump(out_meta, open(f"{DST}/phase_meta.json", "w"), indent=1)
    shutil.copy(SRC + "/phase_meta.json", DST + "/phase_meta_src.json")
    print("saved", DST, "groups", len(out_meta))


if __name__ == "__main__":
    main()
