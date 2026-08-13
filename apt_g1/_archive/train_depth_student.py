"""Train the P2-lite depth student: (ego depth + proprio) -> 9x9 elevation patch.

Paper stage-4 distillation analogue with a real sensor input: a small CNN on a
48x64 ego depth image plus an MLP on 34-d proprio regress the privileged local
elevation patch (81 = 9x9 @ 0.15 m, lookahead 0.6 m).  Trains on seeds 0/1 and
validates on seed 2 (held-out trajectory).

Outputs: outputs/depth_student/{model.pt, meta.json}
"""

from __future__ import annotations

import io
import json
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn

LOCAL = r"C:\Users\zyz\Documents\gr00t\apt_g1"
D = os.path.join(LOCAL, "outputs", "depth_data")
OUT = os.path.join(LOCAL, "outputs", "depth_student")


class DepthStudent(nn.Module):
    def __init__(self, d_prop=34, d_out=81):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_prop, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 32, 256),
            nn.GELU(),
            nn.Linear(256, d_out),
        )

    def forward(self, depth, prop):
        z = self.cnn(depth[:, None]).flatten(1)
        p = self.mlp(prop)
        return self.head(torch.cat([z, p], dim=1))


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(LOCAL, "outputs", "depth_data"))
    ap.add_argument("--out", default=os.path.join(LOCAL, "outputs", "depth_student"))
    cli = ap.parse_args()
    D = cli.data
    OUT = cli.out
    depth = np.load(os.path.join(D, "depth.npy"))
    patch = np.load(os.path.join(D, "patch.npy"))
    prop = np.load(os.path.join(D, "proprio.npy"))
    n = len(depth)
    print("data", depth.shape, patch.shape, prop.shape, flush=True)

    # depth normalization: meters -> [0, 1] over [0.2, 8.0]
    d_norm = np.clip((depth - 0.2) / 7.8, 0.0, 1.0).astype(np.float32)
    # split by seed: 700 frames = seeds 0(252), 1(248), 2(200)
    tr = slice(0, 500)
    va = slice(500, 700)
    Xd_tr = torch.from_numpy(d_norm[tr])
    Xp_tr = torch.from_numpy(prop[tr])
    Y_tr = torch.from_numpy(patch[tr])
    Xd_va = torch.from_numpy(d_norm[va])
    Xp_va = torch.from_numpy(prop[va])
    Y_va = torch.from_numpy(patch[va])
    print("train", tr.stop - tr.start, "val", va.stop - va.start, flush=True)

    model = DepthStudent().train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    lossf = nn.MSELoss()
    best = 1e9
    os.makedirs(OUT, exist_ok=True)
    for ep in range(201):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xd_tr, Xp_tr), Y_tr)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 20 == 0 or ep == 200:
            model.eval()
            with torch.no_grad():
                p = model(Xd_va, Xp_va)
                mae = float((p - Y_va).abs().mean())
                corr = float(
                    np.corrcoef(p.numpy().ravel(), Y_va.numpy().ravel())[0, 1]
                )
            print(
                f"ep {ep:3d} train {float(loss):.5f} val_mae {mae:.5f} m "
                f"corr {corr:.4f}",
                flush=True,
            )
            if mae < best:
                best = mae
                torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
    model.eval()
    with torch.no_grad():
        p = model(Xd_va, Xp_va).numpy()
    y = Y_va.numpy()
    per_cell = np.abs(p - y).mean(axis=0)
    json.dump(
        {
            "best_val_mae": round(best, 5),
            "best_val_corr": round(
                float(np.corrcoef(p.ravel(), y.ravel())[0, 1]), 4
            ),
            "per_cell_mae_max": round(float(per_cell.max()), 5),
            "per_cell_mae_mean": round(float(per_cell.mean()), 5),
            "split": "train seeds 0,1 / val seed 2",
            "p1_comparison": {"corr": 0.954, "mae": 0.0085, "note": "coarse 3x3 proxy"},
        },
        open(os.path.join(OUT, "meta.json"), "w"),
        indent=1,
    )
    print("saved", OUT, "best val MAE", round(best, 5), flush=True)


if __name__ == "__main__":
    main()
