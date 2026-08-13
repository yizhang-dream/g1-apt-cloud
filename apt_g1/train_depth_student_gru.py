"""P2-lite v2: depth student with temporal structure (CNN + GRU + BPTT).

The paper's student is a CNN + GRU encoder trained with truncated BPTT
(DAgger in the loop).  Our frame-wise CNN student (P2-lite v1) only reached
corr ~0.74 while the geometric unprojection upper bound is 0.93-0.97.  This
script tests whether temporal structure recovers most of that gap.

Data: outputs/depth_data_hi2 (700 frames; seeds 0/1 train, seed 2 val).
Output: outputs/depth_student_gru/{model.pt, meta.json}
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
SEQ = 32


class DepthStudentGRU(nn.Module):
    def __init__(self, d_prop=34, d_out=81, hidden=128):
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
        self.gru = nn.GRU(64 + 32, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, d_out),
        )

    def forward(self, depth, prop):
        # depth/prop: (B, T, ...)
        B, T = depth.shape[:2]
        z = self.cnn(depth.reshape(B * T, 1, depth.shape[2], depth.shape[3])).flatten(1)
        z = z.reshape(B, T, -1)
        p = self.mlp(prop.reshape(B * T, -1)).reshape(B, T, -1)
        h, _ = self.gru(torch.cat([z, p], dim=-1))
        return self.head(h)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dirs",
        default=",".join(
            [
                os.path.join(LOCAL, "outputs", "depth_data_hi2"),
                os.path.join(LOCAL, "outputs", "depth_data_004"),
                os.path.join(LOCAL, "outputs", "depth_data_008"),
            ]
        ),
    )
    ap.add_argument("--out", default=os.path.join(LOCAL, "outputs", "depth_student_gru_all"))
    cli = ap.parse_args()
    OUT = cli.out

    depth_parts, patch_parts, prop_parts = [], [], []
    for d in cli.dirs.split(","):
        depth_parts.append(np.load(os.path.join(d, "depth.npy")))
        patch_parts.append(np.load(os.path.join(d, "patch.npy")))
        prop_parts.append(np.load(os.path.join(d, "proprio.npy")))
    depth = np.concatenate(depth_parts)
    patch = np.concatenate(patch_parts)
    prop = np.concatenate(prop_parts)
    d_norm = np.clip((depth - 0.2) / 7.8, 0.0, 1.0).astype(np.float32)

    # split by seed within each dir (seeds 0/1 train, seed 2 val)
    train_seqs, val_seqs = [], []
    off = 0
    for d in cli.dirs.split(","):
        seeds = np.load(os.path.join(d, "seeds.npy"))
        n = len(seeds)
        tr = np.where(seeds != 2)[0]
        va = np.where(seeds == 2)[0]
        train_seqs.append((off + tr.min(), off + tr.max() + 1))
        val_seqs.append((off + va.min(), off + va.max() + 1))
        off += n
    print("frames", len(depth), "train ranges", train_seqs, "val ranges", val_seqs, flush=True)

    def chunks(seqs_):
        out = []
        for a, b in seqs_:
            for s in range(a, b - SEQ + 1, SEQ):
                out.append((s, s + SEQ))
            if b - a > SEQ:
                out.append((b - SEQ, b))
        return out

    tr_chunks = chunks(train_seqs)
    va_chunks = chunks(val_seqs)
    print("train chunks", len(tr_chunks), "val chunks", len(va_chunks), flush=True)

    model = DepthStudentGRU().train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
    lossf = nn.MSELoss()
    best = 1e9
    os.makedirs(OUT, exist_ok=True)
    for ep in range(151):
        model.train()
        order = np.random.permutation(len(tr_chunks))
        for k in order:
            a, b = tr_chunks[k]
            xd = torch.from_numpy(d_norm[a:b][None])
            xp = torch.from_numpy(prop[a:b][None])
            y = torch.from_numpy(patch[a:b][None])
            opt.zero_grad()
            loss = lossf(model(xd, xp), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 15 == 0 or ep == 150:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for a, b in va_chunks:
                    xd = torch.from_numpy(d_norm[a:b][None])
                    xp = torch.from_numpy(prop[a:b][None])
                    preds.append(model(xd, xp)[0].numpy())
                    trues.append(patch[a:b])
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            mae = float(np.abs(preds - trues).mean())
            corr = float(np.corrcoef(preds.ravel(), trues.ravel())[0, 1])
            print(
                f"ep {ep:3d} train {float(loss):.5f} val_mae {mae:.5f} corr {corr:.4f}",
                flush=True,
            )
            if mae < best:
                best = mae
                torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for a, b in va_chunks:
            xd = torch.from_numpy(d_norm[a:b][None])
            xp = torch.from_numpy(prop[a:b][None])
            preds.append(model(xd, xp)[0].numpy())
            trues.append(patch[a:b])
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    json.dump(
        {
            "best_val_mae": round(best, 5),
            "val_corr": round(float(np.corrcoef(preds.ravel(), trues.ravel())[0, 1]), 4),
            "seq_len": SEQ,
            "compare": {
                "cnn_frame": {"mae": 0.0468, "corr": 0.743},
                "geom_upper": {"mae": 0.0275, "corr": 0.93},
                "p1_proxy": {"mae": 0.0085, "corr": 0.954},
            },
        },
        open(os.path.join(OUT, "meta.json"), "w"),
        indent=1,
    )
    print("saved", OUT, "best", round(best, 5), flush=True)


if __name__ == "__main__":
    main()
