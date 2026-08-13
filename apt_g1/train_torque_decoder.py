"""Train the paper-style torque decoder: (phase + command) -> 12-d leg torque.

Input  : [sin(phi), cos(phi)] from the v9 phase router + 14-d command feature.
Target : normalized PD torque tau = kp*(q_des - q) - kd*qdot for the 12 lower
         MuJoCo-order joints, recovered by ``recover_torque_data.py``.

Model  : MLP 16 -> 512 -> 512 -> 12 (GELU, dropout).  Saves the best checkpoint
         by val RMSE (raw N*m) to outputs/torque_decoder_v9/.
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


class TorqueDecoder(nn.Module):
    def __init__(self, d_in=16, d_out=12, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(LOCAL, "data", "torque_data"))
    ap.add_argument("--out", default=os.path.join(LOCAL, "outputs", "torque_decoder_v9"))
    cli = ap.parse_args()
    D = cli.data
    OUT = cli.out
    x = np.load(os.path.join(D, "input.npy"))
    tau = np.load(os.path.join(D, "tau.npy"))
    gid_path = os.path.join(D, "group_ids.npy")
    gid = (
        np.load(gid_path)
        if os.path.exists(gid_path)
        else np.zeros(x.shape[0], dtype=np.int64)
    )
    tau_mean = np.load(os.path.join(D, "tau_norm_mean.npy"))
    tau_std = np.load(os.path.join(D, "tau_norm_std.npy"))
    y = (tau - tau_mean) / tau_std
    print("data", x.shape, y.shape, flush=True)

    # stratified 90/10 split by group
    rng = np.random.default_rng(0)
    tr_idx, va_idx = [], []
    for gi in np.unique(gid):
        rows = np.where(gid == gi)[0]
        perm = rng.permutation(rows)
        n_va = max(1, int(0.1 * len(rows)))
        va_idx.append(perm[:n_va])
        tr_idx.append(perm[n_va:])
    tr_idx = np.concatenate(tr_idx)
    va_idx = np.concatenate(va_idx)
    print("train", len(tr_idx), "val", len(va_idx), flush=True)

    model = TorqueDecoder().train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    lossf = nn.HuberLoss(delta=1.0)
    Xtr = torch.from_numpy(x[tr_idx])
    Ytr = torch.from_numpy(y[tr_idx])
    Xva = torch.from_numpy(x[va_idx])
    Yva = torch.from_numpy(y[va_idx])
    B = 512
    best = 1e9
    os.makedirs(OUT, exist_ok=True)
    for ep in range(301):
        model.train()
        perm = torch.randperm(len(tr_idx))
        for s in range(0, len(tr_idx), B):
            mb = perm[s : s + B]
            opt.zero_grad()
            loss = lossf(model(Xtr[mb]), Ytr[mb])
            loss.backward()
            opt.step()
        sched.step()
        if ep % 20 == 0 or ep == 300:
            model.eval()
            with torch.no_grad():
                p = model(Xva)
                rmse_norm = float((p - Yva).pow(2).mean().sqrt())
                rmse_raw = float(((p * tau_std) - (Yva * tau_std)).pow(2).mean().sqrt())
                mae_raw = float(((p * tau_std) - (Yva * tau_std)).abs().mean())
            print(
                f"ep {ep:3d} train {float(loss):.5f} val_rmse_norm {rmse_norm:.5f} "
                f"val_rmse_raw {rmse_raw:.4f} Nm val_mae {mae_raw:.4f} Nm",
                flush=True,
            )
            if mae_raw < best:
                best = mae_raw
                torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
                print("  saved best", flush=True)
    # final per-group report
    model.eval()
    with torch.no_grad():
        p = model(Xva)
    p_raw = p.numpy() * tau_std
    y_raw = Yva.numpy() * tau_std
    per_joint = np.abs(p_raw - y_raw).mean(axis=0)
    per_group = {}
    for gi in np.unique(gid[va_idx]):
        m = gid[va_idx] == gi
        per_group[str(gi)] = round(float(np.abs(p_raw[m] - y_raw[m]).mean()), 4)
    print("best val MAE raw:", round(best, 4), "Nm", flush=True)
    print("per-joint MAE:", np.round(per_joint, 3).tolist(), flush=True)
    print("per-group MAE:", per_group, flush=True)
    json.dump(
        {
            "best_val_mae_raw": round(best, 4),
            "per_joint_mae": per_joint.tolist(),
            "per_group_mae": per_group,
            "tau_mean": tau_mean.tolist(),
            "tau_std": tau_std.tolist(),
            "input_dim": 16,
            "output_dim": 12,
        },
        open(os.path.join(OUT, "meta.json"), "w"),
        indent=1,
    )
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
