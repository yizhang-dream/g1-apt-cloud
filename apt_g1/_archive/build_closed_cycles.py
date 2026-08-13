"""Data track: build closed periodic token cycles and relabeled dataset.

For each command group:
  1. PCA circular phase on the group's official tokens (same as the router);
  2. per-phase-bin mean cycle (this is what the router prototypes are);
  3. close the cycle by blending the last `m` bins toward bin 0 so the wrap
     discontinuity (closure gap) is removed;
  4. save closed prototypes and a relabeled dataset (each row's token replaced
     by the closed cycle token at its phase bin).

Closure metric reported per group: ||mu[B-1] - mu[0]|| (should be ~0 after
closing) vs the original value, alongside the per-step change for context.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np


D = "/home/cvgluser/ros2_data/apt_g1/data/exp_all"
SRC = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_final"
DST = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_closed"
B = 40
M_BLEND = 8  # bins blended toward bin 0 to close the cycle


def main():
    token = np.load(D + "/token.npy")
    mode = np.load(D + "/mode.npy")
    speed = np.load(D + "/speed.npy")
    ab = np.load(D + "/angle_bin.npy")
    meta = json.load(open(SRC + "/phase_meta.json"))
    os.makedirs(DST, exist_ok=True)

    closed_token = token.copy()
    report = {}
    for gi, md in meta.items():
        gi = int(gi)
        g = tuple(md["group"])
        rows = np.where(
            (mode == g[0]) & (np.abs(speed - g[1]) < 1e-6) & (ab == g[2])
        )[0]
        T = token[rows]
        mu = np.array(md["mu"], dtype=np.float32)
        V2 = np.array(md["V2"], dtype=np.float32)
        proj = (T - mu) @ V2.T
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        bi = np.floor((phi + np.pi) / (2 * np.pi) * B).astype(int) % B
        proto = np.zeros((B, 64), dtype=np.float32)
        cnt = np.zeros(B, dtype=np.float32)
        for k in range(len(rows)):
            proto[bi[k]] += T[k]
            cnt[bi[k]] += 1
        proto = proto / np.maximum(cnt, 1)[:, None]
        # closure gap: wrap discontinuity from last bin back to bin 0
        gap = proto[0] - proto[B - 1]
        proto_closed = proto.copy()
        for k in range(M_BLEND):
            idx = B - M_BLEND + k
            proto_closed[idx] = proto[idx] + ((k + 1) / M_BLEND) * gap
        proto_closed = np.clip(np.round(proto_closed * 16) / 16, -1, 1).astype(np.float32)
        np.save(f"{DST}/proto_g{gi}.npy", proto_closed)
        # relabel dataset rows
        for k in range(len(rows)):
            closed_token[rows[k]] = proto_closed[bi[k]]
        step = float(np.mean(np.linalg.norm(T[1:] - T[:-1], axis=1))) if len(T) > 1 else 0.0
        report[str(gi)] = {
            "group": list(g),
            "rows": len(rows),
            "closure_gap_before": round(float(np.linalg.norm(proto[0] - proto[B - 1])), 4),
            "closure_gap_after": round(float(np.linalg.norm(proto_closed[0] - proto_closed[B - 1])), 6),
            "per_step_change": round(step, 4),
        }
        print(
            f"gi {gi} group {g} rows {len(rows)} gap_before={report[str(gi)]['closure_gap_before']:.3f} "
            f"gap_after={report[str(gi)]['closure_gap_after']:.5f} step={step:.3f}",
            flush=True,
        )

    # copy the rest of the router artifacts
    for f in ["phase_meta.json", "phase_norm.npz"]:
        shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
    for f in os.listdir(SRC):
        if f.startswith("phase_g") and f.endswith(".pt"):
            shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
    np.save(D + "/token_closed.npy", closed_token)
    json.dump(report, open(f"{DST}/closure_report.json", "w"), indent=1)
    print("saved", DST, "and", D + "/token_closed.npy")


if __name__ == "__main__":
    main()
