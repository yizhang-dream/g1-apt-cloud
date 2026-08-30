"""TO37b: torque-decoder learnability on the TO36 dircol solution family.

Data: apt_g1/outputs/to37_v*.npz (hybrid foot solutions, multi-speed).
Each solution contributes 2 phases x (N+1 knots + N midpoints) samples:
features (sin phi, cos phi, v, phase_bit) -> 6D actuator torque in MJCF
sign convention [L_hip, L_knee, L_ankle, R_hip, R_knee, R_ankle].

Phase-coordinate -> MJCF sign map (B-gate FK-verified, see to36 tracker):
  phase order [stance_ankle, stance_knee, stance_hip, other_hip,
               other_knee, other_ankle], sign [-1,-1,-1,+1,+1,-1].

Baseline: TO10 kinematic-gait tau_clean MAE hip 0.61 / knee 0.14 /
ankle 0.03 N.m.  Run on the server under .venv_mjlab (torch).
"""
import glob
import json
import sys

import numpy as np
import torch
import torch.nn as nn

OUT = "apt_g1/outputs/to37_decoder.json"
SIGN = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0])


def phase_to_mjcf(ph, U):
    """Phase-ordered torques -> MJCF 6-joint order [Lh,Lk,La,Rh,Rk,Ra]."""
    st = "left" if ph == "l" else "right"
    us = U * SIGN  # in phase order, MJCF sign
    out = np.zeros((U.shape[0], 6))
    cols = {f"{s}_{p}": i for i, (s, p) in enumerate(
        ([(st, "ankle"), (st, "knee"), (st, "hip"),
          ("right" if st == "left" else "left", "hip"),
          ("right" if st == "left" else "left", "knee"),
          ("right" if st == "left" else "left", "ankle")]))}
    for name, i in cols.items():
        side, part = name.split("_")
        mj = {"left": 0, "right": 3}[side] + \
            {"hip": 0, "knee": 1, "ankle": 2}[part]
        out[:, mj] = us[:, i]
    return out


def load_family():
    X, Y, meta = [], [], []
    for f in sorted(glob.glob("apt_g1/outputs/to37_*.npz")):
        if "seed" in f:
            continue
        d = np.load(f, allow_pickle=True)
        if str(d.get("mode", "foot")) != "foot":
            continue
        N = int(d["knots"])
        v = float(d["v_avg"])
        n_s = 0
        for ph in ("left", "right"):
            short = ph[0]
            Xp = np.array(d[f"X_{ph}"])
            XMp = np.array(d[f"XM_{ph}"])
            Up = np.array(d[f"U_{ph}"])
            Tp = float(d[f"T_{ph}"])
            Umid = 0.5 * (Up[:-1] + Up[1:])
            ts = list(np.linspace(0.0, Tp, N + 1))[:-1]  # knots 0..N-1
            ts_mid = list((np.arange(N) + 0.5) * Tp / N)
            for t, u in zip(ts + ts_mid,
                            list(Up[:-1]) + list(Umid)):
                phi = 2.0 * np.pi * t / Tp
                X.append([np.sin(phi), np.cos(phi), v,
                          0.0 if short == "l" else 1.0])
                Y.append(phase_to_mjcf(short, u[None, :])[0])
            n_s += N + N
            del Xp, XMp
        meta.append({"file": f, "v": v, "samples": n_s,
                     "audit_drift_L": float(d.get("energy_drift_left", np.nan)),
                     "audit_drift_R": float(d.get("energy_drift_right", np.nan))})
        print(f"[data] {f}: v={v:.3f} samples={n_s} "
              f"drift L/R={meta[-1]['audit_drift_L']:.2f}/"
              f"{meta[-1]['audit_drift_R']:.2f} J")
    return (np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32),
            meta)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    X, Y, meta = load_family()
    if len(X) < 100:
        raise SystemExit(f"[to37b] 样本不足 {len(X)}——先跑完 TO37a 解族")
    print(f"[to37b] 样本 {len(X)}，速度点 {len(meta)}，"
          f"tau 范围 [{Y.min():+.1f},{Y.max():+.1f}] N·m")

    # 标签方差（「预测均值」基线）：MAE 必须显著小于 std 才叫可学
    ystd = Y.std(axis=0)
    print(f"[to37b] 各关节 tau std = {np.round(ystd, 2).tolist()} "
          f"(Lh,Lk,La,Rh,Rk,Ra)")

    # 划分：留一速度泛化（holdout = 最大速度点——跨速度泛化才是
    # 「速度条件化」的核心验证；全量拟合另报）
    v_test = max(m["v"] for m in meta)
    te = X[:, 2] == v_test
    tr = ~te

    def train(Xtr, Ytr, epochs=4000, hidden=64):
        net = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(),
                            nn.Linear(hidden, hidden), nn.SiLU(),
                            nn.Linear(hidden, 6))
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        lossf = nn.L1Loss()
        xt = torch.tensor(Xtr)
        yt = torch.tensor(Ytr)
        for ep in range(epochs):
            opt.zero_grad()
            loss = lossf(net(xt), yt)
            loss.backward()
            opt.step()
        return net

    net_full = train(X, Y)
    with torch.no_grad():
        pred = net_full(torch.tensor(X)).numpy()
    mae_full = np.abs(pred - Y).mean(axis=0)

    out = {"n_samples": len(X), "tau_std": ystd.tolist(),
           "speeds": [m["v"] for m in meta], "meta": meta,
           "mae_full": mae_full.tolist()}
    if tr.sum() > 200 and te.sum() > 50:
        net_tr = train(X[tr], Y[tr])
        with torch.no_grad():
            pred_te = net_tr(torch.tensor(X[te])).numpy()
        mae_te = np.abs(pred_te - Y[te]).mean(axis=0)
        out["mae_holdout_speed"] = mae_te.tolist()
        out["holdout_v"] = float(v_test)
        print(f"[to37b] 留出速度 v={v_test:.3f}（{int(te.sum())} 样本）"
              f"MAE = {np.round(mae_te, 3).tolist()}")
    print(f"[to37b] 全量 MAE = {np.round(mae_full, 3).tolist()} N·m "
          f"(Lh,Lk,La,Rh,Rk,Ra)——对照 TO10 hip 0.61/knee 0.14/ankle 0.03")
    ok = bool((mae_full < 0.1 * ystd + 0.05).all())
    print(f"[to37b] 可学性判定（MAE < 0.1·std+0.05）: "
          f"{'PASS' if ok else 'FAIL'}")
    json.dump(out, open(OUT, "w"), indent=2)
    torch.save(net_full.state_dict(),
               "apt_g1/outputs/to37_decoder.pt")
    print(f"[to37b] 已存 {OUT} + to37_decoder.pt")


if __name__ == "__main__":
    main()
