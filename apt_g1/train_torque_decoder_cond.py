"""TO40a: 条件化力矩解码器插值测试（主线 Rung 1 前置，2026-08-31）。

TO37b 已证：标量 v 输入跨速度泛化 FAIL（0.277 训 -> 0.678 测 MAE ~86% std）。
本脚本在**同一份 3 点速度网格**（0.277/0.435/0.678，全部审计 PASS）上比较：

  base_raw   MLP(sin,cos,v,phase_bit)      —— TO37b 原样复刻（对照）
  base_norm  同上但 v 归一化到 [0,1]        —— 排除「只是尺度问题」
  embed      v -> 学习嵌入(16) 拼接进输入
  film       v -> FiLM(每层 gamma/beta) 调制隐层

协议：留一速度 x3 折（{0.277,0.435}->0.678 上外推 / {0.277,0.678}->0.435
**插值(关键折)** / {0.435,0.678}->0.277 下外推）+ 全量拟合参照。每配置
3 seed 取均值。判定：插值折平均 MAE < 20% test-std => PASS（TO37b 基线 86%）。

用法（任意有 torch+numpy 的环境；数据/输出路径可控）：
  python apt_g1/train_torque_decoder_cond.py \
      --data-dir apt_g1/outputs/sync/to3637_sol --out-dir apt_g1/outputs/sync
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

SIGN = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
# 干净三点：seed=F11b(0.277)；to37_v0.08/v0.12 是 0.277 重复点，排除
FILES = [
    ("to37_seed.npz", None),
    ("to37_fast56_v0435.npz", None),
    ("to37_fast48b_v0678.npz", None),
]
ARCHS = ["base_raw", "base_norm", "embed", "film"]
FOLDS = [  # (train speeds, test speed, 标签)
    ((0.277, 0.435), 0.678, "extrap_up"),
    ((0.277, 0.678), 0.435, "interpolation"),
    ((0.435, 0.678), 0.277, "extrap_down"),
]


def phase_to_mjcf(ph, U):
    st = "left" if ph == "l" else "right"
    us = U * SIGN
    out = np.zeros((U.shape[0], 6))
    ot = "right" if st == "left" else "left"
    cols = [(st, "ankle"), (st, "knee"), (st, "hip"),
            (ot, "hip"), (ot, "knee"), (ot, "ankle")]
    for i, (s, p) in enumerate(cols):
        mj = {"left": 0, "right": 3}[s] + {"hip": 0, "knee": 1, "ankle": 2}[p]
        out[:, mj] = us[:, i]
    return out


def load_family(data_dir):
    X, Y, V = [], [], []
    vmap = {}
    for fn, _ in FILES:
        d = np.load(os.path.join(data_dir, fn), allow_pickle=True)
        if str(d.get("mode", "foot")) != "foot":
            continue
        N = int(d["knots"])
        v = round(float(d["v_avg"]), 3)
        for ph in ("left", "right"):
            short = ph[0]
            Up = np.array(d[f"U_{ph}"])
            Tp = float(d[f"T_{ph}"])
            Umid = 0.5 * (Up[:-1] + Up[1:])
            ts = list(np.linspace(0.0, Tp, N + 1))[:-1]
            ts_mid = list((np.arange(N) + 0.5) * Tp / N)
            for t, u in zip(ts + ts_mid, list(Up[:-1]) + list(Umid)):
                phi = 2.0 * np.pi * t / Tp
                X.append([np.sin(phi), np.cos(phi), v,
                          0.0 if short == "l" else 1.0])
                Y.append(phase_to_mjcf(short, u[None, :])[0])
                V.append(v)
        vmap[fn] = v
        print(f"[data] {fn}: v={v:.3f}")
    return (np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32),
            np.array(V, dtype=np.float32), vmap)


class CondNet(nn.Module):
    def __init__(self, mode: str, hidden: int = 64, vemb: int = 16):
        super().__init__()
        self.mode = mode
        self.vemb = vemb
        if mode in ("embed", "film"):
            self.venc = nn.Sequential(nn.Linear(1, vemb), nn.SiLU(),
                                      nn.Linear(vemb, vemb))
        trunk_in = 3 + (vemb if mode == "embed" else 0)
        self.t1 = nn.Linear(trunk_in, hidden)
        self.t2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 6)
        if mode == "film":
            self.f1 = nn.Linear(vemb, hidden)
            self.g1 = nn.Linear(vemb, hidden)
            self.f2 = nn.Linear(vemb, hidden)
            self.g2 = nn.Linear(vemb, hidden)

    def forward(self, sin, cos, v, pbit):
        if self.mode in ("embed", "film"):
            e = self.venc(v)
        if self.mode == "embed":
            h = torch.cat([sin, cos, pbit, e], dim=1)
            h = torch.nn.functional.silu(self.t1(h))
            h = torch.nn.functional.silu(self.t2(h))
        elif self.mode == "film":
            h = torch.nn.functional.silu(self.t1(torch.cat([sin, cos, pbit], 1)))
            h = h * (1.0 + self.g1(e)) + self.f1(e)
            h = torch.nn.functional.silu(self.t2(h))
            h = h * (1.0 + self.g2(e)) + self.f2(e)
        else:  # base_raw / base_norm: v 直接进输入
            h = torch.cat([sin, cos, v, pbit], dim=1)
            h = torch.nn.functional.silu(self.t1(h))
            h = torch.nn.functional.silu(self.t2(h))
        return self.head(h)


def train_one(mode, X, Y, vnorm, epochs, seed):
    torch.manual_seed(seed)
    net = CondNet(mode)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.L1Loss()
    sin = torch.tensor(X[:, 0:1], dtype=torch.float32)
    cos = torch.tensor(X[:, 1:2], dtype=torch.float32)
    pbit = torch.tensor(X[:, 3:4], dtype=torch.float32)
    v = torch.tensor((X[:, 2:3] - vnorm[0]) / (vnorm[1] - vnorm[0]),
                     dtype=torch.float32)
    yt = torch.tensor(Y)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(sin, cos, v, pbit), yt)
        loss.backward()
        opt.step()
    return net


def predict(net, X, vnorm):
    with torch.no_grad():
        sin = torch.tensor(X[:, 0:1], dtype=torch.float32)
        cos = torch.tensor(X[:, 1:2], dtype=torch.float32)
        pbit = torch.tensor(X[:, 3:4], dtype=torch.float32)
        v = torch.tensor((X[:, 2:3] - vnorm[0]) / (vnorm[1] - vnorm[0]),
                         dtype=torch.float32)
        return net(sin, cos, v, pbit).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="apt_g1/outputs/sync/to3637_sol")
    ap.add_argument("--out-dir", default="apt_g1/outputs/sync")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    X, Y, V, vmap = load_family(args.data_dir)
    speeds = sorted(set(V.tolist()))
    print(f"[to40a] 样本 {len(X)}，速度点 {speeds}，"
          f"tau 范围 [{Y.min():+.1f},{Y.max():+.1f}] N·m")
    ystd = Y.std(axis=0)
    vnorm = (min(speeds), max(speeds))

    results = {"speeds": speeds, "n_samples": len(X),
               "tau_std": ystd.tolist(), "folds": {}}
    for tr_v, te_v, tag in FOLDS:
        tr = np.isin(V, tr_v)
        te = V == te_v
        te_std = float(Y[te].std(axis=0).mean())
        entry = {}
        for arch in ARCHS:
            maes = []
            for seed in range(args.seeds):
                net = train_one(arch, X[tr], Y[tr], vnorm, args.epochs, seed)
                pred = predict(net, X[te], vnorm)
                maes.append(float(np.abs(pred - Y[te]).mean()))
            m = float(np.mean(maes))
            entry[arch] = {"mae_mean": round(m, 4),
                           "mae_seeds": [round(x, 4) for x in maes],
                           "pct_std": round(100 * m / te_std, 1)}
        results["folds"][tag] = {
            "train_speeds": list(tr_v), "test_speed": te_v,
            "n_test": int(te.sum()), "test_std_mean": round(te_std, 3),
            "archs": entry}
        print(f"[fold {tag}] train={tr_v} test={te_v} "
              f"(std={te_std:.2f} N·m):")
        for arch in ARCHS:
            e = entry[arch]
            print(f"    {arch:9s} MAE={e['mae_mean']:.3f} "
                  f"({e['pct_std']:.1f}% std)")

    # 全量拟合参照（最佳 arch）+ 保存解码器供 TO40-C 使用
    best = min(ARCHS, key=lambda a: results["folds"]["interpolation"]["archs"][a]["mae_mean"])
    results["best_arch"] = best
    net = train_one(best, X, Y, vnorm, args.epochs, 0)
    pred = predict(net, X, vnorm)
    results["mae_full_best"] = np.abs(pred - Y).mean(axis=0).round(4).tolist()
    torch.save(net.state_dict(), os.path.join(args.out_dir, "to40_cond_decoder.pt"))
    json.dump(results, open(os.path.join(args.out_dir, "to40_cond_results.json"), "w"),
              indent=1, ensure_ascii=False)
    ip = results["folds"]["interpolation"]["archs"]
    ok = ip[best]["pct_std"] < 20.0
    print(f"[to40a] 最佳 arch = {best}；全量 MAE = "
          f"{results['mae_full_best']}")
    print(f"[to40a] 插值判定（{best} 折 MAE < 20% std）: "
          f"{'PASS' if ok else 'FAIL'} —— TO37b 标量 v 基线 86%")
    print(f"[to40a] 已存 to40_cond_results.json + to40_cond_decoder.pt")


if __name__ == "__main__":
    main()
