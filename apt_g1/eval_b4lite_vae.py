"""D046: B4-lite VAE v1 预注册判据评测（J1–J4，只前向不训练）。

判据（预注册，2026-09-07）：
  J1  最终 dev 重建 MSE ≤ 1.5× train 重建 MSE（超出 = overfit，给曲线）
  J2  dir_head 8-bin dev 准确率 > 12.5%（随机线）且 speed_head 3-bin > 33%
      （D045 冒烟常数假设 acc=1.0 退化必须消失）；逐族分桶报告
  J3  全程零 NaN、exit 正常（本脚本负责指标有限性检查；exit 码由训练日志核）
  J4  T-fam 22 段重建 MSE 单独报告（不训练只前向，流形连续性基线数）

评测口径：确定性 mu 前向（非采样 z）；重建/准确率只在段内合法窗口帧上算
（与训练同一边界掩码：向后窗 10 帧）；v_bin 用 run1/vbin_meta.json 的训练
分位边界 + pca.npz 的训练 PCA 重放（dev/tfam 不重新拟合任何统计量）。
默认评测全部 vae_ep*.pt 快照（dev 曲线）+ 最末快照出 J1–J4 终判。

用法（服务器 .venv_isaac python）：
  python eval_b4lite_vae.py --run-dir .../vae_v1/run1 \
    --train-dir .../vae_v1 --dev-dir .../vae_v1/dev --tfam-dir .../vae_v1/tfam
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_token_vae_e39 import DirSpeedPhaseTokenVAE, build_windows  # noqa: E402

WINDOW = 10


def wrap_pi(a):
    return np.mod(a + np.pi, 2.0 * np.pi) - np.pi


def valid_mask(bounds: np.ndarray, n: int, window: int = WINDOW) -> np.ndarray:
    seg_start = np.zeros(n, dtype=np.int64)
    for s0, e0 in bounds:
        seg_start[s0:e0] = s0
    return (np.arange(n) - seg_start) >= (window - 1)


def conditioning(tok, mode, pca, edges):
    proj = (tok - pca["pmean"]) @ pca["V2"]
    phi = np.arctan2(proj[:, 1], proj[:, 0]).astype(np.float32)
    rate = np.zeros(len(tok), dtype=np.float32)
    dphi = wrap_pi(np.diff(phi))
    rate[1:] = np.abs(dphi)
    rate[0] = rate[1]
    vb = np.clip(np.digitize(rate, edges), 0, 2).astype(np.int64)
    phase2 = np.stack([np.sin(phi), np.cos(phi)], axis=1).astype(np.float32)
    return phase2, vb


def load_ds(d):
    return (np.load(os.path.join(d, "token.npy")).astype(np.float32),
            np.load(os.path.join(d, "mode.npy")).astype(np.int64),
            np.load(os.path.join(d, "angle_bin.npy")).astype(np.int64),
            np.load(os.path.join(d, "segment_bounds.npy")).astype(np.int64))


def eval_dataset(model, heads, run_dir, ds_dir, pca, edges, dev="cuda"):
    tok, mode, ab, bounds = load_ds(ds_dir)
    n = len(tok)
    phase2, vb = conditioning(tok, mode, pca, edges)
    x = build_windows(tok, WINDOW)
    vm = valid_mask(bounds, n)
    idx = np.where(vm)[0]
    FAMILIES = json.load(open(os.path.join(ds_dir, "build_meta.json")))["family_order"]
    with torch.no_grad():
        xt = torch.from_numpy(x[idx]).to(dev)
        yt = torch.from_numpy(tok[idx]).to(dev)
        pt = torch.from_numpy(phase2[idx]).to(dev)
        vbt = torch.from_numpy(vb[idx]).to(dev)
        dbt = torch.from_numpy(ab[idx]).to(dev)
        mu, _ = model.encode(xt)
        recon = model.decode(mu, pt, vbt, dbt)
        per_frame = ((recon - yt) ** 2).mean(dim=1).cpu().numpy()
        dir_ok = (heads["dir_head"](mu).argmax(1) == dbt).cpu().numpy()
        spd_ok = (heads["speed_head"](mu).argmax(1) == vbt).cpu().numpy()
    mse = float(per_frame.mean())
    dir_acc = float(dir_ok.mean())
    spd_acc = float(spd_ok.mean())
    modes_i = mode[idx]
    per_fam = {}
    for m in np.unique(modes_i):
        sel = modes_i == m
        per_fam[FAMILIES[int(m)]] = {
            "n": int(sel.sum()),
            "mse": float(per_frame[sel].mean()),
            "dir_acc": float(dir_ok[sel].mean()),
            "spd_acc": float(spd_ok[sel].mean()),
        }
    out = {"dir": ds_dir, "n_frames": n, "n_valid": int(vm.sum()),
           "recon_mse": mse, "dir_acc": dir_acc, "spd_acc": spd_acc,
           "per_family": per_fam,
           "finite": bool(np.isfinite(mse) and np.isfinite(dir_acc) and np.isfinite(spd_acc)),
           "bin_hist_db": {int(b): int(c) for b, c in zip(*np.unique(ab[idx], return_counts=True))},
           "bin_hist_vb": {int(b): int(c) for b, c in zip(*np.unique(vb[idx], return_counts=True))}}
    return out


def load_ckpt(run_dir, vae_path, heads_path, dev="cuda"):
    model = DirSpeedPhaseTokenVAE().to(dev)
    model.load_state_dict(torch.load(vae_path, map_location=dev))
    model.eval()
    hd = torch.load(heads_path, map_location=dev)
    heads = {
        "dir_head": (torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(),
                                         torch.nn.Linear(64, 8))).to(dev),
        "speed_head": (torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(),
                                           torch.nn.Linear(64, 3))).to(dev),
    }
    heads["dir_head"].load_state_dict(hd["dir_head"])
    heads["speed_head"].load_state_dict(hd["speed_head"])
    for h in heads.values():
        h.eval()
    return model, heads


def main():
    ap = argparse.ArgumentParser(description="D046: B4-lite VAE v1 J1-J4 评测")
    base = os.path.expanduser("~/ros2_data/apt_g1/data/ds_bones/g1_b4lite/vae_v1")
    ap.add_argument("--run-dir", default=os.path.join(base, "run1"))
    ap.add_argument("--train-dir", default=base)
    ap.add_argument("--dev-dir", default=os.path.join(base, "dev"))
    ap.add_argument("--tfam-dir", default=os.path.join(base, "tfam"))
    ap.add_argument("--out", default=None, help="默认 <run-dir>/metrics_d046.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    pca = dict(np.load(os.path.join(args.run_dir, "pca.npz")))
    edges = json.load(open(os.path.join(args.run_dir, "vbin_meta.json")))["edges"]

    # ---- 快照曲线（dev/train 重建 + dev head acc）
    snaps = sorted(glob.glob(os.path.join(args.run_dir, "vae_ep*.pt")))
    curves = []
    for sp in snaps:
        ep = int(re.search(r"vae_ep(\d+)\.pt", sp).group(1))
        hp = os.path.join(args.run_dir, f"heads_ep{ep:03d}.pt")
        model, heads = load_ckpt(args.run_dir, sp, hp, dev)
        tr = eval_dataset(model, heads, args.run_dir, args.train_dir, pca, edges, dev)
        dv = eval_dataset(model, heads, args.run_dir, args.dev_dir, pca, edges, dev)
        curves.append({"epoch": ep, "train_mse": tr["recon_mse"],
                       "dev_mse": dv["recon_mse"],
                       "dir_acc_dev": dv["dir_acc"], "spd_acc_dev": dv["spd_acc"]})
        print(f"ep {ep:3d} train_mse={tr['recon_mse']:.6f} dev_mse={dv['recon_mse']:.6f} "
              f"dir_acc_dev={dv['dir_acc']:.3f} spd_acc_dev={dv['spd_acc']:.3f}", flush=True)
    curves.sort(key=lambda c: c["epoch"])

    # ---- 终判：最末快照（无快照则退回 vae.pt + heads.pt）
    if snaps:
        last_ep = curves[-1]["epoch"]
        vae_path = os.path.join(args.run_dir, f"vae_ep{last_ep:03d}.pt")
        heads_path = os.path.join(args.run_dir, f"heads_ep{last_ep:03d}.pt")
        final_id = f"vae_ep{last_ep:03d}.pt"
    else:
        vae_path = os.path.join(args.run_dir, "vae.pt")
        heads_path = os.path.join(args.run_dir, "heads.pt")
        final_id = "vae.pt"
    model, heads = load_ckpt(args.run_dir, vae_path, heads_path, dev)
    tr = eval_dataset(model, heads, args.run_dir, args.train_dir, pca, edges, dev)
    dv = eval_dataset(model, heads, args.run_dir, args.dev_dir, pca, edges, dev)
    tf = eval_dataset(model, heads, args.run_dir, args.tfam_dir, pca, edges, dev)

    J1 = {"train_mse": tr["recon_mse"], "dev_mse": dv["recon_mse"],
          "ratio": dv["recon_mse"] / tr["recon_mse"],
          "threshold": 1.5, "pass": bool(dv["recon_mse"] <= 1.5 * tr["recon_mse"])}
    J2 = {"dir_acc_dev": dv["dir_acc"], "dir_random_line": 0.125,
          "spd_acc_dev": dv["spd_acc"], "spd_random_line": 1.0 / 3.0,
          "dir_per_family": dv["per_family"], "spd_note": "v_bin 由训练分位重放",
          "pass": bool(dv["dir_acc"] > 0.125 and dv["spd_acc"] > 1.0 / 3.0)}
    J3 = {"all_finite": bool(tr["finite"] and dv["finite"] and tf["finite"]
                             and all(np.isfinite([c["train_mse"], c["dev_mse"],
                                                  c["dir_acc_dev"], c["spd_acc_dev"]]).all()
                                    for c in curves)),
          "note": "exit 码以训练日志为准，本脚本核指标有限性"}
    J4 = {"tfam_mse": tf["recon_mse"], "n_frames": tf["n_valid"],
          "per_family": tf["per_family"], "note": "流形连续性基线数，不设阈值"}

    out = {"final_snapshot": final_id, "J1": J1, "J2": J2, "J3": J3, "J4": J4,
           "curves": curves,
           "final_full": {"train": tr, "dev": dv, "tfam": tf}}
    path = args.out or os.path.join(args.run_dir, "metrics_d046.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("=== J1", "PASS" if J1["pass"] else "FAIL",
          f"dev/train = {J1['dev_mse']:.6f}/{J1['train_mse']:.6f} = {J1['ratio']:.3f}")
    print("=== J2", "PASS" if J2["pass"] else "FAIL",
          f"dir {dv['dir_acc']:.3f} (>0.125), spd {dv['spd_acc']:.3f} (>0.333)")
    print("=== J3", "PASS" if J3["all_finite"] else "FAIL")
    print("=== J4 tfam_mse =", f"{J4['tfam_mse']:.6f}")
    print("saved", path)


if __name__ == "__main__":
    main()
