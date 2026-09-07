"""D046b 交叉 z 探针（单轴交叉转向矩阵）——判别 v2.1 VAE 条件通路「真琴键 vs 贴纸」。

背景：v2.1 探针（probe_vae_v2.py）显示 locomotion 窗 z + 条件 IDLE 解码不向 IDLE
质心转向（0.004）；而此前「条件=WALK 输出像 WALK」的高对角可能被 z 驱动混杂。
本探针把 z 驱动与条件驱动分开：**每次只交叉一个条件轴**，其余条件轴保持窗口
自身真值——z 从条件值 i 的训练窗取（确定性 mu），被测轴条件置 j 解码，输出对
train 前向按真值条件分组的 token 质心做最近质心判定 → 转向矩阵
M[i][j] = P(命中 j)。若输出跟条件走，off-diag 高；若跟 z 走，对角高。

预注册判读量（owner 指令 2026-09-07）：
  cond_follow = 逐轴 off-diagonal (i≠j) 的 P(命中 j) 均值
  z_follow    = 逐轴 off-diagonal (i≠j) 的 P(命中 i) 均值（=对角均值，逐行等权）
  chance      = 1/k
  判定（逐轴，按序）：cond_follow >= 2*chance -> 条件通路可用（贴纸问题不成立）；
  elif z_follow >= 3*cond_follow -> z 主导（架构问题证实）；else -> 部分可控
  （逐格报告断裂 (i,j) 对，重点 IDLE 行/列）。
样本门槛：矩阵行（z 组）>= --min-win（默认 150）训练窗（respect segment_bounds
+ keep mask）；低于门槛的条件值不入阵、单独登记。

J9 复测：mode 矩阵 IDLE 行（z_IDLE + 条件 j -> 能否被命令离开 hub）与 IDLE 列
（z_i + 条件 IDLE -> 能否被命令进入 hub）直接回答「hub 能否被命令」；列与
v2.1 探针 dev-z 版 0.004 直接对照。

诊断：IDLE z vs locomotion z 分布（train mu 全体按 locomotion 子集 PCA 的前
8 主成分投影：逐 PC 标准化均值位移 |Δmean|/std_loco 与 std 比）——排除
「IDLE z 本身无信息/塌缩」的平凡解释。

注意：mode 轴按训练窗实况取值——v2.1 中 SLOW_WALK(1)/INJURED_WALK(6) 为 T-fam
零训练窗，指令所列 9 座中 INJURED 不满足「z 从训练窗口取」，实做 8×8（剩余
mode 全部 >=150 窗门槛运行时强制校验并登记计数）。

Usage (CVGL det 容器或 lab-ts venv_isaac，纯 torch 前向，分钟级):
  python probe_vae_cross.py \
    --run-dir /home/cvgluser/ros2_data/g1_b4lite_probe/vae_v21/run1 \
    --inputs-dir /home/cvgluser/ros2_data/g1_b4lite_probe/vae_inputs_v21
产出 <run-dir>/../probe_cross/{metrics_d046b.json, summary_d046b.txt}
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from train_token_vae_e39_v2 import (DirSpeedPhaseTokenVAE, build_windows_bounded,
                                    per_frame_labels, phi_rate_from)

UL_NAMES = {0: "none", 1: "sym", 2: "asym"}


def load_split(d: str, use_ul: bool):
    tok = np.load(os.path.join(d, "token.npy")).astype(np.float32)
    bnd = np.load(os.path.join(d, "segment_bounds.npy"))
    mode = np.load(os.path.join(d, "mode_id.npy")).astype(np.int64)
    ang = np.load(os.path.join(d, "angle_bin.npy")).astype(np.int64)
    ul = (np.load(os.path.join(d, "ul.npy")).astype(np.int64)
          if use_ul else np.zeros(len(tok), dtype=np.int64))
    return tok, bnd, mode, ang, ul


def fmt_matrix(M: np.ndarray, row_names, col_names) -> str:
    w = 7
    head = "cond_j-> ".ljust(9) + "".join(str(c)[:6].rjust(w) for c in col_names)
    lines = [head]
    for r, row in zip(row_names, M):
        lines.append(str(r)[:8].ljust(9) + "".join(f"{v:7.3f}" for v in row))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    base = f"{home}/ros2_data/apt_g1/data/ds_bones/g1_b4lite"
    ap.add_argument("--run-dir", default=f"{base}/vae_v21/run1")
    ap.add_argument("--inputs-dir", default=f"{base}/vae_inputs_v21")
    ap.add_argument("--snapshot", default="vae_ep100.pt",
                    help="与 probe_vae_v2 v2.1 评测同快照，保持可比")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--min-win", type=int, default=150)
    ap.add_argument("--n-pc", type=int, default=8)
    ap.add_argument("--out-dir", default=None,
                    help="缺省 <run-dir>/../probe_cross")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={dev}", flush=True)

    # ---- 模型加载（与 probe_vae_v2.py 同口径） ----
    meta = json.load(open(os.path.join(args.run_dir, "meta.json")))
    use_ul = bool(meta["use_ul"])
    model = DirSpeedPhaseTokenVAE(window=meta["window"], latent_dim=meta["latent_dim"],
                                  hidden_dim=meta["hidden"], n_vbins=3, n_dbins=8,
                                  n_modes=int(meta["n_modes"]), use_ul=use_ul).to(dev)
    sd_path = os.path.join(args.run_dir, args.snapshot)
    model.load_state_dict(torch.load(sd_path, map_location=dev))
    model.eval()
    build_meta = json.load(open(os.path.join(args.inputs_dir, "build_meta.json")))
    mode_names = {t["embed_idx"]: t["mode_name"] for t in build_meta["mode_table"]}
    walk_idx = meta["walk_embed_idx"]

    # ---- train 窗口 + 逐窗条件（复制 probe_vae_v2.py 的 train 侧口径） ----
    tok, bnd, mode, ang, ul = load_split(args.inputs_dir, use_ul)
    pca = np.load(os.path.join(args.run_dir, "pca.npz"))
    pmean, V2 = pca["pmean"], pca["V2"]
    base_mode = ((mode == walk_idx).astype(int) * 2 if walk_idx is not None
                 else np.asarray(mode))
    rate, phi = phi_rate_from(tok, pmean, V2)
    edges = np.quantile(rate[base_mode == 2], [1 / 3, 2 / 3])
    vb = np.clip(np.digitize(rate, edges), 0, 2).astype(np.int64)
    phase2 = np.stack([np.sin(phi), np.cos(phi)], 1).astype(np.float32)

    W = meta["window"]
    x, _y = build_windows_bounded(tok, bnd, W)
    mb = per_frame_labels(mode, bnd, W)
    ub = per_frame_labels(ul, bnd, W)
    pb = per_frame_labels(phase2, bnd, W)
    vbw = per_frame_labels(vb, bnd, W)
    dbw = per_frame_labels(ang, bnd, W)
    km = os.path.join(args.inputs_dir, "window_keep_mask.npy")
    if os.path.isfile(km):
        m = np.load(km)
        x, pb, vbw, dbw, mb, ub = (v[m] for v in (x, pb, vbw, dbw, mb, ub))
    N = len(x)
    print(f"[data] train windows N={N}", flush=True)

    def batches(*arrays):
        for i in range(0, N, args.batch):
            yield [torch.from_numpy(np.ascontiguousarray(a[i:i + args.batch])).to(dev)
                   for a in arrays]

    # ---- 全窗 mu（z 源）+ 自条件前向 rec（质心源） ----
    mus, recs = [], []
    with torch.no_grad():
        for bx, bp, bv, bd, bm, bu in batches(x, pb, vbw, dbw, mb, ub):
            mu, _ = model.encode(bx)
            mus.append(mu.cpu().numpy())
            recs.append(model.decode(mu, bp, bv, bd, bm, bu).cpu().numpy())
    mus = np.concatenate(mus)
    recs = np.concatenate(recs)
    assert np.isfinite(mus).all() and np.isfinite(recs).all(), "NaN/Inf in forward"
    mu_t = torch.from_numpy(mus).float().to(dev)

    # ---- 轴定义：mode / ul / vb / db（各轴独立矩阵） ----
    def axis_values(labels: np.ndarray) -> tuple[list[int], dict[int, str]]:
        cnt = {int(v): int((labels == v).sum()) for v in np.unique(labels)}
        keep = sorted(v for v, c in cnt.items() if c >= args.min_win)
        dropped = {v: c for v, c in cnt.items() if c < args.min_win}
        return keep, dropped

    axes = {
        "mode": (mb, {v: mode_names.get(v, str(v)) for v in range(int(meta["n_modes"]))}),
        "ul": (ub, UL_NAMES),
        "vb": (vbw, {0: "vb0", 1: "vb1", 2: "vb2"}),
        "db": (dbw, {v: f"db{v}" for v in range(8)}),
    }
    results: dict[str, dict] = {}
    for axis, (labels, names) in axes.items():
        values, dropped = axis_values(labels)
        K = len(values)
        row_names = [names.get(v, str(v)) for v in values]
        counts = [int((labels == v).sum()) for v in values]
        cents = np.stack([recs[labels == v].mean(0) for v in values])
        cent_t = torch.from_numpy(cents).float().to(dev)
        M = np.zeros((K, K), dtype=np.float64)
        top = np.zeros((K, K), dtype=np.int64)      # 每格最近质心 argmax（断裂归因用）
        top_p = np.zeros((K, K), dtype=np.float64)  # 每格最近质心概率
        with torch.no_grad():
            for i, vi in enumerate(values):
                idx = np.where(labels == vi)[0]
                mu_i = mu_t[idx]
                ph_i = torch.from_numpy(pb[idx]).float().to(dev)
                cv = torch.from_numpy(vbw[idx]).to(dev)
                cd = torch.from_numpy(dbw[idx]).to(dev)
                cm = torch.from_numpy(mb[idx]).to(dev)
                cu = torch.from_numpy(ub[idx]).to(dev)
                for j, vj in enumerate(values):
                    if axis == "vb":
                        out = model.decode(mu_i, ph_i,
                                           torch.full_like(cv, vj), cd, cm, cu)
                    elif axis == "db":
                        out = model.decode(mu_i, ph_i, cv,
                                           torch.full_like(cd, vj), cm, cu)
                    elif axis == "mode":
                        out = model.decode(mu_i, ph_i, cv, cd,
                                           torch.full_like(cm, vj), cu)
                    elif axis == "ul":
                        out = model.decode(mu_i, ph_i, cv, cd, cm,
                                           torch.full_like(cu, vj))
                    else:
                        raise ValueError(axis)
                    d = torch.cdist(out, cent_t)
                    pred = d.argmin(1)
                    M[i, j] = float((pred == j).float().mean())
                    counts_ij = torch.bincount(pred, minlength=K).float()
                    top[i, j] = int(counts_ij.argmax())
                    top_p[i, j] = float(counts_ij.max() / counts_ij.sum())
        off = ~np.eye(K, dtype=bool)
        cond_follow = float(M[off].mean())
        z_follow = float(np.diag(M).mean())
        chance = 1.0 / K
        if cond_follow >= 2 * chance:
            verdict = "cond_available(条件通路可用,贴纸问题不成立)"
        elif z_follow >= 3 * cond_follow:
            verdict = "z_dominant(z 主导,架构问题证实)"
        else:
            verdict = "partial(部分可控)"
        # 断裂对归因：off-diag 中条件命中所指质心不是输出 argmax 的格
        broken = [[row_names[i], row_names[j], round(float(M[i, j]), 4),
                   names.get(int(top[i, j]), str(int(top[i, j]))),
                   round(float(top_p[i, j]), 4)]
                  for i in range(K) for j in range(K)
                  if i != j and int(top[i, j]) != j]
        results[axis] = {
            "values": values, "row_names": row_names, "row_train_windows": counts,
            "below_min_win_dropped": {names.get(v, str(v)): c for v, c in dropped.items()},
            "matrix_hit_j": M.tolist(),
            "matrix_argmax_target": [[names.get(int(t), str(int(t))) for t in row]
                                     for row in top],
            "matrix_argmax_prob": top_p.tolist(),
            "cond_follow": round(cond_follow, 4),
            "z_follow": round(z_follow, 4),
            "chance": round(chance, 4),
            "z_over_cond_ratio": round(z_follow / max(cond_follow, 1e-9), 2),
            "cond_over_chance": round(cond_follow / chance, 2),
            "verdict": verdict,
            "offdiag_broken_cells_top1_not_j": broken,
            "diag_mean": round(float(np.diag(M).mean()), 4),
        }
        print(f"\n[axis={axis}] k={K} chance={chance:.3f} "
              f"cond_follow={cond_follow:.4f} z_follow={z_follow:.4f} -> {verdict}",
              flush=True)
        print(fmt_matrix(M, row_names, row_names), flush=True)

    # ---- J9 复测：IDLE 行/列（mode 矩阵） ----
    j9 = None
    mres = results["mode"]
    if 0 in mres["values"]:
        r = mres["values"].index(0)
        Mm = np.asarray(mres["matrix_hit_j"])
        rn = mres["row_names"]
        K = len(rn)
        col_off = [float(Mm[i, r]) for i in range(K) if i != r]
        row_off = [float(Mm[r, j]) for j in range(K) if j != r]
        j9 = {
            "idle_embed_idx": 0,
            "idle_row_P_hit_j": {rn[j]: round(float(Mm[r, j]), 4)
                                 for j in range(K)},          # 命令离开 hub
            "idle_row_offdiag_mean": round(float(np.mean(row_off)), 4),
            "idle_col_P_hit_IDLE": {rn[i]: round(float(Mm[i, r]), 4)
                                    for i in range(K)},        # 命令进入 hub
            "idle_col_offdiag_mean": round(float(np.mean(col_off)), 4),
            "reference_probe_v21_devz_idle_hit": 0.004,
            "chance": mres["chance"],
            "note": "行=z_IDLE 条件 j（能否被命令离开）；列=z_i 条件 IDLE"
                    "（能否被命令进入；与 v2.1 dev-z 版 0.004 对照）",
        }
        print(f"\n[J9] IDLE 行(离开) off-diag mean={j9['idle_row_offdiag_mean']}; "
              f"IDLE 列(进入) off-diag mean={j9['idle_col_offdiag_mean']}", flush=True)

    # ---- 诊断：IDLE z vs locomotion z 分布 ----
    idle_mask = mb == 0
    z_diag = None
    if idle_mask.any() and (~idle_mask).any():
        mu_loco = mus[~idle_mask]
        mu_idle = mus[idle_mask]
        mu_c = mu_loco - mu_loco.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(mu_c, full_matrices=False)
        Vp = Vt[:args.n_pc].T  # (16, n_pc)
        pl, pi_ = mu_c @ Vp, (mu_idle - mu_loco.mean(0, keepdims=True)) @ Vp
        sd_l = pl.std(0)
        sd_i = pi_.std(0)
        shift = np.abs(pi_.mean(0) - pl.mean(0)) / np.maximum(sd_l, 1e-9)
        z_diag = {
            "n_idle": int(idle_mask.sum()), "n_loco": int((~idle_mask).sum()),
            "pca_basis": "locomotion 子集 mu（train 全 locomotion 窗）",
            "n_pc": args.n_pc,
            "per_pc_std_mean_shift": [round(float(s), 3) for s in shift],
            "per_pc_std_loco": [round(float(s), 4) for s in sd_l],
            "per_pc_std_idle": [round(float(s), 4) for s in sd_i],
            "std_ratio_idle_over_loco_mean": round(float(sd_i.mean() / sd_l.mean()), 4),
            "mean_abs_shift_top8": round(float(shift.mean()), 3),
            "interpretation_hint": "shift>>1 且 std_ratio 不塌缩 -> IDLE z 有信息"
                                   "（非无信息/塌缩）；shift~0 或 std_ratio~0 -> "
                                   "IDLE z 无信息，hub 断裂属平凡解释",
        }
        print(f"\n[zdiag] mean|shift| top{args.n_pc}={z_diag['mean_abs_shift_top8']}; "
              f"std_ratio(idle/loco)={z_diag['std_ratio_idle_over_loco_mean']}",
              flush=True)

    out_dir = args.out_dir or os.path.join(args.run_dir, "..", "probe_cross")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    overall = {a: {"cond_follow": results[a]["cond_follow"],
                   "z_follow": results[a]["z_follow"],
                   "chance": results[a]["chance"],
                   "verdict": results[a]["verdict"]} for a in results}
    out = {
        "_meta": {"script": "apt_g1/probe_vae_cross.py", "experiment": "D046b",
                  "snapshot": sd_path, "device": str(dev),
                  "train_windows": int(N), "min_win": args.min_win,
                  "design": "单轴交叉：z 取条件值 i 的训练窗 mu，被测轴置 j，"
                            "其余轴保持窗口真值；质心=train 自条件前向按真值分组均值",
                  "criteria": "预注册：cond_follow>=2*chance->条件通路可用；"
                              "elif z_follow>=3*cond_follow->z 主导；else 部分可控",
                  "mode_axis_note": "SLOW_WALK(1)/INJURED_WALK(6) 为 T-fam 零训练窗，"
                                    "mode 轴按 >=150 窗实况取值（8 座）"},
        "axes": results,
        "j9_hub": j9,
        "z_diag_idle_vs_loco": z_diag,
        "verdict_overall": overall,
    }
    jp = os.path.join(out_dir, "metrics_d046b.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    sp = os.path.join(out_dir, "summary_d046b.txt")
    with open(sp, "w", encoding="utf-8") as f:
        f.write(f"D046b 交叉 z 探针 summary\nsnapshot={sd_path}\nN={N}\n")
        for a, r in results.items():
            f.write(f"\n[axis={a}] k={len(r['values'])} chance={r['chance']} "
                    f"cond_follow={r['cond_follow']} z_follow={r['z_follow']} "
                    f"z/c={r['z_over_cond_ratio']} cond/chance={r['cond_over_chance']}"
                    f"\nverdict: {r['verdict']}\n")
            f.write(fmt_matrix(np.asarray(r["matrix_hit_j"]),
                               r["row_names"], r["row_names"]) + "\n")
            f.write("argmax target (per cell):\n")
            for rn_, trow, prow in zip(r["row_names"],
                                       r["matrix_argmax_target"],
                                       r["matrix_argmax_prob"]):
                cells = ", ".join(f"j={cn_}->{t_}({p_:.2f})"
                                  for cn_, t_, p_ in zip(r["row_names"], trow, prow))
                f.write(f"  z={rn_:<14s} {cells}\n")
        if j9:
            f.write(f"\nJ9 hub: 离开={j9['idle_row_offdiag_mean']} "
                    f"进入={j9['idle_col_offdiag_mean']} "
                    f"(v2.1 dev-z 参考 0.004, chance={j9['chance']})\n")
        if z_diag:
            f.write(f"zdiag: mean|shift|={z_diag['mean_abs_shift_top8']} "
                    f"std_ratio={z_diag['std_ratio_idle_over_loco_mean']}\n")
    print(f"\n[write] {jp}\n[write] {sp}", flush=True)
    print(json.dumps(overall, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
