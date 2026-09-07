"""D046 v2/v2.1 评测探针（owner 判据更正版，2026-09-07；v2.1 适配 IDLE 入集）。

判据（更正后生效）：
  J1  dev 重建 <=1.5x train；另做 v1 同款域偏移定性（tfam(T-fam 留出 mode) 重建
      vs dev/train 相对水平 -> 区分记忆 vs 域偏移）。
  J2' 条件通路可控性探针（替代作废的原 J2 head 阈值）：dev mu(>=200 窗) 固定，
      遍历条件 bin 解码，与 train 前向按条件分组的 token 质心做最近质心判定。
      PASS: db(8) 每条件命中 >=25%（2x 随机）、vb(3) >=66%、mode(已训练值)
      每条件 >=2x 随机。head acc 降级为描述性统计。
  J5  生成侧 mode 最近质心混淆矩阵：任意两 in-subset mode 不得互相合并（对角
      占优：每行 argmax == 自身条件）。
  J6' UL 轴同法：v2.1 起站立段回归（IDLE+UL），sym 池非空则 3 条件矩阵可测
      （sym 分支=本次新增可测项）；另 asym/sym vs none 二分区分率。
  J7  全程零 NaN/Inf。
  J8  留出 mode（T-fam: SLOW_WALK/INJURED_WALK）重建单列（流形连续性基线）。
  J9  （v2.1 新增）IDLE 枢纽指标：IDLE 重建 MSE（train/dev 单列）+ mode 扫描
      IDLE 对角命中——hub 状态重建质量 = 将来 A->IDLE->B 切换链的底座指标，
      只报数不下结论。

Usage (server, venv_isaac):
  python probe_vae_v2.py --run-dir data/ds_bones/g1_b4lite/vae_v21/run1 \
      --inputs-dir data/ds_bones/g1_b4lite/vae_inputs_v21 \
      --snapshot vae_ep100.pt
产出 <run-dir>/metrics_d046v21.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from train_token_vae_e39_v2 import (DirSpeedPhaseTokenVAE, build_windows_bounded,
                                    per_frame_labels, phi_rate_from)

HOME = os.path.expanduser("~")
BASE = f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite"
V1_RUN = f"{BASE}/vae_v1/run1"
V2_RUN = f"{BASE}/vae_v2/run1"


def load_split(d: str, use_ul: bool):
    tok = np.load(os.path.join(d, "token.npy")).astype(np.float32)
    bnd = np.load(os.path.join(d, "segment_bounds.npy"))
    mode = np.load(os.path.join(d, "mode_id.npy")).astype(np.int64)
    ang = np.load(os.path.join(d, "angle_bin.npy")).astype(np.int64)
    ul = (np.load(os.path.join(d, "ul.npy")).astype(np.int64)
          if use_ul else np.zeros(len(tok), dtype=np.int64))
    return tok, bnd, mode, ang, ul


def sweep_decode(model, mu: torch.Tensor, phase: torch.Tensor,
                 cond_name: str, values: list[int], mb: torch.Tensor,
                 ub: torch.Tensor, vb: torch.Tensor, db: torch.Tensor,
                 centroids: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """对固定 mu 逐条件解码 -> 最近质心。返回 (hit_rate_per_cond, matrix[K,K])。

    矩阵行=条件值（扫入的条件），列=质心（train 各条件值）。其余条件轴固定为
    mu 对应窗的真值（单轴扫入，隔离被测条件通路）。
    """
    K = len(values)
    mat = np.zeros((K, K), dtype=np.int64)
    cents = np.stack([centroids[v] for v in values])  # (K, 64)
    cent_t = torch.from_numpy(cents).float().cuda()
    with torch.no_grad():
        for ci, c in enumerate(values):
            if cond_name == "db":
                out = model.decode(mu, phase, vb, torch.full_like(db, c), mb, ub)
            elif cond_name == "vb":
                out = model.decode(mu, phase, torch.full_like(vb, c), db, mb, ub)
            elif cond_name == "mode":
                out = model.decode(mu, phase, vb, db,
                                   torch.full_like(mb, c), ub)
            elif cond_name == "ul":
                out = model.decode(mu, phase, vb, db, mb,
                                   torch.full_like(ub, c))
            else:
                raise ValueError(cond_name)
            d = torch.cdist(out, cent_t)  # (N, K)
            pred = d.argmin(1).cpu().numpy()
            mat[ci] = np.bincount(pred, minlength=K)
    hits = np.diag(mat)
    rates = hits / np.maximum(mat.sum(1), 1)
    return rates, mat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=f"{BASE}/vae_v21/run1")
    ap.add_argument("--inputs-dir", default=f"{BASE}/vae_inputs_v21")
    ap.add_argument("--snapshot", default="vae_ep100.pt")
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()
    dev = torch.device("cuda")

    meta = json.load(open(os.path.join(args.run_dir, "meta.json")))
    use_ul = bool(meta["use_ul"])
    n_modes = int(meta["n_modes"])
    walk_idx = meta["walk_embed_idx"]
    model = DirSpeedPhaseTokenVAE(window=meta["window"], latent_dim=meta["latent_dim"],
                                  hidden_dim=meta["hidden"], n_vbins=3, n_dbins=8,
                                  n_modes=n_modes, use_ul=use_ul).to(dev)
    sd_path = os.path.join(args.run_dir, args.snapshot)
    model.load_state_dict(torch.load(sd_path, map_location=dev))
    model.eval()

    build_meta = json.load(open(os.path.join(args.inputs_dir, "build_meta.json")))
    mode_names = {t["embed_idx"]: t["mode_name"] for t in build_meta["mode_table"]}

    tok, bnd, mode, ang, ul = load_split(args.inputs_dir, use_ul)
    dtok, dbnd, dmode, dang, dul = load_split(
        os.path.join(args.inputs_dir, "dev"), use_ul)
    ttok, tbnd, tmode, tang, tul = load_split(
        os.path.join(args.inputs_dir, "tfam"), use_ul)

    # phase/v-bin 与训练侧同口径：复用 train 冻结投影基（run1/pca.npz）+ train edges
    pca = np.load(os.path.join(args.run_dir, "pca.npz"))
    pmean, V2 = pca["pmean"], pca["V2"]
    base = ((mode == walk_idx).astype(int) * 2 if walk_idx is not None
            else np.asarray(mode))
    rate, phi = phi_rate_from(tok, pmean, V2)
    edges = np.quantile(rate[base == 2], [1 / 3, 2 / 3])
    vb = np.clip(np.digitize(rate, edges), 0, 2).astype(np.int64)
    phase2 = np.stack([np.sin(phi), np.cos(phi)], 1).astype(np.float32)

    W = meta["window"]
    x, y = build_windows_bounded(tok, bnd, W)
    mb = per_frame_labels(mode, bnd, W)
    ub = per_frame_labels(ul, bnd, W)
    pb = per_frame_labels(phase2, bnd, W)
    vbw = per_frame_labels(vb, bnd, W)
    dbw = per_frame_labels(ang, bnd, W)
    km = os.path.join(args.inputs_dir, "window_keep_mask.npy")
    if os.path.isfile(km):
        m = np.load(km)
        x, y, pb, vbw, dbw, mb, ub = (v[m] for v in (x, y, pb, vbw, dbw, mb, ub))
    dx, dy = build_windows_bounded(dtok, dbnd, W)
    dmb = per_frame_labels(dmode, dbnd, W)
    dub = per_frame_labels(dul, dbnd, W)
    tx, ty = build_windows_bounded(ttok, tbnd, W)
    tmb = per_frame_labels(tmode, tbnd, W)
    tub = per_frame_labels(tul, tbnd, W)

    def batches(*arrays):
        n = len(arrays[0])
        for i in range(0, n, args.batch):
            yield [torch.from_numpy(np.ascontiguousarray(a[i:i + args.batch])).to(dev)
                   for a in arrays]

    # ---- train/dev/tfam 前向（重建 MSE + mu 收集，J1/J7/J8） ----
    def forward_split(xa, ya, pba, vbwa, dbwa, mba, uba):
        rec_mses, mus, ys = [], [], []
        finite = True
        with torch.no_grad():
            for i in range(0, len(xa), args.batch):
                sl = slice(i, i + args.batch)
                bx = torch.from_numpy(np.ascontiguousarray(xa[sl])).to(dev)
                by = torch.from_numpy(np.ascontiguousarray(ya[sl])).to(dev)
                bp = torch.from_numpy(np.ascontiguousarray(pba[sl])).to(dev)
                bv = torch.from_numpy(np.ascontiguousarray(vbwa[sl])).to(dev)
                bd = torch.from_numpy(np.ascontiguousarray(dbwa[sl])).to(dev)
                bm = torch.from_numpy(np.ascontiguousarray(mba[sl])).to(dev)
                bu = torch.from_numpy(np.ascontiguousarray(uba[sl])).to(dev)
                mu, _ = model.encode(bx)
                rec = model.decode(mu, bp, bv, bd, bm, bu)
                mse = ((rec - by) ** 2).mean(dim=tuple(range(1, rec.dim())))
                rec_mses.append(mse.cpu().numpy())
                mus.append(mu.cpu().numpy())
                ys.append(by.cpu().numpy())
                finite = finite and bool(np.isfinite(mu.cpu().numpy()).all()
                                         and np.isfinite(rec.cpu().numpy()).all())
        return (np.concatenate(rec_mses), np.concatenate(mus),
                np.concatenate(ys), finite)

    train_mse, train_mu, _, fin_tr = forward_split(x, y, pb, vbw, dbw, mb, ub)
    # dev phase/v-bin：复用 train 投影基（不自算 PCA；dev 可能无 WALK 帧）
    drate, dphi = phi_rate_from(dtok, pmean, V2)
    dvb = np.clip(np.digitize(drate, edges), 0, 2).astype(np.int64)
    dphase2 = np.stack([np.sin(dphi), np.cos(dphi)], 1).astype(np.float32)
    dpb = per_frame_labels(dphase2, dbnd, W)
    dvbw = per_frame_labels(dvb, dbnd, W)
    ddbw = per_frame_labels(dang, dbnd, W)
    dev_mse, dev_mu, _, fin_dev = forward_split(dx, dy, dpb, dvbw, ddbw, dmb, dub)

    # tfam 同口径：train 投影基
    trate, tphi = phi_rate_from(ttok, pmean, V2)
    tvb = np.clip(np.digitize(trate, edges), 0, 2).astype(np.int64)
    tphase2 = np.stack([np.sin(tphi), np.cos(tphi)], 1).astype(np.float32)
    tpb = per_frame_labels(tphase2, tbnd, W)
    tvb_w = per_frame_labels(tvb, tbnd, W)
    tdbw = per_frame_labels(tang, tbnd, W)
    tfam_mse, _, _, fin_tf = forward_split(tx, ty, tpb, tvb_w, tdbw, tmb, tub)

    # J1
    tr_mse, dv_mse = float(train_mse.mean()), float(dev_mse.mean())
    ratio = dv_mse / max(tr_mse, 1e-12)
    j1 = {"train_mse": tr_mse, "dev_mse": dv_mse, "ratio": round(ratio, 4),
          "threshold": 1.5, "pass": bool(ratio <= 1.5)}

    # ---- train 条件质心（train 前向，按真值条件分组） ----
    # 需要逐窗 token_out：再跑一遍 train 前向收集 rec 向量
    recs_all = []
    with torch.no_grad():
        for i in range(0, len(x), args.batch):
            sl = slice(i, i + args.batch)
            bx = torch.from_numpy(np.ascontiguousarray(x[sl])).to(dev)
            bp = torch.from_numpy(np.ascontiguousarray(pb[sl])).to(dev)
            bv = torch.from_numpy(np.ascontiguousarray(vbw[sl])).to(dev)
            bd = torch.from_numpy(np.ascontiguousarray(dbw[sl])).to(dev)
            bm = torch.from_numpy(np.ascontiguousarray(mb[sl])).to(dev)
            bu = torch.from_numpy(np.ascontiguousarray(ub[sl])).to(dev)
            mu, _ = model.encode(bx)
            recs_all.append(model.decode(mu, bp, bv, bd, bm, bu).cpu().numpy())
    recs_all = np.concatenate(recs_all)

    def centroids_by(cond_arr: np.ndarray, values: list[int]) -> dict[int, np.ndarray]:
        return {v: recs_all[cond_arr == v].mean(0) for v in values}

    trained_modes = sorted({int(v) for v in np.unique(mb)})
    db_values = [v for v in range(8) if int((dbw == v).sum()) > 0]
    vb_values = [v for v in range(3) if int((vbw == v).sum()) > 0]
    ul_values = [v for v in range(3) if int((ub == v).sum()) > 0]
    zero_support = {
        "mode": [v for v in range(n_modes) if v not in trained_modes],
        "db": [v for v in range(8) if v not in db_values],
        "vb": [v for v in range(3) if v not in vb_values],
        "ul": [v for v in range(3) if v not in ul_values],
    }

    mode_cents = centroids_by(mb, trained_modes)
    db_cents = centroids_by(dbw, db_values)
    vb_cents = centroids_by(vbw, vb_values)
    ul_cents = centroids_by(ub, ul_values)

    dev_mu_t = torch.from_numpy(dev_mu).float().to(dev)
    dev_pb_t = torch.from_numpy(dpb).float().to(dev)
    dev_vb_t = torch.from_numpy(dvbw).to(dev)
    dev_db_t = torch.from_numpy(ddbw).to(dev)
    dev_mb_t = torch.from_numpy(dmb).to(dev)
    dev_ub_t = torch.from_numpy(dub).to(dev)
    n_dev = len(dev_mu)
    cond_ok = n_dev >= 200

    def run_sweep(name, values, cents):
        vals_t = {"db": dev_db_t, "vb": dev_vb_t, "mode": dev_mb_t, "ul": dev_ub_t}
        rates, mat = sweep_decode(model, dev_mu_t, dev_pb_t, name, values,
                                  dev_mb_t, dev_ub_t, dev_vb_t, dev_db_t, cents)
        return rates, mat

    # J2' + J5
    mode_rates, mode_mat = run_sweep("mode", trained_modes, mode_cents)
    random_mode = 1.0 / len(trained_modes)
    j5 = {
        "trained_mode_values": [f"{v}:{mode_names.get(v, v)}" for v in trained_modes],
        "matrix_rows=condition_cols=centroid": mode_mat.tolist(),
        "row_labels": [mode_names.get(v, v) for v in trained_modes],
        "col_labels": [mode_names.get(v, v) for v in trained_modes],
        "per_condition_hit": {mode_names.get(v, v): round(float(r), 4)
                              for v, r in zip(trained_modes, mode_rates)},
        "random_line": round(random_mode, 4),
        "pass_line": round(2 * random_mode, 4),
        "diagonal_dominant": bool(all(mode_mat[i].argmax() == i
                                      for i in range(len(trained_modes)))),
        "n_dev_windows": int(n_dev),
    }
    j5["pass"] = bool(j5["diagonal_dominant"] and
                      all(r >= 2 * random_mode for r in mode_rates))

    db_rates, db_mat = run_sweep("db", db_values, db_cents)
    j2_db = {"per_condition_hit": {str(v): round(float(r), 4)
                                   for v, r in zip(db_values, db_rates)},
             "matrix": db_mat.tolist(), "random_line": 0.125,
             "pass_line": 0.25, "n_dev_windows": int(n_dev),
             "pass": bool(all(r >= 0.25 for r in db_rates))}
    vb_rates, vb_mat = run_sweep("vb", vb_values, vb_cents)
    j2_vb = {"per_condition_hit": {str(v): round(float(r), 4)
                                   for v, r in zip(vb_values, vb_rates)},
             "matrix": vb_mat.tolist(), "random_line": round(1 / 3, 4),
             "pass_line": round(2 / 3, 4), "n_dev_windows": int(n_dev),
             "pass": bool(all(r >= 2 / 3 for r in vb_rates))}

    # J6'（UL 轴；v2.1 起 sym 池非空则 3 条件可测，sym 分支为新增可测项）
    ul_rates, ul_mat = run_sweep("ul", ul_values, ul_cents)
    ul_names = {0: "none", 1: "sym", 2: "asym"}
    # 二分区分（条件 c vs none）：两条件解码分别对 {none, c} 两质心判定
    bin_tests: dict[str, dict] = {}
    for tname, (c_pos, c_neg) in (("asym_vs_none", (2, 0)),
                                  ("sym_vs_none", (1, 0))):
        sub_idx = [v for v in (c_pos, c_neg) if v in ul_values]
        if len(sub_idx) < 2:
            bin_tests[tname] = None
            continue
        cents2 = np.stack([ul_cents[v] for v in sub_idx])
        cent_t = torch.from_numpy(cents2).float().to(dev)
        hits = {}
        with torch.no_grad():
            for c in sub_idx:
                out = model.decode(dev_mu_t, dev_pb_t, dev_vb_t, dev_db_t,
                                   dev_mb_t, torch.full_like(dev_ub_t, c))
                pred = torch.cdist(out, cent_t).argmin(1).cpu().numpy()
                hits[c] = float((pred == sub_idx.index(c)).mean())
        bin_tests[tname] = {
            f"{ul_names[c_pos]}_cond_hit": round(hits[c_pos], 4),
            f"{ul_names[c_neg]}_cond_hit": round(hits[c_neg], 4),
            "random_line_binary": 0.5,
            "pass_line_binary_2x": 1.0,
        }
    j6 = {"per_condition_hit": {ul_names[v]: round(float(r), 4)
                                for v, r in zip(ul_values, ul_rates)},
          "matrix": ul_mat.tolist(),
          "row_labels": [ul_names[v] for v in ul_values],
          "random_line_3way": round(1 / len(ul_values), 4),
          "pass_line_3way": round(2 / len(ul_values), 4),
          "zero_support_values": [ul_names[v] for v in zero_support["ul"]],
          "zero_support_note": ("v2.1 站立段回归（IDLE+UL）后 sym 有训练窗，"
                                "J6' 三元矩阵含 sym 分支（本次新增可测项）"
                                if 1 in ul_values else
                                "sym 无训练窗不可探（v2 教训：站立 clap 类描述"
                                "被剔），J6 实测为 none/asym 二元"),
          "asym_vs_none": bin_tests["asym_vs_none"],
          "sym_vs_none": bin_tests["sym_vs_none"],
          "ul_train_counts": {ul_names[v]: int((ub == v).sum())
                              for v in ul_values},
          "pass": bool(all(r >= 2 / len(ul_values) for r in ul_rates)
                       and bin_tests["asym_vs_none"] is not None
                       and bin_tests["asym_vs_none"]["asym_cond_hit"] >= 1.0)}

    # J8（tfam 按 mode 分桶 + 域偏移定性）
    per_mode = {}
    for v in sorted({int(t) for t in tmb}):
        sel = tmb == v
        per_mode[mode_names.get(v, str(v))] = {
            "n_windows": int(sel.sum()), "mse": round(float(tfam_mse[sel].mean()), 6)}
    j8 = {"tfam_mse": round(float(tfam_mse.mean()), 6),
          "per_mode": per_mode,
          "reference": {"train_mse": tr_mse, "dev_mse": dv_mse},
          "note": "流形连续性基线，不设阈值；tfam <= dev 量级则 dev 偏移定性为"
                  "域偏移而非过拟合（v1 先例口径）"}

    # J9（v2.1 新增）：IDLE 枢纽指标——重建 MSE + mode 扫描对角命中，只报数不下结论
    idle_idx = next((t["embed_idx"] for t in build_meta["mode_table"]
                     if t["mode_name"] == "IDLE"), None)
    if idle_idx is not None:
        tr_sel = mb == idle_idx
        dv_sel = dmb == idle_idx
        j9 = {
            "idle_embed_idx": idle_idx,
            "train_mse_idle": (round(float(train_mse[tr_sel].mean()), 6)
                               if tr_sel.any() else None),
            "n_train_windows_idle": int(tr_sel.sum()),
            "dev_mse_idle": (round(float(dev_mse[dv_sel].mean()), 6)
                             if dv_sel.any() else None),
            "n_dev_windows_idle": int(dv_sel.sum()),
            "tfam_has_idle": bool((tmb == idle_idx).any()),
            "mode_sweep_idle_hit": j5["per_condition_hit"].get("IDLE"),
            "mode_sweep_idle_row": (mode_mat[trained_modes.index(idle_idx)].tolist()
                                    if idle_idx in trained_modes else None),
            "reference": {"train_mse_all": tr_mse, "dev_mse_all": dv_mse},
            "note": "hub 底座指标（将来 A->IDLE->B 切换链的重建质量/条件可控性"
                    "基线），只报数不下结论",
        }
    else:
        j9 = {"note": "本 run 无 IDLE 座位（非 v2.1 产物？）"}

    # J7
    j7 = {"all_finite": bool(fin_tr and fin_dev and fin_tf
                             and np.isfinite(dev_mu).all()
                             and np.isfinite(recs_all).all()
                             and all(np.isfinite(c).all()
                                     for c in list(mode_cents.values()) +
                                     list(db_cents.values()) +
                                     list(vb_cents.values()) +
                                     list(ul_cents.values()))),
          "note": "零支持条件值（无训练窗的 embedding 类）不参与质心/扫描，"
                  "单独登记于 zero_support"}

    # 描述性统计：head 终态 acc（meta 记录）+ 各 sweep 矩阵
    descriptive = {"dir_head_acc": meta.get("dir_head_acc"),
                   "speed_head_acc": meta.get("speed_head_acc"),
                   "ul_head_acc": meta.get("ul_head_acc"),
                   "note": "owner 更正：head acc 不作 PASS 判据（对抗成功态=近随机）"}

    # v1 对照（若可得）
    v1_block = None
    try:
        v1m = json.load(open(os.path.join(V1_RUN, "meta.json")))
        v1e = json.load(open(os.path.join(V1_RUN, "metrics_d046.json")))
        v1_block = {
            "run": V1_RUN,
            "val_mse_best": v1m.get("val_mse"),
            "J1": v1e.get("J1"),
            "J4_tfam": v1e.get("J4"),
            "heads": {"dir": v1m.get("dir_head_acc"), "spd": v1m.get("speed_head_acc")},
            "cond_dims": {"v1": "mode 10 语义族(2 dev+2 tfam 族不进 train)+db8+vb3",
                          "v2": "mode 7 已训练值(8 子集 mode 中 2 个 T-fam 不进 train"
                                "+intermediate)+db8+vb3+ul3"},
        }
    except Exception as e:  # noqa: BLE001
        v1_block = {"error": str(e)}

    # v2 对照块（v2.1 vs v2 对照表数据源；v2 池 sym 0 窗、IDLE 未入集）
    v2_block = None
    try:
        v2e = json.load(open(os.path.join(V2_RUN, "metrics_d046v2.json")))
        v2_block = {
            "run": V2_RUN,
            "J1": v2e.get("J1"),
            "J5_per_condition_hit": (v2e.get("J5") or {}).get("per_condition_hit"),
            "J5_diagonal_dominant": (v2e.get("J5") or {}).get("diagonal_dominant"),
            "J6_per_condition_hit": (v2e.get("J6") or {}).get("per_condition_hit"),
            "J6_zero_support_note": (v2e.get("J6") or {}).get("zero_support_note"),
            "J8_tfam": v2e.get("J8"),
            "J2p": {k: v for k, v in (v2e.get("J2p") or {}).items()
                    if k in ("db", "vb")},
            "train_windows": (v2e.get("_meta") or {}).get("train_windows"),
            "n_dev_windows": (v2e.get("_meta") or {}).get("n_dev_windows"),
        }
    except Exception as e:  # noqa: BLE001
        v2_block = {"error": str(e)}

    out = {
        "_meta": {"script": "apt_g1/probe_vae_v2.py", "experiment": "D046(v2.1)",
                  "snapshot": sd_path, "n_dev_windows": int(n_dev),
                  "train_windows": int(len(x)), "tfam_windows": int(len(tx)),
                  "zero_support": zero_support,
                  "criteria": "owner 更正版 J1/J2'/J5(生成侧)/J6'/J7/J8 + v2.1 新增 J9(IDLE 枢纽)"},
        "J1": j1,
        "J2p": {"db": j2_db, "vb": j2_vb,
                "mode_controllability": {
                    "per_condition_hit": j5["per_condition_hit"],
                    "random_line": j5["random_line"],
                    "pass_line": j5["pass_line"],
                    "pass": bool(all(r >= 2 * random_mode for r in mode_rates))},
                "pass": bool(j2_db["pass"] and j2_vb["pass"] and
                             all(r >= 2 * random_mode for r in mode_rates))},
        "J5": j5,
        "J6": j6,
        "J7": j7,
        "J8": j8,
        "J9": j9,
        "descriptive": descriptive,
        "curves": meta.get("curves"),
        "train_wall_sec": meta.get("train_wall_sec"),
        "v1_comparison": v1_block,
        "v2_comparison": v2_block,
    }
    out_path = os.path.join(args.run_dir, "metrics_d046v21.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[write] {out_path}")
    print(json.dumps({k: out[k] for k in ("J1", "J2p", "J5", "J6", "J7", "J8", "J9")},
                     ensure_ascii=False, indent=1)[:3500])


if __name__ == "__main__":
    main()
