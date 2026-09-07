"""D046b 交叉 z 探针 v2（单轴交叉转向矩阵，owner 评审修正版）——判别 v2.1
VAE 条件通路「真琴键 vs 贴纸」。

背景：v2.1 探针（probe_vae_v2.py）显示 locomotion 窗 z + 条件 IDLE 解码不向 IDLE
质心转向（0.004）；而此前「条件=WALK 输出像 WALK」的高对角可能被 z 驱动混杂。
本探针把 z 驱动与条件驱动分开：**每次只交叉一个条件轴**，其余条件轴保持窗口
自身真值——z 从条件值 i 的训练窗取（确定性 mu），被测轴条件置 j 解码，输出对
train 前向按真值条件分组的 token 质心做最近质心判定 → 转向矩阵
M[i][j] = P(命中 j)。若输出跟条件走，off-diag 高；若跟 z 走，off-diag 高的
P(命中 i)。

v2 修正（owner 代码级评审 2026-09-07 五项发现，v1 数值矩阵仍可用、机制结论作废）：
  ① z_follow 实现错：v1 取对角均值（=完美跟条件时反而 1.0）。预注册定义是
     off-diagonal 格的 P(pred==i)（输出仍命中 z 源类）。v2 每格保存完整命中
     向量，z_follow = off-diag P(pred==i) 均值。v1 台账 z/c=1.53 作废。
  ② mode 输出名错位：v1 把矩阵位置当 embed_idx 查名（实际参与标签
     [0,2,3,4,5,7,8,9]），argmax 目标名全体错位（如 WALK→WALK 0.724 被写成
     预测 SLOW_WALK）。v2 一律经 values[t] 映射，出报告前打印映射核对。
  ③ 质心未校准 + 基线缺失：v1 质心取模型自重建均值。v2 改用真实 train 窗末
     token（y，与 decode 输出同空间）按真值分组均值做质心；新增同条件（对角）
     识别率=可辨识性校准；新增正确基线
     P(命中 j | z_i, cond_i)（同 z 不改条件=自条件前向），逐轴报
     mean delta = 交叉 off-diag − 基线 off-diag。
  ④ (mode,UL) 训练支持分层：交叉 mode 时 UL 保持窗口真值，而训练集 sym 全来自
     IDLE、零 (IDLE,none)——大量格子是训练未见组合，IDLE 归因混入 OOD。
     v2 从 train 窗标签构建 (mode,UL) 支持集 S，mode/ul 轴每格按交叉后组合
     是否 ∈S 拆 trained-combo / unseen-combo 两套矩阵；IDLE 归因只允许引用
     trained-combo 格；支持矩阵本身作为产物输出。
  ⑤ （门脚本侧，不在本脚本）D047 path ratio 时间窗 diff，见
     b3p_gate_isaac.py playback/hold instrumentation。

预注册判读量（owner 指令 2026-09-07；z_follow 按 ① 修正实现）：
  cond_follow = 逐轴 off-diagonal (i≠j) 的 P(命中 j) 均值
  z_follow    = 逐轴 off-diagonal (i≠j) 的 P(命中 i) 均值
  baseline    = off-diagonal (i≠j) 的 P(命中 j | z_i, cond_i) 均值（v2 增补）
  chance      = 1/k
  判定（逐轴，按序）：cond_follow >= 2*chance -> 条件通路可用（贴纸问题不成立）；
  elif z_follow >= 3*cond_follow -> z 主导（架构问题证实）；else -> 部分可控
  （逐格报告断裂 (i,j) 对，重点 IDLE 行/列）。
  判定措辞只允许到「该指标下响应强弱」，不做机制实锤断言（owner 评审口径）。
样本门槛：矩阵行（z 组）>= --min-win（默认 150）训练窗（respect segment_bounds
+ keep mask）；低于门槛的条件值不入阵、单独登记。

J9 复测：mode 矩阵 IDLE 行（z_IDLE + 条件 j -> 能否被命令离开 hub）与 IDLE 列
（z_i + 条件 IDLE -> 能否被命令进入 hub）直接回答「hub 能否被命令」；列与
v2.1 探针 dev-z 版 0.004 直接对照。v2 起行/列另拆 trained/unseen-combo。

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
产出 <run-dir>/../probe_cross_v2/{metrics_d046b_v2.json, summary_d046b_v2.txt}
（v1 产物在 probe_cross/ 不覆盖，append-only 对照）
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
        cells = "".join((f"{v:7.3f}" if v is not None and np.isfinite(v)
                         else "    --").rjust(w) for v in row)
        lines.append(str(r)[:8].ljust(9) + cells)
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
                    help="缺省 <run-dir>/../probe_cross_v2")
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
    x, tok_y = build_windows_bounded(tok, bnd, W)  # tok_y=窗末真 token（decode 目标空间）
    mb = per_frame_labels(mode, bnd, W)
    ub = per_frame_labels(ul, bnd, W)
    pb = per_frame_labels(phase2, bnd, W)
    vbw = per_frame_labels(vb, bnd, W)
    dbw = per_frame_labels(ang, bnd, W)
    km = os.path.join(args.inputs_dir, "window_keep_mask.npy")
    if os.path.isfile(km):
        m = np.load(km)
        x, tok_y, pb, vbw, dbw, mb, ub = (v[m] for v in
                                          (x, tok_y, pb, vbw, dbw, mb, ub))
    N = len(x)
    print(f"[data] train windows N={N}", flush=True)

    def batches(*arrays):
        for i in range(0, N, args.batch):
            yield [torch.from_numpy(np.ascontiguousarray(a[i:i + args.batch])).to(dev)
                   for a in arrays]

    # ---- 全窗 mu（z 源）+ 自条件前向 rec（v2 基线源：同 z 不改条件） ----
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

    # ---- [修正④] (mode,UL) 训练支持集 S（window 级组合计数） ----
    n_modes_total = int(meta["n_modes"])
    sup_counts = np.zeros((n_modes_total, 3), dtype=np.int64)
    np.add.at(sup_counts, (mb, ub), 1)
    sup_ok = sup_counts > 0
    sup_rows = [m for m in range(n_modes_total) if sup_counts[m].any()]
    sup_row_names = [mode_names.get(m, str(m)) for m in sup_rows]
    print("\n[support] (mode,UL) train-window 支持矩阵（行=mode 列=UL，0=缺席）：",
          flush=True)
    print("mode\\UL    ".ljust(11) + "".join(UL_NAMES[u].rjust(9) for u in range(3)),
          flush=True)
    for m, rn_ in zip(sup_rows, sup_row_names):
        print(rn_[:10].ljust(11) + "".join(str(int(sup_counts[m, u])).rjust(9)
                                           for u in range(3)), flush=True)

    def combo_trained_mask(axis: str, idx: np.ndarray, vj: int):
        """axis=mode/ul 时，格 (i,j) 内各窗交叉后组合 (交叉值vj, 保持值) 是否 ∈S。"""
        if axis == "mode":
            return sup_ok[vj, ub[idx]]          # (mode=vj, ul=窗真值)
        if axis == "ul":
            return sup_ok[mb[idx], vj]          # (mode=窗真值, ul=vj)
        return None

    # ---- 轴定义：mode / ul / vb / db（各轴独立矩阵） ----
    def axis_values(labels: np.ndarray) -> tuple[list[int], dict[int, str]]:
        cnt = {int(v): int((labels == v).sum()) for v in np.unique(labels)}
        keep = sorted(v for v, c in cnt.items() if c >= args.min_win)
        dropped = {v: c for v, c in cnt.items() if c < args.min_win}
        return keep, dropped

    axes = {
        "mode": (mb, {v: mode_names.get(v, str(v)) for v in range(n_modes_total)}),
        "ul": (ub, UL_NAMES),
        "vb": (vbw, {0: "vb0", 1: "vb1", 2: "vb2"}),
        "db": (dbw, {v: f"db{v}" for v in range(8)}),
    }
    results: dict[str, dict] = {}
    for axis, (labels, names) in axes.items():
        values, dropped = axis_values(labels)
        K = len(values)
        if K < 2:
            # 单值轴无 off-diag（readout 会 nan），登记后跳过
            results[axis] = {
                "values": values,
                "row_names": [names.get(v, str(v)) for v in values],
                "row_train_windows": [int((labels == v).sum()) for v in values],
                "below_min_win_dropped": {},
                "verdict": "k1_skipped(单值轴，无 off-diag 可判读)",
            }
            print(f"\n[axis={axis}] k={K}<2 -> skipped(单值轴)", flush=True)
            continue
        # [修正②] 行列名一律经 values[t] 映射；argmax 目标名查 col_names（位置->名）
        row_names = [names.get(v, str(v)) for v in values]
        col_names = list(row_names)
        print(f"\n[axis={axis}] values 核对（矩阵位置->标签值->名）: "
              f"{[(t, v, row_names[t]) for t, v in enumerate(values)]}", flush=True)
        counts = [int((labels == v).sum()) for v in values]
        # [修正③] 质心 = 真实 train 窗末 token（y，decode 目标空间）按真值分组均值
        cents = np.stack([tok_y[labels == v].mean(0) for v in values])
        cent_t = torch.from_numpy(cents).float().to(dev)

        # [修正③] 基线：同 z 不改条件（自条件前向 recs）对同一质心集的命中分布
        base = np.zeros((K, K), dtype=np.float64)   # base[i,j]=P(pred==j | z_i, cond_i)
        with torch.no_grad():
            for i, vi in enumerate(values):
                idx = np.where(labels == vi)[0]
                rr = torch.from_numpy(recs[idx]).float().to(dev)
                pred_b = torch.cdist(rr, cent_t).argmin(1).cpu().numpy()
                base[i] = np.bincount(pred_b, minlength=K) / len(idx)

        M = np.zeros((K, K), dtype=np.float64)      # M[i,j]=P(pred==j | z_i, cond_j)
        hit_i = np.zeros((K, K), dtype=np.float64)  # hit_i[i,j]=P(pred==i)（z 源类命中）
        top = np.zeros((K, K), dtype=np.int64)      # 每格最近质心 argmax（断裂归因用）
        top_p = np.zeros((K, K), dtype=np.float64)  # 每格最近质心概率
        stratified = axis in ("mode", "ul")
        sub_names = ("trained", "unseen") if stratified else ()
        # 分层存格：hit 向量 + 窗数；空格 = None
        s_hit = {s: [[None] * K for _ in range(K)] for s in sub_names}
        s_n = {s: np.zeros((K, K), dtype=np.int64) for s in sub_names}
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
                    hv = (torch.bincount(pred, minlength=K).float()
                          / float(len(idx))).cpu().numpy()  # [修正①] 完整命中向量
                    M[i, j] = float(hv[j])
                    hit_i[i, j] = float(hv[i])
                    top[i, j] = int(hv.argmax())
                    top_p[i, j] = float(hv.max())
                    if stratified:
                        tm = combo_trained_mask(axis, idx, vj)
                        for s, sm in (("trained", tm), ("unseen", ~tm)):
                            if sm.any():
                                sm_t = torch.from_numpy(sm).to(dev)
                                bcs = torch.bincount(pred[sm_t], minlength=K).float()
                                s_hit[s][i][j] = (bcs / float(sm.sum())).cpu().numpy()
                                s_n[s][i, j] = int(sm.sum())
        off = ~np.eye(K, dtype=bool)
        cond_follow = float(M[off].mean())
        z_follow = float(hit_i[off].mean())      # [修正①] off-diag P(pred==i)
        base_follow = float(base[off].mean())    # [修正③] 基线 off-diag
        mean_delta = cond_follow - base_follow   # [修正③] 交叉 − 基线
        diag_ident = float(np.diag(M).mean())    # [修正③] 同条件可辨识率
        chance = 1.0 / K
        if cond_follow >= 2 * chance:
            verdict = "cond_available(条件通路可用,贴纸问题不成立)"
        elif z_follow >= 3 * cond_follow:
            verdict = "z_dominant(z 主导,架构问题证实)"
        else:
            verdict = "partial(部分可控)"
        # 断裂对归因：off-diag 中条件命中所指质心不是输出 argmax 的格（名经 values 映射）
        broken = [[row_names[i], col_names[j], round(float(M[i, j]), 4),
                   col_names[int(top[i, j])], round(float(top_p[i, j]), 4)]
                  for i in range(K) for j in range(K)
                  if i != j and int(top[i, j]) != j]

        def strat_pack(s: str) -> dict | None:
            if not stratified:
                return None
            mj = [[(float(s_hit[s][i][j][j]) if s_hit[s][i][j] is not None else None)
                   for j in range(K)] for i in range(K)]
            mi = [[(float(s_hit[s][i][j][i]) if s_hit[s][i][j] is not None else None)
                   for j in range(K)] for i in range(K)]
            tg = [[(col_names[int(s_hit[s][i][j].argmax())]
                    if s_hit[s][i][j] is not None else None)
                   for j in range(K)] for i in range(K)]
            tp = [[(float(s_hit[s][i][j].max()) if s_hit[s][i][j] is not None else None)
                   for j in range(K)] for i in range(K)]
            offv_j = [mj[i][j] for i in range(K) for j in range(K)
                      if i != j and mj[i][j] is not None]
            offv_i = [mi[i][j] for i in range(K) for j in range(K)
                      if i != j and mi[i][j] is not None]
            return {
                "cell_n": s_n[s].tolist(),
                "matrix_hit_j": mj,
                "matrix_hit_i": mi,
                "argmax_target": tg,
                "argmax_prob": tp,
                "cond_follow_offdiag_nonempty": (round(float(np.mean(offv_j)), 4)
                                                 if offv_j else None),
                "z_follow_offdiag_nonempty": (round(float(np.mean(offv_i)), 4)
                                              if offv_i else None),
                "n_nonempty_offdiag_cells": len(offv_j),
            }

        combo_strat = None
        if stratified:
            unseen_share = float((s_n["unseen"][off].sum()
                                  / max(int(s_n["trained"][off].sum()
                                            + s_n["unseen"][off].sum()), 1)))
            combo_strat = {
                "combo_definition": (
                    "trained: 交叉后 (mode,UL) ∈ train 窗支持集 S；unseen: ∉S；"
                    "空格=None。IDLE 归因只允许引用 trained 格" if axis == "mode"
                    else "trained: 交叉后 (mode,UL) ∈ train 窗支持集 S；unseen: ∉S；"
                         "空格=None"),
                "trained": strat_pack("trained"),
                "unseen": strat_pack("unseen"),
                "offdiag_window_share_unseen": round(unseen_share, 4),
            }

        results[axis] = {
            "values": values, "row_names": row_names, "row_train_windows": counts,
            "below_min_win_dropped": {names.get(v, str(v)): c for v, c in dropped.items()},
            "matrix_hit_j": M.tolist(),
            "matrix_hit_i_zsource": hit_i.tolist(),
            "matrix_argmax_target": [[col_names[int(t)] for t in row] for row in top],
            "matrix_argmax_prob": top_p.tolist(),
            "baseline_hit_j_samecond": base.tolist(),
            "matrix_delta_hit_j_minus_baseline": (M - base).tolist(),
            "cond_follow": round(cond_follow, 4),
            "z_follow": round(z_follow, 4),
            "baseline_follow_offdiag": round(base_follow, 4),
            "mean_delta_offdiag_cross_minus_baseline": round(mean_delta, 4),
            "diag_identification": round(diag_ident, 4),
            "chance": round(chance, 4),
            "z_over_cond_ratio": round(z_follow / max(cond_follow, 1e-9), 2),
            "cond_over_chance": round(cond_follow / chance, 2),
            "cond_over_baseline": round(cond_follow / max(base_follow, 1e-9), 2),
            "verdict": verdict,
            "offdiag_broken_cells_top1_not_j": broken,
            "combo_stratified": combo_strat,
        }
        print(f"\n[axis={axis}] k={K} chance={chance:.3f} "
              f"cond_follow={cond_follow:.4f} z_follow={z_follow:.4f} "
              f"baseline={base_follow:.4f} mean_delta={mean_delta:+.4f} "
              f"diag_ident={diag_ident:.4f} -> {verdict}", flush=True)
        print("M_hit_j (P(pred==j | z_i, cond_j)):", flush=True)
        print(fmt_matrix(M, row_names, col_names), flush=True)
        print("baseline P(pred==j | z_i, cond_i):", flush=True)
        print(fmt_matrix(base, row_names, col_names), flush=True)
        if stratified:
            for s in sub_names:
                print(f"stratified[{s}] M_hit_j:", flush=True)
                print(fmt_matrix(np.array([[np.nan if v is None else v for v in r]
                                           for r in results[axis]["combo_stratified"][s]["matrix_hit_j"]],
                                          dtype=np.float64), row_names, col_names),
                      flush=True)

    # ---- J9 复测：IDLE 行/列（mode 矩阵；含 trained/unseen 分层） ----
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
                    "（能否被命令进入；与 v2.1 dev-z 版 0.004 对照）；"
                    "分层解读只引用 trained 格",
        }
        st = mres.get("combo_stratified")
        if st:
            mj_t = st["trained"]["matrix_hit_j"]
            mj_u = st["unseen"]["matrix_hit_j"]
            n_t = st["trained"]["cell_n"]
            for key, mat in (("trained", mj_t), ("unseen", mj_u)):
                rowv = [mat[r][j] for j in range(K) if j != r and mat[r][j] is not None]
                colv = [mat[i][r] for i in range(K) if i != r and mat[i][r] is not None]
                j9[f"idle_row_offdiag_{key}"] = (round(float(np.mean(rowv)), 4)
                                                 if rowv else None)
                j9[f"idle_col_offdiag_{key}"] = (round(float(np.mean(colv)), 4)
                                                 if colv else None)
                j9[f"idle_row_nonempty_offdiag_cells_{key}"] = len(rowv)
                j9[f"idle_col_nonempty_offdiag_cells_{key}"] = len(colv)
        print(f"\n[J9] IDLE 行(离开) off-diag mean={j9['idle_row_offdiag_mean']}; "
              f"IDLE 列(进入) off-diag mean={j9['idle_col_offdiag_mean']}", flush=True)
        if st:
            print(f"[J9 stratified] 行 trained={j9.get('idle_row_offdiag_trained')}"
                  f"(n_cells={j9.get('idle_row_nonempty_offdiag_cells_trained')}) "
                  f"unseen={j9.get('idle_row_offdiag_unseen')}"
                  f"(n_cells={j9.get('idle_row_nonempty_offdiag_cells_unseen')}); "
                  f"列 trained={j9.get('idle_col_offdiag_trained')}"
                  f"(n_cells={j9.get('idle_col_nonempty_offdiag_cells_trained')}) "
                  f"unseen={j9.get('idle_col_offdiag_unseen')}"
                  f"(n_cells={j9.get('idle_col_nonempty_offdiag_cells_unseen')})",
                  flush=True)

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

    out_dir = args.out_dir or os.path.join(args.run_dir, "..", "probe_cross_v2")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    overall = {}
    for a in results:
        ra = results[a]
        if "cond_follow" not in ra:      # k1_skipped 单值轴
            overall[a] = {"verdict": ra["verdict"]}
            continue
        overall[a] = {"cond_follow": ra["cond_follow"],
                      "z_follow": ra["z_follow"],
                      "baseline_follow_offdiag": ra["baseline_follow_offdiag"],
                      "mean_delta_offdiag": ra["mean_delta_offdiag_cross_minus_baseline"],
                      "diag_identification": ra["diag_identification"],
                      "chance": ra["chance"],
                      "verdict": ra["verdict"]}
    out = {
        "_meta": {"script": "apt_g1/probe_vae_cross.py", "experiment": "D046b",
                  "version": "v2 (owner 评审修正版 2026-09-07)",
                  "snapshot": sd_path, "device": str(dev),
                  "train_windows": int(N), "min_win": args.min_win,
                  "design": "单轴交叉：z 取条件值 i 的训练窗 mu，被测轴置 j，"
                            "其余轴保持窗口真值；质心=train 窗末真 token（y，与"
                            "decode 输出同空间）按真值分组"
                            "均值（v2 修正③）；基线=同 z 自条件前向命中分布",
                  "corrections": {
                      "z_follow": "off-diag P(pred==i)（v1 误用对角均值，作废 z/c=1.53）",
                      "name_mapping": "行/列/argmax 目标名一律经 values[t] 映射"
                                      "（v1 位置当标签，mode 轴全体错位）",
                      "centroid": "train 窗末真 token（y=decode 目标空间；v1 为模型"
                                  "自重建均值，未校准）",
                      "baseline": "P(命中 j | z_i, cond_i) vs 交叉逐轴 mean delta",
                      "combo_stratify": "(mode,UL) 训练支持集 S 分层 trained/unseen"
                                        " 矩阵；IDLE 归因只引用 trained 格",
                      "db_bin0": "bin0=段首方向±22.5°=前向（build_b4lite_vae_inputs.py）；"
                                 "v1 台账「db5=前向邻域」表述有误",
                  },
                  "criteria": "预注册：cond_follow>=2*chance->条件通路可用；"
                              "elif z_follow>=3*cond_follow->z 主导；else 部分可控；"
                              "措辞只到「该指标下响应强弱」",
                  "mode_axis_note": "SLOW_WALK(1)/INJURED_WALK(6) 为 T-fam 零训练窗，"
                                    "mode 轴按 >=150 窗实况取值（8 度）"},
        "mode_ul_support": {
            "note": "train 窗级 (mode,UL) 组合计数；0=训练未见组合（数据侧修复候选"
                    "留给 owner：如 IDLE+none 语料有但池未选）",
            "col_names": [UL_NAMES[u] for u in range(3)],
            "row_names": sup_row_names,
            "row_embed_idx": sup_rows,
            "counts": [[int(sup_counts[m, u]) for u in range(3)] for m in sup_rows],
            "trained_pairs": {f"{mode_names.get(m, str(m))}|{UL_NAMES[u]}":
                              int(sup_counts[m, u])
                              for m in sup_rows for u in range(3)
                              if sup_counts[m, u] > 0},
            "absent_pairs": [f"{mode_names.get(m, str(m))}|{UL_NAMES[u]}"
                             for m in sup_rows for u in range(3)
                             if sup_counts[m, u] == 0],
        },
        "axes": results,
        "j9_hub": j9,
        "z_diag_idle_vs_loco": z_diag,
        "verdict_overall": overall,
    }
    jp = os.path.join(out_dir, "metrics_d046b_v2.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    sp = os.path.join(out_dir, "summary_d046b_v2.txt")
    with open(sp, "w", encoding="utf-8") as f:
        f.write(f"D046b 交叉 z 探针 v2 summary（owner 评审修正版）\nsnapshot={sd_path}\nN={N}\n")
        f.write("\n(mode,UL) train-window 支持矩阵（0=缺席）：\n")
        f.write("mode\\UL    ".ljust(11) + "".join(UL_NAMES[u].rjust(9) for u in range(3)) + "\n")
        for m, rn_ in zip(sup_rows, sup_row_names):
            f.write(rn_[:10].ljust(11) + "".join(str(int(sup_counts[m, u])).rjust(9)
                                                 for u in range(3)) + "\n")
        for a, r in results.items():
            if "cond_follow" not in r:
                f.write(f"\n[axis={a}] skipped: {r['verdict']}\n")
                continue
            f.write(f"\n[axis={a}] k={len(r['values'])} chance={r['chance']} "
                    f"cond_follow={r['cond_follow']} z_follow={r['z_follow']} "
                    f"baseline={r['baseline_follow_offdiag']} "
                    f"mean_delta={r['mean_delta_offdiag_cross_minus_baseline']} "
                    f"diag_ident={r['diag_identification']} "
                    f"z/c={r['z_over_cond_ratio']} cond/chance={r['cond_over_chance']}"
                    f"\nverdict: {r['verdict']}\n")
            f.write("values 核对: " + str(list(zip(r["values"], r["row_names"]))) + "\n")
            f.write("M_hit_j (P(pred==j | z_i, cond_j)):\n")
            f.write(fmt_matrix(np.asarray(r["matrix_hit_j"]),
                               r["row_names"], r["row_names"]) + "\n")
            f.write("baseline_hit_j (P(pred==j | z_i, cond_i)):\n")
            f.write(fmt_matrix(np.asarray(r["baseline_hit_j_samecond"]),
                               r["row_names"], r["row_names"]) + "\n")
            f.write("delta (cross - baseline):\n")
            f.write(fmt_matrix(np.asarray(r["matrix_delta_hit_j_minus_baseline"]),
                               r["row_names"], r["row_names"]) + "\n")
            f.write("matrix_hit_i (P(pred==i | z_i, cond_j), z_follow 源):\n")
            f.write(fmt_matrix(np.asarray(r["matrix_hit_i_zsource"]),
                               r["row_names"], r["row_names"]) + "\n")
            f.write("argmax target (per cell):\n")
            for rn_, trow, prow in zip(r["row_names"],
                                       r["matrix_argmax_target"],
                                       r["matrix_argmax_prob"]):
                cells = ", ".join(f"j={cn_}->{t_}({p_:.2f})"
                                  for cn_, t_, p_ in zip(r["row_names"], trow, prow))
                f.write(f"  z={rn_:<14s} {cells}\n")
            if r.get("combo_stratified"):
                for s in ("trained", "unseen"):
                    st = r["combo_stratified"][s]
                    f.write(f"stratified[{s}] M_hit_j "
                            f"(cond_follow_nonempty={st['cond_follow_offdiag_nonempty']}, "
                            f"z_follow_nonempty={st['z_follow_offdiag_nonempty']}, "
                            f"n_nonempty_offdiag={st['n_nonempty_offdiag_cells']}):\n")
                    f.write(fmt_matrix(np.array(
                        [[np.nan if v is None else v for v in row] for row in st["matrix_hit_j"]],
                        dtype=np.float64), r["row_names"], r["row_names"]) + "\n")
                    f.write(f"stratified[{s}] argmax target:\n")
                    for rn_, trow in zip(r["row_names"], st["argmax_target"]):
                        cells = ", ".join(f"j={cn_}->{t_}" for cn_, t_
                                          in zip(r["row_names"], trow) if t_ is not None)
                        f.write(f"  z={rn_:<14s} {cells}\n")
        if j9:
            f.write(f"\nJ9 hub: 离开={j9['idle_row_offdiag_mean']} "
                    f"进入={j9['idle_col_offdiag_mean']} "
                    f"(v2.1 dev-z 参考 0.004, chance={j9['chance']})\n")
            for s in ("trained", "unseen"):
                f.write(f"J9 stratified[{s}]: 行离开={j9.get(f'idle_row_offdiag_{s}')}"
                        f"(cells={j9.get(f'idle_row_nonempty_offdiag_cells_{s}')}) "
                        f"列进入={j9.get(f'idle_col_offdiag_{s}')}"
                        f"(cells={j9.get(f'idle_col_nonempty_offdiag_cells_{s}')})\n")
        if z_diag:
            f.write(f"zdiag: mean|shift|={z_diag['mean_abs_shift_top8']} "
                    f"std_ratio={z_diag['std_ratio_idle_over_loco_mean']}\n")
    print(f"\n[write] {jp}\n[write] {sp}", flush=True)
    print(json.dumps(overall, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
