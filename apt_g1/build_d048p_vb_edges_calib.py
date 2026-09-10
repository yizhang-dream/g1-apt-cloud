"""D048p: edges 命令带反标定 vb 标签构建（D048n 的 edges 校准版，numpy-only）。

§5h 预注册（DS_CONTINUOUS_EXECUTION_PLAN.md，一次不动）：单变量 = D048n
speedA 的 edges 来源——语料 WALK 三分位 [0.42,0.98] → 命令带均值目标校准。
段对齐 / 速度计算 / checksum / 产物格式全部 import 复用
build_d048n_vb_speed_labels（不复制实现），差异仅在 edges：

  参考帧（mode==--ref-mode，默认 2，D048n 同口径）速度分布 v_ref 上网格搜索
  (e1, e2)：e1∈[0.10,0.50]、e2∈[0.50,1.50]、步长 0.01、e1<e2；
  vb=clip(digitize(v_all,edges),0,2) 全帧（与非参考帧、IDLE 一并编目，E39 同构）；
  目标函数 = Σ_i (mean(v_ref[label==i])−t_i)²，i∈{0,1,2} 等权，t=--target-means
  （默认 0.20/0.45/0.65 m/s 命令带）；
  可行性约束 = 每档参考帧占比 ≥ --min-frac（默认 0.10，防退化档）。
  无可行 (e1,e2) → 打印分布事实（v_ref 分位数 5/25/50/75/95、可行域分析）
  并落 vb_speed_calibration_rejected.json 后以非零码退出（负结果路径）。

产物与 D048n 同格式：<out-dir>/vb_speed.npy（(N,) int64）+ vb_speed_meta.json，
meta 增加 calibration 块（target_means/min_frac/搜索网格/选中 edges/三档 mean/
三档 frac/top-5 候选），label_policy 改写为校准口径；experiment="D048p"；
供 train_token_vae_e39.py --vb-npy 消费（消费方式与 D048n 完全一致）。

用法（服务器 .venv_isaac python，本脚本不依赖 torch；本机 numpy 可自测）：
  python build_d048p_vb_edges_calib.py                    # 默认 v21 目录
  python build_d048p_vb_edges_calib.py --target-means 0.20,0.45,0.65
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

# D048n 现役实现单一事实源（段对齐/速度/抽查/默认目录），本脚本只换 edges 口径
try:                      # 服务器执行根平铺 import / 仓库根包 import 双兼容
    from build_d048n_vb_speed_labels import (
        DEFAULT_INPUTS, DEFAULT_NPZ_DIRS, align_segments, frame_speed_bwd,
        md5_file, resolve_npz, token_md5_spot_check,
    )
except ImportError:
    from apt_g1.build_d048n_vb_speed_labels import (
        DEFAULT_INPUTS, DEFAULT_NPZ_DIRS, align_segments, frame_speed_bwd,
        md5_file, resolve_npz, token_md5_spot_check,
    )

# §5h 预注册搜索网格（一次不动；改网格 = 新实验另号）
E1_LO, E1_HI = 0.10, 0.50
E2_LO, E2_HI = 0.50, 1.50
GRID_STEP = 0.01
DEFAULT_TARGET_MEANS = "0.20,0.45,0.65"
DEFAULT_MIN_FRAC = 0.10


def parse_target_means(s: str) -> tuple[float, float, float]:
    """\"t0,t1,t2\" → 三档目标均值（米/秒）。"""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--target-means 需 3 个逗号分隔值，收到 {s!r}")
    t = tuple(float(p) for p in parts)
    return t


def make_edge_grids() -> tuple[np.ndarray, np.ndarray]:
    """预注册网格：(e1∈[0.10,0.50], e2∈[0.50,1.50])，步长 0.01，双端含。"""
    e1 = np.round(np.arange(E1_LO, E1_HI + GRID_STEP / 2.0, GRID_STEP), 2)
    e2 = np.round(np.arange(E2_LO, E2_HI + GRID_STEP / 2.0, GRID_STEP), 2)
    return e1, e2


def calibrate_edges(v_ref: np.ndarray, target_means, min_frac: float,
                    e1_grid: np.ndarray | None = None,
                    e2_grid: np.ndarray | None = None):
    """§5h 预注册网格搜索（与逐对暴力枚举同口径，向量化加速）。

    对每对 (e1,e2)：label = np.digitize(v_ref, [e1,e2])（即 (v>=e1)+(v>=e2)），
    目标 = Σ_i (mean(v_ref[label==i])−t_i)²，可行 = 三档 count/n_ref ≥ min_frac。
    返回 (edges, info)：无可行解时 edges=None；info 携带 n_feasible / top5 /
    v_ref 分位数 / 可行域诊断（best_unconstrained、逐档可达 frac 上界），供
    负结果路径落盘。确定性：同 obj 取网格序（e1 升序为主序）首个。
    """
    v_ref = np.asarray(v_ref, dtype=np.float64)
    t = np.asarray(target_means, dtype=np.float64)
    assert v_ref.ndim == 1 and len(v_ref) > 0, "v_ref 须为非空一维数组"
    assert t.shape == (3,), "target_means 须为三档"
    assert 0.0 < min_frac < 1.0, "min_frac 须 ∈(0,1)"
    n_ref = len(v_ref)
    e1g = make_edge_grids()[0] if e1_grid is None else np.asarray(e1_grid, dtype=np.float64)
    e2g = make_edge_grids()[1] if e2_grid is None else np.asarray(e2_grid, dtype=np.float64)

    # label = (v>=e1) + (v>=e2) 的三档 count/sum 全部由排序前缀和分解
    vs = np.sort(v_ref)
    ps = np.concatenate([[0.0], np.cumsum(vs)])
    k1 = np.searchsorted(vs, e1g, side="left")            # #{v < e1}
    k2 = np.searchsorted(vs, e2g, side="left")            # #{v < e2}
    K1, K2 = k1[:, None], k2[None, :]
    shape = (len(e1g), len(e2g))
    c0 = np.broadcast_to(K1, shape)                       # bin0: v < e1
    c1 = K2 - K1                                          # bin1: e1 <= v < e2
    c2 = np.broadcast_to(n_ref - K2, shape)               # bin2: v >= e2
    s0 = np.broadcast_to(ps[K1], shape)
    s1 = ps[K2] - ps[K1]
    s2 = np.broadcast_to(ps[n_ref] - ps[K2], shape)
    E1, E2 = e1g[:, None], e2g[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        obj_raw = ((s0 / c0 - t[0]) ** 2 + (s1 / c1 - t[1]) ** 2
                   + (s2 / c2 - t[2]) ** 2)
    feasible = (E1 < E2) & (c0 >= min_frac * n_ref) \
        & (c1 >= min_frac * n_ref) & (c2 >= min_frac * n_ref)
    obj = np.where(feasible, obj_raw, np.inf)
    pair_ok = E1 < E2
    obj_relaxed = np.where(pair_ok & np.isfinite(obj_raw), obj_raw, np.inf)
    fracs = np.stack([c0, c1, c2]) / n_ref                # (3, G1, G2)

    info: dict = {
        "n_ref": n_ref,
        "n_grid_pairs": int(obj.size),
        "grid": {"e1": [float(e1g[0]), float(e1g[-1]), GRID_STEP],
                 "e2": [float(e2g[0]), float(e2g[-1]), GRID_STEP],
                 "rule": "e1<e2"},
        "target_means": [float(x) for x in t],
        "min_frac": float(min_frac),
        "v_ref_quantiles": {str(q): float(np.quantile(v_ref, q))
                            for q in (0.05, 0.25, 0.50, 0.75, 0.95)},
        "n_feasible": int(np.isfinite(obj).sum()),
    }
    # 可行域诊断（负结果归因用）：无视 frac 约束的最优对 + 逐档可达 frac 上界
    ij = np.unravel_index(np.argmin(obj_relaxed), obj.shape)
    info["best_unconstrained"] = {
        "e1": float(e1g[ij[0]]), "e2": float(e2g[ij[1]]),
        "obj": float(obj_relaxed[ij]),
        "fracs": [float(f) for f in fracs[:, ij[0], ij[1]]],
    }
    info["max_frac_per_bin"] = [float(fracs[i][pair_ok].max()) for i in range(3)]

    if not np.isfinite(obj).any():
        return None, info
    order = np.argsort(obj.ravel(), kind="stable")
    top5 = []
    for flat in order[:5]:
        i, j = int(flat) // obj.shape[1], int(flat) % obj.shape[1]
        top5.append({"e1": float(e1g[i]), "e2": float(e2g[j]),
                     "obj": float(obj[i, j]),
                     "ref_means": [float(s0[i, j] / c0[i, j]),
                                   float(s1[i, j] / c1[i, j]),
                                   float(s2[i, j] / c2[i, j])],
                     "ref_fracs": [float(f) for f in fracs[:, i, j]]})
    info["top5"] = top5
    bi, bj = int(order[0]) // obj.shape[1], int(order[0]) % obj.shape[1]
    info["edges"] = [float(e1g[bi]), float(e2g[bj])]
    info["obj"] = float(obj[bi, bj])
    info["ref_means"] = top5[0]["ref_means"]
    info["ref_fracs"] = top5[0]["ref_fracs"]
    return np.array(info["edges"], dtype=np.float64), info


def main():
    ap = argparse.ArgumentParser(
        description="D048p: edges 命令带反标定 vb 标签构建"
                    "（D048n 校准版；段对齐/速度复用 build_d048n_vb_speed_labels）")
    ap.add_argument("--inputs-dir", default=DEFAULT_INPUTS,
                    help="VAE 组装输入目录（token/mode[/mode_id]/angle_bin/"
                         "segment_bounds/build_meta.json）")
    ap.add_argument("--npz-dir", nargs="+", default=DEFAULT_NPZ_DIRS,
                    help="源 npz 目录（可多个，builder v2 同款先到先得；"
                         "build_meta 段级 npz_path 优先）")
    ap.add_argument("--out-dir", default="",
                    help="输出目录（默认 = inputs-dir，写 vb_speed.npy + "
                         "vb_speed_meta.json）")
    ap.add_argument("--ref-mode", type=int, default=2,
                    help="参考帧 mode id（默认 2，D048n 同口径）")
    ap.add_argument("--fps", type=float, default=50.0, help="采样率（B4-lite=50Hz）")
    ap.add_argument("--axes", choices=["xy", "xyz"], default="xy",
                    help="速度差分维度（默认 xy 水平面，D046 frame_speed 同口径）")
    ap.add_argument("--target-means", default=DEFAULT_TARGET_MEANS,
                    help="三档目标均值 t0,t1,t2（m/s；§5h 预注册默认 0.20,0.45,0.65）")
    ap.add_argument("--min-frac", type=float, default=DEFAULT_MIN_FRAC,
                    help="每档参考帧最低占比（§5h 预注册默认 0.10，防退化档）")
    args = ap.parse_args()

    target_means = parse_target_means(args.target_means)
    inputs_dir = args.inputs_dir
    out_dir = args.out_dir or inputs_dir
    mode_path = os.path.join(inputs_dir, "mode.npy")
    mode_file = "mode.npy"
    if not os.path.isfile(mode_path):
        mode_path = os.path.join(inputs_dir, "mode_id.npy")
        mode_file = "mode_id.npy"
    token = np.load(os.path.join(inputs_dir, "token.npy")).astype(np.float32)
    mode = np.load(mode_path).astype(np.int64)
    n = len(token)
    assert len(mode) == n, f"mode 长度 {len(mode)} != token N {n}"

    print(f"[load] {inputs_dir}: N={n} mode_file={mode_file}")
    (bounds, src_idx, orig_stems, seg_records, provenance, trans_cache,
     npz_paths, build_meta) = align_segments(inputs_dir, args.npz_dir)
    assert int(bounds[-1, 1]) == n, f"bounds 总帧数 {bounds[-1, 1]} != token N {n}"

    # 段级速度 → 组装帧序；全帧拼成 (N,)（D048n 同口径 import 复用）
    v = np.zeros(n, dtype=np.float64)
    for r in seg_records:
        s0, s1 = r["interval"]
        trans_i, _ = trans_cache[r["stem"]]
        v[s0:s1] = frame_speed_bwd(trans_i[:s1 - s0], args.fps, args.axes)

    # §5h 预注册：参考帧分布上网格搜索 edges（差异点，其余口径 = D048n）
    ref_mask = mode == args.ref_mode
    n_ref = int(ref_mask.sum())
    assert n_ref > 0, f"参考帧为空：mode=={args.ref_mode} 无帧（--ref-mode 口径错误？）"
    v_ref = v[ref_mask]
    print(f"[calib] target_means={list(target_means)} min_frac={args.min_frac} "
          f"ref_frames={n_ref}")
    edges, info = calibrate_edges(v_ref, target_means, args.min_frac)
    if edges is None:
        # 负结果路径：打印分布事实 + 落盘 rejected json + 非零码退出
        q = info["v_ref_quantiles"]
        print("[NEG-RESULT] 无可行 (e1,e2)（语料量程不足？）——v_ref 分位数表:")
        for k, qv in info["v_ref_quantiles"].items():
            print(f"    q{k} = {qv:.4f}")
        print(f"[NEG-RESULT] 可行域分析: n_feasible=0 / "
              f"{info['n_grid_pairs']} 网格对; "
              f"best_unconstrained={info['best_unconstrained']}; "
              f"max_frac_per_bin={[round(f, 4) for f in info['max_frac_per_bin']]} "
              f"(约束 min_frac={args.min_frac})")
        rej = {
            "experiment": "D048p",
            "status": "rejected_no_feasible_edges",
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "inputs_dir": inputs_dir,
            "ref_mode": args.ref_mode,
            "ref_frames": n_ref,
            "target_means": list(target_means),
            "min_frac": args.min_frac,
            "calibration_info": info,
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "vb_speed_calibration_rejected.json"),
                  "w", encoding="utf-8") as f:
            json.dump(rej, f, ensure_ascii=False, indent=1)
        print(f"[write] {out_dir}/vb_speed_calibration_rejected.json")
        raise SystemExit(1)
    print(f"[calib] edges={info['edges']} obj={info['obj']:.6g} "
          f"ref_means={[round(x, 4) for x in info['ref_means']]} "
          f"ref_fracs={[round(x, 4) for x in info['ref_fracs']]} "
          f"n_feasible={info['n_feasible']}")

    vb = np.clip(np.digitize(v, edges), 0, 2).astype(np.int64)
    counts = {int(b): int(c) for b, c in zip(*np.unique(vb, return_counts=True))}
    print(f"[vb] edges={edges.tolist()} all-frame counts={counts}")

    spot = token_md5_spot_check(seg_records, trans_cache, token, bounds)
    print(f"[md5] token 逐帧抽查 {len(spot)} 段全 match: "
          + ", ".join(f"{r['stem']}({r['kind']},{r['frames_checked']}f)" for r in spot))

    ref_mode_name = None
    if "mode_table" in build_meta:
        for t_row in build_meta["mode_table"]:
            if int(t_row["embed_idx"]) == args.ref_mode:
                ref_mode_name = t_row["mode_name"]
    elif "family_order" in build_meta:
        if 0 <= args.ref_mode < len(build_meta["family_order"]):
            ref_mode_name = build_meta["family_order"][args.ref_mode]

    used_files = sorted({resolve_npz(r["stem"], args.npz_dir, npz_paths)
                         for r in seg_records})
    meta = {
        "experiment": "D048p",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "inputs_dir": inputs_dir,
        "npz_dirs": args.npz_dir,
        "mode_file_used": mode_file,
        "n_frames": n,
        "n_segments_out": len(bounds),
        "n_segments_orig": len(orig_stems),
        "fps": args.fps,
        "speed_axes": args.axes,
        "speed_rule": "v[t]=||trans[t]-trans[t-1]||×fps（后向差分），v[0]=v[1]"
                      " 镜像 E39 rate[0]=rate[1]",
        "ref_mode": args.ref_mode,
        "ref_mode_name": ref_mode_name,
        "ref_frames": n_ref,
        "edges": [float(e) for e in edges],
        "bin_counts": counts,
        "calibration": {
            "target_means": [float(t) for t in target_means],
            "min_frac": args.min_frac,
            "grid": info["grid"],
            "edges": info["edges"],
            "obj": info["obj"],
            "ref_means": info["ref_means"],
            "ref_fracs": info["ref_fracs"],
            "n_grid_pairs": info["n_grid_pairs"],
            "n_feasible": info["n_feasible"],
            "top5": info["top5"],
            "rule": "参考帧(mode==ref_mode)速度分布上网格搜索 (e1,e2) 最小化 "
                    "Σ_i (mean_i(v_ref)−t_i)² 等权；vb=clip(digitize(v,edges),0,2) "
                    "全帧；约束三档参考帧占比≥min_frac（§5h 预注册一次不动）",
        },
        "label_policy": "vb=clip(digitize(v,edges),0,2) 全帧；edges=命令带反标定"
                        "（参考帧(mode==ref_mode)速度分布上网格搜索最小化"
                        " Σ_i (mean_i(v_ref)−t_i)²，t=target_means 等权，约束"
                        "三档参考帧占比≥min_frac；对照 D048n=参考帧三分位）；"
                        "替代 train 内部相位 rate 分位标签（对照臂仍用内部标签，"
                        "e39 架构/超参冻结）",
        "alignment_provenance": provenance,
        "checksums": {
            "n_segments": {"bounds": len(bounds), "src_idx": len(src_idx),
                           "match": len(bounds) == len(src_idx)},
            "copies_per_source": {orig_stems[i]: int(c) for i, c in
                                  zip(*np.unique(src_idx, return_counts=True))},
            "total_frames": {"bounds_last": int(bounds[-1, 1]),
                             "token_n": n, "vb_n": len(vb),
                             "match": int(bounds[-1, 1]) == n == len(vb)},
            "token_md5_spot_checks": spot,
        },
        "source_npz_md5": {os.path.splitext(os.path.basename(p))[0]: md5_file(p)
                           for p in used_files},
        "inputs_md5": {f: md5_file(os.path.join(inputs_dir, f))
                       for f in ["token.npy", mode_file, "angle_bin.npy",
                                 "segment_bounds.npy", "build_meta.json"]
                       if os.path.isfile(os.path.join(inputs_dir, f))},
        "outputs": ["vb_speed.npy", "vb_speed_meta.json"],
    }
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "vb_speed.npy"), vb)
    with open(os.path.join(out_dir, "vb_speed_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[write] {out_dir}/vb_speed.npy (N={n}) + vb_speed_meta.json")


if __name__ == "__main__":
    main()
