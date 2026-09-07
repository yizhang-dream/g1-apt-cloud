"""B4-lite VAE v1 训练输入构建（D046）。

把 B4-lite 冻结数据集（D045 manifest_v2，147 段）的 npz 段转成
train_token_vae_e39.py 的 {token,mode,angle_bin}.npy 三件套，并补上
segment_bounds.npy（窗口不跨段硬约束的边界表）。

口径（coordinator 预定设计，2026-09-07）：
  1. 划分：manifest_v2 set_role 为准，且排除 t_role=boundary_ref 3 段
     （A029/A470 在 train、A005 在 dev——boundary 3 段保留原 set_role，
     必须二次排除才得到 train 84 / dev 20 / T-fam 22）。
  2. 段→帧：逐帧速度 = 中心差分 v_t=(trans[t+1]-trans[t-1])/(2dt)，dt=0.02，
     取 XY 模长，5 帧（0.1s）滑动平均平滑；端点一阶差分。
  3. angle_bin（8-bin，45°）：段首前 K=10 个有效帧（平滑速度 >=0.05 m/s）
     方向角的圆均值 = theta0；逐帧 delta=wrap(atan2(vy,vx)-theta0)，
     bin=floor((delta+22.5°)/45°) mod 8（bin0=±22.5°=前向）；速度不足帧
     继承前一有效 bin，段首无有效帧则 bin0；全程纯站立段全段 bin0 并记录。
  4. mode = 语义族固定序 0-9（train 实际只含 6 训练族；dev/tfam 各 2 族，
     mode 编码与 train 同表，方便跨集评 head 时对齐）。
  5. 族平衡过采样（仅 train）：按族补齐到最大族帧预算，确定性平铺复制
     （stem 排序轮转整段复制，末段截断到预算），因子写 meta。
  6. norm stats：token mean/std 只从 train 84 段原始帧算（过采样前），
     落 norm_stats_d046.npz。v_bin 不预计算——训练脚本 L172 phase-rate
     三分位机制自派生（保持 E39 canonical 不动）。

用法（服务器 .venv_isaac python）：
  python build_b4lite_vae_inputs.py --split train --out-dir .../g1_b4lite/vae_v1
  python build_b4lite_vae_inputs.py --split dev  --out-dir .../g1_b4lite/vae_v1
  python build_b4lite_vae_inputs.py --split tfam --out-dir .../g1_b4lite/vae_v1
产出：<out-dir>/{token,mode,angle_bin,segment_bounds}.npy + build_meta.json
（train 另有 norm_stats_d046.npz）；dev/tfam 落 <out-dir>/dev|tfam/ 子目录。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np

FAMILIES = [
    "forward_walk",        # 0
    "slow_walk",           # 1
    "fast_walk_run",       # 2  <- E39 机制 mode==2（phase PCA + v-bin 分位）
    "forward_jump",        # 3
    "dance_rhythm",        # 4
    "turn_walk",           # 5
    "lateral",             # 6
    "start_stop_transition",  # 7
    "asym_upper",          # 8
    "posture_change",      # 9
]
FAM_ID = {f: i for i, f in enumerate(FAMILIES)}

SET_ROLE = {"train": "train", "dev": "dev", "tfam": "test"}
T_ROLE = {"train": "train", "dev": "dev", "tfam": "T-fam"}


def wrap_pi(a: np.ndarray) -> np.ndarray:
    return np.mod(a + np.pi, 2.0 * np.pi) - np.pi


def frame_speed(trans: np.ndarray, dt: float, smooth: int) -> np.ndarray:
    """逐帧 XY 速度模长：中心差分 + 端点一阶差分 + 滑动平均平滑。"""
    v = np.zeros_like(trans, dtype=np.float64)
    v[1:-1] = (trans[2:] - trans[:-2]) / (2.0 * dt)
    v[0] = (trans[1] - trans[0]) / dt
    v[-1] = (trans[-1] - trans[-2]) / dt
    spd = np.hypot(v[:, 0], v[:, 1])
    if smooth > 1:
        pad = smooth // 2
        sp = np.concatenate([np.full(pad, spd[0]), spd, np.full(pad, spd[-1])])
        spd = np.convolve(sp, np.ones(smooth) / smooth, mode="valid")
    return spd


def angle_bins(trans: np.ndarray, spd: np.ndarray, v_thresh: float,
               theta0_k: int) -> tuple[np.ndarray, dict]:
    """8-bin 逐帧方向 bin（bin0=段首方向 ±22.5° = 前向）。"""
    n = len(spd)
    # 速度方向 = trans 差分方向（非位置角）；差分窗口与 frame_speed 一致
    d = np.zeros_like(trans, dtype=np.float64)
    d[1:-1] = trans[2:] - trans[:-2]
    d[0] = trans[1] - trans[0]
    d[-1] = trans[-1] - trans[-2]
    ang = np.arctan2(d[:, 1], d[:, 0])
    valid = spd >= v_thresh
    info = {"n_frames": n, "n_valid": int(valid.sum())}
    if not valid.any():
        info["standing_segment"] = True
        return np.zeros(n, dtype=np.int64), info
    info["standing_segment"] = False
    vi = np.where(valid)[0]
    head = vi[:theta0_k]
    theta0 = np.arctan2(np.sin(ang[head]).mean(), np.cos(ang[head]).mean())
    info["theta0_rad"] = float(theta0)
    delta = wrap_pi(ang - theta0)
    bins = np.floor((delta + np.pi / 8.0) / (np.pi / 4.0)).astype(np.int64) % 8
    out = np.zeros(n, dtype=np.int64)
    last = 0
    for i in range(n):
        if valid[i]:
            last = int(bins[i])
        out[i] = last  # 段首无有效帧继承 bin0
    return out, info


def load_split(manifest_path: str, npz_dir: str, split: str) -> list[dict]:
    segs = json.load(open(manifest_path))
    role, trole = SET_ROLE[split], T_ROLE[split]
    out = []
    for e in segs:
        b = e.get("b4lite", {})
        sp = b.get("split") or {}
        if sp.get("set_role") != role or sp.get("t_role") != trole:
            continue
        fam = (b.get("semantics") or {}).get("family")
        stem = e["stem"]
        path = os.path.join(npz_dir, stem + ".npz")
        z = np.load(path)
        tok = z["tokens"].astype(np.float32)
        assert len(tok) == int(e["n_rows_enc"]), f"{stem}: n_rows mismatch"
        out.append({"stem": stem, "family": fam, "tokens": tok,
                    "trans": z["trans_m"].astype(np.float64)})
    out.sort(key=lambda s: s["stem"])
    return out


def build_arrays(segs: list[dict], dt: float, smooth: int, v_thresh: float,
                 theta0_k: int) -> tuple[np.ndarray, ...]:
    toks, modes, bins, bounds, stems = [], [], [], [], []
    off = 0
    for s in segs:
        n = len(s["tokens"])
        spd = frame_speed(s["trans"], dt, smooth)
        ab, info = angle_bins(s["trans"], spd, v_thresh, theta0_k)
        s["speed_med"] = float(np.median(spd))
        s["bin_info"] = info
        toks.append(s["tokens"])
        modes.append(np.full(n, FAM_ID[s["family"]], dtype=np.int64))
        bins.append(ab)
        bounds.append([off, off + n])
        stems.append(s["stem"])
        off += n
    return (np.concatenate(toks), np.concatenate(modes), np.concatenate(bins),
            np.asarray(bounds, dtype=np.int64), stems)


def oversample(tok, mode, ab, bounds, stems, fam_of):
    """族平衡：按族补齐到最大族帧预算，stem 排序轮转整段复制，末段截断。"""
    fam_frames = {}
    for i, st in enumerate(stems):
        f = fam_of[i]
        fam_frames.setdefault(f, 0)
        fam_frames[f] += int(bounds[i][1] - bounds[i][0])
    budget = max(fam_frames.values())
    seg_idx_by_fam = {}
    for i, st in enumerate(stems):
        seg_idx_by_fam.setdefault(fam_of[i], []).append(i)

    new_tok, new_mode, new_ab, new_bounds, new_meta = [tok], [mode], [ab], [], []
    off = len(tok)
    # 原始 84 段区间原样保留在头部
    for i, st in enumerate(stems):
        new_bounds.append([int(bounds[i][0]), int(bounds[i][1]), i, 0])
    copies_plan = {}
    for f, base in sorted(fam_frames.items()):
        need = budget - base
        if need <= 0:
            copies_plan[f] = {"budget": budget, "orig_frames": base,
                              "factor": 1.0, "copies": {}}
            continue
        idxs = sorted(seg_idx_by_fam[f])
        added = 0
        plan = {}
        while added < need:
            for i in idxs:
                if added >= need:
                    break
                n = int(bounds[i][1] - bounds[i][0])
                take = min(n, need - added)
                new_tok.append(tok[bounds[i][0]:bounds[i][0] + take])
                new_mode.append(mode[bounds[i][0]:bounds[i][0] + take])
                new_ab.append(ab[bounds[i][0]:bounds[i][0] + take])
                new_bounds.append([off, off + take, i, 1])
                off += take
                added += take
                plan[stems[i]] = plan.get(stems[i], 0) + 1
        copies_plan[f] = {"budget": budget, "orig_frames": base,
                          "factor": round(budget / base, 4), "copies": plan}
    tok2 = np.concatenate(new_tok)
    mode2 = np.concatenate(new_mode)
    ab2 = np.concatenate(new_ab)
    bounds2 = np.asarray([[b[0], b[1]] for b in new_bounds], dtype=np.int64)
    return tok2, mode2, ab2, bounds2, copies_plan, budget


def main():
    ap = argparse.ArgumentParser(description="D046: B4-lite VAE v1 输入构建")
    ap.add_argument("--manifest", default=os.path.expanduser(
        "~/ros2_data/apt_g1/data/ds_bones/g1_b4lite/manifest_v2.json"))
    ap.add_argument("--npz-dir", default=os.path.expanduser(
        "~/ros2_data/apt_g1/data/ds_bones/g1_b4lite/npz"))
    ap.add_argument("--out-dir", default=os.path.expanduser(
        "~/ros2_data/apt_g1/data/ds_bones/g1_b4lite/vae_v1"))
    ap.add_argument("--split", choices=["train", "dev", "tfam"], required=True)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--smooth", type=int, default=5, help="速度滑动平均帧数")
    ap.add_argument("--v-thresh", type=float, default=0.05, help="有效方向速度阈值 m/s")
    ap.add_argument("--theta0-k", type=int, default=10, help="theta0 圆均值取前 K 个有效帧")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    segs = load_split(args.manifest, args.npz_dir, args.split)
    print(f"[{args.split}] segments: {len(segs)}")
    assert len(segs) == {"train": 84, "dev": 20, "tfam": 22}[args.split], \
        f"段数不符预注册口径"

    tok, mode, ab, bounds, stems = build_arrays(
        segs, args.dt, args.smooth, args.v_thresh, args.theta0_k)
    fams = [s["family"] for s in segs]
    print(f"[{args.split}] frames: {len(tok)}  seg_bounds: {len(bounds)}")

    meta = {
        "experiment": "D046",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "manifest": args.manifest,
        "npz_dir": args.npz_dir,
        "split": args.split,
        "selection_rule": "manifest_v2 set_role + t_role 双过滤，排除 t_role=boundary_ref（train 86->84, dev 21->20）",
        "bin_rules": {
            "dt_s": args.dt, "speed_smooth_frames": args.smooth,
            "v_thresh_mps": args.v_thresh, "theta0_k": args.theta0_k,
            "n_dbins": 8, "bin_width_deg": 45.0,
            "bin0": "段首 theta0 ±22.5° = 前向（注意：与 exp_all3 bin4=前向 约定不同）",
            "low_speed_rule": "平滑速度<0.05 m/s 帧继承前一有效 bin，段首无有效帧则 bin0",
            "standing_rule": "全程无有效帧段全段 bin0（本集无此情况则为空表）",
            "theta0": f"前 {args.theta0_k} 个有效帧速度方向圆均值",
        },
        "family_order": FAMILIES,
        "family_census": {f: fams.count(f) for f in FAMILIES if fams.count(f)},
        "total_frames": int(len(tok)),
        "n_segments": len(stems),
        "angle_bin_histogram": {int(b): int(c) for b, c in
                                zip(*np.unique(ab, return_counts=True))},
        "mode_histogram": {int(m): int(c) for m, c in
                           zip(*np.unique(mode, return_counts=True))},
        "angle_bin_hist_by_family": {
            f: {int(b): int(c) for b, c in
                zip(*np.unique(ab[mode == FAM_ID[f]], return_counts=True))}
            for f in sorted(set(fams))},
        "segments": [
            {"stem": st, "family": fm, "frames": int(b[1] - b[0]),
             "interval": [int(b[0]), int(b[1])],
             "speed_med_mps": segs[i]["speed_med"],
             "standing": segs[i]["bin_info"]["standing_segment"]}
            for i, (st, fm, b) in enumerate(zip(stems, fams, bounds))],
        "standing_segments": [stems[i] for i, s in enumerate(segs)
                              if s["bin_info"]["standing_segment"]],
    }

    out_dir = args.out_dir
    if args.split == "train":
        tok2, mode2, ab2, bounds2, copies_plan, budget = oversample(
            tok, mode, ab, bounds, stems, fams)
        meta["oversampling"] = {
            "rule": "按族补齐到最大族帧预算；stem 排序轮转整段复制、末段截断到预算（确定性，无随机数）",
            "budget_frames": int(budget),
            "by_family": copies_plan,
            "frames_after": int(len(tok2)),
            "angle_bin_histogram_after": {int(b): int(c) for b, c in
                                          zip(*np.unique(ab2, return_counts=True))},
        }
        meta["final"] = {"frames": int(len(tok2)), "segments": len(bounds2)}
        # norm stats 只从 train 84 段原始帧（过采样前）算
        mean = tok.astype(np.float64).mean(0)
        std = tok.astype(np.float64).std(0)
        np.savez(os.path.join(out_dir, "norm_stats_d046.npz"),
                 mean=mean.astype(np.float32), std=std.astype(np.float32),
                 n_frames=np.int64(len(tok)),
                 stems=np.array(stems, dtype=object))
        meta["norm_stats"] = {
            "file": "norm_stats_d046.npz", "n_frames": int(len(tok)),
            "rule": "train 84 段原始帧（过采样前）token 逐维 mean/std",
            "mean_preview": [float(x) for x in mean[:4]],
            "std_preview": [float(x) for x in std[:4]],
            "std_min": float(std.min()), "std_max": float(std.max())}
        np.save(os.path.join(out_dir, "token.npy"), tok2)
        np.save(os.path.join(out_dir, "mode.npy"), mode2)
        np.save(os.path.join(out_dir, "angle_bin.npy"), ab2)
        np.save(os.path.join(out_dir, "segment_bounds.npy"), bounds2)
        print(f"[train] after oversample: frames={len(tok2)} segs={len(bounds2)}")
    else:
        sub = os.path.join(out_dir, {"dev": "dev", "tfam": "tfam"}[args.split])
        os.makedirs(sub, exist_ok=True)
        np.save(os.path.join(sub, "token.npy"), tok)
        np.save(os.path.join(sub, "mode.npy"), mode)
        np.save(os.path.join(sub, "angle_bin.npy"), ab)
        np.save(os.path.join(sub, "segment_bounds.npy"), bounds)
        out_dir = sub
    with open(os.path.join(out_dir, "build_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"[{args.split}] saved -> {out_dir}")
    print("bin hist:", meta["angle_bin_histogram"])
    print("mode hist:", meta["mode_histogram"])


if __name__ == "__main__":
    main()
