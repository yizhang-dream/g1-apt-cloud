"""D046 v2 训练输入构建：SONIC mode 词表 + UL 正交轴（v1 复用 + 增轴）。

读 build_b4lite_mode_map.py 的 candidates_v2.json（133 段 = 121 mode 段 + 12
intermediate 材料），npz 双目录解析（v1 g1_b4lite/npz/ 只读 + v2 专用
g1_b4lite_v2conv/npz/），产出 train_token_vae_e39_v2.py 的四件套
{token, mode_id, ul, angle_bin}.npy + segment_bounds + window_keep_mask +
norm_stats（只算训练段）。

与 v1（build_b4lite_vae_inputs.py，D046 v1 基线）的关系：
  - 复用其 frame_speed / angle_bins / wrap_pi 代码路径（import，单一实现）；
  - bin 规则同款：dt=0.02、5 帧平滑、v_thresh=0.05、theta0_k=10、bin0=段首前向；
  - mode 轴从 10 语义族改为 SONIC 27-mode 子集（embed idx 表落 build_meta）；
  - 新增 UL 轴（none=0/sym=1/asym=2，段级常量）；
  - intermediate 连通性材料：不参与 mode 预算平衡，窗口级 keep 概率 0.3
    （window_keep_mask.npy，seed=0，owner 倾向的低权重进训练落法）；
  - 过采样预算按 train mode 平衡（STEALTH_WALK 池薄将按因子复制，如实记录）。

用法（服务器 .venv_isaac python）：
  python build_b4lite_vae_inputs_v2.py                # train + dev + tfam 全建
产出 data/ds_bones/g1_b4lite/vae_inputs_v2/：
  {token,mode_id,ul,angle_bin,segment_bounds,window_keep_mask}.npy
  norm_stats_d046v2.npz  build_meta.json  dev/  tfam/
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np

# v1 代码路径复用（帧速度/方向 bin/角度 wrap 单一实现，D046 v1 基线脚本）
from build_b4lite_vae_inputs import angle_bins, frame_speed

HOME = os.path.expanduser("~")
BASE = f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite"
DEFAULT_CAND = f"{BASE}/mode_map_v2/candidates_v2.json"
DEFAULT_NPZ_DIRS = [f"{BASE}/npz", f"{BASE}/../g1_b4lite_v2conv/npz"]
DEFAULT_OUT = f"{BASE}/vae_inputs_v2"

# embed idx 表：in-subset mode 按 hpp id 升序，intermediate 殿后（idx 最大）
INTERMEDIATE_NAME = "intermediate"
UL_ORDER = {"none": 0, "sym": 1, "asym": 2}


def mode_table(in_subset: list[str]) -> list[dict]:
    tabs = []
    for i, name in enumerate(sorted(in_subset, key=lambda x: MODE_HPPO[x])):
        tabs.append({"embed_idx": i, "mode_name": name,
                     "mode_id_hpp": MODE_HPPO[name]})
    tabs.append({"embed_idx": len(tabs), "mode_name": INTERMEDIATE_NAME,
                 "mode_id_hpp": 100})
    return tabs


MODE_HPPO = {
    "SLOW_WALK": 1, "WALK": 2, "RUN": 3, "FORWARD_JUMP": 17,
    "STEALTH_WALK": 18, "INJURED_WALK": 19, "LEDGE_WALKING": 20,
    "HAPPY_DANCE_WALK": 23,
}


def load_npz(stem: str, npz_dirs: list[str]):
    for d in npz_dirs:
        p = os.path.join(d, stem + ".npz")
        if os.path.isfile(p):
            z = np.load(p)
            return z["tokens"].astype(np.float32), z["trans_m"].astype(np.float64), p
    raise FileNotFoundError(stem)


def build_split(segs: list[dict], args, split: str) -> dict:
    toks, modes, uls, bins, bounds, stems = [], [], [], [], [], []
    materials = []
    off = 0
    for s in segs:
        tok, trans, path = load_npz(s["stem"], args.npz_dirs)
        n = len(tok)
        spd = frame_speed(trans, args.dt, args.smooth)
        ab, info = angle_bins(trans, spd, args.v_thresh, args.theta0_k)
        toks.append(tok)
        modes.append(np.full(n, s["embed_idx"], dtype=np.int64))
        uls.append(np.full(n, UL_ORDER[s["ul"]], dtype=np.int64))
        bins.append(ab)
        bounds.append([off, off + n])
        stems.append(s["stem"])
        materials.append(1 if s.get("material") == "intermediate" else 0)
        s["n_frames"] = n
        s["npz_path"] = path
        s["speed_med"] = round(float(np.median(spd)), 4)
        s["bin_info"] = {k: v for k, v in info.items() if k != "theta0_rad"}
        off += n
    return {"token": np.concatenate(toks), "mode_id": np.concatenate(modes),
            "ul": np.concatenate(uls), "angle_bin": np.concatenate(bins),
            "bounds": np.asarray(bounds, dtype=np.int64), "stems": stems,
            "materials": np.asarray(materials, dtype=np.int64)}


def oversample_by_mode(arrs: dict, table: list[dict]) -> tuple[dict, dict]:
    """v1 同款预算平铺：按 train mode 补齐到最大 mode 帧预算（stem 排序轮转整段复制）。"""
    tok, mode, ul, ab = arrs["token"], arrs["mode_id"], arrs["ul"], arrs["angle_bin"]
    bounds, stems, materials = arrs["bounds"], arrs["stems"], arrs["materials"]
    mode_frames: dict[int, int] = {}
    seg_idx: dict[int, list[int]] = {}
    for i, st in enumerate(stems):
        if materials[i]:
            continue  # intermediate 不参与预算
        m = int(mode[bounds[i][0]])
        mode_frames[m] = mode_frames.get(m, 0) + int(bounds[i][1] - bounds[i][0])
        seg_idx.setdefault(m, []).append(i)
    budget = max(mode_frames.values())
    new_tok, new_mode, new_ul, new_ab = [tok], [mode], [ul], [ab]
    new_bounds = [[int(b[0]), int(b[1])] for b in bounds]
    new_materials = [int(v) for v in materials]
    off = len(tok)
    copies_plan = {}
    for m in sorted(mode_frames):
        base = mode_frames[m]
        need = budget - base
        plan = {"budget": budget, "orig_frames": base, "factor": 1.0, "copies": {}}
        if need > 0:
            idxs = sorted(seg_idx[m])
            added = 0
            while added < need:
                for i in idxs:
                    if added >= need:
                        break
                    n = int(bounds[i][1] - bounds[i][0])
                    take = min(n, need - added)
                    new_tok.append(tok[bounds[i][0]:bounds[i][0] + take])
                    new_mode.append(mode[bounds[i][0]:bounds[i][0] + take])
                    new_ul.append(ul[bounds[i][0]:bounds[i][0] + take])
                    new_ab.append(ab[bounds[i][0]:bounds[i][0] + take])
                    new_bounds.append([off, off + take])
                    new_materials.append(int(materials[i]))
                    off += take
                    added += take
                    plan["copies"][stems[i]] = plan["copies"].get(stems[i], 0) + 1
            plan["factor"] = round(budget / base, 4)
        copies_plan[str(m)] = plan
    return {"token": np.concatenate(new_tok), "mode_id": np.concatenate(new_mode),
            "ul": np.concatenate(new_ul), "angle_bin": np.concatenate(new_ab),
            "bounds": np.asarray(new_bounds, dtype=np.int64),
            "stems": stems, "materials": np.asarray(new_materials, dtype=np.int64),
            "n_orig_segments": len(stems)}, copies_plan


def window_keep_mask(bounds: np.ndarray, materials: np.ndarray,
                     window: int, keep_p: float, seed: int) -> np.ndarray:
    """窗口 keep 掩码：intermediate 段窗口以 keep_p 概率保留（0.3 权重落法）。

    窗口枚举与训练侧一致：段 i 的有效窗口 = n_i-(window-1) 个，段内顺序编号，
    段间拼接（不跨段）。
    """
    rng = np.random.default_rng(seed)
    mask = []
    for i in range(len(bounds)):
        n = int(bounds[i][1] - bounds[i][0])
        w = max(n - (window - 1), 0)
        if materials[i]:
            mask.extend((rng.random(w) < keep_p).tolist())
        else:
            mask.extend([True] * w)
    return np.asarray(mask, dtype=bool)


def n_valid_windows(bounds: np.ndarray, window: int) -> int:
    return sum(max(int(b[1] - b[0]) - (window - 1), 0) for b in bounds)


def main() -> None:
    ap = argparse.ArgumentParser(description="D046 v2: VAE 输入构建（mode+UL 轴）")
    ap.add_argument("--cand", default=DEFAULT_CAND)
    ap.add_argument("--npz-dirs", nargs="+", default=DEFAULT_NPZ_DIRS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--v-thresh", type=float, default=0.05)
    ap.add_argument("--theta0-k", type=int, default=10)
    ap.add_argument("--inter-keep-p", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cand = json.load(open(args.cand))
    segs_all = cand["segments"]
    in_subset = cand["_meta_in_subset"] if "_meta_in_subset" in cand else None
    # candidates_v2.json 的 in_subset 在 quota_table 键里（铁定来源）
    in_subset = sorted(cand["quota_table"].keys(),
                       key=lambda n: MODE_HPPO.get(n, 999))
    table = mode_table(in_subset)
    idx_by_name = {t["mode_name"]: t["embed_idx"] for t in table}

    # 防御过滤：boundary_ref / excluded（当前池已排除，双保险）
    segs_all = [s for s in segs_all
                if s.get("t_role") != "boundary_ref" and not s.get("excluded")]
    for s in segs_all:
        s["embed_idx"] = idx_by_name[s["target"] if s["target"] != "intermediate"
                                     else "intermediate"]

    splits = {"train": [s for s in segs_all if s["set_role"] == "train"],
              "dev": [s for s in segs_all if s["set_role"] == "dev"],
              "tfam": [s for s in segs_all if s["t_role"] == "T-fam"]}
    assert splits["train"] and splits["dev"] and splits["tfam"]

    built = {}
    for name, segs in splits.items():
        segs = sorted(segs, key=lambda s: s["stem"])
        built[name] = build_split(segs, args, name)
        print(f"[{name}] segments={len(segs)} frames={len(built[name]['token'])}")

    # train 过采样（mode 预算平衡）
    train_os, copies_plan = oversample_by_mode(built["train"], table)
    print(f"[train-oversample] frames {len(built['train']['token'])} -> "
          f"{len(train_os['token'])}")

    # norm stats：只算训练段（过采样前原始帧）
    tok_tr = built["train"]["token"]
    mean, std = tok_tr.mean(0), tok_tr.std(0) + 1e-6
    np.savez(os.path.join(args.out_dir, "norm_stats_d046v2.npz"),
             mean=mean, std=std)

    keep = window_keep_mask(train_os["bounds"], train_os["materials"],
                            args.window, args.inter_keep_p, args.seed)
    n_inter_w = int(sum(
        max(int(train_os["bounds"][i][1] - train_os["bounds"][i][0])
            - (args.window - 1), 0)
        for i in range(len(train_os["bounds"])) if train_os["materials"][i]))
    print(f"[keep-mask] intermediate windows={n_inter_w} keep@{args.inter_keep_p} "
          f"-> {int(keep.sum())} of {len(keep)} total windows kept")

    def save_arrays(d: dict, out: str, with_mask: bool) -> None:
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "token.npy"), d["token"])
        np.save(os.path.join(out, "mode_id.npy"), d["mode_id"])
        np.save(os.path.join(out, "ul.npy"), d["ul"])
        np.save(os.path.join(out, "angle_bin.npy"), d["angle_bin"])
        np.save(os.path.join(out, "segment_bounds.npy"), d["bounds"])
        if with_mask:
            np.save(os.path.join(out, "window_keep_mask.npy"), keep)

    save_arrays(train_os, args.out_dir, with_mask=True)
    for name in ("dev", "tfam"):
        save_arrays(built[name], os.path.join(args.out_dir, name), with_mask=False)

    walk_idx = idx_by_name["WALK"]
    meta = {
        "experiment": "D046(v2)",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "candidates": args.cand,
        "npz_dirs": args.npz_dirs,
        "mode_table": table,
        "walk_embed_idx": walk_idx,
        "ul_order": UL_ORDER,
        "bin_rules": {"dt_s": args.dt, "speed_smooth_frames": args.smooth,
                      "v_thresh_mps": args.v_thresh, "theta0_k": args.theta0_k,
                      "n_dbins": 8, "bin0": "段首 theta0 ±22.5° = 前向（v1 同款）",
                      "low_speed_rule": "继承前一有效 bin，段首无有效帧则 bin0"},
        "inter_material": {"keep_p": args.inter_keep_p, "mask": "window_keep_mask.npy",
                            "rationale": "owner 倾向：intermediate 以 0.3 权重进训练保连通性"
                                         "（窗口级确定性 Bernoulli，seed=0）"},
        "oversample_plan": copies_plan,
        "window": args.window,
        "n_windows": {"train_total": n_valid_windows(train_os["bounds"], args.window),
                      "train_kept": int(keep.sum()),
                      "dev": n_valid_windows(built["dev"]["bounds"], args.window),
                      "tfam": n_valid_windows(built["tfam"]["bounds"], args.window)},
        "norm_stats": "norm_stats_d046v2.npz（只算训练段原始帧，过采样前；"
                      "E39 canonical 不做输入归一化，本统计为审计与探针用）",
        "per_split_census": {
            name: {str(i): int(c) for i, c in enumerate(np.bincount(
                [s["embed_idx"] for s in splits[name]],
                minlength=len(table)))}
            for name in splits},
        "segment_detail": {
            name: [{"stem": s["stem"], "target": s["target"],
                    "mode_id_hpp": s.get("mode_id"), "ul": s["ul"],
                    "set_role": s["set_role"], "t_role": s["t_role"],
                    "n_frames": s.get("n_frames"), "speed_med": s.get("speed_med"),
                    "npz_path": s.get("npz_path")} for s in splits[name]]
            for name in splits},
    }
    with open(os.path.join(args.out_dir, "build_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[write] {args.out_dir}")


if __name__ == "__main__":
    main()
