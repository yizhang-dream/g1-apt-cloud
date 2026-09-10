"""D048n: 速度锚定 vb 标签构建（B4-lite 语料，numpy-only，服务器秒级）。

把 E39/V21 VAE 输入目录（build_b4lite_vae_inputs_v2.py 产物 vae_inputs_v21/
等）的 vb 标签从「相位变化率三分位」（train_token_vae_e39.walk_phase_rate
内部自派生）替换为「实现速度三分位」：源 npz trans_m 逐帧差分 × fps。

口径（§5g 设计稿，D048n 预注册）：
  1. 段对齐：按 D046b-R3 确定性重放机制把每原始段的 trans_m 对齐到组装帧序
     ——优先读构建器落盘 segment_source.npy，同时用 holdout_ident.
     derive_segment_sources（oversample_plan 确定性重放）交叉校验（copies
     计数 + 逐段帧数 + 总帧数三重 checksum，不符即 raise）；副本段 = 源段
     帧前缀（截段复制），标签随帧逐位一致。
  2. 速度：v[t] = ||trans[t]-trans[t-1]|| × fps（后向差分；XY 平面，与
     D046 frame_speed 口径一致；--axes xyz 可切全 3D）；v[0]=v[1]（镜像
     E39 rate[0]=rate[1] 约定）。
  3. edges = 参考帧（mode==--ref-mode，默认 2；v21 语义 = SONIC WALK 前向
     walk 族，与 train 脚本 walk_phase_rate 的 mode==2 掩码同口径）速度的
     1/3、2/3 分位数；vb = clip(digitize(v, edges), 0, 2) 全帧（含非参考帧
     与 IDLE 等，E39 全帧 vb 同构）。
  4. 产物 <out-dir>/vb_speed.npy（(N,) int64）+ vb_speed_meta.json（edges/
     counts/label_policy/checksum 结果/源 npz md5 清单）；供
     train_token_vae_e39.py --vb-npy 消费（PCA/rate/pca.npz/dbin 落盘照旧，
     env 依赖不动）。

用法（服务器 .venv_isaac python，本脚本不依赖 torch）：
  python build_d048n_vb_speed_labels.py                 # 默认 v21 目录
  python build_d048n_vb_speed_labels.py --ref-mode 2    # 参考帧口径可改
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os

import numpy as np

# D046b-R3 现役重放函数（单一实现，probe_vae_cross v3.1 / 回归测试共用）
try:                      # 服务器执行根平铺 import / 仓库根包 import 双兼容
    from holdout_ident import derive_segment_sources
except ImportError:
    from apt_g1.holdout_ident import derive_segment_sources

HOME = os.path.expanduser("~")
DEFAULT_INPUTS = f"{HOME}/ros2_data/apt_g1/data/g1_b4lite/vae_inputs_v21"
# builder v2 同款双目录解析（v1 npz/ 只读 + v2conv 新转换段；段级 npz_path 优先）
DEFAULT_NPZ_DIRS = [f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite/npz",
                    f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite_v2conv/npz"]


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_frame(row: np.ndarray) -> str:
    """单帧 token 的 md5（float32 连续字节；与构建器 astype(np.float32) 同构）。"""
    return hashlib.md5(
        np.ascontiguousarray(row, dtype=np.float32).tobytes()).hexdigest()


def frame_speed_bwd(trans: np.ndarray, fps: float, axes: str) -> np.ndarray:
    """逐帧速度：后向差分 v[t]=||trans[t]-trans[t-1]||×fps，v[0]=v[1]。

    axes="xy" 取水平面（D046 frame_speed 同口径），"xyz" 取全 3D 范数。
    """
    t = trans if axes == "xyz" else trans[:, :2]
    v = np.zeros(len(trans), dtype=np.float64)
    if len(trans) >= 2:
        v[1:] = np.linalg.norm(np.diff(t, axis=0), axis=1) * fps
        v[0] = v[1]  # 镜像 E39 walk_phase_rate 的 rate[0]=rate[1]
    return v


def resolve_npz(stem: str, npz_dirs: list[str], npz_paths: dict[str, str]) -> str:
    """段 → 源 npz 路径：构建器 build_meta 记录的 npz_path 优先，否则按序搜目录。"""
    p = npz_paths.get(stem)
    if p and os.path.isfile(p):
        return p
    for d in npz_dirs:
        p = os.path.join(d, stem + ".npz")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"源段 npz 找不到: {stem}（npz_dirs={npz_dirs}）")


def align_segments(inputs_dir: str, npz_dirs: list[str]):
    """组装帧序 ← 源段 trans_m 确定性对齐。

    返回 (bounds, src_idx, orig_stems, seg_records, provenance)；任一 checksum
    不符 raise（不得静默错标——D046b-R3 纪律）。
    """
    bounds = np.load(os.path.join(inputs_dir, "segment_bounds.npy")).astype(np.int64)
    assert bounds.ndim == 2 and bounds.shape[1] == 2, "segment_bounds 形状异常"
    with open(os.path.join(inputs_dir, "build_meta.json"), encoding="utf-8") as f:
        build_meta = json.load(f)
    for k in ("mode_table", "oversample_plan", "segment_detail"):
        if k not in build_meta:
            raise RuntimeError(f"build_meta.json 缺 {k}——本脚本面向 "
                               "build_b4lite_vae_inputs_v2(v2/v21) 材料")

    # 确定性重放（v2 builder oversample_by_mode 同款；三重 checksum 内建）
    src_replay, orig_stems = derive_segment_sources(bounds, build_meta)
    ss_path = os.path.join(inputs_dir, "segment_source.npy")
    if os.path.isfile(ss_path):
        src_disk = np.load(ss_path).astype(np.int64)
        if len(src_disk) != len(bounds):
            raise RuntimeError("segment_source.npy 段数与 segment_bounds 不符")
        if not np.array_equal(src_disk, src_replay):
            raise RuntimeError("segment_source.npy 与 oversample_plan 重放不一致"
                               "（材料非当前构建器产物？）")
        provenance = "segment_source.npy(磁盘)×derive_segment_sources(重放)双验证一致"
    else:
        provenance = "derive_segment_sources(oversample_plan 确定性重放；无磁盘 segment_source.npy)"
    src_idx = src_replay

    npz_paths = {d["stem"]: d.get("npz_path") or ""
                 for d in build_meta.get("segment_detail", {}).get("train", [])}
    n_orig = len(orig_stems)
    seg_records, trans_cache = [], {}
    for j in range(len(bounds)):
        i = int(src_idx[j])
        n_j = int(bounds[j, 1] - bounds[j, 0])
        stem = orig_stems[i]
        if stem not in trans_cache:
            z = np.load(resolve_npz(stem, npz_dirs, npz_paths))
            trans_cache[stem] = (z["trans_m"].astype(np.float64),
                                 z["tokens"].astype(np.float32))
        trans_i, _ = trans_cache[stem]
        if j < n_orig:
            assert n_j == len(trans_i), f"原始段 {stem} 帧数 {len(trans_i)} != bounds {n_j}"
            kind = "orig"
        else:
            assert n_j <= len(trans_i), f"副本段 {stem} 越界（{n_j} > 源 {len(trans_i)}）"
            kind = "copy"
        seg_records.append({"seg": j, "src": i, "stem": stem, "kind": kind,
                            "frames": n_j,
                            "interval": [int(bounds[j, 0]), int(bounds[j, 1])]})
    return bounds, src_idx, orig_stems, seg_records, provenance, trans_cache, \
        npz_paths, build_meta


def token_md5_spot_check(seg_records: list[dict], trans_cache: dict,
                         token: np.ndarray, bounds: np.ndarray) -> list[dict]:
    """tokens 逐帧 md5 抽查：首原始段 / 中部原始段 / 首副本段 / 末段（截断副本）。"""
    n_orig = sum(1 for r in seg_records if r["kind"] == "orig")
    first_copy = next((r["seg"] for r in seg_records if r["kind"] == "copy"), None)
    picks = [0, n_orig // 2, first_copy, len(seg_records) - 1]
    picks = sorted({j for j in picks if j is not None})
    results = []
    for j in picks:
        r = seg_records[j]
        s0, s1 = int(bounds[j, 0]), int(bounds[j, 1])
        _, toks_src = trans_cache[r["stem"]]
        n_check = s1 - s0
        for k in range(n_check):
            if md5_frame(token[s0 + k]) != md5_frame(toks_src[k]):
                raise RuntimeError(
                    f"token md5 抽查失败：段 {j}（{r['stem']}）帧 {k} 组装帧与源 npz 不一致")
        results.append({"seg": j, "stem": r["stem"], "kind": r["kind"],
                        "frames_checked": n_check, "match": True})
    return results


def main():
    ap = argparse.ArgumentParser(
        description="D048n: 速度锚定 vb 标签构建（trans_m 差分×fps → 参考帧三分位）")
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
                    help="参考帧 mode id（默认 2；v21=SONIC WALK 前向 walk 族，"
                         "与 train 脚本 mode==2 rate 边界口径一致；v1 语义="
                         "fast_walk_run）")
    ap.add_argument("--fps", type=float, default=50.0, help="采样率（B4-lite=50Hz）")
    ap.add_argument("--axes", choices=["xy", "xyz"], default="xy",
                    help="速度差分维度（默认 xy 水平面，D046 frame_speed 同口径）")
    args = ap.parse_args()

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

    # 段级速度 → 组装帧序；全帧拼成 (N,)
    v = np.zeros(n, dtype=np.float64)
    for r in seg_records:
        s0, s1 = r["interval"]
        trans_i, _ = trans_cache[r["stem"]]
        v[s0:s1] = frame_speed_bwd(trans_i[:s1 - s0], args.fps, args.axes)

    # 参考帧分位 edges → 全帧 digitize（E39 同构：quantile + clip(digitize)）
    ref_mask = mode == args.ref_mode
    n_ref = int(ref_mask.sum())
    assert n_ref > 0, f"参考帧为空：mode=={args.ref_mode} 无帧（--ref-mode 口径错误？）"
    edges = np.quantile(v[ref_mask], [1.0 / 3.0, 2.0 / 3.0])
    vb = np.clip(np.digitize(v, edges), 0, 2).astype(np.int64)
    counts = {int(b): int(c) for b, c in zip(*np.unique(vb, return_counts=True))}
    print(f"[vb] ref_frames={n_ref} edges={edges.tolist()} counts={counts}")

    spot = token_md5_spot_check(seg_records, trans_cache, token, bounds)
    print(f"[md5] token 逐帧抽查 {len(spot)} 段全 match: "
          + ", ".join(f"{r['stem']}({r['kind']},{r['frames_checked']}f)" for r in spot))

    # ref-mode 语义标注（v21: mode_table 表；v1: family_order 表）——只记录不裁定
    ref_mode_name = None
    if "mode_table" in build_meta:
        for t in build_meta["mode_table"]:
            if int(t["embed_idx"]) == args.ref_mode:
                ref_mode_name = t["mode_name"]
    elif "family_order" in build_meta:
        if 0 <= args.ref_mode < len(build_meta["family_order"]):
            ref_mode_name = build_meta["family_order"][args.ref_mode]

    used_files = sorted({resolve_npz(r["stem"], args.npz_dir, npz_paths)
                         for r in seg_records})
    meta = {
        "experiment": "D048n",
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
        "label_policy": "vb=clip(digitize(v,edges),0,2) 全帧；edges=参考帧"
                        "(mode==ref_mode)速度 1/3、2/3 分位；替代 train 内部"
                        "相位 rate 分位标签（对照臂仍用内部标签，e39 架构/超参冻结）",
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
