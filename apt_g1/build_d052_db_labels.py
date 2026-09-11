"""D052 G2: db 方向标签构建（beta 参考系 8 档，B4-lite 语料，numpy-only）。

预注册：refine-logs/DS_CONTINUOUS_EXECUTION_PLAN.md §5l（D052，G1 已裁定
**beta 参考系**：beta=wrap_pi(运动方向 ang−机体 yaw)，bin=floor((beta+π)/(2π)·8)
%8（env L854-855 同构），bin4=前向；低速帧 v<0.05 继承前一有效 bin、段首无
有效帧=bin4）。把 E39/V21 VAE 输入目录的现役 angle_bin.npy（theta0 锚定，
D052-G0 已证与闭环前向命令语义错位）替换为 C 版 db 标签，供
train_token_vae_e39.py --db-npy 消费（G2 重训 dirA VAE：vb 沿用 speedA 标签）。

实现（与 G0 审计 build_d052_db_audit.py 的 C 版**逐位同构**——计算函数
import 复用单一实现，不复制第三份）：
  1. 段对齐：build_d048n.align_segments 确定性三重 checksum 链（D046b-R3
     纪律，不符即 raise）+ resolve_npz/md5_file/token_md5_spot_check 同源复用。
  2. 逐段：完整源段上算 ang=direction_ang(trans)（构建器 L83-87 差分窗口
     同构）、yaw=quat_yaw_wxyz(quat_wxyz)（D044 i:zyx 口径）、
     beta=wrap_pi(ang−yaw)、raw=abs_dir_bin(beta)，按 bounds 前缀截取组装
     帧序——副本段携带方式与构建器/审计逐位同构。
  3. v=frame_speed_bwd(前缀, fps, "xy")（后向差分，v[0]=v[1]），有效帧
     v≥--v-thresh（默认 0.05 m/s）；全帧 db=bins_with_inherit(raw 前缀,
     valid, start_bin=4)（C 版段首锚 bin4，非现役 bin0）。
  4. 产物 <out-dir>/db_labels.npy ((N,) int64) + db_labels_meta.json：bin
     公式/参考系/继承规则/段首锚/直方图（全帧+有效帧）/vs 现役 angle_bin
     差异摘要（预期大差异=契约改动本身，如实落盘不判错）/输入输出 md5。

用法（服务器 .venv_isaac 或本机 python，不依赖 torch）：
  python build_d052_db_labels.py                       # 默认 v21+双 npz 目录
  python build_d052_db_labels.py --v-thresh 0.05       # 有效帧阈值可改
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np

# 计算函数 import 复用（单一实现，不复制）：段对齐/checksum 链 from D048n，
# beta 参考系 C 版语义（ang/yaw/bin 公式/继承规则/段首锚）from D052 审计，
# wrap_pi from D046 构建器（审计同源）
try:                      # 服务器执行根平铺 import / 仓库根包 import 双兼容
    from build_d048n_vb_speed_labels import (align_segments, frame_speed_bwd,
                                             resolve_npz, md5_file,
                                             token_md5_spot_check)
    from build_d052_db_audit import (FWD_BIN, N_BINS, abs_dir_bin,
                                     bins_with_inherit, direction_ang,
                                     hist8, load_quat_cache, quat_yaw_wxyz)
    from build_b4lite_vae_inputs import wrap_pi
except ImportError:
    from apt_g1.build_d048n_vb_speed_labels import (align_segments,
                                                    frame_speed_bwd,
                                                    resolve_npz, md5_file,
                                                    token_md5_spot_check)
    from apt_g1.build_d052_db_audit import (FWD_BIN, N_BINS, abs_dir_bin,
                                            bins_with_inherit, direction_ang,
                                            hist8, load_quat_cache,
                                            quat_yaw_wxyz)
    from apt_g1.build_b4lite_vae_inputs import wrap_pi

HOME = os.path.expanduser("~")
DEFAULT_INPUTS = f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite/vae_inputs_v21"
# builder v2 同款双目录解析（D048n DEFAULT_NPZ_DIRS 同款；段级 npz_path 优先）
DEFAULT_NPZ_DIRS = [f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite/npz",
                    f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite_v2conv/npz"]


def main():
    ap = argparse.ArgumentParser(
        description="D052 G2: db 方向标签构建（beta 参考系 8 档，bin4=前向，"
                    "低速继承/段首锚 bin4）")
    ap.add_argument("--inputs-dir", default=DEFAULT_INPUTS,
                    help="VAE 组装输入目录（token/angle_bin/segment_bounds/"
                         "build_meta.json）")
    ap.add_argument("--npz-dirs", nargs="+", default=DEFAULT_NPZ_DIRS,
                    help="源 npz 目录（trans_m+quat_wxyz；先到先得，"
                         "build_meta 段级 npz_path 优先）")
    ap.add_argument("--out-dir", default="",
                    help="输出目录（默认 = inputs-dir，写 db_labels.npy + "
                         "db_labels_meta.json）")
    ap.add_argument("--fps", type=float, default=50.0, help="采样率（B4-lite=50Hz）")
    ap.add_argument("--v-thresh", type=float, default=0.05,
                    help="有效帧速度阈值 m/s（站立帧 = v < 该值，继承前一有效 bin）")
    args = ap.parse_args()

    inputs_dir = args.inputs_dir
    out_dir = args.out_dir or inputs_dir
    token = np.load(os.path.join(inputs_dir, "token.npy")).astype(np.float32)
    disk_ab = np.load(os.path.join(inputs_dir, "angle_bin.npy")).astype(np.int64)
    n = len(token)
    assert len(disk_ab) == n, \
        f"angle_bin.npy 长度 {len(disk_ab)} != token N {n}"

    print(f"[load] {inputs_dir}: N={n}")
    (bounds, src_idx, orig_stems, seg_records, provenance, trans_cache,
     npz_paths, build_meta) = align_segments(inputs_dir, args.npz_dirs)
    assert int(bounds[-1, 1]) == n, f"bounds 总帧数 {bounds[-1, 1]} != token N {n}"

    # 逐段：完整源段上算 ang/yaw/beta/raw → 前缀截取 → 继承规则拼组装帧序
    quat_cache: dict[str, np.ndarray] = {}
    db = np.zeros(n, dtype=np.int64)
    v_all = np.zeros(n, dtype=np.float64)
    for r in seg_records:
        s0, s1 = r["interval"]
        stem, take = r["stem"], r["frames"]
        trans_full = trans_cache[stem][0]
        quat_full = load_quat_cache(stem, args.npz_dirs, npz_paths, quat_cache)

        ang_full = direction_ang(trans_full)            # 世界系运动方向
        yaw_full = quat_yaw_wxyz(quat_full)             # 机体 yaw（i:zyx）
        beta_full = wrap_pi(ang_full - yaw_full)        # 机体偏差角
        raw = abs_dir_bin(beta_full)                    # C 版公式（env 同构）

        v = frame_speed_bwd(trans_full[:take], args.fps, "xy")
        valid = v >= args.v_thresh
        db[s0:s1] = bins_with_inherit(raw[:take], valid, FWD_BIN)  # 段首锚 bin4
        v_all[s0:s1] = v

    valid_all = v_all >= args.v_thresh
    n_valid = int(valid_all.sum())
    counts_all = hist8(db)
    counts_valid = hist8(db[valid_all])
    print(f"[db] N={n} valid={n_valid} ({n_valid / n * 100:.2f}%) "
          f"counts_all={counts_all['counts']} counts_valid={counts_valid['counts']}")

    spot = token_md5_spot_check(seg_records, trans_cache, token, bounds)
    print(f"[md5] token 逐帧抽查 {len(spot)} 段全 match: "
          + ", ".join(f"{r['stem']}({r['kind']},{r['frames_checked']}f)" for r in spot))

    # vs 现役 angle_bin 差异统计（预期大差异=theta0→beta 契约改动本身，如实落盘）
    diff_mask = db != disk_ab
    n_diff = int(diff_mask.sum())
    conf: dict[str, int] = {}
    for d, rr in zip(disk_ab[diff_mask], db[diff_mask]):
        key = f"angle_bin{int(d)}->db{int(rr)}"
        conf[key] = conf.get(key, 0) + 1
    diff_block = {
        "target": "现役 angle_bin.npy（theta0 锚定版，D052-G0 判定前向语义错位）",
        "mismatch_frames": n_diff, "n_frames": n,
        "diff_rate": round(n_diff / n, 6) if n else 0.0,
        "expected": "大差异=契约改动本身（beta 参考系替换 theta0 锚定），"
                    "非错误、不设上限门",
        "confusion_top": dict(sorted(conf.items(), key=lambda kv: -kv[1])[:16]),
    }
    print(f"[diff] vs angle_bin: {n_diff}/{n} ({diff_block['diff_rate'] * 100:.2f}%) "
          f"top={dict(list(diff_block['confusion_top'].items())[:4])}")

    used_files = sorted({resolve_npz(r["stem"], args.npz_dirs, npz_paths)
                         for r in seg_records})
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, "db_labels.npy")
    np.save(npy_path, db)
    meta = {
        "experiment": "D052-G2",
        "title": "db 方向标签构建（beta 参考系 8 档，bin4=前向）",
        "prereg": "refine-logs/DS_CONTINUOUS_EXECUTION_PLAN.md §5l（G1 裁定 "
                  "beta 参考系）",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "inputs_dir": inputs_dir,
        "npz_dirs": args.npz_dirs,
        "n_frames": n,
        "n_segments_out": len(bounds),
        "n_segments_orig": len(orig_stems),
        "n_valid_frames": n_valid,
        "fps": args.fps,
        "v_thresh_mps": args.v_thresh,
        "reference_frame": "beta=wrap_pi(ang−yaw)（机体偏差角；ang=中心差分运动"
                           "方向角、yaw=quat_wxyz ZYX 提取 i:zyx 口径）",
        "bin_rule": "db=floor((beta+π)/(2π)·8) mod 8（apt_flat_env.py L854-855 "
                    "env 侧公式同构）；bin4=前向（beta=0→bin4 语义锚由 "
                    "build_d052_db_audit 模块级 assert 强制）",
        "inherit_rule": "低速帧（v<%s m/s）继承前一有效 bin；段首无有效帧=bin4"
                        "（C 版语义锚，非现役 bin0）；v=frame_speed_bwd 后向差分"
                        "×%sHz XY、v[0]=v[1]" % (args.v_thresh, args.fps),
        "start_bin": FWD_BIN,
        "n_bins": N_BINS,
        "segment_note": "ang/yaw/beta/raw 在完整源段上算、按 bounds 前缀截取"
                        "（副本段携带方式与构建器/审计逐位同构）",
        "label_policy": "db_labels.npy 全帧替换现役 angle_bin.npy（train "
                        "--db-npy 消费）；与 D048n speedA vb 标签独立成轴",
        "histograms": {
            "all_frames": counts_all,
            "valid_frames": counts_valid,
        },
        "diff_vs_angle_bin": diff_block,
        "alignment_provenance": provenance,
        "checksums": {
            "n_segments": {"bounds": len(bounds), "src_idx": len(src_idx),
                           "match": len(bounds) == len(src_idx)},
            "copies_per_source": {orig_stems[i]: int(c) for i, c in
                                  zip(*np.unique(src_idx, return_counts=True))},
            "total_frames": {"bounds_last": int(bounds[-1, 1]),
                             "token_n": n, "db_n": len(db),
                             "match": int(bounds[-1, 1]) == n == len(db)},
            "token_md5_spot_checks": spot,
        },
        "source_npz_md5": {os.path.splitext(os.path.basename(p))[0]: md5_file(p)
                           for p in used_files},
        "inputs_md5": {f: md5_file(os.path.join(inputs_dir, f))
                       for f in ["token.npy", "angle_bin.npy",
                                 "segment_bounds.npy", "segment_source.npy",
                                 "build_meta.json"]
                       if os.path.isfile(os.path.join(inputs_dir, f))},
        "output_md5": {"db_labels.npy": md5_file(npy_path)},
        "outputs": ["db_labels.npy", "db_labels_meta.json"],
    }
    with open(os.path.join(out_dir, "db_labels_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[write] {npy_path} (N={n}) + {out_dir}/db_labels_meta.json")


if __name__ == "__main__":
    main()
