"""D052 G0：B4-lite 语料方向审计（三参考系 8 档直方图，零训练，numpy-only）。

预注册：refine-logs/DS_CONTINUOUS_EXECUTION_PLAN.md §5l（D052，2026-09-12 立项）。
输入 vae_inputs_v21/ + 双 npz 目录（trans_m+quat_wxyz），逐段逐帧算三参考系
8 档直方图，为 db 方向轴语义重建（G1 契约统一 → G2 标签重训）提供材料判定：
  - A 现役 theta0 锚定版：逐字复刻 build_b4lite_vae_inputs.angle_bins（import
    单一实现不复制），并 vs 磁盘 angle_bin.npy 逐帧对账（差异率 >1% =
    REPLICATE_MISMATCH，退出码 2——复刻或对齐有 bug，不得用于判读）。
  - B 世界系绝对角版：bin=floor((ang+π)/(2π)·8) mod 8（apt_flat_env.py
    L854-855 env 侧公式同构）。
  - C 机体偏差角 beta 版（首选）：beta=wrap_pi(ang−yaw)，同 B 公式作用在
    beta 上；内置 assert beta=0 → bin4（C 版语义锚自检，bin4=前向）。

口径（§5l G0 冻结）：
  1. 段对齐：build_d048n.align_segments 确定性三重 checksum 链（D046b-R3
     纪律，不符即 raise）；ang/yaw/beta/A 版均在完整源段上算、按 bounds 前缀
     截取组装帧序——与构建器标签携带方式逐位同构（副本段末帧含源段中心差分
     前瞻，忠实复刻现役 angle_bin 携带路径）。
  2. 速度 v：frame_speed_bwd（后向差分 ×50Hz，XY 平面，v[0]=v[1]）；
     有效帧 = v ≥ 0.05 m/s（--v-thresh 可改）。三版「有效帧」直方图共用此
     掩码（可横向对比）；A 版「全帧」直方图 = 构建器原样输出（含其自带
     平滑速度掩码与继承规则）。
  3. ang：中心差分方向角（同 build_b4lite_vae_inputs.angle_bins L83-87，含
     端点一阶差分）；yaw：quat_wxyz → atan2(2(wz+xy), 1−2(y²+z²))（D044
     已定 i:zyx 口径的 ZYX yaw 提取，与 isaac/eval_apt_isaac.py L387 同式；
     首段前 5 帧四元数+yaw spot check 打印）。
  4. 直方图继承规则：全帧版低速帧（v<0.05）继承前一有效帧 bin；段首无有效
     帧——A 版=bin0（现役规则），B/C 版=bin4（§5l G1 契约：维持前向语义
     非现役 bin0，差异入账 meta）。全程无有效帧段：A 版全段 bin0（构建器
     standing 规则）、B/C 版全段 bin4。
  5. 判据（§5l 两级材料判定，只计算打印不替判读）：
     L1 db4 前向档语义恢复（联合门必需）= C 版有效帧 db4 占比 ≥15% 且
        站立帧（v<0.05）全语料占比 ≤50%；
     L2 8 方位全域语义恢复 = C 版有效档（有效帧占比 ≥1%）数 ≥5/8；
     B 版同判据作对照。

用法（服务器 .venv_isaac 或本机 python，不依赖 torch）：
  python build_d052_db_audit.py --out ~/ros2_data/apt_g1/data/ds_bones/g1_b4lite/d052_g0_direction_audit.json
  python build_d052_db_audit.py --repo-hash <git hash> --out <json>
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys

import numpy as np

# import 复用（单一实现，不复制）：段对齐/速度/checksum 链 from D048n，
# theta0 锚定版方向 bin + 平滑速度 from D046 现役构建器
try:                      # 服务器执行根平铺 import / 仓库根包 import 双兼容
    from build_d048n_vb_speed_labels import (align_segments, frame_speed_bwd,
                                             resolve_npz, md5_file,
                                             token_md5_spot_check)
    from build_b4lite_vae_inputs import angle_bins, frame_speed, wrap_pi
except ImportError:
    from apt_g1.build_d048n_vb_speed_labels import (align_segments,
                                                    frame_speed_bwd,
                                                    resolve_npz, md5_file,
                                                    token_md5_spot_check)
    from apt_g1.build_b4lite_vae_inputs import angle_bins, frame_speed, wrap_pi

HOME = os.path.expanduser("~")
DEFAULT_INPUTS = f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite/vae_inputs_v21"
# builder v2 同款双目录解析（D048n DEFAULT_NPZ_DIRS 同款；段级 npz_path 优先）
DEFAULT_NPZ_DIRS = [f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite/npz",
                    f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite_v2conv/npz"]
DEFAULT_OUT = f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite/d052_g0_direction_audit.json"

N_BINS = 8
RECON_DIFF_LIMIT = 0.01  # A 版对账差异率上限（>1% = REPLICATE_MISMATCH，退出码 2）
STRAIGHT_DEG = 22.5      # 直线度：|beta| < 22.5°
SIDEBACK_DEG = 45.0      # 侧后移：|beta| >= 45°
FWD_BIN = 4              # 前向档（env 侧语义；B/C 版段首锚）


def quat_yaw_wxyz(quat: np.ndarray) -> np.ndarray:
    """quat_wxyz (N,4) → 机体 yaw (N,)（rad，世界系）。

    D044 已定 i:zyx（R=Rz@Ry@Rx）口径的 ZYX yaw 提取，与
    isaac/eval_apt_isaac.py L387 / replay_token_speed_d048r.py L337 同式。
    """
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def direction_ang(trans: np.ndarray) -> np.ndarray:
    """世界系运动方向角：中心差分 d[t]=trans[t+1]−trans[t−1]，端点一阶差分。

    逐字复刻 build_b4lite_vae_inputs.angle_bins L83-87 的差分窗口（仅提出
    ang；该文件逻辑零改动）。
    """
    t = np.asarray(trans, dtype=np.float64)
    d = np.zeros_like(t)
    if len(t) >= 2:
        d[1:-1] = t[2:] - t[:-2]
        d[0] = t[1] - t[0]
        d[-1] = t[-1] - t[-2]
    return np.arctan2(d[:, 1], d[:, 0])


def abs_dir_bin(x: np.ndarray) -> np.ndarray:
    """env 侧公式同构（apt_flat_env.py L854-855）：floor((x+π)/(2π)·8) mod 8。"""
    return np.floor((np.asarray(x, dtype=np.float64) + math.pi)
                    / (2.0 * math.pi) * float(N_BINS)).astype(np.int64) % N_BINS


# C 版语义锚自检：beta=0（机体前向）必须落 bin4（§5l G1：bin4=前向）
assert int(abs_dir_bin(np.array([0.0]))[0]) == FWD_BIN, "beta=0 → bin4 语义锚自检失败"


def bins_with_inherit(raw_bins: np.ndarray, valid: np.ndarray,
                      start_bin: int) -> np.ndarray:
    """全帧 bin：有效帧取公式 bin，低速帧继承前一有效 bin，段首锚 start_bin。"""
    out = np.empty(len(raw_bins), dtype=np.int64)
    last = int(start_bin)
    for i in range(len(raw_bins)):
        if valid[i]:
            last = int(raw_bins[i])
        out[i] = last
    return out


def hist8(bins: np.ndarray) -> dict:
    """8 档直方图（counts + frac，bin 0-7 全列）。"""
    c = np.bincount(np.asarray(bins, dtype=np.int64), minlength=N_BINS)
    n = int(c.sum())
    return {"counts": [int(v) for v in c],
            "frac": [round(float(v) / n, 6) if n else 0.0 for v in c]}


def load_quat_cache(stem: str, npz_dirs: list[str], npz_paths: dict,
                    cache: dict) -> np.ndarray:
    """源段 quat_wxyz (L,4) 惰性缓存（trans_cache 只缓存 trans/tokens，此处补 quat）。"""
    if stem not in cache:
        z = np.load(resolve_npz(stem, npz_dirs, npz_paths))
        if "quat_wxyz" not in z:
            raise KeyError(f"源段 {stem} npz 缺 quat_wxyz 字段（本审计必需）")
        cache[stem] = z["quat_wxyz"].astype(np.float64)
    return cache[stem]


def effective_bins(hist: dict, n_valid: int) -> tuple[int, list[int]]:
    """有效档数：有效帧占比 ≥1% 的 bin 个数（§5l 判据口径）。"""
    if n_valid <= 0:
        return 0, []
    ids = [b for b in range(N_BINS) if hist["frac"][b] >= 0.01]
    return len(ids), ids


def main():
    ap = argparse.ArgumentParser(
        description="D052 G0: B4-lite 语料方向审计（三参考系 8 档直方图+现役对账）")
    ap.add_argument("--inputs-dir", default=DEFAULT_INPUTS,
                    help="VAE 组装输入目录（angle_bin/segment_bounds/"
                         "segment_source/build_meta.json）")
    ap.add_argument("--npz-dirs", nargs="+", default=DEFAULT_NPZ_DIRS,
                    help="源 npz 目录（trans_m+quat_wxyz；先到先得，"
                         "build_meta 段级 npz_path 优先）")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 JSON 路径")
    ap.add_argument("--fps", type=float, default=50.0, help="采样率（B4-lite=50Hz）")
    ap.add_argument("--v-thresh", type=float, default=0.05,
                    help="有效帧速度阈值 m/s（站立帧 = v < 该值）")
    ap.add_argument("--dt", type=float, default=0.02,
                    help="A 版复刻：构建器 frame_speed dt（现役默认 0.02）")
    ap.add_argument("--smooth", type=int, default=5,
                    help="A 版复刻：构建器速度平滑帧数（现役默认 5）")
    ap.add_argument("--theta0-k", type=int, default=10,
                    help="A 版复刻：theta0 圆均值前 K 有效帧（现役默认 10）")
    ap.add_argument("--repo-hash", default="",
                    help="脚本仓 git hash（可选，记录进 meta）")
    args = ap.parse_args()

    print(f"[D052-G0] inputs={args.inputs_dir}")
    print(f"[D052-G0] npz_dirs={args.npz_dirs}")

    # ---- 段对齐（D048n 确定性三重 checksum 链，不符即 raise）+ tokens md5 抽查
    (bounds, src_idx, orig_stems, seg_records, provenance, trans_cache,
     npz_paths, build_meta) = align_segments(args.inputs_dir, args.npz_dirs)
    disk_ab = np.load(os.path.join(args.inputs_dir, "angle_bin.npy")).astype(np.int64)
    n_total = int(bounds[-1, 1])
    assert len(disk_ab) == n_total, \
        f"angle_bin.npy 长度 {len(disk_ab)} != bounds 总帧数 {n_total}"
    token_path = os.path.join(args.inputs_dir, "token.npy")
    if os.path.isfile(token_path):
        assert len(np.load(token_path)) == n_total, "token.npy 长度与 bounds 不符"
    print(f"[align] N={n_total} segs={len(bounds)}（orig={len(orig_stems)}）"
          f" provenance={provenance}")

    # ---- 逐段计算：v / ang / yaw / beta / 三版 bin（完整源段上算 → 前缀截取）
    quat_cache: dict[str, np.ndarray] = {}
    g = {k: np.zeros(n_total, dtype=np.int64)
         for k in ("A", "B_all", "C_all", "B_raw", "C_raw")}
    v_all = np.zeros(n_total, dtype=np.float64)
    seg_rows, recon_per_seg = [], []
    spot_printed = False
    for r in seg_records:
        j, stem, take = r["seg"], r["stem"], r["frames"]
        s0, s1 = r["interval"]
        trans_full = trans_cache[stem][0]
        quat_full = load_quat_cache(stem, args.npz_dirs, npz_paths, quat_cache)

        ang_full = direction_ang(trans_full)            # 世界系运动方向
        yaw_full = quat_yaw_wxyz(quat_full)             # 机体 yaw（i:zyx）
        beta_full = wrap_pi(ang_full - yaw_full)        # 机体偏差角

        spd_full = frame_speed(trans_full, args.dt, args.smooth)  # 现役平滑速度
        ab_full, _ = angle_bins(trans_full, spd_full, args.v_thresh,
                                args.theta0_k)          # A 版=构建器单一实现
        raw_b = abs_dir_bin(ang_full)
        raw_c = abs_dir_bin(beta_full)

        v = frame_speed_bwd(trans_full[:take], args.fps, "xy")
        valid = v >= args.v_thresh
        g["A"][s0:s1] = ab_full[:take]
        g["B_all"][s0:s1] = bins_with_inherit(raw_b[:take], valid, FWD_BIN)
        g["C_all"][s0:s1] = bins_with_inherit(raw_c[:take], valid, FWD_BIN)
        g["B_raw"][s0:s1] = raw_b[:take]
        g["C_raw"][s0:s1] = raw_c[:take]
        v_all[s0:s1] = v

        # spot check：首段前 5 帧四元数与 yaw（D052 风险预声明②：quat yaw 提取）
        if not spot_printed:
            for k in range(min(5, take)):
                print(f"[spotcheck] seg0({stem}) t{k} "
                      f"quat_wxyz={np.round(quat_full[k], 4).tolist()} "
                      f"yaw={math.degrees(yaw_full[k]):.2f}deg "
                      f"ang={math.degrees(ang_full[k]):.2f}deg "
                      f"beta={math.degrees(beta_full[k]):.2f}deg")
            spot_printed = True

        # 段级统计
        n_valid = int(valid.sum())
        if n_valid:
            absv = np.abs(beta_full[:take])
            straight = float((absv[valid] < math.radians(STRAIGHT_DEG)).mean())
            sideback = float((absv[valid] >= math.radians(SIDEBACK_DEG)).mean())
        else:
            straight, sideback = 0.0, 0.0
        mism = int((g["A"][s0:s1] != disk_ab[s0:s1]).sum())
        recon_per_seg.append(mism)
        seg_rows.append({
            "seg": j, "stem": stem, "kind": r["kind"], "src_stem": stem if r["kind"] == "orig" else orig_stems[r["src"]],
            "n_frames": take, "n_valid": n_valid,
            "standing_frac": round(float((v < args.v_thresh).mean()), 6),
            "straight_frac": round(straight, 6),
            "sideback_frac": round(sideback, 6),
            "speed_med_mps": round(float(np.median(v)), 6),
            "theta0_mode_bin": int(np.bincount(g["A"][s0:s1],
                                               minlength=N_BINS).argmax()),
            "recon_mismatch_frames": mism,
            "recon_mismatch_frac": round(mism / take, 6) if take else 0.0,
        })

    # ---- A 版 vs 现役 angle_bin.npy 逐帧对账（>1% = 复刻/对齐 bug）
    mism_total = int((g["A"] != disk_ab).sum())
    recon_rate = mism_total / n_total if n_total else 0.0
    recon_block = {
        "target": "现役 angle_bin.npy（build_b4lite_vae_inputs[_v2].angle_bins 复刻）",
        "mismatch_frames": mism_total, "n_frames": n_total,
        "diff_rate": round(recon_rate, 6), "diff_limit": RECON_DIFF_LIMIT,
        "per_segment_mismatch": {seg_rows[i]["stem"]: recon_per_seg[i]
                                 for i in range(len(seg_rows)) if recon_per_seg[i]},
    }
    if recon_rate > RECON_DIFF_LIMIT:
        conf = {}
        dm = np.where(g["A"] != disk_ab)[0]
        for d, rr in zip(disk_ab[dm][:20000], g["A"][dm][:20000]):
            key = f"disk{int(d)}->rebuilt{int(rr)}"
            conf[key] = conf.get(key, 0) + 1
        recon_block["status"] = "REPLICATE_MISMATCH"
        recon_block["confusion_top"] = dict(sorted(conf.items(),
                                                   key=lambda kv: -kv[1])[:16])
        recon_block["worst_segments"] = sorted(
            [seg_rows[i] for i in np.argsort(recon_per_seg)[-8:][::-1]
             if recon_per_seg[i] > 0],
            key=lambda s: -s["recon_mismatch_frames"])
    else:
        recon_block["status"] = "OK"

    # ---- 三参考系直方图（全帧 + 有效帧共用 v>=v_thresh 掩码）
    valid_all = v_all >= args.v_thresh
    n_valid_total = int(valid_all.sum())
    hists = {
        "A_theta0_anchored": {
            "note": "现役版：构建器 angle_bins 原样（全帧含其自带继承/standing 规则、"
                    "段首锚 bin0）；有效帧版=同输出采样于审计有效帧掩码",
            "all_frames": hist8(g["A"]),
            "valid_frames": hist8(g["A"][valid_all])},
        "B_world_absolute": {
            "note": "世界系绝对角版：bin=floor((ang+π)/(2π)·8)%8（env L854-855 同构）；"
                    "全帧低速继承+段首锚 bin4（§5l G1 契约）",
            "all_frames": hist8(g["B_all"]),
            "valid_frames": hist8(g["B_raw"][valid_all])},
        "C_body_beta": {
            "note": "机体偏差角 beta 版（首选）：beta=wrap_pi(ang−yaw)，同 B 公式；"
                    "全帧低速继承+段首锚 bin4；beta=0→bin4 语义锚 assert 内置",
            "all_frames": hist8(g["C_all"]),
            "valid_frames": hist8(g["C_raw"][valid_all])},
    }

    # ---- §5l 两级判据（只计算打印，不替判读）；B 版同判据对照
    standing_share = float((~valid_all).mean())

    def level12(tag: str, hist_valid: dict) -> dict:
        db4 = hist_valid["frac"][FWD_BIN]
        n_eff, eff_ids = effective_bins(hist_valid, n_valid_total)
        l1_fwd = db4 >= 0.15
        l1_stand = standing_share <= 0.50
        l2 = n_eff >= 5
        crit = {"reference": tag, "db4_share_valid": round(db4, 6),
                "db4_ge_15pct": bool(l1_fwd),
                "standing_share_all": round(standing_share, 6),
                "standing_le_50pct": bool(l1_stand),
                "level1_pass": bool(l1_fwd and l1_stand),
                "effective_bins": n_eff, "effective_bin_ids": eff_ids,
                "effective_ge_5of8": bool(l2), "level2_pass": bool(l2)}
        print(f"[D052-G0 CRIT L1] {tag}: db4_valid={db4 * 100:.2f}% "
              f"(>=15%: {'PASS' if l1_fwd else 'FAIL'}) | "
              f"standing={standing_share * 100:.2f}% "
              f"(<=50%: {'PASS' if l1_stand else 'FAIL'}) | "
              f"level1={'PASS' if crit['level1_pass'] else 'FAIL'}")
        print(f"[D052-G0 CRIT L2] {tag}: effective_bins={n_eff}/8 "
              f"(>=5: {'PASS' if l2 else 'FAIL'}) bins={eff_ids}")
        return crit

    print(f"[D052-G0] N={n_total} valid={n_valid_total} "
          f"({n_valid_total / n_total * 100:.2f}%) recon_diff="
          f"{recon_rate * 100:.4f}% ({recon_block['status']})")
    crit_c = level12("C_body_beta", hists["C_body_beta"]["valid_frames"])
    crit_b = level12("B_world_absolute", hists["B_world_absolute"]["valid_frames"])

    # ---- JSON 落盘
    out = {
        "experiment": "D052-G0",
        "title": "B4-lite 语料方向审计（三参考系 8 档直方图）",
        "prereg": "refine-logs/DS_CONTINUOUS_EXECUTION_PLAN.md §5l（判据先冻结）",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "repo_hash": args.repo_hash or None,
        "inputs_dir": args.inputs_dir,
        "npz_dirs": args.npz_dirs,
        "out": args.out,
        "params": {
            "fps": args.fps, "v_thresh_mps": args.v_thresh,
            "A_replica": {"dt_s": args.dt, "speed_smooth_frames": args.smooth,
                          "theta0_k": args.theta0_k},
            "bin_rule_env": "bin=floor((x+π)/(2π)·8) mod 8（apt_flat_env.py "
                            "L854-855 同构；x=ang 为 B 版、x=beta 为 C 版）",
            "bin_rule_A": "theta0=段首前 K 有效帧方向圆均值；delta=wrap_pi(ang−theta0)；"
                          "bin=floor((delta+π/8)/(π/4))%8；低速继承前一有效 bin、"
                          "段首无有效帧=bin0（构建器原样 import 复用）",
            "beta_rule": "beta=wrap_pi(ang−yaw)；yaw=atan2(2(wz+xy),1−2(y²+z²))"
                         "（D044 i:zyx 口径 ZYX yaw）",
            "valid_frames": f"v≥{args.v_thresh}（v=frame_speed_bwd 后向差分×50Hz XY）；"
                            "三版有效帧直方图共用此掩码",
            "inherit_rule": "全帧版：低速帧继承前一有效 bin；段首锚 A=bin0（现役）、"
                            "B/C=bin4（§5l G1 前向语义契约，与现役差异入账）",
            "segment_note": "ang/yaw/beta/A 版在完整源段上算、按 bounds 前缀截取"
                            "（副本段携带方式与构建器逐位同构）",
        },
        "n_frames": n_total, "n_segments": len(bounds),
        "n_segments_orig": len(orig_stems), "n_valid_frames": n_valid_total,
        "alignment_provenance": provenance,
        "speed": {
            "standing_share": round(standing_share, 6),
            "v_med_mps": round(float(np.median(v_all)), 6),
            "v_quantiles_mps": {q: round(float(np.quantile(v_all, q)), 4)
                                for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        },
        "histograms": hists,
        "criteria": {
            "prereg": "L1 前向档语义恢复（联合门必需）=C 版有效帧 db4 占比≥15% 且 "
                      "站立帧占比≤50%；L2 8 方位全域=C 版有效档（占比≥1%）≥5/8",
            "C_body_beta": crit_c,
            "B_world_absolute": crit_b,
        },
        "reconciliation": recon_block,
        "token_md5_spot_checks": token_md5_spot_check(
            seg_records, trans_cache,
            np.load(token_path).astype(np.float32), bounds)
        if os.path.isfile(token_path) else "skipped(token.npy 缺失)",
        "segments": seg_rows,
        "inputs_md5": {f: md5_file(os.path.join(args.inputs_dir, f))
                       for f in ["token.npy", "angle_bin.npy",
                                 "segment_bounds.npy", "segment_source.npy",
                                 "build_meta.json"]
                       if os.path.isfile(os.path.join(args.inputs_dir, f))},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[write] {args.out}")
    for tag in ("A_theta0_anchored", "B_world_absolute", "C_body_beta"):
        h = hists[tag]
        print(f"[hist {tag}] all={h['all_frames']['frac']}")
        print(f"[hist {tag}] valid={h['valid_frames']['frac']}")

    if recon_block["status"] == "REPLICATE_MISMATCH":
        print("[D052-G0] REPLICATE_MISMATCH：A 版对账差异率 >1%——复刻或对齐有 bug，"
              "本审计结果不得用于 §5l 判读；差异模式见 JSON reconciliation 块")
        sys.exit(2)


if __name__ == "__main__":
    main()
