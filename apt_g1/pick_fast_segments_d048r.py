"""D048r R1/R2 挑段：v2 g1 npz 池扫描段速，挑参考 v_med 最高段清单。

DS_CONTINUOUS_EXECUTION_PLAN §5i（D048r decoder 上限包络三角测量）预注册：
R1/R2 用同一段清单——v_med 口径 = trans_m 后向差分 ×fps（build_d048n
frame_speed_bwd 同一实现，import 复用不复制）取段中位数（med，D048n/D042
口径），门槛 --min-vmed 1.0 m/s，取 top --top 12（不足 12 取实际数，预注册
分支；全池分位数分布随 json 落盘供该分支判读）。段 = 整段 npz（B4-lite
b4lite/npz + b4lite_v2conv/npz 双目录，builder v2 同款先到先得解析，stem
跨目录去重取先者）。

用法（本机 numpy 可跑；服务器 .venv_isaac python 同样可跑）：
  python apt_g1/pick_fast_segments_d048r.py                 # 默认双目录 + 默认 out
  python apt_g1/pick_fast_segments_d048r.py --top 12 --min-vmed 1.0
产物 <out>（默认 ~/ros2_data/apt_g1/data/d048r/d048r_fast_segments.json）：
  segments[] = {stem, npz_path, v_med, v_p90, frames, fps}，供
  isaac/replay_token_speed_d048r.py --segments-json 消费。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np

try:                      # 服务器执行根平铺 import / 仓库根包 import 双兼容
    from build_d048n_vb_speed_labels import DEFAULT_NPZ_DIRS, frame_speed_bwd
except ImportError:
    from apt_g1.build_d048n_vb_speed_labels import DEFAULT_NPZ_DIRS, frame_speed_bwd

HOME = os.path.expanduser("~")
DEFAULT_OUT = f"{HOME}/ros2_data/apt_g1/data/d048r/d048r_fast_segments.json"


def build_args():
    ap = argparse.ArgumentParser(
        description="D048r: npz 池段速扫描 → top 快段清单（v_med 中位口径）")
    ap.add_argument("--npz-dir", nargs="+", default=DEFAULT_NPZ_DIRS,
                    help="源 npz 目录（可多个，先到先得、stem 去重）")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="输出 json 路径（d048r_fast_segments.json）")
    ap.add_argument("--top", type=int, default=12,
                    help="挑段数上限（§5i 预注册 12；不足取实际数）")
    ap.add_argument("--min-vmed", type=float, default=1.0,
                    help="段速门槛 m/s（§5i 预注册 1.0）")
    ap.add_argument("--min-frames", type=int, default=0,
                    help="最短段帧数过滤（0=不过滤，预注册字面口径）")
    ap.add_argument("--fps", type=float, default=50.0,
                    help="采样率（B4-lite=50Hz；npz meta 内含 fps 时以 meta 为准）")
    ap.add_argument("--axes", choices=["xy", "xyz"], default="xy",
                    help="速度差分维度（默认 xy 水平面，D046/D048n 同口径）")
    return ap


def meta_fps(npz, fallback: float) -> float:
    """npz 内嵌 meta（json 字符串）里找 fps；找不到用 fallback。"""
    if "meta" in npz:
        try:
            meta = json.loads(str(npz["meta"]))
            v = meta.get("fps")
            if v:
                return float(v)
        except Exception:
            pass
    return fallback


def scan_pool(npz_dirs: list[str], fps: float, axes: str,
              min_frames: int) -> tuple[list[dict], list[str]]:
    """扫全部 npz → 段记录列表（stem 去重先到先得）；返回 (records, skipped)。"""
    records, skipped, seen = [], [], set()
    for d in npz_dirs:
        if not os.path.isdir(d):
            skipped.append(f"DIR_MISSING {d}")
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".npz"):
                continue
            stem = fn[: -len(".npz")]
            if stem in seen:            # 双目录先到先得（builder v2 同款）
                continue
            path = os.path.join(d, fn)
            try:
                z = np.load(path, allow_pickle=False)
                if "trans_m" not in z:
                    skipped.append(f"NO_TRANS_M {path}")
                    continue
                trans = z["trans_m"].astype(np.float64)
            except Exception as e:      # 损坏文件只记录不中断（池扫描宽容）
                skipped.append(f"LOAD_FAIL {path}: {e}")
                continue
            if len(trans) < max(min_frames, 2):
                skipped.append(f"TOO_SHORT {path} ({len(trans)}f)")
                continue
            v = frame_speed_bwd(trans, meta_fps(z, fps), axes)
            seen.add(stem)
            records.append({
                "stem": stem,
                "npz_path": path,
                "v_med": float(np.median(v)),
                "v_p90": float(np.percentile(v, 90)),
                "frames": int(len(trans)),
                "fps": meta_fps(z, fps),
            })
    return records, skipped


def main():
    cli = build_args().parse_args()
    records, skipped = scan_pool(cli.npz_dir, cli.fps, cli.axes, cli.min_frames)
    if not records:
        raise SystemExit(f"[pick] FAIL: 池为空（npz_dirs={cli.npz_dir}）")
    v_all = np.array([r["v_med"] for r in records])
    q = {p: float(np.percentile(v_all, p))
         for p in (5, 25, 50, 75, 90, 95, 100)}
    above = [r for r in records if r["v_med"] >= cli.min_vmed]
    above.sort(key=lambda r: (-r["v_med"], r["stem"]))
    picked = above[: cli.top]
    print(f"[pool] n={len(records)} v_med 分位数(m/s): "
          + " ".join(f"p{p}={v:.3f}" for p, v in q.items()))
    print(f"[pick] 门槛 {cli.min_vmed} m/s 以上 {len(above)} 段 → 取 top {len(picked)}"
          f"（预注册 top {cli.top}，不足取实际数）")
    for i, r in enumerate(picked):
        print(f"  #{i + 1:02d} {r['stem'][:48]:48s} v_med={r['v_med']:.3f} "
              f"v_p90={r['v_p90']:.3f} frames={r['frames']} {r['npz_path']}")
    for s in skipped[:20]:
        print(f"  [skip] {s}")
    if len(skipped) > 20:
        print(f"  [skip] ... 共 {len(skipped)} 条（明细略）")
    out = {
        "experiment": "D048r",
        "purpose": "R1/R2 快段清单（§5i 预注册：v_med top12，门槛 1.0 m/s）",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "npz_dirs": cli.npz_dir,
        "fps_default": cli.fps,
        "speed_axes": cli.axes,
        "speed_rule": "v[t]=||trans[t]-trans[t-1]||×fps 后向差分、v[0]=v[1]"
                      "（build_d048n.frame_speed_bwd 同一实现）；段值=段中位数",
        "n_pool": len(records),
        "pool_v_med_quantiles": q,
        "n_above_threshold": len(above),
        "threshold_v_med": cli.min_vmed,
        "top_requested": cli.top,
        "n_picked": len(picked),
        "picked_short_of_top": len(picked) < cli.top,
        "segments": picked,
        "skipped": skipped,
    }
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    with open(cli.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[write] {cli.out}（n_picked={len(picked)}）")


if __name__ == "__main__":
    main()
