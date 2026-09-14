"""D057 G0：UnifoLM-WBT 子集行走含量扫描与选料（MODULE；核心 numpy-only，parquet 读取 pyarrow）。

22 个 UnifoLM-WBT 子集已由部署脚本下载到服务器
~/ros2_data/apt_g1/data/ds_wbt/raw/<RepoName>/{meta/*.json, data/**/*.parquet}
（LeRobot 格式，meta/info.json 带 total_episodes/total_frames/fps，标称 30fps）。
下载由部署脚本完成，本脚本只分析本地已下数据、不做任何网络访问。

双 schema 自适应（同 tmp/wbt_probe.py read_file 口径；读取实现已收敛至
wbt_common.read_wbt_parquet 共享模块，D057 重构）：
  - 老款 q36：observation.state.robot_q_current[36] = 根 7（xyz + wxyz）+ 29 关节；
  - 新款 grouped：observation.state.state_base_pose[7] 显式根位姿 + 分组关节
    lower_body[15] / left_arm[7] / right_arm[7]。
两款根 XY 均取根 7 维的前 2 维（本扫描只要根位置，关节列不读，省 IO）。

每 episode：根 XY 后向差分 ×fps 得逐帧速度 v（v[t]=‖p[t]−p[t−1]‖×fps，
速度实现已收敛至 wbt_common.xy_speed_series 共享模块，D057 重构）→
v_med / v_p90 / v_max；行走窗 = 非重叠 window_s（默认 4s）整窗内 v 中位数
≥walk-thresh（默认 0.10 m/s），每个行走窗计 window_s 入行走时长，尾部不足
一个整窗的余帧不计。数据文件只取 data/**/*.parquet（glob 根限定在 data/
下，meta/ 内的 parquet 结构上不可能混入，另做防御性过滤）。

预注册选料判据（D057 G0）：子集内行走窗总时长 ≥min-walk-min（默认 10 分钟）
入选。

用法（服务器 .venv_isaac）：
  python apt_g1/scan_wbt_datasets.py --raw-root ~/ros2_data/apt_g1/data/ds_wbt/raw
  python apt_g1/scan_wbt_datasets.py --selftest   # 本机 numpy-only 自测，不 import pyarrow/pandas
产物：--out（默认 raw-root 同级 scan_wbt_g0.json），stdout 附按 walk_min
降序的选料表。
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys

import numpy as np

try:                      # 服务器执行根平铺 import / 仓库根包 import 双兼容
    from wbt_common import CheckLog, read_wbt_parquet, xy_speed_series
except ImportError:
    from apt_g1.wbt_common import CheckLog, read_wbt_parquet, xy_speed_series


def r2(x):
    return round(float(x), 3)


def episode_walk_stats(pos_xy, fps, window_s, thresh):
    """单 episode 根 XY 速度统计 + 行走窗（纯函数，numpy-only，selftest 单测对象）。

    参数：
      pos_xy: (N,2) 根位置 XY 序列（米），按时间升序；
      fps: 采样率（Hz）；
      window_s: 行走窗长（秒），非重叠整窗；
      thresh: 窗中位速度阈值（m/s），窗内 v_med ≥ thresh 判为行走窗。
    行走时长只数完整窗（floor((N−1)/窗帧数) 个），尾部余帧不计。
    返回 dict：frames / v_med / v_p90 / v_max / n_walk_windows /
    walk_frames / walk_min（分钟）。
    """
    pos_xy = np.asarray(pos_xy, dtype=np.float64)
    v = xy_speed_series(pos_xy, fps)
    w = max(1, int(round(float(window_s) * float(fps))))
    n_win = len(v) // w
    n_walk = 0
    if n_win:
        meds = np.median(v[: n_win * w].reshape(n_win, w), axis=1)
        n_walk = int((meds >= float(thresh)).sum())
    walk_frames = n_walk * w
    return {
        "frames": int(len(pos_xy)),
        "v_med": float(np.median(v)) if len(v) else 0.0,
        "v_p90": float(np.percentile(v, 90)) if len(v) else 0.0,
        "v_max": float(v.max()) if len(v) else 0.0,
        "n_walk_windows": int(n_walk),
        "walk_frames": int(walk_frames),
        "walk_min": walk_frames / float(fps) / 60.0,
    }


def load_info(repo_dir):
    """读 meta/info.json（缺失返回空 dict；fps 缺省 30）。"""
    p = os.path.join(repo_dir, "meta", "info.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def scan_dataset(repo, repo_dir, args):
    """扫一个子集目录（<RepoName>/{meta, data}）→ 选料表一行（dict）。"""
    info = load_info(repo_dir)
    fps = float(info.get("fps") or 30.0)
    # glob 根限定在 data/ 下：meta/ 内 parquet 结构上不可能混入（D057 验收点）；
    # 仍防御性再过滤一次路径中含 meta 段的文件
    files = sorted(glob.glob(os.path.join(repo_dir, "data", "**", "*.parquet"),
                             recursive=True))
    files = [f for f in files
             if "meta" not in os.path.relpath(f, repo_dir).split(os.sep)]
    entry = {
        "repo": repo,
        "n_parquet": len(files),
        "fps": fps,
        "info_total_episodes": info.get("total_episodes"),
        "info_total_frames": info.get("total_frames"),
    }
    zero = {"episodes": 0, "frames": 0, "hours": 0.0, "schema": "-",
            "v_p90_ep_max": 0.0,
            "v_xy_percentiles": {"50": 0.0, "90": 0.0, "99": 0.0},
            "walk_min": 0.0, "n_walk_windows": 0, "selected": False}
    if not files:
        entry.update(zero)
        entry["error"] = "no data parquet under data/"
        return entry
    recs, n_err = [], 0
    for f in files:
        try:
            recs.append(read_wbt_parquet(f, need_joints=False))
        except Exception as e:  # noqa: BLE001 — 单文件损坏不拖垮整个扫描
            n_err += 1
            print(f"WARN read fail {f}: {e!r}", file=sys.stderr)
    if n_err:
        entry["n_read_err"] = n_err
    if not recs:
        entry.update(zero)
        entry["error"] = "all data parquet read failed"
        return entry
    ep = np.concatenate([r["ep"] for r in recs])
    root_xy = np.concatenate([r["root7"][:, :2] for r in recs])
    ep_stats, v_parts = [], []
    for e in np.unique(ep):
        m = ep == e
        p2 = root_xy[m]
        ep_stats.append(episode_walk_stats(p2, fps, args.window_s,
                                           args.walk_thresh))
        v_parts.append(xy_speed_series(p2, fps))  # 全帧池化（同一速度实现）
    v = np.concatenate(v_parts) if v_parts else np.empty(0)
    walk_min = sum(s["walk_min"] for s in ep_stats)
    entry.update({
        "episodes": len(ep_stats),
        "frames": int(len(ep)),
        "hours": round(len(ep) / fps / 3600, 2),
        "schema": "+".join(sorted({r["schema"] for r in recs})),
        "v_p90_ep_max": (r2(max(s["v_p90"] for s in ep_stats))
                         if ep_stats else 0.0),
        "v_xy_percentiles": ({str(p): r2(np.percentile(v, p))
                              for p in (50, 90, 99)} if len(v)
                             else {"50": 0.0, "90": 0.0, "99": 0.0}),
        "walk_min": round(walk_min, 2),
        "n_walk_windows": int(sum(s["n_walk_windows"] for s in ep_stats)),
        "selected": bool(walk_min >= args.min_walk_min),
    })
    return entry


def print_table(datasets):
    """stdout 选料表：按 walk_min 降序。"""
    print(f"\n{'repo':<68}{'schema':<9}{'npq':>4}{'eps':>5}{'frames':>9}"
          f"{'hours':>7}{'vp90mx':>7}{'walk_min':>9}{'n_win':>7}  sel")
    for d in sorted(datasets, key=lambda d: (-d["walk_min"], d["repo"])):
        print(f"{d['repo']:<68}{d['schema']:<9}{d['n_parquet']:>4}"
              f"{d['episodes']:>5}{d['frames']:>9}{d['hours']:>7.2f}"
              f"{d['v_p90_ep_max']:>7.2f}{d['walk_min']:>9.2f}"
              f"{d['n_walk_windows']:>7}  {'YES' if d['selected'] else '-'}")


def run_selftest():
    """--selftest：合成数组测 episode_walk_stats 四个预注册行为（numpy-only）。"""
    fps, window_s, thresh = 30.0, 4.0, 0.10
    w_frames = int(round(window_s * fps))  # 120 帧/窗
    log = CheckLog()

    # T1 纯站立（v=0）→ walk_min=0
    s = episode_walk_stats(np.zeros((20 * int(fps), 2)), fps, window_s, thresh)
    log.check("T1 纯站立 walk_min=0",
              s["walk_min"] == 0.0 and s["n_walk_windows"] == 0, f"{s}")

    # T2 匀速 0.3 m/s 整 60s（1800 个间隔=15 个整窗）→ walk_min=1.0 分钟
    t = np.arange(int(60 * fps) + 1) / fps
    s = episode_walk_stats(np.stack([0.3 * t, np.zeros_like(t)], axis=1),
                           fps, window_s, thresh)
    log.check("T2 匀速0.3m/s×60s → walk_min≈1min",
              s["n_walk_windows"] == 15 and abs(s["walk_min"] - 1.0) < 1e-6, f"{s}")

    # T3 匀速 50s：12 整窗=48s，尾部 2s（60 帧）不足 4s 不计
    t = np.arange(int(50 * fps) + 1) / fps
    s = episode_walk_stats(np.stack([0.3 * t, np.zeros_like(t)], axis=1),
                           fps, window_s, thresh)
    log.check("T3 尾部不足4s不计（12窗/0.8min）",
              s["n_walk_windows"] == 12 and abs(s["walk_min"] - 0.8) < 1e-6, f"{s}")

    # T4 站立序列中单帧 0.5 m 跳变（v=15 m/s 尖峰）不误判行走
    pos = np.zeros((20 * int(fps), 2))
    pos[100:, 0] = 0.5
    s = episode_walk_stats(pos, fps, window_s, thresh)
    log.check("T4 单帧尖峰不误判行走",
              s["v_max"] > thresh and s["n_walk_windows"] == 0
              and s["walk_min"] == 0.0, f"{s}")

    # 附：窗帧数换算 sanity（30fps × 4s = 120 帧）
    log.check("T0 窗帧数=120 sanity",
              episode_walk_stats(np.zeros((w_frames + 1, 2)), fps, window_s,
                                 thresh)["frames"] == w_frames + 1)
    print("SELFTEST "
          + ("ALL PASS" if not log.failures else f"FAILED: {log.failures}"))
    return 0 if not log.failures else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="D057 G0 UnifoLM-WBT 子集行走含量扫描与选料（只分析本地已下数据）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--raw-root",
                    default=os.path.expanduser(
                        "~/ros2_data/apt_g1/data/ds_wbt/raw"),
                    help="raw 根目录，每子目录=<RepoName>/{meta,data}")
    ap.add_argument("--out", default=None,
                    help="汇总 JSON 输出路径（默认 raw-root 同级 scan_wbt_g0.json）")
    ap.add_argument("--walk-thresh", type=float, default=0.10,
                    help="行走窗中位速度阈值 (m/s)")
    ap.add_argument("--window-s", type=float, default=4.0,
                    help="行走窗长（秒，非重叠整窗）")
    ap.add_argument("--min-walk-min", type=float, default=10.0,
                    help="入选门槛：子集行走窗总时长（分钟）")
    ap.add_argument("--selftest", action="store_true",
                    help="本机 numpy-only 自测（不 import pyarrow/pandas）")
    args = ap.parse_args(argv)
    if args.selftest:
        return run_selftest()

    raw_root = os.path.expanduser(args.raw_root)
    if not os.path.isdir(raw_root):
        print(f"ERROR raw-root 不存在: {raw_root}", file=sys.stderr)
        return 2
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(raw_root)), "scan_wbt_g0.json")
    repos = sorted(d for d in os.listdir(raw_root)
                   if os.path.isdir(os.path.join(raw_root, d)))
    if not repos:
        print(f"ERROR raw-root 下没有子集目录: {raw_root}", file=sys.stderr)
        return 2

    datasets = [scan_dataset(repo, os.path.join(raw_root, repo), args)
                for repo in repos]
    selected = [d["repo"] for d in
                sorted(datasets, key=lambda d: -d["walk_min"])
                if d["selected"]]
    out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "selection_rule": {
            "rule": (f"子集行走窗（非重叠 {args.window_s:g}s 窗内 v_med≥"
                     f"{args.walk_thresh:g} m/s）总时长 ≥{args.min_walk_min:g} "
                     f"分钟入选"),
            "walk_thresh_m_s": args.walk_thresh,
            "window_s": args.window_s,
            "min_walk_min": args.min_walk_min,
            "v_def": ("v[t]=‖p_xy[t]−p_xy[t−1]‖×fps（episode 内根 XY 后向差分，"
                      "非重叠整窗内中位数过阈，尾部不足一整窗不计）"),
        },
        "datasets": datasets,
        "selected": selected,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"D057 G0 扫描 {len(datasets)} 个子集（fps/frames 实测 vs info.json "
          f"total_frames 可对账下载完整性）")
    print_table(datasets)
    print(f"selected ({len(selected)}): {selected}")
    print(f"OUT {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
