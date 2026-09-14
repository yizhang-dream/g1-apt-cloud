"""D057 G3: WBT 行走窗挖掘（转换后 npz → walk_segments.json，numpy-only）。

上游契约（D057 转换器产出，manifest 与 D044 同构）：<conv-dir>/manifest.json
条目 {stem, class, npz_path, wbt:{repo, episode, run_idx,...}, n_rows_enc,...}，
带 "error" 键的条目 = 转换失败，过滤；npz 键：tokens (n,64)@50Hz / jp_isaac /
jp_mj / quat_wxyz / trans_m (n,3 米) / jv_isaac / meta。

口径（D057 G3）：
  1. 逐 npz 全段逐帧速度 = frame_speed_bwd(trans_m, 50.0, "xy")（speedA-D048n
     零漂移 import：XY 后向差分 ×fps，v[0]=v[1]）。
  2. 滑窗：窗长 4s=200 帧、步长 2s=100 帧（默认），窗内 v_med≥0.10 入选；
     每 episode ≤3 窗——episode = stem 中 `_epNN_`（`_ep` + 数字 + `_`）段
     相同者同类（无该段的 stem 自成一类），同 episode 多窗取 v_med 最高的
     前 3（并列保持文件序）。
  3. 产物 <conv-dir>/walk_segments.json：[{stem, npz_path, i0, i1, dur_s,
     v_med, v_p90}] 按文件序（manifest 序，同文件内 i0 升序）；stdout 汇总
     总窗数 / 总时长 / v_med 分布分位数。

用法（服务器 .venv_isaac python；numpy-only，不依赖 torch）：
  python mine_wbt_walk_segments.py --conv-dir <conv-dir>
  python mine_wbt_walk_segments.py --selftest            # 本机 numpy-only 自测
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

try:                      # 服务器执行根平铺 import / 仓库根包 import 双兼容
    from build_d048n_vb_speed_labels import frame_speed_bwd
except ImportError:
    from apt_g1.build_d048n_vb_speed_labels import frame_speed_bwd

FPS = 50.0                      # 上游契约：转换 npz 统一 50Hz（口径勿改）
EP_RE = re.compile(r"_ep\d+_")  # episode = stem 中该段相同者同类


def _safe_join(base_dir: str, name: str) -> str:
    """常量文件名拼 + 落盘路径限制在 base_dir 内（防目录参数携带 ../ 逃逸）。"""
    base = os.path.realpath(base_dir)
    p = os.path.realpath(os.path.join(base_dir, name))
    if not p.startswith(base + os.sep):
        raise ValueError(f"unsafe output path escapes {base}: {name}")
    return p


def episode_key(stem: str) -> str:
    r"""stem -> episode 键：`_ep\d+_` 段（如 `_ep03_`）；无该段的 stem 自成一类。"""
    m = EP_RE.search(stem)
    return m.group(0) if m else stem


def slide_windows(n: int, win_len: int, stride: int) -> list[tuple[int, int]]:
    """满窗滑窗切分：起点 0, stride, 2*stride, ...（尾部不满一整窗丢弃）。"""
    assert win_len >= 1 and stride >= 1, "win_len/stride 必须 >=1 帧"
    return [(i0, i0 + win_len) for i0 in range(0, n - win_len + 1, stride)]


def mine_file(v: np.ndarray, win_len: int, stride: int,
              v_med_min: float) -> list[dict]:
    """单个 npz 的逐帧速度序列 -> 过阈值窗候选（时间序，未做 episode 截断）。"""
    out = []
    for i0, i1 in slide_windows(len(v), win_len, stride):
        vw = v[i0:i1]
        v_med = float(np.median(vw))
        if v_med >= v_med_min:
            out.append({"i0": int(i0), "i1": int(i1),
                        "dur_s": round((i1 - i0) / FPS, 4),
                        "v_med": round(v_med, 4),
                        "v_p90": round(float(np.quantile(vw, 0.9)), 4)})
    return out


def cap_per_episode(cands: list[dict], per_ep_max: int) -> list[dict]:
    r"""每 episode（stem 的 `_ep\d+_` 段同类）保留 v_med 最高的前 per_ep_max 窗。

    cands 列表序 = 文件序（manifest 序，同文件内 i0 升序）；返回保持该序的子集。
    """
    by_ep: dict[str, list[int]] = {}
    for idx, c in enumerate(cands):
        by_ep.setdefault(episode_key(c["stem"]), []).append(idx)
    keep: set[int] = set()
    for idxs in by_ep.values():
        ranked = sorted(idxs, key=lambda i: -cands[i]["v_med"])  # 稳定：并列保序
        keep.update(ranked[:per_ep_max])
    return [c for i, c in enumerate(cands) if i in keep]


def _const_trans(n: int, speed: float) -> np.ndarray:
    """n 帧匀速 +x 直线合成轨迹（z=0）：bwd 差分速度逐帧 == speed（m/s）。"""
    x = np.cumsum(np.full(n, speed / FPS))
    return np.stack([x, np.zeros(n), np.zeros(n)], axis=1)


def mine(conv_dir: str, win_s: float, stride_s: float, v_med_min: float,
         per_ep_max: int) -> list[dict]:
    """主挖掘流程：manifest -> 过阈值候选 -> episode 截断 -> walk 窗列表（文件序）。"""
    win_len = int(round(win_s * FPS))
    stride = int(round(stride_s * FPS))
    mpath = os.path.join(conv_dir, "manifest.json")
    with open(mpath, encoding="utf-8") as f:
        entries = json.load(f)
    assert isinstance(entries, list), f"manifest.json 应为条目列表: {mpath}"
    ok = [e for e in entries if isinstance(e, dict) and "error" not in e]
    n_err = len(entries) - len(ok)

    cands: list[dict] = []
    for k, e in enumerate(ok):
        stem, npz_path = e["stem"], e["npz_path"]
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f"非 error 条目但 npz 缺失: {stem} -> {npz_path}（转换器契约破坏？）")
        trans = np.load(npz_path)["trans_m"].astype(np.float64)
        if "n_rows_enc" in e:
            assert len(trans) == int(e["n_rows_enc"]), \
                f"{stem}: trans_m 帧数 {len(trans)} != manifest n_rows_enc {e['n_rows_enc']}"
        v = frame_speed_bwd(trans, FPS, "xy")
        wins = mine_file(v, win_len, stride, v_med_min)
        print(f"[mine] ({k + 1}/{len(ok)}) {stem}: n={len(trans)} "
              f"cand={len(wins)}", flush=True)
        for w in wins:
            cands.append({"stem": stem, "npz_path": npz_path, **w})
    kept = cap_per_episode(cands, per_ep_max)
    print(f"[mine] manifest {len(entries)} 条（error 过滤 {n_err}）-> "
          f"cand {len(cands)} -> per-ep<={per_ep_max} 截断后 {len(kept)} 窗")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(
        description="D057 G3: WBT 行走窗挖掘（滑窗 v_med 阈值 + 每 episode 限窗）")
    ap.add_argument("--conv-dir", default="",
                    help="转换器输出目录（含 manifest.json；产物 walk_segments.json 落此）")
    ap.add_argument("--win-s", type=float, default=4.0, help="窗长秒（默认 4.0=200 帧@50Hz）")
    ap.add_argument("--stride-s", type=float, default=2.0, help="步长秒（默认 2.0=100 帧@50Hz）")
    ap.add_argument("--v-med-min", type=float, default=0.10,
                    help="窗内速度中位数阈值 m/s（默认 0.10）")
    ap.add_argument("--per-ep-max", type=int, default=3,
                    help="每 episode 最多保留窗数（v_med 最高的前 N，默认 3）")
    ap.add_argument("--selftest", action="store_true", help="numpy-only 自测（不读文件）")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.conv_dir:
        ap.error("--conv-dir 必填（或用 --selftest）")

    kept = mine(args.conv_dir, args.win_s, args.stride_s,
                args.v_med_min, args.per_ep_max)

    out_path = _safe_join(args.conv_dir, "walk_segments.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)

    total_dur = float(sum(w["dur_s"] for w in kept))
    n_ep = len({episode_key(w["stem"]) for w in kept})
    print(f"[write] {out_path}: {len(kept)} windows / {n_ep} episodes / "
          f"total {total_dur:.1f}s")
    if kept:
        v_meds = np.asarray([w["v_med"] for w in kept], dtype=np.float64)
        qs = np.quantile(v_meds, [0.0, 0.25, 0.5, 0.75, 1.0])
        print("[mine] v_med quantiles min/q25/med/q75/max: "
              + "/".join(f"{q:.3f}" for q in qs))
    else:
        print("[mine] 0 windows（无窗过阈值）")


def selftest() -> None:
    """numpy-only 自测：①窗切分边界 ②v_med 阈值 ③per-episode 截断 ④步长语义。"""
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("[selftest] PASS " if cond else "[selftest] FAIL ") + name)
        if not cond:
            failures.append(name)

    # (1) 窗切分边界：满窗才收、右端点恰为 i0+win_len、尾部不满丢弃
    check("win-bounds 500/200/100",
          slide_windows(500, 200, 100) == [(0, 200), (100, 300), (200, 400), (300, 500)])
    check("win-bounds exact fit 200", slide_windows(200, 200, 100) == [(0, 200)])
    check("win-bounds too short 199", slide_windows(199, 200, 100) == [])

    # (4) 步长语义：起点间距 == stride 帧；stride==win 时无缝不重叠平铺
    starts = [i0 for i0, _ in slide_windows(1000, 200, 100)]
    check("stride spacing 100", all(b - a == 100 for a, b in zip(starts, starts[1:])))
    check("stride non-overlap tiling",
          slide_windows(600, 200, 200) == [(0, 200), (200, 400), (400, 600)])
    check("stride drops partial tail",
          slide_windows(650, 200, 200) == [(0, 200), (200, 400), (400, 600)])

    # (2) v_med 阈值：匀速 0.20 m/s 全过、0.05 m/s 全拒（frame_speed_bwd 零漂移路径）
    v_hi = frame_speed_bwd(_const_trans(500, 0.20), FPS, "xy")
    v_lo = frame_speed_bwd(_const_trans(500, 0.05), FPS, "xy")
    check("v_med threshold hi pass", len(mine_file(v_hi, 200, 100, 0.10)) == 4)
    check("v_med threshold lo reject", mine_file(v_lo, 200, 100, 0.10) == [])
    w0 = mine_file(v_hi, 200, 100, 0.0)[0]
    check("v_med value ~0.20", abs(w0["v_med"] - 0.20) < 1e-6)
    check("dur_s == win_s", abs(w0["dur_s"] - 4.0) < 1e-6)

    # (3) per-episode 截断：`_ep\d+_` 段同类（跨 stem），top-3 by v_med，保文件序
    check("episode_key regex", episode_key("repoA_ep03_run2_x") == "_ep03_")
    check("episode_key fallback", episode_key("no_ep_here") == "no_ep_here")
    cands = [
        {"stem": "repoA_ep01_s1", "v_med": 0.30, "i0": 0},
        {"stem": "repoA_ep01_s1", "v_med": 0.25, "i0": 100},
        {"stem": "repoA_ep01_s1", "v_med": 0.20, "i0": 200},
        {"stem": "repoB_ep01_s2", "v_med": 0.40, "i0": 0},   # 同 ep01（跨 stem 合并）
        {"stem": "repoB_ep01_s2", "v_med": 0.22, "i0": 100},
        {"stem": "repoC_ep02_s3", "v_med": 0.15, "i0": 0},   # ep02 不足上限全保
        {"stem": "repoC_ep02_s3", "v_med": 0.14, "i0": 100},
    ]
    got = [(c["stem"], c["v_med"]) for c in cap_per_episode(cands, 3)]
    check("per-ep cap top3 keeps file order",
          got == [("repoA_ep01_s1", 0.30), ("repoA_ep01_s1", 0.25),
                  ("repoB_ep01_s2", 0.40), ("repoC_ep02_s3", 0.15),
                  ("repoC_ep02_s3", 0.14)])

    if failures:
        raise SystemExit(f"selftest FAILED: {failures}")
    print("[selftest] mine_wbt_walk_segments: ALL PASS")


if __name__ == "__main__":
    main()
