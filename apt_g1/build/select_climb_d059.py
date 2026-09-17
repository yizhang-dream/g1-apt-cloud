"""D059 G0-② / DS_TERRAIN_AUTHOR_PLAN：BONES-SEED 爬障（climbing box）选样 +
膝签名四分位带基线重建（numpy + pyarrow，只读语料；产物只落 --out-dir）。

背景（预注册 = refine-logs/ds/DS_CONTINUOUS_EXECUTION_PLAN.md §5s ②）：
冻结 SONIC decoder 地形线生死判别 G0-② 的选料与签名门工具。D057 的膝签名门
（膝 median > -0.25 ∧ max < 2.2）是平地行走系阈值，爬障系膝角分布不同——
预注册要求"对全量爬障段打签名分布取四分位带为门，不照抄平地阈值"。本脚本
即该基线：逐 statistic 取【逐段统计量在段间的 [Q1, Q3] 带】为门；平地阈值
（D057_ABS）只作绝对物理界观察项打印，不参与判定。

三段式（扫描 / 基线 / 选样，一次跑完）：
  1. 扫描：parquet 过滤 content_type_of_movement == --content-type（默认
     "climbing box"，lab-ts 实测 3,402 段，其中 1,701 段为 _M 镜像副本）→
     逐段读 CSV 算签名。CSV 解析**零漂移复用现役转换器**
     `convert_bones_g1_csv.load_csv`（表头 36 列断言 + 29 语义名列名→MuJoCo 序
     + deg→rad + cm→m，与转换侧同一条代码路径；import 失败即 fast-fail 报错，
     不静默换实现——如需独立实现请在 docstring 注明并补等价性断言）。
  2. 基线：对全部扫描段取每个统计量的段间四分位带 {q1, median, q3, min, max}
     + 经验命中率（四项全在带内的段占比）+ 平地绝对界观察计数。膝（左/右）
     × {median, max} 四项为门；肩滚（左/右）median 仅观察项（D057 口径）。
  3. 选样：非镜像段（_M 排除，D044/D045 先例；镜像与母段同 actor_uid，进了
     选池会破坏"演员不重复"）且源行数 ≥ --min-frames（默认 240 = 2s@120fps，
     D044 build_sample_list 同口径）构成选池；按段速 v_med 的 --n-bands
     （默认 3）等分位分快/中/慢档，每档取 --per-band（默认 2）段，
     actor_uid 六段全局互不重复；档内取最接近该档 v_med 中位数者（确定性，
     同距按 v_med、路径字典序）。产 --list 清单（绝对 CSV 路径，直接喂
     `convert_bones_g1_csv.py --list`）。阶段方向（上/下箱）只记录不设门——
     G0-③ 的"障碍类型×段速"分层在预注册里另有归属，不在本步。

段速口径（metadata 无速度字段，已核 seed_metadata_v004.parquet 51 列）：
root translate XY 前向差分 × --fps（原生 120），load_csv 已 cm→m；与 D048n
frame_speed_bwd 同量（长度 N-1，不镜像首帧，勿与 50Hz npz 侧 kinemathics 的
v_med 混用——后者是重采样后口径，本脚本另记 conversion 侧对照用）。

判读门（G0-②，--gate-manifest 模式读转换产物复算；D059 主会话 2026-09-17 重锚：
--gate-mode 双语义）：
  **band**（初版口径，保留向后可比）= 6 段膝四项全落 [Q1,Q3] 联合四分位带；
  **envelope**（重锚后生效口径，照 D057 G2 454ab84 先例）= 硬拦只有"粗错"——
  四项膝统计任一落在总体包络 [pop_min − --env-margin, pop_max + --env-margin]
  （默认 ε=0.05 rad）之外即 FAIL（拦错单位/错列映射/镜像这类转换粗错）；四分位
  带内与否降级为 warning 观察项（逐项输出带外项与余量）。
  两语义都另报 roundtrip MAE 对照 B4-lite 现役带（mean 0.1468 / p90 0.2074 /
  max 0.2539，g1_b4lite/manifest.json，D057 §5q G2 重锚）与带外/失败段如实列出。
  门语义错配的实证（D059 首跑）：[Q1,Q3] 四项联合接受域只有 8.2%（278/3402 段，
  方向分层 up 13.0% / down 0.3%），度量的是"形态典型性"而非正确性——故 envelope。

选样（--per-tier，G0-③ K≥12 用）：速度档（默认 3 档）每档 --per-tier 段（默认 2），
actor_uid 全局互斥、排 _M 镜像、不按签名滤；条目记方向（stem 前缀 come_up/come_down）。

replay 接口（gate 模式内自动产）：D048r R1 replay 兼容 segments-json
（probe/pick_fast_segments_d048r.py 产物同构：顶层 experiment/purpose/created_at/
npz_dirs/fps_default/speed_axes/speed_rule/n_pool/n_picked/picked_short_of_top/
segments/skipped；段 stem/npz_path/v_med/v_p90/frames/fps），默认落
<out-dir>/<门 JSON 同目录>/d059_climb_segments.json（--segments-json-out 可改）。
v_med 用 D048n frame_speed_bwd 口径（npz XY 后向差分×50、v[0]=v[1]、段中位），
与 R1 的 realized_ratio 分母口径一致。

用法（服务器 lab-ts；.venv_mjlab 有 mujoco+onnxruntime+pyarrow，.venv_isaac
无 mujoco ⇒ 带 --roundtrip 的转换只能用 mjlab venv，D044/D057 先例一致）：
  cd ~/ros2_data/apt_g1
  # 1) 扫描+基线+选样（分钟级；--per-tier 4 = 3 档 ×4 = 12 段，G0-③ 用）：
  ~/ros2_data/.venv_mjlab/bin/python select_climb_d059.py \
      --out-dir data/ds_bones/g1_d059_climb_smoke --per-tier 4
  # 2) 转换（现役转换器，串行——encoder 批维静态 [1,·]，勿批）：
  ~/ros2_data/.venv_mjlab/bin/python convert_bones_g1_csv.py \
      --list data/ds_bones/g1_d059_climb_smoke/climb12_list.txt \
      --calibration-json data/ds_bones/g1_b3p/calibration.json \
      --out-dir data/ds_bones/g1_d059_climb_smoke --roundtrip --fresh-manifest --force
  # 3) 门判定（envelope 语义）+ replay segments-json 导出：
  ~/ros2_data/.venv_mjlab/bin/python select_climb_d059.py \
      --gate-manifest data/ds_bones/g1_d059_climb_smoke/manifest.json \
      --band-json data/ds_bones/g1_d059_climb_smoke/select_d059.json \
      --gate-mode envelope
  # 3b) 双语义对照时必须显式给 --segments-json-out（防呆：band/envelope 两趟
  #     都默认写同目录 d059_climb_segments.json，后跑的那趟会覆盖前一趟）：
  ~/ros2_data/.venv_mjlab/bin/python select_climb_d059.py \
      --gate-manifest data/ds_bones/g1_d059_climb_smoke/manifest.json \
      --band-json data/ds_bones/g1_d059_climb_smoke/select_d059.json \
      --gate-mode band --gate-out <dir>/gate_d059_bandmode.json \
      --segments-json-out <dir>/d059_climb_segments_band.json
  # 自检（numpy-only，秒级）：--selftest

产物：<out-dir>/select_d059.json（带 + 选中段 + 计数）、
<out-dir>/select_d059_per_segment.csv（全量逐段统计，审计/D060 复用）、
<out-dir>/climb6_list.txt（--list 清单）；gate 模式另产 <out-dir>/gate_d059.json。
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

# ---------------------------------------------------------------- 常量/口径
FPS_SRC_DEFAULT = 120.0  # BONES-SEED G1 CSV 原生帧率（D043）
# MuJoCo 序下标（25/29 列语义名断言后与 D043/D044 同一映射）
KNEE_STATS = ("knee_left", "knee_right")
KNEE_IDX = {"knee_left": 3, "knee_right": 9}
SHOULDER_ROLL_IDX = {"shoulder_roll_left": 16, "shoulder_roll_right": 23}
GATE_STATS = tuple(f"{k}_{s}" for k in KNEE_STATS for s in ("median", "max"))
OBS_STATS = tuple(f"{k}_median" for k in SHOULDER_ROLL_IDX)  # 肩滚只观察
# D057 平地行走系绝对物理界（§5q/convert_wbt_g1_parquet.assert_joint_signature）
# ——G0-② 只作观察项打印，不参与判定（预注册：不照抄平地阈值）
D057_ABS = {"knee_median_min": -0.25, "knee_max_max": 2.2}
# B4-lite 现役 roundtrip 带（D057 §5q G2 重锚自 g1_b4lite/manifest.json 147 段）
B4LITE_BAND = {"mean": 0.1468, "p90": 0.2074, "max": 0.2539}

DEFAULT_PARQUET = "data/ds_bones/seed_metadata_v004.parquet"
DEFAULT_DS_ROOT = "data/ds_bones"
DEFAULT_OUT_DIR = "data/ds_bones/g1_d059_climb_smoke"
DEFAULT_CONTENT_TYPE = "climbing box"


# ------------------------------------------------------------- 依赖复用
def load_converter():
    """零漂移 import 现役转换器（CSV 解析/stem 语义单一事实源）。

    lab-ts 部署根 = ~/ros2_data/apt_g1（平铺布局，脚本与 data/ 同级）；
    本仓分域布局 = apt_g1/build/。两条 sys.path 都试；失败即 raise（不静默
    换独立实现——语义漂移风险大于收益，宁可 fast-fail）。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.getcwd()):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    last = None
    for name in ("convert_bones_g1_csv", "build.convert_bones_g1_csv",
                 "apt_g1.build.convert_bones_g1_csv"):
        try:
            mod = importlib.import_module(name)
            print(f"[select] CSV 解析复用现役转换器: {mod.__file__}")
            return mod
        except Exception as exc:  # noqa: BLE001 —— 逐候选回退
            last = exc
    raise ImportError(
        "现役转换器 convert_bones_g1_csv 不可 import（cwd 与脚本目录都不在其"
        f"所在路径？最后错误: {type(last).__name__}: {last}）")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- 单段统计
def speed_stats(trans_m, fps):
    """段速统计（XY 前向差分 ×fps；load_csv 已 cm→m）。返回 (v_med, v_p90)。"""
    d = np.diff(np.asarray(trans_m, dtype=np.float64)[:, :2], axis=0)
    if len(d) == 0:
        return 0.0, 0.0
    v = np.linalg.norm(d, axis=1) * float(fps)
    return float(np.median(v)), float(np.percentile(v, 90))


def segment_signature(dof_mj):
    """逐段签名统计（rad，MuJoCo 序）：膝 L/R × {median, max} + 肩滚 L/R median。"""
    jp = np.asarray(dof_mj, dtype=np.float64)
    out = {}
    for key, idx in KNEE_IDX.items():
        col = jp[:, idx]
        out[f"{key}_median"] = float(np.median(col))
        out[f"{key}_max"] = float(np.max(col))
    for key, idx in SHOULDER_ROLL_IDX.items():
        out[f"{key}_median"] = float(np.median(jp[:, idx]))
    return out


def quartile_band(values):
    """段间四分位带（[Q1, Q3] 为门；median/min/max 供审计）。"""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return None
    return {"q1": float(np.percentile(a, 25.0)),
            "median": float(np.median(a)),
            "q3": float(np.percentile(a, 75.0)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "n": int(a.size)}


def in_band(value, band):
    return band is not None and band["q1"] <= float(value) <= band["q3"]


def envelope_margin(value, stat_band, margin=0.05):
    """总体包络余量（envelope 硬门口径）：正 = 在 [pop_min−ε, pop_max+ε] 内距最近
    边界的距离，负 = 包络外超出的距离；stat_band 缺失返回 None（不判）。"""
    if stat_band is None:
        return None
    lo = stat_band["min"] - margin
    hi = stat_band["max"] + margin
    return min(float(value) - lo, hi - float(value))


def gate_segment(sig, band, mode="band", env_margin=0.05):
    """单段膝签名判定（纯函数，双语义，selftest 可测）。

    band 语义（D059 初版，保留向后可比）：四项膝统计全落 [Q1,Q3] 联合带 = 硬门；
    envelope 语义（主会话 2026-09-17 重锚，照 D057 G2 454ab84 口径修正先例）：
    硬拦 = 四项任一落总体包络 [pop_min−ε, pop_max+ε] 之外（拦错单位/错列映射/
    镜像这类粗错）；四分位带外降级 warning（band_warnings，含逐项余量）。
    返回 dict：mode/hard_ok/band_ok/band_margin/env_margin/band_warnings。
    """
    band_ok = {k: in_band(sig[k], band.get(k)) for k in GATE_STATS}
    band_margin = {k: (min(sig[k] - band[k]["q1"], band[k]["q3"] - sig[k])
                       if band.get(k) else None) for k in GATE_STATS}
    env = {k: envelope_margin(sig[k], band.get(k), env_margin) for k in GATE_STATS}
    if mode == "envelope":
        hard_ok = bool(all(v is not None and v >= 0.0 for v in env.values()))
    else:
        hard_ok = bool(all(band_ok.values()))
    return {"mode": mode, "hard_ok": hard_ok, "band_ok": band_ok,
            "band_margin": band_margin, "envelope_margin": env,
            "band_warnings": [k for k in GATE_STATS if not band_ok[k]]}


def direction_of(stem):
    """段方向（stem 前缀启发式）：首 4 个 token 里出现 up/onto→"up"、
    down→"down"，否则 "unknown"（come_up_50cm_box_* / come_down_50cm_box_* /
    lift_crate_come_up_* / neutral_come_down_* / dancing_routine_*）。"""
    for tok in stem.split("_")[:4]:
        if tok in ("up", "onto"):
            return "up"
        if tok == "down":
            return "down"
    return "unknown"


def speed_stats_bwd(trans_m, fps):
    """npz 侧段速（D048n frame_speed_bwd 口径：v[t]=||Δxy||×fps、v[0]=v[1]）。
    返回 (v_med, v_p90)。与源侧 speed_stats（N-1 前向差分）区分——导出给
    D048r R1 replay 的 v_med（realized_ratio 分母）用本口径。"""
    d = np.diff(np.asarray(trans_m, dtype=np.float64)[:, :2], axis=0)
    if len(d) == 0:
        return 0.0, 0.0
    v = np.concatenate([np.zeros(1), np.linalg.norm(d, axis=1)]) * float(fps)
    if len(v) > 1:
        v[0] = v[1]
    return float(np.median(v)), float(np.percentile(v, 90))


def b4lite_label(mae):
    """单段 roundtrip MAE 对照 B4-lite 现役带的标签（先 p90 后 max 档）。"""
    if mae is None:
        return "n/a"
    if mae <= B4LITE_BAND["p90"]:
        return "within_p90"
    if mae <= B4LITE_BAND["max"]:
        return "within_max"
    return "over_max"


def fmt_num(x, spec):
    """None 安全格式化（gate 报表用）：None -> "n/a"，与 b4lite_label(None) 同约定。

    转换未带 --roundtrip 时 manifest 条目无 roundtrip_mae/lattice_rate，
    报表路径不得对 None 做 float 格式化（reviewer should-fix 2026-09-17）。"""
    return "n/a" if x is None else format(x, spec)


# ------------------------------------------------------------- 选样
def pick_bands(rows, per_band=2, n_bands=3):
    """按 v_med 等分位分档抽样（确定性）。

    rows: 选池 list[dict]（需含 v_med / actor_uid / csv_path）。
    档边界 = 选池 v_med 的 (100/n_bands)% 分位（half-open [lo, hi)）；
    档内排序 = |v_med - 档内 v_med 中位数| ↑，再 v_med ↑，再 csv_path ↑；
    actor_uid 全局互斥。返回 (picks, edges, band_info)；凑不满时如实返回不足。
    """
    if not rows:
        return [], [], []
    v = np.asarray([r["v_med"] for r in rows], dtype=np.float64)
    edges = [float(np.percentile(v, 100.0 * i / n_bands))
             for i in range(1, n_bands)]
    band_id = np.digitize(v, edges)  # right=False 语义：[lo, hi)
    picks, band_info = [], []
    used_actors = set()
    for b in range(n_bands):
        idx = np.nonzero(band_id == b)[0]
        cands = [rows[int(i)] for i in idx]
        target = float(np.median([c["v_med"] for c in cands])) if cands else None
        cands.sort(key=lambda r: (abs(r["v_med"] - target) if target is not None else 0.0,
                                  r["v_med"], r["csv_path"]))
        taken, skipped_actor = 0, 0
        for r in cands:
            if r["actor_uid"] in used_actors:
                skipped_actor += 1
                continue
            if taken >= per_band:
                break
            picks.append({**r, "band": b, "band_target_v_med": target})
            used_actors.add(r["actor_uid"])
            taken += 1
        band_info.append({"band": b, "lo": edges[b - 1] if b > 0 else None,
                          "hi": edges[b] if b < len(edges) else None,
                          "n_candidates": int(len(cands)),
                          "v_med_median_in_band": target,
                          "n_picked": int(taken),
                          "n_skipped_actor_dup": int(skipped_actor),
                          "short": bool(taken < per_band)})
    return picks, edges, band_info


def resolve_csv_path(move_g1_path, ds_root, csv_root):
    """parquet move_g1_path（相对 ds_bones 根，如 g1/csv/240529/x.csv）→ 绝对路径。

    依次试：绝对原值 / ds_root 拼接 / csv_root 下的相对尾段 / csv_root 下的一级
    日期目录 + basename。全失败返回 None（不猜）。
    """
    if not move_g1_path:
        return None
    cands = [move_g1_path, os.path.join(ds_root, move_g1_path)]
    tail = move_g1_path.replace("\\", "/")
    if "/g1/csv/" in tail:
        cands.append(os.path.join(csv_root, tail.split("/g1/csv/", 1)[1]))
    cands.append(os.path.join(csv_root, os.path.basename(tail)))
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def read_parquet_rows(parquet, content_type, limit=None):
    """pyarrow 读 metadata → [(move_g1_path, actor_uid, is_mirror, n_frames_meta)]。"""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    cols = ["move_g1_path", "content_type_of_movement", "actor_uid", "is_mirror",
            "move_duration_frames"]
    tbl = pq.read_table(parquet, columns=cols)
    n_total = tbl.num_rows
    sub = tbl.filter(pc.equal(tbl.column("content_type_of_movement"), content_type))
    n_hit = sub.num_rows
    rows = []
    paths = sub.column("move_g1_path").to_pylist()
    actors = sub.column("actor_uid").to_pylist()
    mirrors = sub.column("is_mirror").to_pylist()
    frames = sub.column("move_duration_frames").to_pylist()
    for p, a, m, f in zip(paths, actors, mirrors, frames):
        rows.append({"move_g1_path": p, "actor_uid": a, "is_mirror": bool(m),
                     "n_frames_meta": int(f) if f is not None else None})
    rows.sort(key=lambda r: (r["move_g1_path"] or ""))  # 确定性
    if limit:
        rows = rows[:limit]
    return rows, n_total, n_hit


# ------------------------------------------------------------- 扫描主流程
def scan_select(args):
    t0 = time.time()
    cb = load_converter()
    rows, n_total, n_hit = read_parquet_rows(args.parquet, args.content_type,
                                             limit=args.limit)
    print(f"[select] parquet {args.parquet}: {n_total} rows | "
          f"content_type={args.content_type!r}: {n_hit} | 本次扫描 {len(rows)}")

    scanned, failed = [], []
    for i, r in enumerate(rows):
        csv_path = resolve_csv_path(r["move_g1_path"], args.ds_root, args.csv_root)
        if csv_path is None:
            failed.append({**r, "error": "csv_not_found"})
            continue
        try:
            dof_mj, _euler, trans_m, n_src, _fs, _fe, warn = cb.load_csv(csv_path)
            sig = segment_signature(dof_mj)
            v_med, v_p90 = speed_stats(trans_m, args.fps)
            stem, actor, mirror, cls = cb.parse_stem(csv_path)
            scanned.append({
                "stem": stem, "actor_uid": r["actor_uid"], "csv_path": csv_path,
                "csv_relpath": os.path.relpath(csv_path, args.ds_root).replace("\\", "/"),
                "mirror": bool(mirror), "class": cls, "direction": direction_of(stem),
                "n_frames_src": int(n_src), "n_frames_meta": r["n_frames_meta"],
                "v_med": v_med, "v_p90": v_p90, "warnings": list(warn), **sig,
            })
        except Exception as exc:  # noqa: BLE001 —— 逐段记账，不静默跳过
            failed.append({**r, "csv_path": csv_path,
                           "error": f"{type(exc).__name__}: {exc}"})
        if (i + 1) % 250 == 0 or i + 1 == len(rows):
            print(f"[select]   scanned {i + 1}/{len(rows)} "
                  f"(ok {len(scanned)}, fail {len(failed)}) "
                  f"{time.time() - t0:.0f}s", flush=True)

    # ---- 基线：段间四分位带（门 = 膝四项；肩滚观察）
    band = {}
    for stat in GATE_STATS + OBS_STATS:
        vals = [s[stat] for s in scanned]
        band[stat] = quartile_band(vals) if vals else None
    for s in scanned:  # 逐段带内标记（四项全含）
        s["in_band_stats"] = {k: in_band(s[k], band[k]) for k in GATE_STATS}
        s["in_band_all4"] = bool(all(s["in_band_stats"].values()))
    n_all4 = int(sum(1 for s in scanned if s["in_band_all4"]))
    rate_per_stat = {k: (float(np.mean([s["in_band_stats"][k] for s in scanned]))
                         if scanned else 0.0) for k in GATE_STATS}
    # D057 平地绝对界观察（只看不判）
    abs_obs = {
        "rule": f"knee median > {D057_ABS['knee_median_min']} and "
                f"knee max < {D057_ABS['knee_max_max']} (D057 平地系，仅观察)",
        "knee_left": {"median_ok": int(sum(s["knee_left_median"] > D057_ABS["knee_median_min"]
                                           for s in scanned)),
                      "max_ok": int(sum(s["knee_left_max"] < D057_ABS["knee_max_max"]
                                        for s in scanned))},
        "knee_right": {"median_ok": int(sum(s["knee_right_median"] > D057_ABS["knee_median_min"]
                                            for s in scanned)),
                       "max_ok": int(sum(s["knee_right_max"] < D057_ABS["knee_max_max"]
                                         for s in scanned))},
        "n_scanned": len(scanned),
    }

    # ---- 选池 + 等分位选样（--per-tier）
    pool = [s for s in scanned if (not s["mirror"]) and s["n_frames_src"] >= args.min_frames]
    picks, edges, band_info = pick_bands(pool, args.per_tier, args.n_bands)
    vpool = np.asarray([s["v_med"] for s in pool], dtype=np.float64)
    speed_dist = ({"n": int(vpool.size), "min": float(vpool.min()),
                   "q1": float(np.percentile(vpool, 25)), "median": float(np.median(vpool)),
                   "q3": float(np.percentile(vpool, 75)), "max": float(vpool.max())}
                  if vpool.size else None)

    # ---- 产物
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = args.out_json or os.path.join(args.out_dir, "select_d059.json")
    out_csv = args.per_segment_csv or os.path.join(args.out_dir, "select_d059_per_segment.csv")
    out_list = args.list_out or os.path.join(
        args.out_dir, f"climb{len(picks)}_list.txt")

    with open(out_csv, "w", encoding="utf-8") as f:
        cols = (["stem", "actor_uid", "mirror", "direction", "n_frames_src", "v_med", "v_p90"]
                + list(GATE_STATS) + list(OBS_STATS) + ["in_band_all4", "csv_relpath"])
        f.write(",".join(cols) + "\n")
        for s in scanned:
            f.write(",".join([str(s["stem"]), str(s["actor_uid"]), str(s["mirror"]),
                              str(s["direction"]), str(s["n_frames_src"]),
                              f"{s['v_med']:.6f}", f"{s['v_p90']:.6f}"]
                             + [f"{s[k]:.6f}" for k in GATE_STATS + OBS_STATS]
                             + [str(s["in_band_all4"]), str(s["csv_relpath"])]) + "\n")
    with open(out_list, "w", encoding="utf-8") as f:
        for p in picks:
            f.write(p["csv_path"] + "\n")

    doc = {
        "exp": "D059 G0-②",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": os.path.abspath(__file__),
        "input": {
            "parquet": os.path.abspath(args.parquet),
            "parquet_md5": md5_file(args.parquet),
            "content_type": args.content_type,
            "ds_root": args.ds_root, "csv_root": os.path.abspath(args.csv_root),
            "converter_module": cb.__file__,
            "converter_md5": md5_file(cb.__file__),
        },
        "methods": {
            "parse": "零漂移复用 convert_bones_g1_csv.load_csv（表头断言 + 语义名"
                     "→MuJoCo 序 + deg→rad + cm→m）",
            "speed": f"root translate XY 前向差分 × {args.fps} fps（load_csv 已 cm→m），"
                     "N-1 长度；与 D048n frame_speed_bwd 同量不同长度口径",
            "band": "逐 statistic 取【逐段统计量在段间的 [Q1, Q3] 带】为门"
                    "（预注册 §5s ②：爬障系自分布，不照抄 D057 平地阈值）",
            "gate": "双语义（--gate-mode）：band=四项膝统计全在 [Q1,Q3] 联合带；"
                    "envelope=硬拦总体包络 [pop_min−ε, pop_max+ε] 之外（粗错），"
                    "四分位带外降级 warning；肩滚 L/R median 仅观察",
            "gate_stats": list(GATE_STATS),
            "n_bands": args.n_bands, "per_tier": args.per_tier,
            "selection": f"非镜像 ∧ 行数 ≥ {args.min_frames} 为选池；v_med "
                         f"{args.n_bands} 等分位分档；每档 --per-tier {args.per_tier} 段；"
                         "actor_uid 全局互斥；档内取最接近档中位者（确定性）；"
                         "不按签名滤（保门判读无偏）",
            "direction": "stem 前缀启发式（首 4 token 含 up/onto→up、down→down）",
            "min_frames_src": args.min_frames,
            "fps": args.fps,
        },
        "counts": {
            "parquet_rows": n_total,
            "content_type_rows": n_hit,
            "scanned": len(scanned),
            "failed": len(failed),
            "band_n": len(scanned),
            "mirror_scanned": int(sum(1 for s in scanned if s["mirror"])),
            "selection_pool": len(pool),
            "limit_applied": args.limit,
        },
        "signature_band": {
            "unit": "rad", "order": "MuJoCo 29 序（D043/D044 口径）",
            "gate_stats": list(GATE_STATS), "observation_stats": list(OBS_STATS),
            "band": band,
            "empirical": {
                "n_scanned": len(scanned),
                "n_all4_in_band": n_all4,
                "rate_all4_in_band": (n_all4 / len(scanned)) if scanned else 0.0,
                "rate_per_stat": rate_per_stat,
            },
            "d057_abs_observation": abs_obs,
        },
        "speed": {"source_field": None,
                  "note": "parquet 无速度字段，按 methods.speed 计算",
                  "pool_dist": speed_dist, "band_edges": edges, "band_info": band_info},
        "selected": [{"stem": p["stem"], "actor_uid": p["actor_uid"], "band": p["band"],
                      "band_name": ["slow", "mid", "fast"][p["band"]] if p["band"] < 3 else f"b{p['band']}",
                      "direction": p["direction"],
                      "v_med": p["v_med"], "v_p90": p["v_p90"],
                      "n_frames_src": p["n_frames_src"],
                      "band_target_v_med": p["band_target_v_med"],
                      "csv_path": p["csv_path"], "csv_relpath": p["csv_relpath"],
                      "in_band_stats": p["in_band_stats"], "in_band_all4": p["in_band_all4"],
                      **{k: p[k] for k in GATE_STATS + OBS_STATS}} for p in picks],
        "failed": failed,
        "artifacts": {"per_segment_csv": os.path.abspath(out_csv),
                      "list": os.path.abspath(out_list)},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)

    # ---- stdout 摘要
    print(f"\n[band] 段间四分位带（n={len(scanned)} 段，单位 rad）：")
    print("  stat                     Q1       median   Q3       min      max")
    for stat in GATE_STATS + OBS_STATS:
        b = band[stat]
        mark = "GATE" if stat in GATE_STATS else "obs "
        if b is None:  # 无可用段（如 --content-type 打空）时不得对 None 下标
            print(f"  {mark} {stat:20s} n/a（本次无可用段）")
            continue
        print(f"  {mark} {stat:20s} {b['q1']:+.4f} {b['median']:+.4f} {b['q3']:+.4f} "
              f"{b['min']:+.4f} {b['max']:+.4f}")
    print(f"[band] 四项全在带内: {n_all4}/{len(scanned)} = "
          f"{100.0 * n_all4 / max(len(scanned), 1):.1f}%")
    print(f"[band] D057 平地绝对界观察（膝盖 median>{D057_ABS['knee_median_min']} / "
          f"max<{D057_ABS['knee_max_max']}）：left med_ok={abs_obs['knee_left']['median_ok']}/"
          f"{len(scanned)} max_ok={abs_obs['knee_left']['max_ok']}/{len(scanned)} | "
          f"right med_ok={abs_obs['knee_right']['median_ok']}/{len(scanned)} "
          f"max_ok={abs_obs['knee_right']['max_ok']}/{len(scanned)}")
    print(f"\n[speed] 选池 n={len(pool)} 分位边界={['%.3f' % e for e in edges]} m/s")
    print(f"[select] 选中 {len(picks)} 段：")
    for p in picks:
        print(f"  band={p['band']} {p['stem']:52s} actor={p['actor_uid']} "
              f"dir={p['direction']:7s} v_med={p['v_med']:.3f} frames={p['n_frames_src']} "
              f"in_band={'Y' if p['in_band_all4'] else 'N'} | "
              f"kneeL med={p['knee_left_median']:+.3f} max={p['knee_left_max']:.3f} | "
              f"kneeR med={p['knee_right_median']:+.3f} max={p['knee_right_max']:.3f}")
    print(f"[select] 扫描 {len(scanned)}/{len(rows)}（fail {len(failed)}）| "
          f"耗时 {time.time() - t0:.1f}s")
    print(f"[select] -> {out_json}")
    print(f"[select] -> {out_csv}")
    print(f"[select] -> {out_list}")
    return doc


# ------------------------------------------------------------- 门判定（读 npz）
def export_replay_segments(band_doc, rows, out_path, out_dir):
    """D048r R1 replay 兼容 segments-json（probe/pick_fast_segments_d048r.py 产物
    同构：顶层 experiment/purpose/created_at/npz_dirs/fps_default/speed_axes/
    speed_rule/n_pool/n_picked/picked_short_of_top/segments/skipped；段条目
    stem/npz_path/v_med/v_p90/frames/fps）。v_med 用 npz 侧 D048n 口径（=R1 的
    realized_ratio 分母口径）。"""
    segs = [{"stem": r["stem"], "npz_path": r["npz_path"],
             "v_med": r["v_med_npz"], "v_p90": r["v_p90_npz"],
             "frames": r["n_rows_enc"], "fps": 50.0}
            for r in rows if "error" not in r]
    n_req = (band_doc.get("methods", {}).get("n_bands", len(segs))
             * band_doc.get("methods", {}).get("per_tier", len(segs)))
    doc = {
        "experiment": "D059",
        "purpose": ("D059 G0-② 爬障段清单（BONES-SEED climbing box；速度档分层 ×"
                    "--per-tier、actor_uid 全局互斥、排 _M 镜像、不按签名滤）→ "
                    "G0-③ oracle 回放（R1 方法复用）；格式对照 D048r R1 "
                    "probe/pick_fast_segments_d048r.py 产物"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "npz_dirs": [os.path.join(out_dir, "npz")],
        "fps_default": 50.0,
        "speed_axes": "xy",
        "speed_rule": ("v[t]=||trans[t]-trans[t-1]||×fps 后向差分、v[0]=v[1]"
                       "（build_d048n.frame_speed_bwd 同一实现）；段值=段中位数"),
        "n_pool": band_doc.get("counts", {}).get("selection_pool"),
        "selection_rule": band_doc.get("methods", {}).get("selection"),
        "tier_edges_v_med": band_doc.get("speed", {}).get("band_edges"),
        "n_requested": n_req,
        "n_picked": len(segs),
        "picked_short_of_top": bool(len(segs) < n_req),
        "segments": segs,
        "skipped": [],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    return doc


def gate_main(args):
    band_doc = json.load(open(args.band_json, encoding="utf-8"))
    band = band_doc["signature_band"]["band"]
    sel = {s["stem"]: s for s in band_doc.get("selected", [])}
    man = json.load(open(args.gate_manifest, encoding="utf-8"))
    entries = [e for e in man if isinstance(e, dict)]
    ok_entries = [e for e in entries if "error" not in e]
    err_entries = [e for e in entries if "error" in e]

    rows = []
    for e in ok_entries:
        stem = e.get("stem")
        npz_path = e.get("npz_path")
        try:
            with np.load(npz_path) as z:  # NpzFile 关后不可读 -> 本块内取全部键
                jp = z["jp_mj"]
                tr = z["trans_m"]
            sig = segment_signature(jp)
            v_med_npz, v_p90_npz = speed_stats_bwd(tr, e.get("fps_enc", 50.0))
        except Exception as exc:  # noqa: BLE001
            rows.append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"})
            continue
        gs = gate_segment(sig, band, args.gate_mode, args.env_margin)
        src = sel.get(stem)
        # 稀疏/旧版 band JSON 的 selected 条目可能缺统计键或 v_med -> 一律 .get，
        # 缺项即不计入源↔npz 对照（不得 KeyError）
        src_delta = ({k: abs(sig[k] - src[k]) for k in GATE_STATS
                      if src.get(k) is not None} if src else None)
        rows.append({
            "stem": stem, "actor": e.get("actor"),
            "band": src.get("band_name") if src else None,
            "direction": src.get("direction") if src else None,
            "n_rows_src": e.get("n_rows_src"), "n_rows_enc": e.get("n_rows_enc"),
            "lattice_rate": e.get("lattice_rate"), "roundtrip_mae": e.get("roundtrip_mae"),
            "roundtrip_label": b4lite_label(e.get("roundtrip_mae")),
            "roundtrip_mae_default_baseline": e.get("roundtrip_mae_default_baseline"),
            "v_med_npz": v_med_npz, "v_p90_npz": v_p90_npz,
            "v_med_src": src.get("v_med") if src else None,
            **sig,
            "hard_ok": gs["hard_ok"], "gate_mode": gs["mode"],
            "in_band_stats": gs["band_ok"], "in_band_all4": bool(all(gs["band_ok"].values())),
            "band_warnings": gs["band_warnings"], "band_margin": gs["band_margin"],
            "envelope_margin": gs["envelope_margin"],
            "src_stat_absdelta": src_delta,
            "npz_path": npz_path,
        })

    rts = [r["roundtrip_mae"] for r in rows
           if r.get("roundtrip_mae") is not None and "error" not in r]
    n_ok = len([r for r in rows if "error" not in r])
    pop_rate = band_doc.get("signature_band", {}).get(
        "empirical", {}).get("rate_all4_in_band")
    pop_rate_pct = None if pop_rate is None else 100.0 * pop_rate
    hard_ok = [r for r in rows if "error" not in r and r["hard_ok"]]
    band_ok = [r for r in rows if "error" not in r and r["in_band_all4"]]
    warn_rows = [r for r in rows if "error" not in r and r["band_warnings"]]
    summary = {
        "exp": "D059 G0-② gate",
        "gate_mode": args.gate_mode,
        "envelope_margin_rad": args.env_margin,
        "band_json": os.path.abspath(args.band_json),
        "manifest": os.path.abspath(args.gate_manifest),
        "population_joint_in_band_rate": pop_rate,
        "n_entries": len(entries), "n_converted": n_ok, "n_errors": len(err_entries),
        "n_hard_ok": len(hard_ok),
        "hard_pass_all": bool(n_ok > 0 and len(hard_ok) == n_ok and not err_entries),
        "n_in_band_all4": len(band_ok),
        "in_band_pass_all": bool(n_ok > 0 and len(band_ok) == n_ok and not err_entries),
        "n_band_warn": len(warn_rows),
        "roundtrip": {"n": len(rts), "mean": float(np.mean(rts)) if rts else None,
                      "p90": float(np.percentile(rts, 90)) if rts else None,
                      "max": float(np.max(rts)) if rts else None,
                      "b4lite_band": B4LITE_BAND,
                      "b4lite_verdict": ("pass" if rts and
                                         float(np.mean(rts)) <= B4LITE_BAND["mean"]
                                         and float(np.percentile(rts, 90)) <= B4LITE_BAND["p90"]
                                         else ("see_per_segment" if rts else "n/a"))},
        "lattice_max": max([r["lattice_rate"] for r in rows
                            if r.get("lattice_rate") is not None], default=None),
        "rows": rows, "errors": err_entries,
    }
    out = args.gate_out or os.path.join(os.path.dirname(os.path.abspath(args.gate_manifest)),
                                        "gate_d059.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    seg_out = args.segments_json_out or os.path.join(
        os.path.dirname(os.path.abspath(out)), "d059_climb_segments.json")
    export_replay_segments(band_doc, rows, seg_out,
                           os.path.dirname(os.path.abspath(args.gate_manifest)))

    print(f"\n[gate] 模式={args.gate_mode}"
          + (f"（envelope 硬门余量 ±{args.env_margin} rad）" if args.gate_mode == "envelope" else "")
          + f" | 转换条目 {len(entries)}（ok {n_ok} / error {len(err_entries)}）")
    print(f"{'stem':44s} {'tier':4s} {'dir':4s} {'kneeL_med':>9s} {'kneeL_max':>9s} "
          f"{'kneeR_med':>9s} {'kneeR_max':>9s} {'hard':>4s} {'band':>4s} "
          f"{'rt_mae':>7s} {'lat':>8s}")
    for r in rows:
        if "error" in r:
            print(f"{r['stem']:44s} ERROR {r['error']}")
            continue
        print(f"{r['stem']:44s} {str(r['band']):4s} {str(r['direction']):4s} "
              f"{r['knee_left_median']:+9.3f} {r['knee_left_max']:9.3f} "
              f"{r['knee_right_median']:+9.3f} {r['knee_right_max']:9.3f} "
              f"{'Y' if r['hard_ok'] else 'N':>4s} {'Y' if r['in_band_all4'] else 'N':>4s} "
              f"{fmt_num(r['roundtrip_mae'], '7.4f'):>7s} "
              f"{fmt_num(r['lattice_rate'], '8.1e'):>8s}")
        if r["band_warnings"]:
            print(f"{'':44s}   band warning(带外项/余量): "
                  + ", ".join(f"{k}({fmt_num(r['band_margin'][k], '+.3f')})"
                              for k in r["band_warnings"]))
        envbad = [k for k in GATE_STATS
                  if r["envelope_margin"][k] is not None and r["envelope_margin"][k] < 0]
        if envbad:
            print(f"{'':44s}   envelope 越界(粗错): "
                  + ", ".join(f"{k}({fmt_num(r['envelope_margin'][k], '+.3f')})"
                              for k in envbad))
        if r.get("src_stat_absdelta"):
            print(f"{'':44s}   源↔npz 膝四项 |Δ| max = "
                  f"{max(r['src_stat_absdelta'].values()):.2e}（转换映射保真度）")
    print(f"[gate] 硬门（{args.gate_mode}）{len(hard_ok)}/{n_ok}"
          f" | 四分位带内（warning 口径）{len(band_ok)}/{n_ok}"
          f"（全量联合带接受域先验 {fmt_num(pop_rate_pct, '.1f')}%）"
          f" | 带 warning 段 {len(warn_rows)}")
    print(f"[gate] roundtrip MAE mean "
          f"{fmt_num(summary['roundtrip']['mean'], '.4f')} "
          f"p90 {fmt_num(summary['roundtrip']['p90'], '.4f')} "
          f"max {fmt_num(summary['roundtrip']['max'], '.4f')} vs B4-lite "
          f"{B4LITE_BAND['mean']}/{B4LITE_BAND['p90']}/{B4LITE_BAND['max']} "
          f"-> {summary['roundtrip']['b4lite_verdict']}")
    print(f"[gate] -> {out}")
    print(f"[gate] replay segments-json -> {seg_out}")
    return summary


# ------------------------------------------------------------------ selftest
def selftest():
    ok = []

    def chk(name, cond, detail=""):
        ok.append((name, bool(cond)))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    # ① 段速：1 cm/帧 @120fps 直线 = 1.2 m/s
    tr = np.stack([np.arange(0, 101, 1.0), np.zeros(101)], axis=1) / 100.0
    vm, vp = speed_stats(tr, 120.0)
    chk("speed: 1cm/帧@120fps -> 1.2 m/s", abs(vm - 1.2) < 1e-12 and abs(vp - 1.2) < 1e-12,
        f"v_med={vm}")
    vm0, _ = speed_stats(np.zeros((10, 3)), 120.0)
    chk("speed: 原地 -> 0", vm0 == 0.0)

    # ② 签名：合成 29 列
    jp = np.zeros((50, 29))
    jp[:, 3] = np.linspace(0.2, 1.5, 50)   # 左膝
    jp[:, 9] = np.linspace(0.1, 1.2, 50)   # 右膝
    jp[:, 16] = 0.1                        # 左肩滚
    jp[:, 23] = -0.1                       # 右肩滚
    sig = segment_signature(jp)
    chk("signature: 膝 median/max 取自 idx3/idx9",
        abs(sig["knee_left_median"] - float(np.median(jp[:, 3]))) < 1e-12
        and abs(sig["knee_left_max"] - 1.5) < 1e-12
        and abs(sig["knee_right_max"] - 1.2) < 1e-12)
    chk("signature: 肩滚 median 观察项", sig["shoulder_roll_left_median"] == 0.1
        and sig["shoulder_roll_right_median"] == -0.1)

    # ③ 四分位带
    b = quartile_band(np.arange(0, 101, 1.0))
    chk("band: Q1/median/Q3 = 25/50/75", (b["q1"], b["median"], b["q3"]) == (25.0, 50.0, 75.0)
        and b["min"] == 0.0 and b["max"] == 100.0)
    chk("band: 闭区间判定", in_band(25.0, b) and in_band(75.0, b)
        and not in_band(24.999, b) and not in_band(75.001, b))

    # ④ B4-lite 标签
    chk("b4lite label: p90/max 两档", b4lite_label(0.19) == "within_p90"
        and b4lite_label(0.23) == "within_max" and b4lite_label(0.30) == "over_max"
        and b4lite_label(None) == "n/a")

    # ⑤ 选样：3 档 × 2 段、演员互斥、确定性、贴近档中位
    rows = []
    for i in range(30):
        rows.append({"v_med": 0.1 + 0.05 * i, "actor_uid": f"A{i:03d}",
                     "csv_path": f"/x/{i:02d}.csv", "stem": f"s{i:02d}"})
    picks, edges, info = pick_bands(rows, per_band=2, n_bands=3)
    chk("pick: 3 档 × 2 = 6 段", len(picks) == 6 and [p["band"] for p in picks] ==
        [0, 0, 1, 1, 2, 2])
    chk("pick: actor 全局互斥", len({p["actor_uid"] for p in picks}) == 6)
    chk("pick: 确定性（重跑同结果）",
        [p["stem"] for p in pick_bands(rows, 2, 3)[0]] == [p["stem"] for p in picks])
    chk("pick: 档内取最接近档中位（含并列取小路径）",
        picks[0]["stem"] == min([r["stem"] for r in rows[:10]],
                                key=lambda s: abs(float(s[1:]) - 4.5)))
    chk("pick: 边界 half-open（30 段三等分各 10）",
        [i["n_candidates"] for i in info] == [10, 10, 10])
    # 演员耗尽 -> 只能选到 1 段，short 如实标记（不静默放宽演员互斥）
    rows2 = [dict(r, actor_uid="SAME") for r in rows]
    picks2, _, info2 = pick_bands(rows2, per_band=2, n_bands=3)
    chk("pick: actor 耗尽时只选 1 段且 short 标记", len(picks2) == 1
        and all(i["short"] for i in info2)
        and [i["n_skipped_actor_dup"] for i in info2] == [9, 10, 10]
        and [i["n_picked"] for i in info2] == [1, 0, 0])

    # ⑥ 路径解析（不存在即 None，不猜）
    chk("resolve: 不存在 -> None",
        resolve_csv_path("g1/csv/999999/nope.csv", os.getcwd(), os.getcwd()) is None)
    tmpf = os.path.join(tempfile.gettempdir(), "selftest_d059_tmp.csv")
    try:
        open(tmpf, "w").close()
        chk("resolve: 绝对路径直通",
            resolve_csv_path(tmpf, os.getcwd(), os.getcwd()) == os.path.abspath(tmpf))
        chk("resolve: ds_root 拼接分支",
            resolve_csv_path(os.path.basename(tmpf), os.path.dirname(tmpf),
                             os.getcwd()) == os.path.abspath(tmpf))
    finally:
        if os.path.isfile(tmpf):
            os.remove(tmpf)

    # ⑦ 逐段带内标记 + 双语义门（band / envelope）
    s = {"knee_left_median": 25.0, "knee_left_max": 75.0,
         "knee_right_median": 25.0, "knee_right_max": 75.0}
    chk("gate: 四项全在带 -> True", all(in_band(s[k], b) for k in GATE_STATS))
    sb = {k: dict(b) for k in GATE_STATS}  # 合成带：q1=25 q3=75 min=0 max=100
    g_mid = gate_segment({k: 50.0 for k in GATE_STATS}, sb, "band")
    g_env_mid = gate_segment({k: 50.0 for k in GATE_STATS}, sb, "envelope")
    chk("gate_segment: 典型值 band/envelope 双过",
        g_mid["hard_ok"] and g_env_mid["hard_ok"] and not g_mid["band_warnings"])
    # 90 在四分位带外（>75）但在总体包络内（<100）——重锚语义的关键分支
    sig90 = {k: 90.0 for k in GATE_STATS}
    g_band_90 = gate_segment(sig90, sb, "band")
    g_env_90 = gate_segment(sig90, sb, "envelope")
    chk("gate_segment: 带外包络内 -> band 硬门 FAIL、envelope 硬门 PASS 且记 warning",
        (not g_band_90["hard_ok"]) and g_env_90["hard_ok"]
        and len(g_env_90["band_warnings"]) == len(GATE_STATS)
        and all(g_env_90["band_margin"][k] < 0 for k in GATE_STATS))
    # 100.2 超出包络上界+ε(0.05) —— envelope 硬门必须拦
    sig_bad = {k: 100.2 for k in GATE_STATS}
    g_env_bad = gate_segment(sig_bad, sb, "envelope")
    chk("gate_segment: 包络外 -> envelope 硬门 FAIL（粗错拦截）",
        (not g_env_bad["hard_ok"])
        and all(g_env_bad["envelope_margin"][k] < 0 for k in GATE_STATS))
    chk("envelope_margin: 边界内 ε 内仍算过 / 越界为负",
        envelope_margin(100.04, sb["knee_left_max"], 0.05) > 0
        and envelope_margin(100.06, sb["knee_left_max"], 0.05) < 0
        and envelope_margin(-0.04, sb["knee_left_max"], 0.05) > 0)

    # ⑧ npz 侧段速（D048n bwd 口径：v[0]=v[1]）+ 方向启发式
    vm_bwd, _ = speed_stats_bwd(np.stack([np.arange(0, 51, 1.0), np.zeros(51)], axis=1) / 100.0, 50.0)
    chk("speed_stats_bwd: 1cm/帧@50fps -> 0.5 m/s", abs(vm_bwd - 0.5) < 1e-12, f"v_med={vm_bwd}")
    chk("direction_of: up/down/unknown",
        direction_of("come_up_50cm_box_R_004__A356") == "up"
        and direction_of("come_down_50cm_box_R_002__A389") == "down"
        and direction_of("neutral_come_down_50cm_box_R_001__A001") == "down"
        and direction_of("dancing_routine_V001_001__A002") == "unknown")

    # ⑨ --per-tier 4 -> 3 档 × 4 = 12 段、actor 全局互斥
    rows40 = [{"v_med": 0.05 * i, "actor_uid": f"B{i:03d}", "csv_path": f"/y/{i:02d}.csv",
               "stem": f"t{i:02d}"} for i in range(40)]
    picks12, _, info12 = pick_bands(rows40, per_band=4, n_bands=3)
    chk("pick(per-tier=4): 12 段 + actor 互斥 + 三档各 4",
        len(picks12) == 12 and len({p["actor_uid"] for p in picks12}) == 12
        and [i["n_picked"] for i in info12] == [4, 4, 4])

    # ⑩ replay 导出格式（D048r R1 兼容契约）
    seg_file = os.path.join(tempfile.gettempdir(), "selftest_d059_segments.json")
    try:
        fake_doc = {"counts": {"selection_pool": 40},
                    "methods": {"selection": "selftest", "n_bands": 3, "per_tier": 4},
                    "speed": {"band_edges": [0.1, 0.2]}}
        fake_rows = [{"stem": f"t{i:02d}", "npz_path": f"/y/npz/t{i:02d}.npz",
                      "v_med_npz": 0.1 * i, "v_p90_npz": 0.15 * i, "n_rows_enc": 100 + i}
                     for i in range(12)]
        export_replay_segments(fake_doc, fake_rows, seg_file, "/y")
        got = json.load(open(seg_file, encoding="utf-8"))
        need_top = {"experiment", "purpose", "segments"}
        need_seg = {"stem", "npz_path", "v_med", "frames", "fps"}
        chk("replay 导出: 顶层/段字段满足 R1 兼容契约",
            need_top <= set(got) and len(got["segments"]) == 12
            and all(need_seg <= set(s) for s in got["segments"])
            and got["n_picked"] == 12 and got["fps_default"] == 50.0)
    finally:
        if os.path.isfile(seg_file):
            os.remove(seg_file)

    # ⑪ 门报表 None 安全：manifest 无 roundtrip_mae/lattice_rate（转换未带
    #    --roundtrip）时 gate 报表不得 TypeError（reviewer should-fix）
    gdir = tempfile.mkdtemp(prefix="selftest_d059_gate_")
    try:
        npz_p = os.path.join(gdir, "t00.npz")
        jp_f = np.zeros((20, 29))
        jp_f[:, 3], jp_f[:, 9] = 0.30, 0.40
        np.savez(npz_p, jp_mj=jp_f, trans_m=np.zeros((20, 3)), tokens=np.zeros((20, 64)))
        man_p = os.path.join(gdir, "manifest.json")
        with open(man_p, "w", encoding="utf-8") as fh:  # 刻意缺 roundtrip/lattice 字段
            json.dump([{"stem": "t00", "actor": "A000", "npz_path": npz_p,
                        "n_rows_src": 30, "n_rows_enc": 20}], fh)
        bdoc = {"counts": {"selection_pool": 1},
                "methods": {"selection": "selftest", "n_bands": 3, "per_tier": 2},
                "speed": {"band_edges": None},
                "signature_band": {"band": {k: dict(b) for k in GATE_STATS},
                                   "empirical": {"rate_all4_in_band": 0.5}},
                "selected": [{"stem": "t00", "band_name": "slow", "direction": "up",
                              "v_med": 0.1}]}
        bj_p = os.path.join(gdir, "band.json")
        with open(bj_p, "w", encoding="utf-8") as fh:
            json.dump(bdoc, fh)
        ns = argparse.Namespace(band_json=bj_p, gate_manifest=man_p,
                                gate_out=os.path.join(gdir, "gate.json"),
                                segments_json_out=os.path.join(gdir, "seg.json"),
                                gate_mode="band", env_margin=0.05)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summ_none = gate_main(ns)
        txt = buf.getvalue()
        chk("gate 报表 None 安全: manifest 无 roundtrip 字段不炸、打 n/a、产物齐",
            summ_none["n_converted"] == 1 and "n/a" in txt
            and os.path.isfile(os.path.join(gdir, "gate.json"))
            and os.path.isfile(os.path.join(gdir, "seg.json"))
            and summ_none["rows"][0]["roundtrip_label"] == "n/a",
            f"rt_label={summ_none['rows'][0]['roundtrip_label']}")
    finally:
        shutil.rmtree(gdir, ignore_errors=True)

    n_fail = sum(1 for _, c in ok if not c)
    print(f"\n[selftest] {len(ok) - n_fail}/{len(ok)} passed")
    return n_fail == 0


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--content-type", default=DEFAULT_CONTENT_TYPE)
    ap.add_argument("--ds-root", default=DEFAULT_DS_ROOT)
    ap.add_argument("--csv-root", default=os.path.join(DEFAULT_DS_ROOT, "g1", "csv"))
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--per-segment-csv", default=None)
    ap.add_argument("--list-out", default=None)
    ap.add_argument("--fps", type=float, default=FPS_SRC_DEFAULT)
    ap.add_argument("--min-frames", type=int, default=240,
                    help="选池源行数下限（默认 240 = 2s@120fps，D044 同口径）")
    ap.add_argument("--per-tier", type=int, default=None,
                    help="速度档每档段数（默认 2；G0-③ K≥12 用 --per-tier 4）")
    ap.add_argument("--per-band", type=int, default=None,
                    help="--per-tier 的旧名别名（二者都给时以 --per-tier 为准）")
    ap.add_argument("--n-bands", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="只扫前 N 段（冒烟用）")
    ap.add_argument("--gate-manifest", default=None,
                    help="门判定模式：读转换产物 manifest.json")
    ap.add_argument("--band-json", default=None, help="门判定模式：步骤 1 的 JSON")
    ap.add_argument("--gate-out", default=None)
    ap.add_argument("--gate-mode", default="band", choices=("band", "envelope"),
                    help="band=联合四分位带硬门（预注册 §5s 字面口径，保留向后可比）；"
                         "envelope=总体包络硬拦 + 四分位带外 warning（2026-09-17 主会话"
                         "重锚口径，照 D057 G2 454ab84 口径修正先例）——"
                         "09-17 重锚口径=envelope（D057 语义），默认 band 只为向后可比，"
                         "新判读请显式 --gate-mode envelope")
    ap.add_argument("--env-margin", type=float, default=0.05,
                    help="envelope 硬门余量 ε（rad，默认 0.05）")
    ap.add_argument("--segments-json-out", default=None,
                    help="gate 模式：D048r R1 replay 兼容 segments-json 输出路径"
                         "（默认 <gate-out 同目录>/d059_climb_segments.json）")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    # --per-tier 生效解析（--per-band 兼容别名；缺省 2 = 初版行为）
    args.per_tier = (args.per_tier if args.per_tier is not None
                     else (args.per_band if args.per_band is not None else 2))
    # 相对路径按调用 cwd 归一（用法约定 cwd=执行根 ~/ros2_data/apt_g1）
    for k in ("parquet", "ds_root", "csv_root", "out_dir"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    if args.gate_manifest:
        if not args.band_json:
            ap.error("--gate-manifest 需要 --band-json")
        gate_main(args)
        return
    scan_select(args)


if __name__ == "__main__":
    main()
