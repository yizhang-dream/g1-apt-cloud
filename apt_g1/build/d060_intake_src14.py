"""D060 §5t G1 集成驱动：源 1（planner 剧本）+ 源 4（闭环记录）切窗 → 四源装配 → 消费者测试。

预注册 = `refine-logs/ds/DS_CONTINUOUS_EXECUTION_PLAN.md` §5t（窗格式冻结）与其中
内部 gate **G1 格式冒烟**（四源各 ≥100 窗 + 统一 loader + 消费者测试 + 字段契约断言）。
窗格式的**唯一实现**是 `d060_windows.py`（本文件只 import 复用，不改它、不另开口径）；
本文件是 G1 的集成侧：把两条"另一线"产出的源（源 1 / 源 4）切成同格式窗，做四源
统一装配台账，并驱动消费者测试取证。

源 1 = planner 命令剧本（`data/d060/planner_scripts_smoke/`，每点若干 npz）
--------------------------------------------------------------
每脚本 npz：tokens (n,64) f32@50Hz / root_pos (n,3) PRE-step / root_quat (n,4) wxyz /
jp_mujoco (n,29) / jp_isaaclab / meta JSON。口径（生成器 docstring 已在案）：
root_pos[t]/jp[t] 是**执行 tokens[t] 之前**的态 → 与语料一致，直接喂
`windows_from_arrays`（不需要错位）。
command 取 **planner 名义值**（§5t："command 存 planner 名义值、meta 并记该点实测净速"）：
  target_vel = nominal.target_vel（真值）; mode = nominal.mode_id（真值）;
  movement_direction = atan2(y, x)（nominal.movement_direction 是**体坐标单位向量**
    [x,y,z]，官方 hpp MovementState 同构；命令槽是**标量**，本驱动用体坐标水平偏航角
    （弧度）编码：front=0.0、left=+π/2、right=-π/2；原向量与方向名同记进窗 meta，
    编码规则写在 manifest.format.command_encoding——**编码是我们的约定，非官方字段**）
  height = nominal.height（本批 = -1 哨兵 → has_truth=False）
逐窗 meta 同时记 **实测**：speedA（窗内实测根速中位，与源 2/3 同口径）+ 脚本级
realized.net_speed_xy / ratio_to_nominal（D033 衰减差逐点实测，不假设）。
intent_tokens = 剧本 tokens 自身，intent_source="planner_script"；terrain=plane；
state_source="kinematics"（jp 实测运动学，与源 2/3 同）。

源 4 = 闭环回放记录（`data/d060/s4_records/{records_n004,records_n008,records_fallseg}`）
----------------------------------------------------------------------------------------
每 run npz（`isaac/replay_token_terrain_d059.py` 增量⑦ 产物）：q_des (n,29) 当步下发的
关节位置目标 / root_pos (n,3) step **后** / root_quat (n,4) step 后 / tokens_exec (n,64)
当步喂 decoder 的 token / root_pos_pre,root_quat_pre step 前 / fall_step 标量(-1=未摔) /
meta JSON（stem/seed/terrain/terrain_noise/v_med/token_source/steps/survived…）。
**state 语义与源 1/2/3 不同（必须显式登记，不静默）**：
  state = cmd 关节目标 q_des + 根位姿（step 后），**不是实测关节角** →
  窗 meta / manifest 记 state_source="q_des_cmd_root"（源 1/2/3 = "kinematics"）。
  step 后根位姿与 q_des[t] 是"当步动作 + 当步结果"的因果对（记录器 playback_window
  口径），与源 1 的 PRE-step 口径差一帧——同记 state_align="post_step" 供训练侧裁量；
  源 npz 里另有 *_pre 数组，若要改口径不必重跑本卡（改本文件一处即可重切）。
command = None → 逐窗 speedA 锚定标签（窗内实测根速中位，源 2/3 同口径）；
intent_tokens = tokens_exec，intent_source="oracle_replay"；
terrain = make_terrain(meta.terrain, noise=meta.terrain_noise)（meta 缺 terrain 才回落
rough_paper，回落在 manifest 里计数，不静默）；terrain=plane 时不塞 noise 参数。
**摔倒 run**：fall_step 之后的数据是复位前的最后有效步（记录器已不记 done 步）——
窗只取 fall_step 前的完整窗，不足 win 的尾部丢弃并计数，窗 meta 记 fall_step /
frames_after_fall / fall_clip_applied；run 级 survived / completed 一并入账。
防泄漏：run stem 命中演员 held-out 划分表（`data/ds_bones/actor_split_d060.json`）或为
`_M` 镜像段 → 该 run 的窗 split 标 "heldout"（§5t：held-out 演员段与镜像段不进训练窗），
计数写进 manifest，不静默改判。

四个模式
--------
  src1      planner 剧本 → 窗（默认出 `data/d060/windows_src1/`）
  src4      闭环记录 → 窗（默认出 `data/d060/windows_src4/`；可 --wait-minutes 轮询等落盘）
  assembly  四源 manifest → `data/d060/g1_assembly/assembly_manifest.json`（**只引用不合并**：
            分片留各源目录，loader 天然按 manifest 读；本文件给四源计数/帧数/来源统计/
            speedA 分位/terrain 分布/state_source 口径分列 + 逐源对账结论）
  consumer  对四个 manifest 各跑 `d060_consumer_stub.py`（默认 16 窗 / 1 步 SGD），
            汇总四个 PASS/FAIL + loss 表 → `consumer_summary.json`
  selftest  合成 fixture 全链路自检（纯 numpy：源 1/源 4 切窗 + 摔倒截断 + 划分 + 装配
            对账 + 负例；不跑 torch）

用法（本仓 = apt_g1/build/；lab-ts 部署根 = ~/ros2_data/apt_g1 平铺顶层，两条 sys.path
都自洽——numpy-only + d060_windows 同目录/同盘 import）
    python d060_intake_src14.py selftest
    python d060_intake_src14.py src1 --npz-dirs data/d060/planner_scripts_smoke \
        --out-dir data/d060/windows_src1
    python d060_intake_src14.py src4 --records-dir data/d060/s4_records \
        --out-dir data/d060/windows_src4 --wait-minutes 360 --poll-seconds 600
    python d060_intake_src14.py assembly --out-dir data/d060/g1_assembly
    python d060_intake_src14.py consumer --out-dir data/d060/g1_assembly --limit 16 --steps 1

D060 / §5t G1；本文件不改 `d060_windows.py`（格式脊柱）与 `d060_consumer_stub.py`（消费侧）。
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import subprocess
import sys
import time

import numpy as np

# ------------------------------------------------------------------ import 脊柱（多布局）
def import_d060():
    """import 窗格式脊柱（执行根平铺 / 本仓 build 分域 / 包导入 三种 sys.path 布局）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.path.dirname(here), os.getcwd()):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    last = None
    for name in ("d060_windows", "build.d060_windows", "apt_g1.build.d060_windows"):
        try:
            return importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 —— 逐候选回退
            last = exc
    raise ImportError(f"d060_windows 不可 import（最后错误 {type(last).__name__}: {last}）")


W = import_d060()

SCRIPT_PATH = os.path.abspath(__file__)

# ------------------------------------------------------------------ 常量（口径登记处）
SRC1_ID = "src1_planner_script"
SRC2_ID = "src2_climb_corpus"
SRC3_ID = "src3_official_flat"
SRC4_ID = "src4_closed_loop"
STATE_SOURCE_KIN = "kinematics"
STATE_SOURCE_QDES = "q_des_cmd_root"
INTENT_PLANNER = getattr(W, "INTENT_SOURCE_PLANNER", "planner_script")
# 源 4 的 intent 来源：脊柱另有 INTENT_SOURCE_CLOSED_LOOP="closed_loop"（未被任何生产者使用
# 的死常量）。G1 派发口径定为 "oracle_replay" 且已写进 169 个源 4 窗的 meta/manifest —— 改值
# = 改判据（须 owner 裁量），故此处保留该值，并把与脊柱常量的分歧显式登记、纳入 selftest 断言
# （见 selftest 5），不让它变成静默的第二份口径。
INTENT_ORACLE = "oracle_replay"
# 源 1 npz 键（gen_planner_scripts_d060.py 产物）
SRC1_KEYS = {"tokens": "tokens", "jp": "jp_mujoco", "quat": "root_quat",
             "trans": "root_pos"}
# 源 4 npz 键（replay_token_terrain_d059.py 增量⑦：RECORD_KEYS）
SRC4_KEYS = {"tokens": "tokens_exec", "jp": "q_des", "quat": "root_quat",
             "trans": "root_pos"}
SRC4_DIR_TAGS = ("records_n004", "records_n008", "records_fallseg")
SRC4_EXPECT = {"records_n004": 12, "records_n008": 12, "records_fallseg": 8}
DEFAULT_SRC1_DIR = "data/d060/planner_scripts_smoke"
DEFAULT_SRC1_OUT = "data/d060/windows_src1"
DEFAULT_SRC4_RECORDS = "data/d060/s4_records"
DEFAULT_SRC4_OUT = "data/d060/windows_src4"
DEFAULT_ASSEMBLY_OUT = "data/d060/g1_assembly"
DEFAULT_SPLIT_JSON = "data/ds_bones/actor_split_d060.json"
# 四源 manifest 缺省位置（相对 data 根；仓库/服务器同布局）
DEFAULT_SOURCES = [
    (SRC1_ID, "src1", "data/d060/windows_src1/manifest.json"),
    (SRC2_ID, "src2", "data/ds_bones/g1_d060_climb/windows/manifest.json"),
    (SRC3_ID, "src3", "data/ds_bones/d060_windows_src3/manifest.json"),
    (SRC4_ID, "src4", "data/d060/windows_src4/manifest.json"),
]
G1_MIN_WINDOWS = 100          # G1 判据：四源各 ≥100 窗
G5_FRAME_GATE = 1500000       # §5t 产量门①（≈8.3h@50Hz）；G1 只报数不做 G5 判读


# ------------------------------------------------------------------ 小工具
def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(doc, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    return path


def quantiles(values, prefix="speedA"):
    """标量列表 → 分位块（与脊柱 manifest.speedA_dist 同构，含三分位界）。"""
    a = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
    if a.size == 0:
        return None
    return {"n": int(a.size), "min": float(a.min()),
            "q1": float(np.percentile(a, 25)), "median": float(np.median(a)),
            "q3": float(np.percentile(a, 75)), "max": float(a.max()),
            "tertile_edges": [float(np.percentile(a, 100 / 3)),
                              float(np.percentile(a, 200 / 3))],
            "label": prefix}


DIST_FIELDS = ("n", "min", "q1", "median", "q3", "max")


def dist_fields_diff(declared, computed, tol=1e-9):
    """两个分位块**逐字段**比对（n 精确、min/q1/median/q3/max 与 tertile_edges 逐元素 tol 内）。

    返回差异列表（空 = 一致）。declared=None（源 manifest 无自报块）→ 返回 []（无可对账，
    由调用方另记"未声明"）。**必须全字段比**：只比 n+median 在"同一生产者算两遍"时近乎
    恒真、掩盖系统性偏移（如逐窗标签整体平移只动 min/max/分位界）。
    """
    if declared is None:
        return []
    if not computed:
        return [{"field": "*", "declared": "有自报块", "computed": None}]
    diff = []
    for k in DIST_FIELDS:
        if k == "n":
            if int(declared.get("n", -1)) != int(computed["n"]):
                diff.append({"field": "n", "declared": declared.get("n"),
                             "computed": computed["n"]})
            continue
        try:
            a, b = float(declared.get(k, float("nan"))), float(computed[k])
        except (TypeError, ValueError):
            diff.append({"field": k, "declared": declared.get(k), "computed": "非数值"})
            continue
        if not (abs(a - b) <= tol):
            diff.append({"field": k, "declared": a, "computed": b, "abs_diff": abs(a - b)})
    td, tc = declared.get("tertile_edges"), computed.get("tertile_edges")
    if td is None or tc is None or len(td) != len(tc):
        diff.append({"field": "tertile_edges", "declared": td, "computed": tc})
    else:
        for i, (a, b) in enumerate(zip(td, tc)):
            if not (abs(float(a) - float(b)) <= tol):
                diff.append({"field": f"tertile_edges[{i}]", "declared": float(a),
                             "computed": float(b), "abs_diff": abs(float(a) - float(b))})
    return diff


def distribution(keys):
    """键列表 → 计数 dict（None 归 "None"）。"""
    out = {}
    for k in keys:
        kk = "None" if k is None else str(k)
        out[kk] = out.get(kk, 0) + 1
    return dict(sorted(out.items()))


def read_npz_arrays(path, keys):
    """按 keys（role->npz key）读数组；缺键即 KeyError（不猜、不静默）。"""
    out = {}
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        miss = [v for v in keys.values() if v not in files]
        if miss:
            raise KeyError(f"{os.path.basename(path)} 缺键 {miss}（有 {sorted(files)}）")
        for role, k in keys.items():
            out[role] = np.asarray(z[k])
        out["_files"] = sorted(files)
    return out


def to_np_int(x):
    """npz 标量（含 0-d 数组）→ python int；None/缺失 → None。"""
    if x is None:
        return None
    a = np.asarray(x).reshape(-1)
    if a.size == 0:
        return None
    return int(a[0])


def fall_clip(n_frames, fall_step, win=W.WIN, stride=W.STRIDE):
    """摔倒截断：只取 fall_step 之前的完整帧（返回 (n_use, info)，info 逐项入账）。

    fall_step = -1/None 表未摔（-1 是脚手架哨兵；0 是合法步号）。记录器口径：done 步
    不可记（auto-reset 污染），故源 4 通常 n_frames <= fall_step；源 1 的 npz 保留到
    摔倒当步的 PRE 态（n_frames 可能 = fall_step+1）→ 两种情形都按"只取 fall_step 前"
    处理，并把"记录是否含 fall_step 当步/之后"如实记进 info。
    `windows_dropped_by_fall_clip` 用**本次运行的 win/stride**（不是脊柱缺省）算，
    否则 G2-G4 换 stride 时该计数会系统性错报。
    """
    n = int(n_frames)
    info = {"record_frames": n, "fall_step": None, "fall_clip_applied": False,
            "frames_after_fall": 0, "windows_dropped_by_fall_clip": 0,
            "windows_dropped_win": int(win), "windows_dropped_stride": int(stride),
            "record_includes_fall_step_or_later": False}
    if fall_step is None:
        info["fall_step_null_reason"] = "meta 无 fall_step 字段（缺失≠未摔，如实记）"
        return n, info
    fs = int(fall_step)
    info["fall_step"] = fs
    if fs < 0:
        return n, info
    n_use = min(n, fs)
    info["fall_clip_applied"] = bool(n_use < n)
    info["frames_after_fall"] = n - n_use
    info["record_includes_fall_step_or_later"] = bool(n > fs)
    info["windows_dropped_by_fall_clip"] = (
        len(W.slice_starts(n, win=win, stride=stride))
        - len(W.slice_starts(n_use, win=win, stride=stride)))
    return n_use, info


def resolve_split(split_json, verbose=True):
    """演员划分表（防泄漏）：缺失/None 即"无划分"（源 1 无演员；源 4 应给真表）。"""
    if not split_json or not os.path.isfile(split_json):
        if verbose and split_json:
            print(f"[split] 警告：{split_json} 不存在 → 不做 held-out 判定（全部记 train）")
        return {"heldout": set(), "train": set(), "doc": None, "path": None}
    return W.load_actor_split(split_json)


def seg_split_tag(stem, split):
    """段级 tag："heldout" = held-out 演员 或 `_M` 镜像段（§5t 两者都不进训练窗）。"""
    if split["heldout"] and W.is_heldout(stem, split):
        return "heldout"
    if W.is_mirror_stem(stem):
        return "heldout"
    return "train"


def segment_reconciliation(rec, win=W.WIN):
    """段级对账（脊柱同规则）：windows×win = 覆盖并集 + 重叠冗余；frames = 覆盖 + 未覆盖。"""
    visits = int(rec.get("windows", 0)) * int(win)
    cov = int(rec.get("frames_covered_unique") or 0)
    n_use = int(rec.get("frames_used") or 0)
    return {"windows": int(rec.get("windows", 0)), "window_frames_visits": visits,
            "frames_used": n_use,
            "check": (visits == cov + int(rec.get("overlap_redundancy") or 0)
                      and n_use == cov + int(rec.get("frames_uncovered") or 0))}


# ------------------------------------------------------------------ 源 1：planner 剧本
def src1_command(meta):
    """planner 名义值 → command dict + 记账块（见模块 docstring 的编码约定）。"""
    nom = meta["nominal"]
    tv = float(nom["target_vel"])
    mdir = list(nom.get("movement_direction") or [1.0, 0.0, 0.0])
    ang = float(np.arctan2(float(mdir[1]), float(mdir[0])))
    mode = float(nom["mode_id"])
    height_raw = nom.get("height", None)
    height = float(height_raw) if height_raw is not None else float(W.SENTINEL)
    has_truth = {"target_vel": True, "movement_direction": True, "mode": True,
                 "height": bool(height != float(W.SENTINEL))}
    cmd = W.make_command(tv, ang, mode, height, has_truth=has_truth)
    info = {"cmd_source": "planner_nominal",
            "nominal_target_vel": tv, "nominal_target_vel_sent": nom.get("target_vel_sent"),
            "nominal_mode_id": int(nom["mode_id"]), "nominal_mode_name": nom.get("mode_name"),
            "nominal_height_raw": height_raw,
            "nominal_direction_name": nom.get("direction_name"),
            "movement_direction_xyz": [float(v) for v in mdir],
            "movement_direction_encoding": ("atan2(y,x) rad，体坐标水平偏航"
                                            "（front=0, left=+pi/2, right=-pi/2）"),
            "facing_direction_xyz": nom.get("facing_direction"),
            "ctrl_hz": nom.get("ctrl_hz"), "replan_hz": nom.get("replan_hz")}
    return cmd, info


def cut_src1_script(path, *, terrain, split, win, stride, fps, stem_rel=None):
    """单剧本 npz → (windows, seg_rec, err)。失败如实记 error（不猜、不跳过静默）。

    stem = 相对 `--npz-dirs` 的路径（分隔符 → `__`），如
    `p00_vel0.6_SLOW_WALK_front__script_000`——同一 `script_000` 在多命令点下重名，
    必须带点目录；不带盘上路径前缀（可读、可搬移）。
    """
    rel_key = (stem_rel if stem_rel is not None else os.path.basename(path)[:-4])
    stem = rel_key.replace("\\", "/").replace("/", "__")
    try:
        arr = read_npz_arrays(path, SRC1_KEYS)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
        rec = {"stem": stem, "npz_path": os.path.abspath(path),
               "npz_md5": W.md5_file(path), "point_tag": meta.get("point_tag"),
               "script_idx": meta.get("script_idx"), "seed": meta.get("seed"),
               "schema": meta.get("schema"), "terrain": terrain["type"]}
        cmd, info = src1_command(meta)
        real = meta.get("realized") or {}
        n_full = len(arr["tokens"])
        n_use, finfo = fall_clip(n_full, real.get("fall_step"), win=win, stride=stride)
        rec.update(finfo)
        rec["frames"] = n_full
        rec["frames_used"] = n_use
        rec["n_steps_requested"] = meta.get("n_steps_requested")
        rec["n_steps_planned"] = meta.get("n_steps_planned")
        rec["terminated_cause"] = real.get("terminated_cause")
        rec["script_fall"] = bool(real.get("fall")) or (
            real.get("fall_step") is not None and int(real.get("fall_step") or -1) >= 0)
        rec["realized"] = {"net_speed_xy": real.get("net_speed_xy"),
                           "ratio_to_nominal": real.get("ratio_to_nominal"),
                           "disp_xy": real.get("disp_xy"),
                           "path_len_xy": real.get("path_len_xy"),
                           "straightness": real.get("straightness"),
                           "h_min": real.get("h_min"), "h_end": real.get("h_end"),
                           "n_steps": real.get("n_steps")}
        rec["nominal"] = {k: info[k] for k in
                          ("nominal_target_vel", "nominal_mode_id", "nominal_mode_name",
                           "nominal_direction_name", "movement_direction_xyz",
                           "nominal_height_raw")}
        rec["command"] = {"cmd": cmd["cmd"], "has_truth": cmd["has_truth"]}
        rec["command_encoding"] = info["movement_direction_encoding"]
        rec["split"] = seg_split_tag(rec["stem"], split)
        rec["actor"] = W.parse_actor(rec["stem"])
        rec["mirror"] = W.is_mirror_stem(rec["stem"])
        meta_extra = {"state_source": STATE_SOURCE_KIN,
                      "state_align": "pre_step",
                      "state_source_note": ("jp_mujoco/root_* 为执行 tokens[t] 之前的实测"
                                            "态（生成器口径），与源 2/3 同为实测运动学"),
                      "src": SRC1_ID, "npz_path": rec["npz_path"],
                      "npz_md5": rec["npz_md5"], "point_tag": rec.get("point_tag"),
                      "script_idx": rec.get("script_idx"), "seed": rec.get("seed"),
                      "terminated_cause": rec["terminated_cause"],
                      "script_fall": rec["script_fall"], "fall_step": finfo["fall_step"],
                      "fall_clip_applied": finfo["fall_clip_applied"],
                      "frames_after_fall": finfo["frames_after_fall"],
                      "record_frames": finfo["record_frames"],
                      "speedA_source": ("measured：窗内实测根速中位（XY 后向差分×fps，"
                                        "整段差分、窗帧取中位；与源 2/3 同口径）。"
                                        "command.target_vel 是 planner 名义值，两者分开记"),
                      **info,
                      "realized_net_speed_xy": real.get("net_speed_xy"),
                      "realized_ratio_to_nominal": real.get("ratio_to_nominal")}
        ws = W.windows_from_arrays(arr["tokens"][:n_use], arr["jp"][:n_use],
                                   arr["quat"][:n_use], arr["trans"][:n_use],
                                   terrain_desc=terrain, command=cmd,
                                   intent_tokens=arr["tokens"][:n_use],
                                   win=win, stride=stride, fps=fps, stem=rec["stem"],
                                   intent_source=INTENT_PLANNER, meta_extra=meta_extra)
        # 固定 command 时脊柱不写 speedA（那是"无命令源"的标签）——本驱动显式补实测 speedA
        trans_use = arr["trans"][:n_use]
        for w in ws:
            m = w["meta"]
            m["speedA"] = W.window_speedA(trans_use, m["frame_start"], m["frame_end"],
                                          fps=fps)
        starts = [w["meta"]["frame_start"] for w in ws]
        rec["windows"] = len(ws)
        rec["frames_covered_unique"] = W.covered_unique(starts, win, n_use)
        rec["frames_uncovered"] = int(n_use) - rec["frames_covered_unique"]
        rec["window_frames_visits"] = len(ws) * int(win)
        rec["overlap_redundancy"] = rec["window_frames_visits"] - rec["frames_covered_unique"]
        rec["frames_tail_incomplete"] = rec["frames_uncovered"]  # = frames_uncovered：
        # 被丢弃的未覆盖尾帧（不足 win 的部分）；摔后帧单列在 frames_after_fall，不混入此项
        if ws:
            sp = np.asarray([w["meta"]["speedA"] for w in ws])
            rec["speedA_windows"] = {"min": float(sp.min()),
                                     "median": float(np.median(sp)),
                                     "max": float(sp.max())}
            rec["frame_start_first"] = starts[0]
            rec["frame_end_last"] = ws[-1]["meta"]["frame_end"]
        return ws, rec, None
    except Exception as exc:  # noqa: BLE001 —— 失败段如实记，不静默跳过
        return [], {"stem": stem, "npz_path": os.path.abspath(path),
                    "error": f"{type(exc).__name__}: {exc}"}, \
            f"{type(exc).__name__}: {exc}"


def run_src1_mode(args):
    terrain = W.make_terrain("plane")
    split = resolve_split(args.split_json)
    fps, win, stride = float(args.fps), int(args.win), int(args.stride)
    t0 = time.time()
    files = []   # (npz 绝对路径, 相对所属 --npz-dirs 的相对路径)
    for d in args.npz_dirs:
        base = os.path.abspath(d)
        for p in sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)):
            ap = os.path.abspath(p)
            files.append((ap, os.path.relpath(ap, base)))
    seen, uniq = set(), []
    for ap, rel in files:
        if ap in seen:
            continue
        seen.add(ap)
        uniq.append((ap, rel))
    files = sorted(uniq)
    print(f"[src1] 剧本 npz {len(files)}（目录 {args.npz_dirs}）terrain="
          f"{terrain['type']} win={win} stride={stride}")
    all_w, tags, srecs, errs = [], [], [], []
    for p, rel in files:
        ws, rec, err = cut_src1_script(p, terrain=terrain, split=split, win=win,
                                       stride=stride, fps=fps, stem_rel=rel[:-4])
        srecs.append(rec)
        if err:
            errs.append({"npz_path": p, "error": err})
            continue
        all_w.extend(ws)
        tags.extend([rec["split"]] * len(ws))
    shards, wrecs = W.write_shards(all_w, os.path.join(args.out_dir, "npz"),
                                   shard_size=args.shard_size, prefix="src1",
                                   split_tags=tags,
                                   extra_manifest={"terrain": terrain,
                                                   "source_id": SRC1_ID,
                                                   "stride": stride, "fps": fps})
    doc = build_source_manifest(
        source_id=SRC1_ID, source_label="src1（planner 命令剧本）",
        terrain_default=terrain, win=win, stride=stride, fps=fps,
        segments=srecs, shards=shards, wrecs=wrecs, windows_raw=all_w,
        counts_extra={"scripts_found": len(files), "scripts_error": len(errs),
                      "windows_heldout": int(sum(1 for t in tags if t == "heldout"))
                      if tags else 0},
        format_extra={"command_encoding": (
            "command.movement_direction = atan2(y,x) rad（体坐标水平偏航；"
            "front=0, left=+pi/2, right=-pi/2）；原 [x,y,z] 向量与方向名记在窗 meta"
            "（movement_direction_xyz / nominal_direction_name）。command.target_vel = "
            "planner 名义值；窗 meta 另记 speedA（实测中位）与 realized.net_speed_xy"
            "（脚本级实测净速）+ ratio_to_nominal（D033 衰减差逐点实测）"),
            "state_source": STATE_SOURCE_KIN,
            "intent_source": INTENT_PLANNER,
            "speedA_role": ("实测锚定标签（与源 2/3 同）；本源 command 另有真值 → "
                            "target_vel 与 speedA 语义分离，训练侧自选")},
        inputs={"npz_dirs": [os.path.abspath(d) for d in args.npz_dirs],
                "npz_keys": SRC1_KEYS, "split_json": split.get("path"),
                "terrain": terrain, "script_glob": "**/*.npz"},
        elapsed=time.time() - t0)
    dump_manifest(doc, args.manifest_out or os.path.join(args.out_dir, "manifest.json"))
    c = doc["counts"]
    print(f"[src1] 窗 {c['windows']}（train {c['windows_train']}/heldout "
          f"{c['windows_heldout']}）帧 {c['window_frames']} 分片 {c['shards']} "
          f"错误 {len(errs)} 耗时 {c['elapsed_s']}s")
    if doc["speedA_dist"]:
        s = doc["speedA_dist"]
        print(f"[src1] speedA(实测) min {s['min']:.3f} / med {s['median']:.3f} / "
              f"max {s['max']:.3f}")
    print(f"[src1] -> {doc['manifest_path']}")
    if len(all_w) < G1_MIN_WINDOWS:
        print(f"[src1] [!] 窗数 {len(all_w)} < G1 判据 {G1_MIN_WINDOWS}")
    return doc


# ------------------------------------------------------------------ 源 4：闭环记录
def discover_record_tags(records_dir, tags=None):
    """records 目录 → {tag: [npz…]}；tags=None 时自动发现 `records*` 子目录。"""
    if tags:
        cand = list(tags)
    else:
        cand = sorted(d for d in os.listdir(records_dir)
                      if os.path.isdir(os.path.join(records_dir, d))
                      and d.startswith("records")) if os.path.isdir(records_dir) else []
    out = {}
    for tag in cand:
        d = os.path.join(records_dir, tag)
        out[tag] = sorted(glob.glob(os.path.join(d, "*.npz"))) if os.path.isdir(d) else []
    if not cand:  # 记录直接平铺在 records_dir
        out["records_flat"] = sorted(glob.glob(os.path.join(records_dir, "*.npz")))
    return out


def records_ready(counts, expect):
    """就绪判据：每个期望 tag 的文件数 ≥ 期望（tag 匹配 = 精确名 或 双向包含，容目录改名）；
    匹配到的 tag 不得为空目录；总数亦须达期望之和。返回 (ok, problems, over_expect)。"""
    problems, matched = [], set()
    total = int(sum(int(v) for v in counts.values()))
    if expect:
        for key, need in expect.items():
            m = [t for t in counts if t == key] or [t for t in counts
                                                    if key in t or t in key]
            matched.update(m)
            have = sum(int(counts[t]) for t in m)
            if have < int(need):
                problems.append(f"{key} {have}/{need}")
        need_total = int(sum(int(v) for v in expect.values()))
        if total < need_total:
            problems.append(f"总文件数 {total}/{need_total}")
    for t in sorted(matched):
        if int(counts[t]) == 0:
            problems.append(f"{t} 空目录（在等写入，不当作就绪）")
    extra = [f"{t}={counts[t]}" for t in sorted(counts)
             if t not in matched and int(counts[t]) > 0]
    return (not problems), problems, extra


def wait_for_records(records_dir, expect, *, wait_minutes, poll_seconds, tags=None,
                     trace_path=None, verbose=True):
    """轮询等源 4 记录落盘（默认每 10 分钟；两轮计数一致 = 稳定）；超时如实返回 WAITING。"""
    t_end = time.time() + max(0.0, float(wait_minutes) * 60.0)
    trace, prev, poll = [], None, 0
    while True:
        found = discover_record_tags(records_dir, tags=tags)
        counts = {t: len(v) for t, v in found.items()}
        ok, problems, extra = records_ready(counts, expect)
        stable = (prev is not None and counts == prev)
        poll += 1
        entry = {"poll": poll, "time": now_str(), "counts": counts, "ready": bool(ok),
                 "stable": bool(stable), "problems": problems, "over_expect": extra}
        trace.append(entry)
        if verbose:
            print(f"[src4-wait] poll {poll} {entry['time']} counts={counts} "
                  f"ready={ok} stable={stable} {problems or ''}", flush=True)
        if trace_path:
            dump_json({"records_dir": os.path.abspath(records_dir), "expect": expect,
                       "wait_minutes": wait_minutes, "poll_seconds": poll_seconds,
                       "polls": trace}, trace_path)
        if ok and stable:
            return {"status": "READY", "counts": counts, "polls": poll, "trace": trace,
                    "found": found}
        if time.time() >= t_end:
            return {"status": "WAITING", "counts": counts, "polls": poll, "trace": trace,
                    "found": found,
                    "note": (f"等待 {wait_minutes} 分钟未达就绪（期望 {expect}）——"
                             "如实报 WAITING，不用部分数据硬凑")}
        prev = counts
        time.sleep(max(1.0, float(poll_seconds)))


def src4_terrain(meta, registered):
    """meta.terrain/terrain_noise → terrain_desc（缺 terrain 才回落 rough_paper，落账）。"""
    t = meta.get("terrain")
    if t not in W.TERRAIN_TYPES:
        registered["terrain_fallback_from"] = t
        registered["terrain_fallback_to"] = "rough_paper"
        t = "rough_paper"
    noise = meta.get("terrain_noise")
    kw = {}
    if t == "rough_paper" and noise is not None:
        kw["noise"] = float(noise)
    elif t == "climbing_box" and meta.get("terrain_height") is not None:
        kw["height"] = float(meta["terrain_height"])
    return W.make_terrain(t, **kw)


def cut_src4_record(path, tag, *, split, win, stride, fps):
    """单 run npz → (windows, seg_rec, err)。摔倒截断 + state_source 口径登记。"""
    fstem = os.path.basename(path)[:-4]
    try:
        arr = read_npz_arrays(path, SRC4_KEYS)
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            meta = json.loads(str(z["meta"])) if "meta" in files else {}
            fall_raw = to_np_int(z["fall_step"]) if "fall_step" in files else None
        reg = {}
        terrain = src4_terrain(meta, reg)
        # 窗 stem = 记录目录 tag + **文件 stem**（记录器命名 = `<段 stem>__seed<k>`，含演员
        # `__A\d+`）+ 必要时补 meta.stem（防文件名与 meta 不同源时丢身份/镜像标记）→
        # 跨目录唯一（同一段同 seed 在 n004/n008 下是两档 noise 的两份数据，不能撞名）。
        meta_stem = str(meta.get("stem") or fstem)
        stem = f"{tag}__{fstem}" + ("" if meta_stem in fstem else f"__{meta_stem}")
        n_full = len(arr["tokens"])
        fs = fall_raw if fall_raw is not None else (
            meta.get("fall_step") if isinstance(meta.get("fall_step"), (int, float))
            else None)
        n_use, finfo = fall_clip(n_full, fs, win=win, stride=stride)
        rec = {"stem": stem, "record_file": os.path.basename(path), "record_dir": tag,
               "npz_path": os.path.abspath(path), "npz_md5": W.md5_file(path),
               "meta_stem": meta_stem, "seed": meta.get("seed"),
               "terrain": terrain["type"], "terrain_params": terrain["params"],
               "terrain_meta": {"terrain": meta.get("terrain"),
                                "terrain_noise": meta.get("terrain_noise"), **reg},
               "v_med": meta.get("v_med"), "token_source": meta.get("token_source"),
               "steps": meta.get("steps"), "steps_budget": meta.get("steps_budget"),
               "survived": meta.get("survived"), "completed": meta.get("completed"),
               "dur_s": meta.get("dur_s"),
               "npz_keys_present": arr["_files"],
               "has_pre_step_arrays": all(k in arr["_files"] for k in
                                          ("root_pos_pre", "root_quat_pre")),
               # 划分判定用**段身份**（文件 stem / meta.stem，含演员与 `_M$`）——
               # 台账 stem 另带目录 tag/seed 后缀（唯一性），故两者分开记
               "split": seg_split_tag(meta_stem, split),
               "actor": W.parse_actor(meta_stem),
               "mirror": W.is_mirror_stem(meta_stem),
               "state_source": STATE_SOURCE_QDES}
        rec.update(finfo)
        rec["frames"] = n_full
        rec["frames_used"] = n_use
        meta_extra = {
            "state_source": STATE_SOURCE_QDES,
            "state_align": "post_step",
            "state_source_note": ("state 槽 = q_des（当步下发的关节位置目标，**cmd 非实测"
                                  "关节角**）+ 根位姿（step 后）；与源 1/2/3 的实测运动学"
                                  "（state_source=kinematics）**不同口径**，训练侧须知；"
                                  "记录器另有 root_*_pre（step 前）数组，如需 PRE 对齐"
                                  "不必重跑——改 d060_intake_src14.py 一处重切"),
            "src": SRC4_ID, "record_dir": tag, "record_file": rec["record_file"],
            "npz_path": rec["npz_path"], "npz_md5": rec["npz_md5"],
            "meta_stem": meta_stem, "seed": meta.get("seed"),
            "mirror_stem": rec["mirror"], "actor": rec["actor"],
            "terrain_meta": rec["terrain_meta"], "v_med": rec.get("v_med"),
            "token_source": rec.get("token_source"), "steps": rec.get("steps"),
            "steps_budget": rec.get("steps_budget"), "survived": rec.get("survived"),
            "completed": rec.get("completed"), "dur_s": rec.get("dur_s"),
            "fall_step": finfo["fall_step"], "fall_clip_applied": finfo["fall_clip_applied"],
            "frames_after_fall": finfo["frames_after_fall"],
            "record_frames": finfo["record_frames"],
            "record_includes_fall_step_or_later":
                finfo["record_includes_fall_step_or_later"],
            "speedA_source": ("measured：窗内实测根速中位（XY 后向差分×fps，整段差分、"
                              "窗帧取中位）= command.target_vel（无命令真值源口径）")}
        ws = W.windows_from_arrays(arr["tokens"][:n_use], arr["jp"][:n_use],
                                   arr["quat"][:n_use], arr["trans"][:n_use],
                                   terrain_desc=terrain, command=None,
                                   intent_tokens=arr["tokens"][:n_use],
                                   win=win, stride=stride, fps=fps, stem=stem,
                                   intent_source=INTENT_ORACLE, meta_extra=meta_extra)
        starts = [w["meta"]["frame_start"] for w in ws]
        rec["windows"] = len(ws)
        rec["frames_covered_unique"] = W.covered_unique(starts, win, n_use)
        rec["frames_uncovered"] = int(n_use) - rec["frames_covered_unique"]
        rec["window_frames_visits"] = len(ws) * int(win)
        rec["overlap_redundancy"] = rec["window_frames_visits"] - rec["frames_covered_unique"]
        if ws:
            sp = np.asarray([w["meta"]["speedA"] for w in ws])
            rec["speedA_windows"] = {"min": float(sp.min()),
                                     "median": float(np.median(sp)),
                                     "max": float(sp.max())}
        return ws, rec, None
    except Exception as exc:  # noqa: BLE001
        return [], {"stem": f"{tag}__{fstem}", "record_dir": tag,
                    "npz_path": os.path.abspath(path),
                    "error": f"{type(exc).__name__}: {exc}"}, \
            f"{type(exc).__name__}: {exc}"


def run_src4_mode(args):
    split = resolve_split(args.split_json)
    fps, win, stride = float(args.fps), int(args.win), int(args.stride)
    expect = parse_expect(args.expect) if args.expect else (
        SRC4_EXPECT if not args.skip_wait else {})
    trace_path = os.path.join(args.out_dir, "src4_wait_trace.json")
    wait = {"status": "SKIPPED", "counts": {}, "polls": 0, "note": "未等待（--skip-wait）"}
    if not args.skip_wait:
        wait = wait_for_records(args.records_dir, expect,
                                wait_minutes=args.wait_minutes,
                                poll_seconds=args.poll_seconds,
                                tags=args.dirs, trace_path=trace_path)
        print(f"[src4] 等待状态 {wait['status']} 轮询 {wait['polls']} 次 counts={wait['counts']}")
    if args.wait_trace and os.path.isfile(args.wait_trace):
        # 复用既有等待 trace（--skip-wait 重跑时把**到达时间线**挂回 manifest，不丢证据）
        tr = load_json(args.wait_trace)
        polls = tr.get("polls") or []
        last = polls[-1] if polls else {}
        trace_path = os.path.abspath(args.wait_trace)
        wait = {"status": ("READY" if last.get("ready") else "UNKNOWN"),
                "counts": last.get("counts"), "polls": len(polls),
                "note": ("复用等待 trace（--wait-trace），本 run 未重新等待；"
                         "status 取末轮 ready 判定"),
                "trace_path": trace_path, "polls_detail": polls,
                "expect": tr.get("expect"), "records_dir": tr.get("records_dir")}
    found = discover_record_tags(args.records_dir, tags=args.dirs)
    if args.limit_dirs:
        found = {k: v for k, v in list(found.items())[:int(args.limit_dirs)]}
    if args.limit_per_dir:
        found = {k: v[:int(args.limit_per_dir)] for k, v in found.items()}
    files = [(tag, p) for tag, ps in found.items() for p in ps]
    if not files:
        raise SystemExit(f"{args.records_dir} 下无源 4 记录（目录 {list(found)}）"
                         "——等待状态 "
                         f"{wait['status']}，不硬凑")
    print(f"[src4] run npz {len(files)}（{ {t: len(v) for t, v in found.items()} }）"
          f"win={win} stride={stride}")
    t0 = time.time()
    all_w, tags, srecs, errs = [], [], [], []
    for tag, p in files:
        ws, rec, err = cut_src4_record(p, tag, split=split, win=win, stride=stride,
                                       fps=fps)
        srecs.append(rec)
        if err:
            errs.append({"npz_path": p, "error": err})
            continue
        all_w.extend(ws)
        tags.extend([rec["split"]] * len(ws))
    terrain_stats = {}
    for r in srecs:
        if "error" in r:
            continue
        key = f"{r['terrain']}{r['terrain_params']}"
        terrain_stats[key] = terrain_stats.get(key, 0) + 1
    shards, wrecs = W.write_shards(all_w, os.path.join(args.out_dir, "npz"),
                                   shard_size=args.shard_size, prefix="src4",
                                   split_tags=tags,
                                   extra_manifest={"source_id": SRC4_ID,
                                                   "stride": stride, "fps": fps})
    doc = build_source_manifest(
        source_id=SRC4_ID, source_label="src4（闭环回放记录）",
        terrain_default=None, win=win, stride=stride, fps=fps,
        segments=srecs, shards=shards, wrecs=wrecs, windows_raw=all_w,
        counts_extra={"runs_found": len(files), "runs_error": len(errs),
                      "runs_fall": int(sum(1 for r in srecs
                                           if r.get("survived") is False)),
                      "runs_survived": int(sum(1 for r in srecs
                                              if r.get("survived") is True)),
                      "windows_heldout": int(sum(1 for t in tags if t == "heldout")),
                      "frames_after_fall": int(sum(int(r.get("frames_after_fall") or 0)
                                                   for r in srecs)),
                      "windows_dropped_by_fall_clip": int(sum(
                          int(r.get("windows_dropped_by_fall_clip") or 0)
                          for r in srecs))},
        format_extra={"state_source": STATE_SOURCE_QDES,
                      "state_align": "post_step",
                      "state_semantics_note": (
                          "**口径差异（显式登记）**：源 4 的 36 维 state = cmd 关节目标 "
                          "q_des(29) + 根位姿(step 后 quat4+trans3)，**不是实测关节角**；"
                          "源 1/2/3 = kinematics（实测）。两口径不得混读"),
                      "intent_source": INTENT_ORACLE,
                      "command_source": "per-window speedA（无命令真值源口径）",
                      "terrain_rule": ("type=meta.terrain（缺→rough_paper 并计数）；"
                                       "rough_paper 传 meta.terrain_noise；"
                                       "plane 不传 noise"),
                      "fall_rule": ("窗只取 fall_step 前的完整帧；不足 win 的尾部丢弃"
                                    "并计数（frames_uncovered/frames_tail_incomplete）；"
                                    "窗 meta 记 fall_step / frames_after_fall / "
                                    "fall_clip_applied / survived")},
        inputs={"records_dir": os.path.abspath(args.records_dir),
                "dirs": list(found), "npz_keys": SRC4_KEYS, "split_json": split.get("path"),
                "split_rule": ("held-out 演员段与 `_M` 镜像段的窗标 heldout（不进训练窗，"
                               "§5t）；源 4 run 的 stem 前置记录目录 tag 以保证唯一"),
                "expect": expect,
                "wait": {k: v for k, v in wait.items()
                         if k in ("status", "counts", "polls", "note", "expect")},
                "wait_trace_path": trace_path},
        extra={"terrain_runs": terrain_stats,
               "wait": {"status": wait["status"], "counts": wait.get("counts"),
                        "polls": wait.get("polls"), "note": wait.get("note"),
                        "trace_path": trace_path,
                        "polls_detail": wait.get("polls_detail")}},
        elapsed=time.time() - t0)
    dump_manifest(doc, args.manifest_out or os.path.join(args.out_dir, "manifest.json"))
    c = doc["counts"]
    print(f"[src4] 窗 {c['windows']}（train {c['windows_train']}/heldout "
          f"{c['windows_heldout']}）帧 {c['window_frames']} 分片 {c['shards']} "
          f"错误 {len(errs)} 耗时 {c['elapsed_s']}s | 摔倒 run {c['runs_fall']}"
          f"（fall 截断丢窗 {c['windows_dropped_by_fall_clip']}、摔后帧 "
          f"{c['frames_after_fall']}）")
    if doc["speedA_dist"]:
        s = doc["speedA_dist"]
        print(f"[src4] speedA min {s['min']:.3f} / med {s['median']:.3f} / max "
              f"{s['max']:.3f}")
    print(f"[src4] -> {doc['manifest_path']}")
    if len(all_w) < G1_MIN_WINDOWS:
        print(f"[src4] [!] 窗数 {len(all_w)} < G1 判据 {G1_MIN_WINDOWS}")
    return doc


def parse_expect(spec):
    """'records_n004=12,records_n008=12' → {tag: n}。"""
    out = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--expect 需 tag=n 形式，得 {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = int(v)
    return out


# ------------------------------------------------------------------ manifest 组装（源 1/4 共用）
def build_source_manifest(*, source_id, source_label, terrain_default, win, stride, fps,
                          segments, shards, wrecs, counts_extra, format_extra, inputs,
                          extra=None, elapsed=None, windows_raw=None):
    """源 manifest（脊柱 window 模式同构 + 本源口径登记）；逐段对账内建，出结论 ok。"""
    ok_segs = [r for r in segments if "error" not in r]
    err_segs = [r for r in segments if "error" in r]
    windows = len(wrecs)
    visits = int(sum(int(r.get("window_frames_visits") or 0) for r in ok_segs))
    covered = int(sum(int(r.get("frames_covered_unique") or 0) for r in ok_segs))
    frames_used = int(sum(int(r.get("frames_used") or 0) for r in ok_segs))
    uncovered = int(sum(int(r.get("frames_uncovered") or 0) for r in ok_segs))
    overlap = int(sum(int(r.get("overlap_redundancy") or 0) for r in ok_segs))
    seg_checks = [segment_reconciliation(r, win=win)["check"] for r in ok_segs]
    counts = {"segments_found": len(segments), "segments_used": len(ok_segs),
              "segments_error": len(err_segs),
              "windows": windows, "window_frames": windows * int(win),
              "windows_train": int(sum(1 for w in wrecs if w["split"] == "train")),
              "windows_heldout": int(sum(1 for w in wrecs if w["split"] == "heldout")),
              "shards": len(shards), "shard_bytes": int(sum(s["bytes"] for s in shards)),
              "frames_windowed": frames_used,
              "frames_covered_unique": covered, "frames_uncovered": uncovered,
              "window_frame_visits": visits, "overlap_redundancy": overlap,
              "elapsed_s": round(float(elapsed), 2) if elapsed is not None else None,
              **counts_extra}
    if windows_raw:
        # 消费者侧（d060_consumer_stub）读 counts.token_min/token_max 定 vocab
        counts["token_min"] = int(min(int(w["token_stream"].min()) for w in windows_raw))
        counts["token_max"] = int(max(int(w["token_stream"].max()) for w in windows_raw))
    doc = {
        "exp": "D060", "gate": "G1 格式冒烟（源 1/源 4 切窗）",
        "created_at": now_str(), "source_id": source_id, "source_label": source_label,
        "script": SCRIPT_PATH, "script_md5": W.md5_file(SCRIPT_PATH),
        "spine": {"module": "d060_windows", "path": os.path.abspath(
            getattr(W, "__file__", "d060_windows.py")),
            "md5": W.md5_file(os.path.abspath(getattr(W, "__file__",
                                                      "d060_windows.py"))),
            "format_version": getattr(W, "FORMAT_VERSION", "d060.1")},
        "format": {"format_version": getattr(W, "FORMAT_VERSION", "d060.1"),
                   "fps": float(fps), "win": int(win), "stride": int(stride),
                   "token_dim": int(getattr(W, "TOKEN_DIM", 64)),
                   "state_dim": int(getattr(W, "STATE_DIM", 36)),
                   "state_layout": getattr(W, "STATE_LAYOUT", ""),
                   "token_dtype": "int16", "state_dtype": "float32",
                   "token_scale": (wrecs[0]["token_scale"] if wrecs else None),
                   "token_decode_rule": "token_stream 存 int16 码；反量化值 = code / token_scale",
                   "sentinel": float(W.SENTINEL),
                   "command_fields": list(W.COMMAND_FIELDS),
                   "terrain_default": terrain_default,
                   **format_extra},
        "inputs": inputs,
        "counts": counts,
        "speedA_dist": quantiles([w.get("speedA") for w in wrecs]),
        "command_dist": distribution(
            [f"tv={w.get('target_vel')}|mode={w.get('mode')}|dir="
             f"{w.get('movement_direction')}|has_truth={','.join(w.get('has_truth') or [])}"
             for w in wrecs]),
        "terrain_dist": distribution([w.get("terrain") for w in wrecs]),
        "terrain_param_dist": distribution(
            [f"{w.get('terrain')}{json.dumps(w.get('terrain_params') or {}, sort_keys=True)}"
             for w in wrecs]),
        "intent_source_dist": distribution([w.get("intent_source") for w in wrecs]),
        "split_dist": distribution([w.get("split") for w in wrecs]),
        "segments": segments,
        "segments_error": err_segs,
        "shards": shards,
        "windows": wrecs,
        "reconciliation": {
            "rule": ("段级：windows×win = 覆盖并集 + 重叠冗余；frames_used = 覆盖并集 + "
                     "未覆盖（含摔后帧与不足 win 的尾帧）；全局：manifest 窗数 = 分片窗数"
                     "之和 = counts.windows"),
            "windows_from_shards": int(sum(s["n"] for s in shards)),
            "windows_in_manifest": windows,
            "counts_windows": counts["windows"],
            "frames_used": frames_used, "frames_covered_unique": covered,
            "frames_uncovered": uncovered, "overlap_redundancy": overlap,
            "segments_all_ok": bool(all(seg_checks)) if seg_checks else True,
            "errors": len(err_segs),
            "ok": bool(sum(s["n"] for s in shards) == windows == counts["windows"]
                       and (all(seg_checks) if seg_checks else True)
                       and visits == covered + overlap
                       and frames_used == covered + uncovered),
        },
    }
    if extra:
        doc.update(extra)
    doc["manifest_path"] = None  # 落盘后回填（见 dump_manifest）
    return doc


def dump_manifest(doc, path):
    doc["manifest_path"] = os.path.abspath(path)
    dump_json(doc, path)
    return doc["manifest_path"]


# ------------------------------------------------------------------ 四源装配
def source_summary(source_id, label, manifest_path, doc, declared_state_source=None):
    """单源统计块（窗数/帧数/来源统计/speedA 分位/terrain 分布/state_source/对账）。"""
    counts = doc.get("counts") or {}
    wrecs = doc.get("windows") or []
    shards = doc.get("shards") or []
    fmt = doc.get("format") or {}
    n_from_shards = int(sum(int(s.get("n") or 0) for s in shards))
    state_source = (fmt.get("state_source") or declared_state_source or None)
    evidence = ("declared：窗 meta/manifest.format.state_source 由产出脚本写入"
                if fmt.get("state_source") else
                ("inferred：manifest 无 state_source 字段 → 按产出路径推断"
                 "（d060_windows.build_segment_windows 用 state_from_kinematics 装配"
                 "实测 jp/quat/trans）"))
    # 分布块**一律从逐窗记录重算**（跨源同口径），源 manifest 自报值另存全字段对账
    sp_re = quantiles([w.get("speedA") for w in wrecs])
    sp_decl = doc.get("speedA_dist")
    sp_diff = dist_fields_diff(sp_decl, sp_re)
    sp_match = (sp_decl is None) or (not sp_diff)
    cw = counts.get("windows")
    cw = -1 if cw is None else int(cw)     # 0 窗是合法值：不能被 `or` 吞成 -1（假 FAIL）
    return {
        "source_id": source_id, "label": label,
        "manifest_path": os.path.abspath(manifest_path),
        "manifest_created_at": doc.get("created_at"),
        "manifest_script": doc.get("script"),
        "gate": doc.get("gate"),
        "counts": {k: counts.get(k) for k in
                   ("segments_found", "segments_used", "segments_error", "windows",
                    "window_frames", "windows_train", "windows_heldout", "shards",
                    "shard_bytes", "frames_windowed", "frames_covered_unique",
                    "frames_uncovered", "overlap_redundancy", "token_min", "token_max",
                    "token_scale", "elapsed_s", "runs_found", "runs_error", "runs_fall",
                    "runs_survived", "scripts_found", "frames_after_fall",
                    "windows_dropped_by_fall_clip", "frames_too_short", "frames_error")
                   if k in counts},
        "format": {k: fmt.get(k) for k in
                   ("format_version", "fps", "win", "stride", "token_dim", "state_dim",
                    "state_layout", "token_scale", "state_source", "state_align",
                    "intent_source", "command_encoding")
                   if k in fmt},
        "state_source": state_source,
        "state_source_evidence": evidence,
        "state_semantics_note": fmt.get("state_semantics_note"),
        "speedA_dist": sp_re,
        "speedA_dist_declared": sp_decl,
        "speedA_dist_recomputed_matches_declared": sp_match,
        "speedA_dist_recomputed_vs_declared_diff": sp_diff,
        "terrain_dist": distribution([w.get("terrain") for w in wrecs]),
        "terrain_param_dist": doc.get("terrain_param_dist")
        or distribution([f"{w.get('terrain')}{json.dumps(w.get('terrain_params') or {}, sort_keys=True)}"
                         for w in wrecs]),
        "intent_source_dist": doc.get("intent_source_dist")
        or distribution([w.get("intent_source") for w in wrecs]),
        "split_dist": doc.get("split_dist")
        or distribution([w.get("split") for w in wrecs]),
        "command_dist": doc.get("command_dist"),
        "reconciliation": {
            "windows_from_shards": n_from_shards,
            "windows_in_manifest": len(wrecs),
            "counts_windows": counts.get("windows"),
            "counts_windows_used": cw,
            "source_manifest_ok": (doc.get("reconciliation") or {}).get("ok"),
            "ok": bool(n_from_shards == len(wrecs) == cw),
        },
        "g1_min_windows": G1_MIN_WINDOWS,
        "g1_windows_ok": bool(int(counts.get("windows") or 0) >= G1_MIN_WINDOWS),
    }


def build_assembly(sources, *, notes=None, data_root=None):
    """四源装配台账（只引用不合并）：sources = [(source_id, label, manifest_path,
    declared_state_source)]。"""
    blocks, bad = [], []
    for sid, label, mpath, declared in sources:
        if not os.path.isfile(mpath):
            bad.append({"source_id": sid, "manifest_path": os.path.abspath(mpath),
                        "problem": "manifest 缺失"})
            continue
        doc = load_json(mpath)
        blk = source_summary(sid, label, mpath, doc, declared_state_source=declared)
        blocks.append(blk)
    total_windows = int(sum(int(b["reconciliation"]["windows_in_manifest"])
                            or 0 for b in blocks))
    total_frames = int(sum(int((b["counts"].get("window_frames") or 0)) for b in blocks))
    wins = {b["source_id"]: b["format"].get("win") for b in blocks}
    sds = {b["source_id"]: b["format"].get("state_dim") for b in blocks}
    fvs = {b["source_id"]: b["format"].get("format_version") for b in blocks}
    ts = {b["source_id"]: b["format"].get("token_scale") for b in blocks}
    state_sources = distribution([b.get("state_source") for b in blocks])
    speedA_all = []
    for b in blocks:
        if b["speedA_dist"]:
            speedA_all.append(b["speedA_dist"]["median"])
    doc = {
        "exp": "D060", "gate": "G1 格式冒烟（四源统一装配）",
        "created_at": now_str(), "script": SCRIPT_PATH,
        "script_md5": W.md5_file(SCRIPT_PATH),
        "purpose": ("四源统一窗语料的**装配台账**：只引用各源 manifest（分片留各源目录，"
                    "loader 按 manifest 天然跨源读），不做物理合并、不复制数据；"
                    "给出窗数/帧数/来源统计/speedA 分位/terrain 分布/state_source 分列"),
        "data_root": os.path.abspath(data_root) if data_root else None,
        "format_common": {
            "win_by_source": wins, "state_dim_by_source": sds,
            "format_version_by_source": fvs, "token_scale_by_source": ts,
            "all_win_equal": len(set(wins.values())) == 1,
            "all_state_dim_equal": len(set(sds.values())) == 1,
            "all_format_version_equal": len(set(fvs.values())) == 1,
            "win": wins.get(list(wins)[0]) if wins else None,
            "state_dim": sds.get(list(sds)[0]) if sds else None,
            "state_dim_note": ("owner 2026-09-17 裁决 canonical = 36 维 = jp29 + quat4 + "
                               "trans3，无保留位；四源 state_dim 必须一致（本装配逐源核对）"),
            "token_decode_rule": "反量化值 = code / token_scale（逐源 scale 见上）",
        },
        "state_source_dist": state_sources,
        "state_source_rule": {
            "kinematics": "实测运动学（jp 实测 + 根位姿）——源 1（planner 剧本 npz 实测）、"
                          "源 2/源 3（ds_bones 语料实测）",
            "q_des_cmd_root": "cmd 关节目标 q_des + 根位姿（step 后）——源 4 专有，"
                              "**非实测关节角**，训练侧不得与 kinematics 混读",
        },
        "totals": {"sources": len(blocks), "sources_missing": len(bad),
                   "windows": total_windows, "window_frames": total_frames,
                   "windows_train": int(sum(int(b["counts"].get("windows_train") or 0)
                                            for b in blocks)),
                   "windows_heldout": int(sum(int(b["counts"].get("windows_heldout") or 0)
                                              for b in blocks)),
                   "shards": int(sum(int(b["counts"].get("shards") or 0) for b in blocks)),
                   "shard_bytes": int(sum(int(b["counts"].get("shard_bytes") or 0)
                                          for b in blocks)),
                   "segments_used": int(sum(int(b["counts"].get("segments_used") or 0)
                                            for b in blocks)),
                   "segments_error": int(sum(int(b["counts"].get("segments_error") or 0)
                                             for b in blocks)),
                   "source_median_speedA_min": (min(speedA_all) if speedA_all else None),
                   "source_median_speedA_max": (max(speedA_all) if speedA_all else None),
                   "g5_frame_gate": G5_FRAME_GATE,
                   "g5_gate_note": ("§5t 产量门①（≥150 万帧）是 **G5 全量** 判据；本装配是 "
                                    "G1 冒烟子集，只报数不做 G5 判读，不把子集当产量门过")},
        "g1_judgement_material": {
            "rule": f"G1 = 四源各 ≥{G1_MIN_WINDOWS} 窗 ∧ 统一 loader 读通 ∧ 消费者测试 PASS",
            "windows_per_source": {b["source_id"]: b["reconciliation"]["windows_in_manifest"]
                                   for b in blocks},
            "per_source_ok": {b["source_id"]: b["g1_windows_ok"] for b in blocks},
            "all_sources_ge_min": bool(blocks and all(b["g1_windows_ok"] for b in blocks)),
            "reconciliation_ok": bool(blocks and all(b["reconciliation"]["ok"] for b in blocks)
                                      and not bad),
        },
        "sources": {b["source_id"]: b for b in blocks},
        "sources_missing": bad,
        "terrain_dist_total": distribution(
            [f"{b['source_id']}:{k}" for b in blocks
             for k, v in (b.get("terrain_dist") or {}).items() for _ in range(int(v))]),
        "intent_source_dist_total": distribution(
            [f"{b['source_id']}:{k}" for b in blocks
             for k, v in (b.get("intent_source_dist") or {}).items()
             for _ in range(int(v))]),
        "consumer": None, "notes": notes,
        "reconciliation": {
            "rule": ("装配计数 = 各源 manifest 逐窗记录数 = 各源分片窗数之和；三处不一致"
                     "即 ok=False（不静默）"),
            "per_source": {b["source_id"]: b["reconciliation"] for b in blocks},
            "ok": bool(blocks and not bad
                       and all(b["reconciliation"]["ok"] for b in blocks)),
        },
    }
    return doc


def run_assembly_mode(args):
    srcs = [(SRC1_ID, "src1（planner 命令剧本）", args.src1, None),
            (SRC2_ID, "src2（ds_bones 爬障全量窗）", args.src2, STATE_SOURCE_KIN),
            (SRC3_ID, "src3（官方平地存量窗）", args.src3, STATE_SOURCE_KIN),
            (SRC4_ID, "src4（闭环回放记录）", args.src4, STATE_SOURCE_QDES)]
    doc = build_assembly(srcs, notes=args.notes, data_root=args.data_root)
    out = os.path.join(args.out_dir, "assembly_manifest.json")
    doc["consumer"] = load_json(args.consumer_summary) if (
        args.consumer_summary and os.path.isfile(args.consumer_summary)) else None
    dump_json(doc, out)
    t = doc["totals"]
    print(f"[assembly] 源 {t['sources']}（缺 {t['sources_missing']}）窗 {t['windows']}"
          f"（train {t['windows_train']}/heldout {t['windows_heldout']}）"
          f"帧 {t['window_frames']} 分片 {t['shards']}")
    for sid, b in doc["sources"].items():
        c = b["counts"]
        print(f"[assembly]   {sid}: 窗 {b['reconciliation']['windows_in_manifest']} "
              f"（对账 {'ok' if b['reconciliation']['ok'] else 'FAIL'}，G1≥"
              f"{G1_MIN_WINDOWS} {'ok' if b['g1_windows_ok'] else 'FAIL'}）"
              f"帧 {c.get('window_frames')} terrain {list((b['terrain_dist'] or {}).keys())} "
              f"state_source={b['state_source']}")
    print(f"[assembly] g1 all_sources_ge_min={doc['g1_judgement_material']['all_sources_ge_min']}"
          f" reconciliation_ok={doc['reconciliation']['ok']}")
    print(f"[assembly] -> {out}")
    return doc


# ------------------------------------------------------------------ 消费者测试
def run_consumer_mode(args):
    stub = args.stub or os.path.join(os.path.dirname(SCRIPT_PATH),
                                     "d060_consumer_stub.py")
    if not os.path.isfile(stub):
        raise SystemExit(f"消费者脚本缺失：{stub}")
    py = args.python or sys.executable
    srcs = [(SRC1_ID, args.src1), (SRC2_ID, args.src2), (SRC3_ID, args.src3),
            (SRC4_ID, args.src4)]
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for sid, mpath in srcs:
        rec = {"source_id": sid, "manifest": os.path.abspath(mpath)}
        if not os.path.isfile(mpath):
            rec.update({"verdict": "FAIL", "error": "manifest 缺失"})
            rows.append(rec)
            continue
        out_json = os.path.join(args.out_dir, f"consumer_{sid}.json")
        log = os.path.join(args.out_dir, f"consumer_{sid}.log")
        cmd = [py, stub, "--manifest", mpath, "--limit", str(args.limit),
               "--steps", str(args.steps), "--seed", str(args.seed),
               "--split", args.split, "--lr", str(args.lr), "--out-json", out_json]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        with open(log, "w", encoding="utf-8") as f:
            f.write("cmd: " + " ".join(cmd) + "\n\n")
            f.write(p.stdout or "")
            f.write("\n--- stderr ---\n")
            f.write(p.stderr or "")
        rec.update({"cmd": cmd, "rc": p.returncode, "elapsed_s": round(time.time() - t0, 2),
                    "out_json": out_json, "log": log})
        if os.path.isfile(out_json):
            o = load_json(out_json)
            rec.update({k: o.get(k) for k in
                        ("verdict", "n_windows", "contract_ok", "contract_violations",
                         "loss_first", "loss_last", "loss_finite", "params_changed",
                         "grads_finite", "max_param_delta_after_step", "n_params",
                         "code_min", "vocab", "batch_shapes", "token_src_range",
                         "split", "limit", "steps") if k in o})
            rec["losses"] = o.get("losses")
        else:
            rec.update({"verdict": "FAIL", "error": f"消费者未产出 JSON（rc={p.returncode}）",
                        "stderr_tail": (p.stderr or "")[-800:]})
        rows.append(rec)
        print(f"[consumer] {sid}: {rec.get('verdict')} rc={rec.get('rc')} "
              f"窗 {rec.get('n_windows')} loss {rec.get('loss_first')} → "
              f"{rec.get('loss_last')} 契约 {'ok' if rec.get('contract_ok') else 'FAIL'}")
    summary = {"exp": "D060", "gate": "G1 消费者测试（四源）", "created_at": now_str(),
               "script": SCRIPT_PATH, "stub": os.path.abspath(stub),
               "stub_md5": W.md5_file(stub), "python": py,
               "args": {"limit": args.limit, "steps": args.steps, "split": args.split,
                        "seed": args.seed, "lr": args.lr},
               "table": [{"source_id": r["source_id"],
                          "verdict": r.get("verdict"),
                          "n_windows": r.get("n_windows"),
                          "contract_ok": r.get("contract_ok"),
                          "loss_first": r.get("loss_first"),
                          "loss_last": r.get("loss_last"),
                          "grad_norm": ((r.get("losses") or [{}])[-1].get("grad_norm")
                                        if r.get("losses") else None),
                          "next_token_top1_acc": ((r.get("losses") or [{}])[-1].get(
                              "next_token_top1_acc") if r.get("losses") else None),
                          "rc": r.get("rc"), "elapsed_s": r.get("elapsed_s")}
                         for r in rows],
               "rows": rows,
               "all_pass": all(r.get("verdict") == "PASS" for r in rows) and len(rows) == 4,
               "note": ("消费者 = d060_consumer_stub.py（stub causal transformer forward + "
                        "1 步 SGD + 字段契约断言）；PASS 判据 = 契约 ok ∧ loss 有限 ∧ "
                        "参数有更新 ∧ grad>0")}
    out = os.path.join(args.out_dir, "consumer_summary.json")
    dump_json(summary, out)
    print(f"[consumer] 四源 {'全 PASS' if summary['all_pass'] else '存在 FAIL'}"
          f" -> {out}")
    return 0 if summary["all_pass"] else 1


# ------------------------------------------------------------------ selftest
def run_selftest():
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="d060_intake_selftest_")
    n_ok = n_bad = 0

    def chk(name, cond, extra=""):
        nonlocal n_ok, n_bad
        if cond:
            n_ok += 1
            print(f"  [ok]   {name} {extra}")
        else:
            n_bad += 1
            print(f"  [FAIL] {name} {extra}")

    def synth(n=520, seed=0, speed=0.3):
        rng = np.random.default_rng(seed)
        step = speed / W.FPS
        trans = np.stack([np.arange(n) * step, rng.normal(0, 1e-4, n),
                          np.full(n, 0.75)], axis=1).astype(np.float32)
        jp = rng.normal(0, 0.2, (n, 29)).astype(np.float32)
        q = np.tile(np.array([1.0, 0, 0, 0], np.float32), (n, 1))
        q += rng.normal(0, 1e-3, (n, 4)).astype(np.float32)
        q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
        tok = (rng.integers(-11, 12, (n, 64)) / 16.0).astype(np.float32)
        return tok, jp, q, trans

    def write_src1(dirpath, point, name, n, *, tv, mode_id, mode_name, fall_step=None,
                  seed=0):
        os.makedirs(os.path.join(dirpath, point), exist_ok=True)
        tok, jp, q, tr = synth(n, seed=seed)
        meta = {"schema": "d060_src1_v1", "point_tag": point, "script_idx": 0,
                "seed": seed, "seed_base": 0, "n_steps_requested": n,
                "n_steps_planned": n, "env_episode_length": n,
                "nominal": {"ctrl_hz": 50.0, "replan_hz": 10.0, "target_vel": tv,
                            "target_vel_sent": tv, "mode_id": mode_id,
                            "mode_name": mode_name, "height": -1.0,
                            "direction_name": "front", "movement_direction": [1.0, 0.0, 0.0],
                            "facing_direction": [1.0, 0.0, 0.0]},
                "realized": {"net_speed_xy": tv * 0.55, "ratio_to_nominal": 0.55,
                             "disp_xy": 6.6, "path_len_xy": 7.3, "straightness": 0.9,
                             "h_min": 0.67, "h_end": 0.75, "n_steps": n,
                             "fall": fall_step is not None, "fall_step": fall_step,
                             "terminated_cause": ("fall" if fall_step is not None
                                                  else "horizon_cap")}}
        np.savez_compressed(os.path.join(dirpath, point, name + ".npz"), tokens=tok,
                            root_pos=tr, root_quat=q, jp_mujoco=jp, jp_isaaclab=jp,
                            meta=np.array(json.dumps(meta)))
        return os.path.join(dirpath, point, name + ".npz")

    def write_src4(dirpath, tag, name, n, *, terrain="rough_paper", noise=0.04,
                   fall_step=-1, survived=True, seed=1, stem=None):
        d = os.path.join(dirpath, tag)
        os.makedirs(d, exist_ok=True)
        tok, jp, q, tr = synth(n, seed=seed, speed=0.5)
        meta = {"stem": stem or name, "seed": seed, "terrain": terrain,
                "terrain_noise": noise, "v_med": 0.42, "token_source": "orig",
                "steps": n + (0 if fall_step < 0 else 1), "steps_budget": n,
                "dur_s": 20.0, "completed": fall_step < 0, "survived": survived}
        np.savez_compressed(os.path.join(d, name + ".npz"), q_des=jp, root_pos=tr,
                            root_quat=q, tokens_exec=tok, root_pos_pre=tr,
                            root_quat_pre=q, fall_step=np.int64(fall_step),
                            meta=np.array(json.dumps(meta)))
        return os.path.join(d, name + ".npz")

    def expect_windows(n_use, win=W.WIN, stride=W.STRIDE):
        return len(W.slice_starts(n_use, win=win, stride=stride))

    try:
        root = os.path.join(tmp, "data")
        src1_dir = os.path.join(root, "src1")
        # 源 1 fixture：520 帧（4 窗）/ 295 帧摔倒 fall_step=294（1 窗）/ 405 帧（3 窗）
        write_src1(src1_dir, "p00_vel0.6_SLOW_WALK_front", "script_000", 520, tv=0.6,
                   mode_id=1, mode_name="SLOW_WALK", seed=0)
        write_src1(src1_dir, "p00_vel0.6_SLOW_WALK_front", "script_001", 295, tv=0.6,
                   mode_id=1, mode_name="SLOW_WALK", fall_step=294, seed=1)
        write_src1(src1_dir, "p01_vel1_RUN_front", "script_000", 405, tv=1.0,
                   mode_id=3, mode_name="RUN", seed=2)

        print("[selftest 1] 源 1 切窗（nominal command + 实测 speedA + 摔倒截断）")
        out1 = os.path.join(root, "windows_src1")
        doc1 = run_src1_mode(argparse.Namespace(
            npz_dirs=[src1_dir], out_dir=out1, manifest_out=None, split_json=None,
            win=W.WIN, stride=W.STRIDE, fps=W.FPS, shard_size=4, limit=None))
        want1 = expect_windows(520) + expect_windows(294) + expect_windows(405)
        chk(f"窗数 = 4+1+3 = {want1}", doc1["counts"]["windows"] == want1,
            f"{doc1['counts']['windows']}")
        chk("每脚本一条段记录且无 error", doc1["counts"]["segments_found"] == 3
            and doc1["counts"]["segments_error"] == 0)
        chk("摔倒脚本截断入账（fall_step=294，摔后帧 1）",
            any(r.get("fall_step") == 294 and r.get("frames_after_fall") == 1
                for r in doc1["segments"]))
        chk("manifest 对账 ok（分片 = 逐窗记录 = counts）",
            doc1["reconciliation"]["ok"]
            and doc1["reconciliation"]["windows_from_shards"] == want1)
        wr1 = {w["stem"]: w for w in doc1["windows"]}
        chk("command 用 nominal（target_vel/has_truth）",
            all(w["has_truth"] == ["target_vel", "movement_direction", "mode"]
                for w in doc1["windows"])
            and all(w["target_vel"] in (0.6, 1.0) for w in doc1["windows"]))
        chk("height 无真值 → 哨兵 -1 且不在 has_truth",
            all(w["height"] == -1.0 for w in doc1["windows"]))
        chk("movement_direction 编码 = 体坐标偏航角（front=0.0）",
            all(abs(w["movement_direction"]) < 1e-9 for w in doc1["windows"]))
        chk("实测 speedA 逐窗入账且与命令分离",
            all(w["speedA"] is not None and 0.2 < w["speedA"] < 0.8
                for w in doc1["windows"]))
        man1 = load_json(out1 + "/manifest.json")
        chk("名义/实测双记（段级 realized.net_speed_xy + ratio_to_nominal）",
            all(r["realized"]["net_speed_xy"] is not None
                and r["realized"]["ratio_to_nominal"] is not None
                for r in man1["segments"]))
        chk("counts.token_min/token_max 就位（消费者定 vocab 用）",
            man1["counts"]["token_min"] == -11 and man1["counts"]["token_max"] == 11,
            f"{man1['counts'].get('token_min')}..{man1['counts'].get('token_max')}")
        chk("manifest_path 回填（落盘后自指）",
            man1["manifest_path"] == os.path.abspath(
                os.path.join(out1, "manifest.json")))
        seg_noms = {(r["nominal"]["nominal_target_vel"], r["nominal"]["nominal_mode_id"])
                    for r in doc1["segments"]}
        chk("段级 nominal 双命令点（0.6/SLOW_WALK, 1.0/RUN）",
            seg_noms == {(0.6, 1), (1.0, 3)}, f"{sorted(seg_noms)}")
        chk("段 stem = 点目录__剧本名（可读、不带盘上路径前缀）",
            sorted(r["stem"] for r in doc1["segments"])
            == ["p00_vel0.6_SLOW_WALK_front__script_000",
                "p00_vel0.6_SLOW_WALK_front__script_001",
                "p01_vel1_RUN_front__script_000"],
            f"{sorted(r['stem'] for r in doc1['segments'])}")
        # loader 回读 + 契约
        wins1 = list(W.load_windows(os.path.join(out1, "manifest.json"), limit=3))
        chk("统一 loader 回读源 1 窗（3 窗）", len(wins1) == 3)
        chk("回读窗契约合规 + state_source=kinematics + intent_source=planner_script",
            all(not W.window_contract_violations(w) for w in wins1)
            and all(w["meta"]["state_source"] == STATE_SOURCE_KIN for w in wins1)
            and all(w["meta"]["intent_source"] == INTENT_PLANNER for w in wins1))
        chk("回读 speedA 与 manifest 一致",
            all(abs(w["speedA"] - w["meta"]["speedA"]) < 1e-12 for w in wins1))
        chk("token 码无损（scale=16 且 code/16 == 源值）",
            all(w["meta"]["token_scale"] == 16 for w in wins1))

        print("[selftest 2] 源 4 切窗（q_des 口径 + 摔倒截断 + 划分）")
        src4_dir = os.path.join(root, "s4_records")
        write_src4(src4_dir, "records_n004", "runA__seed1", 520, noise=0.04, seed=1,
                   stem="climb_seg_001__A111")
        write_src4(src4_dir, "records_n004", "runB__seed1", 350, noise=0.04, seed=2,
                   fall_step=340, survived=False, stem="climb_seg_002__A222")
        write_src4(src4_dir, "records_n008", "runC__seed1", 250, noise=0.08, seed=3,
                   stem="climb_seg_003__A999")           # held-out 演员
        write_src4(src4_dir, "records_n008", "runD_M__seed1", 250, noise=0.08, seed=4,
                   stem="climb_seg_004__A111_M")          # 镜像段
        split_path = os.path.join(root, "actor_split.json")
        dump_json({"seed": 0, "heldout": ["A999"], "train": ["A111", "A222"],
                   "actors": {}}, split_path)
        out4 = os.path.join(root, "windows_src4")
        doc4 = run_src4_mode(argparse.Namespace(
            records_dir=src4_dir, out_dir=out4, manifest_out=None, split_json=split_path,
            win=W.WIN, stride=W.STRIDE, fps=W.FPS, shard_size=4, skip_wait=True,
            wait_minutes=0, poll_seconds=1, dirs=None, expect=None, limit_dirs=None,
            limit_per_dir=None, wait_trace=None))
        want4 = expect_windows(520) + expect_windows(340) + expect_windows(250) * 2
        chk(f"窗数 = 4+2+1+1 = {want4}", doc4["counts"]["windows"] == want4,
            f"{doc4['counts']['windows']}")
        chk("摔倒 run 截断（fall_step=340 → frames_after_fall=10、丢窗 0）",
            any(r.get("fall_step") == 340 and r.get("frames_after_fall") == 10
                for r in doc4["segments"]))
        chk("摔倒 run 的 survived/completed 入账",
            any(r.get("survived") is False and r.get("completed") is False
                for r in doc4["segments"]))
        chk("state_source=q_des_cmd_root 写进窗 meta 与 manifest.format",
            doc4["format"]["state_source"] == STATE_SOURCE_QDES)
        wins4 = list(W.load_windows(os.path.join(out4, "manifest.json")))
        chk("全部源 4 窗 meta 记 state_source/state_align 与非静默注记",
            all(w["meta"]["state_source"] == STATE_SOURCE_QDES
                and w["meta"]["state_align"] == "post_step"
                and "cmd 非实测关节角" in w["meta"]["state_source_note"] for w in wins4))
        chk("intent_source=oracle_replay 且 intent == token_stream（回放自条件）",
            all(w["meta"]["intent_source"] == INTENT_ORACLE
                and np.array_equal(w["intent_tokens"], w["token_stream"]) for w in wins4))
        chk("command = 逐窗 speedA（无命令源口径）",
            all(abs(w["command"]["cmd"]["target_vel"] - w["speedA"]) < 1e-12
                and w["command"]["has_truth"]["target_vel"] for w in wins4))
        chk("terrain 来自 meta（noise 0.04/0.08 两档）",
            set(doc4["terrain_dist"]) <= {"rough_paper"}
            and {json.dumps(w["terrain_desc"]["params"], sort_keys=True)
                 for w in wins4} == {'{"noise": 0.04}', '{"noise": 0.08}'},
            f"{doc4['terrain_dist']}")
        chk("held-out 演员段标 heldout（A999: 1 窗）",
            doc4["counts"]["windows_heldout"] == 2
            and sorted(doc4["split_dist"]) == ["heldout", "train"],
            f"{doc4['split_dist']}")
        chk("镜像 `_M` 段同样不进训练窗（标 heldout）",
            any(r.get("mirror") and r.get("split") == "heldout"
                for r in doc4["segments"]))
        chk("run stem 带记录目录 tag（跨目录唯一）",
            all("__" in r["stem"] and r["stem"].startswith(r["record_dir"] + "__")
                for r in doc4["segments"]))
        chk("manifest 对账 ok", doc4["reconciliation"]["ok"])
        chk("fall 截断计数进 counts",
            doc4["counts"]["frames_after_fall"] == 10)
        # 负例：记录含 fall_step 之后的帧 → 截断必须触发并计数
        fs_n_use, fs_info = fall_clip(1000, 400, win=200)
        chk("fall_clip 负例：1000 帧 + fall_step=400 → 用 400 帧、丢 9 窗、记 600 摔后帧",
            fs_n_use == 400 and fs_info["frames_after_fall"] == 600
            and fs_info["windows_dropped_by_fall_clip"] == expect_windows(1000)
            - expect_windows(400), f"{fs_info}")
        chk("fall_clip 未摔（-1/None）→ 不截断",
            fall_clip(300, -1)[0] == 300 and fall_clip(300, None)[0] == 300
            and fall_clip(300, None)[1]["fall_step"] is None)
        # stride 必须跟随本次运行（写死脊柱缺省 100 会在 G2-G4 换 stride 时错报）
        d50 = fall_clip(1000, 400, win=200, stride=50)[1]["windows_dropped_by_fall_clip"]
        d100 = fall_clip(1000, 400, win=200, stride=100)[1]["windows_dropped_by_fall_clip"]
        chk("fall_clip 丢弃窗数用本次 stride（stride=50 → 12，stride=100 → 6，非同一值）",
            d50 == 12 and d100 == 6, f"stride50={d50} stride100={d100}")
        chk("fall_clip info 记录本次 win/stride（可追溯）",
            fall_clip(1000, 400, win=200, stride=50)[1]["windows_dropped_stride"] == 50
            and fall_clip(1000, 400, win=200, stride=50)[1]["windows_dropped_win"] == 200)

        print("[selftest 3] 源 4 等待判据（不真等）")
        chk("records_ready：未达期望 → False 且报差",
            not records_ready({"records_n004": 3}, {"records_n004": 12})[0])
        chk("records_ready：达标 → True",
            records_ready({"records_n004": 12, "records_n008": 12},
                          {"records_n004": 12, "records_n008": 12})[0])
        chk("records_ready：期望内的目录为空 → False（不把空当就绪）",
            not records_ready({"records_n008": 0}, {"records_n008": 8})[0])
        chk("records_ready：目录改名（n004 子串匹配 records_n004）仍认就绪",
            records_ready({"n004": 12}, {"records_n004": 12})[0])
        chk("records_ready：总数不足 → False（防只到一类就开跑）",
            not records_ready({"records_n004": 12}, {"records_n004": 12,
                                                     "records_n008": 12,
                                                     "records_fallseg": 8})[0])
        chk("discover_record_tags 发现 records* 三目录",
            sorted(discover_record_tags(src4_dir)) == ["records_n004", "records_n008"])
        chk("parse_expect 解析 tag=n",
            parse_expect("records_n004=12,records_n008=12")
            == {"records_n004": 12, "records_n008": 12})

        print("[selftest 4] 四源装配（含 src2/3 桩 manifest + 负例）")
        # src2/src3 桩：只有装配需要的字段（结构与真 manifest 同）
        def stub_manifest(path, n_win, terrain, params, tv, split="train"):
            doc = {"exp": "D060", "created_at": "stub",
                   "gate": "G1 格式冒烟（stub 源）",
                   "script": "stub", "format": {"format_version": "d060.1", "fps": 50.0,
                                                "win": W.WIN, "stride": W.STRIDE,
                                                "state_dim": W.STATE_DIM,
                                                "token_scale": 16},
                   "counts": {"windows": n_win, "window_frames": n_win * W.WIN,
                              "windows_train": n_win, "windows_heldout": 0, "shards": 1,
                              "segments_used": 1, "token_min": -8, "token_max": 8},
                   "shards": [{"name": "w.npz", "n": n_win, "path": path}],
                   "windows": [{"stem": f"stub_{i}", "split": split, "speedA": tv,
                                "target_vel": tv, "terrain": terrain,
                                "terrain_params": params,
                                "intent_source": "self_corpus"} for i in range(n_win)],
                   "reconciliation": {"ok": True}}
            dump_json(doc, path)
            return path
        p2 = stub_manifest(os.path.join(root, "src2_stub.json"), 131, "climbing_box",
                           {"height": 0.5}, 0.074)
        p3 = stub_manifest(os.path.join(root, "src3_stub.json"), 732, "plane", {}, 0.128)
        asm = build_assembly([(SRC1_ID, "src1", os.path.join(out1, "manifest.json"), None),
                              (SRC2_ID, "src2", p2, STATE_SOURCE_KIN),
                              (SRC3_ID, "src3", p3, STATE_SOURCE_KIN),
                              (SRC4_ID, "src4", os.path.join(out4, "manifest.json"), None)])
        want_total = want1 + 131 + 732 + want4
        chk(f"四源窗数合计 = {want_total}", asm["totals"]["windows"] == want_total,
            f"{asm['totals']['windows']}")
        chk("四源帧数 = 窗数×200",
            asm["totals"]["window_frames"] == want_total * W.WIN)
        chk("四源对账全 ok（分片 = 逐窗记录 = counts）", asm["reconciliation"]["ok"]
            and all(b["reconciliation"]["ok"] for b in asm["sources"].values()))
        chk("state_source 分列：kinematics 三源 + q_des_cmd_root 一源",
            asm["state_source_dist"] == {STATE_SOURCE_KIN: 3, STATE_SOURCE_QDES: 1},
            f"{asm['state_source_dist']}")
        chk("state_source 证据分列（改编排声明 vs 推断）",
            "declared" in asm["sources"][SRC1_ID]["state_source_evidence"]
            and "inferred" in asm["sources"][SRC3_ID]["state_source_evidence"])
        chk("terrain 分布分列（四源合计按源前缀）",
            set(asm["terrain_dist_total"]) == {f"{SRC1_ID}:plane",
                                               f"{SRC2_ID}:climbing_box",
                                               f"{SRC3_ID}:plane",
                                               f"{SRC4_ID}:rough_paper"},
            f"{asm['terrain_dist_total']}")
        chk("speedA 分位逐源存在（四源全有，从逐窗记录重算）",
            all(asm["sources"][s]["speedA_dist"] for s in
                (SRC1_ID, SRC2_ID, SRC3_ID, SRC4_ID))
            and all(asm["sources"][s]["speedA_dist_recomputed_matches_declared"]
                    for s in (SRC1_ID, SRC2_ID, SRC3_ID, SRC4_ID)))
        chk("speedA 自报 vs 重算：一致时 diff 为空（全字段比对）",
            all(asm["sources"][s]["speedA_dist_recomputed_vs_declared_diff"] == []
                for s in (SRC1_ID, SRC4_ID)))
        # 负例：逐窗标签系统性偏移（自报块不动）→ 全字段比对必须抓到（只比 n+median 抓不到）
        shifted = os.path.join(root, "src1_shifted.json")
        sd = load_json(os.path.join(out1, "manifest.json"))
        for w in sd["windows"]:
            if w.get("speedA") is not None:
                w["speedA"] = float(w["speedA"]) + 0.02
        dump_json(sd, shifted)
        sb = source_summary(SRC1_ID, "src1-shifted", shifted, sd)
        chk("负例：逐窗 speedA 整体 +0.02 → 对账 FAIL 且 diff 列明字段",
            sb["speedA_dist_recomputed_matches_declared"] is False
            and {f["field"] for f in sb["speedA_dist_recomputed_vs_declared_diff"]}
            >= {"min", "q1", "median", "q3", "max"},
            f"{[f['field'] for f in sb['speedA_dist_recomputed_vs_declared_diff']]}")
        # 负例：0 窗源 → 对账必须 ok=True（`or -1` 会把它误判成 FAIL）
        p0 = stub_manifest(os.path.join(root, "src0_empty.json"), 0, "plane", {}, 0.0)
        b0 = source_summary(SRC3_ID, "src-empty", p0, load_json(p0))
        chk("负例/边界：0 窗源对账 ok=True（0 就是 0，不被 or 吞成 -1）",
            b0["reconciliation"]["ok"] is True
            and b0["reconciliation"]["counts_windows"] == 0
            and b0["reconciliation"]["counts_windows_used"] == 0
            and b0["g1_windows_ok"] is False,
            f"{b0['reconciliation']}")
        chk("format_common 逐源核对（win/state_dim/version 一致）",
            asm["format_common"]["all_win_equal"]
            and asm["format_common"]["all_state_dim_equal"]
            and asm["format_common"]["all_format_version_equal"]
            and asm["format_common"]["state_dim"] == 36)
        chk("G1 判据块：四源窗数逐源 + all_sources_ge_min（本 fixture 源 1/4 < 100 如实记 False）",
            asm["g1_judgement_material"]["windows_per_source"][SRC4_ID] == want4
            and asm["g1_judgement_material"]["all_sources_ge_min"] is False)
        chk("装配只引用不合并：每源 manifest_path 指向原 manifest 且可读，窗记录不复制到装配件",
            all(os.path.isfile(b["manifest_path"]) for b in asm["sources"].values())
            and all("windows" not in b for b in asm["sources"].values()))
        # 负例：窗数与 counts 不一致 → 该源对账 FAIL，装配整体 ok=False
        bad = stub_manifest(os.path.join(root, "src3_bad.json"), 732, "plane", {}, 0.1)
        d = load_json(bad)
        d["counts"]["windows"] = 700
        dump_json(d, bad)
        asm_bad = build_assembly([(SRC3_ID, "src3", bad, STATE_SOURCE_KIN)])
        chk("负例：counts.windows ≠ 逐窗记录 → 该源对账 FAIL",
            asm_bad["sources"][SRC3_ID]["reconciliation"]["ok"] is False
            and asm_bad["reconciliation"]["ok"] is False)
        miss = build_assembly([(SRC2_ID, "src2", os.path.join(root, "nope.json"),
                                STATE_SOURCE_KIN)])
        chk("负例：manifest 缺失 → sources_missing 记名（不静默）",
            miss["totals"]["sources_missing"] == 1 and miss["sources_missing"][0]["source_id"]
            == SRC2_ID)

        print("[selftest 5] 边界与口径")
        cmd, info = src1_command({"nominal": {"target_vel": 0.4, "mode_id": 2,
                                             "mode_name": "WALK", "height": 0.5,
                                             "direction_name": "left",
                                             "movement_direction": [0.0, 1.0, 0.0]}})
        chk("movement_direction 左 = +pi/2；height 有真值即入 has_truth",
            abs(cmd["cmd"]["movement_direction"] - np.pi / 2) < 1e-12
            and cmd["has_truth"]["height"] and cmd["cmd"]["height"] == 0.5)
        chk("未知 terrain 回落 rough_paper 并落账（不静默）",
            src4_terrain({"terrain": "stairs", "terrain_noise": 0.05}, {})["type"]
            == "rough_paper")
        chk("plane terrain 不塞 noise 参数",
            src4_terrain({"terrain": "plane", "terrain_noise": 0.05}, {})["params"] == {})
        chk("源 4 键名与记录器 RECORD_KEYS 同构",
            set(SRC4_KEYS.values()) == {"tokens_exec", "q_des", "root_quat", "root_pos"})
        chk("源 1 键名与生成器产物同构",
            set(SRC1_KEYS.values()) == {"tokens", "jp_mujoco", "root_quat", "root_pos"})
        chk("源 4 intent 口径与脊柱常量**显式分歧**（oracle_replay ≠ closed_loop："
            "G1 派发口径已写入产物，改值=改判据，故登记而非静默并存）",
            getattr(W, "INTENT_SOURCE_CLOSED_LOOP", None) == "closed_loop"
            and INTENT_ORACLE == "oracle_replay"
            and INTENT_ORACLE != getattr(W, "INTENT_SOURCE_CLOSED_LOOP", None),
            f"脊柱={getattr(W, 'INTENT_SOURCE_CLOSED_LOOP', None)!r} 本驱动={INTENT_ORACLE!r}")
        chk("dist_fields_diff：仅 median 相同、min/max 偏移 → 仍判 FAIL（全字段比对）",
            dist_fields_diff({"n": 2, "min": 0.1, "q1": 0.2, "median": 0.3,
                              "q3": 0.4, "max": 0.5, "tertile_edges": [0.2, 0.4]},
                             {"n": 2, "min": 0.2, "q1": 0.25, "median": 0.3,
                              "q3": 0.45, "max": 0.6, "tertile_edges": [0.25, 0.45]},
                             tol=1e-9) != []
            and dist_fields_diff({"n": 2, "min": 0.1, "q1": 0.2, "median": 0.3,
                                  "q3": 0.4, "max": 0.5,
                                  "tertile_edges": [0.2, 0.4]},
                                 {"n": 2, "min": 0.1, "q1": 0.2, "median": 0.3,
                                  "q3": 0.4, "max": 0.5,
                                  "tertile_edges": [0.2, 0.4]}) == [])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n[selftest] 通过 {n_ok} / 失败 {n_bad}")
    return n_bad == 0


# ------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(
        description="D060 §5t G1 集成驱动：源 1/源 4 切窗 + 四源装配 + 消费者测试")
    ap.add_argument("cmd", nargs="?", default=None,
                    choices=["src1", "src4", "assembly", "consumer", "selftest"])
    ap.add_argument("--selftest", action="store_true", help="合成 fixture 自检")
    ap.add_argument("--win", type=int, default=W.WIN)
    ap.add_argument("--stride", type=int, default=W.STRIDE)
    ap.add_argument("--fps", type=float, default=W.FPS)
    ap.add_argument("--shard-size", type=int, default=128)
    # src1
    ap.add_argument("--npz-dirs", nargs="+", default=[DEFAULT_SRC1_DIR])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument("--split-json", default=DEFAULT_SPLIT_JSON)
    # src4
    ap.add_argument("--records-dir", default=DEFAULT_SRC4_RECORDS)
    ap.add_argument("--dirs", nargs="+", default=None, help="记录子目录 tag（缺省自动发现）")
    ap.add_argument("--expect", default=None, help="就绪期望，如 records_n004=12,...")
    ap.add_argument("--wait-minutes", type=float, default=0.0)
    ap.add_argument("--poll-seconds", type=float, default=600.0)
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--wait-trace", default=None,
                    help="复用既有 src4_wait_trace.json（--skip-wait 重跑时挂回到达时间线）")
    ap.add_argument("--limit-dirs", type=int, default=None)
    ap.add_argument("--limit-per-dir", type=int, default=None)
    # assembly / consumer
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--src1", default=DEFAULT_SOURCES[0][2])
    ap.add_argument("--src2", default=DEFAULT_SOURCES[1][2])
    ap.add_argument("--src3", default=DEFAULT_SOURCES[2][2])
    ap.add_argument("--src4", default=DEFAULT_SOURCES[3][2])
    ap.add_argument("--notes", default=None)
    ap.add_argument("--consumer-summary", default=None)
    ap.add_argument("--stub", default=None)
    ap.add_argument("--python", default=None)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--split", default="train", choices=["train", "heldout"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.02)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    cmd = args.cmd
    if args.selftest and cmd is None:
        cmd = "selftest"
    if cmd == "selftest" or args.selftest:
        return 0 if run_selftest() else 1
    if cmd == "src1":
        if not args.out_dir:
            args.out_dir = args.manifest_out or DEFAULT_SRC1_OUT
        if not args.split_json:
            args.split_json = None
        run_src1_mode(args)
        return 0
    if cmd == "src4":
        if not args.out_dir:
            args.out_dir = DEFAULT_SRC4_OUT
        if not args.split_json:
            args.split_json = None
        run_src4_mode(args)
        return 0
    if cmd == "assembly":
        if not args.out_dir:
            args.out_dir = DEFAULT_ASSEMBLY_OUT
        args.consumer_summary = args.consumer_summary or os.path.join(
            args.out_dir, "consumer_summary.json")
        run_assembly_mode(args)
        return 0
    if cmd == "consumer":
        if not args.out_dir:
            args.out_dir = DEFAULT_ASSEMBLY_OUT
        return run_consumer_mode(args)
    raise SystemExit("用法：d060_intake_src14.py {src1|src4|assembly|consumer|selftest} …"
                     "（--help）")


if __name__ == "__main__":
    sys.exit(main())
