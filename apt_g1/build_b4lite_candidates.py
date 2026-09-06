"""B4-lite S2 后半 + S3 清洗台账 + S4 划分冻结（D045，DS_B4LITE_ACTION_SPLIT_PLAN §2/§3/§6）。

S2 后半：读 desc_family_per_segment.csv（L1a 段级族归属）+ seed_metadata_v004.parquet
（51 列官方标注），候选池硬条件筛选（逐段原因记录）+ 每段五维语义标签（移动方式/
上肢/姿态/接触/时间，三级来源声明，temporal_labels 缺失走 §7.2 整段粗标签 fallback）。
S3：三层清洗台账（L1 空占位待 S5 回填；L2 逐段原因；L3 不清洗——A005 显式登记
boundary_ref 保留）。S4：划分冻结（演员 md5 三桶 80/10/10；6/2/2 族角色；T-fam 族
演员与训练/开发 8 族零交集；泄漏检查三项）+ 每族抽 5 段轨迹核实（translate 差分
v_med/v_p90/航向变化，120fps，cm→m，D040 M1 同法）。

owner 裁定（2026-09-07 固定，脚本内不许改）：
  - 训练 6 族: forward_walk/fast_walk_run/forward_jump/dance_rhythm/turn_walk/
    start_stop_transition；开发 2 族: asym_upper/posture_change；
    T-fam 留出 2 族: lateral/slow_walk（slow_walk=L1a 212 段名称命中，不分档扩展）。
  - 配额: 训练族各 18（~3 段 T-seg 从测试演员桶）、开发族各 10、T-fam 族各 12。
  - 演员桶: md5(actor_uid) mod 10，0-7 训练桶 / 8 开发桶 / 9 测试桶。
  - 边界段: walk_forward_grab_injured_L_leg_002__A005 登记 L3 保留 + t_role=boundary_ref。

v2 修订（D045 fix，2026-09-07）：v1 用 is_neutral=1 一刀切排除，标定发现该旗标覆盖
10 族候选 90.3%（标准语料总旗标，Dancing/Gestures 全覆盖），dance_rhythm/lateral/
slow_walk 三族零入选。改为描述文本定向排除 box/系统攀爬/正障碍族（EXCLUDE_PATTERNS，
命中原因 box_climb_desc），is_neutral 仅作审计字段保留；其余硬条件与选择规则不动。

Usage (server, venv_isaac):
    cd ~/ros2_data/apt_g1 && python apt_g1/build_b4lite_candidates.py
产物（--out-dir，默认 data/ds_bones/g1_b4lite/）:
    candidates.json / candidates_stats.json / cleaning_ledger.json /
    split_assignments.json / leak_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HOME = os.path.expanduser("~")
DS_DIR = f"{HOME}/ros2_data/apt_g1/data/ds_bones"
DEFAULT_PARQUET = f"{DS_DIR}/seed_metadata_v004.parquet"
DEFAULT_SEG_CSV = f"{DS_DIR}/g1_b4lite/desc_family_per_segment.csv"
DEFAULT_OUT_DIR = f"{DS_DIR}/g1_b4lite"

FPS = 120  # G1 原生 CSV 帧率（D043）
MIN_FRAMES, MAX_FRAMES = 360, 7200  # 3s–60s
MAX_PER_ACTOR_FAM = 2  # 每演员每族上限（多样性）
TSEG_MIN_PER_TRAIN_FAM = 3  # 训练族 T-seg 目标段数（测试演员桶）

# D045 v2 修订（fix：三族零入选）：is_neutral 旗标不再单独排除。标定（seed_metadata_v004）：
# 10 族候选 112,547/124,609（90.3%）带旗，category 横跨 Dancing/Gestures/Object Manipulation
# 全部类别——该旗标是"标准动作语料"总标记，非 50cm box 族专用；dance_rhythm 14,155/14,161
# 段带旗，抽查 macarena/moonwalk 均为普通舞蹈，一刀切导致 dance_rhythm/lateral/slow_walk
# 三族零入选。协议本意（§2）=「50cm box 正障碍语料另用，不入首版」→ 改为描述文本定向排除
# box/系统攀爬族。合并文本 = filename stem + content_name + 全部描述列（lower+下划线转空格）。
EXCLUDE_PATTERNS: list[tuple[str, str]] = [
    (r"50\s*cm\s*box", "50cm_box"),
    (r"\bcome\s+(?:up|down)\b[^.;]{0,40}\bbox\b", "come_updown_box"),
    # D043 记录名 neutral_come_up/down_50cm_box 在 v004 stem 实为 come_up/down_50cm_box_R
    #（无 neutral 前缀），上一模式已覆盖；此条为防御性保留（标定 0 命中）。
    (r"\bneutral\s+come\s+(?:up|down)\b", "neutral_come_updown"),
    (r"\bget\s+(?:up\s+onto|down\s+from)\b[^.;]{0,40}\bbox\b", "get_ontofrom_box"),
    (r"\bclimb\w*\b", "climb"),
    # 以下两条为"拿不准宁可多排除"（quality_notes 登记）：obstacle=正障碍语料近邻
    #（10 族内 1,464 段 push/avoid/jump over 系列）；jump*box=箱上/箱间跳跃（216 段）。
    (r"\bobstacle\b", "obstacle"),
    (r"\b(?:jump|leap)\w*[^.;]{0,30}\bbox\b|\bbox\b[^.;]{0,30}\b(?:jump|leap)\w*\b", "jump_box"),
]
EXCL_RE = [(re.compile(p, re.IGNORECASE), n) for p, n in EXCLUDE_PATTERNS]
# 保护声明：dance_hiphop_box_step（舞步名含 box，86 段）不含 jump/come_updown/50cm，
# 不被任何模式命中，不受排除影响。

EXCL_TEXT_COLS = ("content_name", "content_short_description", "content_short_description_2",
                  "content_technical_description", "content_natural_desc_1",
                  "content_natural_desc_2", "content_natural_desc_3", "content_natural_desc_4")


def excl_text(row: dict) -> str:
    parts = [str(row.get(c)) for c in EXCL_TEXT_COLS if row.get(c) is not None]
    parts.append(re.sub(r"_\d+__A\d+(_M)?$", "", str(row.get("filename") or "")))
    return " ".join(parts).lower().replace("_", " ")


def excl_pattern_hits(text: str) -> list[str]:
    return [name for rx, name in EXCL_RE if rx.search(text)]


TRAIN_FAMS = ["forward_walk", "fast_walk_run", "forward_jump", "dance_rhythm",
              "turn_walk", "start_stop_transition"]
DEV_FAMS = ["asym_upper", "posture_change"]
TFAM_FAMS = ["lateral", "slow_walk"]
ALL_FAMS = TRAIN_FAMS + DEV_FAMS + TFAM_FAMS
QUOTA = {**{f: 18 for f in TRAIN_FAMS}, **{f: 10 for f in DEV_FAMS},
         **{f: 12 for f in TFAM_FAMS}}
BOUNDARY_STEM = "walk_forward_grab_injured_L_leg_002__A005"

NAME_RE = re.compile(r"^(?P<desc>.+?)_(?P<seq>\d+)__A(?P<actor>\d+)(?P<mir>_M)?$")
CONF_RANK = {"high": 0, "med": 1, "low": 2, "other": 3}

# 桶判定：md5(actor_uid) mod 10；0-7 训练 / 8 开发 / 9 测试（owner 裁定②）
def actor_bucket(actor_uid: str) -> int:
    return int(hashlib.md5(actor_uid.encode("utf-8")).hexdigest(), 16) % 10

BUCKET_ROLE = lambda b: "train" if b <= 7 else ("dev" if b == 8 else "test")

# ---------------------------------------------------------------------------
# 五维标签词表（§1.1；filename/metadata 通道，manual 通道后续人工抽看另做）
# ---------------------------------------------------------------------------
FWD_RE = re.compile(r"\b(?:walk|forward|march|step|stroll|jog|run|sprint|jump|hop|leap)\w*\b")
TURN_RE = re.compile(r"\b(?:turn|pivot|spin)\w*\b")
SIDE_RE = re.compile(r"\b(?:strafe\w*|lateral(?:ly)?|sideways?|sidestep\w*|side\s+steps?\b)")
SS_RE = re.compile(r"\b(?:start\w*|stop(?:s|ping|ped)?|transition\w*)\b")

UPPER_ANY_RE = re.compile(
    r"\b(?:wav(?:e|es|ing|ed)|clap(?:s|ping|ped)?|arms?|hands?|reach(?:es|ing|ed)?|"
    r"grab(?:s|bing|bed)?|punch(?:es|ing|ed)?|push(?:es|ing|ed)?|pull(?:s|ing|ed)?|"
    r"throw(?:s|ing)?|catch(?:es|ing)?|carry(?:ing)?|lift(?:s|ing|ed)?|point(?:s|ing|ed)?|"
    r"thumbs?|fists?|palms?|fingers?|shoulders?|elbows?|wrists?|gestures?(?:ing)?|"
    r"salut(?:e|es|ing|ed)|hugs?(?:ging|ged)?|stretch(?:es|ing|ed)?|hold(?:s|ing)?|"
    r"knock(?:s|ing|ed)?|swings?(?:ing)?)\b")
UPPER_ASYM_RE = re.compile(
    r"\b(?:wav(?:e|es|ing|ed)|salut(?:e|es|ing|ed)|point(?:s|ing|ed)?|thumbs?|"
    r"punch(?:es|ing|ed)?|grab(?:s|bing|bed)?|reach(?:es|ing|ed)?|carry(?:ing)?|"
    r"lift(?:s|ing|ed)?|push(?:es|ing|ed)?|pull(?:s|ing|ed)?|throw(?:s|ing)?|"
    r"knock(?:s|ing|ed)?|hold(?:s|ing)?|left|right|single|one)\b")
UPPER_SYM_RE = re.compile(
    r"\b(?:clap(?:s|ping|ped)?|both|hugs?(?:ging|ged)?|crossed|folded|joined|stretch(?:es|ing|ed)?)\b")
MAG_LARGE_RE = re.compile(
    r"\b(?:big|large|wide|overhead|above\s+(?:the\s+)?head|high|vigorous|energetic|"
    r"raised|extends?|wildly|rapidly)\b")
MAG_SMALL_RE = re.compile(r"\b(?:slight(?:ly)?|small|gentl(?:e|y)|light(?:ly)?|little|soft(?:ly)?)\b")

LAND_RE = re.compile(r"\bland(?:s|ing|ed)?\b")
JUMP_RE = re.compile(r"\b(?:jump|leap|hop)\w*\b")

MOVEMENT_DEFAULT = {
    "forward_walk": "前进", "fast_walk_run": "前进", "slow_walk": "前进",
    "forward_jump": "前进", "turn_walk": "转向", "lateral": "侧移",
    "start_stop_transition": "启停", "dance_rhythm": "混合",
    "asym_upper": "混合", "posture_change": "混合",
}
CONTACT_DEFAULT = {
    "dance_rhythm": "混合", "start_stop_transition": "混合",
    "asym_upper": "双支撑", "posture_change": "双支撑",
}
# content_body_position 非直立词表：(词干, 标签, 严重度)；severity 高者胜（§1.1 姿态维）
POSTURE_WORDS = [
    ("kneel", "屈膝", 2), ("crouch", "屈膝", 2), ("croach", "屈膝", 2), ("squat", "屈膝", 2),
    ("lean", "倾斜", 1), ("tilt", "倾斜", 1), ("bend", "倾斜", 1),
    ("sit", "其他", 3), ("crawl", "其他", 3), ("all fours", "其他", 3), ("lying", "其他", 3),
]


def norm_parts(filename: str) -> dict | None:
    """filename -> {desc, actor, is_mirror, take_id, stem}；解析失败返回 None。"""
    m = NAME_RE.match(filename)
    if m is None:
        return None
    is_mir = m.group("mir") == "_M"
    return {
        "stem": filename,
        "desc": m.group("desc").lower(),
        "actor_from_name": f"A{m.group('actor')}",
        "is_mirror": is_mir,
        "mother": filename[:-2] if is_mir else filename,
    }


def meta_text(row: dict) -> str:
    return " ".join(str(row.get(f)) for f in
                    ("content_natural_desc_1", "content_natural_desc_2",
                     "content_natural_desc_3", "content_natural_desc_4",
                     "content_short_description", "content_short_description_2",
                     "content_technical_description")
                    if row.get(f) is not None).lower().replace("_", " ")


def five_dim_labels(family: str, desc: str, row: dict) -> tuple[dict, dict, bool]:
    """返回 (labels, signals, metadata_used)。manual 通道恒 false（后续人工抽看另做）。"""
    text = meta_text(row)
    hmove = int(row.get("content_horizontal_move") or 0)
    vmove = int(row.get("content_vertical_move") or 0)
    tom = str(row.get("content_type_of_movement") or "").strip().lower()
    bp = str(row.get("content_body_position") or "").strip().lower()
    metadata_used = False

    # -- 移动方式 {前进,转向,侧移,启停,混合}
    sig: set[str] = set()
    if FWD_RE.search(desc):
        sig.add("前进")
    if TURN_RE.search(desc):
        sig.add("转向")
    if SIDE_RE.search(desc):
        sig.add("侧移")
    if SS_RE.search(desc):
        sig.add("启停")
    if len(sig) == 1:
        movement = sig.pop()
    elif len(sig) > 1:
        movement = "混合"
    else:
        movement = MOVEMENT_DEFAULT[family]
        if movement == "前进" and hmove != 1:
            movement = "混合"  # 前进族但官方标注无水平位移 -> 归混合并在 signals 注明
        metadata_used = metadata_used or (hmove != 1 and MOVEMENT_DEFAULT[family] == "前进")

    # -- 上肢 {对称,非对称,小幅,大幅,最小}
    up_any = bool(UPPER_ANY_RE.search(text))
    if not up_any and family != "asym_upper":
        upper = "最小"
    else:
        if MAG_LARGE_RE.search(text):
            upper = "大幅"
        elif MAG_SMALL_RE.search(text):
            upper = "小幅"
        elif UPPER_ASYM_RE.search(text) or family == "asym_upper":
            upper = "非对称"
        else:
            upper = "对称"
        metadata_used = True

    # -- 姿态 {直立,屈膝,倾斜,其他}（content_body_position 直读，非直立取严重度最高者）
    posture, psev = "直立", 0
    if bp and bp != "standing":
        metadata_used = True
        for stem_w, lab, sev in POSTURE_WORDS:
            if stem_w in bp and sev > psev:
                posture, psev = lab, sev
        if psev == 0:
            posture = "其他"  # 非站立且词表未命中（如 sitting on floor 变体）

    # -- 接触 {双支撑,单支撑,腾空,落地,混合}（tom+族；族先验 jump=腾空 walk=单支撑）
    csig: list[str] = []
    if LAND_RE.search(text) or LAND_RE.search(desc):
        csig.append("落地")
    if family == "forward_jump" or tom == "jumping" or (vmove == 1 and JUMP_RE.search(desc)):
        csig.append("腾空")
    if tom in ("walking", "jogging", "turning") or family in (
            "forward_walk", "fast_walk_run", "slow_walk", "turn_walk", "lateral"):
        csig.append("单支撑")
    if not csig and (hmove == 0 or tom in ("standing idle", "gesture", "action",
                                           "sitting", "standing")):
        csig.append("双支撑")
    csig = list(dict.fromkeys(csig))
    if len(csig) == 1:
        contact = csig[0]
    elif len(csig) > 1:
        contact = "混合"
    else:
        contact = CONTACT_DEFAULT[family]
    if csig or tom or hmove or vmove:
        metadata_used = True

    # -- 时间结构 {稳态,含启停,连续过渡}（整段粗标签，§7.2 fallback，置信度降级）
    if family == "start_stop_transition" or SS_RE.search(desc):
        temporal = "含启停"
    elif family in ("posture_change", "asym_upper"):
        temporal = "连续过渡"
    else:
        temporal = "稳态"

    labels = {
        "movement": movement,
        "upper_body": upper,
        "posture": posture,
        "contact": contact,
        "temporal": temporal,
    }
    signals = {
        "movement_filename_hits": sorted(s for s in
                                         (("前进" if FWD_RE.search(desc) else ""),
                                          ("转向" if TURN_RE.search(desc) else ""),
                                          ("侧移" if SIDE_RE.search(desc) else ""),
                                          ("启停" if SS_RE.search(desc) else "")) if s),
        "upper_metadata_hit": up_any,
        "body_position_raw": bp,
        "contact_signals": csig,
        "type_of_movement_raw": tom,
    }
    return labels, signals, metadata_used


def hard_filter_reasons(row: dict) -> list[str]:
    """L2 硬条件（逐段全部命中的原因）：时长/道具/box-攀爬描述定向排除（D045 v2）。"""
    reasons = []
    n = int(row["move_duration_frames"])
    if n < MIN_FRAMES:
        reasons.append("duration_short")
    if n > MAX_FRAMES:
        reasons.append("duration_long")
    props = str(row.get("content_props") or "").strip()
    if props not in ("", "0"):
        reasons.append("props")
    if row.get("_excl_hits"):
        reasons.append("box_climb_desc")
    return reasons


def traj_check(csv_path: str, family: str, expect_frames: int) -> dict:
    """translate 差分速度/航向（120fps 原始，cm→m，D040 M1 同法）+ 矛盾 flag。"""
    out: dict = {"csv_path": csv_path, "flags": []}
    try:
        df = pd.read_csv(csv_path, usecols=["root_translateX", "root_translateY"])
    except Exception as e:  # noqa: BLE001 — 缺文件/坏行记 flag 不中断
        out["error"] = f"read_failed: {type(e).__name__}: {e}"
        out["flags"].append("csv_read_failed")
        return out
    xy = df.to_numpy(dtype=float) / 100.0  # cm -> m
    out["n_frames"] = int(xy.shape[0])
    if xy.shape[0] != expect_frames:
        out["flags"].append("frame_count_mismatch")
    if xy.shape[0] < 3:
        out["error"] = "too_few_frames"
        return out
    d = np.diff(xy, axis=0)
    speed = np.hypot(d[:, 0], d[:, 1]) * FPS
    out["v_med_ms"] = round(float(np.median(speed)), 4)
    out["v_p90_ms"] = round(float(np.percentile(speed, 90)), 4)
    heading = np.arctan2(d[:, 1], d[:, 0])
    valid = speed > 0.1  # 低速帧航向噪声大，不计入
    if valid.sum() > 1:
        hd = np.diff(np.unwrap(heading[valid]))
        out["heading_change_total_rad"] = round(float(np.abs(hd).sum()), 4)
        out["heading_change_total_deg"] = round(float(np.degrees(np.abs(hd).sum())), 2)
    else:
        out["heading_change_total_rad"] = 0.0
        out["heading_change_total_deg"] = 0.0
        out["flags"].append("mostly_stationary")
    # 标签矛盾 flag（与 parquet/族标签对照）
    if family in ("forward_walk", "slow_walk") and out["v_med_ms"] > 2.5:
        out["flags"].append("speed_too_fast_for_walk_family")
    if family == "forward_jump" and out["v_med_ms"] < 0.05:
        out["flags"].append("no_forward_motion_for_jump")
    if family == "lateral" and out.get("heading_change_total_rad", 0.0) > np.pi:
        out["flags"].append("excess_heading_for_lateral")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--seg-csv", default=DEFAULT_SEG_CSV)
    ap.add_argument("--g1-root", default=DS_DIR,
                    help="move_g1_path 的挂载根（含 g1/csv/...）")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--traj-per-family", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = pd.Timestamp.now()
    df = pd.read_parquet(args.parquet)
    seg = pd.read_csv(args.seg_csv)
    assert len(df) == len(seg), f"row mismatch parquet={len(df)} seg_csv={len(seg)}"
    df = df.merge(seg, on="filename", how="left", validate="one_to_one")
    assert df["family"].notna().all(), "seg_csv 未覆盖全部 filename"
    print(f"[load] rows={len(df)} cols={len(df.columns)}")

    rows = df.to_dict("records")
    cand: dict[str, list[dict]] = {f: [] for f in ALL_FAMS}  # 过硬条件候选（族内）
    fail_entries: list[dict] = []  # L2 台账
    reason_any: Counter = Counter()
    reason_primary: Counter = Counter()
    actor_mismatch: list[dict] = []
    parse_fail: list[str] = []
    REASON_PRIORITY = ["box_climb_desc", "props", "duration_short", "duration_long"]

    for row in rows:
        stem = row["filename"]
        fam = row["family"]
        parts = norm_parts(stem)
        if parts is None or fam not in ALL_FAMS:
            if parts is None:
                parse_fail.append(stem)
            continue
        # 演员键交叉校验：actor_uid 为准；文件名 __A\d+ 与 take_actor 不一致段记录
        a_uid = str(row.get("actor_uid") or "")
        a_take = str(row.get("take_actor") or "")
        if a_uid and a_take and a_uid != a_take:
            actor_mismatch.append({"stem": stem, "actor_uid": a_uid, "take_actor": a_take})
        rec = {
            "stem": stem, "family": fam,
            "confidence": row["confidence"],
            "norm_desc": row.get("norm_desc"), "backward": bool(row.get("backward")),
            **parts,
            "actor_uid": a_uid or parts["actor_from_name"],
            "take_actor": a_take,
            "is_mirror_col": bool(row.get("is_mirror")),
            "move_duration_frames": int(row["move_duration_frames"]),
            "content_props": str(row.get("content_props") or ""),
            "is_neutral": float(row.get("is_neutral") or 0.0),
            "move_g1_path": str(row.get("move_g1_path") or ""),
        }
        rec["actor_consistent"] = rec["actor_uid"] == parts["actor_from_name"] == (
            a_take or rec["actor_uid"])
        rec["actor_bucket"] = actor_bucket(rec["actor_uid"])
        row["_excl_hits"] = excl_pattern_hits(excl_text(row))
        reasons = hard_filter_reasons(row)
        if reasons:
            primary = next(r for r in REASON_PRIORITY if r in reasons)
            reason_primary[primary] += 1
            for r in reasons:
                reason_any[r] += 1
            fail_entries.append({
                "stem": stem, "family": fam, "layer": "L2", "reason": primary,
                "all_reasons": reasons,
                "detail": {"frames": rec["move_duration_frames"],
                           "props": rec["content_props"],
                           "is_neutral": rec["is_neutral"],
                           "excl_patterns": row.get("_excl_hits") or [],
                           "confidence": rec["confidence"]},
            })
            continue
        cand[fam].append(rec)

    # 镜像去重规则登记：母 take 同族过筛的镜像段恒不可选（选母不选镜像，§3.2）
    mirror_dedup: list[dict] = []
    pools: dict[str, list[dict]] = {}
    pass_set: dict[str, set[str]] = {}
    for fam in ALL_FAMS:
        lst = cand[fam]
        pass_set[fam] = {r["stem"] for r in lst}
        for r in lst:
            if r["is_mirror"] and r["mother"] in pass_set[fam]:
                mirror_dedup.append({
                    "stem": r["stem"], "family": fam, "layer": "L2",
                    "reason": "mirror_dedup",
                    "all_reasons": ["mirror_dedup"],
                    "detail": {"mother": r["mother"],
                               "note": "同 take _M 镜像不重复选（§3.2 派生同集合）"},
                })
        lst.sort(key=lambda r: (CONF_RANK.get(r["confidence"], 3), r["stem"]))
        pools[fam] = lst

    # ---------------- 选择：先训练+开发族（收集演员集），再 T-fam 族零交集 ----------------
    selections: dict[str, list[dict]] = {}
    fam_actor_count: dict[str, Counter] = {}
    actors_used_8fams: set[str] = set()
    tfam_conflict_skips: Counter = Counter()

    def try_pick(rec: dict, fam: str, picked: list[dict], cnt: Counter,
                 extra_excl: set[str] | None = None) -> bool:
        if any(p["stem"] == rec["stem"] for p in picked):
            return False
        if extra_excl and rec["actor_uid"] in extra_excl:
            return False
        if cnt[rec["actor_uid"]] >= MAX_PER_ACTOR_FAM:
            return False
        if rec["is_mirror"] and rec["mother"] in pass_set[fam]:
            return False
        return True

    for fam in TRAIN_FAMS + DEV_FAMS:
        pool = pools[fam]
        picked: list[dict] = []
        cnt: Counter = Counter()
        if fam in TRAIN_FAMS:  # 先保 ~3 段 T-seg（测试演员桶 bucket 9）
            n_tseg = 0
            for rec in pool:
                if n_tseg >= TSEG_MIN_PER_TRAIN_FAM:
                    break
                if rec["actor_bucket"] != 9 or not try_pick(rec, fam, picked, cnt):
                    continue
                picked.append(rec)
                cnt[rec["actor_uid"]] += 1
                n_tseg += 1
        for rec in pool:
            if len(picked) >= QUOTA[fam]:
                break
            if not try_pick(rec, fam, picked, cnt):
                continue
            picked.append(rec)
            cnt[rec["actor_uid"]] += 1
        selections[fam] = picked
        fam_actor_count[fam] = cnt
        actors_used_8fams |= {r["actor_uid"] for r in picked}

    for fam in TFAM_FAMS:  # T-fam：演员与 8 族选段演员零交集（§3.2 泄漏防线）
        pool = pools[fam]
        picked: list[dict] = []
        cnt: Counter = Counter()
        skipped_conflict = 0
        for rec in pool:
            if len(picked) >= QUOTA[fam]:
                break
            if rec["actor_uid"] in actors_used_8fams:
                skipped_conflict += 1
                continue
            if not try_pick(rec, fam, picked, cnt):
                continue
            picked.append(rec)
            cnt[rec["actor_uid"]] += 1
        selections[fam] = picked
        fam_actor_count[fam] = cnt
        tfam_conflict_skips[fam] = skipped_conflict

    # set_role / t_role（owner 规则：训练族∩测试演员桶->test(T-seg)；其余训练族->train；
    # 开发族->dev；T-fam 族->test(T-fam)；边界段->boundary_ref）
    for fam, picked in selections.items():
        for rec in picked:
            if fam in TFAM_FAMS:
                rec["set_role"], rec["t_role"] = "test", "T-fam"
            elif fam in DEV_FAMS:
                rec["set_role"], rec["t_role"] = "dev", "dev"
            elif rec["actor_bucket"] == 9:
                rec["set_role"], rec["t_role"] = "test", "T-seg"
            else:
                rec["set_role"], rec["t_role"] = "train", "train"

    # 边界段 A005：L3 保留（不清洗），显式入划分清单 t_role=boundary_ref
    boundary_rec: dict | None = None
    for row in rows:
        if row["filename"] == BOUNDARY_STEM:
            parts = norm_parts(row["filename"])
            boundary_rec = {
                "stem": BOUNDARY_STEM, "family": row["family"],
                "confidence": row["confidence"], "norm_desc": row.get("norm_desc"),
                "backward": bool(row.get("backward")), **parts,
                "actor_uid": str(row.get("actor_uid") or ""),
                "take_actor": str(row.get("take_actor") or ""),
                "actor_bucket": actor_bucket(str(row.get("actor_uid") or "")),
                "move_duration_frames": int(row["move_duration_frames"]),
                "content_props": str(row.get("content_props") or ""),
                "is_neutral": float(row.get("is_neutral") or 0.0),
                "move_g1_path": str(row.get("move_g1_path") or ""),
                "set_role": "dev",  # 族属开发族 asym_upper；L2 props 未过不入配额，仅边界登记
                "t_role": "boundary_ref",
            }
            break
    l3_entries = []
    if boundary_rec is not None:
        l3_entries.append({
            "stem": BOUNDARY_STEM, "family": boundary_rec["family"], "layer": "L3",
            "reason": "capability_boundary_record", "disposition": "保留",
            "detail": {"note": "D044 实例：离线回环 0.114 rad 健康 + Isaac 闭环 3-seed "
                               "系统性摔倒 @≈12.7s = 闭环空间漂移型（协议 §2，执行失败≠脏数据）；"
                               f"L2 通道该段 props={boundary_rec['content_props']!r} 未过筛，"
                               "不入首版配额，仅以 boundary_ref 登记",
                       "frames": boundary_rec["move_duration_frames"],
                       "actor_uid": boundary_rec["actor_uid"]},
        })

    # ---------------- 五维标签 + label_sources（对入选段 + 边界段）----------------
    row_by_stem = {r["filename"]: r for r in rows}

    def label_rec(rec: dict) -> dict:
        row = row_by_stem[rec["stem"]]
        labels, signals, meta_used = five_dim_labels(rec["family"], rec["norm_desc"] or "", row)
        rec["labels"] = labels
        rec["label_signals"] = signals
        rec["label_sources"] = {
            "filename": rec["confidence"] in ("high", "med"),
            "metadata": bool(meta_used),
            "manual": False,  # 人工抽看为后续工序（§1.3 三级来源第 3 级）
        }
        rec["temporal_note"] = "整段粗标签（temporal_labels 缺失，§7.2 fallback，置信度降级）"
        rec["repr_participation"] = {"vae": "适配训练待定", "norm_stats": "待定",
                                     "sonic_pretrain": "not_excludable"}
        return rec

    all_selected = [rec for picked in selections.values() for rec in picked]
    all_selected = [label_rec(r) for r in all_selected]
    if boundary_rec is not None:
        all_selected.append(label_rec(boundary_rec))

    # ---------------- 轨迹核实：每族抽 ≤5 段（均匀铺开）----------------
    traj_flags: Counter = Counter()
    by_fam_sel: dict[str, list[dict]] = defaultdict(list)
    for rec in all_selected:
        by_fam_sel[rec["family"]].append(rec)
    for fam, recs in sorted(by_fam_sel.items()):
        n = len(recs)
        idx = sorted(set(np.linspace(0, n - 1, min(args.traj_per_family, n))
                         .round().astype(int).tolist()))
        for i in idx:
            rec = recs[i]
            path = os.path.join(args.g1_root, rec["move_g1_path"])
            tc = traj_check(path, rec["family"], rec["move_duration_frames"])
            rec["traj_check"] = tc
            for f in tc["flags"]:
                traj_flags[f] += 1

    # ---------------- 泄漏检查三项 ----------------
    # ① _M 镜像与母 take 同集合=0 违例（入选段内同 take 只允许单侧 + 集合一致）
    take_sets: dict[str, set[str]] = defaultdict(set)
    for rec in all_selected:
        take_sets[rec["mother"]].add(f"{rec['set_role']}|{rec['t_role']}")
    mirror_viol = [
        {"take_id": k, "members": sorted(v)}
        for k, v in take_sets.items() if len(v) > 1
    ]
    # ② T-fam 族演员 ∩ 训练/开发族演员 = 0
    tfam_actors = {r["actor_uid"] for f in TFAM_FAMS for r in selections[f]}
    train_dev_actors = set(actors_used_8fams)
    tfam_overlap = sorted(tfam_actors & train_dev_actors)
    # ③ 同演员跨 set_role 段计数（演员留出报告用；T-fam 零交集为硬约束）
    actor_sets: dict[str, set[str]] = defaultdict(set)
    actor_sets_detail: dict[str, dict] = {}
    for rec in all_selected:
        actor_sets[rec["actor_uid"]].add(rec["set_role"])
    for a, ss in sorted(actor_sets.items()):
        if len(ss) > 1:
            fams = sorted({r["family"] for r in all_selected if r["actor_uid"] == a})
            n_seg = sum(1 for r in all_selected if r["actor_uid"] == a)
            actor_sets_detail[a] = {"set_roles": sorted(ss), "families": fams,
                                    "n_segments": n_seg}
    tfam_fams_set = {r["family"] for f in TFAM_FAMS for r in selections[f]}
    cross_set_in_tfam = {a: d for a, d in actor_sets_detail.items()
                         if any(f in tfam_fams_set for f in d["families"])}
    leak = {
        "_meta": {
            "script": "apt_g1/build_b4lite_candidates.py",
            "experiment": "D045",
            "protocol": "DS_B4LITE_ACTION_SPLIT_PLAN §3.2/§6-S4",
            "bucket_rule": "md5(actor_uid) mod 10: 0-7 训练桶 / 8 开发桶 / 9 测试桶",
        },
        "check1_mirror_mother_same_set": {
            "pass": len(mirror_viol) == 0, "n_violations": len(mirror_viol),
            "violations": mirror_viol,
            "rule": "入选段内同 take（母/_M）只出现一侧；跨集合即违例",
        },
        "check2_tfam_actor_zero_overlap": {
            "pass": len(tfam_overlap) == 0, "n_overlap": len(tfam_overlap),
            "tfam_actors": sorted(tfam_actors),
            "n_train_dev_actors": len(train_dev_actors),
            "overlap": tfam_overlap,
            "rule": "T-fam（lateral/slow_walk）选段演员与训练/开发 8 族选段演员零交集",
        },
        "check3_cross_set_actor_report": {
            "pass": len(cross_set_in_tfam) == 0,
            "n_actors_cross_set": len(actor_sets_detail),
            "detail": actor_sets_detail,
            "tfam_violations": cross_set_in_tfam,
            "rule": "同演员跨 set_role 允许存在于 train/test 内部统计（演员留出独立报告，"
                    "§3.2）；但 T-fam 族段所在演员必须单一 set_role（=test）",
        },
    }
    leak["pass_all"] = all(leak[k]["pass"] for k in
                           ("check1_mirror_mother_same_set", "check2_tfam_actor_zero_overlap",
                            "check3_cross_set_actor_report"))

    # ---------------- stats ----------------
    fam_quota_table = {}
    for fam in ALL_FAMS:
        picked = selections[fam]
        pool = pools[fam]
        conf_dist = Counter(r["confidence"] for r in picked)
        fam_quota_table[fam] = {
            "role": ("train" if fam in TRAIN_FAMS else
                     "dev" if fam in DEV_FAMS else "test(T-fam)"),
            "quota": QUOTA[fam],
            "selected": len(picked),
            "shortfall": QUOTA[fam] - len(picked),
            "actors": len({r["actor_uid"] for r in picked}),
            "tseg_selected": sum(1 for r in picked if r["t_role"] == "T-seg"),
            "pool_total_candidates": sum(1 for r in rows if r["family"] == fam),
            "pool_pass_hard_filters": len(pool),
            "pool_pass_takes": len({r["mother"] for r in pool}),
            "pool_remaining_takes": len({r["mother"] for r in pool}) - len(
                {r["mother"] for r in picked}),
            "mirror_dedup_pool": sum(1 for r in pool
                                     if r["is_mirror"] and r["mother"] in pass_set[fam]),
            "tfam_actor_conflict_skips": tfam_conflict_skips.get(fam, 0),
            "confidence_dist_selected": dict(conf_dist),
            "bucket_dist_selected": dict(Counter(BUCKET_ROLE(r["actor_bucket"])
                                                 for r in picked)),
        }

    # 候选池逐族逐原因失败计数（候选 = 10 族段；失败原因可叠加）
    fail_by_fam: dict[str, Counter] = {f: Counter() for f in ALL_FAMS}
    for e in fail_entries:
        for r in e["all_reasons"]:
            fail_by_fam[e["family"]][r] += 1

    stats = {
        "_meta": {
            "script": "apt_g1/build_b4lite_candidates.py",
            "experiment": "D045",
            "protocol": "DS_B4LITE_ACTION_SPLIT_PLAN §1.1/§2/§3/§6-S2/S3/S4",
            "owner_ruling": "2026-09-07：6/2/2 族角色 + slow_walk=212 段名称命中不分档 + "
                            "演员 md5 三桶 80/10/10 + 配额 6×18/2×10/2×12=152 + 边界双轨",
            "hard_filters": {
                "duration_frames": f"[{MIN_FRAMES}, {MAX_FRAMES}]（3s–60s）",
                "content_props": "非 '0'/非空 -> 排除进扩展池",
                "box_climb_desc": "描述文本定向排除 box/系统攀爬/正障碍族"
                                  "（模式见 box_climb_patterns）；is_neutral 旗标"
                                  "不单独排除（D045 v2 修订，依据见 quality_notes）",
                "mirror": "同 take 选母不选镜像",
                "per_actor_per_family": f"<={MAX_PER_ACTOR_FAM}",
                "tfam_actor_disjoint": "T-fam 族演员与训练/开发 8 族选段演员零交集",
            },
            "box_climb_patterns": {name: pat for pat, name in EXCLUDE_PATTERNS},
            "quality_notes": [
                "D045 v2 标定：is_neutral=1 覆盖 10 族候选 112,547/124,609（90.3%），"
                "category 横跨 Dancing(11,006)/Gestures/Object Manipulation 全部类别"
                "——标准语料总旗标而非 box 族专用；dance_rhythm 14,155/14,161 段带旗，"
                "抽查 macarena_001__A545 / moonwalk_R_001__A533 / buckets_R_001__A531 "
                "均为普通舞蹈应保留，一刀切系三族零入选根因。",
                "50cm box 族 v004 确切模式：stem come_up_50cm_box_R / come_down_50cm_box_R"
                "（+lift_crate 变体），sd='get up onto/down from 50 cm box || come up/down "
                "50cm box'；D043 记录名 neutral_come_up/down_50cm_box 在 v004 无 neutral "
                "前缀，neutral_come_updown 模式防御性保留（标定 0 命中）。",
                "拿不准宁可多排除（记录）：obstacle 模式打中正障碍语料近邻 1,464 段"
                "（push/avoid/jump over obstacle 系列，start_stop_transition 占 3,436 段带旗"
                "中的大部分）；jump_box 模式打中箱上/箱间跳跃 216 段"
                "（jump_form_box_to_safety_roll / jumping on a box 等）。",
                "保护声明：dance_hiphop_box_step（舞步名含 box，86 段）不含 "
                "jump/come_updown/50cm，不被任何模式命中。",
            ],
            "label_rule_note": "五维标签=filename/metadata 双通道规则打标（manual=false 待人工抽看）；"
                               "temporal=整段粗标签（§7.2 fallback）",
            "t_pair_note": "T-pair=动作×地形组合测试，属训练发射后评测口径：本划分不单独切段，"
                           "由 train 段的动作多样性支撑（五维标签覆盖表即其依据）",
        },
        "candidates_10fam_total": sum(1 for r in rows if r["family"] in ALL_FAMS),
        "pass_candidates_10fam": sum(len(v) for v in pools.values()),
        "selected_total": len(all_selected) - (1 if boundary_rec else 0),
        "selected_total_incl_boundary": len(all_selected),
        "quota_table": fam_quota_table,
        "shortfall_summary": {f: fam_quota_table[f]["shortfall"]
                              for f in ALL_FAMS if fam_quota_table[f]["shortfall"] > 0},
        "cleaning": {
            "L1": {"status": "placeholder_empty", "note": "格式错层 S5 转换时回填（转换器硬校验）",
                   "n_entries": 0},
            "L2_by_primary_reason": dict(reason_primary),
            "L2_by_any_reason": dict(reason_any),
            "L2_mirror_dedup": len(mirror_dedup),
            "L2_n_entries": len(fail_entries),
            "L3_n_entries": len(l3_entries),
            "by_family": {f: dict(c) for f, c in fail_by_fam.items() if c},
            "parse_fail_stems": len(parse_fail),
        },
        "actor": {
            "actor_uid_vs_take_actor_mismatch": len(actor_mismatch),
            "mismatch_detail": actor_mismatch[:20],
            "actor_uid_wins": True,
            "bucket_rule": "md5(actor_uid) mod 10: 0-7/8/9",
            "selected_bucket_dist": dict(Counter(BUCKET_ROLE(r["actor_bucket"])
                                                 for r in all_selected)),
        },
        "confidence_selected_dist": dict(Counter(r["confidence"] for r in all_selected)),
        "traj_check": {
            "per_family": args.traj_per_family,
            "n_checked": sum(1 for r in all_selected if "traj_check" in r),
            "flag_count_total": sum(traj_flags.values()),
            "flags": dict(traj_flags),
            "flag_rule": "speed_too_fast_for_walk_family: forward_walk/slow_walk v_med>2.5m/s; "
                         "no_forward_motion_for_jump: v_med<0.05; excess_heading_for_lateral: "
                         "航向总量>pi; frame_count_mismatch: CSV 行数≠parquet 帧数; csv_read_failed",
        },
        "leak_pass_all": leak["pass_all"],
        "elapsed_sec": round((pd.Timestamp.now() - t0).total_seconds(), 1),
    }

    # ---------------- 落盘 ----------------
    candidates = {
        "_meta": stats["_meta"],
        "segments": [{k: v for k, v in r.items() if k != "is_mirror_col"}
                     for r in all_selected],
    }
    split_segments = [{
        "stem": r["stem"], "family": r["family"], "actor_uid": r["actor_uid"],
        "actor_bucket": r["actor_bucket"],
        "set_role": r["set_role"], "t_role": r["t_role"],
        "repr_participation": r["repr_participation"],
    } for r in all_selected]
    split = {
        "_meta": {
            **stats["_meta"],
            "set_role_rule": "T-fam 族段->test(T-fam)；训练族∩测试演员桶->test(T-seg)；"
                             "训练族其余->train；开发族->dev；边界段 t_role=boundary_ref",
            "repr_participation_note": "§3.3 三处声明：VAE=适配训练待定；norm_stats=待定；"
                                       "SONIC 官方预训练 not_excludable（不可由本次划分排除）",
            "t_pair_note": stats["_meta"]["t_pair_note"],
        },
        "segments": split_segments,
    }
    ledger = {
        "_meta": {
            "script": "apt_g1/build_b4lite_candidates.py",
            "experiment": "D045",
            "protocol": "DS_B4LITE_ACTION_SPLIT_PLAN §2 三层清洗",
            "layer_rule": "L1 格式错=S5 转换时回填（本表空占位）；L2 内容适用范围=逐段原因；"
                          "L3 执行能力=保留不清洗（能力边界记录）",
        },
        "L1": [],
        "L2": fail_entries + mirror_dedup,
        "L3": l3_entries,
        "summary": {
            "L1": 0, "L2": len(fail_entries) + len(mirror_dedup), "L3": len(l3_entries),
            "L2_by_primary_reason": dict(reason_primary),
            "L2_by_any_reason": dict(reason_any),
            "L2_mirror_dedup": len(mirror_dedup),
        },
    }

    def w(name: str, obj: dict, compact: bool = False) -> None:
        p = os.path.join(args.out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            if compact:
                json.dump(obj, f, ensure_ascii=False,
                          separators=(",", ":"))
            else:
                json.dump(obj, f, ensure_ascii=False, indent=1)
        print(f"[write] {p}")

    w("candidates.json", candidates)
    w("candidates_stats.json", stats)
    w("cleaning_ledger.json", ledger, compact=True)
    w("split_assignments.json", split)
    w("leak_check.json", leak)

    print(f"\n[done] elapsed={stats['elapsed_sec']}s "
          f"selected={stats['selected_total']}/152 (incl boundary {len(all_selected)})")
    for fam in ALL_FAMS:
        t = fam_quota_table[fam]
        print(f"  {fam:22s} [{t['role']:12s}] quota={t['quota']:3d} sel={t['selected']:3d} "
              f"gap={t['shortfall']:3d} actors={t['actors']:3d} "
              f"pool={t['pool_pass_hard_filters']:5d}")
    print(f"  shortfall: {stats['shortfall_summary'] or 'none'}")
    print(f"  cleaning L2 primary: {dict(reason_primary)}  mirror_dedup: {len(mirror_dedup)}")
    print(f"  traj flags: {dict(traj_flags)}")
    print(f"  leak pass_all: {leak['pass_all']}  "
          f"tfam_actors={sorted(tfam_actors)}")


if __name__ == "__main__":
    main()
