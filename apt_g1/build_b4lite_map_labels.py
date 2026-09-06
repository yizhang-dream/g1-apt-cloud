"""B4-lite S2 前半（D045）：描述名 -> 语义族映射层（DS_B4LITE_ACTION_SPLIT_PLAN §1.3/§6-S2）。

读 seed_metadata_v004.parquet（142,220 段，51 列），把文件名细粒度描述名归一后按
关键词规则映射到 owner 裁定①的 10 语义族（2026-09-07 固定），产出：
  - desc_family_map.json   归一描述类 -> {family, confidence, matched, hits, n_segments}
  - map_stats.json         每族段数/占比、覆盖率、other 数、置信度分布（段级）
  - desc_family_per_segment.csv  段级映射（S2 后半五维打标直接复用，附加产物）

归一口径与 D040 M1 scan_smpl_metadata.py 的 NAME_RE 一致：
``<desc>_<seq>__A<actor>[_M]`` -> desc 小写、去 seq、去演员/镜像后缀。
映射在归一描述类层面做一次，段级展开；temporal_labels 缺失（fallback，协议 §7.2）
时映射即整段粗标签，置信度照实记录。

置信度（owner 裁定③口径的落法）：
  high = 名称命中且仅命中一个族；med = 名称命中多族取更具体者（优先级序）；
  low  = 名称无命中、仅 metadata 描述字段命中；other = 全无命中（低置信）。

Usage (server, venv_isaac):
    cd ~/ros2_data/apt_g1 && python build_b4lite_map_labels.py \
        --parquet data/ds_bones/seed_metadata_v004.parquet \
        --out-dir data/ds_bones/g1_b4lite
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

HOME = os.path.expanduser("~")
DEFAULT_PARQUET = f"{HOME}/ros2_data/apt_g1/data/ds_bones/seed_metadata_v004.parquet"
DEFAULT_OUT_DIR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite"

# 归一描述名解析：与 D040 NAME_RE 同口径（filename 无 .pkl 后缀，为 G1 CSV 名）
NAME_RE = re.compile(r"^(?P<desc>.+?)_(?P<seq>\d+)__A(?P<actor>\d+)(?P<mir>_M)?$")

# ---------------------------------------------------------------------------
# 10 语义族（owner 裁定① 2026-09-07 固定）。priority 序 = 具体度序：命中多族时
# 取序号更小（更具体）者。词表用词边界匹配（归一名下划线先转空格）。
# ---------------------------------------------------------------------------
FAMILY_RULES: list[tuple[str, str]] = [
    ("forward_jump", r"\bjump(?:s|ing|ed)?\b|\bhop(?:s|ing|ped)?\b|\bleap(?:s|ing|ed)?\b"),
    ("dance_rhythm", r"\bdance(?:s|d|ing|r)?\b"),
    ("posture_change", r"\bkneel(?:s|ing|ed)?\b|\bcrouch(?:es|ing|ed)?\b|\bbend(?:s|ing|ed)?\b|\blean(?:s|ing|ed)?\b|\btilt(?:s|ing|ed)?\b|\bsquat(?:s|ing|ted)?\b"),
    ("asym_upper", r"\bwav(?:e|es|ing|ed)\b|\bclap(?:s|ping|ped)?\b|\bgrab(?:s|bing|bed)?\b|\breach(?:es|ing|ed)?\b|\bpump(?:s|ing|ed)?\b|\bthumb(?:s)?\b|\barms?\b"),
    ("lateral", r"\bstrafe(?:s|ing)?\b|\blateral(?:ly)?\b|\bsideways?\b|\bside\s(?:step|walk|ways)\b|\bsidestep\w*\b"),
    ("turn_walk", r"\bturn(?:s|ing|ed)?\b|\bpivot(?:s|ing|ed)?\b|\bspin(?:s|ning|ned)?\b"),
    ("start_stop_transition", r"\bstart(?:s|ing|ed)?\b|\bstop(?:s|ping|ped)?\b|\btransition\w*\b"),
    ("fast_walk_run", r"\bjog(?:s|ging|ged)?\b|\brun(?:s|ning)?\b|\bsprint(?:s|ing|ed)?\b"),
    ("slow_walk", r"\bslow(?:ly)?\b"),
    ("forward_walk", r"\bwalk(?:s|ing|ed)?\b|\bforward\b|\bmarch(?:es|ing|ed)?\b|\bstep(?:s|ping|ped)?\b|\bstroll(?:s|ing)?\b"),
]
FAMILY_PRIORITY = {fam: i for i, (fam, _) in enumerate(FAMILY_RULES)}
COMPILED = [(fam, re.compile(pat)) for fam, pat in FAMILY_RULES]

# 方向修饰词（不改族，仅记录；B4-lite 首版前向为主，backward 段后续单独归置）
BACKWARD_RE = re.compile(r"\bbackwards?\b")

# content_type_of_movement 直接映射（描述字段 low 置信通道的一部分）
TOM_MAP = {
    "walking": "forward_walk",
    "jogging": "fast_walk_run",
    "dancing": "dance_rhythm",
    "jumping": "forward_jump",
    "turning": "turn_walk",
    "transition": "start_stop_transition",
}
# 描述字段（low 通道）规则 = 名称规则去掉 slow_walk：描述里 slowly/slow 常为
# 副词修饰（"slowly sneezes"）而非慢走族证据（D045 抽验修正）。
DESC_EXCLUDE_FAMS = {"slow_walk"}
DESC_RULES = [(f, re.compile(p)) for f, p in FAMILY_RULES if f not in DESC_EXCLUDE_FAMS]
DESC_FIELDS = [
    "content_type_of_movement",
    "content_short_description",
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
]


def norm_desc(filename: str) -> str | None:
    """`crossed_arms_idle_R_002__A549` -> `crossed_arms_idle_r`；解析失败返回 None。"""
    m = NAME_RE.match(filename)
    if m is None:
        return None
    return m.group("desc").lower()


def _word(text: str) -> str:
    return (text or "").lower().replace("_", " ")


def match_name(desc: str) -> tuple[list[str], list[str]]:
    """返回 (命各族列表按优先级序, 命中关键词列表)。"""
    words = _word(desc)
    hits, kws = [], []
    for fam, rx in COMPILED:
        found = rx.findall(words)
        if found:
            hits.append(fam)
            kws.extend(sorted(set(found)))
    return hits, kws


def match_desc_fields(row) -> tuple[list[str], str]:
    """metadata 描述字段匹配，返回 (命各族按优先级序, 来源字段)。"""
    hits: list[str] = []
    src = ""
    for field in DESC_FIELDS:
        val = row.get(field)
        if not isinstance(val, str) or not val.strip():
            continue
        words = _word(val)
        if field == "content_type_of_movement":
            fam = TOM_MAP.get(val.strip().lower())
            fam_hits = [fam] if fam else []
        else:
            fam_hits = [fam for fam, rx in DESC_RULES if rx.search(words)]
        if fam_hits:
            if not hits:
                src = field
            hits.extend(f for f in fam_hits if f not in hits)
    fam_sorted = sorted(set(hits), key=lambda f: FAMILY_PRIORITY.get(f, 99))
    return fam_sorted, src


def classify(row) -> dict:
    """单段分类：优先名称命中（high/med），否则描述字段（low），否则 other。"""
    desc = norm_desc(row["filename"])
    if desc is None:
        return {"norm_desc": None, "family": "other", "confidence": "other",
                "hits": [], "matched": [], "backward": False}
    hits, kws = match_name(desc)
    backward = bool(BACKWARD_RE.search(_word(desc)))
    if hits:
        fam = hits[0]
        conf = "high" if len(hits) == 1 else "med"
        return {"norm_desc": desc, "family": fam, "confidence": conf,
                "hits": hits, "matched": kws, "backward": backward}
    d_hits, src = match_desc_fields(row)
    backward = backward or bool(BACKWARD_RE.search(" ".join(
        _word(str(row.get(f))) for f in DESC_FIELDS if isinstance(row.get(f), str))))
    if d_hits:
        return {"norm_desc": desc, "family": d_hits[0], "confidence": "low",
                "hits": d_hits, "matched": [f"desc:{src}"], "backward": backward}
    return {"norm_desc": desc, "family": "other", "confidence": "other",
            "hits": [], "matched": [], "backward": backward}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    n_seg = len(df)
    print(f"[load] parquet rows={n_seg} cols={len(df.columns)}")

    seg_rows = []
    class_info: dict[str, dict] = {}
    for row in df.to_dict("records"):
        r = classify(row)
        seg_rows.append((row["filename"], r))
        desc = r["norm_desc"]
        if desc is None:
            continue
        ci = class_info.setdefault(desc, {
            "n_segments": 0,
            "fam_counter": Counter(), "conf_counter": Counter(),
            "matched": [], "hits": [],
        })
        ci["n_segments"] += 1
        ci["fam_counter"][r["family"]] += 1
        ci["conf_counter"][r["confidence"]] += 1
        if r["matched"] and not ci["matched"]:
            ci["matched"] = r["matched"]
            ci["hits"] = r["hits"]

    # 类级族 = 同归一类内的多数族（名称通道天然一致；仅 low/other 通道可能因
    # 描述字段差异分裂）；歧义类在条目里保留 fam_all 供抽验
    for desc, ci in class_info.items():
        fam, n_top = ci["fam_counter"].most_common(1)[0]
        ci["family"] = fam
        ci["confidence"] = ci["conf_counter"].most_common(1)[0][0]
        ci["fam_ambiguous"] = len(ci["fam_counter"]) > 1 and ci["fam_counter"][fam] != sum(ci["fam_counter"].values())

    fam_seg = Counter(r["family"] for _, r in seg_rows)
    conf_seg = Counter(r["confidence"] for _, r in seg_rows)
    fam_conf = Counter((r["family"], r["confidence"]) for _, r in seg_rows)
    n_backward = sum(1 for _, r in seg_rows if r["backward"])
    n_other = fam_seg.get("other", 0)
    covered = n_seg - n_other

    fam_stats = {}
    for fam, _ in FAMILY_RULES:
        n = fam_seg.get(fam, 0)
        fam_stats[fam] = {
            "segments": n,
            "share": round(n / n_seg, 4),
            "n_classes": sum(1 for ci in class_info.values() if ci["family"] == fam),
        }
    fam_stats["other"] = {
        "segments": n_other,
        "share": round(n_other / n_seg, 4),
        "n_classes": sum(1 for ci in class_info.values() if ci["family"] == "other"),
    }

    stats = {
        "_meta": {
            "script": "apt_g1/build_b4lite_map_labels.py",
            "experiment": "D045",
            "protocol": "refine-logs/DS_B4LITE_ACTION_SPLIT_PLAN.md §1.3/§6-S2",
            "family_list": "owner 裁定① 2026-09-07（10 族固定）",
            "parquet": args.parquet,
            "parquet_rows": n_seg,
            "temporal_labels": "缺失（两端无有效 HF token，D043 revoke）-> "
                               "fallback 协议 §7.2：整段粗标签过渡 + 标注置信度降级记录",
            "confidence_def": "high=名称单一族强命中; med=名称多族命中取更具体者; "
                              "low=仅 metadata 描述字段命中; other=无命中",
        },
        "total_segments": n_seg,
        "n_classes": len(class_info),
        "coverage": round(covered / n_seg, 4),
        "covered_segments": covered,
        "other_segments": n_other,
        "backward_modifier_segments": n_backward,
        "per_family": fam_stats,
        "confidence_dist_segments": dict(conf_seg),
        "family_x_confidence_segments": {f"{f}|{c}": n for (f, c), n in sorted(fam_conf.items())},
    }

    map_json = {
        "_meta": stats["_meta"],
        "classes": {d: {k: v for k, v in ci.items()
                        if k not in ("fam_counter", "conf_counter")}
                    for d, ci in sorted(class_info.items())},
    }

    map_path = os.path.join(args.out_dir, "desc_family_map.json")
    stats_path = os.path.join(args.out_dir, "map_stats.json")
    csv_path = os.path.join(args.out_dir, "desc_family_per_segment.csv")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(map_json, f, ensure_ascii=False, indent=1)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("filename,norm_desc,family,confidence,backward\n")
        for fname, r in seg_rows:
            f.write(f"{fname},{r['norm_desc']},{r['family']},{r['confidence']},{r['backward']}\n")

    print(f"[done] classes={len(class_info)} coverage={stats['coverage']} other={n_other}")
    for fam, s in fam_stats.items():
        print(f"  {fam:22s} segs={s['segments']:7d} share={s['share']:.4f} classes={s['n_classes']}")
    print(f"  confidence(segments): {dict(conf_seg)}")
    print(f"[write] {map_path}\n[write] {stats_path}\n[write] {csv_path}")


if __name__ == "__main__":
    main()
