"""D045 / B4-lite S6 reporting: per-family gate aggregation + publication
table (DS_B4LITE_ACTION_SPLIT_PLAN §2 公布纪律), from the FROZEN v2 manifest
(backfill_b4lite_manifest.py output) -- read-only joins, pure stdlib.

Inputs (all under --dir, g1_b4lite):
  manifest_v2.json          frozen manifest (147 stems, flat_replay backfilled,
                            split filled, npz meta isomorphic)
  split_assignments.json    S4 frozen split (excluded flags, family/actors)
  candidates_stats.json     S2/S3 pool + cleaning counters (candidate pool size)
  cleaning_ledger.json      L1/L2/L3 ledger (ruling exclusions included)
  gate_A/d045_gate_result_A.json + gate_B/d045_gate_result_B.json   raw gates
  gate_3seed/d045_gate_3seed_falls.json                             3-seed gate
  candidates.v2_prelabel.bak.json (optional --prev-candidates)
                            pre-fix labels -> diff stats for the lateral
                            locomotion rule fix (noise register)

Outputs:
  gate_summary_d045.json    per-family n/survived/rate/PASS-FAIL (>=95%, n>=10)
                            + T-seg/dev/T-fam/boundary buckets + 3-seed verdicts
  publication_table_d045.json  candidate pool, per-reason exclusion counts,
                            final 147 coverage (10 fam x n/actors/5-dim label
                            dist), test-split counts, repr_participation,
                            known-noise register

Usage (lab-ts, any python3):
  python3 apt_g1/build_b4lite_reports.py --dir .../g1_b4lite
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

HOME = os.path.expanduser("~")
DEFAULT_DIR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite"

FAMILIES = ["forward_walk", "fast_walk_run", "forward_jump", "dance_rhythm",
            "turn_walk", "start_stop_transition", "asym_upper", "posture_change",
            "lateral", "slow_walk"]
FAM_ROLE = {**{f: "train" for f in FAMILIES[:6]},
            **{f: "dev" for f in FAMILIES[6:8]},
            **{f: "test(T-fam)" for f in FAMILIES[8:]}}
LABEL_DIMS = ["locomotion", "upper_body", "posture", "contact", "temporal"]
PASS_RATE = 0.95
MIN_N = 10


def load(path):
    with open(path) as f:
        return json.load(f)


def three_seed_verdict(n_seeds, n_falls):
    if n_falls >= 2:
        return "systematic_fall"
    if n_falls == 1:
        return "seed_variance"
    return "clean"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--manifest", default="manifest_v2.json")
    ap.add_argument("--prev-candidates", default=None,
                    help="pre-label-fix candidates.json backup -> label diff stats")
    ap.add_argument("--summary-out", default="gate_summary_d045.json")
    ap.add_argument("--publication-out", default="publication_table_d045.json")
    args = ap.parse_args()
    d = args.dir

    manifest = [e for e in load(os.path.join(d, args.manifest)) if "error" not in e]
    split = load(os.path.join(d, "split_assignments.json"))["segments"]
    stats = load(os.path.join(d, "candidates_stats.json"))
    ledger = load(os.path.join(d, "cleaning_ledger.json"))
    gate_a = load(os.path.join(d, "gate_A", "d045_gate_result_A.json"))
    gate_b = load(os.path.join(d, "gate_B", "d045_gate_result_B.json"))
    seed3 = load(os.path.join(d, "gate_3seed", "d045_gate_3seed_falls.json"))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---------------- per-stem view from the frozen manifest ----------------
    stems = {}
    for e in manifest:
        b4 = e.get("b4lite") or {}
        fr = b4.get("flat_replay") or {}
        stems[e["stem"]] = {
            "family": (b4.get("semantics") or {}).get("family"),
            "actor": e.get("actor"),
            "t_role": (b4.get("split") or {}).get("t_role"),
            "set_role": (b4.get("split") or {}).get("set_role"),
            "survived": fr.get("survived"),
            "n_seeds": fr.get("n_seeds"),
            "n_falls": fr.get("n_falls"),
            "labels": (b4.get("semantics") or {}).get("labels_5dim") or {},
        }

    # 3-seed verdicts (gate_3seed per_segment, keyed stem__seedN)
    s3 = defaultdict(lambda: {"n_seeds": 0, "n_falls": 0})
    for key, r in seed3.get("per_segment", {}).items():
        stem = key.rpartition("__seed")[0]
        s3[stem]["n_seeds"] += 1
        s3[stem]["n_falls"] += 1 if r.get("fall_step") is not None else 0
    three_seed = {stem: {**v, "verdict": three_seed_verdict(**v)}
                  for stem, v in s3.items()}

    # ---------------- per-family aggregation ----------------
    # 口径（D044 先例，A005 计入 walk 类 24/25）：族 n = 该族配额段全集（含执行后
    # t_role 改 boundary_ref 的系统性 fall 段——它们仍是该族成员，不得摘除失败段）；
    # 但 S4 即 boundary_ref 的段（A005，props 未过 L2、从不在族配额）不入族存活率、
    # 仅单列 boundary。boundary 另作附加桶单列。族 n 加总 = 146，overall = 147（含 A005）。
    split_orig = {s["stem"]: s.get("t_role") for s in split}
    fams = {f: {"n": 0, "survived": 0, "actors": set(), "tseg": [0, 0],
                "boundary_members": []}
            for f in FAMILIES}
    buckets = {"T-seg": [0, 0], "dev": [0, 0], "T-fam": [0, 0], "boundary": [0, 0]}
    boundary_items = []
    for st, v in stems.items():
        f = v["family"]
        surv = 1 if v["survived"] else 0
        is_s4_boundary = split_orig.get(st) == "boundary_ref"
        if not is_s4_boundary and f in fams:
            fams[f]["n"] += 1
            fams[f]["survived"] += surv
            fams[f]["actors"].add(v["actor"])
            if v["t_role"] == "T-seg":
                fams[f]["tseg"][0] += 1
                fams[f]["tseg"][1] += surv
        if v["t_role"] == "boundary_ref":
            buckets["boundary"][0] += 1
            buckets["boundary"][1] += surv
            if not is_s4_boundary and f in fams:
                fams[f]["boundary_members"].append(st)
            boundary_items.append({
                "stem": st, "family": f, "actor": v["actor"],
                "n_seeds": v["n_seeds"], "n_falls": v["n_falls"],
                "s4_boundary_ref": is_s4_boundary,
                "verdict": three_seed.get(st, {}).get("verdict",
                            three_seed_verdict(v["n_seeds"] or 0, v["n_falls"] or 0)),
            })
        elif v["t_role"] in buckets:
            buckets[v["t_role"]][0] += 1
            buckets[v["t_role"]][1] += surv

    def rate_block(n, surv):
        rate = round(surv / n, 4) if n else None
        return {"n": n, "survived": surv, "rate": rate,
                "pass": bool(n >= MIN_N and rate is not None
                             and rate >= PASS_RATE)}

    per_family = {}
    for f in FAMILIES:
        v = fams[f]
        n_excl_b = v["n"] - len(v["boundary_members"])
        s_excl_b = v["survived"] - sum(1 for st in v["boundary_members"]
                                       if stems[st]["survived"])
        blk = {"role": FAM_ROLE[f], **rate_block(v["n"], v["survived"]),
               "n_actors": len(v["actors"]),
               "boundary_ref_members": sorted(v["boundary_members"]),
               "rate_excl_boundary_ref": (rate_block(n_excl_b, s_excl_b)
                                          if len(v["boundary_members"]) else None),
               "tseg": {"n": v["tseg"][0], "survived": v["tseg"][1],
                        "rate": round(v["tseg"][1] / v["tseg"][0], 4)
                        if v["tseg"][0] else None},
               "three_seed": {st: three_seed[st] for st in sorted(three_seed)
                              if stems.get(st, {}).get("family") == f}}
        per_family[f] = blk

    summary = {
        "_meta": {
            "script": "apt_g1/build_b4lite_reports.py",
            "experiment": "D045",
            "generated_at": now,
            "manifest": args.manifest,
            "criteria": f"族级存活 >= {int(PASS_RATE*100)}% PASS（n>={MIN_N} 口径）；"
                        "族 n = 配额段全集（执行失败段保留族内，D044 先例）；S4 即 "
                        "boundary_ref 的 A005 非配额段不入族存活率、仅单列 boundary"
                        "（族 n 加总=146，overall n=147 含 A005）；排除 5 段（owner 裁定）"
                        "不在冻结 manifest，自动不计",
            "three_seed_criterion": "3 seed 中 >=2 fall=系统性(boundary)；1/3=seed-variance"
                                    "（quality_notes）；双门 seed0 + 3-seed 合并回填",
        },
        "overall": {
            **rate_block(len(stems),
                         sum(1 for v in stems.values() if v["survived"])),
            "n_three_seed_stems": len(three_seed),
            "three_seed_verdicts": dict(Counter(v["verdict"] for v in three_seed.values())),
        },
        "per_family": per_family,
        "buckets": {k: ({"n": v[0], "survived": v[1],
                         "rate": round(v[1] / v[0], 4) if v[0] else None}
                        if k != "T-fam" else
                        {"n": v[0], "survived": v[1],
                         "rate": round(v[1] / v[0], 4) if v[0] else None,
                         "per_family": {f: {"n": per_family[f]["n"],
                                            "survived": per_family[f]["survived"],
                                            "rate": per_family[f]["rate"],
                                            "pass": per_family[f]["pass"]}
                                        for f in FAMILIES[8:]}})
                    for k, v in buckets.items()},
        "boundary_detail": boundary_items,
        "pass_all_families": all(b["pass"] for b in per_family.values()),
    }

    # ---------------- publication table (协议 §2 公布纪律) ----------------
    pool = stats.get("candidates_10fam_total")
    clean = stats.get("cleaning", {})
    l2_primary = dict(clean.get("L2_by_primary_reason") or {})
    ledger_l2_primary = dict((ledger.get("summary") or {}).get("L2_by_primary_reason")
                             or {})
    ruling = stats.get("excluded_ruling_L2_label_mismatch") or {}

    coverage = {}
    for f in FAMILIES:
        recs = [v for v in stems.values() if v["family"] == f]
        coverage[f] = {
            "n": len(recs),
            "n_actors": len({v["actor"] for v in recs}),
            "n_boundary_ref": sum(1 for v in recs if v["t_role"] == "boundary_ref"),
            "labels_dist": {dim: dict(Counter(v["labels"].get(dim) or "缺失"
                                              for v in recs))
                            for dim in LABEL_DIMS},
        }

    split_counts = Counter(v["t_role"] for v in stems.values())
    sys_moved = [st for st, v in stems.items()
                 if v["t_role"] == "boundary_ref" and st != 
                 "walk_forward_grab_injured_L_leg_002__A005"]

    # label-rule fix diff (lateral locomotion 前进/混合 -> 侧移)
    label_diff_stems = []
    if args.prev_candidates and os.path.isfile(args.prev_candidates):
        prev = {c["stem"]: c for c in
                load(args.prev_candidates)["segments"]}
        for st, v in stems.items():
            old = (prev.get(st) or {}).get("labels", {}).get("movement")
            new = v["labels"].get("locomotion")
            if old is not None and old != new:
                label_diff_stems.append({"stem": st, "family": v["family"],
                                         "movement_before": old, "movement_after": new})

    excl_stems = sorted(ruling.get("stems") or
                        [e["stem"] for e in ledger.get("L2", [])
                         if e.get("reason") == "L2_label_mismatch"])

    publication = {
        "_meta": {
            "script": "apt_g1/build_b4lite_reports.py",
            "experiment": "D045",
            "generated_at": now,
            "protocol": "DS_B4LITE_ACTION_SPLIT_PLAN §2 公布纪律",
            "manifest_frozen": args.manifest,
        },
        "candidate_pool": {
            "n_mapped_10fam_segments": pool,
            "source": "seed_metadata_v004.parquet x desc_family_per_segment.csv"
                      "（L1a 10 语义族映射候选）",
        },
        "exclusions_by_layer": {
            "L2_hard_filters_by_primary_reason": l2_primary,
            "L2_hard_filters_incl_mirror_dedup": {
                **ledger_l2_primary,
                "mirror_dedup": (ledger.get("summary") or {}).get("L2_mirror_dedup"),
            },
            "L2_label_mismatch_ruling": {"n": ruling.get("n"), "stems": excl_stems,
                                          "note": "owner 裁定 2026-09-07，渲染复核证据留档"},
            "L1_conversion_failures": len(ledger.get("L1") or []),
            "L3_boundary_records": len(boundary_items),
        },
        "final_dataset": {
            "n_segments": len(stems),
            "note": "覆盖表=147 段全集按 family 归族（含 boundary_ref 段，"
                    "n_boundary_ref 标注），族 n 加总=147；存活率口径见 gate_summary",
            "coverage_by_family": coverage,
        },
        "test_splits": {
            "counts_frozen": {k: split_counts.get(k, 0)
                              for k in ["train", "T-seg", "dev", "T-fam", "boundary_ref"]},
            "note": "冻结口径=排除 5 段后 + 系统性 fall 段 t_role 改 boundary_ref"
                    + (f"；本次移入 boundary 的原 S4 段: {sys_moved}" if sys_moved else
                       "（本次无 S4 段被移入 boundary）"),
        },
        "repr_participation": {
            "vae": "适配训练(以冻结 manifest 为准)",
            "norm_stats": "B4-lite 样本内标定",
            "sonic_pretrain": "not_excludable（不可由本次划分排除）",
            "scope": "全部 147 段统一声明（backfill --split 写入，npz meta 同构）",
        },
        "known_noise": {
            "label_rule_fix_lateral": {
                "issue": "lateral 族 locomotion 曾被 filename 通道误标「前进/混合」"
                         "（jog_sideway 词根命中 FWD_RE）",
                "fix": "lateral 族 locomotion 恒「侧移」（build_b4lite_candidates.py "
                       "S5 渲染复核覆写，owner 裁定 2026-09-07）",
                "n_stems_changed": len(label_diff_stems),
                "stems_changed": label_diff_stems,
            },
            "a005_tpose_frames": "首尾各 ~2 帧 T-pose 校准帧（原始数据自带）；"
                                 "quality_notes 登记不做数据处理",
            "render_review_exclusions": [
                {"stem": st,
                 "reason": next((e.get("detail", {}).get("note") or e.get("detail")
                                 for e in ledger.get("L2", [])
                                 if e.get("stem") == st), None)}
                for st in excl_stems],
            "render_review_doubt_stems": excl_stems,
            "render_review_doubt_note": "渲染复看存疑 4 组共 5 段（A020 / A465 / "
                                        "crawl 三连窗 A125-A127），全部排除留档",
        },
        "gate_numbers": {
            "gate_A_raw": {"n_segments": gate_a.get("n_segments"),
                           "n_falls": sum(1 for r in gate_a.get("per_segment", {}).values()
                                          if not r.get("completed")),
                           "note": "含后续裁定排除段（A020/crawl 三连窗），证据原档"},
            "gate_B_raw": {"n_segments": gate_b.get("n_segments"),
                           "n_falls": sum(1 for r in gate_b.get("per_segment", {}).values()
                                          if not r.get("completed")),
                           "note": "含后续裁定排除段 A465，证据原档"},
            "three_seed": {"n_segments": seed3.get("n_segments"),
                           "seeds": seed3.get("seeds"),
                           "per_stem": three_seed},
        },
    }

    for name, obj in ((args.summary_out, summary),
                      (args.publication_out, publication)):
        p = os.path.join(d, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        print(f"[write] {p}")

    print(f"[summary] overall {summary['overall']['survived']}/{summary['overall']['n']}"
          f" = {summary['overall']['rate']}")
    for f in FAMILIES:
        b = per_family[f]
        print(f"  {f:22s} {b['survived']:3d}/{b['n']:3d} = {b['rate']} "
              f"{'PASS' if b['pass'] else 'FAIL'}  3seed={len(b['three_seed'])}")
    for k, v in buckets.items():
        print(f"  [{k:8s}] {v[1]}/{v[0]}"
              + (f" = {round(v[1]/v[0],4)}" if v[0] else ""))
    print(f"  boundary: {[(b['stem'], b['verdict']) for b in boundary_items]}")
    print(f"  pass_all_families: {summary['pass_all_families']}")


if __name__ == "__main__":
    main()
