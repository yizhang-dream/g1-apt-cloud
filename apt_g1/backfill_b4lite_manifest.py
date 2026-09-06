"""D045 / B4-lite protocol §5 (refine-logs/DS_B4LITE_ACTION_SPLIT_PLAN.md):
join the B3' gate results back into the conversion manifest, freeze it as
v2, and emit the capability-boundary set.

Two inputs, one join:
  - manifest.json   from convert_bones_g1_csv.py (entries carry the nested
    "b4lite" block, D045; the npz-embedded meta stays the fact source,
    D044 pitfall #1)
  - gate_result.json from isaac/b3p_gate_isaac.py, per_segment keyed
    "{stem}__seed{seed}"

Per stem, aggregated over seeds:
  b4lite.flat_replay = {
    survived                all-seed completed AND,
    n_seeds,
    fall_steps              per-seed list (seed order),
    fall_during_playback    any seed,
    steps_budget            gate-level budget (fallback: max total_steps),
    q_track_mae_med         median over seeds with data,
    realized_path_ratio_med median over seeds with data,
    fall_form               None -- qualitative field, filled manually later }

L3 ruling (protocol §2: execution failure = capability-boundary record,
NEVER cleaned away): survived=False -> quality.layer3="recorded",
quality.boundary_set=True, plus split.t_role prefilled "boundary_ref"
(S4 may still revise it) and a quality_notes audit line. Stems absent from
the gate keep flat_replay=None -- never fabricated. Legacy manifest entries
without a b4lite block get the default block created (warned).

boundary_set.json (owner ruling ③ dual-track, schema independently designed
in the per-scene-per-seed spirit of apt_g1/stress_isolate.json):
  { generated_at, criteria: "B3' oracle replay, any-seed fall 或未完成",
    items: [{stem, actor, family, fall_steps, q_track_mae_med,
             evidence_note}] }

Pure stdlib (any python >= 3.9; no numpy needed). Usage (lab-ts):
  python apt_g1/backfill_b4lite_manifest.py \
      --manifest data/ds_bones/g1_b4lite/manifest.json \
      --gate data/ds_bones/g1_b4lite/gate_result.json \
      --out data/ds_bones/g1_b4lite/manifest_v2.json \
      --boundary-out data/ds_bones/g1_b4lite/boundary_set.json
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from statistics import median

DEFAULT_DIR = "/home/cvgluser/ros2_data/apt_g1/data/ds_bones/g1_b4lite"

BOUNDARY_CRITERIA = "B3' oracle replay, any-seed fall 或未完成"


def default_b4lite_block():
    """Same defaults as convert_bones_g1_csv.build_b4lite_block (duplicated
    on purpose: this script stays stdlib-only and must not import the
    converter chain). Used only for legacy entries missing the block."""
    return {
        "semantics": {"family": None},
        "quality": {"layer1": "pass", "layer2": "pass", "layer3": "none",
                    "boundary_set": False, "quality_notes": []},
        "split": {"set_role": None, "t_role": None,
                  "repr_participation": {"vae": None, "norm_stats": None,
                                         "sonic_pretrain": "not_excludable"}},
        "flat_replay": None,
    }


def split_gate_key(key):
    """'{stem}__seed{seed}' -> (stem, seed). Stems themselves contain '__'
    (e.g. ...__A001), so split on the LAST '__seed' occurrence only."""
    stem, sep, seed = key.rpartition("__seed")
    if not sep:
        raise ValueError(f"bad per_segment key (expected __seed suffix): {key}")
    return stem, int(seed)


def backfill(manifest, gate):
    """Mutates+returns the manifest entries with flat_replay / L3 fields."""
    per = gate.get("per_segment", {})
    budget = gate.get("steps_budget")
    by_stem = {}
    for key, r in per.items():
        stem, seed = split_gate_key(key)
        by_stem.setdefault(stem, {})[seed] = r

    for e in manifest:
        if "error" in e:
            continue
        if "b4lite" not in e:
            print(f"[warn] {e.get('stem')}: no b4lite block (legacy manifest) "
                  "-> default block created")
            e["b4lite"] = default_b4lite_block()
        seeds = by_stem.get(e.get("stem"))
        if not seeds:
            continue  # 没跑门/无 gate 条目：flat_replay 保持 null，不伪造
        ordered = [seeds[s] for s in sorted(seeds)]
        fall_steps = [r.get("fall_step") for r in ordered]
        survived = all(bool(r.get("completed")) for r in ordered)
        q_mae = [r["q_track_mae_vs_ref_rad"] for r in ordered
                 if r.get("q_track_mae_vs_ref_rad") is not None]
        rpr = [r["realized_path_ratio"] for r in ordered
               if r.get("realized_path_ratio") is not None]
        e["b4lite"]["flat_replay"] = {
            "survived": survived,
            "n_seeds": len(ordered),
            "fall_steps": fall_steps,
            "fall_during_playback": any(bool(r.get("fall_during_playback"))
                                        for r in ordered),
            "steps_budget": budget if budget is not None else max(
                (r.get("total_steps") or 0) for r in ordered),
            "q_track_mae_med": round(median(q_mae), 4) if q_mae else None,
            "realized_path_ratio_med": round(median(rpr), 3) if rpr else None,
            "fall_form": None,  # 定性字段：摔倒形态留人工/后续填
        }
        if not survived:
            # L3：执行失败 = 能力边界记录，不清洗（协议 §2）
            b4 = e["b4lite"]
            b4["quality"]["layer3"] = "recorded"
            b4["quality"]["boundary_set"] = True
            b4["quality"]["quality_notes"].append(
                "L3 recorded by backfill: B3' oracle replay failure "
                f"(n_seeds={len(ordered)}, fall_steps={fall_steps})")
            b4["split"]["t_role"] = "boundary_ref"  # 预填候选，S4 可改

    manifest_stems = {e.get("stem") for e in manifest if "error" not in e}
    for stem in sorted(by_stem):
        if stem not in manifest_stems:
            print(f"[warn] gate stem not present in manifest: {stem}")
    return manifest


def build_boundary_set(manifest):
    items = []
    for e in manifest:
        b4 = e.get("b4lite") or {}
        fr = b4.get("flat_replay")
        if not fr or fr.get("survived", True):
            continue  # 无 gate 条目或全存活：不入边界集
        items.append({
            "stem": e.get("stem"),
            "actor": e.get("actor"),
            "family": (b4.get("semantics") or {}).get("family"),
            "fall_steps": fr.get("fall_steps"),
            "q_track_mae_med": fr.get("q_track_mae_med"),
            "evidence_note": (
                f"B3' oracle replay: {fr.get('n_seeds')} seed(s), "
                f"fall_steps={fr.get('fall_steps')}, "
                f"fall_during_playback={fr.get('fall_during_playback')}, "
                f"realized_path_ratio_med={fr.get('realized_path_ratio_med')}, "
                f"steps_budget={fr.get('steps_budget')}"),
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criteria": BOUNDARY_CRITERIA,
        "items": items,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=f"{DEFAULT_DIR}/manifest.json",
                    help="manifest.json from convert_bones_g1_csv.py (D045)")
    ap.add_argument("--gate", default=f"{DEFAULT_DIR}/gate_result.json",
                    help="gate_result.json from isaac/b3p_gate_isaac.py")
    ap.add_argument("--out", default=f"{DEFAULT_DIR}/manifest_v2.json",
                    help="frozen v2 manifest path")
    ap.add_argument("--boundary-out",
                    default=f"{DEFAULT_DIR}/boundary_set.json",
                    help="capability-boundary set path")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.gate) as f:
        gate = json.load(f)

    manifest = backfill(manifest, gate)
    boundary = build_boundary_set(manifest)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=1)
    with open(args.boundary_out, "w") as f:
        json.dump(boundary, f, indent=1)

    ok = [e for e in manifest if "error" not in e]
    n_fr = sum(1 for e in ok if e.get("b4lite", {}).get("flat_replay"))
    n_bd = sum(1 for e in ok
               if e.get("b4lite", {}).get("quality", {}).get("boundary_set"))
    print(f"[backfill] {len(ok)} segments | flat_replay filled {n_fr} | "
          f"boundary_set {n_bd} | items {len(boundary['items'])}")
    print(f"saved {args.out}")
    print(f"saved {args.boundary_out}")


if __name__ == "__main__":
    main()
