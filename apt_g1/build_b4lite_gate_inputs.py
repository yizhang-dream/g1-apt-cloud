"""D045 / B4-lite protocol §6-S5 (DS_B4LITE_ACTION_SPLIT_PLAN): build the
conversion + replay inputs from the frozen S4 split -- three modes, one
script, pure stdlib (any python >= 3.9).

  bridge      split_assignments.json + candidates.json
              -> labels_split_bridge.json (CB --labels-json format:
                 {stem: {family / labels_5dim(5键) / label_confidence /
                 label_sources(filename,trajectory,manual)}})
              + csv_list.txt (one absolute CSV path per line -> CB --list)
  gate-split  manifest.json (CB output) + split_assignments.json
              -> gate_A/ + gate_B/ (npz copies + subset manifests with
                 rewritten npz_path), so TWO b3p_gate_isaac.py processes can
                 run in parallel WITHOUT touching the gate code (t003
                 double-launch防线: separate dirs, separate outputs).
                 A = t_role in {train, T-seg};  B = {dev, T-fam, boundary_ref}
  ledger-l1   manifest.json + cleaning_ledger.json -> append L1 entries
              (格式错=S5 转换时回填, ledger layer_rule) for conversion-failed
              stems; failures never block the batch (protocol §2: record,
              don't clean).

Key-name reconciliation (candidates vs CB b4lite semantics template):
  labels["movement"]      -> labels_5dim["locomotion"]   (其余 4 键同名直传)
  label_sources["metadata"] -> label_sources["trajectory"]
  (协议 §5「文件名/轨迹/人工」三级来源；S2 的轨迹级由官方 parquet metadata
   通道实现，故映射而非丢弃)

Usage (lab-ts, any python3):
  python3 apt_g1/build_b4lite_gate_inputs.py bridge   [--dir ...]
  python3 apt_g1/build_b4lite_gate_inputs.py gate-split [--dir ...]
  python3 apt_g1/build_b4lite_gate_inputs.py ledger-l1 [--dir ...]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

HOME = os.path.expanduser("~")
DEFAULT_DIR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b4lite"
DEFAULT_DS_BONES = f"{HOME}/ros2_data/apt_g1/data/ds_bones"

LABEL_KEY_MAP = {"movement": "locomotion"}  # 其余键 CB 模板同名
LABEL_5DIM_KEYS = ["locomotion", "upper_body", "posture", "contact", "temporal"]
SOURCE_KEY_MAP = {"metadata": "trajectory"}  # 其余键同名直传
SOURCE_KEYS = ["filename", "trajectory", "manual"]

GATE_ROLE_A = {"train", "T-seg"}  # 训练 6 族：train + T-seg
GATE_ROLE_B = {"dev", "T-fam", "boundary_ref"}  # 开发 2 族 + T-fam 2 族 + A005


def _load_split_and_candidates(d):
    split = json.load(open(os.path.join(d, "split_assignments.json")))["segments"]
    cand = {c["stem"]: c for c in
            json.load(open(os.path.join(d, "candidates.json")))["segments"]}
    missing = [s["stem"] for s in split if s["stem"] not in cand]
    if missing:
        sys.exit(f"[bridge] split stems absent from candidates: {missing[:5]}")
    return split, cand


# ------------------------------------------------------------------ bridge
def mode_bridge(args):
    split, cand = _load_split_and_candidates(args.dir)
    bridge, csv_lines, bad = {}, [], []
    for s in split:
        stem = s["stem"]
        c = cand[stem]
        labels_5dim = {LABEL_KEY_MAP.get(k, k): v for k, v in c["labels"].items()}
        sources = {SOURCE_KEY_MAP.get(k, k): bool(v)
                   for k, v in c["label_sources"].items()}
        if sorted(labels_5dim) != sorted(LABEL_5DIM_KEYS):
            bad.append(f"{stem}: labels keys {sorted(labels_5dim)}")
        if sorted(sources) != sorted(SOURCE_KEYS):
            bad.append(f"{stem}: sources keys {sorted(sources)}")
        bridge[stem] = {
            "family": c["family"],
            "labels_5dim": labels_5dim,
            "label_confidence": c["confidence"],
            "label_sources": sources,
        }
        csv_abs = os.path.join(args.ds_bones, c["move_g1_path"])
        if not os.path.isfile(csv_abs):
            bad.append(f"{stem}: csv missing {csv_abs}")
        csv_lines.append(csv_abs)
    if bad:
        sys.exit("[bridge] consistency failures (abort, fix first):\n  " + "\n  ".join(bad[:10]))

    out_bridge = os.path.join(args.dir, "labels_split_bridge.json")
    out_list = os.path.join(args.dir, "csv_list.txt")
    with open(out_bridge, "w") as f:
        json.dump(bridge, f, indent=1, ensure_ascii=False)
    with open(out_list, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"[bridge] {len(bridge)} stems -> {out_bridge}")
    print(f"[bridge] {len(csv_lines)} csv paths -> {out_list}")


# --------------------------------------------------------------- gate-split
def mode_gate_split(args):
    split = json.load(open(os.path.join(args.dir, "split_assignments.json")))["segments"]
    t_role = {s["stem"]: s["t_role"] for s in split}
    manifest = json.load(open(os.path.join(args.dir, "manifest.json")))
    ok = [e for e in manifest if "error" not in e]
    failed = [e for e in manifest if "error" in e]
    if failed:
        print(f"[gate-split] WARNING {len(failed)} conversion failures excluded "
              f"(run ledger-l1): {[e['stem'] for e in failed][:5]}")

    parts = {"A": [], "B": []}
    unassigned = []
    for e in ok:
        role = t_role.get(e["stem"])
        if role in GATE_ROLE_A:
            parts["A"].append(e)
        elif role in GATE_ROLE_B:
            parts["B"].append(e)
        else:
            unassigned.append((e["stem"], role))
    if unassigned:
        sys.exit(f"[gate-split] stems with unknown t_role: {unassigned[:5]}")

    for name, entries in parts.items():
        gdir = os.path.join(args.dir, f"gate_{name}")
        npz_dir = os.path.join(gdir, "npz")
        os.makedirs(npz_dir, exist_ok=True)
        out_entries = []
        for e in entries:
            dst = os.path.join(npz_dir, e["stem"] + ".npz")
            if not os.path.isfile(dst):
                shutil.copy2(e["npz_path"], dst)
            e2 = dict(e)
            e2["npz_path"] = os.path.abspath(dst)
            e2["gate_partition"] = f"d045_gate_{name}"
            out_entries.append(e2)
        with open(os.path.join(gdir, "manifest.json"), "w") as f:
            json.dump(out_entries, f, indent=1, ensure_ascii=False)
        print(f"[gate-split] gate_{name}: {len(out_entries)} segments, "
              f"t_roles={sorted({t_role[e['stem']] for e in entries})} -> {gdir}")


# ---------------------------------------------------------------- ledger-l1
def mode_ledger_l1(args):
    split = json.load(open(os.path.join(args.dir, "split_assignments.json")))["segments"]
    family = {s["stem"]: s["family"] for s in split}
    lpath = os.path.join(args.dir, "cleaning_ledger.json")
    ledger = json.load(open(lpath))
    manifest = json.load(open(os.path.join(args.dir, "manifest.json")))
    have = {e.get("stem") for e in ledger["L1"]}
    added = 0
    for e in manifest:
        if "error" not in e or e["stem"] in have:
            continue
        err = e["error"]
        ledger["L1"].append({
            "stem": e["stem"], "family": family.get(e["stem"]),
            "layer": "L1", "reason": "conversion_format_error",
            "all_reasons": [err.split(":")[0]],
            "detail": {"error": err, "csv": e.get("csv"),
                       "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        })
        added += 1
    ledger["summary"]["L1"] = len(ledger["L1"])
    with open(lpath, "w") as f:
        json.dump(ledger, f, indent=1, ensure_ascii=False)
    print(f"[ledger-l1] added {added} L1 entries (total {len(ledger['L1'])}) -> {lpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["bridge", "gate-split", "ledger-l1"])
    ap.add_argument("--dir", default=DEFAULT_DIR, help="g1_b4lite dir")
    ap.add_argument("--ds-bones", default=DEFAULT_DS_BONES,
                    help="ds_bones root (csv move_g1_path resolution)")
    args = ap.parse_args()
    {"bridge": mode_bridge, "gate-split": mode_gate_split,
     "ledger-l1": mode_ledger_l1}[args.mode](args)


if __name__ == "__main__":
    main()
