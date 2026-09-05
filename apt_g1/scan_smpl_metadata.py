"""DS plan M1: metadata scan over the full bones_seed_smpl mirror (131,455 pkls).

Per-file stats -> CSV; aggregate class/actor/direction/speed tables -> JSON.
This is the data-analysis workhorse for the official-data mainline
(DS_OFFICIAL_DATA_PLAN §3.1 M1): class inventory & duration, speed histograms,
direction skew quantification, actor coverage, mirror-duplicate accounting,
run-class scarcity check (758 files at filename-classification).

Filename convention (verified on server): ``<Class_Name>_<NNN>__A<actor>[_M].pkl``
- ``_M`` suffix = left-right mirrored duplicate of the same take.
- class guess = ``__`` prefix with trailing ``_<NNN>`` index stripped.

Resumable: rows are appended; already-scanned filenames are skipped on restart.

Usage (server, any venv with numpy; joblib auto-vendored):
    cd ~/ros2_data/apt_g1 && nohup ~/ros2_data/.venv_mjlab/bin/python \
        scan_smpl_metadata.py > data/ds_bones/m1/scan.log 2>&1 < /dev/null & disown
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import re
import sys
import zlib
from multiprocessing import Pool

import numpy as np

HOME = os.path.expanduser("~")
DEFAULT_MIRROR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/b1_smpl/smpl_filtered"
DEFAULT_OUT_DIR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/m1"
VENDOR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/b2/_vendor"
NAME_RE = re.compile(r"^(?P<cls>.+?)_(?P<take>\d+)__A(?P<actor>\d+)(?P<mir>_M)?\.pkl$")


def _load_pkl(path):
    raw = open(path, "rb").read()
    data = zlib.decompress(raw) if raw[:1] == b"\x78" else raw
    sys.path.insert(0, VENDOR)  # vendored joblib (absent from server venvs)
    import joblib

    return joblib.load(io.BytesIO(data))


def scan_one(path):
    """Worker: one pkl -> one metadata row (never raises)."""
    row = {"file": os.path.basename(path), "error": ""}
    try:
        m = NAME_RE.match(row["file"])
        if m:
            row["class_guess"] = m.group("cls")
            row["actor"] = int(m.group("actor"))
            row["is_mirror_dup"] = bool(m.group("mir"))
        else:
            row["class_guess"] = row["file"].split("__")[0]
            row["actor"] = -1
            row["is_mirror_dup"] = "_M.pkl" in row["file"]

        obj = _load_pkl(path)
        v = obj
        if isinstance(obj, dict) and len(obj) == 1:
            v = next(iter(obj.values()))
        fps = float(v.get("fps", 50.0))
        tr = np.asarray(v["transl"], dtype=np.float64)
        j = np.asarray(v["smpl_joints"], dtype=np.float64)
        pa = np.asarray(v["pose_aa"], dtype=np.float64).reshape(len(tr), 24, 3)
        n = len(tr)
        row["n_frames"] = n
        row["fps"] = fps
        row["duration_s"] = n / fps

        d = np.diff(tr, axis=0)
        seg = np.linalg.norm(d, axis=1)
        row["path_len_m"] = float(seg.sum())
        row["mean_speed"] = float(seg.mean() * fps)
        row["p95_speed"] = float(np.percentile(seg, 95) * fps)
        row["net_disp_m"] = float(np.linalg.norm(tr[-1] - tr[0]))
        std = tr.std(axis=0)
        row["up_axis"] = int(np.argmin(std))  # least-spread axis = vertical
        row["horiz_span_m"] = float(np.linalg.norm(np.delete(std, row["up_axis"])))

        # heading turn from transl (when actually moving)
        mov = seg > 0.15 / fps
        if mov.sum() > 10:
            steps = d[mov]
            hd = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
            row["transl_turn_deg"] = float(np.degrees(np.abs(np.diff(hd)).sum()))
        else:
            row["transl_turn_deg"] = 0.0

        # root orientation yaw (axis-angle -> quat wxyz -> yaw)
        aa = pa[:, 0, :]
        th = np.linalg.norm(aa, axis=1)
        cos = np.cos(th / 2)
        s = np.sin(th / 2)
        axis = np.divide(aa, np.where(th[:, None] < 1e-12, 1.0, th)[:, None])
        qw = cos
        qv = axis * s[:, None]
        yaw = np.degrees(np.arctan2(2 * (qw * qv[:, 2] + qv[:, 0] * qv[:, 1]), 1 - 2 * (qv[:, 1] ** 2 + qv[:, 2] ** 2)))
        row["root_yaw_start_deg"] = float(yaw[0])
        row["root_yaw_end_deg"] = float(yaw[-1])
        dyaw = np.diff(np.unwrap(np.radians(yaw)))
        row["root_turn_deg"] = float(np.degrees(np.abs(dyaw).sum()))

        # body scale proxy: pelvis-relative joint bbox diagonal (frame mean)
        rel = j - j[:, 0:1, :]
        ext = rel.reshape(-1, 3)
        row["body_span_m"] = float(np.linalg.norm(ext.max(axis=0) - ext.min(axis=0)))
    except Exception as e:  # noqa: BLE001 - record, never block the scan
        row["error"] = f"{type(e).__name__}: {e}"
    return row


COLUMNS = [
    "file", "class_guess", "actor", "is_mirror_dup", "n_frames", "fps",
    "duration_s", "path_len_m", "mean_speed", "p95_speed", "net_disp_m",
    "up_axis", "horiz_span_m", "transl_turn_deg", "root_yaw_start_deg",
    "root_yaw_end_deg", "root_turn_deg", "body_span_m", "error",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", default=DEFAULT_MIRROR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap file count")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "metadata_rows.csv")
    files = sorted(glob.glob(os.path.join(args.mirror, "*.pkl")))
    if args.limit:
        files = files[: args.limit]

    done = set()
    if os.path.exists(csv_path):  # resume
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                done.add(r["file"])
        files = [p for p in files if os.path.basename(p) not in done]
        print(f"[resume] {len(done)} rows already present, {len(files)} left", flush=True)

    new_file = not os.path.exists(csv_path)
    if new_file:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    t0 = __import__("time").time()
    n_err = 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        with Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(scan_one, files, chunksize=32)):
                w.writerow({k: row.get(k, "") for k in COLUMNS})
                if row["error"]:
                    n_err += 1
                if (i + 1) % 5000 == 0:
                    f.flush()
                    dt = __import__("time").time() - t0
                    print(f"[scan] {i + 1}/{len(files)} rows "
                          f"({(i + 1) / dt:.0f} files/s, errors {n_err})", flush=True)
    dt = __import__("time").time() - t0
    print(f"[scan] done: {len(files)} files in {dt:.0f}s, errors {n_err}", flush=True)

    # ---- aggregate summary from the full CSV (fresh + resumed rows)
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    ok = [r for r in rows if not r["error"]]
    print(f"[summary] total rows {len(rows)}, ok {len(ok)}", flush=True)

    def fnum(r, k, default=0.0):
        try:
            return float(r[k])
        except (KeyError, ValueError, TypeError):
            return default

    classes = {}
    for r in ok:
        c = classes.setdefault(r["class_guess"], {"n": 0, "n_unique": 0, "dur": 0.0,
                                                  "speeds": [], "turns": [], "actors": set()})
        c["n"] += 1
        if r["is_mirror_dup"] != "True":
            c["n_unique"] += 1
        c["dur"] += fnum(r, "duration_s")
        if fnum(r, "mean_speed") > 0.1:
            c["speeds"].append(fnum(r, "mean_speed"))
            c["turns"].append(fnum(r, "transl_turn_deg"))
        c["actors"].add(r["actor"])

    table = []
    for name, c in classes.items():
        sp = np.array(c["speeds"]) if c["speeds"] else np.array([0.0])
        tn = np.array(c["turns"]) if c["turns"] else np.array([0.0])
        table.append({
            "class": name, "n_files": c["n"], "n_unique_takes": c["n_unique"],
            "total_hours": round(c["dur"] / 3600, 2),
            "n_actors": len(c["actors"] - {-1}),
            "moving_n": len(c["speeds"]),
            "speed_median_ms": round(float(np.median(sp)), 3),
            "speed_p10_p90": [round(float(np.percentile(sp, 10)), 3), round(float(np.percentile(sp, 90)), 3)],
            "turn_deg_median": round(float(np.median(tn)), 1),
            "turn_deg_p90": round(float(np.percentile(tn, 90)), 1),
        })
    table.sort(key=lambda x: -x["total_hours"])

    summary = {
        "mirror": args.mirror,
        "n_files": len(rows),
        "n_ok": len(ok),
        "n_errors": len(rows) - len(ok),
        "n_classes": len(table),
        "n_actors": len({r["actor"] for r in ok if r["actor"] not in ("-1", -1, "")}),
        "total_hours": round(sum(x["total_hours"] for x in table), 1),
        "up_axis_hist": {},
        "class_table_top40": table[:40],
    }
    up_hist: dict[str, int] = {}
    for r in ok:
        up_hist[str(r["up_axis"])] = up_hist.get(str(r["up_axis"]), 0) + 1
    summary["up_axis_hist"] = up_hist

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("[summary] JSON -> summary.json", flush=True)
    for x in table[:15]:
        print(f"  {x['class']:<28} n={x['n_files']:>6} uniq={x['n_unique_takes']:>6} "
              f"{x['total_hours']:>7.2f}h v={x['speed_median_ms']:.2f} turn={x['turn_deg_median']:.0f}", flush=True)


if __name__ == "__main__":
    main()
