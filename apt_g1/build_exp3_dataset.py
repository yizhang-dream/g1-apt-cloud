"""Build a merged dataset from exp1_raw + exp2_raw + exp3_raw with one
consistent command encoding, then export to apt_g1/data/exp_all3/.

Command feature (14-d): 5-mode one-hot [0,1,2,17,18] + mdir(3) + fdir(3) +
speed + height + planner. angle_bin: 8 bins from atan2(mdir.y, mdir.x).
"""
import collections
import os

import numpy as np

RAW = "/home/cvgluser/ros2_data/apt_g1/data"
OUT = "/home/cvgluser/ros2_data/apt_g1/data/exp_all3"
MODES = [0, 1, 2, 17, 18]  # union across batches


def load_csv(path, dtype, skip=0):
    with open(path) as f:
        lines = f.read().splitlines()
    rows = []
    for ln in lines[skip:]:
        if not ln.strip():
            continue
        parts = ln.split(",")
        if parts and parts[-1] == "":
            parts = parts[:-1]
        rows.append([float(x) for x in parts])
    return np.array(rows, dtype=dtype)


def process(raw_dir, tag):
    pi = load_csv(f"{raw_dir}/policy_input.csv", np.float32)
    cmd = load_csv(f"{raw_dir}/commands.csv", np.float64)
    tok = load_csv(f"{raw_dir}/logs/token_state.csv", np.float64, skip=1)
    tok_cols = tok[:, 5:69].astype(np.float32)
    diff = None
    if len(tok_cols) == len(pi):
        diff = float(np.abs(pi[:, :64] - tok_cols).max())
    else:
        # token_state may include extra rows at a different sample rate; try to
        # align by offset, otherwise skip the verification (policy_input is the
        # authoritative token source, verified for exp1/exp2 with diff=0).
        for off in range(0, min(len(tok_cols) - len(pi) + 1, 2000)):
            cand = np.abs(pi[:, :64] - tok_cols[off : off + len(pi)]).max()
            if cand < 1e-3:
                diff = float(cand)
                print(f"[{tag}] token_state aligned at offset {off}")
                break
        if diff is None:
            print(f"[{tag}] WARNING: token_state length mismatch "
                  f"({len(tok_cols)} vs {len(pi)}), skipping verification")
    print(f"[{tag}] PI {pi.shape} CMD {cmd.shape} tok {tok.shape} token diff "
          f"{'n/a' if diff is None else f'{diff:.3e}'}")
    n, m = len(pi), len(cmd)
    j = np.floor(np.arange(n) * m / n).astype(int)
    cm = cmd[j]
    mode = cm[:, 7].astype(int)
    mdir = cm[:, 8:11]
    fdir = cm[:, 11:14]
    speed = cm[:, 14]
    height = cm[:, 15]
    planner = cm[:, 5]
    mode_oh = np.zeros((n, len(MODES)), dtype=np.float32)
    for k, mm in enumerate(MODES):
        mode_oh[:, k] = mode == mm
    cmd_feat = np.concatenate(
        [
            mode_oh,
            mdir.astype(np.float32),
            fdir.astype(np.float32),
            speed[:, None].astype(np.float32),
            height[:, None].astype(np.float32),
            planner[:, None].astype(np.float32),
        ],
        axis=1,
    )
    ang = np.arctan2(mdir[:, 1], mdir[:, 0])
    angle_bin = np.floor((ang + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    return dict(
        proprio=pi[:, 64:].copy(),
        cmd=cmd_feat,
        token=pi[:, :64].copy(),
        mode=mode.astype(np.int64),
        speed=speed.astype(np.float32),
        angle_bin=angle_bin.astype(np.int64),
    )


parts = []
for name, raw in [("exp1", "exp1_raw"), ("exp2", "exp2_raw"), ("exp3", "exp3_raw")]:
    raw_dir = f"{RAW}/{raw}"
    if not os.path.isdir(raw_dir):
        print(f"skip {name}: {raw_dir} missing")
        continue
    parts.append((name, process(raw_dir, name)))

if not parts:
    raise SystemExit("no raw data")

merged = {}
for k in ["proprio", "cmd", "token", "mode", "speed", "angle_bin"]:
    arrs = [p[k] for _, p in parts]
    merged[k] = np.concatenate(arrs, axis=0)
    print(k, merged[k].shape)

os.makedirs(OUT, exist_ok=True)
for k, v in merged.items():
    np.save(f"{OUT}/{k}.npy", v)
np.save(f"{OUT}/meta_modes.npy", np.array(MODES, dtype=np.int32))

lat = np.unique(np.round(merged["token"].ravel() * 16) / 16.0)
print("token lattice values:", len(lat))
cnt = collections.Counter(
    (int(m), round(float(s), 2), int(b))
    for m, s, b in zip(merged["mode"], merged["speed"], merged["angle_bin"])
)
for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:40]:
    print("  ", k, v)
print("saved to", OUT)
