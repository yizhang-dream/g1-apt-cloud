"""Terrain experiment summary figure: survival vs noise for noaux / blind aux
(E15) / elevation aux (E16), fixed terrain seed 0."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/cvgluser/ros2_data/apt_g1/outputs"


def load(path):
    d = json.load(open(path))
    return d["A_walk60"]


# (noise, variant) -> list of fall_step (None = survived full 3000)
rows = {
    "prior(noaux)": [
        (0.04, "terr_sweep_n0.04_s0.json", "noaux"),
        (0.06, "terr_fix_noaux_n0.06_s0.json", "noaux"),
        (0.08, "terr_fix_noaux_n0.08_s0.json", "noaux"),
        (0.10, "terr_sweep_n0.10_s0.json", "noaux"),
    ],
    "aux blind (E15)": [
        (0.04, "terr_e15_n0.04_s0.json", "aux"),
        (0.06, "terr_fix_e15_n0.06_s0.json", "aux"),
        (0.08, "terr_fix_e15_n0.08_s0.json", "aux"),
        (0.10, "terr_e15_n0.10_s0.json", "aux"),
    ],
    "aux + elevation (E16)": [
        (0.06, "terr_e16_n0.06_s0.json", "aux"),
        (0.08, "terr_e16_n0.08_s0.json", "aux"),
    ],
}

fig, ax = plt.subplots(figsize=(9, 5.5))
colors = {"prior(noaux)": "#1f77b4", "aux blind (E15)": "#d62728", "aux + elevation (E16)": "#2ca02c"}
marks = {"prior(noaux)": "o", "aux blind (E15)": "s", "aux + elevation (E16)": "^"}
for name, items in rows.items():
    xs, ys = [], []
    for noise, fname, key in items:
        try:
            d = load(f"{OUT}/{fname}")[key]
        except Exception:
            continue
        falls = [3000 if r["fall_step"] is None else r["fall_step"] for r in d.values()]
        xs.append(noise)
        ys.append(np.mean(falls))
        for f in falls:
            ax.plot(
                noise,
                f,
                marker=marks[name],
                mfc="none",
                mec=colors[name],
                ms=7,
                ls="",
            )
    ax.plot(xs, ys, color=colors[name], label=name, marker=marks[name], lw=2)

ax.axhline(3000, color="gray", ls="--", lw=1)
ax.text(0.101, 3050, "60s full walk", fontsize=9, color="gray")
ax.set_xlabel("rough terrain noise (m)")
ax.set_ylabel("mean survival time (control steps, 50Hz)")
ax.set_title("G1 frozen-router prior on rough terrain: blind aux vs elevation aux\n(fixed terrain seed 0, 3 seeds each)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/terrain_summary.png", dpi=150)
print("saved", f"{OUT}/terrain_summary.png")
