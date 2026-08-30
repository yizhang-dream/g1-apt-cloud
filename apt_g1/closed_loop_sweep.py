"""Closed-loop terrain-amplitude sweep: where does blind re-planning break?

The official SONIC planner is KINEMATIC and terrain-blind (no heightmap input):
its rough-terrain generalization comes purely from 10Hz re-planning on live
proprioception. This sweeps the terrain bump amplitude and runs the closed-loop
walk at each, averaging over terrain seeds to remove spawn-location luck, to
locate the true boundary of that blind re-planning.

Terrain: uniform bumps (0..amp) on 0.1 m cells, 5-cell box blur, NO flat center
(a hard flat edge tripped the robot in an earlier version -- see tracker MQ10).

Usage (server, mjlab venv):
    python closed_loop_sweep.py
"""

from __future__ import annotations

import os
import re
import struct
import sys

import numpy as np

REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
SCENE_DIR = f"{REPO}/gear_sonic/data/robot_model/model_data/g1"
SRC = f"{SCENE_DIR}/scene_43dof.xml"
OUT = f"{SCENE_DIR}/scene_43dof_rough.xml"
MESH_DIR = f"{SCENE_DIR}/meshes"

SIZE_M, RES, BASE_Z = 40.0, 0.1, 0.5
AMPS = [0.08, 0.10, 0.12, 0.14, 0.16, 0.20]
SEEDS = [0, 1, 2]


def build_terrain(amp: float, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = int(SIZE_M / RES)
    h = rng.uniform(0, amp, (n, n))  # bumps only (Isaac noise_range=(0, amp))
    h = (h + np.roll(h, 1, 0) + np.roll(h, -1, 0) + np.roll(h, 1, 1) + np.roll(h, -1, 1)) / 5.0
    hmin, hmax = float(h.min()), float(h.max())
    data = ((h - hmin) / (hmax - hmin)).astype(np.float64)
    os.makedirs(MESH_DIR, exist_ok=True)
    with open(f"{MESH_DIR}/rough_hfield.bin", "wb") as f:
        f.write(struct.pack("ii", n, n))
        f.write(data.astype(np.float32).tobytes())
    xml = open(SRC, encoding="utf-8").read()
    hfield_asset = (f'<hfield name="rough" size="{SIZE_M/2} {SIZE_M/2} {hmax-hmin:.4f} {BASE_Z}" '
                    f'nrow="{n}" ncol="{n}" file="rough_hfield.bin" content_type="bin"/>\n')
    xml = xml.replace("<asset>", "<asset>\n" + hfield_asset, 1)
    xml = re.sub(r'<geom name="floor"[^>]*/>',
                 f'<geom name="floor" type="hfield" hfield="rough" pos="0 0 {hmin:.4f}" material="groundplane"/>',
                 xml)
    open(OUT, "w", encoding="utf-8").write(xml)
    return hmax


if __name__ == "__main__":
    sys.path.insert(0, "/home/cvgluser/ros2_data")
    sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
    from planner_closed_loop import run

    print(f"{'amp':>6}{'survived':>9}{'adv_mean':>9}{'adv_list':>22}{'h_min_mean':>10}")
    for amp in AMPS:
        advs, hmins, falls = [], [], []
        for seed in SEEDS:
            hmax = build_terrain(amp, seed)
            r = run("walk", n_steps=300, scene="rough", spawn_x=-6.0)
            advs.append(r["adv"]); hmins.append(r["h_min"])
            falls.append(r["fall"] is None)
        survived = sum(falls)
        adv_mean = round(float(np.mean(advs)), 2)
        hmin_mean = round(float(np.mean(hmins)), 2)
        print(f"{amp:>6}{survived:>6}/{len(SEEDS)}"
              f"{adv_mean:>9}{str(advs):>22}{hmin_mean:>10}")
