"""Closed-loop gait-mode lever test at the terrain boundary (amp 0.14).

MQ10 located the blind re-planning boundary at ~0.12-0.14. This tests whether
the gait MODE (walk / stealth / crawl) is a lever that moves the boundary: does
a lower/slower gait survive amp 0.14 where walk fails?

Usage (server, mjlab venv):
    python closed_loop_levers.py
"""

from __future__ import annotations

import sys

REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
AMP = 0.14
SEEDS = [0, 1, 2]
MODES = ["walk", "stealth", "crawl"]


if __name__ == "__main__":
    sys.path.insert(0, "/home/cvgluser/ros2_data")
    sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
    from closed_loop_sweep import build_terrain
    from planner_closed_loop import run

    print(f"amp={AMP}")
    print(f"{'seed':>5}{'mode':>9}{'fall':>7}{'n':>5}{'adv':>7}{'h_min':>7}{'h_end':>7}")
    for seed in SEEDS:
        build_terrain(AMP, seed)
        for mode in MODES:
            r = run(mode, n_steps=300, scene="rough", spawn_x=-6.0)
            print(f"{seed:>5}{mode:>9}{str(r['fall']):>7}{r['n']:>5}{r['adv']:>7}"
                  f"{r['h_min']:>7}{r['h_end']:>7}")
