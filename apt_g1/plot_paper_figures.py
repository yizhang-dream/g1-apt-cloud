"""Paper figures for APT-RL x SONIC x G1 (2026-08-27 doc-completion pass).

Generates the summary figures that the final reports cite but that were never
drawn. Data sources, in order of preference:

1. E-line ladder / Pareto / terrain matrix / E48 residual — read the server
   eval JSONs under apt_g1/outputs/ (schema: {test}.{arm}.seed{N} ->
   {steps, completed, fall_step, h_min, vx, disp, v_speed}, 50 Hz sim).
2. MQ08/MQ10/MQ11 planner line, TO-line survival battle, mjlab reference —
   embedded from refine-logs/EXPERIMENT_TRACKER.md canonical rows (each table
   cites its tracker section; those experiments have no uniform JSON).

Outputs -> apt_g1/outputs/figs/*.png  (labels in English: server has no CJK font)

Run: ~/ros2_data/.venv_isaac/bin/python apt_g1/plot_paper_figures.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/home/cvgluser/ros2_data/apt_g1/outputs"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

HZ = 50.0  # sim step rate -> steps/Hz = seconds


# ---------------------------------------------------------------- helpers
def load_eval(name_pattern):
    """Find first JSON matching pattern; aggregate A_walk60 over seeds/arms.

    Prefer the 'noaux' arm when present (residual/aux-off is the clean-policy
    view); fall back to any arm. Returns (disp, vx, straight, n) or None.
    """
    matches = sorted(glob.glob(os.path.join(OUT_DIR, name_pattern)))
    if not matches:
        return None
    data = json.load(open(matches[0]))
    if "A_walk60" not in data or not isinstance(data["A_walk60"], dict):
        return None
    a = data["A_walk60"]
    if isinstance(a.get("noaux"), dict):
        arms = [a["noaux"]]
    elif isinstance(a.get("aux"), dict):
        arms = [a["aux"]]
    else:  # pool every arm (e.g. E26 uses arm name 'phaseaux')
        arms = [v for v in a.values() if isinstance(v, dict)]
    seeds = [s for arm in arms for k, s in sorted(arm.items())
             if k.startswith("seed") and isinstance(s, dict)]
    if not seeds:
        return None
    disp = sum(s["disp"] for s in seeds) / len(seeds)
    vx = sum(s["vx"] for s in seeds) / len(seeds)
    secs = sum(s["steps"] for s in seeds) / len(seeds) / HZ
    straight = disp / (vx * secs) if vx > 1e-6 else 0.0
    return disp, vx, min(straight, 1.0), len(seeds)


def survival(name_pattern, test="A_walk60", arm=None):
    """Fraction of seeds with fall_step == None across seeds (optionally one arm)."""
    matches = sorted(glob.glob(os.path.join(OUT_DIR, name_pattern)))
    ok, tot = 0, 0
    for m in matches:
        data = json.load(open(m))
        arms_dict = data.get(test, {})
        if not isinstance(arms_dict, dict):
            continue
        arms = ([arms_dict[arm]] if arm and isinstance(arms_dict.get(arm), dict)
                else [v for v in arms_dict.values() if isinstance(v, dict)])
        for armd in arms:
            for k, s in armd.items():
                if k.startswith("seed") and isinstance(s, dict):
                    tot += 1
                    ok += 1 if s.get("fall_step") is None else 0
    return (ok, tot) if tot else None


def bar_annotate(ax, bars, texts, fontsize=8):
    for b, t in zip(bars, texts):
        ax.annotate(t, (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=fontsize)


# ================================================================ FIG 1
# Latent-line ladder: A-60s displacement per experiment.
# JSON-driven; fallback numbers from TRACKER rows (cited inline).
def fig1_latent_ladder():
    rows = [  # (label, json glob, fallback (disp, vx), group)
        ("Frozen prior\n(router+decoder)", "isaac_eval_noaux.json", (47.0, 0.79), "ref"),
        ("E26 phase-offset", "isaac_eval_e26.json", (45.9, 0.77), "phase"),
        ("E27 latent VAE", "isaac_eval_e27.json", (19.1, 0.32), "e27"),
        ("E29 +KL prior", "isaac_eval_e29.json", (18.3, 0.348), "e27"),
        ("E31 speed-cond", "isaac_eval_e31.json", (10.0, 0.535), "cond"),
        ("E35 dir-cond", "isaac_eval_e35.json", (16.3, 0.295), "cond"),
        ("E36 +speed inc.", "isaac_eval_e36.json", (15.4, 0.372), "cond"),
        ("E37 dir-disent", "isaac_eval_e37.json", (21.55, 0.370), "disent"),
        ("E38 +hi inc.", "isaac_eval_e38.json", (19.4, 0.399), "disent"),
        ("E39 dual-disent", "isaac_eval_e39.json", (24.65, 0.417), "sweet"),
        ("E40 +hi inc.", "isaac_eval_e40.json", (24.40, 0.456), "disent"),
        ("E43 fast-weight", "isaac_eval_e43.json", (13.5, 0.347), "disent"),
        ("E45 from-scratch", "eval_from0_dec_01.json", (14.0, 0.27), "from0"),
        ("E46 +E39 VAE", "eval_e45_e39_from0.json", (21.7, 0.418), "from0"),
        ("E47 +heading", "eval_e47_heading.json", (23.8, 0.42), "from0sweet"),
        ("vanilla joint-space\n(from0_01)", "eval_from0_01.json", (0.0, 0.57), "fail"),
        ("mjlab from-scratch\n(native sim)", None, (46.2, 0.775), "ref"),
    ]
    colors = {
        "ref": "#888888", "phase": "#b0b0b0", "e27": "#7f9fc4",
        "cond": "#f2a04e", "disent": "#5ba85f", "sweet": "#1d7a30",
        "from0": "#5f7fd7", "from0sweet": "#2653c9", "fail": "#c0392b",
    }
    labels, disps, vxs, cs, used_json = [], [], [], [], []
    for label, pat, fb, grp in rows:
        got = load_eval(pat) if pat else None
        disp, vx = (got[0], got[1]) if got else fb
        labels.append(label)
        disps.append(disp)
        vxs.append(vx)
        cs.append(colors[grp])
        used_json.append("J" if got else "T")

    fig, ax = plt.subplots(figsize=(13, 5.2))
    bars = ax.bar(range(len(labels)), disps, color=cs, edgecolor="k", lw=0.4)
    for i, (d, v) in enumerate(zip(disps, vxs)):
        ax.annotate(f"vx {v:.2f}" if d > 0 else "0/3 fall",
                    (i, d), ha="center", va="bottom", fontsize=8)
        ax.annotate(used_json[i], (i, 0.5), ha="center", fontsize=7, color="w",
                    weight="bold")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("A 60s straight-walk displacement (m)")
    ax.set_title("Latent-line ladder (A 60s): E27 -> E47 vs frozen prior / vanilla / mjlab\n"
                 "(J = read from eval JSON, T = TRACKER canonical row)")
    ax.axhline(47.0, color="k", ls="--", lw=1)
    ax.annotate("frozen prior 47m", (len(labels) - 1.2, 47.5), fontsize=8)
    ax.set_ylim(0, 55)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_latent_ladder.png"), dpi=150)
    plt.close(fig)


# ================================================================ FIG 2
# Speed vs straightness Pareto (the "fast AND straight" story).
def fig2_speed_straight_pareto():
    # (label, json, fallback (vx, straight, disp), group)
    pts = [
        ("E27", "isaac_eval_e27.json", (0.32, 0.98, 19.1), "e27"),
        ("E29", "isaac_eval_e29.json", (0.348, 0.88, 18.3), "e27"),
        ("E31", "isaac_eval_e31.json", (0.535, 0.31, 10.0), "cond"),
        ("E35", "isaac_eval_e35.json", (0.295, 0.93, 16.3), "cond"),
        ("E36", "isaac_eval_e36.json", (0.372, 0.69, 15.4), "cond"),
        ("E37", "isaac_eval_e37.json", (0.370, 0.97, 21.55), "disent"),
        ("E38", "isaac_eval_e38.json", (0.399, 0.81, 19.4), "disent"),
        ("E39", "isaac_eval_e39.json", (0.417, 0.98, 24.65), "sweet"),
        ("E40", "isaac_eval_e40.json", (0.456, 0.89, 24.40), "disent"),
        ("E43", "isaac_eval_e43.json", (0.347, 0.65, 13.5), "disent"),
        ("E45", "eval_from0_dec_01.json", (0.27, 1.0, 14.0), "from0"),
        ("E46", "eval_e45_e39_from0.json", (0.418, 0.86, 21.7), "from0"),
        ("E47", "eval_e47_heading.json", (0.42, 0.944, 23.8), "from0sweet"),
    ]
    colors = {"e27": "#7f9fc4", "cond": "#f2a04e", "disent": "#5ba85f",
              "sweet": "#1d7a30", "from0": "#5f7fd7", "from0sweet": "#2653c9"}
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, pat, fb, grp in pts:
        got = load_eval(pat) if pat else None
        if got:
            disp, vx, straight = got[0], got[1], got[2]
        else:
            vx, straight, disp = fb
        ax.scatter(vx, straight, s=disp * 14, c=colors[grp], alpha=0.75,
                   edgecolors="k", lw=0.6, zorder=3)
        ax.annotate(label, (vx, straight), xytext=(5, 4),
                    textcoords="offset points", fontsize=9, weight="bold")
    ax.axhline(0.95, color="gray", ls=":", lw=1)
    ax.annotate("straightness 0.95", (0.245, 0.955), fontsize=8, color="gray")
    ax.annotate("marker size = A-60s displacement", (0.245, 0.28), fontsize=8,
                color="gray")
    ax.set_xlabel("mean vx (m/s)")
    ax.set_ylabel("straightness = disp / (vx * 60s)")
    ax.set_title("Fast AND straight: speed-straightness Pareto of the latent line\n"
                 "(E31 fast-but-drifts; E39 dual-disentangle & E47 from-scratch+heading are the sweet spots)")
    ax.set_xlim(0.22, 0.58)
    ax.set_ylim(0.25, 1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_speed_straight_pareto.png"), dpi=150)
    plt.close(fig)


# ================================================================ FIG 3
# Terrain boundary, resorted by SHAPE (G0 revision):
# convex-only passes 0.06 / pits at +-0.06 kill / convex-only 0.08 kills.
def fig3_terrain_boundary():
    rows = [
        "frozen prior (T1)",
        "E39 sweet spot (E41)",
        "E42 terrain-trained",
        "E47 from-scratch best",
        "E48 residual-off (it700)",
        "mjlab from-scratch (native)",
    ]
    # mjlab: TRACKER M-FROM0 — 0.06: 1/3 full-60s (2/3 fall 38-53s);
    # 0.08: 1/3 full-60s; pits never tested (native sim has no paper-shape terrain).
    mjlab = {"convex 0.06": (1, 3), "convex 0.08": (1, 3),
             "pits ±0.06 (0.1m)": None, "pits ±0.06 (paper 0.2m)": None}

    cols = ["convex 0.06", "convex 0.08",
            "pits ±0.06 (0.1m)", "pits ±0.06 (paper 0.2m)"]
    colmap = {
        "frozen prior (T1)": ["terr_fix_noaux_n0.06_s0.json",
                              "terr_sweep_n008_s0.json",
                              None, None],
        "E39 sweet spot (E41)": ["terr_e41_n0.06_s0.json",
                                 ["terr_e41_n0.08_s0.json", "terr_e41_n0.08_s1.json"],
                                 None, "eval_e39_paper006_s*.json"],
        "E42 terrain-trained": ["terr_e42_n0.06_s0.json",
                                ["terr_e42_n0.08_s0.json", "terr_e42_n0.08_s1.json"],
                                None, None],
        "E47 from-scratch best": ["eval_e47_terrain_006_s*.json",
                                  "eval_e47_terrain_008_s*.json",
                                  "eval_e47_sym006_s*.json",
                                  "eval_e47_paper006_s*.json"],
        "E48 residual-off (it700)": ["eval_e48_r006_s0.json",
                                     "eval_e48_r008_s0.json",
                                     "eval_e48_sym006_s0.json",
                                     "eval_e48_paper006_s0.json"],
    }
    # frozen-prior convex cells backed by terr_fix_noaux_*; paper-shape terrain
    # was only measured for E39/E47/E48 (G0) — other cells stay blank.

    grid_ok = [[None] * len(cols) for _ in rows]
    grid_tot = [[0] * len(cols) for _ in rows]
    for r, label in enumerate(rows):
        if label.startswith("mjlab"):
            for c, col in enumerate(cols):
                if mjlab[col]:
                    grid_ok[r][c], grid_tot[r][c] = mjlab[col]
            continue
        for c, spec in enumerate(colmap[label]):
            if spec is None:
                continue
            pats = spec if isinstance(spec, list) else [spec]
            ok = tot = 0
            # E48 row: only the residual-OFF (noaux) arm represents the base policy
            arm_filter = "noaux" if label.startswith("E48") else None
            for p in pats:
                s = survival(p, arm=arm_filter)
                if s:
                    ok += s[0]
                    tot += s[1]
            if tot:
                grid_ok[r][c], grid_tot[r][c] = ok, tot

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for r in range(len(rows)):
        for c in range(len(cols)):
            if grid_tot[r][c] == 0:
                txt, v = "—", float("nan")
            else:
                v = grid_ok[r][c] / grid_tot[r][c]
                txt = f"{grid_ok[r][c]}/{grid_tot[r][c]}"
            ax.add_patch(plt.Rectangle((c, len(rows) - 1 - r), 1, 1,
                         facecolor=plt.cm.RdYlGn(v) if v == v else "#eeeeee",
                         edgecolor="k", lw=0.8))
            fg = "w" if (v == v and 0.25 < v < 0.8) else "k"
            ax.text(c + 0.5, len(rows) - 1 - r + 0.5, txt, ha="center",
                    va="center", fontsize=11, weight="bold", color=fg)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.set_xticks([c + 0.5 for c in range(len(cols))], cols, fontsize=9)
    ax.set_yticks([len(rows) - 0.5 - r for r in range(len(rows))],
                  rows, fontsize=9)
    ax.set_title("Terrain survival by SHAPE at amplitude 0.06/0.08 (A-60s, seeds pooled)\n"
                 "G0 revision: pits (negative obstacles) at ±0.06 kill everything on the "
                 "distilled path; mjlab native sim has no cliff")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_terrain_boundary.png"), dpi=150)
    plt.close(fig)


# ================================================================ FIG 4
# Planner replication line (MQ08/MQ10/MQ11) — embedded from TRACKER tables.
def fig4_planner_line():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    # (a) MQ08: flat vs rough adv per mode (TRACKER MQ08 结论 table)
    modes = ["run", "walk", "slow_walk", "stealth", "crawl", "squat", "idle"]
    flat = [1.13, 0.78, 0.38, 0.84, 0.51, 0.32, 0.45]
    rough = [1.00, 0.26, 0.43, 0.73, 0.44, 0.35, 0.44]
    x = range(len(modes))
    w = 0.38
    ax = axes[0]
    ax.bar([i - w / 2 for i in x], flat, w, label="flat", color="#7f9fc4")
    ax.bar([i + w / 2 for i in x], rough, w, label="rough 0.08", color="#c0504d")
    ax.set_xticks(list(x), modes, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("advance (m)")
    ax.set_title("(a) MQ08 open-loop: gait aggressiveness\npredicts degradation, not CoM height")
    ax.legend(fontsize=8)

    # (b) MQ10: closed-loop blind replanning amp sweep (TRACKER MQ10 table)
    ax = axes[1]
    amps = [0.08, 0.10, 0.12, 0.14, 0.16, 0.20]
    surv = [1, 2, 3, 0, 1, 1]
    adv = [1.83, 2.07, 2.83, 1.02, 0.96, 0.67]
    ax.bar([str(a) for a in amps], adv, color="#5ba85f",
           yerr=None, capsize=3)
    for i, (s, a) in enumerate(zip(surv, adv)):
        ax.annotate(f"{s}/3", (i, a), ha="center", va="bottom",
                    fontsize=9, weight="bold")
    ax.axvspan(2.5, 5.5, color="#c0392b", alpha=0.12)
    ax.annotate("boundary\n0.12–0.14", (3.4, 2.6), fontsize=9, color="#c0392b")
    ax.set_xlabel("terrain amplitude (m)")
    ax.set_ylabel("advance (m)")
    ax.set_title("(b) MQ10 closed-loop blind replan:\nseed-sensitive, dies at amp ≥ 0.14")

    # (c) MQ11: gait mode lever at amp 0.14 (TRACKER MQ11 table)
    ax = axes[2]
    gaits = ["walk", "stealth", "crawl"]
    ok = [0, 1, 3]
    ax.bar(gaits, ok, color=["#c0392b", "#f2a04e", "#1d7a30"])
    for i, s in enumerate(ok):
        ax.annotate(f"{s}/3 survive", (i, s + 0.05), ha="center", fontsize=10,
                    weight="bold")
    ax.set_ylim(0, 3.6)
    ax.set_ylabel("survivors of 3 seeds")
    ax.set_title("(c) MQ11 gait mode is the real lever\n(crawl 3/3 at walk's boundary 0.14)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_planner_line.png"), dpi=150)
    plt.close(fig)


# ================================================================ FIG 5
# TO-line survival battle (torque control line), embedded from TRACKER TO rows.
def fig5_to_battle():
    # (label, seconds, note, color)
    bars = [
        ("A-ID feedforward\n(paper-style)", 2.5, "stands, no walk", "#c0392b"),
        ("TO06 SRB torque", 2.0, "sign+1 falls 1.6–2.0s", "#c0392b"),
        ("TO15 2D-NMP torque", 3.1, "2D->3D gap", "#e07b39"),
        ("TO11 kinematic τ_clean", 3.58, "correct torque, wrong motion", "#e07b39"),
        ("TO18–22 hypotheses", 3.9, "8 cheap fixes eliminated", "#b0b0b0"),
        ("TO23 QP-WBC v1", 8.3, "+qfrc_passive fix\n(decisive)", "#5ba85f"),
        ("TO24 stance balance", 6.0, "perfect stand\n(mechanism verified)", "#5ba85f"),
        ("TO27 capture stepping", 7.9, "but h_mean 0.43\n(half-squat cheat)", "#b0b0b0"),
        ("TO28 LIPM-MPC", 3.3, "CoM ±0.02m but\nROLL diverges", "#b0b0b0"),
        ("TO29 centroidal MPC", 4.24, "angular momentum\nstate: x2 survival", "#5ba85f"),
        ("TO32 honest QP\nnarrow feet", 1.96, "cone-constraint bug\nfixed -> honest", "#c0392b"),
        ("TO32 honest QP\nwide feet f=2.0", 8.52, "first >8s, but lurching\n(knife-edge, not gait)", "#1d7a30"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    xs = range(len(bars))
    bs = ax.bar(xs, [b[1] for b in bars],
                color=[b[3] for b in bars], edgecolor="k", lw=0.4)
    for i, (label, sec, note, _) in enumerate(bars):
        ax.annotate(note, (i, sec + 0.12), ha="center", fontsize=7.5)
    ax.set_xticks(list(xs), [b[0] for b in bars], rotation=38, ha="right",
                  fontsize=8)
    ax.set_ylabel("closed-loop survival (s, flat ground)")
    ax.set_title("Torque line (TO01–TO35): joint-PD wall ~3.5s -> QP-WBC reactive layer doubles it;\n"
                 "cone-bug fix (TO32) makes the QP honest: 1.96s real feet vs 8.52s widened feet — "
                 "lateral under-actuation is the physical boundary")
    ax.axhline(3.5, color="gray", ls="--", lw=1)
    ax.annotate("joint-PD wall ~3.5s", (0.1, 3.6), fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_to_battle.png"), dpi=150)
    plt.close(fig)


# ================================================================ FIG 6
# E48/E48c residual channel: JSON-driven disp per terrain, residual on/off.
def fig6_e48_residual():
    conds = [  # (label, json, arm) — arm 'aux'=residual ON, 'noaux'=residual OFF
        ("E48 it700\nflat", "eval_e48_flat.json"),
        ("it700\nconvex .06", "eval_e48_r006_s0.json"),
        ("it700\nconvex .08", "eval_e48_r008_s0.json"),
        ("it700\npits paper", "eval_e48_paper006_s0.json"),
        ("it700\npits sym", "eval_e48_sym006_s0.json"),
        ("E48b it200\nflat", "eval_e48b_it200_flat.json"),
        ("E48c it800\nflat", "eval_e48c_it800_flat.json"),
        ("E48c it800\nconvex .06", "eval_e48c_it800_r006_s0.json"),
        ("E48c it1000\nconvex .06", "eval_e48c_it1000_r006_s0.json"),
        ("E48c it1000\nconvex .08", "eval_e48c_it1000_r008_s0.json"),
    ]
    on_d, off_d, fall_on, fall_off = [], [], [], []
    for label, pat in conds:
        m = os.path.join(OUT_DIR, pat)
        rec = [0.0, 0.0, 0, 0]
        if os.path.exists(m):
            d = json.load(open(m))
            for arm, tgt in (("aux", 0), ("noaux", 1)):
                seeds = [v for k, v in d["A_walk60"].get(arm, {}).items()
                         if k.startswith("seed")]
                if seeds:
                    rec[tgt] = sum(s["disp"] for s in seeds) / len(seeds)
                    rec[2 + tgt] = sum(1 for s in seeds
                                       if s["fall_step"] is None)
        on_d.append(rec[0]); off_d.append(rec[1])
        fall_on.append(rec[2]); fall_off.append(rec[3])

    fig, ax = plt.subplots(figsize=(12.5, 5))
    x = range(len(conds))
    w = 0.38
    ax.bar([i - w / 2 for i in x], on_d, w, color="#c0392b",
           label="residual ON (aux arm)")
    ax.bar([i + w / 2 for i in x], off_d, w, color="#5ba85f",
           label="residual OFF (noaux arm)")
    for i in x:
        if on_d[i] < 0.5:
            ax.annotate("fall", (i - w / 2, 0.4), fontsize=7, color="#c0392b",
                        ha="center", rotation=90)
    ax.set_xticks(list(x), [c[0] for c in conds], fontsize=8)
    ax.set_ylabel("A 60s displacement (m)")
    ax.set_title("E48/E48c full-joint residual channel: ON destroys every terrain, "
                 "OFF keeps the base healthy\n(it700 OFF on convex .08 = the single 1/9 "
                 "edge survival, not reproduced by E48c)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_e48_residual.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    fig1_latent_ladder()
    fig2_speed_straight_pareto()
    fig3_terrain_boundary()
    fig4_planner_line()
    fig5_to_battle()
    fig6_e48_residual()
    for f in sorted(glob.glob(os.path.join(FIG_DIR, "*.png"))):
        print("wrote", f, os.path.getsize(f), "bytes")
