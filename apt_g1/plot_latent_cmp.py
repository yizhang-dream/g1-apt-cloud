"""Bar comparison of the latent-mode A_walk60 experiments (E27-E30).

Showcase artifact for the latent-direction experiments: mean forward velocity and
60s displacement for each variant. Run on the server (matplotlib) and scp the PNG
back. Update RESULTS as new runs complete.
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# A_walk60 results (6-seed means where available). vx in m/s, disp in meters.
RESULTS = {
    "E27\nbaseline":      {"vx": 0.317, "disp": 19.1, "note": "frozen clock, base reward"},
    "E28a\n+cadence":     {"vx": 0.342, "disp": 13.1, "note": "drift"},
    "E28b\n+reward retune": {"vx": 0.253, "disp": 14.9, "note": "slower"},
    "E29\n+manifold KL":  {"vx": 0.348, "disp": 18.3, "note": "best vx, stable (coef 1e-2)"},
    "E30\nKL coef x10":  {"vx": 0.336, "disp": 16.0, "note": "over-pinned (coef 1e-1)"},
}

WALK_DATA_SPEED = 0.6  # the SONIC walk data was recorded at 0.6 m/s


def main(out: str = "outputs/latent_cmp.png") -> None:
    names = list(RESULTS.keys())
    vx = [RESULTS[n]["vx"] for n in names]
    dp = [RESULTS[n]["disp"] for n in names]
    x = range(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    bars1 = ax1.bar(x, vx, color=["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"][: len(names)])
    ax1.axhline(WALK_DATA_SPEED, color="orange", ls="--", lw=1.2,
                label="walk data speed (0.6)")
    ax1.set_ylabel("mean forward vx (m/s)")
    ax1.set_title("A_walk60 forward speed")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(names, fontsize=8)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_ylim(0, 0.7)
    for b, v in zip(bars1, vx):
        if v is not None:
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                     ha="center", fontsize=8)

    bars2 = ax2.bar(x, dp, color=["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"][: len(names)])
    ax2.set_ylabel("mean displacement (m, 60s)")
    ax2.set_title("A_walk60 displacement (straight-line)")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylim(0, 22)
    for b, v in zip(bars2, dp):
        if v is not None:
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                     ha="center", fontsize=8)

    fig.suptitle("Latent-mode (VAE->SONIC) speed-ceiling ablation — E27/E28/E29/E30",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/latent_cmp.png")
