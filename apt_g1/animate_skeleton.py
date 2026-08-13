"""Animate a logged G1 rollout (npz from rollout_log_joints.py) as a 2D
sagittal-plane stick-figure video, no GPU rendering needed.

Side view: x = forward, z = up. The pelvis comes from the logged base pose; the
leg/arm segment angles come from the sagittal joint angles (hip_pitch / knee /
ankle_pitch / shoulder_pitch / elbow). Link lengths are approximate G1
proportions (qualitative showcase, not a CAD render).

Outputs an mp4 (ffmpeg) or, if ffmpeg is unavailable, a folder of PNG frames.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# --- approximate G1 link lengths (meters) ---
TORSO = 0.46   # pelvis -> shoulder
THIGH = 0.36
SHANK = 0.34
FOOT = 0.10
UARM = 0.26
FARM = 0.24

# G1_ISAACLab_ORDER indices
L_HIP_P, R_HIP_P = 0, 1
L_KNEE, R_KNEE = 9, 10
L_ANK_P, R_ANK_P = 13, 14
L_SHO_P, R_SHO_P = 11, 12
L_ELB, R_ELB = 21, 22


def quat_to_pitch(quat):
    """Pitch (rotation about body y, sagittal) from wxyz quat (approx)."""
    w, x, y, z = quat
    # pitch = atan2(2(w y - z x), 1 - 2(y^2 + x^2)) -- standard, depends on convention
    return np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def leg_pts(hip_pitch, knee, ankle_pitch, px, pz, pitch=0.0):
    """Return (hip, knee, ankle, toe) world XY for one leg given pelvis pose."""
    # thigh absolute angle: hangs down at 0, +hip_pitch swings toe forward.
    # Add torso pitch so the whole leg leans with the body.
    th = pitch + hip_pitch
    kx = px + THIGH * np.sin(th)
    kz = pz - THIGH * np.cos(th)
    sh = th - knee  # knee flexion bends shank back
    ax = kx + SHANK * np.sin(sh)
    az = kz - SHANK * np.cos(sh)
    ft = sh - ankle_pitch
    tx = ax + FOOT * np.cos(ft)   # toe roughly horizontal forward
    tz = az
    return (px, pz), (kx, kz), (ax, az), (tx, tz)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default="outputs/e29_skeleton.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--title", default="E29 latent walk (skeleton)")
    args = ap.parse_args()

    d = np.load(args.npz)
    base = d["base_xyz"]; jp = d["joint_pos"]; quat = d["base_quat"]
    cmd_vx = float(d["command_vx"]) if "command_vx" in d else 0.8
    N = len(base)
    dt = 1.0 / 50.0
    print(f"[anim] {N} steps, base final x={base[-1,0]:.2f}m, "
          f"mean_vx={(base[-1,0]-base[0,0])/(N*dt):.3f} m/s, cmd={cmd_vx}")

    import matplotlib
    matplotlib.use("Agg")
    # prefer a bundled ffmpeg (imageio-ffmpeg) if system ffmpeg is missing
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ground_z = 0.0
    ax.axhline(ground_z, color="k", lw=1.0)
    ln_l, = ax.plot([], [], "-", color="#1f77b4", lw=3)   # left leg
    ln_r, = ax.plot([], [], "-", color="#d62728", lw=3)   # right leg
    ln_torso, = ax.plot([], [], "-", color="k", lw=3.5)
    ln_al, = ax.plot([], [], "-", color="#1f77b4", lw=2)
    ln_ar, = ax.plot([], [], "-", color="#d62728", lw=2)
    head = matplotlib.patches.Circle((0, 0), 0.09, color="k")
    ax.add_patch(head)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=9,
                  bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # camera follows the robot: window of fixed width centered on pelvis x.
    win = 2.5
    ax.set_ylim(-0.1, 1.55)
    ax.set_xlim(-win, win)
    ax.set_aspect("equal")
    ax.set_title(args.title)
    ax.set_xlabel("forward x (m)")

    def pelvis(i):
        px = base[i, 0]
        pz = base[i, 2]
        pitch = quat_to_pitch(quat[i])
        return px, pz, pitch

    def update(i):
        px, pz, pitch = pelvis(i)
        # torso top (shoulder) leans with pitch
        sx = px + TORSO * np.sin(pitch)
        sz = pz + TORSO * np.cos(pitch)
        # legs (feet planted from pelvis)
        L = leg_pts(jp[i, L_HIP_P], jp[i, L_KNEE], jp[i, L_ANK_P], px, pz, pitch)
        R = leg_pts(jp[i, R_HIP_P], jp[i, R_KNEE], jp[i, R_ANK_P], px, pz, pitch)
        ln_l.set_data([L[0][0], L[1][0], L[2][0], L[3][0]],
                      [L[0][1], L[1][1], L[2][1], L[3][1]])
        ln_r.set_data([R[0][0], R[1][0], R[2][0], R[3][0]],
                      [R[0][1], R[1][1], R[2][1], R[3][1]])
        ln_torso.set_data([px, sx], [pz, sz])
        # arms from shoulder
        la_sh = pitch + jp[i, L_SHO_P]
        la_e = (sx + UARM * np.sin(la_sh), sz - UARM * np.cos(la_sh))
        la_h = (la_e[0] + FARM * np.sin(la_sh - jp[i, L_ELB]),
                la_e[1] - FARM * np.cos(la_sh - jp[i, L_ELB]))
        ln_al.set_data([sx, la_e[0], la_h[0]], [sz, la_e[1], la_h[1]])
        ra_sh = pitch + jp[i, R_SHO_P]
        ra_e = (sx + UARM * np.sin(ra_sh), sz - UARM * np.cos(ra_sh))
        ra_h = (ra_e[0] + FARM * np.sin(ra_sh - jp[i, R_ELB]),
                ra_e[1] - FARM * np.cos(ra_sh - jp[i, R_ELB]))
        ln_ar.set_data([sx, ra_e[0], ra_h[0]], [sz, ra_e[1], ra_h[1]])
        head.center = (sx + 0.12 * np.sin(pitch), sz + 0.12)
        # pan camera with the robot
        ax.set_xlim(px - win, px + win)
        t = i * dt
        vx = (base[min(i + 5, N - 1), 0] - base[max(i - 5, 0), 0]) / (10 * dt)
        txt.set_text(f"t={t:4.1f}s  x={px:5.2f}m  vx≈{vx:.2f} m/s")
        return ln_l, ln_r, ln_torso, ln_al, ln_ar, head, txt

    ani = FuncAnimation(fig, update, frames=N, interval=1000 / args.fps, blit=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        ani.save(str(out), writer=FFMpegWriter(fps=args.fps), dpi=110)
        print(f"[anim] wrote {out}")
    except Exception as e:
        print(f"[anim] ffmpeg mp4 failed ({e}); saving PNG frames instead")
        frdir = out.with_suffix("")
        frdir.mkdir(parents=True, exist_ok=True)
        for i in range(0, N, 2):
            update(i)
            fig.savefig(frdir / f"frame_{i:04d}.png", dpi=110)
        print(f"[anim] wrote {len(range(0, N, 2))} frames to {frdir}/")


if __name__ == "__main__":
    sys.exit(main())
