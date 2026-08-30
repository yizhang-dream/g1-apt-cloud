"""LIPM-consistent CoM walking gait + full-model ID torque, closed loop (TO21).

Diagnosis chain that motivates this script:
  TO11/TO18: kinematic-gait tau_clean + PD falls ~3.5 s regardless of whether
             the PD tracks the default pose or the planned gait -> not a
             feedback-targeting problem.
  TO19:      the 2D-NMP (dynamically consistent in 2D) gait + tracking also
             falls (~2-2.7 s) -> not a motion-quality problem either.
  TO20:      ankle-strategy CoM feedback on top cannot rescue it -> the plan
             itself is dynamically inconsistent.
Root cause: foot_gait_id assumes hx = v*t (CoM glides at constant speed
directly over the support foot = "infinite support"), while a real biped is
an inverted pendulum: the CoM must accelerate away from/toward the ZMP.

TO21 replaces the CoM trajectory with the LIPM periodic orbit for the SAME
footstep plan (ZMP = stance foot center, piecewise constant, half-cycle T/2),
then reuses the TO09 pipeline: analytic 2-link IK with hx = x_lipm(t) ->
numeric Qd/Qdd -> manual full-model ID (M qdd + qfrc_bias - qfrc_constraint)
-> closed loop: tau = KP (Q(phi) - q) + KD (Qd(phi) - qd) + tau_ff(phi)
[+ optional stance-ankle CoM feedback tracking x_lipm].

LIPM periodic orbit (relative coord xi = x - p, support shifts S/2 each
half-cycle Ts = T/2, omega^2 = g/z_c):
    xi(Ts)    = xi0 cosh(wTs) + (xid0/w) sinh(wTs),  support shift -> -S/2
    Symmetric periodic orbit solves the 2x2 LINEAR system:
    xi0 (1+cosh) + (xid0/w) sinh = S/2
    xid0 (1-cosh) - xi0 w sinh   = 0

Run on the SERVER under .venv_mjlab.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")

import mujoco

from foot_gait_id import (
    SCENE, DEFAULT_Q, PELVIS_H, HIP_H, ANKLE_H, HIP_DZ,
    LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE_PITCH,
    RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE_PITCH,
    ik, setup,
)
from eval_torque_gait import KP, KD, EFFORT, SAG

G = 9.81


def lipm_orbit(v, T, z_c):
    """Periodic LIPM orbit for the foot_gait_id footstep plan.

    Returns (omega, xi0, xid0): relative CoM state at the START of a LEFT
    stance half-cycle (support at x=L for t in [0,T/2), then RIGHT support).
    """
    omega = np.sqrt(G / z_c)
    Ts = T / 2.0
    S = v * T
    ch, sh = np.cosh(omega * Ts), np.sinh(omega * Ts)
    # symmetric periodic orbit: the relative state at the start of each
    # half-cycle (w.r.t. THAT half-cycle's support) repeats:
    #   xi(Ts) - S/2 = xi0   ->  xi0 (ch-1) + (xid0/w) sh = S/2
    #   xid(Ts)       = xid0 ->  xi0 w sh + xid0 (ch-1)  = 0
    A = np.array([[ch - 1.0, sh / omega], [omega * sh, ch - 1.0]])
    b = np.array([S / 2.0, 0.0])
    xi0, xid0 = np.linalg.solve(A, b)
    return omega, float(xi0), float(xid0)


def lipm_x(t, v, T, z_c, L):
    """Absolute CoM position/velocity at time t on the periodic orbit."""
    omega, xi0, xid0 = lipm_orbit(v, T, z_c)
    Ts = T / 2.0
    S = v * T
    k = int(np.floor(t / Ts))          # half-cycle index
    tau = t - k * Ts                    # time within the half-cycle
    p_k = L + (S / 2.0) * k             # support foot x during half-cycle k
    ch, sh = np.cosh(omega * tau), np.sinh(omega * tau)
    xi = xi0 * ch + (xid0 / omega) * sh
    xid = xi0 * omega * sh + xid0 * ch
    return p_k + xi, xid


def build_gait(v, T, z_c, h_clear=0.06, n=201, hip_h=HIP_H):
    """IK with hx = LIPM CoM -> Q/Qd/Qdd + phase tables for the closed loop.

    hip_h: hip joint height above ground. Default = TO09 geometry (0.657,
    knee ~0.7 rad). TO25 raises it toward ~0.68 (leg slack 0.0166 m) to cut
    the single-support knee-torque budget that made the WBC squat-collapse.
    """
    model, data, qpos_adr, dof_adr = setup()
    S = v * T
    L = S / 4.0
    ts = np.linspace(0, T, n, endpoint=False)
    dt = T / n

    X, Xd = np.zeros(n), np.zeros(n)
    Q = np.zeros((n, len(DEFAULT_Q)))
    for i, t in enumerate(ts):
        x, xd = lipm_x(t, v, T, z_c, L)
        X[i], Xd[i] = x - v * t, xd     # X = drift of CoM rel. to linear ramp
        if t < T / 2.0:
            stance_x = L
            toe_x, heel_x = S / 2.0 + L - S, S / 2.0 + L
        else:
            stance_x = S / 2.0 + L
            toe_x, heel_x = L, S + L
        u = (t % (T / 2.0)) / (T / 2.0)
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        swing_x = toe_x + (heel_x - toe_x) * s
        swing_z = ANKLE_H + h_clear * (1.0 - np.cos(2.0 * np.pi * u)) / 2.0
        q = DEFAULT_Q.copy()
        if t < T / 2.0:
            q[LEFT_HIP_PITCH], q[LEFT_KNEE] = ik(stance_x, ANKLE_H, x, hip_h)
            q[RIGHT_HIP_PITCH], q[RIGHT_KNEE] = ik(swing_x, swing_z, x, hip_h)
        else:
            q[RIGHT_HIP_PITCH], q[RIGHT_KNEE] = ik(stance_x, ANKLE_H, x, hip_h)
            q[LEFT_HIP_PITCH], q[LEFT_KNEE] = ik(swing_x, swing_z, x, hip_h)
        # TO22 ankle fix: the gait previously left the ankle at DEFAULT_Q's
        # -0.363, which keeps the foot flat ONLY at the default (hip-over-
        # ankle) pose. Flat foot requires ankle_pitch = -hip_pitch - knee
        # (empirically verified against MuJoCo FK, resid < 0.01 rad), i.e. a
        # ~0.47 rad sweep over the stance -- the missing 0.35 rad tracking
        # error that locked every closed loop since TO11.
        q[LEFT_ANKLE_PITCH] = -q[LEFT_HIP_PITCH] - q[LEFT_KNEE]
        q[RIGHT_ANKLE_PITCH] = -q[RIGHT_HIP_PITCH] - q[RIGHT_KNEE]
        Q[i] = q

    Qd = np.gradient(Q, dt, axis=0)
    Qdd = np.gradient(Qd, dt, axis=0)

    cols = [LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE_PITCH,
            RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE_PITCH]
    tau = np.zeros((n, 6))
    for i in range(n):
        t = ts[i]
        x, xd = lipm_x(t, v, T, z_c, L)
        data.qpos[0] = x
        data.qpos[2] = hip_h + HIP_DZ
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[qpos_adr] = Q[i]
        data.qvel[:] = 0.0
        data.qvel[0] = xd
        data.qvel[dof_adr] = Qd[i]
        mujoco.mj_forward(model, data)
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)
        qacc_full = np.zeros(model.nv)
        qacc_full[dof_adr] = Qdd[i]
        clean = (M @ qacc_full + data.qfrc_bias)[dof_adr] - data.qfrc_constraint[dof_adr]
        tau[i] = [clean[c] for c in cols]
    return X, Xd, Q, Qd, tau


def main(v=0.5, T=0.5, z_c=0.70, kp_scale=1.0, kd_scale=1.0, seconds=10.0, seed=0,
         stab_kp=0.0, stab_kd=0.0, stab_sign=1, h_clear=0.06, ff_scale=1.0,
         ankle_kp_boost=1.0):
    np.random.seed(seed)
    model, data, qpos_adr, dof_adr, act_ids = None, None, None, None, None
    # eval_torque_gait.setup returns 5 outputs; foot_gait_id.setup returns 4
    from eval_torque_gait import setup as setup5
    model, data, qpos_adr, dof_adr, act_ids = setup5()

    X, Xd, Q, Qd, tau = build_gait(v, T, z_c, h_clear)
    n = tau.shape[0]
    phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
    tau_cols = {j: tau[:, k] for k, j in enumerate(SAG)}

    data.qpos[0:3] = [0.0, 0.0, 0.76]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_adr] = Q[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    dt_ctrl = 0.02
    n_steps = int(seconds / dt_ctrl)
    x0 = float(data.qpos[0])
    h_min = float(data.qpos[2])
    fall = None
    com_prev = None
    S = v * T
    L = S / 4.0
    for step in range(n_steps):
        t = step * dt_ctrl
        phi = 2.0 * np.pi * (t % T / T)
        q_gait = np.array([np.interp(phi, phases, Q[:, j]) for j in range(Q.shape[1])])
        qd_gait = np.array([np.interp(phi, phases, Qd[:, j]) for j in range(Qd.shape[1])])
        x_ref = v * t + np.interp(phi, phases, X)
        xd_ref = np.interp(phi, phases, Xd)
        tau_ff = np.array([np.interp(phi, phases, tau_cols[j]) for j in SAG])

        tau_stab = 0.0
        if stab_kp > 0.0 or stab_kd > 0.0:
            mujoco.mj_forward(model, data)
            com_x = float(data.subtree_com[0][0])
            com_vx = 0.0 if com_prev is None else (com_x - com_prev) / dt_ctrl
            tau_stab = stab_sign * (stab_kp * (com_x - x_ref) + stab_kd * (com_vx - xd_ref))
        stance_ankle = LEFT_ANKLE_PITCH if phi < np.pi else RIGHT_ANKLE_PITCH

        for _ in range(4):
            q = data.qpos[qpos_adr]
            qd = data.qvel[dof_adr]
            torque = kp_scale * KP * (q_gait - q) + kd_scale * KD * (qd_gait - qd)
            for j in (LEFT_ANKLE_PITCH, RIGHT_ANKLE_PITCH):
                # ankle is the weakest joint (KP 28.5) yet carries the whole
                # flat-foot alignment; boost its position term selectively
                torque[j] += (ankle_kp_boost - 1.0) * kp_scale * KP[j] * (q_gait[j] - q[j])
            for k, j in enumerate(SAG):
                torque[j] += ff_scale * tau_ff[k]
            torque[stance_ankle] += tau_stab
            torque = np.clip(torque, -EFFORT, EFFORT)
            ctrl = np.zeros(model.nu)
            ctrl[act_ids] = torque
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
        h_min = min(h_min, float(data.qpos[2]))
        if step % 25 == 0 or step == n_steps - 1:
            w, x_q, y_q, z_q = data.qpos[3:7]
            pitch = np.degrees(np.arcsin(np.clip(2.0 * (w * y_q - z_q * x_q), -1, 1)))
            com_x = float(data.subtree_com[0][0])
            com_vx = (com_x - com_prev) / dt_ctrl if com_prev is not None else 0.0
            err = q_gait - q
            order = np.argsort(-np.abs(err))[:3]
            jn = [model.joint(model.actuator_trnid[act_ids[j], 0]).name for j in order]
            print(f"  t={t:5.2f} pitch={pitch:+6.1f}deg com_err={com_x-x_ref:+.3f} "
                  f"com_vx={com_vx:+.2f}(ref {xd_ref:+.2f}) sat={int(np.sum(np.abs(torque) >= EFFORT - 0.5))} h={data.qpos[2]:.3f}",
                  flush=True)
            print(f"    top-err joints: " + ", ".join(
                f"{jn[k]} tgt={q_gait[order[k]]:+.2f} act={q[order[k]]:+.2f}" for k in range(3)),
                flush=True)
        if com_prev is not None:
            pass
        mujoco.mj_forward(model, data)
        com_prev = float(data.subtree_com[0][0])
        if float(data.qpos[2]) < 0.2 or not np.all(np.isfinite(data.qpos)):
            fall = t
            break

    disp = float(data.qpos[0] - x0)
    nn = n_steps if fall is None else max(1, step + 1)
    print(f"=== TO21 LIPM-gait closed loop (v={v} T={T} z_c={z_c} kp={kp_scale} kd={kd_scale} "
          f"stab=({stab_kp},{stab_kd},{stab_sign}) {seconds}s seed={seed}) ===")
    print(f"  fall={fall}  h_min={h_min:.3f}  disp={disp:+.2f}m  vx={disp/(nn*dt_ctrl):+.2f} m/s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--z-c", type=float, default=0.70)
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--kd-scale", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stab-kp", type=float, default=0.0)
    ap.add_argument("--stab-kd", type=float, default=0.0)
    ap.add_argument("--stab-sign", type=int, default=1, choices=[1, -1])
    ap.add_argument("--ff-scale", type=float, default=1.0,
                    help="feedforward ID-torque scale (0 = pure PD gait tracking)")
    ap.add_argument("--ankle-kp-boost", type=float, default=1.0,
                    help="multiply the ankle position gain (weakest joint)")
    a = ap.parse_args()
    main(a.v, a.T, a.z_c, a.kp_scale, a.kd_scale, a.seconds, a.seed,
         a.stab_kp, a.stab_kd, a.stab_sign, ff_scale=a.ff_scale,
         ankle_kp_boost=a.ankle_kp_boost)
