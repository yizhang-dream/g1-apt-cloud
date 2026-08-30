"""NMP torque -> G1 closed-loop smoke test (TO15).

Loads the planar-biped NMP gait (nmp_biped_gait.npz) and applies its hip/knee torque to
the G1's sagittal joints as a phase-indexed feedforward + light PD.  This tests whether
the DYNAMICALLY-CONSISTENT motion (NMP) gives a more stable closed loop than the
kinematic gait (TO11: 0.82 m, 3.58 s).

Sign convention (from TO09): G1 hip_pitch = -theta_h (thigh-forward is NEGATIVE), knee =
theta_k.  The NMP "hip" is theta_h, so G1 hip_pitch torque = -tau_hip; knee is the same.

2D->3D caveat: the NMP's body pitch theta is a free DOF with no ankle; here the G1 body
pitch is handled by PD on the ankle, so the hip/knee feedforward assumes a body pitch the
G1 may not reproduce.  This is the known migration gap; the result is diagnostic.

Run on the SERVER under .venv_mjlab.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")

import mujoco

from foot_gait_id import SCENE, DEFAULT_Q, LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE_PITCH, \
    RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE_PITCH

KP = np.array([
    99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
    99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
    40.17924, 28.50125, 28.50125,
    14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 16.77833, 16.77833,
    14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 16.77833, 16.77833,
], dtype=np.float64)
KD = np.array([
    6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
    6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
    2.55789, 1.81445, 1.81445,
    0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 1.06814, 1.06814,
    0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 1.06814, 1.06814,
], dtype=np.float64)
EFFORT = np.array([
    88, 88, 88, 139, 50, 50,
    88, 88, 88, 139, 50, 50,
    88, 50, 50,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5,
], dtype=np.float64)


def setup():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    qpos_adr, dof_adr, act_ids = [], [], []
    for act_id in range(model.nu):
        jid = model.actuator_trnid[act_id, 0]
        name = model.joint(jid).name
        if "hand" in name:
            continue
        act_ids.append(act_id)
        qpos_adr.append(model.jnt_qposadr[jid])
        dof_adr.append(model.jnt_dofadr[jid])
    return model, data, np.asarray(qpos_adr, int), np.asarray(dof_adr, int), np.asarray(act_ids, int)


def main(npz="/home/cvgluser/ros2_data/apt_g1/outputs/nmp_biped_gait.npz",
         kp_scale=0.5, kd_scale=1.0, seconds=10.0, tau_scale=1.0, sign=(1, 1, 1, 1), seed=0,
         track_gait=False, anchor_dc=True):
    d = np.load(npz)
    q, tau = d["q"], d["tau"]  # q (7,N+1), tau (4,N)
    N = tau.shape[1]
    T = float(d["T"])
    dt = T / N
    model, data, qpos_adr, dof_adr, act_ids = setup()

    # tau columns: [hipL, kneeL, hipR, kneeR] -> G1 sagittal joints
    # G1 hip_pitch torque = -tau_hip ; knee torque = +tau_knee
    phases = np.linspace(0, 2 * np.pi, N, endpoint=False)
    tau_L_hip = -tau[0]; tau_L_knee = tau[1]; tau_R_hip = -tau[2]; tau_R_knee = tau[3]

    # TO19 --track-gait: PD target from the NMP joint trajectory.
    # Planar q rows: [x_c, z_c, th_t, hipL, kneeL, hipR, kneeR]; position
    # mapping mirrors the torque convention (hip_pitch = -th_h, knee = th_k).
    # anchor_dc: re-center each joint's cycle mean onto DEFAULT_Q so the
    # planar model's zero-posture offset does not yank the robot.
    q_gait_prof = {LEFT_HIP_PITCH: -q[3, :N], LEFT_KNEE: q[4, :N],
                   RIGHT_HIP_PITCH: -q[5, :N], RIGHT_KNEE: q[6, :N]}
    if anchor_dc:
        for j, prof in q_gait_prof.items():
            q_gait_prof[j] = prof - prof.mean() + DEFAULT_Q[j]
    qd_gait_prof = {j: np.gradient(prof, dt) for j, prof in q_gait_prof.items()}

    data.qpos[0:3] = [0.0, 0.0, 0.76]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_adr] = DEFAULT_Q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    dt_ctrl = 0.02
    n_steps = int(seconds / dt_ctrl)
    x0 = float(data.qpos[0]); h_min = float(data.qpos[2]); fall = None
    step = 0
    for step in range(n_steps):
        phi = 2.0 * np.pi * ((step * dt_ctrl) % T / T)
        ff = {LEFT_HIP_PITCH: tau_scale * sign[0] * np.interp(phi, phases, tau_L_hip),
              LEFT_KNEE: tau_scale * sign[1] * np.interp(phi, phases, tau_L_knee),
              RIGHT_HIP_PITCH: tau_scale * sign[2] * np.interp(phi, phases, tau_R_hip),
              RIGHT_KNEE: tau_scale * sign[3] * np.interp(phi, phases, tau_R_knee)}
        if track_gait:
            base_q = DEFAULT_Q.copy()
            qd_ff = np.zeros(len(DEFAULT_Q))
            for j, prof in q_gait_prof.items():
                base_q[j] = np.interp(phi, phases, prof)
                qd_ff[j] = np.interp(phi, phases, qd_gait_prof[j])
        else:
            base_q = DEFAULT_Q
            qd_ff = None
        for _ in range(4):
            qv = data.qpos[qpos_adr]
            qd = data.qvel[dof_adr]
            torque = kp_scale * KP * (base_q - qv) - kd_scale * KD * qd
            if qd_ff is not None:
                torque = torque + kd_scale * KD * qd_ff
            for j, v in ff.items():
                torque[j] += v
            torque = np.clip(torque, -EFFORT, EFFORT)
            ctrl = np.zeros(model.nu)
            ctrl[act_ids] = torque
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
        h_min = min(h_min, float(data.qpos[2]))
        if float(data.qpos[2]) < 0.2 or not np.all(np.isfinite(data.qpos)):
            fall = step * dt_ctrl
            break
    disp = float(data.qpos[0] - x0)
    n = n_steps if fall is None else max(1, step + 1)
    print(f"=== {'TO19 NMP gait-tracking' if track_gait else 'TO15 NMP-torque'} closed loop "
          f"(T={T} kp_scale={kp_scale} tau_scale={tau_scale} anchor_dc={anchor_dc} "
          f"sign={sign} {seconds}s seed={seed}) ===")
    print(f"  fall={fall}  h_min={h_min:.3f}  disp={disp:+.2f}m  vx={disp/(n*dt_ctrl):+.2f} m/s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp-scale", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--tau-scale", type=float, default=1.0)
    ap.add_argument("--sign", type=str, default="1,1,1,1")
    ap.add_argument("--track-gait", action="store_true",
                    help="TO19: PD tracks the NMP joint trajectory (not the default stand)")
    ap.add_argument("--no-anchor-dc", action="store_true",
                    help="use the raw planar->G1 joint mapping without DC re-centering")
    a = ap.parse_args()
    sign = tuple(int(x) for x in a.sign.split(","))
    main(kp_scale=a.kp_scale, seconds=a.seconds, tau_scale=a.tau_scale, sign=sign,
         track_gait=a.track_gait, anchor_dc=not a.no_anchor_dc)
