"""Correct-baseline test: SRB TO torque + PD(zero-token q_des) on the G1.

Baseline (aux=0, no SRB): PD tracks q_des = SONIC_decoder(zero_token) -- the
stable stand pose the A-ID line used (which stood, vx~0.03).
Question: does adding the self-consistent SRB TO torque as feedforward make the
G1 WALK (vs the ID-torque 站住, and vs the SRB-torque-on-default-pose collapse)?

tau = sign * tau_scale * tau_SRB(phase) + kp*(q_des_zero - q) - kd*qdot,
applied to the sagittal leg joints (hip_pitch/knee/ankle_pitch x2).  The gait
phase is a fixed 2 Hz clock (T=0.5 s).
"""

from __future__ import annotations

import sys
import numpy as np
import mujoco

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")

from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from apt_g1.sonic.sonic_wrapper import SonicOnnxDecoder

from srb_to import solve, grf_amp, M, I, G, N
from srb_to_torque import ik, knee_pos, L_HEEL, L_TOE

ONNX = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"

LEFT_SAG = [0, 3, 4]     # left  hip_pitch, knee, ankle_pitch
RIGHT_SAG = [6, 9, 10]   # right hip_pitch, knee, ankle_pitch


class NoQuantDecoder(SonicOnnxDecoder):
    def decode(self, token, history):
        obs = self.build_decoder_obs(token, history)
        return self.session.run([self.output_name], {self.input_name: obs})[0]


def build_torque_table(v, T, d):
    r = solve(v, T=T, d=d)
    z0, zd0, th0, thd0 = r["z0"], r["zd0"], r["th0"], r["thd0"]
    S = v * T
    L = S * d / 4.0
    A = grf_amp(d)
    state = np.array([0.0, z0, th0, zd0, thd0])
    hs = d * T / 2.0 / N
    phis, taus = [], []
    for foot_x, t0 in [(L, 0.0), (S / 2.0 + L, T / 2.0)]:
        for k in range(N + 1):
            t = t0 + k * hs
            x, z, th, zd, thd = state
            Fz = A * np.sin(2 * np.pi * (t - t0) / (d * T))
            th_h, th_k = ik(foot_x, 0.0, x, z)
            kx, kz = knee_pos(th_h, x, z)
            tau_h = Fz * (foot_x - x)
            tau_k = Fz * (foot_x - kx)
            u = (t - t0) / (d * T / 2.0)
            tau_a = Fz * (-L_HEEL + (L_HEEL + L_TOE) * u)
            phis.append(2 * np.pi * t / T)
            taus.append([tau_h, tau_k, tau_a])
            zdd = Fz / M - G
            thdd = (foot_x - x) * Fz / I
            state = state + hs * np.array([v, zd, thd, zdd, thdd])
        if d < 1.0:
            hf = (1.0 - d) * T / 2.0 / N
            for k in range(N + 1):
                x, z, th, zd, thd = state
                state = state + hf * np.array([v, zd, thd, -G, 0.0])
    return np.array(phis), np.array(taus)


class TorqueSrbEnv(MujocoG1FlatEnv):
    def __init__(self, decoder, repo, v, T, d, tau_scale=1.0, sign=(-1, -1, -1), **kw):
        super().__init__(decoder, repo, robot_scene=kw.pop("robot_scene", None), **kw)
        self.phis, self.taus = build_torque_table(v, T, d)
        self.T = T
        self.tau_scale = tau_scale
        self.sign = np.asarray(sign, dtype=np.float32)

    def _srb_feedforward(self):
        phi = 2 * np.pi * ((self.step_count * 0.02) % self.T / self.T)
        tau = np.array([np.interp(phi, self.phis, self.taus[:, j]) for j in range(3)])
        tau12 = np.zeros(12, dtype=np.float32)
        sag = RIGHT_SAG if phi >= np.pi else LEFT_SAG
        for i, idx in enumerate(sag):
            tau12[idx] = self.sign[i] * self.tau_scale * tau[i]
        return tau12

    def _step_physics(self, q_des):
        tau_ff = self._srb_feedforward()
        for _ in range(self.control_decimation):
            if self.use_elastic_band:
                self._apply_elastic_band()
            qpos, qvel = self._get_body_state()
            torque = (
                self.kp[: self.num_body] * (q_des - qpos)
                - self.kd[: self.num_body] * qvel
            )
            torque[:12] += tau_ff
            torque = np.clip(torque, -self.effort_limit[: self.num_body],
                             self.effort_limit[: self.num_body])
            ctrl = np.zeros(self.model.nu, dtype=np.float32)
            ctrl[self.body_act_ids] = torque
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)


def run(v, T, d, tau_scale, sign, steps=1000, seed=0):
    decoder = NoQuantDecoder(ONNX)
    env = TorqueSrbEnv(
        decoder, REPO, v, T, d, tau_scale=tau_scale, sign=sign,
        robot_scene="gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml",
        stand_only=True, use_elastic_band=False,
    )
    env.reset()
    x0 = float(env.data.qpos[0])
    h_min = float(env.data.qpos[2])
    x_last = x0
    fall = None
    for t in range(env.episode_length):
        x_before = float(env.data.qpos[0])
        h_before = float(env.data.qpos[2])
        h_min = min(h_min, h_before)
        obs, reward, terminated, info = env.step(
            {"token": np.zeros(64, dtype=np.float32), "aux": np.zeros(12, dtype=np.float32)}
        )
        x_last = x_before
        if terminated:
            fall = t
            break
    disp = x_last - x0
    n = env.episode_length if fall is None else max(1, fall)
    return dict(fall=fall, disp=round(disp, 3), h_min=round(h_min, 3),
                vx_est=round(disp / (n * 0.02), 3))


if __name__ == "__main__":
    for v in [1.0, 2.0, 3.0]:
        phis, taus = build_torque_table(v, 0.5, 1.0)
        print(f"SRB torque ranges (v={v} walk): hip [{taus[:,0].min():.1f},{taus[:,0].max():.1f}] "
              f"knee [{taus[:,1].min():.1f},{taus[:,1].max():.1f}] "
              f"ankle [{taus[:,2].min():.1f},{taus[:,2].max():.1f}] Nm", flush=True)
        for tau_scale, sign in [(1.0, (-1, -1, -1)), (1.0, (1, 1, 1))]:
            r = run(v, 0.5, 1.0, tau_scale, sign)
            print(f"v={v} scale={tau_scale} sign={sign}: {r}", flush=True)
