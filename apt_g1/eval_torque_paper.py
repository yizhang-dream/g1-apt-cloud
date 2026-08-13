"""MuJoCo closed-loop eval of paper-style torque control (phase+cmd -> torque).

Control law (paper hybrid scheme, aux=0 for now):

    tau = tau_dec(phase, cmd) + kp*(q_default - q) - kd*qdot

against the current baseline (token q_des path):

    tau = kp*(q_des_token - q) - kd*qdot

Scenarios: idle / walk_fwd on flat and on the local rough hfield (+-0.06),
3 seeds x 1000 control steps.  Validates whether the recovered torque decoder
is a drop-in replacement for the token path before the Isaac RL test.
"""

from __future__ import annotations

import io
import json
import os
import sys
import argparse

import numpy as np
import torch

LOCAL = r"C:\Users\zyz\Documents\gr00t\apt_g1"
REPO = r"D:\GR00T-WholeBodyControl"
sys.path.insert(0, os.path.dirname(LOCAL))
sys.path.insert(0, LOCAL)
sys.path.insert(0, REPO)

import mujoco

import make_rough_xml as mrx
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from apt_g1.eval_distill import NoQuantDecoder, hist_to_proprio
from rough_sweep import load_router, feat_for

DEV = "cpu"
STEPS = 1000
SEEDS = [0, 1, 2]


class TorquePaperEnv(MujocoG1FlatEnv):
    """Paper-style control: tau_dec(phase, cmd) + PD(q_default - q)."""

    def __init__(
        self,
        decoder,
        repo,
        torque_model,
        tau_mean,
        tau_std,
        router,
        walk_feat,
        robot_scene=None,
        hybrid: bool = False,
        tau_scale: float = 1.0,
        **kw,
    ):
        super().__init__(decoder, repo, robot_scene=robot_scene, **kw)
        self.torque_model = torque_model
        self.tau_mean = tau_mean
        self.tau_std = tau_std
        self.router = router
        self.walk_feat = walk_feat
        self.sc_prev = None
        self.hybrid = hybrid
        self.tau_scale = tau_scale

    def _torque_ff(self):
        prop = hist_to_proprio(self._get_sonic_history())
        pm, ps, nets, protos, gmap = self.router
        x = np.concatenate([(prop - pm) / ps, self.walk_feat]).astype(np.float32)
        gi = gmap[(2, -1.0, 4)]  # angle 0 rad -> bin 4 (forward walk)
        with torch.no_grad():
            sc = nets[gi](torch.from_numpy(x[None]))[0].numpy().astype(np.float32)
        if self.sc_prev is not None:
            sc = 0.3 * self.sc_prev + 0.7 * sc
        self.sc_prev = sc
        inp = torch.from_numpy(np.concatenate([sc, self.walk_feat])[None]).float()
        with torch.no_grad():
            tau_n = self.torque_model(inp)[0].numpy().astype(np.float32)
        return tau_n * self.tau_std + self.tau_mean

    def _step_physics(self, q_des):
        tau_ff = self._torque_ff()  # 12-d, MuJoCo order
        for _ in range(self.control_decimation):
            if self.use_elastic_band:
                self._apply_elastic_band()
            qpos, qvel = self._get_body_state()
            torque = (
                self.kp[: self.num_body] * (q_des - qpos)
                - self.kd[: self.num_body] * qvel
            )
            # paper scheme: decoder torque + PD; hybrid: token PD + decoder torque
            torque[:12] += self.tau_scale * tau_ff
            torque = np.clip(
                torque,
                -self.effort_limit[: self.num_body],
                self.effort_limit[: self.num_body],
            )
            ctrl = np.zeros(self.model.nu, dtype=np.float32)
            ctrl[self.body_act_ids] = torque
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)


def make_env(
    amp, paper, torque_model, tau_mean, tau_std, router, walk_feat,
    hybrid=False, tau_scale=1.0,
):
    decoder = NoQuantDecoder(os.path.join(LOCAL, "model_decoder.onnx"))
    if amp <= 0.0:
        scene = os.path.join(
            REPO, "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
        )
    else:
        mrx.build(amp=amp, seed=0)
        scene = mrx.OUT
    if paper:
        env = TorquePaperEnv(
            decoder,
            REPO,
            torque_model,
            tau_mean,
            tau_std,
            router,
            walk_feat,
            robot_scene=scene,
            hybrid=hybrid,
            tau_scale=tau_scale,
            use_elastic_band=False,
            stand_only=True,
        )
    else:
        env = MujocoG1FlatEnv(
            decoder,
            REPO,
            robot_scene=scene,
            use_elastic_band=False,
            stand_only=True,
        )
    env.command = np.zeros(3, dtype=np.float32)
    return env


def terrain_z(amp, x, y):
    if amp <= 0.0:
        return 0.0
    h = np.load(os.path.join(LOCAL, "outputs", "rough_h.npy"))
    n = h.shape[0]
    res = 40.0 / n
    i = int(np.clip(round(x / res + n / 2), 0, n - 1))
    j = int(np.clip(round(y / res + n / 2), 0, n - 1))
    return float(h[i, j])


def run(env, amp, mode, seed, steps):
    rng = np.random.default_rng(seed)
    env.reset()
    env.data.qpos[2] = terrain_z(amp, 0.0, 0.0) + 0.76
    env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(
        np.float32
    )
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco.mj_forward(env.model, env.data)
    env._reset_history()
    env._fill_history_from_state()
    x0 = float(env.data.qpos[0])
    y0 = float(env.data.qpos[1])
    xs, ys, heights = [], [], []
    fall = None
    env.sc_prev = None
    for t in range(steps):
        xs.append(float(env.data.qpos[0]))
        ys.append(float(env.data.qpos[1]))
        if isinstance(env, TorquePaperEnv) and not env.hybrid:
            obs, reward, terminated, info = env.step(
                {"token": np.zeros(64, dtype=np.float32), "aux": np.zeros(12)}
            )
        else:
            # baseline / hybrid: v9 router token for walk_fwd group
            pm, ps, nets, protos, gmap = ROUTER
            prop = hist_to_proprio(env._get_sonic_history())
            if mode == "idle":
                tok = np.zeros(64, dtype=np.float32)
            else:
                gi = gmap[(2, -1.0, 4)]  # angle 0 rad -> bin 4 (forward walk)
                x = np.concatenate([(prop - pm) / ps, WALK_FEAT]).astype(np.float32)
                with torch.no_grad():
                    sc = nets[gi](torch.from_numpy(x[None]))[0].numpy().astype(np.float32)
                prev = getattr(env, "sc_prev", None)
                if prev is not None:
                    sc = 0.3 * prev + 0.7 * sc
                env.sc_prev = sc
                phi = float(np.arctan2(sc[0], sc[1]))
                B = len(protos[gi])
                b = int(np.floor((phi + np.pi) / (2 * np.pi) * B) % B)
                tok = protos[gi][b]
            obs, reward, terminated, info = env.step(
                {"token": tok, "aux": np.zeros(12)}
            )
        heights.append(float(env.data.qpos[2]))
        if terminated:
            fall = t
            break
    n_steps = steps if fall is None else fall
    if fall is None:
        dx = float(env.data.qpos[0] - x0)
        dy = float(env.data.qpos[1] - y0)
    else:
        dx = float(xs[fall] - x0)
        dy = float(ys[fall] - y0)
    return {
        "fall": fall,
        "h_min": round(float(min(heights)), 3),
        "disp": round(float(np.hypot(dx, dy)), 2),
        "vx_est": round(float(np.hypot(dx, dy) / max(1e-6, n_steps * 0.02)), 3),
    }


ROUTER = None
WALK_FEAT = None


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--decoder-dir",
        default=os.path.join(LOCAL, "outputs", "torque_decoder_v9"),
    )
    cli = ap.parse_args()
    global ROUTER, WALK_FEAT
    ROUTER = load_router("distill_v9")
    modes_list = np.load(os.path.join(LOCAL, "data", "exp_all3", "meta_modes.npy"))
    WALK_FEAT = feat_for(
        dict(mode=2, speed=-1.0, mdir=[1.0, 0.0, 0.0], fdir=[1.0, 0.0, 0.0])
    )
    td = cli.decoder_dir
    meta = json.load(open(os.path.join(td, "meta.json")))
    tau_mean = np.asarray(meta["tau_mean"], dtype=np.float32)
    tau_std = np.asarray(meta["tau_std"], dtype=np.float32)
    from train_torque_decoder import TorqueDecoder

    tm = TorqueDecoder()
    tm.load_state_dict(torch.load(os.path.join(td, "model.pt"), map_location=DEV))
    tm.eval()

    results = {}
    for amp in [0.0, 0.06]:
        for paper in [False, True]:
            env = make_env(amp, paper, tm, tau_mean, tau_std, ROUTER, WALK_FEAT)
            tag = f"amp{amp}_{'paper' if paper else 'baseline'}"
            results[tag] = {}
            modes = ["walk"] if paper else ["idle", "walk"]
            for mode in modes:
                for seed in SEEDS:
                    r = run(env, amp, mode, seed, STEPS)
                    results[tag][f"{mode}_s{seed}"] = r
                    print(tag, mode, seed, r, flush=True)
        # hybrid: token path + torque feedforward
        env = make_env(amp, True, tm, tau_mean, tau_std, ROUTER, WALK_FEAT, hybrid=True)
        tag = f"amp{amp}_hybrid"
        results[tag] = {}
        for mode in ["walk"]:
            for seed in SEEDS:
                r = run(env, amp, mode, seed, STEPS)
                results[tag][f"{mode}_s{seed}"] = r
                print(tag, mode, seed, r, flush=True)
    json.dump(
        results,
        open(os.path.join(LOCAL, "outputs", "torque_paper_eval.json"), "w"),
        indent=1,
    )
    print("saved outputs/torque_paper_eval.json")


if __name__ == "__main__":
    main()
