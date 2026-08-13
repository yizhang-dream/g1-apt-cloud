"""Closed-loop MuJoCo battery for the TVAE-lite token router."""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from eval_distill import NoQuantDecoder, hist_to_proprio

ap = argparse.ArgumentParser()
ap.add_argument("--ema", type=float, default=0.3)
ap.add_argument("--z-dim", type=int, default=16)
cli = ap.parse_args()


class Encoder(nn.Module):
    def __init__(self, d_in, z):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1),
        )
        self.mu = nn.Linear(512, z)
        self.logvar = nn.Linear(512, z)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, d_cmd, z):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z + d_cmd, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 64),
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=-1))


D = "/home/cvgluser/ros2_data/apt_g1/data/exp_all3"
cmd = np.load(D + "/cmd.npy")
modes_list = np.load(D + "/meta_modes.npy")
ODIR = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_vae"
norm = np.load(ODIR + "/norm.npz")
pmean = norm["pmean"].ravel()
pstd = norm["pstd"].ravel()
enc = Encoder(930 + cmd.shape[1], cli.z_dim).cuda()
dec = Decoder(cmd.shape[1], cli.z_dim).cuda()
enc.load_state_dict(torch.load(f"{ODIR}/encoder.pt", map_location="cuda"))
dec.load_state_dict(torch.load(f"{ODIR}/decoder.pt", map_location="cuda"))
enc.eval()
dec.eval()


def angle_bin_of(a):
    return int(np.floor((a + np.pi) / (2 * np.pi) * 8)) % 8


def feat_for(c):
    oh = np.zeros(len(modes_list), dtype=np.float32)
    oh[int(np.where(modes_list == int(c["mode"]))[0][0])] = 1
    return np.concatenate(
        [
            oh,
            np.array(c["mdir"], dtype=np.float32),
            np.array(c["fdir"], dtype=np.float32),
            np.array([c["speed"], -1.0, 1.0], dtype=np.float32),
        ]
    ).astype(np.float32)


env = MujocoG1FlatEnv(
    NoQuantDecoder(
        "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
    ),
    "/home/cvgluser/ros2_data/GR00T-WholeBodyControl",
    use_elastic_band=False,
    stand_only=True,
)
env.command = np.zeros(3, dtype=np.float32)


def run(c, steps, seed, ema):
    import mujoco

    env.reset()
    rng = np.random.default_rng(seed)
    env.data.qpos[2] += float(rng.normal(0, 0.005))
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(
        np.float32
    )
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco.mj_forward(env.model, env.data)
    env._reset_history()
    env._fill_history_from_state()
    heights, vxs, vys = [], [], []
    fall = None
    z_prev = None
    f = feat_for(c).astype(np.float32)
    for t in range(steps):
        prop = hist_to_proprio(env._get_sonic_history())
        x = np.concatenate([(prop - pmean) / pstd, f]).astype(np.float32)
        with torch.no_grad():
            mu, _ = enc(torch.from_numpy(x[None]).cuda())
            z = mu[0].cpu().numpy().astype(np.float32)
        if ema > 0 and z_prev is not None:
            z = ema * z_prev + (1 - ema) * z
        z_prev = z
        with torch.no_grad():
            tok = (
                dec(
                    torch.from_numpy(z[None]).cuda(),
                    torch.from_numpy(f[None]).cuda(),
                )[0]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        tok = np.clip(np.round(tok * 16) / 16, -1, 1).astype(np.float32)
        obs, reward, terminated, info = env.step(
            {"token": tok, "aux": np.zeros(12, dtype=np.float32)}
        )
        v = env._get_base_linear_velocity()
        vxs.append(float(v[0]))
        vys.append(float(v[1]))
        heights.append(float(env.data.qpos[2]))
        if terminated:
            fall = t
            break
    vxs = np.array(vxs)
    vys = np.array(vys)
    heights = np.array(heights)
    spd = np.sqrt(vxs**2 + vys**2)
    return dict(
        fall_step=fall,
        completed=fall is not None and fall >= steps - 1,
        h_min=round(float(heights.min()), 3),
        vx=round(float(vxs.mean()), 3),
        vy=round(float(vys.mean()), 3),
        path=round(float(spd.sum() * 0.02), 2),
    )


scen = [
    ("idle", 0, -1.0, 0.0, 1000),
    ("slow_fwd", 1, 0.2, 0.0, 1000),
    ("slow_back", 1, 0.2, np.pi, 1000),
    ("walk_fwd", 2, -1.0, 0.0, 1000),
    ("walk_back", 2, -1.0, np.pi, 1000),
    ("jump", 17, -1.0, 0.0, 1000),
    ("turn_right", 1, 0.2, np.pi / 3, 1000),
    ("turn_left", 1, 0.2, -np.pi / 3, 1000),
    ("strafe_right", 1, 0.2, np.pi / 2, 1000),
    ("strafe_left", 1, 0.2, -np.pi / 2, 1000),
    ("stealth", 18, -1.0, -np.pi / 6, 1000),
    ("slow_speed35", 1, 0.35, 0.0, 1000),
]

out = {}
for name, m, sp, a, steps in scen:
    mdir = [float(np.cos(a)), float(np.sin(a)), 0.0]
    c = dict(mode=m, speed=sp, mdir=mdir, fdir=mdir)
    out[name] = {}
    for seed in [0, 1, 2]:
        r = run(c, steps, seed, cli.ema)
        out[name][f"seed{seed}"] = r
        print(
            f"{name:16s} seed{seed} done={r['completed']} fall={r['fall_step']} "
            f"h_min={r['h_min']} vx={r['vx']} vy={r['vy']} path={r['path']}",
            flush=True,
        )
json.dump(out, open(f"{ODIR}/eval_vae_lite_ema{cli.ema}.json", "w"), indent=1)
print("saved", f"{ODIR}/eval_vae_lite_ema{cli.ema}.json")
