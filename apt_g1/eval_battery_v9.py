"""Battery v9: closed-loop eval of a per-group phase router directory.

Same harness/metrics as eval_battery_v6.py, but the router dir is a CLI arg
(default outputs/distill_v9). Runs the standard scenarios plus the walk
direction bins that exp3 was collected to cover.
"""
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
ap.add_argument("--odir", default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_v9")
ap.add_argument("--tag", default="v9")
ap.add_argument("--steps", type=int, default=1000)
cli = ap.parse_args()


class PhaseNet(nn.Module):
    def __init__(self, d_in, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


D = "/home/cvgluser/ros2_data/apt_g1/data/exp_all3"
cmd = np.load(D + "/cmd.npy")
modes_list = np.load(D + "/meta_modes.npy")
odir = cli.odir
norm = np.load(odir + "/phase_norm.npz")
meta = json.load(open(odir + "/phase_meta.json"))
pmean = norm["pmean"].ravel()
pstd = norm["pstd"].ravel()
nets = {}
protos = {}
for gi, md in meta.items():
    if gi.startswith("_"):
        continue
    gi = int(gi)
    net = PhaseNet(930 + cmd.shape[1]).cuda()
    net.load_state_dict(torch.load(f"{odir}/phase_g{gi}.pt", map_location="cuda"))
    net.eval()
    nets[gi] = net
    protos[gi] = np.load(f"{odir}/proto_g{gi}.npy")
gmap = {tuple(md["group"]): int(gi) for gi, md in meta.items() if not gi.startswith("_")}


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


def run(c, gi, steps, seed, ema=0.3):
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
    sc_prev = None
    B = len(protos[gi])
    for t in range(steps):
        prop = hist_to_proprio(env._get_sonic_history())
        x = np.concatenate([(prop - pmean) / pstd, feat_for(c)]).astype(np.float32)
        with torch.no_grad():
            sc = nets[gi](torch.from_numpy(x[None]).cuda())[0].cpu().numpy().astype(
                np.float32
            )
        if ema > 0 and sc_prev is not None:
            sc = ema * sc_prev + (1 - ema) * sc
        sc_prev = sc
        phi = float(np.arctan2(sc[0], sc[1]))
        b = int(np.floor((phi + np.pi) / (2 * np.pi) * B) % B)
        tok = protos[gi][b]
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
        h_mean=round(float(heights.mean()), 3),
        h_min=round(float(heights.min()), 3),
        vx=round(float(vxs.mean()), 3),
        vy=round(float(vys.mean()), 3),
        speed=round(float(spd.mean()), 3),
        path=round(float(spd.sum() * 0.02), 2),
    )


scen = [
    ("idle", 0, -1.0, 0.0),
    ("slow_fwd", 1, 0.2, 0.0),
    ("slow_back", 1, 0.2, np.pi),
    ("walk_fwd", 2, -1.0, 0.0),
    ("walk_back", 2, -1.0, np.pi),
    ("jump", 17, -1.0, 0.0),
    ("turn_right", 1, 0.2, np.pi / 3),
    ("turn_left", 1, 0.2, -np.pi / 3),
    ("strafe_right", 1, 0.2, np.pi / 2),
    ("strafe_left", 1, 0.2, -np.pi / 2),
    ("stealth", 18, -1.0, -np.pi / 6),
    # walk direction bins (exp3 coverage targets)
    ("walk_bin1", 2, -1.0, -3 * np.pi / 8),
    ("walk_bin2", 2, -1.0, -np.pi / 2),
    ("walk_bin3", 2, -1.0, -np.pi / 8),
    ("walk_bin5", 2, -1.0, np.pi / 8),
    ("walk_bin6", 2, -1.0, np.pi / 2),
    ("walk_bin7", 2, -1.0, 3 * np.pi / 8),
    # speed interpolation (continuous-speed coverage)
    ("slow_speed35", 1, 0.35, 0.0),
]
steps = cli.steps
out = {}
for name, m, sp, a in scen:
    gi = gmap.get((int(m), round(float(sp), 2), angle_bin_of(a)))
    if gi is None:
        print(f"{name:16s} NO GROUP", flush=True)
        out[name] = {"note": "no router"}
        continue
    mdir = [float(np.cos(a)), float(np.sin(a)), 0.0]
    c = dict(mode=m, speed=sp, mdir=mdir, fdir=mdir)
    out[name] = {}
    for seed in [0, 1, 2]:
        r = run(c, gi, steps, seed)
        out[name][f"seed{seed}"] = r
        print(
            f"{name:16s} seed{seed} done={r['completed']} fall={r['fall_step']} "
            f"h_min={r['h_min']} vx={r['vx']} vy={r['vy']} path={r['path']}",
            flush=True,
        )
json.dump(out, open(f"{odir}/eval_battery_{cli.tag}.json", "w"), indent=1)
print("saved", f"{odir}/eval_battery_{cli.tag}.json")
