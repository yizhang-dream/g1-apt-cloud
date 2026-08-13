"""Local (Windows) v9 router highlight reel, CPU inference."""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

LOCAL = r"C:\Users\zyz\Documents\gr00t\apt_g1"
REPO = r"D:\GR00T-WholeBodyControl"
sys.path.insert(0, os.path.dirname(LOCAL))
sys.path.insert(0, LOCAL)
sys.path.insert(0, REPO)

from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from apt_g1.eval_distill import NoQuantDecoder, hist_to_proprio

DEV = "cpu"


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


D = os.path.join(LOCAL, "data", "exp_all3")
cmd = np.load(os.path.join(D, "cmd.npy"))
modes_list = np.load(os.path.join(D, "meta_modes.npy"))
odir = os.path.join(LOCAL, "outputs", "distill_v9")
norm = np.load(os.path.join(odir, "phase_norm.npz"))
meta = json.load(open(os.path.join(odir, "phase_meta.json")))
pmean = norm["pmean"].ravel()
pstd = norm["pstd"].ravel()
nets = {}
protos = {}
for gi, md in meta.items():
    if gi.startswith("_"):
        continue
    gi = int(gi)
    net = PhaseNet(930 + cmd.shape[1])
    net.load_state_dict(
        torch.load(os.path.join(odir, f"phase_g{gi}.pt"), map_location=DEV)
    )
    net.eval()
    nets[gi] = net
    protos[gi] = np.load(os.path.join(odir, f"proto_g{gi}.npy"))
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


decoder = NoQuantDecoder(os.path.join(LOCAL, "model_decoder.onnx"))
env = MujocoG1FlatEnv(decoder, REPO, use_elastic_band=False, stand_only=True)
env.command = np.zeros(3, dtype=np.float32)

W, H = 1280, 720
env.model.vis.global_.offwidth = W
env.model.vis.global_.offheight = H
renderer = mujoco.Renderer(env.model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = env.model.body("pelvis").id
cam.distance = 3.4
cam.azimuth = 100
cam.elevation = -18
cam.lookat[:] = [0, 0, 0.8]

try:
    font = ImageFont.truetype("arial.ttf", 34)
except Exception:
    font = ImageFont.load_default()

schedule = [
    ("idle", 3),
    ("walk_fwd", 6),
    ("walk_back", 5),
    ("walk_bin5", 6),
    ("walk_bin3", 6),
    ("walk_bin7", 5),
    ("jump", 4),
    ("slow_fwd", 4),
    ("idle", 3),
]
scen = {
    "idle": (0, -1.0, 0.0),
    "walk_fwd": (2, -1.0, 0.0),
    "walk_back": (2, -1.0, np.pi),
    "walk_bin5": (2, -1.0, np.pi / 8),
    "walk_bin3": (2, -1.0, -np.pi / 8),
    "walk_bin7": (2, -1.0, 3 * np.pi / 8),
    "jump": (17, -1.0, 0.0),
    "slow_fwd": (1, 0.2, 0.0),
}
labels = {
    "idle": "IDLE",
    "walk_fwd": "WALK FWD 0deg",
    "walk_back": "WALK BACK 180deg",
    "walk_bin5": "WALK +45deg (new data)",
    "walk_bin3": "WALK -45deg (new data)",
    "walk_bin7": "WALK +135deg (new data)",
    "jump": "FORWARD JUMP",
    "slow_fwd": "SLOW WALK",
}

out = os.path.join(odir, "v9_reel_local.mp4")
writer = imageio.get_writer(out, fps=50, codec="libx264", quality=7, pixelformat="yuv420p")
total = 0
for name, secs in schedule:
    m, sp, a = scen[name]
    gi = gmap[(int(m), round(float(sp), 2), angle_bin_of(a))]
    c = dict(
        mode=m,
        speed=sp,
        mdir=[float(np.cos(a)), float(np.sin(a)), 0.0],
        fdir=[float(np.cos(a)), float(np.sin(a)), 0.0],
    )
    feat = feat_for(c)
    rng = np.random.default_rng(0)
    env.reset()
    env.data.qpos[2] += float(rng.normal(0, 0.005))
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(
        np.float32
    )
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco.mj_forward(env.model, env.data)
    env._reset_history()
    env._fill_history_from_state()
    sc_prev = None
    B = len(protos[gi])
    for t in range(int(secs * 50)):
        prop = hist_to_proprio(env._get_sonic_history())
        x = np.concatenate([(prop - pmean) / pstd, feat]).astype(np.float32)
        with torch.no_grad():
            sc = nets[gi](torch.from_numpy(x[None]))[0].numpy().astype(np.float32)
        if sc_prev is not None:
            sc = 0.3 * sc_prev + 0.7 * sc
        sc_prev = sc
        phi = float(np.arctan2(sc[0], sc[1]))
        b = int(np.floor((phi + np.pi) / (2 * np.pi) * B) % B)
        obs, reward, terminated, info = env.step(
            {"token": protos[gi][b], "aux": np.zeros(12, dtype=np.float32)}
        )
        renderer.update_scene(env.data, camera=cam)
        frame = renderer.render()
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        d.text((30, 24), labels[name], fill=(255, 220, 80), font=font)
        writer.append_data(np.asarray(img))
        total += 1
        if terminated:
            print("fall in", name, "at step", t, flush=True)
            break
    print("phase", name, "done, frames", total, flush=True)
writer.close()
print("saved", out, total, "frames", flush=True)
