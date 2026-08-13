"""MuJoCo rough-terrain router eval + video (local)."""
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
ROUGH = os.path.join(LOCAL, "outputs", "rough_h.npy")
rough_h = np.load(ROUGH)


def load_router(odir_name):
    odir = os.path.join(LOCAL, "outputs", odir_name)
    norm = np.load(os.path.join(odir, "phase_norm.npz"))
    meta = json.load(open(os.path.join(odir, "phase_meta.json")))
    nets, protos = {}, {}
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
    gmap = {
        tuple(md["group"]): int(gi)
        for gi, md in meta.items()
        if not gi.startswith("_")
    }
    return (
        norm["pmean"].ravel(),
        norm["pstd"].ravel(),
        nets,
        protos,
        gmap,
    )


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


def terrain_z(x, y):
    n = rough_h.shape[0]
    res = 40.0 / n
    i = int(np.clip(round(x / res + n / 2), 0, n - 1))
    j = int(np.clip(round(y / res + n / 2), 0, n - 1))
    return float(rough_h[i, j])


decoder = NoQuantDecoder(os.path.join(LOCAL, "model_decoder.onnx"))
rough_xml = os.path.join(REPO, "gear_sonic/data/robot_model/model_data/g1/scene_43dof_rough.xml")
env = MujocoG1FlatEnv(
    decoder,
    REPO,
    robot_scene=rough_xml,
    use_elastic_band=False,
    stand_only=True,
)
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


def run_one(tag, pmean, pstd, nets, protos, gmap, m, sp, a, steps, render=True):
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
    env.data.qpos[2] = terrain_z(0.0, 0.0) + 0.76
    env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(
        np.float32
    )
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco.mj_forward(env.model, env.data)
    env._reset_history()
    env._fill_history_from_state()
    sc_prev = None
    B = len(protos[gi])
    frames = []
    fall = None
    heights = []
    for t in range(steps):
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
        heights.append(float(env.data.qpos[2]))
        if terminated:
            fall = t
            break
        if render and t % 2 == 0:
            renderer.update_scene(env.data, camera=cam)
            frame = renderer.render()
            img = Image.fromarray(frame)
            d = ImageDraw.Draw(img)
            d.text(
                (30, 24),
                f"{tag} t={t/50:.1f}s",
                fill=(255, 220, 80),
                font=font,
            )
            frames.append(np.asarray(img))
    return dict(fall=fall, h_min=round(float(min(heights)), 3)), frames


pmean9, pstd9, nets9, protos9, gmap9 = load_router("distill_v9")
pmean6, pstd6, nets6, protos6, gmap6 = load_router("distill_final")

results = {}
writer = None
for tag, pm, ps, nets, protos, gmap in [
    ("v9 rough walk", pmean9, pstd9, nets9, protos9, gmap9),
    ("v6 rough walk", pmean6, pstd6, nets6, protos6, gmap6),
]:
    r, frames = run_one(
        tag, pm, ps, nets, protos, gmap, 2, -1.0, 0.0, 1200
    )
    results[tag] = r
    print(tag, r, flush=True)
    out = os.path.join(LOCAL, "outputs", f"rough_{tag.split()[0]}.mp4")
    writer = imageio.get_writer(out, fps=25, codec="libx264", quality=7, pixelformat="yuv420p")
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    print("video", out, len(frames), "frames", flush=True)

json.dump(results, open(os.path.join(LOCAL, "outputs", "rough_mujoco.json"), "w"), indent=1)
