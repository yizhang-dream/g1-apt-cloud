"""Perception-distillation demo (paper stage 4 mechanism).

Teacher: the privileged local elevation patch (9x9 @0.15m, robot frame).
Student: reconstructs the teacher patch from a coarse+noisy "perception"
proxy (3x3 block mean + Gaussian noise) plus the base state -- the analog of
the paper's CNN/GRU student regressing the teacher's exteroceptive latent from
depth/LIDAR. MSE regression, the same loss family as the paper's DAgger step.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ODIR = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_percept"
os.makedirs(ODIR, exist_ok=True)

# ---- collect (fine_patch, coarse_noisy, base_state) pairs from the env ------
from isaaclab.app import AppLauncher

launcher_parser = __import__("argparse").ArgumentParser()
AppLauncher.add_app_launcher_args(launcher_parser)
launcher_args, _ = launcher_parser.parse_known_args()
launcher_args.num_envs = 1
launcher_args.headless = True
launcher_args.enable_cameras = False
launcher_args.env_spacing = 4.0
launcher_args.output_dir = ODIR
app_launcher = AppLauncher(launcher_args)
simulation_app = app_launcher.app

import numpy as np
import torch

from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
from apt_g1.isaac.terrain_cfg import make_terrain_importer_cfg

cfg = AptFlatG1EnvCfg()
cfg.scene.num_envs = 1
cfg.terrain = make_terrain_importer_cfg("rough", 0.08, seed=0)
cfg.router_model_dir = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_final"
cfg.sonic_decoder_path = (
    "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
    "gear_sonic_deploy/policy/release/model_decoder.onnx"
)
cfg.use_elevation = True
cfg.observation_space += cfg.elev_grid * cfg.elev_grid
np.random.seed(0)
torch.manual_seed(0)
env = AptFlatG1Env(cfg)

N_SAMPLES = 3000
fine_list, coarse_list, state_list = [], [], []
obs, _ = env.reset()
env._last_obs = obs["policy"]
rng = np.random.default_rng(0)
for i in range(N_SAMPLES):
    # random-ish poses: walk command for motion, small random teleports
    if i % 50 == 0:
        obs, _ = env.reset()
    action = torch.zeros(1, 14, dtype=torch.float32, device=env.device)
    obs, rew, term, trunc, _ = env.step(action)
    env._last_obs = obs["policy"]
    if term.any():
        obs, _ = env.reset()
    elev = obs["policy"][0, -81:].detach().cpu().numpy().astype(np.float32)
    fine = elev.reshape(9, 9)
    # coarse 3x3 block mean
    coarse = fine.reshape(3, 3, 3, 3).mean(axis=(1, 3))
    coarse = coarse + rng.normal(0, 0.04, coarse.shape).astype(np.float32)
    base = obs["policy"][0, :9].detach().cpu().numpy().astype(np.float32)
    fine_list.append(fine.ravel())
    coarse_list.append(coarse.ravel())
    state_list.append(base)
    if (i + 1) % 500 == 0:
        print("collected", i + 1, flush=True)

X = np.concatenate([np.stack(coarse_list), np.stack(state_list)], axis=1).astype(
    np.float32
)
Y = np.stack(fine_list).astype(np.float32)
print("X", X.shape, "Y", Y.shape, "elev std", float(Y.std()))


class Student(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, d_out),
        )

    def forward(self, x):
        return self.net(x)


torch.manual_seed(0)
n = len(X)
idx = np.random.RandomState(0).permutation(n)
ntr = int(n * 0.8)
tr, va = idx[:ntr], idx[ntr:]
ds = TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(Y[tr]))
ld = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)
net = Student(X.shape[1], Y.shape[1]).cuda()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
lossf = nn.MSELoss()
xv = torch.from_numpy(X[va]).cuda()
yv = torch.from_numpy(Y[va]).cuda()
best = 1e9
for ep in range(60):
    net.train()
    tl, tb = 0.0, 0
    for xb, yb in ld:
        xb = xb.cuda(non_blocking=True)
        yb = yb.cuda(non_blocking=True)
        opt.zero_grad()
        loss = lossf(net(xb), yb)
        loss.backward()
        opt.step()
        tl += loss.item() * len(yb)
        tb += len(yb)
    net.eval()
    with torch.no_grad():
        err = float(lossf(net(xv), yv).item())
    if err < best:
        best = err
        torch.save(net.state_dict(), f"{ODIR}/student.pt")
    if (ep + 1) % 10 == 0:
        print(f"ep {ep+1} loss {tl/tb:.5f} val {err:.5f}", flush=True)

# eval reconstruction quality at several noise levels
with torch.no_grad():
    pred = net(xv).cpu().numpy()
true = yv.cpu().numpy()
mae = float(np.abs(pred - true).mean())
corr = float(np.corrcoef(pred.ravel(), true.ravel())[0, 1])
print("best val MSE", best, "MAE", mae, "corr", round(corr, 4), flush=True)
json.dump(
    {
        "best_val_mse": best,
        "val_mae": mae,
        "val_corr": corr,
        "noise_std": 0.04,
        "note": "student: coarse noisy 3x3 + base state -> privileged 9x9 patch",
    },
    open(f"{ODIR}/percept_meta.json", "w"),
    indent=1,
)
os._exit(0)
