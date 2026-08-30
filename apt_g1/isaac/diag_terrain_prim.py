"""One-shot diagnostic: inspect the USD terrain prim structure under /World/ground."""
import os
import numpy as np
import torch
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
launcher_args, _ = parser.parse_known_args()
launcher_args.num_envs = 1
launcher_args.headless = True
launcher_args.enable_cameras = False
app_launcher = AppLauncher(launcher_args)
simulation_app = app_launcher.app

from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
from apt_g1.isaac.terrain_cfg import make_terrain_importer_cfg

cfg = AptFlatG1EnvCfg()
cfg.scene.num_envs = 1
cfg.terrain = make_terrain_importer_cfg("rough", 0.06, seed=0)
cfg.router_model_dir = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_final"
cfg.sonic_decoder_path = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
env = AptFlatG1Env(cfg)

from pxr import UsdGeom

stage = simulation_app.context.get_stage()
print("DIAG: ground prim:", stage.GetPrimAtPath("/World/ground").IsValid())
children = stage.GetPrimAtPath("/World/ground").GetChildren()
print("DIAG: children:", [(str(p.GetPath()), p.GetTypeName()) for p in children])
for p in children:
    if p.GetTypeName() == "Mesh":
        m = UsdGeom.Mesh(p)
        pts = m.GetPointsAttr().Get()
        print("DIAG: mesh", str(p.GetPath()), "points:", None if pts is None else len(pts))
        if pts is not None and len(pts) > 0:
            a = np.asarray(pts, dtype=np.float32)
            print("DIAG: pts shape", a.shape, "z range", float(a[:, 2].min()), float(a[:, 2].max()))
o = env.scene.env_origins[0].detach().cpu().numpy()
print("DIAG: env_origins[0]", o)
print("DIAG DONE")
os._exit(0)
