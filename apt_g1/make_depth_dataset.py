"""Generate a local depth -> privileged elevation dataset (P2-lite).

Drives the v9 phase-router walk on the local MuJoCo rough hfield (+-0.06) and,
at every control step, records:

  depth.npy    (N, 48, 64)  ego forward-down depth image (meters, clipped)
  patch.npy    (N, 81)      privileged 9x9 local elevation patch (0.15 m,
                            lookahead 0.6 m, yaw-aligned; heights - root z)
  proprio.npy  (N, 34)      base lin/ang vel, gravity, z rel, 12 joint pos
                            (rel to default) and 12 joint vel (MuJoCo order)

This is the sensor-side of the paper's stage-4 distillation: a student that
regresses the teacher's privileged map from depth + proprio.  Outputs to
``outputs/depth_data/``.
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
from rough_sweep import load_router, feat_for, angle_bin_of

DEV = "cpu"
W, H = 128, 96
FOV = 75.0
DEPTH_MAX = 8.0
STEPS = 1500
SEEDS = [0, 1, 2]
GRID_N = 9
RES = 0.15
LOOKAHEAD = 0.6


def quat_to_yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def sample_patch(rough_h, root_xy, root_z, yaw):
    n = rough_h.shape[0]
    span = 40.0
    res_terr = span / n
    half = GRID_N // 2
    gx = (np.arange(GRID_N) - half) * RES + LOOKAHEAD
    gy = (np.arange(GRID_N) - half) * RES
    gxx, gyy = np.meshgrid(gx, gy, indexing="xy")
    c, s = np.cos(yaw), np.sin(yaw)
    wx = gxx * c - gyy * s
    wy = gxx * s + gyy * c
    qx = root_xy[0] + wx
    qy = root_xy[1] + wy
    fx = np.clip(qx / res_terr + n / 2, 0, n - 1.001)
    fy = np.clip(qy / res_terr + n / 2, 0, n - 1.001)
    x0 = fx.astype(int)
    y0 = fy.astype(int)
    x1 = np.minimum(x0 + 1, n - 1)
    y1 = np.minimum(y0 + 1, n - 1)
    wx_ = fx - x0
    wy_ = fy - y0
    h = (
        rough_h[x0, y0] * (1 - wx_) * (1 - wy_)
        + rough_h[x1, y0] * wx_ * (1 - wy_)
        + rough_h[x0, y1] * (1 - wx_) * wy_
        + rough_h[x1, y1] * wx_ * wy_
    )
    return (h - root_z).astype(np.float32).reshape(-1)


def patch_world_coords(root_xy, yaw):
    half = GRID_N // 2
    gx = (np.arange(GRID_N) - half) * RES + LOOKAHEAD
    gy = (np.arange(GRID_N) - half) * RES
    gxx, gyy = np.meshgrid(gx, gy, indexing="xy")
    c, s = np.cos(yaw), np.sin(yaw)
    return root_xy[0] + gxx * c - gyy * s, root_xy[1] + gxx * s + gyy * c


def unproject_patch(rough_h, depth, cam_pos, fwd, up, root_z, fx, fy, W, H, qx, qy):
    """Geometric upper bound: for each patch cell world point, project it into
    the camera, read the rendered depth, and back-project that ray to a terrain
    height.  Returns (81,) meters relative to root z."""
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    n = rough_h.shape[0]
    res_terr = 40.0 / n
    # project patch points (qx, qy, z=0) into the camera
    p = np.stack([qx, qy, np.zeros_like(qx)], axis=-1) - cam_pos[None, None, :]
    x_cam = (p * right[None, None, :]).sum(-1)
    y_cam = (p * up[None, None, :]).sum(-1)
    z_cam = (p * fwd[None, None, :]).sum(-1)
    z_cam = np.clip(z_cam, 1e-3, None)
    uu = fx * x_cam / z_cam + W / 2
    vv = fy * y_cam / z_cam + H / 2
    ui = np.clip(np.rint(uu).astype(int), 0, W - 1)
    vi = np.clip(np.rint(vv).astype(int), 0, H - 1)
    dd = depth[vi, ui]
    # back-project each depth ray to the terrain
    dxp = (uu - W / 2) / fx
    dyp = (vv - H / 2) / fy
    dirs = fwd[None, None, :] + right[None, None, :] * dxp[..., None] + up[
        None, None, :
    ] * dyp[..., None]
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-12)
    pts = cam_pos[None, None, :] + dirs * dd[..., None]
    fi = np.clip(pts[..., 0] / res_terr + n / 2, 0, n - 1.001)
    fj = np.clip(pts[..., 1] / res_terr + n / 2, 0, n - 1.001)
    i0 = fi.astype(int)
    j0 = fj.astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    j1 = np.minimum(j0 + 1, n - 1)
    wi = fi - i0
    wj = fj - j0
    h = (
        rough_h[i0, j0] * (1 - wi) * (1 - wj)
        + rough_h[i1, j0] * wi * (1 - wj)
        + rough_h[i0, j1] * (1 - wi) * wj
        + rough_h[i1, j1] * wi * wj
    )
    return (h - root_z).astype(np.float32).reshape(-1)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(LOCAL, "outputs", "depth_data"))
    ap.add_argument("--amp", type=float, default=0.06)
    ap.add_argument("--terrain-seed", type=int, default=0)
    cli = ap.parse_args()
    mrx.build(amp=cli.amp, seed=cli.terrain_seed)
    rough_h = np.load(os.path.join(LOCAL, "outputs", "rough_h.npy"))
    modes_list = np.load(os.path.join(LOCAL, "data", "exp_all3", "meta_modes.npy"))
    walk_feat = feat_for(
        dict(mode=2, speed=-1.0, mdir=[1.0, 0.0, 0.0], fdir=[1.0, 0.0, 0.0])
    )
    router = load_router("distill_v9")
    pm, ps, nets, protos, gmap = router
    gi = gmap[(2, -1.0, angle_bin_of(0.0))]
    B = len(protos[gi])

    decoder = NoQuantDecoder(os.path.join(LOCAL, "model_decoder.onnx"))
    env = MujocoG1FlatEnv(
        decoder,
        REPO,
        robot_scene=mrx.OUT,
        use_elastic_band=False,
        stand_only=True,
    )
    env.command = np.zeros(3, dtype=np.float32)
    env.model.vis.global_.fovy = FOV
    renderer = mujoco.Renderer(env.model, height=H, width=W)
    renderer.enable_depth_rendering()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.0
    cam.elevation = 160.0
    cam.azimuth = 0.0
    fovx = 2.0 * np.arctan(np.tan(np.radians(FOV) / 2.0) * W / H)
    fx = (W / 2.0) / np.tan(fovx / 2.0)
    fy = (H / 2.0) / np.tan(np.radians(FOV) / 2.0)
    pelvis_id = env.model.body("pelvis").id
    default_q = env.default_motor_angles

    all_depth, all_patch, all_prop, all_pgeom, all_seed = [], [], [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        env.reset()
        env.data.qpos[2] = 0.76
        env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(
            np.float32
        )
        env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
        mujoco.mj_forward(env.model, env.data)
        env._reset_history()
        env._fill_history_from_state()
        sc_prev = None
        n = 0
        for t in range(STEPS):
            prop = hist_to_proprio(env._get_sonic_history())
            x = np.concatenate([(prop - pm) / ps, walk_feat]).astype(np.float32)
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

            pelvis = env.data.xpos[pelvis_id]
            cam_pos = pelvis + np.array([0.0, 0.0, 0.1])
            look_dir = np.array([np.cos(np.radians(20)), 0.0, -np.sin(np.radians(20))])
            cam.lookat[:] = cam_pos + look_dir
            renderer.update_scene(env.data, camera=cam)
            depth = renderer.render().astype(np.float32)
            depth = np.clip(depth, 0.2, DEPTH_MAX)
            scam = renderer.scene.camera[0]
            cam_pos = scam.pos.copy()
            cam_fwd = scam.forward.copy()
            cam_up = scam.up.copy()

            q = env.data.qpos[3:7]
            yaw = quat_to_yaw(q)
            root_xy = env.data.qpos[:2].copy()
            root_z = float(env.data.qpos[2])
            patch = sample_patch(rough_h, root_xy, root_z, yaw)
            qx, qy = patch_world_coords(root_xy, yaw)
            pgeom = unproject_patch(
                rough_h, depth, cam_pos, cam_fwd, cam_up, root_z, fx, fy, W, H, qx, qy
            )

            qpos = env.data.qpos[env.body_qpos_adr]
            qvel = env.data.qvel[env.body_dof_adr]
            grav = env._get_gravity_dir()
            lin = env._get_base_linear_velocity()
            ang = env._get_base_angular_velocity()
            propv = np.concatenate(
                [
                    lin,
                    ang,
                    grav,
                    [root_z - 0.76],
                    (qpos[:12] - default_q[:12]).astype(np.float32),
                    qvel[:12].astype(np.float32),
                ]
            ).astype(np.float32)
            all_depth.append(depth)
            all_patch.append(patch)
            all_pgeom.append(pgeom)
            all_prop.append(propv)
            all_seed.append(seed)
            n += 1
            if terminated:
                print("seed", seed, "fell at", t, "collected", n, flush=True)
                break
        print("seed", seed, "done, total", len(all_depth), flush=True)

    out = cli.out
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "depth.npy"), np.stack(all_depth))
    np.save(os.path.join(out, "patch.npy"), np.stack(all_patch))
    np.save(os.path.join(out, "patch_geom.npy"), np.stack(all_pgeom))
    np.save(os.path.join(out, "proprio.npy"), np.stack(all_prop))
    np.save(os.path.join(out, "seeds.npy"), np.asarray(all_seed, dtype=np.int64))
    json.dump(
        {
            "n": len(all_depth),
            "depth_shape": [H, W],
            "grid_n": GRID_N,
            "res": RES,
            "lookahead": LOOKAHEAD,
            "patch_units": "meters rel root z",
            "terrain": f"rough amp {cli.amp} seed {cli.terrain_seed}",
            "patch_geom": "depth unprojection upper bound (per-cell ray)",
        },
        open(os.path.join(out, "meta.json"), "w"),
        indent=1,
    )
    print("saved", out, len(all_depth), "frames", flush=True)


if __name__ == "__main__":
    main()
