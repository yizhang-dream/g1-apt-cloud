"""Privileged local elevation-map observation for the APT Isaac env.

The paper's teacher policy receives a local elevation map around the robot.
Isaac Lab's ``HfRandomUniformTerrainCfg`` samples heights from the GLOBAL
``np.random`` state, so we wrap the sub-terrain function to record the exact
height grid that the physics mesh is built from. Sampling that grid at
robot-relative query points yields a privileged terrain observation that is
truthful w.r.t. the simulated ground (same seed -> same grid).
"""
from __future__ import annotations

import math

import numpy as np
import torch


# key: (seed, noise) -> list of (mesh_vertices (K,3) in local meters, hscale)
_ELEV_REGISTRY: dict[tuple, list] = {}
_ELEV_HSCALE: dict[tuple, float] = {}
_ELEV_VSCALE: dict[tuple, float] = {}


def wrap_height_function(sub_cfg, key: tuple):
    """Return a recording wrapper for a sub-terrain height function."""
    orig = sub_cfg.function
    if key not in _ELEV_REGISTRY:
        _ELEV_REGISTRY[key] = []

    def recording(difficulty, cfg):
        # sub_cfg.function is the DECORATED height_field_to_mesh wrapper, so it
        # returns (meshes, origin). The first mesh's vertices are the height
        # grid nodes in local meters (grid starts at (0,0), z relative to the
        # sub-terrain origin).
        meshes, origin = orig(difficulty, cfg)
        verts = np.asarray(meshes[0].vertices, dtype=np.float64)
        _ELEV_REGISTRY[key].append(verts)
        return meshes, origin

    sub_cfg.function = recording
    _ELEV_HSCALE[key] = float(sub_cfg.horizontal_scale)
    _ELEV_VSCALE[key] = float(sub_cfg.vertical_scale)


def build_env_grids(
    terrain, key: tuple, num_envs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env (H_grid, origin_xy): the recorded height grid assigned to each env.

    Each env is placed on the nearest sub-terrain origin. Returns
    grids (N, W, H) in METERS relative to the sub-terrain origin z, and
    origins (N, 2) world xy.
    """
    entries = _ELEV_REGISTRY.get(key, [])
    origins = terrain.terrain_origins.cpu().numpy()  # (rows, cols, 3)
    env_origins = terrain.env_origins.cpu().numpy()
    if len(entries) == 0:
        raise RuntimeError(f"no recorded height grids for key {key}")
    flat_origins = origins.reshape(-1, 3)
    hscale = _ELEV_HSCALE[key]
    vscale = _ELEV_VSCALE[key]
    out_g = []
    out_o = []
    for i in range(num_envs):
        d = np.linalg.norm(flat_origins[:, :2] - env_origins[i, :2], axis=1)
        k = int(np.argmin(d))
        verts = entries[k]
        i_idx = np.rint(verts[:, 0] / hscale).astype(int)
        j_idx = np.rint(verts[:, 1] / hscale).astype(int)
        W = int(verts[:, 0].max() / hscale) + 1
        g = np.zeros((W, W), dtype=np.float32)
        g[i_idx, j_idx] = (verts[:, 2] * vscale).astype(np.float32)
        # the decorator's mesh is centered on the sub-terrain by -size/2 (size
        # = 8.0 in our config), so the world min corner is origin - 4.0.
        world_min = flat_origins[k, :2] - 4.0
        out_g.append(g)
        out_o.append(world_min.astype(np.float32))
    grids_t = torch.from_numpy(np.stack(out_g, 0)).float()
    origins_t = torch.from_numpy(np.stack(out_o, 0)).float()
    return grids_t, origins_t, hscale


def sample_elevation(
    grids_t: torch.Tensor,
    origins_t: torch.Tensor,
    hscale: float,
    root_xy: torch.Tensor,
    root_z: torch.Tensor,
    yaw: torch.Tensor,
    grid_n: int,
    res: float,
    lookahead: float,
) -> torch.Tensor:
    """Sample a (N, grid_n*grid_n) local elevation patch (meters, robot frame).

    Patch is centered ``lookahead`` ahead of the robot, aligned with its yaw.
    Height values are terrain height - root height.
    """
    device = root_xy.device
    n = root_xy.shape[0]
    half = grid_n // 2
    gx = (torch.arange(grid_n, device=device) - half) * res + lookahead
    gy = (torch.arange(grid_n, device=device) - half) * res
    gxx, gyy = torch.meshgrid(gx, gy, indexing="xy")
    gxx = gxx.reshape(1, -1).expand(n, -1)
    gyy = gyy.reshape(1, -1).expand(n, -1)
    cy = torch.cos(yaw).unsqueeze(-1)
    sy = torch.sin(yaw).unsqueeze(-1)
    wx = gxx * cy - gyy * sy
    wy = gxx * sy + gyy * cy
    qx = root_xy[:, 0:1] + wx
    qy = root_xy[:, 1:2] + wy
    W = grids_t.shape[1]
    fx = (qx - origins_t[:, 0:1]) / hscale
    fy = (qy - origins_t[:, 1:2]) / hscale
    fx = fx.clamp(0, W - 1.001)
    fy = fy.clamp(0, W - 1.001)
    x0 = fx.floor().long()
    y0 = fy.floor().long()
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=W - 1)
    tx = (fx - x0.float())
    ty = (fy - y0.float())
    flat = grids_t.reshape(n, -1)
    h00 = flat.gather(1, x0 * W + y0)
    h10 = flat.gather(1, x1 * W + y0)
    h01 = flat.gather(1, x0 * W + y1)
    h11 = flat.gather(1, x1 * W + y1)
    h = (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty
    return (h - root_z.unsqueeze(-1)).reshape(n, -1)
