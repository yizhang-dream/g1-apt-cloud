"""Replay an Isaac rollout (npz from rollout_log_joints.py) in MuJoCo and render
a REAL 3D G1 model video, offscreen (EGL, no display server needed).

The npz holds per-step base pose + 29 joint angles (SONIC/IsaacLab order) logged
from the Isaac RL rollout; here we set the same state in the MuJoCo G1 model and
render camera frames with mujoco.Renderer, then encode an mp4 with the bundled
imageio-ffmpeg.

E41+: if the npz carries the Isaac heightfield ("heights" + "tile_origin"), a
MuJoCo hfield ground matching the terrain tile under the robot is injected into
the scene (replacing the flat floor), so rough-terrain rollouts render faithfully.

Run on the server with .venv_mjlab and MUJOCO_GL=egl:
    MUJOCO_GL=egl .venv_mjlab/bin/python replay_render_mujoco.py \
        --npz outputs/e39_rollout.npz --out outputs/e39_mujoco.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# G1 ordering constants (same as envs/mujoco_g1_flat_env.py).
G1_ISAACLAB_TO_MUJOCO_DOF = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
    2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16,
    20, 22, 24, 26, 28,
]

MODEL_PATH = (
    "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
    "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
)

H_SCALE = 0.1  # Isaac horizontal_scale (m per heightmap cell)
TILE_HALF = 4.0  # half tile size (8x8 m tiles)


def extract_tile(H: np.ndarray, origin: np.ndarray, flip: str) -> np.ndarray:
    """Extract the 8x8 m tile centered at `origin` from the Isaac heightmap.

    Isaac convention: heights[i][j] sits at world
      x = (j - (NC-1)/2) * H_SCALE ,  y = ((NR-1)/2 - i) * H_SCALE
    (terrain centered at world origin, row 0 on the +y side).
    `flip` fixes data-orientation mismatches before returning.
    """
    NR, NC = H.shape
    ox, oy = float(origin[0]), float(origin[1])
    j0 = int(round((ox - TILE_HALF) / H_SCALE + (NC - 1) / 2.0))
    j1 = int(round((ox + TILE_HALF) / H_SCALE + (NC - 1) / 2.0))
    i0 = int(round((NR - 1) / 2.0 - (oy + TILE_HALF) / H_SCALE))
    i1 = int(round((NR - 1) / 2.0 - (oy - TILE_HALF) / H_SCALE))
    i0, i1 = max(0, i0), min(NR - 1, i1)
    j0, j1 = max(0, j0), min(NC - 1, j1)
    tile = H[i0:i1 + 1, j0:j1 + 1]
    if flip == "ud":
        tile = np.flipud(tile)
    elif flip == "lr":
        tile = np.fliplr(tile)
    elif flip == "t":
        tile = tile.T
    return np.ascontiguousarray(tile, dtype=np.float32)


def build_model(xml_str: str, tile: np.ndarray, origin: np.ndarray) -> "mujoco.MjModel":
    import mujoco  # local import so --help works without mujoco

    nrow, ncol = tile.shape
    hx = (ncol - 1) * H_SCALE / 2.0 + H_SCALE / 2.0  # half-width incl half cell
    hy = (nrow - 1) * H_SCALE / 2.0 + H_SCALE / 2.0  # half-depth incl half cell
    hmax = float(max(abs(tile.min()), abs(tile.max()))) + 0.02
    data = " ".join(f"{v:.3f}" for v in tile.ravel())
    hfield = (
        f'<hfield name="rough" size="{hx:.3f} {hy:.3f} {hmax:.3f} {hmax:.3f}" '
        f'nrow="{nrow}" ncol="{ncol}">{data}</hfield>'
    )
    if "</asset>" not in xml_str:
        raise ValueError("no </asset> found in scene xml")
    xml_str = xml_str.replace("</asset>", hfield + "</asset>")
    geom = (
        f'<geom name="rough_ground" type="hfield" hfield="rough" '
        f'pos="{float(origin[0]):.3f} {float(origin[1]):.3f} 0" '
        f'euler="0 0 0" friction="1.5"/>'
    )
    # drop the flat floor geom, insert our hfield geom after <worldbody>
    lines = []
    for ln in xml_str.splitlines():
        if 'name="floor"' in ln:
            continue
        lines.append(ln)
    xml_str = "\n".join(lines)
    xml_str = xml_str.replace(
        "<!-- Site -->", f"    {geom}\n    <!-- Site -->", 1
    )
    return mujoco.MjModel.from_xml_string(xml_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="outputs/replay.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=640)  # GLContext framebuffer cap
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--view", choices=["side", "threequarter"], default="threequarter")
    ap.add_argument("--cam-dist", type=float, default=2.6)
    ap.add_argument("--title", default="")
    # E41 terrain rendering knobs
    ap.add_argument("--hfield-flip", choices=["none", "ud", "lr", "t"], default="none")
    ap.add_argument("--test-frame", type=int, default=-1,
                    help="render only this frame to a PNG for alignment checks")
    args = ap.parse_args()

    import mujoco

    d = np.load(args.npz)
    base = d["base_xyz"]
    quat = d["base_quat"]
    jp = d["joint_pos"]  # (N, 29) SONIC order
    N = len(base)
    fell_at = int(d["fell_at"])
    print(f"[replay] {N} steps, fell_at={fell_at}, "
          f"mean_vx={(base[-1,0]-base[0,0])/(N/50.0):.3f} m/s")

    has_heights = "heights" in d.files and "tile_origin" in d.files
    tile = None
    origin = None
    if has_heights and d["heights"] is not None:
        tile = np.asarray(d["heights"], dtype=np.float32)  # window grid (rows=y, cols=x)
        min_corner = np.asarray(d["tile_origin"], dtype=np.float32)
        nrow, ncol = tile.shape
        # grid extents (0.1 m cells), window min corner -> center + half extents
        cx = float(min_corner[0]) + (ncol - 1) * H_SCALE / 2.0
        cy = float(min_corner[1]) + (nrow - 1) * H_SCALE / 2.0
        origin = np.array([cx, cy, 0.0], dtype=np.float32)
        if args.hfield_flip == "ud":
            tile = np.flipud(tile)
        elif args.hfield_flip == "lr":
            tile = np.fliplr(tile)
        elif args.hfield_flip == "t":
            tile = tile.T
        print(f"[replay] heightfield grid {tile.shape} "
              f"(hmin {tile.min():.3f} hmax {tile.max():.3f}) "
              f"center ({cx:.2f}, {cy:.2f})")

    xml_str = Path(MODEL_PATH).read_text()
    # from_xml_string resolves <include> relative to cwd -> run from the model dir
    os.chdir(Path(MODEL_PATH).parent)
    if tile is not None:
        model = build_model(xml_str, tile, origin)
        print("[replay] injected hfield ground into scene")
    else:
        model = mujoco.MjModel.from_xml_string(xml_str)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # joint qpos addresses for the 29 body joints (skipping hands), in MuJoCo order
    body_qpos_adr = []
    for act_id in range(model.nu):
        jid = model.actuator_trnid[act_id, 0]
        if "hand" in model.joint(jid).name:
            continue
        body_qpos_adr.append(model.jnt_qposadr[jid])
    body_qpos_adr = np.asarray(body_qpos_adr, dtype=int)
    isaac_to_mujoco = np.asarray(G1_ISAACLAB_TO_MUJOCO_DOF, dtype=int)

    # camera setup (follow the robot)
    cam = mujoco.MjvCamera()
    cam.distance = args.cam_dist
    cam.elevation = -12.0
    cam.azimuth = 90.0 if args.view == "side" else 135.0
    cam.lookat = base[0].copy()

    renderer = mujoco.Renderer(model, args.height, args.width)
    if args.test_frame >= 0:
        i = min(args.test_frame, N - 1)
        data.qpos[0:3] = base[i]
        data.qpos[3:7] = quat[i]
        data.qpos[body_qpos_adr] = jp[i][isaac_to_mujoco]
        mujoco.mj_forward(model, data)
        # numeric alignment check: contacts with the ground geom + lowest geoms
        bz = base[i]
        for gid in range(model.ngeom):
            if model.geom(gid).type == mujoco.mjtGeom.mjGEOM_HFIELD:
                print(f"[replay] hfield geom {gid} name={model.geom(gid).name} "
                      f"xpos={data.geom_xpos[gid]} size={model.geom(gid).size}")
        low = sorted(range(model.ngeom), key=lambda g: data.geom_xpos[g][2])[:4]
        for g in low:
            print(f"[replay] low geom {g} name={model.geom(g).name} "
                  f"xpos={data.geom_xpos[g]} z={data.geom_xpos[g][2]:.3f}")
        ncon = 0
        for c in range(data.ncon):
            con = data.contact[c]
            if con.geom1 == 0 or con.geom2 == 0:
                ncon += 1
                if ncon <= 6:
                    print(f"[replay] ground contact geom1={con.geom1} geom2={con.geom2} "
                          f"dist={con.dist:.4f}")
        print(f"[replay] total ground contacts: {ncon}")
        for tag, px, py in [("center", 1.65, 0.0),
                            ("base", bz[0], bz[1]),
                            ("far", bz[0] - 5.0, bz[1])]:
            dist = mujoco.mj_ray(
                model, data, np.array([px, py, bz[2] + 0.5], dtype=np.float64),
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
                geomgroup=None, flg_static=True, bodyexclude=0, geomid=None,
            )
            hit = bz[2] + 0.5 - (dist if dist >= 0 else float("nan"))
            print(f"[replay] ray {tag}: hit={hit:.3f} expected_ground={bz[2] - 0.76:.3f} "
                  f"diff={hit - (bz[2] - 0.76):+.3f}")
        cam.lookat = base[i]
        renderer.update_scene(data, cam)
        frame = renderer.render().copy()
        try:
            renderer.close()
        except Exception:  # noqa: BLE001  (EGL teardown quirk; png already written)
            pass
        png = Path(str(args.out).replace(".mp4", f"_frame{i}.png"))
        import imageio.v2 as iio
        iio.imwrite(png, frame)
        print(f"[replay] test frame -> {png}")
        sys.stdout.flush()
        os._exit(0)

    frames = []
    for i in range(N):
        # free joint: base pose (MuJoCo free qpos = xyz + wxyz quat)
        data.qpos[0:3] = base[i]
        data.qpos[3:7] = quat[i]
        # body joints: SONIC order -> MuJoCo order -> qpos addresses
        data.qpos[body_qpos_adr] = jp[i][isaac_to_mujoco]
        mujoco.mj_forward(model, data)
        cam.lookat = base[i]
        renderer.update_scene(data, cam)
        frames.append(renderer.render().copy())
    try:
        renderer.close()
    except Exception:  # noqa: BLE001
        print("[replay] warn: renderer.close raised (EGL teardown), continuing")
    print(f"[replay] rendered {len(frames)} frames {frames[0].shape}")

    # encode mp4 with bundled ffmpeg
    import imageio.v2 as iio

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(
        str(out), fps=args.fps, codec="libx264", quality=8,
        macro_block_size=2, format="FFMPEG",
    ) as w:
        for f in frames:
            w.append_data(f)
    print(f"[replay] wrote {out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
