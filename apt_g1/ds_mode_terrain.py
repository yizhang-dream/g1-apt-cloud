"""DS mode x terrain runner (D030): all 21 non-static SONIC LocomotionModes on
mjlab-official paper-spec terrains, in-memory MjSpec assembly (no XML files).

Parameterized fork of planner_closed_loop.py (MQ09/11 lineage: same obs
assembly / 10Hz replan / metrics). Changes:
  - MODE = all 21 non-static modes (static IDLE/SQUAT/KNEEL/LYING/BOXING
    excluded: no forward motion, pass-rate meaningless)
  - terrain built in-memory: MjSpec.from_file(scene) -> drop floor plane ->
    mjlab official subterrain functions (paper params) OR self-built boxes
    (hurdle/gap, labeled) -> spec.compile() -> MjModel injected into env via
    from_xml_path patch (to_xml cannot serialize embedded hfield/textures)
  - one JSON result line per run for matrix aggregation

Run on lab-ts (.venv_mjlab):
  python ds_mode_terrain.py --mode 2 --terrain rough_mid --seed 0 --n-steps 2250
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
SCENE = f"{REPO}/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"

BODY_IDX = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]
MODES = {
    "slow_walk": 1, "walk": 2, "run": 3, "crawling": 8,
    "walk_boxing": 10, "left_punch": 11, "right_punch": 12,
    "random_punch": 13, "elbow_crawling": 14, "left_hook": 15, "right_hook": 16,
    "forward_jump": 17, "stealth_walk": 18, "injured_walk": 19,
    "ledge_walking": 20, "object_carrying": 21, "stealth_walk_2": 22,
    "happy_dance_walk": 23, "zombie_walk": 24, "gun_walk": 25, "scare_walk": 26,
}

TERRAIN_SPAWN = {  # name -> (spawn_x, spawn_z)
    "flat": (-6.0, 0.85), "rough_mid": (-6.0, 0.95),
    "stairs_mid": (-6.0, 0.85), "stones_mid": (-6.0, 0.95),
    "discrete_mid": (-6.0, 0.95), "highstep_mid": (-6.0, 0.85),
    "hurdle_mid": (-6.0, 0.85), "gap_mid": (-6.0, 0.85),
    "hurdle_h10": (-6.0, 0.85), "hurdle_h20": (-6.0, 0.85),
    "hurdle_h30": (-6.0, 0.85),
    "bar_h10": (-6.0, 0.85), "bar_h20": (-6.0, 0.85), "bar_h30": (-6.0, 0.85),
}


def _qn(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _rotmat(q):
    w, x, y, z = _qn(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def build_model(terrain: str, seed: int):
    """Assemble scene+terrain fully in memory; returns a compiled MjModel."""
    import mujoco
    from mjlab.terrains.primitive_terrains import (
        BoxPyramidStairsTerrainCfg, BoxSteppingStonesTerrainCfg)
    from mjlab.terrains.heightfield_terrains import (
        HfRandomUniformTerrainCfg, HfDiscreteObstaclesTerrainCfg)

    rng = np.random.default_rng(seed)
    spec = mujoco.MjSpec.from_file(SCENE)
    world = spec.worldbody
    for g in list(world.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            world.geoms.remove(g)
    world.add_body(name="terrain")
    size = (16.0, 16.0)

    if terrain == "rough_mid":  # paper shape: symmetric, 0.2m coarse grid
        cfg = HfRandomUniformTerrainCfg(
            size=size, noise_range=(0.0, 0.08), noise_step=0.01,
            downsampled_scale=0.2, horizontal_scale=0.1,
            vertical_scale=0.005, border_width=1.0)
        cfg.function(difficulty=0.5, spec=spec, rng=rng)
        model = spec.compile()
        h = model.hfield_data.reshape(model.hfield_nrow[0], model.hfield_ncol[0])
        h -= (h.max() + h.min()) / 2.0  # symmetric +/-0.04: pits included (G0)
        return model
    if terrain == "stairs_mid":  # paper: width 0.3, height 0.05-0.315 -> mid ~0.18
        cfg = BoxPyramidStairsTerrainCfg(
            size=size, step_height_range=(0.05, 0.315), step_width=0.3,
            platform_width=1.0, border_width=1.0)
        cfg.function(difficulty=0.5, spec=spec, rng=rng)
        return spec.compile()
    if terrain == "stones_mid":  # paper: stones 0.4-0.48, stone h 0-0.18
        cfg = BoxSteppingStonesTerrainCfg(
            size=size, stone_size_range=(0.4, 0.48),
            stone_distance_range=(0.2, 0.5), stone_height=0.09,
            stone_height_variation=0.09, stone_size_variation=0.05,
            floor_depth=0.36, displacement_range=0.1,
            platform_width=1.0, border_width=0.25)
        cfg.function(difficulty=0.5, spec=spec, rng=rng)
        return spec.compile()
    if terrain == "discrete_mid":  # paper: block h 0-0.16, size 0.2-0.8
        cfg = HfDiscreteObstaclesTerrainCfg(
            size=size, obstacle_height_range=(0.0, 0.16),
            obstacle_width_range=(0.2, 0.8), num_obstacles=40,
            obstacle_height_mode="choice", platform_width=1.0,
            border_width=1.0, horizontal_scale=0.1, vertical_scale=0.005)
        cfg.function(difficulty=0.5, spec=spec, rng=rng)
        return spec.compile()

    tb = spec.body("terrain")
    if terrain == "highstep_mid":  # self-built (paper params): 1m-thick, 0.45-high wall
        tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 8.0, 0.225],
                    pos=[0.0, 0.0, 0.225], rgba=[0.4, 0.5, 0.6, 1.0])
    elif terrain == "hurdle_mid":  # self-built (paper params): 0.4 bars, 1.5m apart
        for i in range(8):
            tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 4.0, 0.2],
                        pos=[-5.0 + 1.5 * i, 0.0, 0.2], rgba=[0.4, 0.5, 0.6, 1.0])
    elif terrain == "gap_mid":  # self-built (paper params): 0.8m chasm at x=0
        # slabs leave a true 0.8m gap: [-(9.6+0.4), -0.4] and [0.4, +9.6]
        tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[4.8, 20.0, 0.25],
                    pos=[-5.2, 0.0, -0.25], rgba=[0.5, 0.45, 0.4, 1.0])
        tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[4.8, 20.0, 0.25],
                    pos=[5.2, 0.0, -0.25], rgba=[0.5, 0.45, 0.4, 1.0])
    elif terrain.startswith("hurdle_h"):  # height sweep: h10/h20/h30 = 0.1/0.2/0.3m
        hz = float(terrain[len("hurdle_h"):]) / 100.0
        for i in range(8):
            tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 4.0, hz],
                        pos=[-5.0 + 1.5 * i, 0.0, hz], rgba=[0.4, 0.5, 0.6, 1.0])
    elif terrain.startswith("bar_h"):  # SINGLE bar at x=0 (phase-diagnosis D031)
        hz = float(terrain[len("bar_h"):]) / 100.0
        tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 4.0, hz],
                    pos=[0.0, 0.0, hz], rgba=[0.7, 0.4, 0.3, 1.0])
    else:
        raise ValueError(f"unknown terrain {terrain}")
    return spec.compile()


def run(mode_id: int, terrain: str, spawn_x: float, spawn_z: float,
        n_steps: int, seed: int):
    sys.path.insert(0, "/home/cvgluser/ros2_data")
    sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
    sys.path.insert(0, REPO)
    import mujoco
    import onnxruntime as ort
    from apt_g1.envs.mujoco_g1_flat_env import (
        MujocoG1FlatEnv, SONIC_DEFAULT_ANGLES_MUJOCO, G1_MUJOCO_TO_ISAACLAB_DOF)
    from eval_distill import NoQuantDecoder

    scene_arg = SCENE
    if terrain != "flat":
        model = build_model(terrain, seed)
        _built = {SCENE: model}
        _orig = mujoco.MjModel.from_xml_path

        def _patched(p, *a, **k):
            key = str(p)
            return _built[key] if key in _built else _orig(p, *a, **k)
        mujoco.MjModel.from_xml_path = staticmethod(_patched)

    try:
        env = MujocoG1FlatEnv(NoQuantDecoder(f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"),
                              REPO, robot_scene=scene_arg,
                              use_elastic_band=False, stand_only=True)
    finally:
        if terrain != "flat":
            mujoco.MjModel.from_xml_path = staticmethod(_orig)

    planner = ort.InferenceSession(
        f"{REPO}/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
        providers=["CPUExecutionProvider"])
    enc = ort.InferenceSession(
        f"{REPO}/gear_sonic_deploy/policy/release/model_encoder.onnx",
        providers=["CPUExecutionProvider"])
    iname = enc.get_inputs()[0].name
    m2i = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)
    fk_data = mujoco.MjData(env.model)

    def live_q36():
        q = np.zeros(36, dtype=np.float32)
        q[0:3] = env.data.qpos[0:3]
        q[3:7] = env.data.qpos[3:7]
        q[7:36] = env.data.qpos[env.body_qpos_adr]
        return q

    def fk(q36):
        q = np.zeros(env.model.nq)
        q[:3] = q36[:3]; q[3:7] = q36[3:7]; q[env.body_qpos_adr] = q36[7:36]
        fk_data.qpos[:] = q
        mujoco.mj_forward(env.model, fk_data)
        return fk_data.xquat[BODY_IDX].copy()

    def plan():
        ctx = np.tile(live_q36()[None, None], (1, 4, 1)).astype(np.float32)
        inp = {"context_mujoco_qpos": ctx,
               "target_vel": np.array([-1.0], dtype=np.float32),
               "mode": np.array([mode_id], dtype=np.int64),
               "movement_direction": np.array([[1.0, 0, 0]], dtype=np.float32),
               "facing_direction": np.array([[1.0, 0, 0]], dtype=np.float32),
               "random_seed": np.array([seed], dtype=np.int64),
               "has_specific_target": np.array([[0]], dtype=np.int64),
               "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
               "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
               "allowed_pred_num_tokens": np.ones((1, 11), dtype=np.int64),
               "height": np.array([-1.0], dtype=np.float32)}
        qpos_out, nframes_out = planner.run(None, inp)
        traj = qpos_out[0][:int(nframes_out[0])]
        jp = traj[:, 7:36][:, m2i]
        jv = np.vstack([np.zeros((1, 29)), np.diff(jp, axis=0) * 50.0])
        bq = np.array([fk(t) for t in traj])
        ad = _qn(_qmul(np.array([1.0, 0, 0, 0]), _qconj(_qn(bq[0, 0]))))
        return jp, jv, bq, ad

    env.reset()
    env.data.qpos[0] = spawn_x
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = spawn_z
    for _ in range(40):
        env._step_physics(SONIC_DEFAULT_ANGLES_MUJOCO.copy())

    jp = jv = bq = ad = None
    cur_frame = 0
    term = False
    xs, hs = [], []

    for step in range(n_steps):
        if step % 5 == 0:
            jp, jv, bq, ad = plan()
            cur_frame = 0
        live = env.data.qpos[3:7].astype(np.float64)
        obs = np.zeros(1762, dtype=np.float32); obs[0] = 0.0; p = 4
        for f in range(10):
            idx = min(cur_frame + f * 5, len(jp) - 1); obs[p:p + 29] = jp[idx]; p += 29
        for f in range(10):
            idx = min(cur_frame + f * 5, len(jv) - 1); obs[p:p + 29] = jv[idx]; p += 29
        p += 17
        for f in range(10):
            idx = min(cur_frame + f * 5, len(bq) - 1)
            nr = _qn(_qmul(ad, bq[idx, 0]))
            btr = _qn(_qmul(_qconj(live), nr))
            rot = _rotmat(btr); obs[p:p + 6] = rot[:, :2].flatten(); p += 6
        x_pre = float(env.data.qpos[0]); h_pre = float(env.data.qpos[2])
        tok = enc.run(None, {iname: obs[None]})[0][0].astype(np.float32)
        _, _, term, _ = env.step({"token": tok, "aux": np.zeros(12, dtype=np.float32)})
        cur_frame += 1
        xs.append(x_pre); hs.append(h_pre)
        if term:
            break

    h = np.asarray(hs)
    return dict(fall="term" if term else None, n=len(h),
                x0=round(xs[0], 2), x_end=round(xs[-1], 2),
                adv=round(xs[-1] - xs[0], 2),
                h_min=round(float(h.min()), 2), h_end=round(float(h[-1]), 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, help="mode id 1-26 or name")
    ap.add_argument("--terrain", default="flat",
                    choices=list(TERRAIN_SPAWN.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spawn-x", type=float, default=None,
                    help="override terrain default spawn")
    ap.add_argument("--spawn-z", type=float, default=None)
    ap.add_argument("--n-steps", type=int, default=2250)  # 45 s @ 50 Hz
    args = ap.parse_args()
    mode_id = int(args.mode) if args.mode.isdigit() else MODES[args.mode]
    sx, sz = TERRAIN_SPAWN[args.terrain]
    if args.spawn_x is not None: sx = args.spawn_x
    if args.spawn_z is not None: sz = args.spawn_z
    r = run(mode_id, args.terrain, sx, sz, args.n_steps, args.seed)
    r.update(mode=mode_id, terrain=args.terrain, seed=args.seed, spawn_x=sx)
    print("DSMT " + json.dumps(r))
