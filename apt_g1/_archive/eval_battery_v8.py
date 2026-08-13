"""Battery v8: v6 (per-group fallback) vs v8 (continuous conditional router)
on the existing scenarios plus UNSEEN command scenarios (speed interpolation
and walk directions with no training data).

Run on the MuJoCo venv (same harness as eval_battery_v6.py).
"""
import json
import sys

RUN_SWITCH = "--switch-only" in sys.argv

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from eval_distill import NoQuantDecoder, hist_to_proprio


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=512, drop=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


D = "/home/cvgluser/ros2_data/apt_g1/data/exp_all"
cmd = np.load(D + "/cmd.npy")
modes_list = np.load(D + "/meta_modes.npy")
token = np.load(D + "/token.npy")
mode = np.load(D + "/mode.npy")
speed = np.load(D + "/speed.npy")
ab = np.load(D + "/angle_bin.npy")
ODIR6 = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_v6"
ODIR8 = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_v8"
ODIR8C = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_v8c"
norm = np.load(ODIR6 + "/phase_norm.npz")
meta6 = json.load(open(ODIR6 + "/phase_meta.json"))
pmean = norm["pmean"].ravel()
pstd = norm["pstd"].ravel()

nets6 = {}
protos = {}
for gi, md in meta6.items():
    gi = int(gi)
    net = MLP(930 + cmd.shape[1], 2).cuda()
    net.load_state_dict(torch.load(f"{ODIR6}/phase_g{gi}.pt", map_location="cuda"))
    net.eval()
    nets6[gi] = net
    protos[gi] = np.load(f"{ODIR6}/proto_g{gi}.npy")
gmap = {tuple(md["group"]): int(gi) for gi, md in meta6.items()}


def select_group_fallback(m, sp, b):
    """Mirror PhaseRouterEncoder.select_group: exact -> same mode+bin nearest
    speed -> same mode any group."""
    key = (int(m), round(float(sp), 2), int(b))
    if key in gmap:
        return gmap[key]
    cands = [(g, gi) for g, gi in gmap.items() if g[0] == key[0] and g[2] == key[2]]
    if cands:
        return min(cands, key=lambda t: abs(t[0][1] - key[1]))[1]
    cands = [(g, gi) for g, gi in gmap.items() if g[0] == key[0]]
    if cands:
        return cands[0][1]
    return None

phase_net = MLP(930 + cmd.shape[1], 2).cuda()
phase_net.load_state_dict(torch.load(f"{ODIR8}/phase_net.pt", map_location="cuda"))
phase_net.eval()
tok_dec = MLP(2 + cmd.shape[1], 64, drop=0.1).cuda()
tok_dec.load_state_dict(torch.load(f"{ODIR8}/token_dec.pt", map_location="cuda"))
tok_dec.eval()
groups8c = sorted({int(k) for k in meta6})
tok_dec8c = MLP(2 + cmd.shape[1] + len(groups8c), 64, drop=0.1).cuda()
tok_dec8c.load_state_dict(
    torch.load(f"{ODIR8C}/token_dec.pt", map_location="cuda")
)
tok_dec8c.eval()


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


def run_v6(c, gi, steps, seed, ema=0.3):
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
            sc = nets6[gi](torch.from_numpy(x[None]).cuda())[0].cpu().numpy().astype(
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


def run_v8(c, steps, seed, ema=0.3):
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
    f = feat_for(c).astype(np.float32)
    for t in range(steps):
        prop = hist_to_proprio(env._get_sonic_history())
        x = np.concatenate([(prop - pmean) / pstd, f]).astype(np.float32)
        with torch.no_grad():
            sc = phase_net(torch.from_numpy(x[None]).cuda())[0].cpu().numpy().astype(
                np.float32
            )
        if ema > 0 and sc_prev is not None:
            sc = ema * sc_prev + (1 - ema) * sc
        sc_prev = sc
        n = float(np.linalg.norm(sc))
        if n < 1e-6:
            n = 1.0
        sc = sc / n
        xd = np.concatenate([sc, f]).astype(np.float32)
        with torch.no_grad():
            tok = (
                tok_dec(torch.from_numpy(xd[None]).cuda())[0]
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
        h_mean=round(float(heights.mean()), 3),
        h_min=round(float(heights.min()), 3),
        vx=round(float(vxs.mean()), 3),
        vy=round(float(vys.mean()), 3),
        speed=round(float(spd.mean()), 3),
        path=round(float(spd.sum() * 0.02), 2),
    )


def run_v8c(c, gi, steps, seed, ema=0.3):
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
    f = feat_for(c).astype(np.float32)
    frame_oh = np.zeros(len(groups8c), dtype=np.float32)
    frame_oh[groups8c.index(gi)] = 1.0
    for t in range(steps):
        prop = hist_to_proprio(env._get_sonic_history())
        x = np.concatenate([(prop - pmean) / pstd, f]).astype(np.float32)
        with torch.no_grad():
            sc = nets6[gi](torch.from_numpy(x[None]).cuda())[0].cpu().numpy().astype(
                np.float32
            )
        if ema > 0 and sc_prev is not None:
            sc = ema * sc_prev + (1 - ema) * sc
        sc_prev = sc
        n = float(np.linalg.norm(sc))
        if n < 1e-6:
            n = 1.0
        sc = sc / n
        xd = np.concatenate([sc, f, frame_oh]).astype(np.float32)
        with torch.no_grad():
            tok = (
                tok_dec8c(torch.from_numpy(xd[None]).cuda())[0]
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
        h_mean=round(float(heights.mean()), 3),
        h_min=round(float(heights.min()), 3),
        vx=round(float(vxs.mean()), 3),
        vy=round(float(vys.mean()), 3),
        speed=round(float(spd.mean()), 3),
        path=round(float(spd.sum() * 0.02), 2),
    )


# (name, mode, speed, angle, steps) -- steps=1000 (20 s), switch uses its own
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
    # --- unseen commands (not in training data) ---
    ("slow_speed35", 1, 0.35, 0.0, 1000),
    ("slow_speed35_back", 1, 0.35, np.pi, 1000),
    ("walk_bin1", 2, -1.0, -3 * np.pi / 8, 600),
    ("walk_bin2", 2, -1.0, -np.pi / 2, 600),
    ("walk_bin3", 2, -1.0, -np.pi / 8, 600),
    ("walk_bin5", 2, -1.0, np.pi / 8, 600),
    ("walk_bin6", 2, -1.0, np.pi / 2, 600),
    ("walk_bin7", 2, -1.0, 3 * np.pi / 8, 600),
]

if not RUN_SWITCH:
    out = {}
    for name, m, sp, a, steps in scen:
        gi = select_group_fallback(m, sp, angle_bin_of(a))
        mdir = [float(np.cos(a)), float(np.sin(a)), 0.0]
        c = dict(mode=m, speed=sp, mdir=mdir, fdir=mdir)
        row = {}
        for seed in [0, 1, 2]:
            r6 = run_v6(c, gi, steps, seed) if gi is not None else None
            r8 = run_v8(c, steps, seed)
            r8c = run_v8c(c, gi, steps, seed) if gi is not None else None
            row[f"seed{seed}"] = {"v6": r6, "v8": r8, "v8c": r8c}
            print(
                f"{name:16s} seed{seed} v6={'n/a' if r6 is None else r6}",
                flush=True,
            )
            print(f"{'':16s}         v8={r8}", flush=True)
            print(f"{'':16s}         v8c={'n/a' if r8c is None else r8c}", flush=True)
        out[name] = row
    json.dump(out, open(f"{ODIR8}/eval_battery_v8.json", "w"), indent=1)
    print("battery done")
else:
    print("battery skipped (--switch-only)")


# switch episode (v6 vs v8) -- same schedule as battery v6
sched = [
    ("idle", 5),
    ("walk_fwd", 10),
    ("idle", 5),
    ("slow_back", 10),
    ("turn_left", 8),
    ("slow_fwd", 10),
    ("jump", 5),
    ("idle", 5),
]
scen_map = {x[0]: x for x in scen}


def run_switch(mode_fn, seed=4):
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
    sc_state = {}
    for name, secs in sched:
        _, m, sp, a, _ = scen_map[name]
        gi = select_group_fallback(m, sp, angle_bin_of(a))
        mdir = [float(np.cos(a)), float(np.sin(a)), 0.0]
        c = dict(mode=m, speed=sp, mdir=mdir, fdir=mdir)
        for _t in range(secs * 50):
            prop = hist_to_proprio(env._get_sonic_history())
            tok, sc_state = mode_fn(c, gi, prop, sc_state)
            obs, reward, terminated, info = env.step(
                {"token": tok, "aux": np.zeros(12, dtype=np.float32)}
            )
            v = env._get_base_linear_velocity()
            vxs.append(float(v[0]))
            vys.append(float(v[1]))
            heights.append(float(env.data.qpos[2]))
            if terminated:
                fall = len(heights)
                break
        if terminated:
            break
    vxs = np.array(vxs)
    vys = np.array(vys)
    heights = np.array(heights)
    spd = np.sqrt(vxs**2 + vys**2)
    return dict(
        fall_step=fall,
        completed=fall is None,
        h_min=round(float(heights.min()), 3),
        speed=round(float(spd.mean()), 3),
        path=round(float(spd.sum() * 0.02), 2),
    )


def v6_tok(c, gi, prop, sc_state):
    key = "v6"
    x = np.concatenate([(prop - pmean) / pstd, feat_for(c)]).astype(np.float32)
    with torch.no_grad():
        sc = nets6[gi](torch.from_numpy(x[None]).cuda())[0].cpu().numpy().astype(
            np.float32
        )
    prev = sc_state.get(key)
    if prev is not None:
        sc = 0.3 * prev + 0.7 * sc
    sc_state[key] = sc
    phi = float(np.arctan2(sc[0], sc[1]))
    b = int(np.floor((phi + np.pi) / (2 * np.pi) * len(protos[gi])) % len(protos[gi]))
    return protos[gi][b], sc_state


def v8_tok(c, gi, prop, sc_state):
    key = "v8"
    f = feat_for(c).astype(np.float32)
    x = np.concatenate([(prop - pmean) / pstd, f]).astype(np.float32)
    with torch.no_grad():
        sc = phase_net(torch.from_numpy(x[None]).cuda())[0].cpu().numpy().astype(
            np.float32
        )
    prev = sc_state.get(key)
    if prev is not None:
        sc = 0.3 * prev + 0.7 * sc
    sc_state[key] = sc
    n = float(np.linalg.norm(sc))
    if n < 1e-6:
        n = 1.0
    sc = sc / n
    xd = np.concatenate([sc, f]).astype(np.float32)
    with torch.no_grad():
        tok = tok_dec(torch.from_numpy(xd[None]).cuda())[0].cpu().numpy().astype(
            np.float32
        )
    return np.clip(np.round(tok * 16) / 16, -1, 1).astype(np.float32), sc_state


sw = {}
for name, fn in [("v6", v6_tok), ("v8", v8_tok)]:
    sw[name] = run_switch(fn)
    print("switch", name, sw[name], flush=True)
json.dump(sw, open(f"{ODIR8}/eval_switch_v8.json", "w"), indent=1)
print("v8 battery done")
