"""60s+ stress test for the distilled PhaseRouterEncoder (no elastic band).

Three blocks:
  A. Long-run straight WALK (60 s) x 3 seeds
  B. Disturbance impulses during WALK (45 s): 4 directions x 2 magnitudes x 3 seeds
  C. Command-switch marathon (~66 s) x 3 seeds

Metrics: completion, fall step, min/mean root height, mean vx, displacement,
post-impulse min height and recovery time.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")

from apt_g1.encoder import Command, PhaseRouterEncoder, proprio_vector
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from eval_distill import NoQuantDecoder


REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
MODEL_DIR = "/home/cvgluser/ros2_data/apt_g1/outputs/distill_final"
OUT = f"{MODEL_DIR}/stress_test_results.json"


def make_env(episode_s: float = 70.0):
    decoder = NoQuantDecoder(f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx")
    env = MujocoG1FlatEnv(
        decoder,
        REPO,
        use_elastic_band=False,
        stand_only=True,
        episode_length_s=episode_s,
    )
    env.command = np.zeros(3, dtype=np.float32)
    return env


def reset_with_jitter(env, seed):
    import mujoco

    env.reset()
    rng = np.random.default_rng(1000 + seed)
    env.data.qpos[2] += float(rng.normal(0, 0.005))
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(np.float32)
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco.mj_forward(env.model, env.data)
    env._reset_history()
    env._fill_history_from_state()
    return env


def impulse(env, force_world):
    pid = env.model.body("pelvis").id
    env.data.xfrc_applied[pid, :3] = np.asarray(force_world, dtype=np.float64)


def clear_impulse(env):
    pid = env.model.body("pelvis").id
    env.data.xfrc_applied[pid, :3] = 0.0


def run_episode(encoder, env, schedule, steps, seed, impulses=None):
    """schedule: list of (Command, seconds); impulses: list of (step, force_xyz)."""
    reset_with_jitter(env, seed)
    impulses = impulses or []
    imp = {s: f for s, f in impulses}
    heights, vxs, vys = [], [], []
    fall = None
    t = 0
    for cmd, secs in schedule:
        for _ in range(int(secs * 50)):
            if t in imp:
                impulse(env, imp[t])
            tok = encoder.encode(cmd, env._get_sonic_history())
            obs, reward, terminated, info = env.step({"token": tok, "aux": np.zeros(12, dtype=np.float32)})
            clear_impulse(env)
            v = env._get_base_linear_velocity()
            heights.append(float(env.data.qpos[2]))
            vxs.append(float(v[0]))
            vys.append(float(v[1]))
            if terminated:
                fall = t
                break
            t += 1
        if fall is not None:
            break
    heights = np.array(heights)
    vxs = np.array(vxs)
    vys = np.array(vys)
    spd = np.sqrt(vxs**2 + vys**2)
    return {
        "steps": len(heights),
        "completed": fall is None and len(heights) >= steps - 1,
        "fall_step": fall,
        "h_min": round(float(heights.min()), 3),
        "h_mean": round(float(heights.mean()), 3),
        "vx": round(float(vxs.mean()), 3),
        "vy": round(float(vys.mean()), 3),
        "displacement": round(float(spd.sum() * 0.02), 2),
    }


def recovery_time(vxs, imp_step, cmd_vx, window=25):
    """First step > imp_step where |vx - cmd_vx| < 0.15 sustained for `window` steps."""
    n = len(vxs)
    ok = np.abs(np.asarray(vxs) - cmd_vx) < 0.15
    for t in range(imp_step + 1, n - window):
        if ok[t : t + window].all():
            return round((t - imp_step) / 50.0, 2)
    return None


def main():
    encoder = PhaseRouterEncoder(MODEL_DIR)
    results = {"A_long_walk": {}, "B_disturbance": {}, "C_switch": {}}

    # ---- A. long straight walk 60s ----
    env = make_env(episode_s=70.0)
    cmd = Command.from_vxvy(0.8, 0.0)
    for seed in [0, 1, 2]:
        r = run_episode(encoder, env, [(cmd, 60)], 3000, seed)
        results["A_long_walk"][f"seed{seed}"] = r
        print(f"A walk60 seed{seed} done={r['completed']} h_min={r['h_min']} vx={r['vx']} disp={r['displacement']}", flush=True)

    # ---- B. disturbance grid during walk 45s ----
    env = make_env(episode_s=50.0)
    dirs = {"fwd": [500.0, 0, 0], "back": [-500.0, 0, 0], "left": [0, 500.0, 0], "right": [0, -500.0, 0]}
    t0 = time.time()
    for dname, dvec in dirs.items():
        for mag in [200.0, 500.0]:
            f = [v * mag / 500.0 for v in dvec]
            for seed in [0, 1, 2]:
                imp_steps = [500, 1250]
                imp = [(s, f) for s in imp_steps]
                r = run_episode(encoder, env, [(cmd, 45)], 2250, seed, impulses=imp)
                # per-impulse window metrics need the trajectory; re-derive cheaply:
                hw = []
                rw = []
                # rerun once to capture post-impulse windows (kept simple: reuse episode via a second run)
                reset_with_jitter(env, seed)
                heights, vxs = [], []
                t = 0
                for _ in range(2250):
                    if t in imp:
                        impulse(env, f)
                    tok = encoder.encode(cmd, env._get_sonic_history())
                    obs, reward, term, info = env.step({"token": tok, "aux": np.zeros(12, dtype=np.float32)})
                    clear_impulse(env)
                    heights.append(float(env.data.qpos[2]))
                    v = env._get_base_linear_velocity()
                    vxs.append(float(v[0]))
                    if term:
                        break
                    t += 1
                for s in imp_steps:
                    if s < len(heights):
                        hw.append(round(float(min(heights[s : s + 150])), 3))
                        rw.append(recovery_time(vxs, s, 0.83))
                    else:
                        hw.append(None)
                        rw.append(None)
                r["min_h_after_imp"] = hw
                r["recovery_s"] = rw
                key = f"{dname}_{int(mag)}_seed{seed}"
                results["B_disturbance"][key] = r
                print(
                    f"B {dname:5s} {int(mag):3d}N seed{seed} done={r['completed']} h_min={r['h_min']} "
                    f"after_imp={hw} recovery={rw}",
                    flush=True,
                )
    print("B elapsed", round(time.time() - t0, 1), "s", flush=True)

    # ---- C. command-switch marathon ----
    env = make_env(episode_s=75.0)
    sched_c = [
        (Command.from_vxvy(0.0, 0.0), 5),
        (Command.from_vxvy(0.8, 0.0), 8),
        (Command.from_vxvy(0.0, 0.0), 3),
        (Command.from_vxvy(-0.8, 0.0), 6),
        (Command.from_vxvy(0.0, 0.0), 3),
        (Command.from_vxvy(0.25, 0.0), 6),
        (Command.from_vxvy(0.0, 0.0), 3),
        (Command(mode=17, speed=-1.0, mdir=np.array([1.0, 0, 0], dtype=np.float32), fdir=np.array([1.0, 0, 0], dtype=np.float32)), 5),
        (Command.from_vxvy(0.0, 0.0), 3),
        (Command.from_vxvy(0.25, -0.43), 6),  # ~60 deg left turn
        (Command.from_vxvy(0.0, 0.0), 3),
        (Command.from_vxvy(0.25, 0.43), 6),  # ~60 deg right turn
        (Command.from_vxvy(0.0, 0.0), 3),
        (Command.from_vxvy(0.8, 0.0), 8),
    ]
    total_s = sum(s for _, s in sched_c)
    for seed in [0, 1, 2]:
        r = run_episode(encoder, env, sched_c, int(total_s * 50), seed)
        results["C_switch"][f"seed{seed}"] = r
        print(f"C switch{seed} done={r['completed']} h_min={r['h_min']} vx={r['vx']} disp={r['displacement']} ({total_s}s)", flush=True)

    json.dump(results, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
