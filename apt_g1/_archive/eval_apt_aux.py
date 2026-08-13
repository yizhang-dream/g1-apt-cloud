"""Closed-loop eval: frozen phase-router token prior + PPO-trained aux (APT-style).

Compares aux=0 (pure distilled token) vs aux=PPO policy on:
  A. 60 s straight walk
  B. disturbance impulses during walk (45 s)
  C. vx/vy command-switch marathon (68 s)
  D. jump with explicit mode command (20 s)
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")

from apt_g1.encoder import Command, PhaseRouterEncoder
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from apt_g1.policies.apt_policy import APTPolicy
from eval_distill import NoQuantDecoder


REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"


def make_env(router, episode_s):
    decoder = NoQuantDecoder(f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx")
    return MujocoG1FlatEnv(
        decoder,
        REPO,
        use_elastic_band=False,
        stand_only=True,
        episode_length_s=episode_s,
        phase_router=router,
    )


def reset_jitter(env, seed):
    import mujoco

    env.reset()
    rng = np.random.default_rng(1000 + seed)
    env.data.qpos[2] += float(rng.normal(0, 0.005))
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(np.float32)
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco.mj_forward(env.model, env.data)
    env._reset_history()
    env._fill_history_from_state()


def rollout(env, policy, schedule, seed, use_aux, impulses=None, phase_policy=False):
    """schedule: list of (vx, vy, seconds) or (Command, seconds) pairs."""
    reset_jitter(env, seed)
    total_steps = int(
        sum(entry[2] if len(entry) == 3 else entry[1] for entry in schedule) * 50
    )
    imp = {s: f for s, f in (impulses or [])}
    heights, vxs, vys = [], [], []
    fall = None
    t = 0
    for entry in schedule:
        if len(entry) == 3:
            vx, vy, secs = entry
            item = (vx, vy)
        else:
            item, secs = entry
        if isinstance(item, Command):
            env.router_command = item
            env.command = np.zeros(3, dtype=np.float32)
        else:
            vx, vy = item
            env.router_command = None
            env.command = np.array([vx, vy, 0.0], dtype=np.float32)
        for _ in range(int(secs * 50)):
            if t in imp:
                pid = env.model.body("pelvis").id
                env.data.xfrc_applied[pid, :3] = np.asarray(imp[t], dtype=np.float64)
            obs = env.get_obs().astype(np.float32)
            extra = {}
            if use_aux:
                with torch.no_grad():
                    act, _, _ = policy.act(torch.from_numpy(obs[None]).cuda(), deterministic=True)
                aux = act["aux"][0].cpu().numpy().astype(np.float32)
                if phase_policy:
                    extra["phase"] = act["phase"][0].cpu().numpy().astype(np.float32)
            else:
                aux = np.zeros(12, dtype=np.float32)
            obs2, reward, terminated, info = env.step(
                {"token": np.zeros(64, dtype=np.float32), "aux": aux, **extra}
            )
            pid = env.model.body("pelvis").id
            env.data.xfrc_applied[pid, :3] = 0.0
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
        "completed": fall is None and len(heights) >= total_steps - 1,
        "fall_step": fall,
        "h_min": round(float(heights.min()), 3),
        "h_mean": round(float(heights.mean()), 3),
        "vx": round(float(vxs.mean()), 3),
        "vy": round(float(vys.mean()), 3),
        "displacement": round(float(spd.sum() * 0.02), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--phase-router-dir", default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_final")
    ap.add_argument("--out", default="/home/cvgluser/ros2_data/apt_g1/outputs/apt_aux_router/eval_apt_aux.json")
    ap.add_argument("--phase-policy", action="store_true")
    args = ap.parse_args()

    router = PhaseRouterEncoder(args.phase_router_dir)
    if args.phase_policy:
        from apt_g1.policies.phase_aux_policy import PhaseAuxPolicy

        policy = PhaseAuxPolicy(obs_dim=99, aux_dim=12).cuda()
    else:
        policy = APTPolicy(
            obs_dim=99,
            token_dim=64,
            aux_dim=12,
            num_skills=2,
            use_skill_selection=False,
        ).cuda()
    policy.load_state_dict(torch.load(args.policy, map_location="cuda"))
    policy.eval()
    pp = args.phase_policy

    out = {}
    # ---- A. 60s walk ----
    env = make_env(router, 70.0)
    out["A_walk60"] = {}
    for use_aux in [True, False]:
        key = "aux" if use_aux else "noaux"
        out["A_walk60"][key] = {}
        for seed in [0, 1, 2]:
            r = rollout(env, policy, [(0.8, 0.0, 60)], seed, use_aux, phase_policy=pp)
            out["A_walk60"][key][f"seed{seed}"] = r
            print(f"A walk60 {key} seed{seed} done={r['completed']} h_min={r['h_min']} vx={r['vx']} disp={r['displacement']}", flush=True)

    # ---- B. disturbance grid ----
    env = make_env(router, 50.0)
    dirs = {"fwd": [500.0, 0, 0], "back": [-500.0, 0, 0], "left": [0, 500.0, 0], "right": [0, -500.0, 0]}
    out["B_disturbance"] = {}
    for use_aux in [True, False]:
        out["B_disturbance"]["aux" if use_aux else "noaux"] = {}
        for dname, dvec in dirs.items():
            for seed in [0, 1, 2]:
                imp = [(500, dvec), (1250, dvec)]
                r = rollout(env, policy, [(0.8, 0.0, 45)], seed, use_aux, impulses=imp, phase_policy=pp)
                out["B_disturbance"]["aux" if use_aux else "noaux"][f"{dname}_seed{seed}"] = r
                print(f"B {dname} {'aux' if use_aux else 'noaux'} seed{seed} done={r['completed']} h_min={r['h_min']}", flush=True)

    # ---- C. command-switch marathon (vx/vy only) ----
    env = make_env(router, 75.0)
    sched = [
        (0.0, 0.0, 5), (0.8, 0.0, 8), (0.0, 0.0, 3), (-0.8, 0.0, 6),
        (0.0, 0.0, 3), (0.25, 0.0, 6), (0.0, 0.0, 3), (0.25, -0.43, 6),
        (0.0, 0.0, 3), (0.25, 0.43, 6), (0.0, 0.0, 3), (0.8, 0.0, 8),
    ]
    out["C_switch"] = {}
    for use_aux in [True, False]:
        key = "aux" if use_aux else "noaux"
        out["C_switch"][key] = {}
        for seed in [0, 1, 2]:
            r = rollout(env, policy, [(vx, vy, s) for vx, vy, s in sched], seed, use_aux, phase_policy=pp)
            out["C_switch"][key][f"seed{seed}"] = r
            print(f"C switch {key} seed{seed} done={r['completed']} fall={r['fall_step']} h_min={r['h_min']}", flush=True)

    # ---- D. jump (explicit mode) ----
    env = make_env(router, 30.0)
    jump_cmd = Command(
        mode=17, speed=-1.0,
        mdir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        fdir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    out["D_jump"] = {}
    for use_aux in [True, False]:
        key = "aux" if use_aux else "noaux"
        out["D_jump"][key] = {}
        for seed in [0, 1, 2]:
            r = rollout(env, policy, [(jump_cmd, 20)], seed, use_aux, phase_policy=pp)
            out["D_jump"][key][f"seed{seed}"] = r
            print(f"D jump {key} seed{seed} done={r['completed']} h_min={r['h_min']} vx={r['vx']}", flush=True)

    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
