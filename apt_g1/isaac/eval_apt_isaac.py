"""Isaac Lab A/B/C/D eval for the APT aux policy (mirrors MuJoCo eval_apt_aux).

A. 60 s straight walk @ vx=0.8
B. disturbance impulses (500 N, 4 dirs, t=10 s and t=25 s) during 45 s walk
C. vx/vy command-switch marathon (68 s)
D. jump with explicit mode command (20 s)

Each test runs with aux=0 and with the trained policy aux (and optionally the
policy phase in phase_mode), 3 seeds, and reports the same metrics as the
MuJoCo harness (steps, completed, fall_step, h_min, vx, displacement).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import os

import numpy as np
import torch


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tests", default="A,B,C,D")
    ap.add_argument("--env", choices=["apt", "vanilla"], default="apt")
    ap.add_argument("--out", default="outputs/isaac_eval.json")
    ap.add_argument("--phase-mode", action="store_true")
    ap.add_argument("--phase-anchor", action="store_true")
    ap.add_argument("--aux-scale", type=float, default=0.2)
    ap.add_argument("--latent-mode", action="store_true")
    # E44: decoder fine-tuning policy (29-d joint-target actions). The
    # checkpoint contains the fine-tuned SONIC decoder; run A/B/C only (the
    # D jump test needs the router mode path, which decft bypasses).
    ap.add_argument("--decft", action="store_true")
    ap.add_argument(
        "--latent-vae-path",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/token_vae_e27/vae.pt",
    )
    # E28: command-conditioned gait cadence (must mirror training for consistency)
    ap.add_argument("--latent-cmd-phase-rate", action="store_true")
    ap.add_argument("--latent-phase-rate-ref", type=float, default=0.6)
    ap.add_argument("--latent-phase-rate-max", type=float, default=2.0)
    ap.add_argument("--stillness-vx-scale", type=float, default=0.05)
    # E31: speed-conditioned VAE decoder (must mirror training)
    ap.add_argument("--latent-speed-bins", action="store_true")
    # E35: direction+speed-conditioned VAE decoder (must mirror training)
    ap.add_argument("--latent-dir-bins", action="store_true")
    # E48: full-joint residual escape channel (must mirror training). aux head
    # of the checkpoint is 29-d; --aux-zero gives the residual-off ablation.
    ap.add_argument("--latent-residual", action="store_true")
    ap.add_argument("--res-scale", type=float, default=0.4)
    ap.add_argument("--res-clip", type=float, default=1.0)
    ap.add_argument("--use-elevation", type=int, default=0)
    # E32: heading/yaw reward (rollout dynamics only; no effect on eval metrics)
    ap.add_argument("--yaw-scale", type=float, default=0.5)
    ap.add_argument("--heading-scale", type=float, default=0.0)
    # TO38: override test A's commanded vx (default 0.8 = E-battery standard;
    # low-speed band evals pass e.g. 0.277 / 0.2 / 0.35)
    ap.add_argument("--a-cmd-vx", type=float, default=0.8)
    # TO38: reference obs injection (must match the trained policy's obs dim)
    ap.add_argument("--to-ref", action="store_true")
    ap.add_argument("--to-ref-npz", default="")
    ap.add_argument("--to-ref-obs-zero", action="store_true")
    # E33: open-loop yaw-bias compensation (rad/s). Cancels a systematic
    # turning bias (e.g. E31's ~-0.07 rad/s = -4 deg/s left drift).
    ap.add_argument("--yaw-bias-comp", type=float, default=0.0)
    ap.add_argument("--terrain", choices=["plane", "rough", "rough_paper", "rough_sym"], default="plane")
    ap.add_argument("--terrain-noise", type=float, default=0.04)
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument(
        "--phase-zero",
        action="store_true",
        help="ablation: zero the policy phase output (pure router clock in "
        "anchored mode, no modulation)",
    )
    ap.add_argument(
        "--aux-zero",
        action="store_true",
        help="ablation: zero the policy aux output (isolate phase mechanism)",
    )
    ap.add_argument("--use-2hz-gate", type=int, default=1)
    ap.add_argument("--router-model-dir",
                    default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_final")
    ap.add_argument(
        "--decoder-path",
        default=(
            "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
            "gear_sonic_deploy/policy/release/model_decoder.onnx"
        ),
    )
    return ap


def jitter_and_reset(env, seed: int):
    """Reset all envs (num_envs=1) and apply MuJoCo-parity reset jitter."""
    rng = np.random.default_rng(1000 + seed)
    env.reset()
    device = env.device
    env_ids = torch.arange(env.num_envs, device=device)

    root = env.robot.data.default_root_state[env_ids].clone()
    root[:, 2] = env.scene.env_origins[env_ids, 2] + 0.76 + torch.tensor(
        rng.normal(0.0, 0.005, len(env_ids)), dtype=torch.float32, device=device
    )
    env.robot.write_root_state_to_sim(root, env_ids)

    jp = env.robot.data.default_joint_pos[env_ids].clone()
    sonic = torch.from_numpy(env._sonic_default).to(device)
    noise = torch.tensor(
        rng.normal(0.0, 0.01, (len(env_ids), 29)), dtype=torch.float32, device=device
    )
    jp[:, env._body_idx] = sonic + noise
    jv = torch.tensor(
        rng.normal(0.0, 0.02, (len(env_ids), env.robot.num_joints)),
        dtype=torch.float32,
        device=device,
    )
    env.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

    # refill history from the (jittered) current state
    ang_vel = env._base_ang_vel()
    jpos_rel = env.robot.data.joint_pos[:, env._body_idx] - sonic
    jvel = env.robot.data.joint_vel[:, env._body_idx]
    gravity = env.robot.data.projected_gravity_b
    n = env.num_envs
    env._hist_ang_vel[:] = ang_vel[:, None, :].expand(n, 10, 3).clone()
    env._hist_joint_pos[:] = jpos_rel[:, None, :].expand(n, 10, 29).clone()
    env._hist_joint_vel[:] = jvel[:, None, :].expand(n, 10, 29).clone()
    env._hist_last_actions[:] = 0.0
    env._hist_gravity[:] = gravity[:, None, :].expand(n, 10, 3).clone()
    if env._router_state is not None:
        env._router_state = env._router.reset_state(env.num_envs)
    env._gate_mode[:] = 0
    env._gate_tick[:] = False
    env._gate_count[:] = 0
    env._q_des[:] = sonic
    env._last_phase[:] = 0.0
    env._last_aux[:] = 0.0


def rollout(
    env,
    policy,
    schedule,
    seed,
    use_aux,
    impulses=None,
    phase_policy=False,
    phase_zero=False,
    aux_zero=False,
    latent_policy=False,
    yaw_bias=0.0,  # E33: open-loop yaw command offset (rad/s) to cancel a
    # systematic turning bias (e.g. E31's -4 deg/s left drift)
):
    """schedule: list of (vx, vy, seconds) or (Command, seconds) pairs."""
    from apt_g1.encoder import Command

    jitter_and_reset(env, seed)
    total_steps = int(
        sum(entry[2] if len(entry) == 3 else entry[1] for entry in schedule) * 50
    )
    imp = {s: f for s, f in (impulses or [])}
    heights, vxs, vys = [], [], []
    xys = []
    fall = None
    t = 0
    for entry in schedule:
        if len(entry) == 3:
            vx, vy, secs = entry
            item = (vx, vy)
        else:
            item, secs = entry
        if isinstance(item, Command):
            env.router_commands[0] = item
            env._commands[0] = torch.zeros(3, dtype=torch.float32, device=env.device)
        else:
            vx, vy = item
            env.router_commands[0] = None
            env._commands[0] = torch.tensor(
                [vx, vy, yaw_bias], dtype=torch.float32, device=env.device
            )
        for _ in range(int(secs * 50)):
            if t in imp:
                world_dir = torch.tensor(
                    imp[t], dtype=torch.float32, device=env.device
                ).reshape(1, 1, 3)
                body_dir = env._world_to_body(world_dir)
                forces = torch.zeros(1, 1, 3, dtype=torch.float32, device=env.device)
                forces[0, 0] = body_dir[0, 0]
                env.robot.set_external_force_and_torque(
                    forces,
                    torch.zeros_like(forces),
                    body_ids=[env._root_body_idx[0]],
                )
            if use_aux:
                with torch.no_grad():
                    act, _, _, _, _ = policy.act(env._last_obs, deterministic=True)
                aux = act["aux"]
                if aux_zero:
                    aux = torch.zeros_like(aux)
            else:
                aux = torch.zeros(1, 12, dtype=torch.float32, device=env.device)
            if getattr(env, "_vanilla", False):
                with torch.no_grad():
                    act, _, _, _, _ = policy.act(env._last_obs, deterministic=True)
                action = act["aux"]
            elif getattr(env, "_decft", False):
                # E44: policy action IS the 29-d joint-target vector
                if not use_aux:  # should not happen (decft keys force use_aux)
                    with torch.no_grad():
                        act, _, _, _, _ = policy.act(env._last_obs, deterministic=True)
                action = act["aux"]
            elif getattr(env, "_gate_policy", False):
                with torch.no_grad():
                    act, _, _, _, _ = policy.act(env._last_obs, deterministic=True)
                action = torch.zeros(1, 13, dtype=torch.float32, device=env.device)
                action[:, :12] = aux
                action[:, 12] = act["gate"].float()
            elif latent_policy:
                with torch.no_grad():
                    act, _, _, _, _ = policy.act(env._last_obs, deterministic=True)
                action = act["latent"]
                if getattr(env, "_latent_residual", False):
                    # E48: append the full-joint residual (policy aux head).
                    # noaux / aux_zero rollouts zero it -> pure-prior ablation.
                    if use_aux and not aux_zero:
                        res = act["aux"]
                    else:
                        res = torch.zeros_like(act["aux"])
                    action = torch.cat([action, res], dim=1)
            else:
                action = torch.zeros(1, 14, dtype=torch.float32, device=env.device)
                action[:, 2:] = aux
                if phase_policy:
                    with torch.no_grad():
                        act, _, _, _, _ = policy.act(env._last_obs, deterministic=True)
                    if phase_zero:
                        act["phase"] = torch.zeros_like(act["phase"])
                    action[:, :2] = act["phase"]
            obs_dict, reward, term, trunc, _ = env.step(action)
            env._last_obs = obs_dict["policy"]
            if t in imp:
                env.robot.set_external_force_and_torque(
                    torch.zeros(1, 1, 3, dtype=torch.float32, device=env.device),
                    torch.zeros(1, 1, 3, dtype=torch.float32, device=env.device),
                    body_ids=[env._root_body_idx[0]],
                )
            h = float(env.robot.data.root_pos_w[0, 2].item())
            v = env._base_lin_vel()[0].detach().cpu().numpy()
            xy = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy()
            heights.append(h)
            vxs.append(float(v[0]))
            vys.append(float(v[1]))
            xys.append(xy)
            if term.any():
                fall = t
                break
            t += 1
        if fall is not None:
            break
    heights = np.array(heights)
    vxs = np.array(vxs)
    vys = np.array(vys)
    xys = np.array(xys)
    h_min = float(heights.min()) if len(heights) else 0.0
    displacement = 0.0
    if len(xys) > 1:
        displacement = float(np.linalg.norm(xys[-1] - xys[0]))
    spd = np.sqrt(vxs**2 + vys**2)
    return {
        "steps": len(heights),
        "completed": fall is None and len(heights) >= total_steps - 1,
        "fall_step": fall,
        "h_min": round(h_min, 3),
        "vx": round(float(vxs.mean()), 3),
        "disp": round(displacement, 3),
        "v_speed": round(float(spd.mean()), 3),
    }


def main():
    cli = build_args().parse_args()

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = 1
    launcher_args.headless = cli.headless
    launcher_args.env_spacing = 4.0
    launcher_args.output_dir = str(Path(cli.out).parent)
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    from apt_g1.encoder import Command
    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.apt_flat_env_vanilla import (
        AptFlatG1VanillaEnv,
        AptFlatG1VanillaEnvCfg,
    )
    from apt_g1.isaac.terrain_cfg import make_terrain_importer_cfg
    from apt_g1.isaac.ppo_core import AptPPOPolicy

    if cli.env == "vanilla":
        cfg = AptFlatG1VanillaEnvCfg()
        policy = AptPPOPolicy(
            obs_dim=cfg.observation_space, aux_dim=29, use_phase=False
        ).to("cuda:0")
    elif cli.decft:
        from apt_g1.isaac.decft_policy import DecFtPolicy, OBS_DIM as DECFT_OBS_DIM

        cfg = AptFlatG1EnvCfg()
        cfg.observation_space = DECFT_OBS_DIM
        policy = DecFtPolicy(
            obs_dim=cfg.observation_space,
            vae_path=cli.latent_vae_path,
            decoder_path=cli.decoder_path,
            vx_max=0.8,
        ).to("cuda:0")
    else:
        cfg = AptFlatG1EnvCfg()
        if cli.latent_mode:
            cfg.observation_space += 14  # _last_phase 2 -> 16 in the observation
        cfg.use_elevation = bool(cli.use_elevation)
        if cfg.use_elevation:
            cfg.observation_space += cfg.elev_grid * cfg.elev_grid
        if cli.latent_residual:
            cfg.observation_space += 29  # residual action feedback
        if cli.to_ref:
            cfg.observation_space += 12  # TO38 reference block
        policy = AptPPOPolicy(
            obs_dim=cfg.observation_space,
            aux_dim=29 if cli.latent_residual else 12,
            use_phase=not cli.latent_mode,
            latent_dim=16 if cli.latent_mode else 0,
        ).to("cuda:0")
    cfg.scene.num_envs = 1
    cfg.terrain = make_terrain_importer_cfg(
        cli.terrain, cli.terrain_noise, seed=cli.terrain_seed
    )
    cfg.sonic_decoder_path = cli.decoder_path
    cfg.router_model_dir = cli.router_model_dir
    cfg.use_2hz_gate = bool(cli.use_2hz_gate)
    cfg.phase_mode = cli.phase_mode
    cfg.phase_anchor = cli.phase_anchor
    cfg.latent_mode = cli.latent_mode
    cfg.decft_mode = cli.decft
    cfg.latent_vae_path = cli.latent_vae_path
    cfg.latent_speed_bins = cli.latent_speed_bins
    cfg.latent_dir_bins = cli.latent_dir_bins
    cfg.latent_residual = cli.latent_residual
    cfg.res_scale = cli.res_scale
    cfg.res_clip = cli.res_clip
    cfg.yaw_scale = cli.yaw_scale
    cfg.heading_scale = cli.heading_scale
    cfg.to_ref = cli.to_ref
    cfg.to_ref_npz = cli.to_ref_npz
    cfg.to_ref_obs_zero = cli.to_ref_obs_zero
    cfg.latent_cmd_phase_rate = cli.latent_cmd_phase_rate
    cfg.latent_phase_rate_ref = cli.latent_phase_rate_ref
    cfg.latent_phase_rate_max = cli.latent_phase_rate_max
    cfg.stillness_vx_scale = cli.stillness_vx_scale
    cfg.aux_scale = cli.aux_scale
    cfg.disturbance_prob = 0.0
    cfg.episode_length_s = 120.0
    # pin the global numpy RNG to the terrain seed before env creation
    np.random.seed(cli.terrain_seed)
    if cli.env == "vanilla":
        env = AptFlatG1VanillaEnv(cfg)
    else:
        env = AptFlatG1Env(cfg)
    try:
        policy.load_state_dict(torch.load(cli.checkpoint, map_location="cuda:0"))
        policy.eval()
        print("[eval] loaded checkpoint", cli.checkpoint)
    except FileNotFoundError:
        print("[eval] WARNING: checkpoint not found, aux=0 only")

    # initial obs
    env._vanilla = cli.env == "vanilla"
    env._decft = cli.decft
    env._latent_residual = cli.latent_residual
    obs_dict, _ = env.reset()
    env._last_obs = obs_dict["policy"]

    tests = set(cli.tests.split(","))
    out = {"A_walk60": {}, "B_disturbance": {}, "C_switch": {}, "D_jump": {}}
    pp = cli.phase_mode
    pz = cli.phase_zero
    az = cli.aux_zero
    lp = cli.latent_mode
    if cli.decft:
        # single policy key (no aux/noaux split; D needs the router mode path)
        key_list = ["aux"]
        if "D" in tests:
            tests.discard("D")
    else:
        key_list = ["aux", "noaux"] if not pp else ["phaseaux"]

    # ---- A. 60s walk @ 0.8 ----

    if "A" in tests:
        out["A_walk60"]["cmd_vx"] = cli.a_cmd_vx
        for key in key_list:
            out["A_walk60"][key] = {}
            use_aux = key != "noaux"
            for seed in [0, 1, 2]:
                r = rollout(env, policy, [(cli.a_cmd_vx, 0.0, 60)], seed, use_aux, phase_policy=pp, phase_zero=pz, aux_zero=az, latent_policy=lp, yaw_bias=cli.yaw_bias_comp)
                out["A_walk60"][key][f"seed{seed}"] = r
                print(f"A walk{cli.a_cmd_vx} {key} seed{seed} done={r['completed']} h_min={r['h_min']} vx={r['vx']} disp={r['disp']}", flush=True)

    # ---- B. disturbance grid ----

    if "B" in tests:
        dirs = {"fwd": [500.0, 0, 0], "back": [-500.0, 0, 0], "left": [0, 500.0, 0], "right": [0, -500.0, 0]}
        for key in key_list:
            out["B_disturbance"][key] = {}
            use_aux = key != "noaux"
            for dname, dvec in dirs.items():
                for seed in [0, 1, 2]:
                    imp = [(500, dvec), (1250, dvec)]
                    r = rollout(env, policy, [(0.8, 0.0, 45)], seed, use_aux, impulses=imp, phase_policy=pp, phase_zero=pz, aux_zero=az, latent_policy=lp, yaw_bias=cli.yaw_bias_comp)
                    out["B_disturbance"][key][f"{dname}_seed{seed}"] = r
                    print(f"B {dname} {key} seed{seed} done={r['completed']} h_min={r['h_min']}", flush=True)

    # ---- C. command-switch marathon ----

    if "C" in tests:
        sched = [
            (0.0, 0.0, 5), (0.8, 0.0, 8), (0.0, 0.0, 3), (-0.8, 0.0, 6),
            (0.0, 0.0, 3), (0.25, 0.0, 6), (0.0, 0.0, 3), (0.25, -0.43, 6),
            (0.0, 0.0, 3), (0.25, 0.43, 6), (0.0, 0.0, 3), (0.8, 0.0, 8),
        ]
        for key in key_list:
            out["C_switch"][key] = {}
            use_aux = key != "noaux"
            for seed in [0, 1, 2]:
                r = rollout(env, policy, [(vx, vy, s) for vx, vy, s in sched], seed, use_aux, phase_policy=pp, phase_zero=pz, aux_zero=az, latent_policy=lp, yaw_bias=cli.yaw_bias_comp)
                out["C_switch"][key][f"seed{seed}"] = r
                print(f"C switch {key} seed{seed} done={r['completed']} fall={r['fall_step']} h_min={r['h_min']}", flush=True)

    # ---- D. jump (explicit mode) ----

    if "D" in tests:
        jump_cmd = Command(
            mode=17, speed=-1.0,
            mdir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            fdir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        for key in key_list:
            out["D_jump"][key] = {}
            use_aux = key != "noaux"
            for seed in [0, 1, 2]:
                r = rollout(env, policy, [(jump_cmd, 20)], seed, use_aux, phase_policy=pp, phase_zero=pz, aux_zero=az, latent_policy=lp, yaw_bias=cli.yaw_bias_comp)
                out["D_jump"][key][f"seed{seed}"] = r
                print(f"D jump {key} seed{seed} done={r['completed']} h_min={r['h_min']} vx={r['vx']}", flush=True)

    if tests != {"A", "B", "C", "D"}:
        out = {k: v for k, v in out.items() if k.split("_")[0] in tests}
    with open(cli.out, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", cli.out)
    os._exit(0)


if __name__ == "__main__":
    main()
