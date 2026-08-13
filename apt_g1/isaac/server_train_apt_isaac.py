"""Train the APT (phase-router prior + aux) policy in Isaac Lab.

Controller line on Isaac: frozen phase-router token prior + PPO-trained aux,
with the paper's RL-stage mechanisms:
  - latent KL w.r.t. N(0, I) (coef 2.5e-6)
  - latent exploration bonus decaying to zero
  - 2 Hz gait-gate hold + feedback observation (cfg.use_2hz_gate)

Usage (from the repo root on the training server):
    PYTHONPATH=~/ros2_data/apt_g1:~/ros2_data/GR00T-WholeBodyControl \\
      python ~/ros2_data/apt_g1/isaac/train_apt_isaac.py \\
        --num-envs 64 --iters 500 --out outputs/isaac_apt_aux
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--rollout", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vx-max", type=float, default=0.8)
    ap.add_argument("--disturbance-prob", type=float, default=0.0)
    ap.add_argument("--disturbance-ramp-iters", type=int, default=0)
    ap.add_argument("--use-2hz-gate", type=int, default=1)
    ap.add_argument("--phase-mode", action="store_true")
    ap.add_argument("--latent-kl", type=float, default=2.5e-6)
    ap.add_argument("--latent-expl", type=float, default=0.01)
    ap.add_argument("--aux-scale", type=float, default=0.2)
    ap.add_argument("--aux-l2", type=float, default=0.0)
    ap.add_argument("--aux-rate", type=float, default=0.0)
    ap.add_argument("--yaw-sigma2", type=float, default=0.25)
    ap.add_argument("--vel-sigma2", type=float, default=0.25)
    ap.add_argument("--phase-warmstart-iters", type=int, default=0)
    ap.add_argument("--phase-warmstart-coef", type=float, default=10.0)
    ap.add_argument("--entropy", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/isaac_apt_aux")
    ap.add_argument("--env", choices=["apt", "vanilla"], default="apt")
    ap.add_argument("--terrain", choices=["plane", "rough"], default="plane")
    ap.add_argument("--terrain-noise", type=float, default=0.04)
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument("--use-elevation", type=int, default=0)
    ap.add_argument("--gate-sel", type=int, default=0)
    ap.add_argument("--progress-scale", type=float, default=0.0)
    ap.add_argument("--anti-stop", type=float, default=0.0)
    ap.add_argument("--anti-stop-thresh", type=float, default=0.3)
    ap.add_argument("--resume", default=None)
    ap.add_argument(
        "--router-model-dir",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_final",
    )
    ap.add_argument(
        "--decoder-path",
        default=(
            "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
            "gear_sonic_deploy/policy/release/model_decoder.onnx"
        ),
    )
    return ap


def main():
    ap = build_args()

    # AppLauncher args (must be created before any isaaclab import)
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    cli = ap.parse_args()
    launcher_args.num_envs = cli.num_envs
    launcher_args.headless = cli.headless
    launcher_args.env_spacing = 4.0
    launcher_args.output_dir = cli.out
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    # safe to import isaaclab-dependent modules now
    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.apt_flat_env_vanilla import (
        AptFlatG1VanillaEnv,
        AptFlatG1VanillaEnvCfg,
    )
    from apt_g1.isaac.ppo_core import AptPPOPolicy, PPOTrainer
    from apt_g1.isaac.terrain_cfg import make_terrain_importer_cfg

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    out_dir = Path(cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cli.env == "vanilla":
        cfg = AptFlatG1VanillaEnvCfg()
    else:
        cfg = AptFlatG1EnvCfg()
    cfg.scene.num_envs = cli.num_envs
    cfg.terrain = make_terrain_importer_cfg(
        cli.terrain, cli.terrain_noise, seed=cli.terrain_seed
    )
    cfg.sonic_decoder_path = cli.decoder_path
    cfg.router_model_dir = cli.router_model_dir
    cfg.vx_max = cli.vx_max
    cfg.disturbance_prob = 0.0 if cli.disturbance_ramp_iters > 0 else cli.disturbance_prob
    cfg.use_2hz_gate = bool(cli.use_2hz_gate)
    cfg.use_elevation = bool(cli.use_elevation)
    if cfg.use_elevation:
        cfg.observation_space += cfg.elev_grid * cfg.elev_grid
    cfg.use_gate_sel = bool(cli.gate_sel)
    cfg.progress_scale = cli.progress_scale
    cfg.anti_stop_scale = cli.anti_stop
    cfg.anti_stop_thresh = cli.anti_stop_thresh
    if cfg.use_gate_sel:
        cfg.action_space = 13  # aux(12) + gate(1)
    cfg.phase_mode = cli.phase_mode
    cfg.aux_scale = cli.aux_scale
    cfg.aux_l2_scale = cli.aux_l2
    cfg.aux_rate_scale = cli.aux_rate
    cfg.yaw_sigma2 = cli.yaw_sigma2
    cfg.vel_sigma2 = cli.vel_sigma2
    cfg.episode_length_s = 20.0
    # pin the global numpy RNG to the terrain seed before env creation (see
    # eval_fast.py note: HfRandomUniformTerrainCfg uses the global np.random).
    np.random.seed(cli.terrain_seed)

    if cli.env == "vanilla":
        env = AptFlatG1VanillaEnv(cfg)
        policy = AptPPOPolicy(
            obs_dim=cfg.observation_space,
            aux_dim=29,
            gate_k=0,
            hidden_dim=256,
            use_phase=False,
        ).to("cuda:0")
    else:
        env = AptFlatG1Env(cfg)
        policy = AptPPOPolicy(
            obs_dim=cfg.observation_space,
            aux_dim=12,
            gate_k=3 if cfg.use_gate_sel else 0,
            hidden_dim=256,
            use_phase=not cfg.use_gate_sel,
        ).to("cuda:0")
    trainer = PPOTrainer(
        policy,
        lr=cli.lr,
        entropy_coef=cli.entropy,
        latent_kl_coef=cli.latent_kl,
        latent_expl_coef=cli.latent_expl,
        max_iters=cli.iters,
        device="cuda:0",
    )
    start_it = 0
    if cli.resume:
        policy.load_state_dict(torch.load(cli.resume, map_location="cuda:0"))
        start_it = int(Path(cli.resume).stem.split("_")[-1])
        if start_it >= cli.iters:
            raise SystemExit(
                f"[train] resume iteration {start_it} >= --iters {cli.iters}; "
                "pass a cumulative --iters larger than the checkpoint iteration"
            )

    rollout = cli.rollout
    T, N, D = rollout, env.num_envs, cfg.observation_space
    aux_dim = 29 if cli.env == "vanilla" else 12
    phase_labels_buf = None
    if cli.env == "vanilla":
        buf_phase_none = True
    else:
        buf_phase_none = False
    if cli.phase_mode and cli.phase_warmstart_iters > 0:
        phase_labels_buf = torch.zeros(T, N, 2, device="cuda:0")

    buf = {
        "obs": torch.zeros(T, N, D, device="cuda:0"),
        "phase": torch.zeros(T, N, 2, device="cuda:0"),
        "aux": torch.zeros(T, N, aux_dim, device="cuda:0"),
        "logp": torch.zeros(T, N, device="cuda:0"),
        "value": torch.zeros(T, N, device="cuda:0"),
        "reward": torch.zeros(T, N, device="cuda:0"),
        "done": torch.zeros(T, N, dtype=torch.bool, device="cuda:0"),
        "trunc": torch.zeros(T, N, dtype=torch.bool, device="cuda:0"),
    }
    if cli.gate_sel:
        buf["gate"] = torch.zeros(T, N, dtype=torch.long, device="cuda:0")
        buf["phase"] = None

    if buf_phase_none:
        buf["phase"] = None
    hist = {"rewards": [], "vx": [], "fall_rate": []}
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    for it in range(start_it, cli.iters):
        t0 = time.time()
        if cli.disturbance_ramp_iters > 0:
            env.cfg.disturbance_prob = min(
                cli.disturbance_prob,
                cli.disturbance_prob * (it + 1) / cli.disturbance_ramp_iters,
            )
        ep_rewards = []
        for t in range(T):
            act, logp, ent, val, _ = policy.act(obs)
            buf["obs"][t] = obs
            if cli.env == "vanilla":
                buf["aux"][t] = act["aux"].detach()
                action = act["aux"]
            else:
                buf["aux"][t] = act["aux"].detach()
                if cli.gate_sel:
                    buf["gate"][t] = act["gate"].detach()
                    action = torch.cat(
                        [act["aux"], act["gate"].float().unsqueeze(-1)], dim=1
                    )
                else:
                    buf["phase"][t] = act["phase"].detach()
                    action = torch.cat([act["phase"], act["aux"]], dim=1)
            buf["logp"][t] = logp.detach()
            buf["value"][t] = val.detach()
            obs_dict, rew, term, trunc, _ = env.step(action)
            buf["reward"][t] = rew
            buf["done"][t] = term
            buf["trunc"][t] = trunc
            obs = obs_dict["policy"]
            ep_rewards.append(rew.mean().item())
            if phase_labels_buf is not None and it < cli.phase_warmstart_iters:
                cmds = env._build_commands_list()
                proprio = env._proprio_np()
                sc, _ = env._router.phase_raw_batch(proprio, cmds)
                phase_labels_buf[t] = torch.from_numpy(sc).to("cuda:0")
        last_val = policy.get_value(obs)
        buf["last_value"] = last_val.detach()

        warm_coef = 0.0
        if phase_labels_buf is not None and it < cli.phase_warmstart_iters:
            warm_coef = cli.phase_warmstart_coef * (
                1.0 - it / max(1, cli.phase_warmstart_iters)
            )
        stats = trainer.update(buf, phase_labels=phase_labels_buf, phase_warm_coef=warm_coef)
        it_time = time.time() - t0
        mean_rew = float(np.mean(ep_rewards))
        fall_rate = float(buf["done"].float().mean().item())
        hist["rewards"].append(mean_rew)
        hist["fall_rate"].append(fall_rate)
        # mean vx from env data
        vx = float(
            torch.mean(
                torch.norm(env.robot.data.root_lin_vel_w[:, :2], dim=1).detach().cpu()
            )
        )
        hist["vx"].append(vx)
        if it % 10 == 0 or it == cli.iters - 1:
            print(
                f"[{it}/{cli.iters}] rew={mean_rew:.3f} fall={fall_rate:.3f} "
                f"vx={vx:.3f} loss={stats['loss']:.4f} ploss={stats['ploss']:.4f} "
                f"ent={stats['ent']:.4f} kl={stats['kl']:.6f} "
                f"expl={stats['expl']:.5f} dt={it_time:.1f}s",
                flush=True,
            )
        if (it + 1) % 50 == 0 or it == cli.iters - 1:
            ckpt = out_dir / f"policy_it_{it + 1}.pt"
            torch.save(policy.state_dict(), ckpt)
            with open(out_dir / "train_log.json", "w") as f:
                json.dump(hist, f)

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(hist, f)
    torch.save(policy.state_dict(), out_dir / "policy_final.pt")
    print("saved", out_dir)
    os._exit(0)


if __name__ == "__main__":
    main()
