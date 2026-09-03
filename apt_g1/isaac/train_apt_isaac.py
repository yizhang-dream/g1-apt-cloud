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
    # E34: randomize commanded yaw during training (domain randomization) so
    # the policy learns to steer toward the commanded heading. Default 0,0 =
    # E31 behavior (constant yaw=0, which caused the systematic drift).
    ap.add_argument("--yaw-min", type=float, default=0.0)
    ap.add_argument("--yaw-max", type=float, default=0.0)
    ap.add_argument("--disturbance-prob", type=float, default=0.0)
    ap.add_argument("--disturbance-ramp-iters", type=int, default=0)
    ap.add_argument("--use-2hz-gate", type=int, default=1)
    ap.add_argument("--phase-mode", action="store_true")
    ap.add_argument("--phase-anchor", action="store_true")
    ap.add_argument("--latent-mode", action="store_true")
    # E44: decoder fine-tuning. Action space = 29-d joint targets; the policy
    # owns a trainable SONIC decoder (E39 latent -> VAE -> token -> decoder).
    ap.add_argument("--decft", action="store_true")
    ap.add_argument("--decoder-reg", type=float, default=1.0)
    # E44: separate (smaller) LR for the fine-tuned decoder; None = same LR
    ap.add_argument("--decoder-lr", type=float, default=None)
    # E44v2: weight-space anchor + action noise scale (guards against the
    # decoder drifting off the official manifold, which broke E44 v1)
    ap.add_argument("--decoder-wreg", type=float, default=0.0)
    ap.add_argument("--decft-aux-std", type=float, default=-2.0)
    # E44 two-phase: phase 1 trains z against the FROZEN official decoder
    ap.add_argument("--freeze-decoder", action="store_true")
    # E44p1-fix: z-noise scale (phase_init_std); larger = real z exploration so
    # the z-head gets a usable score-function gradient through the decoder
    ap.add_argument("--decft-phase-std", type=float, default=-4.0)
    ap.add_argument(
        "--latent-vae-path",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/token_vae_e27/vae.pt",
    )
    ap.add_argument("--latent-warmstart-iters", type=int, default=0)
    ap.add_argument("--latent-kl", type=float, default=2.5e-6)
    ap.add_argument("--latent-expl", type=float, default=0.01)
    # E28: command-conditioned gait cadence + forward-speed reward shaping
    ap.add_argument("--latent-cmd-phase-rate", action="store_true")
    ap.add_argument("--latent-phase-rate-ref", type=float, default=0.6)
    ap.add_argument("--latent-phase-rate-max", type=float, default=2.0)
    ap.add_argument("--stillness-vx-scale", type=float, default=0.05)
    # E29: latent KL prior. "zero" = N(0,I) (E27); "walk" = N(z_walk, I) keeps z
    # on the SONIC walk manifold instead of pulling it toward the origin.
    ap.add_argument("--latent-kl-prior", choices=["zero", "walk"], default="zero")
    # E31: speed-conditioned VAE decoder (D(z, phase, v_bin) -> token)
    ap.add_argument("--latent-speed-bins", action="store_true")
    # E35: direction+speed-conditioned VAE decoder (D(z,phi,v_bin,psi_bin))
    ap.add_argument("--latent-dir-bins", action="store_true")
    # E48: full-joint residual escape channel (RuN/ReSkill-style). Action
    # becomes [z(16), res(29)]; q_des = q_decoder(z) + res_scale*clamp(res).
    ap.add_argument("--latent-residual", action="store_true")
    ap.add_argument("--res-scale", type=float, default=0.4)
    ap.add_argument("--res-clip", type=float, default=1.0)
    ap.add_argument("--res-l2", type=float, default=0.0)
    # E48c: freeze the residual (zeroed in the env) for the first N control
    # steps so the z-head first learns a working controller on terrain.
    ap.add_argument("--res-freeze-steps", type=int, default=0)
    # E32: heading/yaw reward strengthening (fights high-speed drift)
    ap.add_argument("--yaw-scale", type=float, default=0.5)
    ap.add_argument("--heading-scale", type=float, default=0.0)
    ap.add_argument("--to-ref", action="store_true",
                    help="TO38: append the 12-d TO reference obs block (zeros without --to-ref-npz)")
    ap.add_argument("--to-ref-npz", default="",
                    help="TO38: LUT from apt_g1/to38_export_ref.py (empty = zero block, paired control arm)")
    ap.add_argument("--to-ref-obs-zero", action="store_true",
                    help="TO38: control arm -- load LUT (clock/diagnostics) but zero the obs block")
    ap.add_argument("--to-ref-w", type=float, default=0.0, help="TO38: tracking reward weight")
    ap.add_argument("--to-ref-sigma2", type=float, default=0.1)
    ap.add_argument("--to-ref-gate-sigma2", type=float, default=0.0036)
    ap.add_argument("--to-tau", action="store_true",
                    help="TO40-C: cmd-gated torque feedforward from the LUT (no obs change)")
    ap.add_argument("--to-tau-w", type=float, default=1.0)
    ap.add_argument("--aux-scale", type=float, default=0.2)
    ap.add_argument("--aux-l2", type=float, default=0.0)
    ap.add_argument("--aux-rate", type=float, default=0.0)
    ap.add_argument("--yaw-sigma2", type=float, default=0.25)
    ap.add_argument("--vel-sigma2", type=float, default=0.25)
    ap.add_argument("--phase-warmstart-iters", type=int, default=0)
    ap.add_argument("--phase-warmstart-coef", type=float, default=10.0)
    ap.add_argument("--entropy", type=float, default=0.001)
    # TO42 修订 v4（论文式大并行操作点）：2048 envs × 500it 配 minibatch 4096
    # （24×2048/4096 = 12 minibatch/epoch，整除）；默认 512 = 既有行为逐字不变
    ap.add_argument("--ppo-minibatch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/isaac_apt_aux")
    ap.add_argument("--env", choices=["apt", "vanilla"], default="apt")
    ap.add_argument("--terrain", choices=["plane", "rough", "rough_paper", "rough_sym"], default="plane")
    ap.add_argument("--terrain-noise", type=float, default=0.04)
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument("--use-elevation", type=int, default=0)
    ap.add_argument("--gate-sel", type=int, default=0)
    # TO42: learned regime selection on the frozen decoder substrate
    # (TO42_PLAN §3；骑在 --latent-mode 上；action 16→17，obs +2)
    ap.add_argument("--to42-sel", choices=["off", "lsel", "fbkt"], default="off")
    ap.add_argument("--to42-hold-steps", type=int, default=25)
    ap.add_argument("--progress-scale", type=float, default=0.0)
    ap.add_argument("--anti-stop", type=float, default=0.0)
    ap.add_argument("--anti-stop-thresh", type=float, default=0.3)
    # E44v3: penalty on yaw-rate (omega_z^2) to suppress the spin gait
    ap.add_argument("--yaw-rate-penalty", type=float, default=0.0)
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
    cfg.yaw_min = cli.yaw_min
    cfg.yaw_max = cli.yaw_max
    cfg.disturbance_prob = 0.0 if cli.disturbance_ramp_iters > 0 else cli.disturbance_prob
    cfg.use_2hz_gate = bool(cli.use_2hz_gate)
    cfg.use_elevation = bool(cli.use_elevation)
    if cfg.use_elevation:
        cfg.observation_space += cfg.elev_grid * cfg.elev_grid
    cfg.use_gate_sel = bool(cli.gate_sel)
    cfg.progress_scale = cli.progress_scale
    cfg.anti_stop_scale = cli.anti_stop
    cfg.anti_stop_thresh = cli.anti_stop_thresh
    cfg.yaw_rate_penalty = cli.yaw_rate_penalty
    if cfg.use_gate_sel:
        cfg.action_space = 13  # aux(12) + gate(1)
    cfg.phase_mode = cli.phase_mode
    cfg.phase_anchor = cli.phase_anchor
    cfg.latent_mode = cli.latent_mode
    cfg.latent_vae_path = cli.latent_vae_path
    cfg.latent_speed_bins = cli.latent_speed_bins
    cfg.latent_dir_bins = cli.latent_dir_bins
    cfg.latent_residual = cli.latent_residual
    cfg.res_scale = cli.res_scale
    cfg.res_clip = cli.res_clip
    cfg.res_l2_scale = cli.res_l2
    cfg.res_freeze_steps = cli.res_freeze_steps
    cfg.yaw_scale = cli.yaw_scale
    cfg.heading_scale = cli.heading_scale
    cfg.to_ref = cli.to_ref
    cfg.to_ref_npz = cli.to_ref_npz
    cfg.to_ref_obs_zero = cli.to_ref_obs_zero
    cfg.to_ref_w = cli.to_ref_w
    cfg.to_ref_sigma2 = cli.to_ref_sigma2
    cfg.to_ref_gate2 = cli.to_ref_gate_sigma2
    cfg.to_tau = cli.to_tau
    cfg.to_tau_w = cli.to_tau_w
    if cfg.to_ref:
        # obs block: [sin psi, cos psi, q_ref6_rel, pitch, z, heel_x_rel, heel_z]
        cfg.observation_space += 12
    cfg.latent_cmd_phase_rate = cli.latent_cmd_phase_rate
    cfg.latent_phase_rate_ref = cli.latent_phase_rate_ref
    cfg.latent_phase_rate_max = cli.latent_phase_rate_max
    cfg.stillness_vx_scale = cli.stillness_vx_scale
    if cli.latent_mode:
        cfg.action_space = 16  # latent z only (no aux / gate)
        cfg.observation_space += 14  # _last_phase 2 -> 16 in the observation
        if cli.latent_residual:
            cfg.action_space = 16 + 29  # z(16) + full-joint residual(29)
            cfg.observation_space += 29  # residual action feedback
    if cli.to42_sel != "off":
        # TO42: selection rides on the latent decode path；两臂 obs/action
        # 布局完全一致，唯一差异 = 选择由策略学出还是由冻结 bucketize 产生
        assert not cli.gate_sel, "--to42-sel and --gate-sel are mutually exclusive"
        assert cli.latent_mode and not cli.decft and not cli.latent_residual, (
            "--to42-sel rides on plain --latent-mode")
        cfg.to42_sel = cli.to42_sel
        cfg.to42_hold_steps = cli.to42_hold_steps
        cfg.action_space = 17        # z(16) + sel bit(1)
        cfg.observation_space += 2   # [sel_state, gate_bool]
    to42_active = cli.to42_sel != "off"
    if cli.decft:
        from apt_g1.isaac.decft_policy import OBS_DIM as DECFT_OBS_DIM

        cfg.decft_mode = True
        cfg.action_space = 29  # normalized joint targets (decoder output)
        cfg.observation_space = DECFT_OBS_DIM  # 91 base + 930 hist + 2 phase
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
    elif cli.decft:
        from apt_g1.isaac.decft_policy import DecFtPolicy

        env = AptFlatG1Env(cfg)
        policy = DecFtPolicy(
            obs_dim=cfg.observation_space,
            vae_path=cli.latent_vae_path,
            decoder_path=cli.decoder_path,
            vx_max=cli.vx_max,
            aux_init_std=cli.decft_aux_std,
            phase_init_std=cli.decft_phase_std,
            freeze_decoder=cli.freeze_decoder,
            device="cuda:0",
        ).to("cuda:0")
    else:
        env = AptFlatG1Env(cfg)
        policy = AptPPOPolicy(
            obs_dim=cfg.observation_space,
            aux_dim=29 if cli.latent_residual else 12,
            gate_k=(2 if to42_active else (3 if cfg.use_gate_sel else 0)),
            hidden_dim=256,
            use_phase=not cfg.use_gate_sel and not cfg.latent_mode,
            latent_dim=16 if cfg.latent_mode else 0,
        ).to("cuda:0")
    latent_prior_mean = None
    if cli.latent_mode and cli.latent_kl_prior == "walk":
        zw = np.load(str(Path(cli.latent_vae_path).parent / "z_walk.npy"))
        latent_prior_mean = torch.from_numpy(zw).float().to("cuda:0")
    if cli.decft and cli.latent_kl_prior == "walk":
        # E44: keep z near the E39 walk manifold while the decoder adapts
        zw = np.load(str(Path(cli.latent_vae_path).parent / "z_walk.npy"))
        latent_prior_mean = torch.from_numpy(zw).float().to("cuda:0")
    trainer = PPOTrainer(
        policy,
        lr=cli.lr,
        entropy_coef=cli.entropy,
        latent_kl_coef=cli.latent_kl,
        latent_expl_coef=cli.latent_expl,
        latent_prior_mean=latent_prior_mean,
        max_iters=cli.iters,
        device="cuda:0",
        decoder_reg_coef=cli.decoder_reg if cli.decft else 0.0,
        decoder_lr=cli.decoder_lr if cli.decft else None,
        decoder_wreg_coef=cli.decoder_wreg if cli.decft else 0.0,
        minibatch_size=cli.ppo_minibatch,
    )
    start_it = 0
    if cli.resume:
        sd = torch.load(cli.resume, map_location="cuda:0")
        if cli.decft:
            # partial warm start: keep encoder/z-head/critic from the E39
            # checkpoint; decoder stays at official init, aux heads are dropped
            cur = policy.state_dict()
            sd = {
                k: v
                for k, v in sd.items()
                if k in cur and tuple(v.shape) == tuple(cur[k].shape)
            }
            policy.load_state_dict(sd, strict=False)
        else:
            policy.load_state_dict(sd)
        start_it = int(Path(cli.resume).stem.split("_")[-1])
        if start_it >= cli.iters:
            raise SystemExit(
                f"[train] resume iteration {start_it} >= --iters {cli.iters}; "
                "pass a cumulative --iters larger than the checkpoint iteration"
            )

    rollout = cli.rollout
    T, N, D = rollout, env.num_envs, cfg.observation_space
    aux_dim = 29 if (cli.env == "vanilla" or cli.decft or cli.latent_residual) else 12
    phase_labels_buf = None
    if cli.env == "vanilla":
        buf_phase_none = True
    else:
        buf_phase_none = False
    if (cli.phase_mode or cli.latent_mode) and (
        cli.phase_warmstart_iters > 0 or cli.latent_warmstart_iters > 0
    ):
        phase_labels_buf = torch.zeros(
            T, N, 16 if cli.latent_mode else 2, device="cuda:0"
        )
        if cli.latent_mode:
            zw = np.load(
                str(Path(cli.latent_vae_path).parent / "z_walk.npy")
            )
            phase_labels_buf[:] = torch.from_numpy(zw).to("cuda:0")

    buf = {
        "obs": torch.zeros(T, N, D, device="cuda:0"),
        "phase": torch.zeros(
            T, N, 16 if (cli.latent_mode or cli.decft) else 2, device="cuda:0"
        ),
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
    if to42_active:
        # phase z 保留 + 增加 gate(sel) 槽位；PPOTrainer.update 的 gate 分支
        # （Categorical log_prob/entropy 重算）原样复用
        buf["gate"] = torch.zeros(T, N, dtype=torch.long, device="cuda:0")

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
            if cli.decft:
                buf["phase"][t] = act["phase"].detach()
                buf["aux"][t] = act["aux"].detach()
                action = act["aux"]
            elif cli.env == "vanilla":
                buf["aux"][t] = act["aux"].detach()
                action = act["aux"]
            else:
                buf["aux"][t] = act["aux"].detach()
                if cli.gate_sel:
                    buf["gate"][t] = act["gate"].detach()
                    action = torch.cat(
                        [act["aux"], act["gate"].float().unsqueeze(-1)], dim=1
                    )
                elif to42_active:
                    buf["phase"][t] = act["phase"].detach()
                    buf["gate"][t] = act["gate"].detach()
                    action = torch.cat(
                        [act["phase"], act["gate"].float().unsqueeze(-1)], dim=1
                    )
                else:
                    buf["phase"][t] = act["phase"].detach()
                    if cfg.latent_mode and cfg.latent_residual:
                        # E48: [z(16), res(29)] -- the aux head IS the residual
                        action = torch.cat([act["phase"], act["aux"]], dim=1)
                    elif cfg.latent_mode:
                        action = act["phase"]
                    else:
                        action = torch.cat([act["phase"], act["aux"]], dim=1)
            buf["logp"][t] = logp.detach()
            buf["value"][t] = val.detach()
            obs_dict, rew, term, trunc, _ = env.step(action)
            buf["reward"][t] = rew
            buf["done"][t] = term
            buf["trunc"][t] = trunc
            obs = obs_dict["policy"]
            ep_rewards.append(rew.mean().item())
            if (
                phase_labels_buf is not None
                and it < (
                    cli.latent_warmstart_iters
                    if cli.latent_mode
                    else cli.phase_warmstart_iters
                )
            ):
                if not cli.latent_mode:  # latent: z_walk labels pre-filled
                    cmds = env._build_commands_list()
                    proprio = env._proprio_np()
                    if cfg.phase_anchor:
                        # anchored mode: the policy phase head is a bounded offset,
                        # warmstart it toward zero offset (pure router clock).
                        phase_labels_buf[t] = torch.zeros(
                            env.num_envs, 2, dtype=torch.float32, device="cuda:0"
                        )
                    else:
                        sc, _ = env._router.phase_raw_batch(proprio, cmds)
                        phase_labels_buf[t] = torch.from_numpy(sc).to("cuda:0")
        last_val = policy.get_value(obs)
        buf["last_value"] = last_val.detach()

        warm_coef = 0.0
        warm_iters = (
            cli.latent_warmstart_iters if cli.latent_mode else cli.phase_warmstart_iters
        )
        if phase_labels_buf is not None and it < warm_iters:
            warm_coef = cli.phase_warmstart_coef * (
                1.0 - it / max(1, warm_iters)
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
            dec_dw = 0.0
            if cli.decft:
                # E44: total weight drift of the fine-tuned decoder vs official
                dec_dw = float(
                    sum(
                        (p.detach() - rp.detach()).pow(2).sum().item()
                        for p, rp in zip(
                            policy.decoder.net.parameters(),
                            policy.decoder_ref.net.parameters(),
                        )
                    )
                    ** 0.5
                )
            print(
                f"[{it}/{cli.iters}] rew={mean_rew:.3f} fall={fall_rate:.3f} "
                f"vx={vx:.3f} loss={stats['loss']:.4f} ploss={stats['ploss']:.4f} "
                f"ent={stats['ent']:.4f} kl={stats['kl']:.6f} "
                f"expl={stats['expl']:.5f} dreg={stats['dreg']:.5f} "
                f"dec_dw={dec_dw:.4f} dt={it_time:.1f}s",
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
