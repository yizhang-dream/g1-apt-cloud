"""E49 smoke: direct-token mode invariants on a real headless Isaac env.

Protocol: DS_OFFICIAL_DATA_PLAN.md §3.2 smoke list (owner R3/R4 reviews).
Run (server):
    bash /tmp/run_apt_isaac.sh /home/cvgluser/ros2_data/apt_g1/isaac/e49_smoke.py \
        --arm B --token-stats /home/cvgluser/ros2_data/apt_g1/outputs/e49/token_stats_e49.npz

Checks:
  1. obs dim == 91 - 2 + 64 (+2 phi for arm B); env's own obs assert holds
  2. decoder receives exactly token_mean + alpha * token_std * a (no tanh,
     no quantization -- linear mapping only)
  3. action feedback slot env._last_phase == raw action a (NOT the mapped
     token) -- same convention as the latent arm's z feedback
  4. arm B: obs tail [sin, cos] == sin/cos(env._latent_phase) bit-exact,
     before the step and after (post-advance value = the phi the NEXT
     decode will pair with -- same pairing the latent VAE uses)
  5. fresh-policy initial action stats recorded (a std ~= exp(-4) = 0.0183;
     mapped token per-dim mean/std vs official) -- owner R4 #1
  6. rewards finite over the smoke steps
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], default="A")
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--token-alpha", type=float, default=1.0)
    ap.add_argument("--token-stats", required=True,
                    help="npz with mean[64]/std[64]/rate from official g1-mode tokens")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--router-model-dir",
                    default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_final")
    ap.add_argument(
        "--decoder-path",
        default=(
            "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
            "gear_sonic_deploy/policy/release/model_decoder.onnx"
        ),
    )
    cli = ap.parse_args()

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = cli.num_envs
    launcher_args.headless = cli.headless
    launcher_args.env_spacing = 4.0
    launcher_args.output_dir = "/tmp/isaac_smoke"
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.ppo_core import AptPPOPolicy

    cfg = AptFlatG1EnvCfg()
    cfg.scene.num_envs = cli.num_envs
    cfg.sonic_decoder_path = cli.decoder_path
    cfg.router_model_dir = cli.router_model_dir
    cfg.vx_max = 0.8
    cfg.disturbance_prob = 0.0
    cfg.episode_length_s = 20.0
    cfg.token_mode = True
    cfg.token_phase_obs = cli.arm == "B"
    cfg.token_alpha = cli.token_alpha
    cfg.token_stats = cli.token_stats
    cfg.observation_space = 91 - 2 + 64 + (2 if cfg.token_phase_obs else 0)
    env = AptFlatG1Env(cfg)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    expect_dim = cfg.observation_space
    print(f"[e49-smoke] arm={cli.arm} obs={tuple(obs.shape)} expected={expect_dim}")
    assert obs.shape[1] == cfg.observation_space == expect_dim

    # (5) initial action stats from a FRESH policy (same builder as train)
    torch.manual_seed(0)
    policy = AptPPOPolicy(
        obs_dim=cfg.observation_space, aux_dim=12, gate_k=0,
        hidden_dim=256, use_phase=False, latent_dim=64,
    ).to(env.device)
    policy.eval()
    with torch.no_grad():
        samples = torch.stack(
            [policy.act(obs, deterministic=False)[0]["latent"] for _ in range(512)]
        )
    a_std = samples.std(dim=0).mean().item()
    tok = env._token_mean + cfg.token_alpha * env._token_std * samples
    print(f"[e49-smoke] init a: mean-of-std={a_std:.5f} (exp(-4)={float(np.exp(-4)):.5f}), "
          f"mean-of|mean|={samples.mean(dim=0).abs().mean().item():.5f}")
    print(f"[e49-smoke] init token: mean-of-std={tok.std(dim=0).mean().item():.5f} "
          f"vs official mean std={env._token_std.mean().item():.5f} "
          f"-> ratio={(tok.std(dim=0).mean() / env._token_std.mean()).item():.4f}")

    # capture what the decoder actually receives
    cap = {}
    orig_decode = env._decoder.decode

    def _cap(*args, **kw):
        cap["args"] = args
        return orig_decode(*args, **kw)

    env._decoder.decode = _cap

    for t in range(cli.steps):
        obs_now = env._get_observations()["policy"]
        if cfg.token_phase_obs:
            expect_phi = torch.stack(
                [torch.sin(env._latent_phase), torch.cos(env._latent_phase)], dim=1
            )
            assert torch.equal(obs_now[:, -2:], expect_phi), f"step {t}: phi obs stale"
        a = torch.randn(env.num_envs, 64, device=env.device) * 0.5
        obs_dict, rew, term, trunc, _ = env.step(a)
        assert torch.equal(env._last_phase, a), f"step {t}: feedback != raw action"
        tok_in = [x for x in cap["args"]
                  if torch.is_tensor(x) and tuple(x.shape) == tuple(a.shape)]
        assert tok_in, f"step {t}: no (N,64) tensor reached the decoder"
        expect_tok = env._token_mean + cfg.token_alpha * env._token_std * a
        assert torch.allclose(tok_in[0], expect_tok, atol=1e-5), f"step {t}: mapping"
        assert torch.isfinite(rew).all(), f"step {t}: non-finite reward"
    print(f"[e49-smoke] {cli.steps} steps: mapping/feedback/phi/reward all OK")
    print("[e49-smoke] PASS")
    os._exit(0)


if __name__ == "__main__":
    main()
