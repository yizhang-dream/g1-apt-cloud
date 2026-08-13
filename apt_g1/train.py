"""Flat-ground APT-RL training entry point.

Smoke test with a dummy environment:

    python -m apt_g1.train --dummy --max-iters 5

Real Isaac Lab integration requires wiring the environment factory and the
SONIC decoder checkpoint first.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from apt_g1.envs.g1_flat_env import DummyG1FlatEnv
from apt_g1.policies.apt_policy import APTPolicy


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_dummy_env(cfg: dict):
    return DummyG1FlatEnv(
        action_dim=cfg["sonic"]["action_dim"],
        obs_dim=64,
        num_envs=1,
    )


def make_mujoco_env(cfg: dict, onnx_path: str, repo_root: str, phase_router_dir=None):
    from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
    from apt_g1.sonic.sonic_wrapper import SonicOnnxDecoder

    decoder = SonicOnnxDecoder(onnx_path)
    phase_router = None
    if phase_router_dir:
        from apt_g1.encoder import PhaseRouterEncoder

        phase_router = PhaseRouterEncoder(phase_router_dir)
        print(f"[make_mujoco_env] phase-router prior from {phase_router_dir}")
    motion_cfg = cfg.get("motion", {})
    reference_tokens = None
    base_token = None
    if motion_cfg.get("reference_tokens_path"):
        reference_tokens = np.load(motion_cfg["reference_tokens_path"])
    if motion_cfg.get("base_token_path"):
        base_token = np.load(motion_cfg["base_token_path"])
    token_vae = None
    if motion_cfg.get("token_vae_path"):
        from apt_g1.sonic.token_vae import TokenVAE

        token_vae = TokenVAE(
            token_dim=64, latent_dim=cfg["apt"]["latent_dim"]
        )
        token_vae.load_state_dict(torch.load(motion_cfg["token_vae_path"], map_location="cpu"))
        token_vae.eval()
    skill_tokens = None
    if motion_cfg.get("skill_tokens_path"):
        skill_tokens = np.load(motion_cfg["skill_tokens_path"])
    token_seq_vae = None
    if motion_cfg.get("token_seq_vae_path"):
        from apt_g1.sonic.token_seq_vae import TokenSeqVAE

        token_seq_vae = TokenSeqVAE(
            token_dim=64,
            latent_dim=cfg["apt"]["latent_dim"],
            window=motion_cfg.get("vae_window", 10),
        )
        token_seq_vae.load_state_dict(
            torch.load(motion_cfg["token_seq_vae_path"], map_location="cpu")
        )
        token_seq_vae.eval()
    joint_seq_vae = None
    if motion_cfg.get("joint_seq_vae_path"):
        from apt_g1.sonic.token_seq_vae import TokenSeqVAE

        joint_seq_vae = TokenSeqVAE(
            token_dim=29,
            latent_dim=cfg["apt"]["latent_dim"],
            window=motion_cfg.get("vae_window", 10),
        )
        joint_seq_vae.load_state_dict(
            torch.load(motion_cfg["joint_seq_vae_path"], map_location="cpu")
        )
        joint_seq_vae.eval()
    return MujocoG1FlatEnv(
        sonic_decoder=decoder,
        repo_root=repo_root,
        robot_scene=cfg["mujoco"]["robot_scene"],
        wbc_config_path=cfg["mujoco"]["wbc_config_path"],
        sim_dt=cfg["mujoco"]["sim_dt"],
        control_decimation=cfg["mujoco"]["control_decimation"],
        episode_length_s=cfg["mujoco"]["episode_length_s"],
        aux_scale=cfg["apt"]["aux_scale"],
        stand_only=cfg["commands"].get("stand_only", False),
        use_elastic_band=cfg["mujoco"].get("use_elastic_band", False),
        band_scale=cfg["mujoco"].get("band_scale", 1.0),
        command_vx_min=cfg["commands"].get("vx_min", 0.0),
        command_vx_max=cfg["commands"].get("vx_max", 1.0),
        command_vy_min=cfg["commands"].get("vy_min", -0.5),
        command_vy_max=cfg["commands"].get("vy_max", 0.5),
        command_yaw_min=cfg["commands"].get("yaw_min", -0.5),
        command_yaw_max=cfg["commands"].get("yaw_max", 0.5),
        reference_token_sequence=reference_tokens,
        residual_scale=motion_cfg.get("residual_scale", 0.1),
        base_token=base_token,
        token_vae=token_vae,
        skill_tokens=skill_tokens,
        token_seq_vae=token_seq_vae,
        add_phase_obs=motion_cfg.get("add_phase_obs", True),
        joint_seq_vae=joint_seq_vae,
        joint_reset_path=motion_cfg.get("joint_reset_path"),
        phase_router=phase_router,
        disturbance_prob=cfg.get("training", {}).get("disturbance_prob", 0.0),
        disturbance_force_range=tuple(
            cfg.get("training", {}).get("disturbance_force_range", [0.0, 0.0])
        ),
    )


def make_obs_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(obs, dtype=torch.float32, device=device)


def collect_rollout(policy, envs, num_steps: int, device: torch.device, gamma: float):
    if not isinstance(envs, (list, tuple)):
        envs = [envs]
    num_envs = len(envs)
    obs = np.stack([env.reset() for env in envs]).astype(np.float32)
    obs_t = make_obs_tensor(obs, device)
    obs_buf = []
    action_buf = {"token": [], "aux": [], "hand": [], "skill": []}
    reward_buf = []
    done_buf = []
    value_buf = []
    logprob_buf = []

    for _ in range(num_steps):
        action, log_prob, value = policy.act(obs_t)
        obs_buf.append(obs_t)
        action_buf["token"].append(action["token"])
        action_buf["aux"].append(action["aux"])
        action_buf["hand"].append(action["hand"])
        action_buf["skill"].append(action["skill"])
        logprob_buf.append(log_prob)
        value_buf.append(value)

        action_np = {key: value.detach().cpu().numpy() for key, value in action.items()}
        next_obs_list = []
        reward_list = []
        done_list = []
        for i, env in enumerate(envs):
            if hasattr(env, "action_dim"):
                full_action = np.zeros((1, env.action_dim), dtype=np.float32)
                lower = np.clip(action_np["aux"][i], -1.0, 1.0)
                full_action[0, : lower.shape[0]] = lower
                next_obs, reward, done, _ = env.step(full_action)
            else:
                per_env_action = {
                    key: value[i] for key, value in action_np.items()
                }
                next_obs, reward, done, _ = env.step(per_env_action)
            next_obs_list.append(np.asarray(next_obs, dtype=np.float32))
            reward_list.append(float(reward))
            done_list.append(bool(done))

        obs = np.stack(next_obs_list)
        obs_t = make_obs_tensor(obs, device)
        reward_buf.append(
            torch.as_tensor(np.asarray(reward_list, dtype=np.float32), device=device)
        )
        done_buf.append(
            torch.as_tensor(np.asarray(done_list, dtype=np.float32), device=device)
        )

    obs = torch.stack(obs_buf)
    actions = {
        "token": torch.stack(action_buf["token"]),
        "aux": torch.stack(action_buf["aux"]),
        "hand": torch.stack(action_buf["hand"]),
        "skill": torch.stack(action_buf["skill"]),
    }
    rewards = torch.stack(reward_buf)
    dones = torch.stack(done_buf)
    values = torch.stack(value_buf)
    logprobs = torch.stack(logprob_buf)

    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(num_steps)):
        running = rewards[t] + gamma * running * (1.0 - dones[t])
        returns[t] = running
    advantages = returns - values
    return obs, actions, logprobs, returns, advantages


def ppo_update(
    policy,
    optimizer,
    obs,
    actions,
    old_logprobs,
    returns,
    advantages,
    cfg: dict,
):
    ppo_cfg = cfg["ppo"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    num_samples = obs.shape[0] * obs.shape[1]
    flat_obs = obs.reshape(-1, obs.shape[-1])
    flat_actions = {
        key: value.detach().reshape(-1, value.shape[-1]) if key != "skill" else value.detach().reshape(-1)
        for key, value in actions.items()
    }
    flat_old = old_logprobs.detach().reshape(-1)
    flat_returns = returns.detach().reshape(-1)
    flat_adv = advantages.detach().reshape(-1)

    indices = torch.randperm(num_samples, device=obs.device)
    minibatch_size = ppo_cfg["minibatch_size"]
    for _ in range(ppo_cfg["num_epochs"]):
        for start in range(0, num_samples, minibatch_size):
            idx = indices[start : start + minibatch_size]
            batch_obs = flat_obs[idx]
            batch_actions = {key: value[idx] for key, value in flat_actions.items()}
            batch_old = flat_old[idx]
            batch_returns = flat_returns[idx]
            batch_adv = flat_adv[idx]

            log_prob, entropy, value = policy.evaluate_actions(batch_obs, batch_actions)
            ratio = torch.exp(log_prob - batch_old)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - ppo_cfg["clip_eps"], 1.0 + ppo_cfg["clip_eps"]
            )
            policy_loss = -torch.min(ratio * batch_adv, clipped_ratio * batch_adv).mean()
            value_loss = torch.nn.functional.mse_loss(value, batch_returns)
            entropy_loss = entropy.mean()
            loss = (
                policy_loss
                + ppo_cfg["value_coef"] * value_loss
                - ppo_cfg["entropy_coef"] * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), ppo_cfg["max_grad_norm"]
            )
            optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="apt_g1/configs/flat_g1.yaml")
    parser.add_argument("--dummy", action="store_true", help="Run on a dummy env")
    parser.add_argument("--mujoco", action="store_true", help="Run on MuJoCo G1")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--onnx-path", default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--load-policy", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--band-scale", type=float, default=None)
    parser.add_argument("--no-band", action="store_true", help="Disable elastic band")
    parser.add_argument("--token-std", type=float, default=None)
    parser.add_argument("--phase-router-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    if args.no_band:
        cfg["mujoco"]["use_elastic_band"] = False
        cfg["mujoco"]["band_scale"] = 0.0
    if args.band_scale is not None:
        cfg["mujoco"]["band_scale"] = args.band_scale
    max_iters = args.max_iters or cfg["training"]["max_iterations"]
    num_steps = args.num_steps or cfg["training"]["num_steps"]
    num_envs = cfg["training"].get("num_envs", 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dummy:
        envs = [make_dummy_env(cfg) for _ in range(num_envs)]
        obs_dim = 64
    elif args.mujoco:
        if args.onnx_path is None or args.repo_root is None:
            raise ValueError("--mujoco requires --onnx-path and --repo-root")
        envs = [
            make_mujoco_env(cfg, args.onnx_path, args.repo_root, args.phase_router_dir)
            for _ in range(num_envs)
        ]
        obs_dim = 3 + 3 + 3 + 29 + 29 + 29 + 3
        if (
            cfg.get("motion", {}).get("reference_tokens_path")
            and cfg.get("motion", {}).get("add_phase_obs", True)
        ):
            obs_dim += 1
    else:
        raise NotImplementedError(
            "Wire IsaacLabG1FlatEnv and the SONIC decoder, or run with --dummy."
        )

    policy = APTPolicy(
        obs_dim=obs_dim,
        token_dim=cfg["apt"]["latent_dim"],
        aux_dim=cfg["apt"]["aux_dim"],
        num_skills=cfg["apt"]["num_skills"],
        use_skill_selection=cfg["apt"]["use_skill_selection"],
    ).to(device)
    if args.load_policy:
        policy.load_state_dict(
            torch.load(args.load_policy, map_location=device, weights_only=False)
        )
        print(f"loaded policy from {args.load_policy}")
    if args.token_std is not None:
        with torch.no_grad():
            policy.token_log_std.data.fill_(math.log(args.token_std))
        print(f"set token std={args.token_std}")
    if args.load_policy is None:
        initial_token_path = cfg["apt"].get("initial_token_path")
        if initial_token_path:
            initial_token = np.load(initial_token_path).astype(np.float32)
            with torch.no_grad():
                policy.token_mean.weight.zero_()
                policy.token_mean.bias.copy_(torch.as_tensor(initial_token, device=device))
                initial_token_std = cfg["apt"].get("initial_token_std", 0.02)
                policy.token_log_std.data.fill_(math.log(initial_token_std))
            print(f"initialized token mean/std from {initial_token_path}")
        initial_aux_std = cfg["apt"].get("initial_aux_std", 0.01)
        with torch.no_grad():
            policy.aux_log_std.data.fill_(math.log(initial_aux_std))
            policy.hand_log_std.data.fill_(math.log(initial_aux_std))
        print(f"initialized aux/hand std={initial_aux_std}")
    train_token = cfg["apt"].get("train_token", True)
    if train_token:
        optimizer_params = list(policy.parameters())
    else:
        optimizer_params = [
            param for name, param in policy.named_parameters()
            if not name.startswith("token")
        ]
        print("token frozen; optimizing aux/hand/skill/critic only")
    optimizer = torch.optim.Adam(optimizer_params, lr=cfg["ppo"]["lr"])

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    band_anneal_iters = cfg["training"].get("band_anneal_iters", 0)
    band_scale_start = getattr(envs[0], "band_scale", 1.0)

    for iteration in range(max_iters):
        if band_anneal_iters > 0:
            scale = band_scale_start * max(0.0, 1.0 - iteration / band_anneal_iters)
            for env in envs:
                env.band_scale = scale
        obs, actions, logprobs, returns, advantages = collect_rollout(
            policy, envs, num_steps, device, cfg["ppo"]["gamma"]
        )
        ppo_update(
            policy,
            optimizer,
            obs,
            actions,
            logprobs,
            returns,
            advantages,
            cfg,
        )
        if (iteration + 1) % cfg["training"]["log_interval"] == 0:
            band_info = (
                f", band_scale={envs[0].band_scale:.3f}" if band_anneal_iters > 0 else ""
            )
            print(
                f"iteration {iteration + 1}/{max_iters}, "
                f"mean return={returns.mean().item():.4f}{band_info}"
            )
        if (iteration + 1) % cfg["training"]["save_interval"] == 0:
            torch.save(policy.state_dict(), output_dir / f"policy_{iteration + 1}.pt")

    torch.save(policy.state_dict(), output_dir / "policy_final.pt")
    print(f"saved final policy to {output_dir / 'policy_final.pt'}")


if __name__ == "__main__":
    main()
