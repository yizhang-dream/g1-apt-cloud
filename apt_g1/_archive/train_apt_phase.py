"""APT-style joint RL over (phase latent + aux) with a warm-started phase head.

The phase policy is pretrained to imitate the distilled phase router, then
fine-tuned with PPO in MuJoCo (single env). The env converts the policy phase
into a phase-bin prototype token (frozen prior) and applies aux.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from apt_g1.encoder import Command, PhaseRouterEncoder
from apt_g1.policies.phase_aux_policy import PhaseAuxPolicy
from apt_g1.train import load_config, make_mujoco_env


def collect(policy, env, num_steps, device, gamma):
    obs = np.asarray(env.reset(), dtype=np.float32)
    obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
    obs_buf, act_buf, rew_buf, done_buf, val_buf, lp_buf = [], {"phase": [], "aux": []}, [], [], [], []
    for _ in range(num_steps):
        action, log_prob, value = policy.act(obs_t)
        obs_buf.append(obs_t)
        act_buf["phase"].append(action["phase"])
        act_buf["aux"].append(action["aux"])
        lp_buf.append(log_prob)
        val_buf.append(value)
        anp = {k: v.detach().cpu().numpy()[0] for k, v in action.items()}
        next_obs, reward, done, _ = env.step(
            {
                "token": np.zeros(64, dtype=np.float32),
                "aux": anp["aux"].astype(np.float32),
                "phase": anp["phase"].astype(np.float32),
            }
        )
        obs_t = torch.as_tensor(np.asarray(next_obs, dtype=np.float32)[None], device=device)
        rew_buf.append(torch.as_tensor([[float(reward)]], device=device))
        done_buf.append(torch.as_tensor([[bool(done)]], device=device))
    obs = torch.cat(obs_buf)
    acts = {k: torch.cat(v) for k, v in act_buf.items()}
    rewards = torch.cat(rew_buf)
    dones = torch.cat(done_buf)
    values = torch.cat(val_buf)
    logprobs = torch.cat(lp_buf)
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(num_steps)):
        running = rewards[t] + gamma * running * (1.0 - dones[t].float())
        returns[t] = running
    return obs, acts, logprobs, returns, returns - values


def ppo_update(policy, opt, obs, acts, old_lp, returns, adv, cfg):
    ppo = cfg["ppo"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = obs.shape[0]
    fo = obs.reshape(-1, obs.shape[-1])
    fa = {k: v.detach().reshape(-1, v.shape[-1]) for k, v in acts.items()}
    fl = old_lp.detach().reshape(-1)
    fr = returns.detach().reshape(-1)
    fadv = adv.detach().reshape(-1)
    idx = torch.randperm(n, device=obs.device)
    for _ in range(ppo["num_epochs"]):
        for s in range(0, n, ppo["minibatch_size"]):
            i = idx[s : s + ppo["minibatch_size"]]
            lp, ent, val = policy.evaluate_actions(fo[i], {k: v[i] for k, v in fa.items()})
            ratio = torch.exp(lp - fl[i])
            clip = torch.clamp(ratio, 1 - ppo["clip_eps"], 1 + ppo["clip_eps"])
            ploss = -torch.min(ratio * fadv[i], clip * fadv[i]).mean()
            vloss = torch.nn.functional.mse_loss(val, fr[i])
            loss = ploss + ppo["value_coef"] * vloss - ppo["entropy_coef"] * ent.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo["max_grad_norm"])
            opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="apt_g1/configs/flat_g1_apt_aux_c1.yaml")
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--onnx-path", required=True)
    ap.add_argument("--phase-router-dir", required=True)
    ap.add_argument("--max-iters", type=int, default=400)
    ap.add_argument("--pretrain-iters", type=int, default=2000)
    ap.add_argument("--output-dir", default="outputs/apt_phase_aux")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["training"]["output_dir"] = args.output_dir
    device = torch.device("cuda")
    env = make_mujoco_env(cfg, args.onnx_path, args.repo_root, args.phase_router_dir)
    router = env.phase_router
    policy = PhaseAuxPolicy(obs_dim=99, aux_dim=12).to(device)

    # --- supervised warm start: collect (env obs -> router phase) pairs ---
    import mujoco

    obs_pairs, label_pairs = [], []
    for ep in range(3):
        env.reset()
        for _ in range(1000):
            obs = env.get_obs().astype(np.float32)
            cmd_c = Command.from_vxvy(
                float(env.command[0]), float(env.command[1]), 0.0
            )
            gi, sc = router.phase_raw(cmd_c, env._get_sonic_history())
            obs_pairs.append(obs)
            label_pairs.append(sc)
            tok = router.encode(cmd_c, env._get_sonic_history())
            env.step({"token": tok, "aux": np.zeros(12, dtype=np.float32)})
    Xt = torch.from_numpy(np.stack(obs_pairs)).cuda()
    Lt = torch.from_numpy(np.stack(label_pairs)).cuda()
    n = len(Xt)
    print(f"warm-start pairs: {n}")
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()
    for it in range(args.pretrain_iters):
        idx = torch.randint(0, n, (512,), device=device)
        opt.zero_grad()
        pm = policy.phase_mean(policy.encoder(Xt[idx]))
        loss = lossf(pm, Lt[idx])
        loss.backward()
        opt.step()
        if (it + 1) % 500 == 0:
            print(f"pretrain {it+1} loss {loss.item():.5f}")
    with torch.no_grad():
        policy.phase_log_std.data.fill_(math.log(0.02))
        policy.aux_log_std.data.fill_(math.log(0.01))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out / "phase_head_pretrained.pt")

    opt = torch.optim.Adam(policy.parameters(), lr=cfg["ppo"]["lr"])
    for it in range(args.max_iters):
        obs, acts, lp, ret, adv = collect(policy, env, cfg["training"]["num_steps"], device, cfg["ppo"]["gamma"])
        ppo_update(policy, opt, obs, acts, lp, ret, adv, cfg)
        if (it + 1) % 25 == 0:
            print(f"iter {it+1}/{args.max_iters} mean_return={ret.mean().item():.2f}")
            torch.save(policy.state_dict(), out / f"policy_{it+1}.pt")
    torch.save(policy.state_dict(), out / "policy_final.pt")
    print("saved", out / "policy_final.pt")


if __name__ == "__main__":
    main()
