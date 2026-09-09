"""D048i 确定性测试：PPO 更新约束臂的步级解析 KL 看门 + 资格检查（无 isaaclab 依赖）。

纯 torch CPU（policy / optimizer / update 全在 CPU 上跑合成 rollout，无需 Isaac）。
覆盖面：
  1. kl_diag_gaussian_reverse 闭式 KL(old‖new) vs 手算标量对照 +
     torch.distributions.kl_divergence oracle 双对拍
  2. KL(同分布)=0：对角高斯（res/z 头）与离散 gate 头（含 oracle 对拍）
  3. ksg_snapshot/ksg_restore：一步真实 Adam step 后还原——参数逐位一致 +
     Adam 状态（exp_avg/exp_avg_sq/step 计数）逐位一致（还原前先证明第二步
     确实移动了参数与状态，排除平凡通过）
  4. 超大 lr（1e6）极小策略 → 看门全程拒绝：rejected==total>0、轮后参数与
     轮前逐位一致（param_rel_move==0）、lr_scale 触底 2^-6、资格检查无违约
  5. 中等 lr → 先以 target=inf 探得全步长 KL，再以 0.5×KL 为阈重跑同一步：
     全步长超阈、回退后以 lr_scale<1 接受（Adam 首步位移 ∝ lr，KL ∝ lr²，
     0.5×lr ≈ 0.25×KL < 0.5×KL）
  6. 篡改 logp_old 后 ksg_qualification 函数级检出（Δlogp / ratio 偏差超容差；
     未篡改时同策略前向偏差 ≈ 0 作对照）
  7. guard 关闭零回归：同种子 guard-on/off 两次 update() 的核心 stats 逐位
     一致，且 guard-off 的 stats 键集合不含任何 D048i 新键

用法（服务器 .venv_isaac；本机无 torch 仅 py_compile）：
    PYTHONPATH=. python apt_g1/isaac/test_kl_step_guard.py
全部用例 PASS 时 exit 0。
"""

from __future__ import annotations

import math
import sys

import torch
from torch.distributions import Categorical, Normal, kl_divergence

from apt_g1.isaac.ppo_core import (
    AptPPOPolicy,
    PPOTrainer,
    kl_categorical_reverse,
    kl_diag_gaussian,
    kl_diag_gaussian_reverse,
)

# D048i guard 开启时 update() stats 的新键全集（与 train_apt_isaac.py
# KSG_HIST_KEYS 一致；用于 case7 的键集合检查）
KSG_STATS_KEYS = (
    "qual_logp_maxdev",
    "qual_ratio_maxdev",
    "qual_kl_self",
    "kl_steps_total",
    "kl_steps_accepted",
    "kl_steps_rejected",
    "kl_lr_scale_min",
    "kl_lr_scale_mean",
    "kl_analytic_joint_mean",
    "kl_analytic_joint_max",
    "kl_analytic_z_mean",
    "kl_analytic_res_mean",
    "kl_gate_mean",
    "param_rel_move",
)


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
        flush=True,
    )
    return ok


def _mb_loss(policy: AptPPOPolicy, obs: torch.Tensor) -> torch.Tensor:
    """测试用的 minibatch 损失（同时驱动 aux 与 phase 两头的梯度）。"""
    p = policy.forward_actor(obs)
    la = Normal(p["aux_mean"], p["aux_log_std"].exp()).log_prob(
        obs[:, : policy.aux_dim]
    ).sum(-1).mean()
    lp = Normal(p["phase_mean"], p["phase_log_std"].exp()).log_prob(
        obs[:, :2]
    ).sum(-1).mean()
    return -(la + lp)


def _rollout(policy: AptPPOPolicy, T: int = 4, N: int = 8, seed: int = 0) -> dict:
    """用 policy.act 采集一份合成 rollout（与 update() 所需键位对齐）。"""
    torch.manual_seed(seed)
    D, aux_dim = policy.obs_dim, policy.aux_dim
    obs = torch.randn(T, N, D)
    has_phase = bool(policy.use_phase or policy.latent_dim > 0)
    rl = {
        "obs": obs,
        "aux": torch.zeros(T, N, aux_dim),
        "logp": torch.zeros(T, N),
        "value": torch.zeros(T, N),
        "reward": torch.randn(T, N),
        "done": torch.zeros(T, N, dtype=torch.bool),
        "trunc": torch.zeros(T, N, dtype=torch.bool),
    }
    if has_phase:
        pdim = policy.latent_dim if policy.latent_dim > 0 else 2
        ph = torch.zeros(T, N, pdim)
    o = obs[0]
    for t in range(T):
        act, logp, _ent, val, _p = policy.act(o)
        rl["aux"][t] = act["aux"].detach()
        if has_phase:
            ph[t] = act["phase"].detach()
        rl["logp"][t] = logp.detach()
        rl["value"][t] = val.detach()
        o = obs[t + 1] if t + 1 < T else obs[T - 1]
    if has_phase:
        rl["phase"] = ph
    rl["last_value"] = torch.zeros(N)
    return rl


# ---------------------------------------------------------------------------
def case1_kl_closed_form() -> bool:
    # 已知 μ/σ 手算标量：KL(old‖new) = log(σn/σo) + (σo²+(μo−μn)²)/(2σn²) − 1/2
    mu_o = torch.tensor([[0.3, -0.7]])
    ls_o = torch.tensor([[0.2, 0.5]])
    mu_n = torch.tensor([[-0.1, 0.4]])
    ls_n = torch.tensor([[-0.3, 0.1]])
    hand = 0.0
    for i in range(2):
        so, sn = math.exp(ls_o[0, i].item()), math.exp(ls_n[0, i].item())
        hand += math.log(sn / so) + (so**2 + (mu_o[0, i].item() - mu_n[0, i].item()) ** 2) / (
            2.0 * sn**2
        ) - 0.5
    got = kl_diag_gaussian_reverse(mu_o, ls_o, mu_n, ls_n)
    ok = torch.allclose(got, torch.tensor([hand]), atol=1e-6)
    # oracle：torch.distributions 的 KL(p||q)（old 在前 = 第一参数）
    ref = kl_divergence(
        Normal(mu_o, ls_o.exp()), Normal(mu_n, ls_n.exp())
    ).sum(-1)
    ok = ok and torch.allclose(got, ref, atol=1e-5)
    # 方向性对照：与 rsl_rl 口径的 kl_diag_gaussian（KL(new‖old)）不相等
    rsl = kl_diag_gaussian(mu_o, ls_o, mu_n, ls_n)
    ok = ok and not torch.allclose(got, rsl, atol=1e-3)
    return _report(
        "1 closed-form KL(old||new) vs hand scalar + torch oracle",
        ok,
        f"got={got.item():.8f} hand={hand:.8f} ref={ref.item():.8f}",
    )


def case2_kl_self_zero() -> bool:
    torch.manual_seed(1)
    mu = torch.randn(5, 3)
    ls = torch.randn(5, 3) * 0.3
    g = kl_diag_gaussian_reverse(mu, ls, mu.clone(), ls.clone())
    ok = bool((g.abs() <= 1e-12).all().item())
    # gate 离散头：同 logits → 0；不同 logits → 与 oracle 一致
    lo = torch.randn(5, 4)
    g0 = kl_categorical_reverse(lo, lo.clone())
    ok = ok and bool((g0.abs() <= 1e-12).all().item())
    ln = torch.randn(5, 4)
    g2 = kl_categorical_reverse(lo, ln)
    ref = kl_divergence(Categorical(logits=lo), Categorical(logits=ln))
    ok = ok and torch.allclose(g2, ref, atol=1e-6)
    return _report("2 KL(same dist) == 0 (gaussian + categorical)", ok,
                   f"max|g|={g.abs().max().item():.2e} max|g0|={g0.abs().max().item():.2e}")


def case3_snapshot_restore() -> bool:
    torch.manual_seed(21)
    policy = AptPPOPolicy(obs_dim=6, aux_dim=3, gate_k=0, hidden_dim=8)
    tr = PPOTrainer(policy, lr=3e-4, device="cpu", kl_step_guard=True)
    # 一步真实 step：填出 Adam 状态（exp_avg/exp_avg_sq/step）
    loss = sum(p.pow(2).sum() for p in policy.parameters())
    tr.optimizer.zero_grad()
    loss.backward()
    tr.optimizer.step()
    snap = tr.ksg_snapshot()
    # 第二步（反号梯度 → 参数必动、Adam 状态必前进；注意必须重建损失图，
    # 首次 backward 已释放原图）
    loss2 = -sum(p.pow(2).sum() for p in policy.parameters())
    tr.optimizer.zero_grad()
    loss2.backward()
    tr.optimizer.step()
    moved = max(
        (p - s).abs().max().item()
        for p, s in zip(policy.parameters(), snap[0])
    )
    so = snap[1]
    state_moved = False
    for i, p in enumerate(policy.parameters()):
        cur = tr.optimizer.state.get(p, {})
        ref = so["state"].get(i, {})
        for kk in ("exp_avg", "exp_avg_sq", "step"):
            if kk in ref and kk in cur:
                if not torch.equal(cur[kk], ref[kk]):
                    state_moved = True
    ok = moved > 0.0 and state_moved
    # 还原：参数 + Adam 状态逐位精确
    tr.ksg_restore(snap)
    ok = ok and all(
        torch.equal(p, s) for p, s in zip(policy.parameters(), snap[0])
    )
    for i, p in enumerate(policy.parameters()):
        cur = tr.optimizer.state.get(p, {})
        ref = so["state"].get(i, {})
        for kk in ("exp_avg", "exp_avg_sq", "step"):
            if kk in ref:
                ok = ok and kk in cur and torch.equal(cur[kk], ref[kk])
    return _report(
        "3 snapshot/restore: params + Adam state bitwise exact",
        ok,
        f"moved={moved:.3e} state_moved={state_moved}",
    )


def case4_huge_lr_all_rejected() -> bool:
    torch.manual_seed(31)
    policy = AptPPOPolicy(obs_dim=6, aux_dim=3, gate_k=0, hidden_dim=8)
    tr = PPOTrainer(
        policy, lr=1e6, device="cpu", kl_step_guard=True,
        minibatch_size=16, kl_backtracks=6,
    )
    rl = _rollout(policy, T=4, N=8, seed=32)
    theta_before = [p.detach().clone() for p in policy.parameters()]
    stats = tr.update(rl)
    total = stats["kl_steps_total"]
    ok = total > 0
    ok = ok and stats["kl_steps_rejected"] == total
    ok = ok and stats["kl_steps_accepted"] == 0
    # 轮后参数与轮前逐位一致（每个被拒步都还原快照）
    ok = ok and all(
        torch.equal(p.detach(), s) for p, s in zip(policy.parameters(), theta_before)
    )
    ok = ok and stats["param_rel_move"] == 0.0
    # lr_scale 触底 = 0.5^backtracks
    floor = 0.5**6
    ok = ok and stats["kl_lr_scale_min"] == floor
    ok = ok and stats["kl_lr_scale_mean"] == floor
    # 资格检查同策略自算，不应违约
    ok = ok and stats["qual_ratio_maxdev"] < 1e-3
    ok = ok and stats["qual_kl_self"] <= 1e-8
    return _report(
        "4 huge lr tiny policy -> all steps rejected, params untouched",
        ok,
        f"total={total} lr_min={stats['kl_lr_scale_min']:.6g} "
        f"prm={stats['param_rel_move']:.2e}",
    )


def case5_moderate_lr_backtrack_accept() -> bool:
    torch.manual_seed(41)
    policy = AptPPOPolicy(obs_dim=6, aux_dim=3, gate_k=0, hidden_dim=8)
    tr = PPOTrainer(
        policy, lr=0.5, device="cpu", kl_step_guard=True,
        kl_backtracks=6, max_grad_norm=0.5,
    )
    torch.manual_seed(42)
    obs = torch.randn(16, 6)
    mb = torch.arange(16)
    old_buf = {
        k: v.detach().clone() for k, v in policy.forward_actor(obs).items()
    }
    lr_round = [float(g["lr"]) for g in tr.optimizer.param_groups]

    def _prep_grads() -> None:
        tr.optimizer.zero_grad()
        _mb_loss(policy, obs).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)

    # 探针 1：target=inf → 全步长必被接受，量得全步长 KL
    _prep_grads()
    snap = tr.ksg_snapshot()
    w1 = tr.ksg_watch_step(obs, mb, old_buf, lr_round, target=float("inf"))
    ok = w1["accepted"] and w1["lr_scale"] == 1.0 and w1["joint"] > 0.0
    kl_full = w1["joint"]
    # 探针 2：还原后以 0.5×KL 为阈重跑同一步——全步长超阈，lr 减半回退后接受
    # （Adam 首步位移 ∝ lr，解析 KL ∝ 位移² → 0.5×lr ≈ 0.25×KL < 0.5×KL）
    tr.ksg_restore(snap)
    _prep_grads()
    w2 = tr.ksg_watch_step(obs, mb, old_buf, lr_round, target=0.5 * kl_full)
    ok = ok and w2["accepted"] and 0.0 < w2["lr_scale"] < 1.0
    ok = ok and w2["joint"] <= 0.5 * kl_full
    return _report(
        "5 moderate lr -> backtracked step accepted with lr_scale < 1",
        ok,
        f"kl_full={kl_full:.4g} w2.scale={w2['lr_scale']:.4g} "
        f"w2.kl={w2['joint']:.4g} bt={w2['backtracks']}",
    )


def case6_qualification_detects_tamper() -> bool:
    torch.manual_seed(51)
    policy = AptPPOPolicy(obs_dim=6, aux_dim=3, gate_k=0, hidden_dim=8)
    tr = PPOTrainer(policy, lr=3e-4, device="cpu", kl_step_guard=True)
    rl = _rollout(policy, T=3, N=8, seed=52)
    T, N = rl["obs"].shape[:2]
    obs_f = rl["obs"].reshape(T * N, -1)
    phase_f = rl["phase"].reshape(T * N, -1)
    aux_f = rl["aux"].reshape(T * N, -1)
    logp_f = rl["logp"].reshape(-1)
    # 未篡改：同策略前向，重算 logp 与存储值一致
    q0 = tr.ksg_qualification(obs_f, phase_f, aux_f, None, logp_f, aux_scored=True)
    ok = q0["qual_logp_maxdev"] <= 1e-5 and q0["qual_ratio_maxdev"] <= 1e-5
    # 篡改 logp_old：函数级检出（Δlogp ≈ 0.5，ratio 偏差 ≈ e^0.5−1 ≈ 0.649）
    q1 = tr.ksg_qualification(
        obs_f, phase_f, aux_f, None, logp_f + 0.5, aux_scored=True
    )
    ok = ok and q1["qual_logp_maxdev"] >= 0.49
    ok = ok and q1["qual_ratio_maxdev"] > 1e-3  # --qual-ratio-tol 默认值
    return _report(
        "6 tampered logp_old detected by qualification (function level)",
        ok,
        f"q0.ratio={q0['qual_ratio_maxdev']:.2e} "
        f"q1.logp={q1['qual_logp_maxdev']:.4f} "
        f"q1.ratio={q1['qual_ratio_maxdev']:.4f}",
    )


def case7_guard_off_zero_regression() -> bool:
    def _run_once(ksg: bool, seed: int) -> dict:
        torch.manual_seed(seed)
        policy = AptPPOPolicy(obs_dim=6, aux_dim=3, gate_k=0, hidden_dim=8)
        tr = PPOTrainer(
            policy, lr=3e-4, device="cpu", minibatch_size=16,
            kl_step_guard=ksg, kl_step_target=0.05, kl_backtracks=6,
        )
        rl = _rollout(policy, T=3, N=8, seed=seed + 100)
        return tr.update(rl)

    s_off = _run_once(False, 61)
    s_on = _run_once(True, 61)
    core = [
        "loss", "ploss", "vloss", "ent", "kl_prior", "approx_kl",
        "clip_frac", "act_std", "ent_aux", "ent_z", "aux_std", "expl",
        "dreg", "nan_skip", "expl_var", "post_update_kl",
    ]
    # guard 开/关核心统计逐位一致（guard 前向无采样、不耗 RNG、不进损失图）
    ok = all(s_off[k] == s_on[k] for k in core)
    # guard-off 键集合不含任何 D048i 新键；guard-on 全部在
    ok = ok and not any(k in s_off for k in KSG_STATS_KEYS)
    ok = ok and all(k in s_on for k in KSG_STATS_KEYS)
    return _report(
        "7 guard-off zero regression (bitwise stats, key sets)",
        ok,
        f"prm(on)={s_on['param_rel_move']:.3e} "
        f"steps={s_on['kl_steps_total']} acc={s_on['kl_steps_accepted']}",
    )


def main() -> None:
    cases = [
        case1_kl_closed_form,
        case2_kl_self_zero,
        case3_snapshot_restore,
        case4_huge_lr_all_rejected,
        case5_moderate_lr_backtrack_accept,
        case6_qualification_detects_tamper,
        case7_guard_off_zero_regression,
    ]
    results = [c() for c in cases]
    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} cases PASS", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
