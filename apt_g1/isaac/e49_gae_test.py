"""E49 确定性测试：PPO 训练器修复的纯 torch 验证（无 isaaclab 依赖）。

覆盖修复面：
  1. GAE 边界：terminated(摔倒)/truncated(超时) 都切断优势递推且不自举
     （Isaac DirectRLEnv 边界返回复位后 obs，终末状态价值不可得——已知
     残留限制，本测试锁定"切断"这一约束而非自举值）
  2. aux_executed=False 时 log_prob/entropy 不含 aux 项（act 与 update 同约定）
  3. update() 真 epoch 循环（num_epochs 生效，每 epoch 重新洗牌）
  4. approx_kl(k3) / clip_frac / act_std 指标口径 + kl → kl_prior 更名

用法（仓库根目录，服务器 CPU 可跑）：
    PYTHONPATH=. python apt_g1/isaac/e49_gae_test.py
全部用例 PASS 时 exit 0。
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from torch.distributions import Categorical, Normal

from apt_g1.isaac.ppo_core import AptPPOPolicy, PPOTrainer


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
        flush=True,
    )
    return ok


def case1_reference_gae() -> bool:
    """用例 1：与测试内独立写的朴素 GAE 对拍（逐 env 显式累加，不同代码风格）。"""
    gamma, lam = 0.97, 0.9
    T, N = 24, 16
    rng = np.random.default_rng(7)
    # float64 输入：让对拍误差落在 ~1e-12（float32 下 24 步递推的舍入累积
    # 可达 ~1e-5，会淹没 atol=1e-6 的判定）
    rewards = torch.tensor(rng.normal(0.0, 1.0, (T, N)), dtype=torch.float64)
    values = torch.tensor(rng.normal(0.0, 2.0, (T, N)), dtype=torch.float64)
    dones = torch.from_numpy(rng.random((T, N)) < 0.2)  # 边界密度 ~20%
    truncs = torch.from_numpy(rng.random((T, N)) < 0.2)
    last_value = torch.tensor(rng.normal(0.0, 1.0, N), dtype=torch.float64)

    trainer = PPOTrainer(
        AptPPOPolicy(obs_dim=4), gamma=gamma, lam=lam, device="cpu"
    )
    got = trainer.compute_gae(rewards, values, dones, truncs, last_value)

    # 朴素参考实现：float 纯量 + 逐 env 循环，刻意不用向量化递推
    expected = np.zeros((T, N), dtype=np.float64)
    for n in range(N):
        acc = 0.0
        for t in range(T - 1, -1, -1):
            cont = 0.0 if (bool(dones[t, n]) or bool(truncs[t, n])) else 1.0
            nxt = float(last_value[n]) if t == T - 1 else float(values[t + 1, n])
            delta = float(rewards[t, n]) + gamma * nxt * cont - float(values[t, n])
            acc = delta + gamma * lam * cont * acc
            expected[t, n] = acc
    exp_t = torch.tensor(expected, dtype=torch.float64)
    ok = torch.allclose(got, exp_t, atol=1e-6)
    return _report(
        "case1 naive-GAE reference", ok, f"max_abs_err={(got - exp_t).abs().max():.2e}"
    )


def case2_hand_example() -> bool:
    """用例 2：手算小例 γ=0.9 λ=0.8 T=4 N=1，t=2 处 done=1。

    手工推导（自后向前；除 t=2 外 cont=1）：
      t=3: cont=1, next=last_value=10 → delta = 4 + 0.9*10 - 8 = 5.0
                                        adv3 = 5.0
      t=2: cont=0（done 切断，不自举）→ delta = 3 + 0.9*8*0 - 7 = -4.0
                                        adv2 = -4.0 + 0.72*0*5.0 = -4.0
      t=1: cont=1, next=values[2]=7  → delta = 2 + 0.9*7 - 6 = 2.3
                                        adv1 = 2.3 + 0.72*(-4.0) = -0.58
      t=0: cont=1, next=values[1]=6  → delta = 1 + 0.9*6 - 5 = 1.4
                                        adv0 = 1.4 + 0.72*(-0.58) = 0.9824
    旧缺陷行为（cont 只看 trunc，t=2 仍 cont=1）会给出
    adv2 = 3 + 0.9*8 - 7 + 0.72*5.0 = 6.8 —— 与手算不符，本用例可区分修复前后。
    """
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    values = torch.tensor([[5.0], [6.0], [7.0], [8.0]])
    dones = torch.tensor([[False], [False], [True], [False]])
    truncs = torch.zeros(4, 1, dtype=torch.bool)
    last_value = torch.tensor([10.0])
    trainer = PPOTrainer(AptPPOPolicy(obs_dim=4), gamma=0.9, lam=0.8, device="cpu")
    got = trainer.compute_gae(rewards, values, dones, truncs, last_value)
    expected = torch.tensor([[0.9824], [-0.58], [-4.0], [5.0]])
    ok = torch.allclose(got, expected, atol=1e-6)
    return _report("case2 hand-computed boundary example", ok,
                   f"got={got.flatten().tolist()}")


def case3_boundary_invariance() -> bool:
    """用例 3：边界切断不变性——done/trunc 步之后的量扰动不影响边界步及之前的 adv。"""
    T, N = 8, 3
    torch.manual_seed(11)
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    trainer = PPOTrainer(AptPPOPolicy(obs_dim=4), device="cpu")

    def adv_of(done, trunc, r, v, lv):
        return trainer.compute_gae(r, v, done, trunc, lv)

    def flags(done_steps, trunc_steps):
        d = torch.zeros(T, N, dtype=torch.bool)
        tr = torch.zeros(T, N, dtype=torch.bool)
        for n, t in done_steps.items():
            d[t, n] = True
        for n, t in trunc_steps.items():
            tr[t, n] = True
        return d, tr

    ok = True
    # (a) done 步之后扰动 rewards/values（×100）→ done 步及之前 adv 不变
    done_steps = {0: 2, 1: 5}  # env2 无边界，作正对照（其 adv 必须变化）
    d, tr = flags(done_steps, {})
    base = adv_of(d, tr, rewards, values, torch.randn(N))
    r2, v2 = rewards.clone(), values.clone()
    for n, t in done_steps.items():
        r2[t + 1:, n] *= 100.0
        v2[t + 1:, n] *= 100.0
    pert = adv_of(d, tr, r2, v2, torch.randn(N) * 100.0)  # last_value 一并放大，
    # 正对照（env2 无边界）的差异量级才有保证；env0/1 的断言窗口被 done 切断，
    # 不受 last_value 影响
    ok &= torch.allclose(base[: done_steps[0] + 1, 0], pert[: done_steps[0] + 1, 0], atol=1e-6)
    ok &= torch.allclose(base[: done_steps[1] + 1, 1], pert[: done_steps[1] + 1, 1], atol=1e-6)
    ok &= not torch.allclose(base[:, 2], pert[:, 2], atol=1e-6)  # 正对照

    # (b) done 恰在 T-1：last_value 被 cont 掩码，扰动它不影响任何一步 adv
    d_b = torch.zeros(T, N, dtype=torch.bool)
    d_b[T - 1] = True
    tr_b = torch.zeros(T, N, dtype=torch.bool)
    base_b = adv_of(d_b, tr_b, rewards, values, torch.randn(N))
    pert_b = adv_of(d_b, tr_b, rewards, values, torch.randn(N) * 100.0)
    ok &= torch.allclose(base_b, pert_b, atol=1e-6)

    # (c) trunc 步同理切断递推（且不自举）
    trunc_steps = {0: 3, 2: 6}
    d_c, tr_c = flags({}, trunc_steps)
    base_c = adv_of(d_c, tr_c, rewards, values, torch.randn(N))
    r3, v3 = rewards.clone(), values.clone()
    for n, t in trunc_steps.items():
        r3[t + 1:, n] *= 100.0
        v3[t + 1:, n] *= 100.0
    pert_c = adv_of(d_c, tr_c, r3, v3, torch.randn(N))
    ok &= torch.allclose(base_c[: trunc_steps[0] + 1, 0], pert_c[: trunc_steps[0] + 1, 0], atol=1e-6)
    ok &= torch.allclose(base_c[: trunc_steps[2] + 1, 2], pert_c[: trunc_steps[2] + 1, 2], atol=1e-6)
    return _report("case3 boundary-cut invariance (done/trunc/last_value)", ok)


def case4_aux_removal() -> bool:
    """用例 4：aux_executed=False 的 policy，logp/entropy 与 aux 采样值无关，
    且 entropy == phase(+gate) 头单独的熵。"""
    torch.manual_seed(3)
    obs = torch.randn(5, 8)
    pol = AptPPOPolicy(obs_dim=8, aux_dim=12, use_phase=False, latent_dim=16, gate_k=2)
    pol.aux_executed = False

    # (i) 返回的 logp/entropy 逐项等于 phase+gate 项（对实际采样的 phase/gate）
    out, lp, ent, _, p = pol.act(obs)
    pdist = Normal(p["phase_mean"], p["phase_log_std"].exp())
    gdist = Categorical(logits=p["gate_logits"])
    lp_expected = pdist.log_prob(out["phase"]).sum(-1) + gdist.log_prob(out["gate"])
    ent_expected = pdist.entropy().sum(-1) + gdist.entropy()
    ok = torch.allclose(lp, lp_expected, atol=1e-6)
    ok &= torch.allclose(ent, ent_expected, atol=1e-6)

    # (ii) 对 aux 采样值不敏感：扰动 aux 头参数（必然改变 aux 采样值），
    #      同 obs 确定性模式下 logp/entropy 必须不变
    out_d, lp_d, ent_d, _, _ = pol.act(obs, deterministic=True)
    with torch.no_grad():
        pol.aux_mean.weight.add_(torch.randn_like(pol.aux_mean.weight))
        pol.aux_mean.bias.add_(torch.randn_like(pol.aux_mean.bias))
        pol.aux_log_std.add_(torch.randn_like(pol.aux_log_std))
    out_d2, lp_d2, ent_d2, _, _ = pol.act(obs, deterministic=True)
    ok &= torch.allclose(lp_d, lp_d2, atol=1e-6)
    ok &= torch.allclose(ent_d, ent_d2, atol=1e-6)
    ok &= not torch.allclose(out_d["aux"], out_d2["aux"])  # aux 采样值确实变了

    # (iii) 无 gate 的 latent-only policy：entropy == 仅 phase 头的熵
    pol2 = AptPPOPolicy(obs_dim=8, aux_dim=12, use_phase=False, latent_dim=16, gate_k=0)
    pol2.aux_executed = False
    out2, lp2, ent2, _, p2 = pol2.act(obs)
    pdist2 = Normal(p2["phase_mean"], p2["phase_log_std"].exp())
    ok &= torch.allclose(ent2, pdist2.entropy().sum(-1), atol=1e-6)
    ok &= torch.allclose(lp2, pdist2.log_prob(out2["phase"]).sum(-1), atol=1e-6)
    return _report("case4 aux-removal invariants (act side)", ok)


def case5_epoch_loop() -> bool:
    """用例 5：num_epochs=2、T*N=1024、minibatch=512 → optimizer.step 恰 4 次。"""
    torch.manual_seed(5)
    T, N, D = 4, 256, 6  # T*N = 1024
    pol = AptPPOPolicy(obs_dim=D, aux_dim=12, use_phase=True)
    trainer = PPOTrainer(pol, device="cpu", num_epochs=2, minibatch_size=512,
                         skip_nan=False)
    rollout = {
        "obs": torch.randn(T, N, D),
        "phase": torch.randn(T, N, 2),
        "aux": torch.randn(T, N, 12),
        "logp": torch.randn(T, N) * 0.1 - 30.0,
        "value": torch.randn(T, N),
        "reward": torch.randn(T, N),
        "done": torch.zeros(T, N, dtype=torch.bool),
        "trunc": torch.zeros(T, N, dtype=torch.bool),
        "last_value": torch.randn(N),
    }
    calls = {"step": 0}
    orig_step = trainer.optimizer.step

    def counting_step(*a, **kw):
        calls["step"] += 1
        return orig_step(*a, **kw)

    trainer.optimizer.step = counting_step
    trainer.update(rollout)
    ok = calls["step"] == 4  # 2 epochs × (1024/512) minibatch
    return _report("case5 epoch loop step count", ok,
                   f"steps={calls['step']} (expect 4)")


def case6_metrics() -> bool:
    """用例 6：lp == logp_old → approx_kl≈0、clip_frac=0；随机扰动后
    clip_frac∈(0,1]、approx_kl ≥ -1e-8（k3 估计逐点非负）。
    直接复用 update() 返回的 agg 字段断言（这些量在 update 内部计算）。"""
    torch.manual_seed(9)
    T, N, D, ZDIM = 2, 128, 6, 8  # T*N = 256 = minibatch → 单 minibatch
    pol = AptPPOPolicy(obs_dim=D, use_phase=False, latent_dim=ZDIM)
    pol.aux_executed = False
    # lr=0：optimizer.step 不改参数 → update 内部重算的 lp 与预存的 logp_old
    # 出自同一参数，逐样本 ratio=1
    trainer = PPOTrainer(pol, device="cpu", lr=0.0, num_epochs=1,
                         minibatch_size=T * N)
    obs = torch.randn(T, N, D)
    with torch.no_grad():
        out, lp, _, val, _ = pol.act(obs.reshape(T * N, D))
    rollout = {
        "obs": obs,
        "phase": out["phase"].reshape(T, N, ZDIM),
        "aux": out["aux"].reshape(T, N, pol.aux_dim),
        "logp": lp.reshape(T, N),
        "value": val.reshape(T, N),
        "reward": torch.randn(T, N),
        "done": torch.zeros(T, N, dtype=torch.bool),
        "trunc": torch.zeros(T, N, dtype=torch.bool),
        "last_value": val[:N].clone(),
    }
    stats = trainer.update(rollout)
    ok = abs(stats["approx_kl"]) < 1e-6
    ok &= stats["clip_frac"] == 0.0

    # 随机扰动 logp_old → ratio 偏离 1：clip_frac 落在 (0,1]，approx_kl 非负
    rollout2 = dict(rollout)
    rollout2["logp"] = rollout["logp"] + torch.randn_like(rollout["logp"])
    stats2 = trainer.update(rollout2)
    ok &= 0.0 < stats2["clip_frac"] <= 1.0
    ok &= stats2["approx_kl"] >= -1e-8
    # 更名同步：新键在、旧 "kl" 键不复存在
    ok &= "kl_prior" in stats and "kl" not in stats
    return _report(
        "case6 ppo metrics sanity",
        ok,
        f"kl0={stats['approx_kl']:.2e} cf0={stats['clip_frac']} "
        f"akl={stats2['approx_kl']:.4f} cf={stats2['clip_frac']:.3f}",
    )


def main() -> int:
    print("=== E49 PPO-trainer fix tests (pure torch, CPU) ===", flush=True)
    results = [
        case1_reference_gae(),
        case2_hand_example(),
        case3_boundary_invariance(),
        case4_aux_removal(),
        case5_epoch_loop(),
        case6_metrics(),
    ]
    print(f"=== {sum(results)}/{len(results)} cases PASS ===", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
