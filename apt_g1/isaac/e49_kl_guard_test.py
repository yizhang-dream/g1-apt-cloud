"""E49-C 确定性测试：KL 信任域守卫的纯 torch 验证（无 isaaclab 依赖）。

覆盖面：
  1. 解析对角高斯 KL 公式 vs torch.distributions.kl_divergence 对拍
     （rsl_rl 口径 KL(new‖old)，动作维 sum、minibatch mean）
  2. kl_guard=None 默认关闭：update() 无守卫键、lr 不变、参数正常更新
     （冻结版行为零回归；expl_var 为唯一无条件新增的日志量）
  3. 极小阈值：每步全回滚 → 参数逐位还原、lr 缩小、连续回滚断路提前结束循环
  4. 巨阈值（=探针等价形态）：零回滚、参数有变化、kl_mb_all 长度 == minibatch 数
  5. grow 恢复路径：小阈值回滚缩 lr 后，巨阈值继续 update → lr 回升且 ≤ 初始 lr
  6. expl_var 口径：returns==values → ≈1；values=大方差独立噪声 → 显著为负

用法（仓库根目录，服务器 CPU 可跑）：
    PYTHONPATH=. python apt_g1/isaac/e49_kl_guard_test.py
全部用例 PASS 时 exit 0。
"""

from __future__ import annotations

import sys

import torch
from torch.distributions import Normal

from apt_g1.isaac.ppo_core import AptPPOPolicy, PPOTrainer, kl_diag_gaussian


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
        flush=True,
    )
    return ok


def _make_policy() -> AptPPOPolicy:
    """小维度合成 policy：obs 8 维、latent(phase) 4 维、aux 4 维（两执行头都进 KL）。

    用 latent_dim=4 的 phase 头（canonical phase_mean/phase_log_std keys），
    aux_executed 默认 True → phase+aux 双头均计入守卫 KL。
    """
    return AptPPOPolicy(
        obs_dim=8, aux_dim=4, use_phase=False, latent_dim=4, hidden_dim=32
    )


def _act_rollout(pol: AptPPOPolicy, T: int, N: int, zd: int, seed: int) -> dict:
    """合成 rollout：logp 由当前 policy 采样自洽生成（ratio≈1，梯度健康，
    不会像随机 logp 那样把 ratio 推到溢出），reward/value 独立随机。"""
    torch.manual_seed(seed)
    obs = torch.randn(T, N, 8)
    with torch.no_grad():
        out, lp, _, val, _ = pol.act(obs.reshape(T * N, 8))
    return {
        "obs": obs,
        "phase": out["phase"].reshape(T, N, zd),
        "aux": out["aux"].reshape(T, N, pol.aux_dim),
        "logp": lp.reshape(T, N),
        "value": val.reshape(T, N),
        "reward": torch.randn(T, N),
        "done": torch.zeros(T, N, dtype=torch.bool),
        "trunc": torch.zeros(T, N, dtype=torch.bool),
        "last_value": val[:N].clone(),
    }


def case1_kl_formula_reference() -> bool:
    """用例 1：kl_diag_gaussian 公式 vs torch.distributions.kl_divergence 对拍。

    KL(new‖old) 的解析式（对角高斯）：
      old_ls - new_ls + (var_new + (mu_new - mu_old)^2) / (2 var_old) - 0.5
    float64 随机 32×64 维，逐样本与 mean 口径均 allclose 1e-6。
    """
    torch.manual_seed(1)
    mu_old = torch.randn(32, 64, dtype=torch.float64)
    ls_old = torch.randn(32, 64, dtype=torch.float64) * 0.3
    mu_new = torch.randn(32, 64, dtype=torch.float64)
    ls_new = torch.randn(32, 64, dtype=torch.float64) * 0.3
    got = kl_diag_gaussian(mu_old, ls_old, mu_new, ls_new)  # 动作维已 sum
    ref = torch.distributions.kl_divergence(
        Normal(mu_new, ls_new.exp()), Normal(mu_old, ls_old.exp())
    ).sum(-1)
    ok = torch.allclose(got, ref, atol=1e-6)
    ok &= torch.allclose(got.mean(), ref.mean(), atol=1e-6)
    return _report(
        "case1 analytic KL vs torch.distributions",
        ok,
        f"max_abs_err={(got - ref).abs().max():.2e}",
    )


def case2_default_off_regression() -> bool:
    """用例 2：kl_guard=None 默认 → update() 无任何守卫键、lr 不变、参数正常
    更新（冻结版行为零回归）；expl_var 是唯一无条件新增键。"""
    torch.manual_seed(2)
    T, N = 2, 32  # T*N = 64, minibatch 32 → 2 minibatch
    pol = _make_policy()
    trainer = PPOTrainer(
        pol, device="cpu", lr=1e-3, minibatch_size=32, num_epochs=1, skip_nan=False
    )
    before = {k: v.clone() for k, v in pol.state_dict().items()}
    stats = trainer.update(_act_rollout(pol, T, N, zd=4, seed=21))
    ok = not any(k in stats for k in ("kl_mb", "kl_roll", "kl_mb_all", "kl_rolls", "lr_now"))
    ok &= trainer._lr_now == trainer._lr0
    ok &= trainer.optimizer.param_groups[0]["lr"] == trainer._lr0
    after = pol.state_dict()
    ok &= any(not torch.equal(before[k], after[k]) for k in before)  # 正常更新
    ok &= "expl_var" in stats
    return _report(
        "case2 kl_guard=None default-off regression",
        ok,
        f"lr={trainer._lr_now} expl_var={stats['expl_var']:.3f}",
    )


def case3_tiny_threshold_full_rollback() -> bool:
    """用例 3：kl_guard=1e-9 极小阈值 → 每个执行了 step 的 minibatch 全回滚。

    断言：update() 后 policy 参数与更新前逐位一致（torch.equal）；lr 被缩小并
    写回 optimizer；max_rolls=2 的连续回滚断路生效（执行数 < 全部 4 个
    minibatch，kl_rolls ≥ max_rolls）。
    """
    torch.manual_seed(3)
    T, N = 2, 32  # T*N = 64, minibatch 16 → 4 minibatch/epoch
    pol = _make_policy()
    trainer = PPOTrainer(
        pol,
        device="cpu",
        lr=1e-3,
        minibatch_size=16,
        num_epochs=1,
        skip_nan=False,
        kl_guard=1e-9,
        kl_guard_max_rolls=2,
    )
    before = {k: v.clone() for k, v in pol.state_dict().items()}
    lr0 = trainer._lr0
    stats = trainer.update(_act_rollout(pol, T, N, zd=4, seed=31))
    after = pol.state_dict()
    ok = all(torch.equal(before[k], after[k]) for k in before)  # 逐位还原
    ok &= trainer._lr_now < lr0
    ok &= trainer.optimizer.param_groups[0]["lr"] == trainer._lr_now
    ok &= stats["kl_rolls"] >= 2  # 连续回滚 ≥ max_rolls
    ok &= len(stats["kl_mb_all"]) < 4  # 提前结束：未跑完全部 minibatch
    ok &= stats["kl_roll"] == 1.0  # 所有执行步均回滚
    return _report(
        "case3 tiny threshold -> full rollback + circuit break",
        ok,
        f"rolls={stats['kl_rolls']} ran={len(stats['kl_mb_all'])}/4 "
        f"lr={trainer._lr_now:.3e}",
    )


def case4_huge_threshold_probe() -> bool:
    """用例 4：kl_guard=1e9 巨阈值（=纯探针等价形态）→ 零回滚、参数有变化、
    kl_mb_all 长度 == minibatch 数。"""
    torch.manual_seed(4)
    T, N = 2, 32  # T*N = 64, minibatch 16 → 4 minibatch/epoch
    pol = _make_policy()
    trainer = PPOTrainer(
        pol,
        device="cpu",
        lr=1e-3,
        minibatch_size=16,
        num_epochs=1,
        skip_nan=False,
        kl_guard=1e9,
    )
    before = {k: v.clone() for k, v in pol.state_dict().items()}
    stats = trainer.update(_act_rollout(pol, T, N, zd=4, seed=41))
    after = pol.state_dict()
    ok = stats["kl_rolls"] == 0
    ok &= len(stats["kl_mb_all"]) == 4  # == minibatch 数
    ok &= all(0.0 <= v < 1e9 for v in stats["kl_mb_all"])
    ok &= any(not torch.equal(before[k], after[k]) for k in before)  # 参数有变化
    ok &= trainer._lr_now <= trainer._lr0  # grow 上限钉在初始 lr
    return _report(
        "case4 huge threshold -> probe-equivalent, zero rollback",
        ok,
        f"kl_mb_all={[f'{v:.3g}' for v in stats['kl_mb_all']]}",
    )


def case5_grow_recovery() -> bool:
    """用例 5：grow 恢复路径——小阈值触发回滚缩 lr 后，切巨阈值继续 update，
    lr 回升且不超过初始 lr（上限钉住）。"""
    torch.manual_seed(5)
    T, N = 2, 32  # T*N = 64, minibatch 16 → 4 minibatch/epoch
    pol = _make_policy()
    trainer = PPOTrainer(
        pol,
        device="cpu",
        lr=1e-3,
        minibatch_size=16,
        num_epochs=1,
        skip_nan=False,
        kl_guard=1e-9,
        kl_guard_max_rolls=99,  # 不断路，保证首次 update 跑完全部 minibatch
    )
    stats1 = trainer.update(_act_rollout(pol, T, N, zd=4, seed=51))
    ok = trainer._lr_now < trainer._lr0  # 回滚已缩 lr
    ok &= stats1["kl_rolls"] >= 1
    lr_after_shrink = trainer._lr_now
    trainer.kl_guard = 1e9  # 切巨阈值：后续步全部通过 → grow 逐 minibatch 回升
    stats2 = trainer.update(_act_rollout(pol, T, N, zd=4, seed=52))
    ok &= stats2["kl_rolls"] == 0
    ok &= trainer._lr_now > lr_after_shrink
    ok &= trainer._lr_now <= trainer._lr0
    return _report(
        "case5 grow recovery path",
        ok,
        f"lr: 1e-3 -> {lr_after_shrink:.3e} -> {trainer._lr_now:.3e}",
    )


def case6_expl_var() -> bool:
    """用例 6：expl_var 口径——T=1、done 全真、gamma=0 → returns == rewards。

    (a) rewards = values → returns == values → 解释方差 = 1；
    (b) rewards 与 values 独立 → var(ret-val) = var(ret)+var(val) ≥ var(ret)
        → expl_var ≤ 0。lr=0 保证参数不动，只看日志量。
    """
    T, N = 1, 256
    pol = _make_policy()
    trainer = PPOTrainer(
        pol,
        device="cpu",
        lr=0.0,
        minibatch_size=T * N,
        num_epochs=1,
        skip_nan=False,
        gamma=0.0,
    )
    # (a) returns == values → expl_var ≈ 1
    r = _act_rollout(pol, T, N, zd=4, seed=61)
    r["reward"] = r["value"].clone()
    r["done"] = torch.ones(T, N, dtype=torch.bool)
    s1 = trainer.update(r)
    ok = s1["expl_var"] > 0.999
    # (b) returns（=rewards）与 values 独立：value 覆写为大方差 fresh-RNG 噪声
    #     （期望 expl_var = -var(val)/var(ret) ≈ -25 量级，免疫 1/√N 协方差
    #     噪声——首版判据 ≤0 在 var(val) 极小时会被 ±0.005 噪声压过，实测
    #     FAIL +0.0049 即此因，非实现问题）
    r2 = _act_rollout(pol, T, N, zd=4, seed=62)
    r2["done"] = torch.ones(T, N, dtype=torch.bool)
    g = torch.Generator().manual_seed(63)
    r2["value"] = torch.randn(T, N, generator=g) * 5.0
    s2 = trainer.update(r2)
    ok &= s2["expl_var"] < -1.0
    return _report("case6 expl_var conventions", ok,
                   f"perfect={s1['expl_var']:.4f} indep={s2['expl_var']:.4f}")


def main() -> int:
    print("=== E49-C KL trust-region guard tests (pure torch, CPU) ===", flush=True)
    results = [
        case1_kl_formula_reference(),
        case2_default_off_regression(),
        case3_tiny_threshold_full_rollback(),
        case4_huge_threshold_probe(),
        case5_grow_recovery(),
        case6_expl_var(),
    ]
    print(f"=== {sum(results)}/{len(results)} cases PASS ===", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
