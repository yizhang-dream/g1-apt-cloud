"""E49-C 确定性测试：KL 信任域守卫的纯 torch 验证（无 isaaclab 依赖）。

v2 口径：守卫 old 侧参照 = rollout 采样时存储的旧分布（数据参照），不再是
v1 的"步前 forward"步内参照——后者对只改方差 / 连续小步累计漂移两 类失效
（见 case7/8/9）。覆盖面：
  1. 解析对角高斯 KL 公式 vs torch.distributions.kl_divergence 对拍
     （rsl_rl 口径 KL(new‖old)，动作维 sum、minibatch mean）
  2. kl_guard=None 默认关闭：update() 无守卫键、lr 不变、参数正常更新
     （冻结版行为零回归；expl_var 为唯一无条件新增的日志量）
  3. 极小阈值：每步全回滚 → 参数逐位还原、lr 缩小、连续回滚断路提前结束循环
  4. 巨阈值（=探针等价形态）：零回滚、参数有变化、kl_mb_all 长度 == minibatch 数
  5. grow 恢复路径：小阈值回滚缩 lr 后，巨阈值继续 update → lr 回升且 ≤ 初始 lr
  6. expl_var 口径：returns==values → ≈1；values=大方差独立噪声 → 显著为负
  7. 方差盲复现（v1 别名缺陷）：仅 log_std 变化的更新 kl_mb > 0 且 ≈ 解析值
     （旧代码在此恒报 0 → 本 case 旧代码必 FAIL）
  8. 累计超阈（v1 步内参照缺陷）：每步单独位移小、相对 rollout 参照累计超阈
     → 第 1 步通过、后续步全拒，kl_mb 序列单调不减
  9. 拒绝步全恢复（v2 新增回滚范围）：参数 + Adam state（exp_avg/exp_avg_sq/
     step）逐张量还原

用法（仓库根目录，服务器 CPU 可跑）：
    PYTHONPATH=. python apt_g1/isaac/e49_kl_guard_test.py
全部用例 PASS 时 exit 0。
"""

from __future__ import annotations

import math
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
    不会像随机 logp 那样把 ratio 推到溢出），reward/value 独立随机。

    v2：附带 KL 守卫所需的旧分布 buffer（与 train_apt_isaac.py 同口径：从
    act() 第 5 个返回值 p（forward 原始 dict，新旧代码都有）取参，log_std
    必须先 detach().clone() 再存——它是参数 expand 视图，detach 不解除共享
    存储；均值是 Linear 新输出无别名问题）。
    """
    torch.manual_seed(seed)
    obs = torch.randn(T, N, 8)
    with torch.no_grad():
        out, lp, _, val, p = pol.act(obs.reshape(T * N, 8))
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
        # E49-C v2：旧分布 buffer（kl_guard 开启时 update() 必需）
        "phase_mean_old": p["phase_mean"].detach().reshape(T, N, zd),
        "phase_log_std_old": p["phase_log_std"].detach().clone().reshape(T, N, zd),
        "aux_mean_old": p["aux_mean"].detach().reshape(T, N, pol.aux_dim),
        "aux_log_std_old": p["aux_log_std"].detach().clone().reshape(T, N, pol.aux_dim),
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

    v2 口径注：old 侧换成 rollout 参照后断言不变——回滚同时还原参数与 Adam
    状态，每个 minibatch 都从同一起点做同一步，相对 rollout 分布的 KL 恒
    > 1e-9，故依旧全回滚（本 case 同时验证 *_old buffer 正确流入守卫）。
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
    kl_mb_all 长度 == minibatch 数。

    v2 口径注：kl_mb_all 现在读 rollout 参照（累计口径，逐 minibatch 递增
    趋势），仍全部非负且远小于巨阈值，断言不变。
    """
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
    lr 回升且不超过初始 lr（上限钉住）。

    v2 口径注：首次 update 每步回滚时 Adam 状态一并还原（v1 动量残留），
    各步等价、lr 缩小次数不变，断言不受影响。
    """
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


def _entropy_only_trainer(
    pol: AptPPOPolicy, lr: float, minibatch_size: int, thr: float, **kw
) -> PPOTrainer:
    """case7/8 共用：entropy-only 更新构造器。

    把 loss 里除 phase 熵以外的梯度全部掐掉：value_coef=0 断 critic/encoder
    路径；latent_kl_coef/latent_expl_coef=0 断先验 KL 与探索项；adv 全 0 见
    各 case 的 reward/value 构造。正态分布熵 = log_std + 常数，与均值无关，
    故唯一带梯度的参数是 phase_log_std（aux_executed=False 时 aux 头不进熵）。
    Adam 首步/同梯度步步长精确 = ±lr（m̂/√v̂ = sign(g)，熵梯度恒为
    -entropy_coef）→ 每接受一步 log_std 精确 +lr，KL 有闭式解析值。
    """
    pol.aux_executed = False  # 只留 phase 头进 log_prob/entropy/KL
    return PPOTrainer(
        pol,
        device="cpu",
        lr=lr,
        minibatch_size=minibatch_size,
        num_epochs=1,
        skip_nan=False,
        kl_guard=thr,
        entropy_coef=1.0,
        value_coef=0.0,
        latent_kl_coef=0.0,
        latent_expl_coef=0.0,
        **kw,
    )


def _zero_adv_rollout(pol: AptPPOPolicy, T: int, N: int, zd: int, seed: int) -> dict:
    """done 全真 + gamma=0 + reward=value → GAE 优势恒 0 → 归一化后仍 0 →
    ploss 梯度精确为 0（surrogate 与 adv 相乘）。"""
    r = _act_rollout(pol, T, N, zd=zd, seed=seed)
    r["done"] = torch.ones(T, N, dtype=torch.bool)
    r["reward"] = r["value"].clone()
    return r


def case7_variance_blindness() -> bool:
    """用例 7（v2 新增）：方差盲复现——仅 log_std 变化的更新必须被守卫看见。

    v1 缺陷：old 侧 log_std 取自步前 forward 的 p，而 forward 返回的是参数
    expand 视图，.detach() 不解除别名 → optimizer.step() 原地改参数后"旧
    log_std"已是新值 → 只改方差时 kl_mb 恒报 0（服务器实测真 KL 0.685 报 0）。
    v2 修复：old 侧用 rollout 采样时 clone 的 log_std。本 case 必须旧代码
    FAIL（报 0）/新代码 PASS。

    构造见 _entropy_only_trainer：单 minibatch，lr=0.35 → 一步后 ls 精确
    +0.35，解析 KL = zd·(e^{2lr}/2 − lr − 0.5) ≈ 0.628。断言 kl_mb_all[0]
    > 0、与解析值偏差 <5%、thr=0.5 小于它时回滚触发、log_std 被逐位还原。
    """
    T, N, zd, lr, thr = 1, 64, 4, 0.35, 0.5
    pol = _make_policy()
    trainer = _entropy_only_trainer(
        pol, lr, T * N, thr, kl_guard_max_rolls=99
    )
    ls0 = pol.phase_log_std.detach().clone()
    stats = trainer.update(_zero_adv_rollout(pol, T, N, zd=zd, seed=71))
    kl_expected = zd * (math.exp(2.0 * lr) / 2.0 - lr - 0.5)
    kl_mb0 = stats["kl_mb_all"][0]
    ok = kl_mb0 > 0.0  # 旧代码在此恒报 0（方差盲）→ FAIL
    ok &= abs(kl_mb0 - kl_expected) < 0.05 * kl_expected  # 与解析值同量级
    ok &= stats["kl_rolls"] == 1  # thr < 真 KL → 回滚触发
    ok &= torch.allclose(pol.phase_log_std.detach(), ls0, atol=1e-6)  # 还原
    return _report(
        "case7 variance-only update visible to guard (alias fix)",
        ok,
        f"kl_mb={kl_mb0:.4f} expected={kl_expected:.4f} rolls={stats['kl_rolls']}",
    )


def case8_cumulative_drift() -> bool:
    """用例 8（v2 新增）：累计超阈——每步单独位移小、相对 rollout 参照累计超阈。

    v1 步内参照的缺陷：每个小步各自动对比"步前"，逐步都过阈、累计却可走远。
    v2 rollout 参照天然含累计效应。构造同 case7 的 entropy-only 单调推进，
    6 个 minibatch、lr=0.1：接受 k 步后 KL = zd·f(k·lr)（f 严格单调增），
    KL1≈0.043、KL2≈0.184，thr=0.1 取两者之间 → 第 1 步通过、后续 5 步全拒；
    shrink/grow 置 1 保证每步位移一致，kl_mb 序列相对 rollout 参照单调不减。
    旧代码（步内参照）每步 KL≈0.043 全部过阈 → 本 case 旧代码必 FAIL。
    """
    T, N, zd, lr, thr = 3, 32, 4, 0.1, 0.1  # T*N=96, minibatch 16 → 6 步
    pol = _make_policy()
    trainer = _entropy_only_trainer(
        pol, lr, 16, thr, kl_guard_max_rolls=99, kl_guard_shrink=1.0,
        kl_guard_grow=1.0,
    )
    ls0 = pol.phase_log_std.detach().clone()
    stats = trainer.update(_zero_adv_rollout(pol, T, N, zd=zd, seed=81))
    kl = stats["kl_mb_all"]
    ok = len(kl) == 6
    ok &= kl[0] < thr  # 第 1 步通过
    ok &= all(v > thr for v in kl[1:])  # 后续步全被拒（累计超阈）
    ok &= stats["kl_rolls"] == 5
    # rollout 参照下单调不减（v1 步内参照无此性质可言）
    ok &= all(b >= a - 1e-6 for a, b in zip(kl, kl[1:]))
    # 只有第 1 步被接受 → log_std 恰好 +lr
    ok &= torch.allclose(pol.phase_log_std.detach(), ls0 + lr, atol=1e-5)
    return _report(
        "case8 cumulative drift caught vs rollout reference",
        ok,
        f"kl={[f'{v:.3f}' for v in kl]} rolls={stats['kl_rolls']}",
    )


def case9_rejected_step_full_restore() -> bool:
    """用例 9（v2 新增）：拒绝步全恢复——参数与 Adam 状态逐张量还原。

    v1 只回滚 policy.state_dict，被拒步留下的 exp_avg/exp_avg_sq/step 会
    污染后续接受步；v2 回滚范围扩到 optimizer 状态。构造：先巨阈值 update
    （步全被接受，Adam state 非空），快照全部参数与 optimizer.state；再切
    极小阈值 update（每步都回滚）→ 断言两者与快照逐张量 torch.equal。
    """
    torch.manual_seed(9)
    T, N = 2, 32  # minibatch 16 → 4 minibatch
    pol = _make_policy()
    trainer = PPOTrainer(
        pol,
        device="cpu",
        lr=1e-3,
        minibatch_size=16,
        num_epochs=1,
        skip_nan=False,
        kl_guard=1e9,  # 巨阈值：步全部接受 → Adam state 被填充
        kl_guard_grow=1.0,
    )
    trainer.update(_act_rollout(pol, T, N, zd=4, seed=91))
    snap_p = {k: v.clone() for k, v in pol.state_dict().items()}
    snap_o = {
        pid: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
        for pid, s in trainer.optimizer.state.items()
    }
    ok = len(snap_o) > 0  # 前提：state 已非空
    ok &= all("exp_avg" in s and "exp_avg_sq" in s for s in snap_o.values())
    trainer.kl_guard = 1e-9
    trainer.kl_guard_max_rolls = 99
    stats = trainer.update(_act_rollout(pol, T, N, zd=4, seed=92))
    ok &= stats["kl_rolls"] >= 1  # 确有回滚发生
    after_p = pol.state_dict()
    ok &= all(torch.equal(snap_p[k], after_p[k]) for k in snap_p)  # 参数逐位
    after_o = trainer.optimizer.state
    ok &= set(after_o.keys()) == set(snap_o.keys())
    for pid, s_snap in snap_o.items():
        s_now = after_o.get(pid, {})
        ok &= set(s_now.keys()) == set(s_snap.keys())
        for k, v in s_snap.items():
            # exp_avg/exp_avg_sq/step 逐张量逐位还原
            ok &= torch.equal(v, s_now[k]) if torch.is_tensor(v) else v == s_now.get(k)
    return _report(
        "case9 rejected step restores params + Adam state exactly",
        ok,
        f"rolls={stats['kl_rolls']} state_tensors={sum(len(s) for s in snap_o.values())}",
    )


def main() -> int:
    print("=== E49-C KL trust-region guard tests (pure torch, CPU) ===", flush=True)
    results = [
        case1_kl_formula_reference(),
        case2_default_off_regression(),
        case3_tiny_threshold_full_rollback(),
        case4_huge_threshold_probe(),
        case5_grow_recovery(),
        case6_expl_var(),
        case7_variance_blindness(),
        case8_cumulative_drift(),
        case9_rejected_step_full_restore(),
    ]
    print(f"=== {sum(results)}/{len(results)} cases PASS ===", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
