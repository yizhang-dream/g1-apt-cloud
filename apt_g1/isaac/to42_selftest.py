"""TO42 G0 selftest——纯 torch、无 isaaclab、negative cases 先行。

角色（SCRIPT_MAP 登记）：**DEV**（G0 wiring 门的本机/云上可跑部分；Isaac 级
wiring 冒烟另在 to42_eval --smoke + 云 wave 驱动里做）。协议依据 = TO42_PLAN
§5 G0：fbkt 臂逐位复现冻结 bucketize 行为；lsel 臂只在决策边界切换、锁存期
内 decode 输入恒定。本文件同时是 G0 判据的机械化定义——checker
（to42_checker.py）对正式 receipt 的同构断言以这里的负例为防伪锚。

用例（负例在前；任何一例失败退出码 1）：
  N1 (neg) 边界值错位被察觉：扰动 vx_max 的 bucketize 与冻结公式可区分
  N2 (neg) 非法 mode / 非法 hold_steps 被拒绝
  N3 (neg) fbkt 出现 gate=True（或 state 偏离自然 bin）即失败
  N4 (neg) lsel 在非边界步切换的实现会被"切换 ⊆ 边界"断言抓到
  T5 冻结公式核对：natural_vb 对照 torch.bucketize 原语 + 网格七点语义
  T6 fbkt 流：随机 cmd 流逐位 == clamp(natural)；gate 恒 False
  T7 lsel 边界/锁存/布尔语义：边界外提案忽略；边界处采纳+gate 单步脉冲；
     锁存期内状态不变；reset 后自然 bin 起步、首边界在第 hold_steps 步
  T8 策略侧：gate_k=1 头 shapes / Bernoulli≡Categorical(2) / deterministic
     =argmax / PPO update（含 gate 分支）有限且 gate 头有梯度

用法：python -m apt_g1.isaac.to42_selftest   （或直接 python 本文件）
"""
from __future__ import annotations

import sys

import torch

from apt_g1.isaac.to42_gate import N_SEL, To42Gate, natural_vb, vae_speed_edges


def _check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not cond:
        raise SystemExit(f"G0-SELFTEST FAIL: {name} {detail}")


def n1_edge_perturbation_detected() -> None:
    v = torch.tensor([0.25, 0.30])
    good = natural_vb(v, vx_max=0.8)
    bad = natural_vb(v, vx_max=0.7)  # 边界变 [0.2333, 0.4667] → 0.25 落 bin1
    _check("N1 edge perturbation detected", not torch.equal(good, bad),
           f"good={good.tolist()} bad={bad.tolist()}")


def n2_bad_args_rejected() -> None:
    for kwargs in (dict(mode="bogus"), dict(mode="lsel", hold_steps=0)):
        try:
            To42Gate(4, "cpu", **kwargs)
        except (ValueError, AssertionError):
            pass
        else:
            raise SystemExit(f"G0-SELFTEST FAIL: N2 bad args accepted: {kwargs}")
    _check("N2 bad args rejected", True)


def n3_fbkt_gate_must_stay_false() -> None:
    torch.manual_seed(0)
    g = To42Gate(4, "cpu", hold_steps=25, mode="fbkt")
    for _ in range(120):
        cmd = torch.rand(4) * 0.8
        state, gate = g.step(cmd, sel_bit=torch.randint(0, 2, (4,)))
        nat = natural_vb(cmd).clamp(0, N_SEL - 1)
        if not torch.equal(state, nat) or gate.any():
            raise SystemExit("G0-SELFTEST FAIL: N3 fbkt deviated from frozen bucketize")
    _check("N3 fbkt deviation would fail (120-step stream stayed exact)", True)


def n4_offboundary_switch_caught() -> None:
    """派生一个"每步跟随提案（无边界/锁存）"的坏实现，确认 checker 同构断言
    （切换步 ⊆ 决策边界）能抓住它——G0 判据对实现错误的敏感性自证。"""

    class BadEveryStepGate(To42Gate):
        def step(self, cmd_v, sel_bit=None):
            self.count += 1
            proposed = sel_bit.clamp(0, self.n_sel - 1)
            changed = proposed != self.state
            self.state = torch.where(changed, proposed, self.state)
            self.gate = changed
            return self.state, self.gate

    torch.manual_seed(1)
    g = BadEveryStepGate(1, "cpu", hold_steps=25, mode="lsel")
    g.reset(torch.tensor([0]), torch.full((1,), 0.200))
    switches = []
    for t in range(1, 76):
        bit = torch.tensor([1 if (t % 10) < 5 else 0])
        _, gate = g.step(torch.full((1,), 0.200), sel_bit=bit)
        if bool(gate[0]):
            switches.append(t)
    violations = [t for t in switches if t % 25 != 0]
    _check("N4 off-boundary switch would be caught", len(violations) > 0,
           f"bad-machine switches={switches[:6]} violations={violations[:6]}")


def t5_frozen_formula() -> None:
    edges = vae_speed_edges(0.8, 3)
    ref = torch.tensor([0.2666667, 0.5333333])
    _check("T5 edges == [0.2667, 0.5333]",
           torch.allclose(edges, ref, atol=1e-6), f"{edges.tolist()}")
    grid = torch.tensor([0.200, 0.225, 0.250, 0.275, 0.277, 0.300, 0.325])
    expect = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    _check("T5 7-point grid natural bins == {0,0,0,1,1,1,1}",
           torch.equal(natural_vb(grid), expect),
           f"{natural_vb(grid).tolist()}")
    # 对照 torch.bucketize 原语（默认 right=False：x < edge 归左 bin）
    rnd = torch.rand(4096) * 0.8
    manual = torch.bucketize(rnd, edges).clamp(0, 2)
    _check("T5 random stream == torch.bucketize reference",
           torch.equal(natural_vb(rnd), manual))


def t6_fbkt_stream() -> None:
    torch.manual_seed(2)
    g = To42Gate(8, "cpu", hold_steps=25, mode="fbkt")
    for _ in range(200):
        cmd = torch.rand(8) * 0.8
        state, gate = g.step(cmd, sel_bit=torch.randint(0, 2, (8,)))
        if not torch.equal(state, natural_vb(cmd).clamp(0, N_SEL - 1)) or gate.any():
            raise SystemExit("G0-SELFTEST FAIL: T6 fbkt stream mismatch")
    g.reset(torch.arange(8), torch.full((8,), 0.277))
    state, gate = g.step(torch.full((8,), 0.277), sel_bit=torch.zeros(8, dtype=torch.long))
    _check("T6 fbkt 200-step exact + v0.277→bin1 constant",
           int(state[0]) == 1 and not gate.any() and int(g.state[0]) == 1)


def t7_lsel_semantics() -> None:
    torch.manual_seed(3)
    g = To42Gate(1, "cpu", hold_steps=25, mode="lsel")
    g.reset(torch.tensor([0]), torch.full((1,), 0.200))  # natural bin0 起步
    _check("T7 reset → natural bin + count0", int(g.state[0]) == 0 and int(g.count[0]) == 0)

    events = []  # (step, state, gate)
    for t in range(1, 101):
        # 非边界步全给提案 1；边界步 t=25/50/75/100 给 1,1,0,0
        bit = 0 if t in (75, 100) else 1
        state, gate = g.step(torch.full((1,), 0.200), sel_bit=torch.tensor([bit]))
        events.append((t, int(state[0]), bool(gate[0])))

    switches = [t for t, s, gv in events if gv]
    # t=25 提案1(0→1 切换)、t=50 提案1(同值→重评但不切换、无脉冲)、
    # t=75 提案0(1→0 切换)、t=100 提案0(同值→无脉冲)
    _check("T7 switches exactly at {25, 75}; same-value boundary re-eval silent",
           switches == [25, 75], f"switches={switches}")
    _check("T7 state stays 0 before first boundary",
           all(s == 0 for t, s, _ in events if t < 25))
    _check("T7 state==1 on [25,75), ==0 on [75,100] (latched between boundaries)",
           all(s == 1 for t, s, _ in events if 25 <= t < 75)
           and all(s == 0 for t, s, _ in events if 75 <= t <= 100))
    _check("T7 gate pulses only on actual switch steps",
           all((not gv) or (t in switches) for t, s, gv in events))
    gate_by_t = {t: gv for t, _, gv in events}
    _check("T7 boundary steps without change do not pulse",
           not gate_by_t[50] and not gate_by_t[100])

    # 锁存 = 两次边界之间 decode 输入（state）恒定，即便提案剧烈翻转
    g2 = To42Gate(1, "cpu", hold_steps=25, mode="lsel")
    g2.reset(torch.tensor([0]), torch.full((1,), 0.277))
    states = []
    for t in range(1, 26):
        bit = torch.tensor([t % 2])  # 每 步翻转提案
        s, _ = g2.step(torch.full((1,), 0.277), sel_bit=bit)
        states.append(int(s[0]))
    _check("T7 latch: state constant across non-boundary steps",
           len(set(states[:-1])) == 1, f"{states}")


def t8_policy_gate_head() -> None:
    from torch.distributions import Categorical

    from apt_g1.isaac.ppo_core import AptPPOPolicy, PPOTrainer

    torch.manual_seed(4)
    # gate_k = 类数（{0=vb0, 1=vb1}），动作 = 类索引 1 维——与旧 gate_sel 同机制
    pol = AptPPOPolicy(obs_dim=64, aux_dim=12, gate_k=2, use_phase=False,
                       latent_dim=16)
    obs = torch.randn(6, 64)
    act, logp, ent, val, p = pol.act(obs)
    _check("T8 act shapes", act["latent"].shape == (6, 16)
           and act["gate"].shape == (6,)
           and set(act["gate"].tolist()) <= {0, 1})
    gd = Categorical(logits=p["gate_logits"])
    _check("T8 deterministic == argmax(probs)",
           torch.equal(pol.act(obs, deterministic=True)[0]["gate"],
                       gd.probs.argmax(-1)))
    bit = act["gate"]
    _check("T8 Bernoulli ≡ Categorical(2) log_prob",
           torch.allclose(gd.log_prob(bit),
                           torch.log(torch.stack([1 - gd.probs[:, 1],
                                                  gd.probs[:, 1]], -1)
                                     .gather(1, bit[:, None])).squeeze(1),
                           atol=1e-6))

    # PPO update（gate 分支激活）一步：有限 loss + gate 头收到梯度
    T, N = 4, 8
    buf = {
        "obs": torch.randn(T, N, 64),
        "phase": torch.randn(T, N, 16),
        "aux": torch.randn(T, N, 12),
        "gate": torch.randint(0, 2, (T, N)),
        "logp": torch.randn(T, N) * 0.01,
        "value": torch.randn(T, N) * 0.1,
        "reward": torch.rand(T, N),
        "done": torch.zeros(T, N, dtype=torch.bool),
        "trunc": torch.zeros(T, N, dtype=torch.bool),
    }
    buf["last_value"] = pol.get_value(buf["obs"][-1]).detach()
    tr = PPOTrainer(pol, max_iters=10, device="cpu")
    stats = tr.update(buf)
    _check("T8 ppo update finite", all(torch.isfinite(torch.tensor(v))
                                       for k, v in stats.items()
                                       if isinstance(v, float)),
           f"loss={stats['loss']:.4f}")
    gw = pol.gate_logits.weight.grad
    _check("T8 gate head receives gradient", gw is not None
           and torch.isfinite(gw).all() and gw.abs().sum() > 0)


def main() -> int:
    print("=== TO42 G0 selftest (pure torch; negative cases first) ===", flush=True)
    n1_edge_perturbation_detected()
    n2_bad_args_rejected()
    n3_fbkt_gate_must_stay_false()
    n4_offboundary_switch_caught()
    t5_frozen_formula()
    t6_fbkt_stream()
    t7_lsel_semantics()
    t8_policy_gate_head()
    print("=== G0-SELFTEST ALL PASS ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
