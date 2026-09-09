"""D048j 确定性测试：progress 项封顶上界改为当前命令（纯 torch CPU，无 isaaclab 依赖）。

被测对象 ``apt_g1.isaac.reward_terms.progress_bonus``（纯函数模块，import 不拉
isaaclab）。覆盖面：
  1. 旗标关（cap_cmd=False）= 旧内联公式 clamp(vx, 0, 1) 逐位一致（网格 + 随机
     + 单点 vx=0.6→0.6；含负 vx 裁零）
  2. 旗标开 + cmd=0.4 + vx=0.6 → 0.4（超速段被封到命令）
  3. 旗标开 + vx 低于封顶不裁（vx=0.3/0.399, cmd=0.4 → 原值）
  4. 旗标开 + cmd=0 → 0（任意 vx）；负命令 clamp(min=0) 同样封到 0
  5. cmd 张量逐元素混合 [0.2, 0.4, 0.6] 对 vx=[0.7, 0.7, 0.7] → [0.2, 0.4, 0.6]
  6. cmd 标量可广播（python float 与 0-d 张量，shape 对齐 vx）
  7. 静态最优数值钉（D048j 预注册数值，stillness_vx_scale=0.05 计入）：
     f(v) = exp(-(v-0.4)^2/0.25) + progress_bonus(v, 0.4, ·) - 0.05 v^2 在
     v∈[0,1] 网格（步长 0.001）上的 argmax——旗标开 = 0.4±0.002（最优恰为
     cmd）、旗标关 ∈ [0.50, 0.57]（旧上界 1.0 的结构性超速偏置存在）
  8. 偏置反转钉：f(0.62) - f(0.4) 旗标关 ≈ +0.033（超速反高）、旗标开 ≈
     -0.187（超速受罚），符号与量级同时钉住

用法（服务器 .venv_isaac / 本机 CPU torch 均可）：
    PYTHONPATH=. python apt_g1/isaac/test_progress_cap_reward.py
全部用例 PASS 时 exit 0。
"""

from __future__ import annotations

import math
import sys

import torch

from apt_g1.isaac.reward_terms import progress_bonus


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
        flush=True,
    )
    return ok


def _f(v: torch.Tensor, cmd: float, cap_cmd: bool) -> torch.Tensor:
    """D048j 静态一维评分：track_xy + progress - stillness(vx 项)。

    track_xy = exp(-(v-cmd)^2/0.25)（权重 1.0，vel_sigma2=0.25），
    stillness_vx_scale=0.05（vy/omega 项在一维切片下为 0）。
    """
    return (
        torch.exp(-((v - cmd) ** 2) / 0.25)
        + progress_bonus(v, cmd, cap_cmd)
        - 0.05 * v**2
    )


# ---------------------------------------------------------------------------
def case1_cap_off_bitwise_legacy() -> bool:
    # 旗标关 = 旧内联公式 torch.clamp(vx, 0.0, 1.0) 逐位一致
    grid = torch.arange(0, 1001, dtype=torch.float32) * 0.001
    torch.manual_seed(0)
    rnd = torch.randn(4096) * 0.7
    ok = True
    for vx in (grid, rnd, torch.tensor([0.6]), torch.tensor([-0.5])):
        legacy = torch.clamp(vx, 0.0, 1.0)
        got = progress_bonus(vx, 0.4, cap_cmd=False)
        ok = ok and torch.equal(got, legacy)  # 逐位（bitwise）一致
    # 单点钉：旗标关 vx=0.6 → 0.6（不被任何新逻辑触碰；torch.equal 逐位）
    ok = ok and torch.equal(
        progress_bonus(torch.tensor([0.6]), 0.4, False), torch.tensor([0.6])
    )
    return _report(
        "1 cap-off == legacy clamp(vx,0,1) bitwise (grid+rand, vx=0.6->0.6)",
        ok,
    )


def case2_cap_on_clips_overspeed() -> bool:
    got = progress_bonus(torch.tensor([0.6]), 0.4, cap_cmd=True)
    ok = torch.equal(got, torch.tensor([0.4]))
    # vx=1.0（旧上界）同样被封到 cmd
    ok = ok and torch.equal(
        progress_bonus(torch.tensor([1.0]), 0.4, True), torch.tensor([0.4])
    )
    return _report("2 cap-on cmd=0.4 vx=0.6 -> 0.4 (overspeed clipped to cmd)", ok,
                   f"got={got.item():.1f}")


def case3_cap_on_below_cap_untouched() -> bool:
    vx = torch.tensor([0.3, 0.399])
    got = progress_bonus(vx, 0.4, cap_cmd=True)
    ok = torch.equal(got, vx)  # 低于封顶不裁，原值返回
    return _report("3 cap-on vx below cap untouched (0.3/0.399 -> same)", ok)


def case4_cap_on_zero_or_negative_cmd() -> bool:
    ok = progress_bonus(torch.tensor([0.6, 1.0]), 0.0, cap_cmd=True).tolist() == [0.0, 0.0]
    # 负命令 clamp(min=0)：封顶不为负
    ok = ok and progress_bonus(torch.tensor([0.6]), -0.5, True).item() == 0.0
    # 负 vx：clamp(vx,0,1)=0，两旗标一致为 0
    ok = ok and progress_bonus(torch.tensor([-0.5]), 0.4, True).item() == 0.0
    ok = ok and progress_bonus(torch.tensor([-0.5]), 0.4, False).item() == 0.0
    return _report("4 cap-on cmd=0 (or negative) -> bonus 0; negative vx -> 0", ok)


def case5_elementwise_cmd_tensor() -> bool:
    vx = torch.tensor([0.7, 0.7, 0.7])
    cmd = torch.tensor([0.2, 0.4, 0.6])
    got = progress_bonus(vx, cmd, cap_cmd=True)
    ok = torch.equal(got, torch.tensor([0.2, 0.4, 0.6]))
    return _report("5 elementwise cmd [0.2,0.4,0.6] x vx 0.7 -> [0.2,0.4,0.6]", ok,
                   f"got={got.tolist()}")


def case6_scalar_cmd_broadcast() -> bool:
    vx = torch.tensor([[0.6], [0.3]])
    got_f = progress_bonus(vx, 0.4, cap_cmd=True)
    got_t = progress_bonus(vx, torch.tensor(0.4), cap_cmd=True)
    ref = torch.tensor([[0.4], [0.3]])
    ok = torch.equal(got_f, ref) and torch.equal(got_t, ref)
    ok = ok and got_f.shape == vx.shape and got_t.shape == vx.shape  # shape 对齐
    return _report("6 scalar cmd broadcast (python float / 0-d tensor)", ok)


def case7_static_optimum_argmax() -> bool:
    v = torch.arange(0, 1001, dtype=torch.float64) * 0.001
    v_on = _f(v, 0.4, cap_cmd=True)
    v_off = _f(v, 0.4, cap_cmd=False)
    arg_on = v[torch.argmax(v_on)].item()
    arg_off = v[torch.argmax(v_off)].item()
    ok = abs(arg_on - 0.4) <= 0.002  # 旗标开：静态最优恰为 cmd
    ok = ok and 0.50 <= arg_off <= 0.57  # 旗标关：结构性超速偏置存在
    return _report(
        "7 static optimum argmax: cap-on 0.4±0.002 / cap-off in [0.50,0.57]",
        ok,
        f"arg_on={arg_on:.3f} arg_off={arg_off:.3f}",
    )


def case8_bias_reversal_at_062() -> bool:
    v = torch.tensor([0.4, 0.62], dtype=torch.float64)
    d_off = (_f(v, 0.4, False)[1] - _f(v, 0.4, False)[0]).item()
    d_on = (_f(v, 0.4, True)[1] - _f(v, 0.4, True)[0]).item()
    ok = 0.030 <= d_off <= 0.036  # 旧奖励：vx=0.62 每步反高 ≈ +0.033
    ok = ok and -0.19 <= d_on <= -0.18  # 修正后：超速每步 ≈ -0.187
    ok = ok and math.isclose(d_off, 0.033, abs_tol=5e-3)
    ok = ok and math.isclose(d_on, -0.187, abs_tol=5e-3)
    return _report(
        "8 bias reversal: f(0.62)-f(0.4) +0.033 (off) -> -0.187 (on)",
        ok,
        f"d_off={d_off:+.4f} d_on={d_on:+.4f}",
    )


def main() -> None:
    cases = [
        case1_cap_off_bitwise_legacy,
        case2_cap_on_clips_overspeed,
        case3_cap_on_below_cap_untouched,
        case4_cap_on_zero_or_negative_cmd,
        case5_elementwise_cmd_tensor,
        case6_scalar_cmd_broadcast,
        case7_static_optimum_argmax,
        case8_bias_reversal_at_062,
    ]
    results = [c() for c in cases]
    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} cases PASS", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
