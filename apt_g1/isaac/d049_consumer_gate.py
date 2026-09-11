"""D049b-fix 消费者测试门：vb 连续动作契约（无 isaaclab 依赖，纯 torch CPU）。

owner 冻结路线（refine-logs/DS_CONTINUOUS_EXECUTION_PLAN.md §5j 判读结果段，
commit 1fe31a8）：修复 P1 动作-概率契约缺陷（旧实现 categorical 采样索引进
log_prob 而 env 执行 softmax(logits) 确定性 w，回报与采样索引独立 ⇒
E[A∇log p(k)]=0，vb 头无合法策略梯度；ksg joint 不含 kl_vb）后、训练发射前
的三个消费者门。判据全部预注册在本文件常量里，数值随 JSON 落盘入档。

子门（--gate 1|2|3|all）：
  G1 契约一致性：act() 采样 → buffer → 送 env → update() 重算的同一张量贯穿。
     ① 真采样（两种子两次 act 的 vb_action 不同）
     ② mock train 循环拼接后 action[45:48]==vb_action（residual 48 维布局，
        _vb_action_slice 同步切片）；plain 布局 [z,aux,vb]=31 切片 28:31 同验
     ③ act() 的 log_prob 分解 = aux+z+vb 高斯密度之和（手算对拍 1e-5）；
        update() 的资格检查（ksg qualification）重算输入=buffer vb_action：
        同值 → ratio 偏差 ≤1e-5；故意喂 forward 的 det 值（=vb_logits）→
        偏差显著（>0.5）= 重算路径确以 buffer 值为准（门自身有效性对照）
     ④ det 模式 vb_action==vb_logits（softmax 与旧确定性 w 逐位一致，
        评测口径不变）
  G2 小环境信用分配：toy bandit（无需 Isaac），reward = 1−‖softmax(vb_action)
     ·v_targets − cmd‖²，v_targets=[0.3,0.5,1.2]、cmd=0.9。初始 softmax 近
     均匀 → 加权速度 2/3≈0.667；最优软权重 w*≈(0,0.43,0.57) 精确命中 0.9。
     主臂复用 ppo_core 真实 update() 跑 400 update（lr=3e-3 + vb_init_std=
     -1.0：首版 lr=1e-2/σ=e^-4 实测过冲——best 1.115 后震荡回落 final 0.639
     带外，2026-09-12 返工修正，超参扫描定档），判据 = 末 20 轮平均加权速度
     落 0.9±0.15=[0.75,1.05]（下界 0.75=0.667+半程收敛余量，上界防跑飞；
     owner 冻结值不改）。
     对照臂 = 缺陷版复现（categorical 簿记+确定性执行，reward 与采样索引独
     立 ⇒ 期望梯度恒 0），3 种子同跑 400 update，≥2 种子末 20 轮均值落在带
     外 = 缺陷可复现（不收敛），证明本门有能力区分缺陷版与修复版。
  G3 单头超限回滚：ksg 流程中人为只对 vb 头注入大梯度（其余参数 grad=None
     → Adam 跳过，z/res 头无位移源），lr=100 使任何尺度的全步长都超阈：
     ① kl_vb 超 0.05 且 6 次回退后整步拒绝；② joint==kl_vb（z=res=0 仍相等
     = kl_vb 确在联合目标里，若 vb 不进 joint 则 joint=0≠vb 必败）；③ 回滚
     后 vb 头参数与其余参数逐位还原、Adam 状态（exp_avg/exp_avg_sq/step）
     逐位还原；④ 对照：正常 lr(3e-4) 只动 vb 头的小步被接受（lr_scale=1、
     kl_vb≤0.05）= 守卫不过度触发。

用法（服务器 .venv_isaac 或本机 torch-cpu 均可实跑；无 torch 环境仅 py_compile）：
    PYTHONPATH=. python apt_g1/isaac/d049_consumer_gate.py --gate all \
        --out apt_g1/outputs/d049_consumer_gate_report.json
所求子门全 PASS 时 exit 0，任一 FAIL exit 1。判据数值写入 --out JSON。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.distributions import Categorical, Normal

from apt_g1.isaac.ppo_core import AptPPOPolicy, PPOTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]

# G2 预注册判据（§5j 工单口径；改这里 = 改判据，需 owner 重冻）
G2_V_TARGETS = (0.3, 0.5, 1.2)
G2_CMD = 0.9
G2_BAND = (0.75, 1.05)  # 0.9±0.15
G2_ITERS = 400
G2_TAIL = 20  # 末 20 轮平均为判据（平滑随机尾动）
G2_TREATED_N = 128
# 首版 lr=1e-2/vb_init_std=-4 实测过冲（best 1.115 → 震荡回落 final 0.639
# 带外）；2026-09-12 返工扫描定档 lr=3e-3 + σ=e^-1：σ 加大让 score-function
# 信号可见（log_prob 曲率 1/σ² 降 → ratio 不易撞 clip 墙），final20=0.9075
# 正中 0.9、尾段 std 全场最小
G2_TREATED_LR = 3e-3
G2_TREATED_VB_INIT_STD = -1.0
G2_CONTROL_N = 512
G2_CONTROL_SEEDS = (11, 12, 13)
G2_CONTROL_MIN_OUT = 2  # ≥2/3 种子带外 = 缺陷复现成立
G2_INIT_TOL = 0.05  # 初始加权速度 vs 2/3 的容差（sanity）

# G3 预注册判据
G3_KL_TARGET = 0.05
G3_BACKTRACKS = 6

# env._vb_action_slice 的两模式切片（apt_flat_env.py 单一事实源；此处硬编码
# 为消费方自检值，若 env 侧布局变更本门应失败提示同步）
SLICE_RESIDUAL = (45, 48)  # [z(16), res(29), vb(3)]
SLICE_PLAIN = (28, 31)  # [z(16), aux(12), vb(3)]


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
        flush=True,
    )
    return ok


# ---------------------------------------------------------------------------
# G1 契约一致性
# ---------------------------------------------------------------------------
def gate1() -> tuple[bool, dict]:
    detail: dict = {}

    # 布局 a：D048i residual 配方 [z(16), res(29), vb(3)] = 48
    torch.manual_seed(101)
    policy = AptPPOPolicy(
        obs_dim=111, aux_dim=29, gate_k=0, vb_head=True, hidden_dim=64,
        use_phase=False, latent_dim=16,
    )
    obs = torch.randn(8, 111)

    # ① 真采样：同 obs 两种子两次 act，vb_action 必不同
    torch.manual_seed(7)
    a1, lp1, _ent1, _v1, fwd1 = policy.act(obs)
    torch.manual_seed(8)
    a2, lp2, _ent2, _v2, fwd2 = policy.act(obs)
    diff = float((a1["vb_action"] - a2["vb_action"]).abs().max())
    detail["g1_resample_maxdiff"] = diff
    ok1 = not torch.equal(a1["vb_action"], a2["vb_action"])
    _report("G1.1 stochastic act(): two seeds -> different vb_action", ok1,
            f"maxdiff={diff:.4g}")
    ok = ok1

    # ② mock train 循环：buffer 槽位与送 env 的动作段同值（同一张量贯穿）。
    #    residual 布局 48 维切片 45:48；plain 布局 31 维切片 28:31
    buf_vb = a1["vb_action"].detach()
    action48 = torch.cat([a1["phase"], a1["aux"], buf_vb], dim=1)
    lo, hi = SLICE_RESIDUAL
    seg48 = action48[:, lo:hi]
    ok2a = action48.shape[1] == 48 and torch.equal(seg48, buf_vb)
    torch.manual_seed(102)
    policy_p = AptPPOPolicy(
        obs_dim=64, aux_dim=12, gate_k=0, vb_head=True, hidden_dim=32,
        use_phase=False, latent_dim=16,
    )
    obs_p = torch.randn(8, 64)
    ap1, _, _, _, _ = policy_p.act(obs_p)
    buf_vb_p = ap1["vb_action"].detach()
    action31 = torch.cat([ap1["phase"], ap1["aux"], buf_vb_p], dim=1)
    lo_p, hi_p = SLICE_PLAIN
    ok2b = action31.shape[1] == 31 and torch.equal(action31[:, lo_p:hi_p], buf_vb_p)
    ok2 = ok2a and ok2b
    detail["g1_slice48_bitwise"] = bool(ok2a)
    detail["g1_slice31_bitwise"] = bool(ok2b)
    _report("G1.2 buffer slot == env action segment (48d 45:48 / 31d 28:31)",
            ok2)
    ok = ok and ok2

    # ③ log_prob 数值 = aux+z+vb 高斯密度之和（手算对拍，容差 1e-5）
    lp_vb = Normal(fwd1["vb_logits"], fwd1["vb_log_std"].exp()).log_prob(
        buf_vb
    ).sum(-1)
    lp_aux = Normal(fwd1["aux_mean"], fwd1["aux_log_std"].exp()).log_prob(
        a1["aux"]
    ).sum(-1)
    lp_z = Normal(fwd1["phase_mean"], fwd1["phase_log_std"].exp()).log_prob(
        a1["phase"]
    ).sum(-1)
    dev = float((lp1 - (lp_aux + lp_z + lp_vb)).detach().abs().max())
    detail["g1_logp_hand_maxdev"] = dev
    ok3 = dev <= 1e-5
    _report("G1.3a act logp == manual aux+z+vb normal densities (1e-5)", ok3,
            f"maxdev={dev:.3e}")
    ok = ok and ok3

    # ③' update() 重算路径以 buffer vb_action 为准：真实 rollout + ksg 资格
    #    检查（同值 → ratio 偏差≈0）；再故意把 vb 槽位换成 det 值
    #    （=vb_logits）→ 偏差显著，证明重算确在消费 buffer 值
    torch.manual_seed(9)
    T, N = 2, 8
    obs_r = torch.randn(T, N, 111)
    rl = {
        "obs": obs_r,
        "aux": torch.zeros(T, N, 29),
        "phase": torch.zeros(T, N, 16),
        "vb_action": torch.zeros(T, N, 3),
        "logp": torch.zeros(T, N),
        "value": torch.zeros(T, N),
        "reward": torch.randn(T, N),
        "done": torch.zeros(T, N, dtype=torch.bool),
        "trunc": torch.zeros(T, N, dtype=torch.bool),
        "last_value": torch.zeros(N),
    }
    for t in range(T):
        at, lpt, _et, vt, _ft = policy.act(obs_r[t])
        rl["aux"][t] = at["aux"].detach()
        rl["phase"][t] = at["phase"].detach()
        rl["vb_action"][t] = at["vb_action"].detach()
        rl["logp"][t] = lpt.detach()
        rl["value"][t] = vt.detach()
    tr = PPOTrainer(
        policy, lr=3e-4, device="cpu", kl_step_guard=True, minibatch_size=64
    )
    obs_f = rl["obs"].reshape(T * N, -1)
    # 两组资格检查都在 update() 改动策略之前做（否则偏差会混入策略移动）：
    # 同值检查：buffer vb_action 重算 vs 存储 logp → ratio 偏差≈0
    q_ok = tr.ksg_qualification(
        obs_f, rl["phase"].reshape(T * N, -1), rl["aux"].reshape(T * N, -1),
        None, rl["logp"].reshape(-1), aux_scored=True,
        vb=rl["vb_action"].reshape(T * N, 3),
    )
    detail["g1_qual_ratio_maxdev"] = q_ok["qual_ratio_maxdev"]
    ok3b0 = q_ok["qual_ratio_maxdev"] <= 1e-5
    _report("G1.3b' ksg qualification on buffer vb_action: ratio dev ~0",
            ok3b0, f"qual_ratio_maxdev={q_ok['qual_ratio_maxdev']:.3e}")
    ok = ok and ok3b0

    # 篡改探针：重算输入换成 forward 的 det 值（≠采样 vb_action）→ 必被检出
    # （若 vb 不进重算或重算读了别的量，此项偏差≈0 = 门无区分力）
    p0 = policy.forward_actor(obs_f)
    q_bad = tr.ksg_qualification(
        obs_f, rl["phase"].reshape(T * N, -1), rl["aux"].reshape(T * N, -1),
        None, rl["logp"].reshape(-1), aux_scored=True,
        vb=p0["vb_logits"].detach(),
    )
    detail["g1_tamper_ratio_maxdev"] = q_bad["qual_ratio_maxdev"]
    ok3c = q_bad["qual_ratio_maxdev"] > 0.5
    _report("G1.3c tamper probe: det value in vb slot is detected", ok3c,
            f"tamper_ratio_maxdev={q_bad['qual_ratio_maxdev']:.4f}")
    ok = ok and ok3c

    # 端到端：真实 update() 内部资格检查同口径复验（此时才允许改策略）
    stats = tr.update(rl)
    q_dev = stats["qual_ratio_maxdev"]
    detail["g1_update_qual_ratio_maxdev"] = q_dev
    ok3b = q_dev <= 1e-5
    _report("G1.3b update() qualification: buffer vb_action recomputes ratio~1",
            ok3b, f"qual_ratio_maxdev={q_dev:.3e}")
    ok = ok and ok3b

    # ④ det 模式：vb_action == vb_logits（softmax 与旧确定性 w 逐位一致）
    ad, _, _, _, fd = policy.act(obs_r[0], deterministic=True)
    ok4 = torch.equal(ad["vb_action"], fd["vb_logits"])
    detail["g1_det_bitwise"] = bool(ok4)
    _report("G1.4 det mode: vb_action == vb_logits (eval semantics kept)", ok4)
    ok = ok and ok4

    return ok, detail


# ---------------------------------------------------------------------------
# G2 小环境信用分配（toy bandit，复用真实 ppo_core.update()）
# ---------------------------------------------------------------------------
def _t_targets() -> torch.Tensor:
    return torch.tensor(G2_V_TARGETS, dtype=torch.float32)


def gate2() -> tuple[bool, dict]:
    detail: dict = {}
    tgt = _t_targets()

    # ---- 主臂：修复版（连续 vb_action 同值贯穿），真实 update() ----
    torch.manual_seed(2001)
    policy = AptPPOPolicy(
        obs_dim=4, aux_dim=12, gate_k=0, vb_head=True, hidden_dim=32,
        use_phase=False, latent_dim=0,
        vb_init_std=G2_TREATED_VB_INIT_STD,
    )
    # 只有 vb 头可学出非平凡行为：aux 头不进 log_prob（与 plain latent 同
    # 约定）；obs 固定，reward 只依赖 softmax(vb_action)
    policy.aux_executed = False
    # 刻意不开 kl_step_guard/kl_guard：本门测的是「契约→信用分配」通路本身，
    # 不是看门（G3 专测看门）；看门的回退重试会让单步更新非平稳、拖慢收敛
    # （reviewer N2 实测：首版超参下 ksg on 200 it 停在 0.6957；2026-09-12
    # 复测：新超参下 ksg on 400 it 亦可入带 0.9208，门内仍关=归因干净）
    tr = PPOTrainer(
        policy, lr=G2_TREATED_LR, device="cpu", minibatch_size=1024
    )
    torch.manual_seed(2002)
    obs = torch.randn(G2_TREATED_N, 4)
    hist: list = []
    for _ in range(G2_ITERS):
        act, logp, _ent, val, _f = policy.act(obs)
        vb = act["vb_action"].detach()
        w = torch.softmax(vb, dim=-1)
        rew = 1.0 - (w @ tgt - G2_CMD) ** 2
        rl = {
            "obs": obs.unsqueeze(0),
            "aux": torch.zeros(1, G2_TREATED_N, 12),
            "phase": None,
            "vb_action": vb.reshape(1, G2_TREATED_N, 3),
            "logp": logp.detach().reshape(1, G2_TREATED_N),
            "value": val.detach().reshape(1, G2_TREATED_N),
            "reward": rew.reshape(1, G2_TREATED_N),
            "done": torch.zeros(1, G2_TREATED_N, dtype=torch.bool),
            "trunc": torch.zeros(1, G2_TREATED_N, dtype=torch.bool),
            "last_value": torch.zeros(G2_TREATED_N),
        }
        tr.update(rl)
        with torch.no_grad():
            p = policy.forward_actor(obs)
            w_det = torch.softmax(p["vb_logits"], dim=-1)
            hist.append(float((w_det @ tgt).mean()))
    init_speed = hist[0]
    final_speed = sum(hist[-G2_TAIL:]) / G2_TAIL
    band_lo, band_hi = G2_BAND
    init_ok = abs(init_speed - 2.0 / 3.0) <= G2_INIT_TOL
    treated_ok = band_lo <= final_speed <= band_hi
    detail["g2_treated_init_speed"] = init_speed
    detail["g2_treated_final_speed_mean20"] = final_speed
    detail["g2_treated_hist_tail5"] = [round(v, 4) for v in hist[-5:]]
    _report(
        f"G2 treated: weighted speed 0.667 -> 0.9+-0.15 in {G2_ITERS} updates",
        init_ok and treated_ok,
        f"init={init_speed:.4f} final20={final_speed:.4f} "
        f"band=[{band_lo},{band_hi}]",
    )
    ok = init_ok and treated_ok

    # ---- 对照臂：G1 缺陷版复现（categorical 簿记 + 确定性执行）----
    # 缺陷力学：log_prob 消费采样索引 k，执行/回报只看 softmax(logits)（与
    # k 独立）⇒ E[A∇log p(k)]=0，logits 无定向漂移 → 加权速度停在 ~0.667
    # 附近。≥2/3 种子末 20 轮均值在带外 = 缺陷可复现（若本对照意外收敛，说
    # 明本门判据无区分力，G2 判 FAIL）
    outs = 0
    finals = []
    for seed in G2_CONTROL_SEEDS:
        hist_c = _defect_arm(seed, tgt)
        fin_c = sum(hist_c[-G2_TAIL:]) / G2_TAIL
        finals.append(fin_c)
        if not (band_lo <= fin_c <= band_hi):
            outs += 1
    control_ok = outs >= G2_CONTROL_MIN_OUT
    detail["g2_control_final_means"] = [round(v, 4) for v in finals]
    detail["g2_control_seeds_outside_band"] = outs
    _report(
        "G2 control (defect repro): categorical bookkeeping does NOT converge",
        control_ok,
        f"final20 per seed={detail['g2_control_final_means']} "
        f"outside_band={outs}/{len(G2_CONTROL_SEEDS)}",
    )
    ok = ok and control_ok
    return ok, detail


def _defect_arm(seed: int, tgt: torch.Tensor) -> list:
    """G1 缺陷版的最小复现臂：categorical 簿记 + 确定性执行。

    与真实缺陷逐点同构：log_prob 消费采样索引 k，回报只依赖 softmax(logits)
    （同 D049b 旧实现的 env 路径），k 与回报独立 ⇒ 期望策略梯度为零。
    独立小 policy（Linear 4→3 零初始化 = 均匀 w 起点）+ Adam，与主臂同 lr
    同迭代数。返回逐步加权速度（det 口径，同主臂指标）。
    """
    torch.manual_seed(seed)
    obs = torch.randn(G2_CONTROL_N, 4)
    head = torch.nn.Linear(4, 3)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    opt = torch.optim.Adam(head.parameters(), lr=G2_TREATED_LR)
    hist: list = []
    for _ in range(G2_ITERS):
        logits = head(obs)
        cat = Categorical(logits=logits)
        k = cat.sample()  # 缺陷①：采样索引进 log_prob
        lpk = cat.log_prob(k)
        w_det = torch.softmax(logits, dim=-1)  # 缺陷②③：执行与 k 无关
        rew = 1.0 - (w_det @ tgt - G2_CMD) ** 2
        adv = (rew - rew.mean()) / (rew.std() + 1e-8)
        loss = -(adv.detach() * lpk).mean() - 1e-3 * cat.entropy().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            hist.append(
                float((torch.softmax(head(obs), dim=-1) @ tgt).mean())
            )
    return hist


# ---------------------------------------------------------------------------
# G3 单头超限回滚（ksg 看门，只对 vb 头施更新）
# ---------------------------------------------------------------------------
def _state_equal(sa: dict, sb: dict) -> bool:
    """Adam optimizer.state_dict 的 state 段逐位相等（exp_avg/exp_avg_sq/step）。"""
    if set(sa["state"].keys()) != set(sb["state"].keys()):
        return False
    for k in sa["state"]:
        da, db = sa["state"][k], sb["state"][k]
        if set(da.keys()) != set(db.keys()):
            return False
        for kk in da:
            va, vb = da[kk], db[kk]
            if isinstance(va, torch.Tensor):
                if not torch.equal(va, vb):
                    return False
            elif va != vb:
                return False
    return True


def gate3() -> tuple[bool, dict]:
    detail: dict = {}
    torch.manual_seed(301)
    policy = AptPPOPolicy(
        obs_dim=8, aux_dim=3, gate_k=0, vb_head=True, hidden_dim=16,
        use_phase=True,
    )
    tr = PPOTrainer(
        policy, lr=100.0, device="cpu", kl_step_guard=True,
        kl_step_target=G3_KL_TARGET, kl_backtracks=G3_BACKTRACKS,
        minibatch_size=64,
    )
    obs = torch.randn(16, 8)
    mb = torch.arange(16)
    # 预热：一次真实小 step，让 Adam 状态非平凡（exp_avg/exp_avg_sq/step
    # 非零非空）——回滚还原检查才有区分力。Adam 首步位移≈lr（梯度幅度被
    # 归一化），故临时压低 lr 做 1e-3 级小步，step 后复位，参数只轻微移动
    torch.manual_seed(302)
    warm = sum(p.pow(2).sum() for p in policy.parameters())
    tr.optimizer.zero_grad()
    warm.backward()
    tr.optimizer.param_groups[0]["lr"] = 1e-3
    tr.optimizer.step()
    tr.optimizer.param_groups[0]["lr"] = 100.0

    old_buf = {k: v.detach().clone() for k, v in policy.forward_actor(obs).items()}
    pre_named = {n: p.detach().clone() for n, p in policy.named_parameters()}
    pre_state = copy.deepcopy(tr.optimizer.state_dict())
    lr_round = [float(g["lr"]) for g in tr.optimizer.param_groups]

    # 场景 A：只对 vb 头注入大梯度（其余参数 grad=None → Adam 跳过，z/res
    # 头本步无位移源）；lr=100 + Adam 首步位移≈lr ⇒ 任何回退尺度都超 0.05
    for p in policy.parameters():
        p.grad = None
    policy.vb_logits.weight.grad = torch.full_like(
        policy.vb_logits.weight, 10.0
    )
    policy.vb_logits.bias.grad = torch.full_like(policy.vb_logits.bias, 10.0)
    wA = tr.ksg_watch_step(obs, mb, old_buf, lr_round)
    detail["g3_A"] = {
        k: wA[k] for k in
        ("accepted", "rejected", "lr_scale", "backtracks",
         "joint", "z", "res", "vb")
    }
    okA_rej = bool(wA["rejected"]) and wA["vb"] > G3_KL_TARGET
    _report("G3.1 vb-only huge update: kl_vb > 0.05 -> step rejected", okA_rej,
            f"rejected={wA['rejected']} kl_vb={wA['vb']:.4g} "
            f"lr_scale={wA['lr_scale']:.4g}")
    okA_joint = abs(wA["joint"] - wA["vb"]) <= 1e-9 * max(1.0, abs(wA["vb"]))
    _report("G3.2 joint == kl_vb (z=res=0): kl_vb is inside the joint target",
            okA_joint,
            f"joint={wA['joint']:.4g} z={wA['z']:.4g} res={wA['res']:.4g}")
    okA_param = all(
        torch.equal(p.detach(), pre_named[n])
        for n, p in policy.named_parameters()
    )
    okA_state = _state_equal(pre_state, tr.optimizer.state_dict())
    detail["g3_A_params_bitwise_restored"] = bool(okA_param)
    detail["g3_A_adam_state_bitwise_restored"] = bool(okA_state)
    okA = okA_rej and okA_joint and okA_param and okA_state
    _report("G3.3 rollback: vb-head params + Adam state bitwise restored",
            okA_param and okA_state)
    ok = okA_rej and okA_joint and okA_param and okA_state

    # 场景 B（对照）：正常 lr 只动 vb 头的小步应被接受（守卫不过度触发）；
    # 接受后再验 ksg_eval_kl 的 joint = z+res+vb 恒等式
    tr2 = PPOTrainer(
        policy, lr=3e-4, device="cpu", kl_step_guard=True,
        kl_step_target=G3_KL_TARGET, kl_backtracks=G3_BACKTRACKS,
        minibatch_size=64,
    )
    old_buf2 = {
        k: v.detach().clone() for k, v in policy.forward_actor(obs).items()
    }
    lr_round2 = [float(g["lr"]) for g in tr2.optimizer.param_groups]
    for p in policy.parameters():
        p.grad = None
    torch.manual_seed(303)
    policy.vb_logits.weight.grad = 0.1 * torch.randn_like(
        policy.vb_logits.weight
    )
    policy.vb_logits.bias.grad = 0.1 * torch.randn_like(policy.vb_logits.bias)
    wB = tr2.ksg_watch_step(obs, mb, old_buf2, lr_round2)
    detail["g3_B"] = {
        k: wB[k] for k in
        ("accepted", "rejected", "lr_scale", "backtracks",
         "joint", "z", "res", "vb")
    }
    okB = (
        bool(wB["accepted"]) and wB["lr_scale"] == 1.0
        and 0.0 < wB["vb"] <= G3_KL_TARGET
    )
    _report("G3.4 contrast: small vb-only step at normal lr accepted", okB,
            f"accepted={wB['accepted']} kl_vb={wB['vb']:.4g}")
    ok = ok and okB

    kl_post = tr2.ksg_eval_kl(obs, mb, old_buf2)
    ident = abs(
        kl_post["joint"] - (kl_post["z"] + kl_post["res"] + kl_post["vb"])
    )
    detail["g3_joint_identity_maxdev"] = ident
    okI = ident <= 1e-6 * max(1.0, abs(kl_post["joint"]))
    _report("G3.5 ksg_eval_kl identity: joint == z + res + vb", okI,
            f"maxdev={ident:.3e}")
    ok = ok and okI

    return ok, detail


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="D049b-fix consumer gate: vb continuous-action contract "
                    "(G1 contract / G2 credit assignment / G3 rollback)"
    )
    ap.add_argument("--gate", choices=["1", "2", "3", "all"], default="all",
                    help="which gate(s) to run (default: all)")
    ap.add_argument("--out", default=None,
                    help="report JSON path (default: apt_g1/outputs/"
                         "d049_consumer_gate_report.json under repo root)")
    args = ap.parse_args()
    out_path = Path(
        args.out
        or (REPO_ROOT / "apt_g1" / "outputs" / "d049_consumer_gate_report.json")
    )

    gates_run: dict = {}
    if args.gate in ("1", "all"):
        ok1, d1 = gate1()
        gates_run["G1"] = {"pass": bool(ok1), **d1}
    if args.gate in ("2", "all"):
        ok2, d2 = gate2()
        gates_run["G2"] = {"pass": bool(ok2), **d2}
    if args.gate in ("3", "all"):
        ok3, d3 = gate3()
        gates_run["G3"] = {"pass": bool(ok3), **d3}

    all_pass = all(g["pass"] for g in gates_run.values()) and bool(gates_run)
    report = {
        "script": "apt_g1/isaac/d049_consumer_gate.py",
        "purpose": "D049b-fix P1 action-probability contract consumer gate "
                   "(owner frozen route, DS_CONTINUOUS_EXECUTION_PLAN §5j / "
                   "commit 1fe31a8)",
        "torch_version": torch.__version__,
        "criteria": {
            "g2_v_targets": list(G2_V_TARGETS),
            "g2_cmd": G2_CMD,
            "g2_band": list(G2_BAND),
            "g2_iters": G2_ITERS,
            "g2_tail_mean": G2_TAIL,
            "g2_treated_N": G2_TREATED_N,
            "g2_treated_lr": G2_TREATED_LR,
            "g2_treated_vb_init_std": G2_TREATED_VB_INIT_STD,
            "g2_control_N": G2_CONTROL_N,
            "g2_control_seeds": list(G2_CONTROL_SEEDS),
            "g3_kl_target": G3_KL_TARGET,
            "g3_backtracks": G3_BACKTRACKS,
        },
        "gates": gates_run,
        "all_pass": bool(all_pass),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    n_pass = sum(1 for g in gates_run.values() if g["pass"])
    print(
        f"\n{n_pass}/{len(gates_run)} gates PASS  (report: {out_path})",
        flush=True,
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
