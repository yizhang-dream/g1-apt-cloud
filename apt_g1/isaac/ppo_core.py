"""Vectorized PPO for the APT Isaac env with paper-style training extras.

Implements the RL-stage mechanisms from APT-RL that were missing in the
MuJoCo attempts:

- latent action KL regularization w.r.t. N(0, I) (paper coefficient 2.5e-6)
- latent-space exploration bonus that decays to zero over training
  (paper: "exploration bonus that progressively decayed to zero")
- standard PPO (clip, GAE, entropy) on the phase latent + aux action
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class AptPPOPolicy(nn.Module):
    """MLP actor-critic: obs -> phase(2) + aux(12) [+ gate logits] + value."""

    def __init__(
        self,
        obs_dim: int,
        aux_dim: int = 12,
        gate_k: int = 0,
        vb_head: bool = False,
        hidden_dim: int = 256,
        phase_init_std: float = -4.0,
        aux_init_std: float = -4.0,
        use_phase: bool = True,
        latent_dim: int = 0,
        vb_init_std: float = -4.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.aux_dim = aux_dim
        self.gate_k = gate_k
        # D049b-fix：连续 vb 软权重臂的可选头（3 维 logits）。False = 不建头，
        # 参数量/前向/RNG 路径与旧逐位一致；True 时动作 = 连续高斯采样
        # vb_action = vb_logits + exp(vb_log_std)·ε（log_std 头与 z/res 的
        # phase_log_std/aux_log_std 完全同款：nn.Parameter 全 init_std 填充），
        # 同一张量贯穿 buffer → env（softmax(vb_action)=带噪软权重 w）→
        # update() 重算，动作-概率契约闭合。旧 categorical 索引簿记实现
        # （sample 进 log_prob 而 env 执行 softmax(logits)，E[A∇log p(k)]=0）
        # 已整体删除，不留旗标——旧 D049b ckpt 缺 vb_log_std 键，strict 加载
        # 即明确报错（owner 冻结口径：判执行失败不兼容是预期）。
        self.vb_head = bool(vb_head)
        self.use_phase = use_phase
        self.latent_dim = latent_dim
        # E49：False = aux 头采样后丢弃（latent/token/to42 模式 action=act["phase"]）。
        # 此时 act() 与 update() 的 log_prob/entropy 都不得含 aux 项——
        # 两处必须是同一约定，否则 ratio 全错。例外：latent_residual 的 aux
        # 头是 29d 残差执行动作，恒 True。
        self.aux_executed = True
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        if use_phase:
            self.phase_mean = nn.Linear(hidden_dim, 2)
            self.phase_log_std = nn.Parameter(torch.full((2,), phase_init_std))
        if latent_dim > 0:
            # E27 latent head: canonical "phase_mean/phase_log_std" keys so the
            # trainer's latent-KL / warmstart machinery works unchanged.
            self.phase_mean = nn.Linear(hidden_dim, latent_dim)
            self.phase_log_std = nn.Parameter(
                torch.full((latent_dim,), phase_init_std)
            )
        self.aux_mean = nn.Linear(hidden_dim, aux_dim)
        self.aux_log_std = nn.Parameter(torch.full((aux_dim,), aux_init_std))
        if gate_k > 0:
            self.gate_logits = nn.Linear(hidden_dim, gate_k)
        if self.vb_head:
            # D049b-fix：vb 头（3 维 logits→softmax=w）+ 高斯 log_std 参数头
            # （z/res 的 log_std 同款写法：Parameter 全量 init 值填充）
            self.vb_logits = nn.Linear(hidden_dim, 3)
            self.vb_log_std = nn.Parameter(torch.full((3,), vb_init_std))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_actor(self, obs: torch.Tensor) -> dict:
        x = self.encoder(obs)
        out = {
            "aux_mean": self.aux_mean(x),
            "aux_log_std": self.aux_log_std.expand_as(self.aux_mean(x)),
        }
        if self.use_phase or self.latent_dim > 0:
            out["phase_mean"] = self.phase_mean(x)
            out["phase_log_std"] = self.phase_log_std.expand_as(self.phase_mean(x))
        if self.gate_k > 0:
            out["gate_logits"] = self.gate_logits(x)
        if self.vb_head:
            out["vb_logits"] = self.vb_logits(x)
            out["vb_log_std"] = self.vb_log_std.expand_as(out["vb_logits"])
        return out

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Return dict of sampled actions, log_prob, entropy, value.

        aux_executed=False（latent/token 模式：aux 采样后丢弃）时 log_prob 与
        entropy 只累计 phase（+ gate）项；aux 照常采样并放进返回 dict。
        """
        p = self.forward_actor(obs)
        ad = Normal(p["aux_mean"], p["aux_log_std"].exp())
        aux = ad.mean if deterministic else ad.sample()
        out = {"aux": aux}
        # E49-C v2：分布参数随 act() 带出（detach 视图；真正 clone 快照由存储
        # 侧做）。纯加键，eval 等只读 phase/aux/gate 的调用方向后兼容，RNG 路径
        # 零变化
        out["aux_mean"] = p["aux_mean"].detach()
        out["aux_log_std"] = p["aux_log_std"].detach()
        if self.aux_executed:
            log_prob = ad.log_prob(aux).sum(-1)
            entropy = ad.entropy().sum(-1)
        else:
            log_prob = 0.0
            entropy = 0.0
        if self.use_phase or self.latent_dim > 0:
            pd = Normal(p["phase_mean"], p["phase_log_std"].exp())
            phase = pd.mean if deterministic else pd.sample()
            out["phase"] = phase
            if self.latent_dim > 0:
                out["latent"] = phase
            # E49-C v2：phase 头分布参数一并带出（use_phase/latent 任一为真即存在）
            out["phase_mean"] = p["phase_mean"].detach()
            out["phase_log_std"] = p["phase_log_std"].detach()
            log_prob = log_prob + pd.log_prob(phase).sum(-1)
            entropy = entropy + pd.entropy().sum(-1)
        if self.gate_k > 0:
            gd = Categorical(logits=p["gate_logits"])
            gate = gd.sample() if not deterministic else gd.probs.argmax(-1)
            out["gate"] = gate
            log_prob = log_prob + gd.log_prob(gate)
            entropy = entropy + gd.entropy()
        if self.vb_head:
            # D049b-fix：vb 动作 = 连续高斯采样（z/res 头完全同构）——
            # vb_action = vb_logits + exp(vb_log_std)·ε，det 模式取均值
            # （=vb_logits，softmax 与旧确定性 w 逐位一致，评测口径不变）。
            # 调用方把 act 返回的 vb_action 同值存进 buffer、拼进送 env 的
            # 动作段（env 侧 softmax(vb_action) 即带噪软权重），update() 用
            # buffer 值对 Normal(vb_logits, vb_log_std) 重算 log_prob——
            # 回报经由 softmax(vb_action) 依赖采样值，E[A∇log p]≠0，
            # vb 头获得合法策略梯度
            vbd = Normal(p["vb_logits"], p["vb_log_std"].exp())
            vb_action = vbd.mean if deterministic else vbd.sample()
            out["vb_action"] = vb_action
            log_prob = log_prob + vbd.log_prob(vb_action).sum(-1)
            entropy = entropy + vbd.entropy().sum(-1)
        return out, log_prob, entropy, self.get_value(obs), p


def kl_normal_std_normal(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """D_KL(N(mean, diag sigma^2) || N(0, I)), summed over latent dims."""
    var = torch.exp(2.0 * log_std)
    return 0.5 * (mean.pow(2) + var - 2.0 * log_std - 1.0).sum(-1)


def kl_normal(
    mean: torch.Tensor, log_std: torch.Tensor, prior_mean: torch.Tensor
) -> torch.Tensor:
    """D_KL(N(mean, sigma^2) || N(prior_mean, I)), summed over latent dims.

    E29: a non-zero prior mean (e.g. the walk posterior z_walk) keeps the policy's
    latent on the SONIC skill manifold instead of pulling it toward the origin.
    """
    var = torch.exp(2.0 * log_std)
    d = mean - prior_mean
    return 0.5 * (d.pow(2) + var - 2.0 * log_std - 1.0).sum(-1)


def kl_diag_gaussian(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    """对角高斯解析 KL，rsl_rl 口径 KL(new‖old)，逐样本（动作维已 sum）。

    E49-C KL 信任域守卫用（v2 口径）：old = rollout 采样时存储的分布参数
    （数据参照），new = optimizer.step 后的前向重算。v1 的 old 取步前 forward
    结果已废弃——那是"单步前后"对比，连续小步相对采样策略可累计走远。
    """
    var_old = torch.exp(2.0 * old_log_std)
    var_new = torch.exp(2.0 * new_log_std)
    return (
        old_log_std
        - new_log_std
        + (var_new + (new_mean - old_mean) ** 2) / (2.0 * var_old)
        - 0.5
    ).sum(-1)


def kl_diag_gaussian_reverse(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    """对角高斯解析 KL，D048i 步级看门口径 KL(old‖new)，逐样本（动作维已 sum）。

    与上方 kl_diag_gaussian（rsl_rl 口径 KL(new‖old)，E49-C rollout 参照）方向
    相反且用途不同：这里 old = 本轮更新开始时的策略（update() 内 B2 缓存，
    不是 rollout 采样分布），new = 候选 optimizer step 后的前向重算。闭式式：
    KL(old‖new) = log(σn/σo) + (σo² + (μo−μn)²)/(2σn²) − 1/2。
    """
    var_old = torch.exp(2.0 * old_log_std)
    var_new = torch.exp(2.0 * new_log_std)
    return (
        new_log_std
        - old_log_std
        + (var_old + (old_mean - new_mean) ** 2) / (2.0 * var_new)
        - 0.5
    ).sum(-1)


def kl_categorical_reverse(
    old_logits: torch.Tensor, new_logits: torch.Tensor
) -> torch.Tensor:
    """离散分布（gate 头）解析 KL(old‖new)，逐样本（D048i）。

    不进步级看门的联合目标（owner 口径「按动作维求和」= 连续动作头）；策略带
    gate 头时仅顺带记录 kl_gate。D049b-fix 后 vb 头为连续高斯动作，其更新约束
    用 kl_diag_gaussian_reverse（解析 KL）并纳入 ksg joint 目标，不再走本函数。
    """
    log_po = torch.log_softmax(old_logits, dim=-1)
    log_pn = torch.log_softmax(new_logits, dim=-1)
    return (torch.softmax(old_logits, dim=-1) * (log_po - log_pn)).sum(-1)


def vb_bc_target_w(
    cmd: torch.Tensor, anchors=(0.29, 0.47, 1.31)
) -> torch.Tensor:
    """D051：BC 目标软权重 w*(cmd)——把 cmd 投影到 anchors 凸包后分段线性
    分配到相邻两档。

    语义 = argmin_{w∈Δ²} (w·v − cmd)²，v = anchors（speedA init 档位图
    D048n；§5k 预注册）。cmd ≤ v0 → (1,0,0)；cmd ≥ v2 → (0,0,1)；
    v0..v1 / v1..v2 在相邻两档间线性分配（档位边界 v1 处两段一致 =(0,1,0)）。
    数值自检点：0.4→(0.389,0.611,0)、0.6→(0,0.845,0.155)。
    纯函数（只依赖 torch），train 预热与 d049_consumer_gate G4 共用同一
    实现（单一事实源）。cmd 任意 shape（m/s），返回 (..., 3)。
    """
    v0, v1, v2 = (float(a) for a in anchors)
    c = torch.clamp(cmd, min=v0, max=v2)
    flat = c.reshape(-1)
    w = torch.zeros(flat.numel(), 3, dtype=c.dtype, device=c.device)
    m_lo = flat <= v1  # 下半段 [v0, v1]：分配到 0/1 两档
    r_lo = (v1 - flat[m_lo]) / (v1 - v0)
    w[m_lo, 0] = r_lo
    w[m_lo, 1] = 1.0 - r_lo
    m_hi = flat > v1  # 上半段 [v1, v2]：分配到 1/2 两档
    r_hi = (v2 - flat[m_hi]) / (v2 - v1)
    w[m_hi, 1] = r_hi
    w[m_hi, 2] = 1.0 - r_hi
    return w.reshape(*c.shape, 3)


def vb_bc_loss(
    policy: "AptPPOPolicy", obs: torch.Tensor, cmd: torch.Tensor
) -> torch.Tensor:
    """D051：vb 头 BC 监督损失——CE(softmax(vb_logits(obs)), w*(cmd))。

    交叉熵以 w* 为目标分布（log_softmax 形式规避 log 0，§5k 预注册）。
    只应配「仅 policy.vb_logits 参数进 optimizer」的更新（encoder/z/res/
    vb_log_std 全冻结 = 单变量纪律）；本函数自身不改任何参数、纯前向+求和。
    """
    p = policy.forward_actor(obs)
    w_star = vb_bc_target_w(cmd)
    return -(w_star * torch.log_softmax(p["vb_logits"], dim=-1)).sum(-1).mean()


class PPOTrainer:
    def __init__(
        self,
        policy: AptPPOPolicy,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        # 默认 1 = 历史 run 的实际行为（旧 update() 从未循环，此参数形同虚设）；
        # 多 epoch 只能经 CLI --ppo-epochs 显式开启
        num_epochs: int = 1,
        minibatch_size: int = 512,
        entropy_coef: float = 0.001,
        # D048b：res（aux）头熵奖励独立系数，只乘 aux 头熵项。默认 1.0 =
        # 历史行为逐位不变；latent_residual 的 aux=29d 执行残差，共享熵奖励
        # 会持续推高残差探索噪声（E48 破坏模式的种子之一）
        res_ent_coef: float = 1.0,
        latent_kl_coef: float = 2.5e-6,
        latent_expl_coef: float = 0.01,
        latent_prior_mean: "torch.Tensor | None" = None,
        max_grad_norm: float = 0.5,
        max_iters: int = 500,
        value_coef: float = 0.5,
        device: str = "cuda:0",
        decoder_reg_coef: float = 1.0,  # E44: drift regularizer for decoder fine-tune
        decoder_lr: "float | None" = None,  # E44: separate (smaller) decoder LR
        decoder_wreg_coef: float = 0.0,  # E44v2: weight-space anchor to official
        skip_nan: bool = True,  # E44: skip optimizer step if any grad is NaN
        fused: bool = False,  # fused Adam（CUDA 融合实现）；默认 False 保持 E 系列历史 run 可比
        kl_guard: "float | None" = None,  # E49-C: KL 信任域阈值；None = 完全关闭（默认）
        kl_guard_shrink: float = 0.8,  # E49-C: 回滚时 lr 乘子（TRPO 回溯惯例）
        kl_guard_grow: float = 1.2,  # E49-C: KL < 阈/2 时 lr 回升乘子（镜像 rsl_rl 自适应；1.0 = 关；上限钉在初始 lr）
        kl_guard_max_rolls: int = 3,  # E49-C: 同一 update() 内连续回滚达此数 → 提前结束整个循环
        # D048i：步级解析 KL 看门（默认 False = 完全关闭；与 kl_guard 互斥，
        # 互斥由 train_apt_isaac CLI 强制）。开启时每个 optimizer step 前快照
        # 参数+Adam 状态，步后在当前 minibatch 观测上测 KL(old‖new)（old =
        # 本轮更新开始策略的分布），超 kl_step_target 则 lr 减半重试至
        # kl_backtracks 次，仍超则拒绝该步（参数+Adam 状态还原、lr 复位）
        kl_step_guard: bool = False,
        kl_step_target: float = 0.05,
        kl_backtracks: int = 6,
    ):
        self.policy = policy.to(device)
        self.device = device
        if decoder_lr is not None and getattr(policy, "decoder_ft", False):
            dec_params = [
                p
                for n, p in policy.named_parameters()
                if n.startswith("decoder.") and p.requires_grad
            ]
            rest_params = [
                p
                for n, p in policy.named_parameters()
                if not n.startswith("decoder.") and p.requires_grad
            ]
            self.optimizer = torch.optim.Adam(
                [
                    {"params": rest_params},
                    {"params": dec_params, "lr": decoder_lr},
                ],
                lr=lr,
                fused=fused,
            )
        else:
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, fused=fused)
        # E49-C：KL 信任域守卫的 lr 状态（回滚缩小 / 通过回升，上限钉在初始 lr）
        self._lr0 = self.optimizer.param_groups[0]["lr"]
        self._lr_now = self._lr0
        self.skip_nan = skip_nan
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.res_ent_coef = res_ent_coef
        self.latent_kl_coef = latent_kl_coef
        self.latent_expl_coef = latent_expl_coef
        # E29: prior mean for the latent KL. None -> N(0, I) (E27 behavior).
        if latent_prior_mean is not None:
            self.latent_prior_mean = latent_prior_mean.to(self.device)
        else:
            self.latent_prior_mean = None
        self.max_grad_norm = max_grad_norm
        self.max_iters = max_iters
        self.value_coef = value_coef
        self.decoder_reg_coef = decoder_reg_coef
        self.decoder_wreg_coef = decoder_wreg_coef
        # E49-C：KL 信任域守卫（kl_guard=None = 完全关闭，默认 = 冻结版行为）
        self.kl_guard = kl_guard
        self.kl_guard_shrink = kl_guard_shrink
        self.kl_guard_grow = kl_guard_grow
        self.kl_guard_max_rolls = kl_guard_max_rolls
        # D048i：步级解析 KL 看门状态（kl_step_guard=False 时 update() 内
        # ksg 分支完全不进入，行为与冻结版逐位一致）
        self.kl_step_guard = kl_step_guard
        self.kl_step_target = kl_step_target
        self.kl_backtracks = kl_backtracks
        self.it = 0

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        truncated: torch.Tensor,
        last_value: torch.Tensor,
        trunc_values: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """rewards/values/dones/truncated: (T, N); trunc_values: (T, N) 或 None。
        returns advantages (T, N).

        边界语义（E49 修订）：terminated（dones，摔倒）= 真终局 → 自举 0 +
        切断优势递推；truncated（超时）= rollout 人为截断 → 递推仍切断，但
        自举改用复位前终末状态的价值 trunc_values[t]（env._reset_idx 在 super()
        之前截留终末 obs，非复位后 obs 的价值）；t == T-1 且 trunc 时
        trunc_values[T-1] 优先于 last_value。trunc_values=None 时保持旧行为
        （done/trunc 自举均为 0，向后兼容旧调用方）。
        """
        T, N = rewards.shape
        adv = torch.zeros_like(rewards)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            rec = ~(dones[t] | truncated[t])  # 递推切断掩码：done 与 trunc 都切断
            if trunc_values is None:
                # 旧行为：cont 同控自举与递推，done/trunc 步自举均为 0
                boot = rec
            else:
                # E49：仅 done 切断自举；超时步自举复位前终末状态价值
                boot = ~dones[t]
                next_value = torch.where(
                    dones[t],
                    torch.zeros_like(next_value),
                    torch.where(truncated[t], trunc_values[t], next_value),
                )
            delta = rewards[t] + self.gamma * next_value * boot.float() - values[t]
            gae = delta + self.gamma * self.lam * rec.float() * gae
            adv[t] = gae
        return adv

    def _grads_nonfinite(self) -> bool:
        """E44 NaN guard 的批量版：任何参数梯度含 NaN/Inf 则返回 True。

        原实现对全部参数逐个 ``torch.isfinite(p.grad).all()``（N 次小 kernel +
        N 次隐式同步）；改用 ``torch._foreach_norm`` 一次算出各参数梯度的 L2
        范数（NaN/Inf 任一元素都会让范数非有限，检出语义不变），再
        ``isfinite(...).all().item()`` —— 每 minibatch 仅一次同步。数值上无假
        阳性：guard 在 clip_grad_norm_ 之后执行，有限梯度已被缩放到总范数
        max_grad_norm 内，逐元素平方求和不会溢出。
        """
        grads = [p.grad for p in self.policy.parameters() if p.grad is not None]
        if not grads:
            return False
        norms = torch.stack(torch._foreach_norm(grads))
        return not bool(torch.isfinite(norms).all().item())

    # ------------------------------------------------------------------
    # D048i：步级解析 KL 看门（kl_step_guard）工具方法。全部纯 torch、
    # 不依赖 isaac，可被 test_kl_step_guard.py 在 CPU 上独立调用。
    # ------------------------------------------------------------------

    def ksg_snapshot(self):
        """D048i B3：optimizer 全部参数（clone）+ Adam 状态（deepcopy）快照。

        返回 (参数 clone 列表, optimizer.state_dict 深拷贝, 参数引用列表)。
        每个被看门的 step 各自快照（单次使用，下个 minibatch 重新快照）。
        """
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        return (
            [p.detach().clone() for p in params],
            copy.deepcopy(self.optimizer.state_dict()),
            params,
        )

    def ksg_restore(self, snap) -> None:
        """D048i B3：还原参数 + Adam 状态（逐位精确）。

        用 copy_ 原位写参数（保留 .grad 引用——回退重跑 step 时梯度仍是在
        还原后的参数处计算的，无需重算前向/反传）；load_state_dict 前 deepcopy
        一次防快照张量被别名进 optimizer.state（与 kl_guard 同款防御）。
        """
        snap_p, snap_o, params = snap
        with torch.no_grad():
            for p, s in zip(params, snap_p):
                p.copy_(s)
        self.optimizer.load_state_dict(copy.deepcopy(snap_o))

    def ksg_joint_logp(
        self, p_fwd, obs, phase, aux, gate, aux_scored=True, vb=None
    ) -> torch.Tensor:
        """D048i B1：从 forward dict 重算存量 action 的联合 logp。

        分支口径照抄 update() 末尾 post_update_kl 的整批重算（decoder_ft /
        aux_scored / phase / gate 四分支同构；D049b-fix 后 vb 分支为高斯
        log_prob——vb 实参 = buffer 存储的连续 vb_action，对
        Normal(vb_logits, vb_log_std) 求密度，与 act() 同一分布参数），
        供资格检查复用——既有 post_update_kl 的计算本身不动。
        """
        if getattr(self.policy, "decoder_ft", False):
            aux_mean, _ = self.policy.action_mean(phase, obs)
            ad = Normal(aux_mean, p_fwd["aux_log_std"].exp())
            lp = ad.log_prob(aux).sum(-1)
        else:
            ad = Normal(p_fwd["aux_mean"], p_fwd["aux_log_std"].exp())
            lp = ad.log_prob(aux).sum(-1) if aux_scored else 0.0
        if phase is not None:
            pd = Normal(p_fwd["phase_mean"], p_fwd["phase_log_std"].exp())
            lp = lp + pd.log_prob(phase).sum(-1)
        if gate is not None:
            gd = Categorical(logits=p_fwd["gate_logits"])
            lp = lp + gd.log_prob(gate)
        if vb is not None:
            # D049b-fix：vb 分量 = 高斯 log_prob（z/res 重算路径同款）
            vbd = Normal(p_fwd["vb_logits"], p_fwd["vb_log_std"].exp())
            lp = lp + vbd.log_prob(vb).sum(-1)
        return lp

    def ksg_qualification(
        self, obs, phase, aux, gate, logp_old, aux_scored=True, vb=None
    ) -> dict:
        """D048i B1：资格检查——重算联合 logp 与存储 logp_old 的偏差。

        返回 dict：qual_logp_maxdev = max|Δlogp|、qual_ratio_maxdev =
        max|exp(Δlogp)−1|。函数级独立可调（测试篡改 logp_old 的入口）。
        """
        with torch.no_grad():
            p0 = self.policy.forward_actor(obs)
            lp = self.ksg_joint_logp(
                p0, obs, phase, aux, gate, aux_scored, vb=vb
            )
            d = lp - logp_old
            return {
                "qual_logp_maxdev": float(d.abs().max().item()),
                "qual_ratio_maxdev": float((d.exp() - 1.0).abs().max().item()),
            }

    def ksg_eval_kl(self, obs_mb, mb, old_buf) -> dict:
        """D048i B3：候选步后 KL(old‖new) 评估（no_grad，当前 minibatch 观测）。

        每头闭式 KL 按动作维求和、再按 minibatch 样本平均；联合 = res(aux)头
        + z(phase)头 + vb 头。vanilla 无 z 头（按键存在性跳过）；latent/token
        模式的 aux 头虽不执行但共享 encoder 主干，照测（KL 仍度量策略移动）；
        gate 头只顺带记录、不进联合目标（owner 口径「按动作维求和」= 连续
        动作头）。D049b-fix：vb 头为连续高斯动作（vb_logits, vb_log_std），
        解析 KL 纳入联合目标——joint = kl_z + kl_res + kl_vb（不再仅记录）。
        返回 dict（全 float）：joint/z/res/gate/vb。
        """
        with torch.no_grad():
            p_new = self.policy.forward_actor(obs_mb)
            kl_z = 0.0
            kl_res = 0.0
            if "phase_mean" in old_buf and "phase_mean" in p_new:
                kl_z = float(
                    kl_diag_gaussian_reverse(
                        old_buf["phase_mean"][mb],
                        old_buf["phase_log_std"][mb],
                        p_new["phase_mean"],
                        p_new["phase_log_std"],
                    )
                    .mean()
                    .item()
                )
            if "aux_mean" in old_buf and "aux_mean" in p_new:
                kl_res = float(
                    kl_diag_gaussian_reverse(
                        old_buf["aux_mean"][mb],
                        old_buf["aux_log_std"][mb],
                        p_new["aux_mean"],
                        p_new["aux_log_std"],
                    )
                    .mean()
                    .item()
                )
            kl_gate = 0.0
            if "gate_logits" in old_buf and "gate_logits" in p_new:
                kl_gate = float(
                    kl_categorical_reverse(
                        old_buf["gate_logits"][mb], p_new["gate_logits"]
                    )
                    .mean()
                    .item()
                )
            kl_vb = 0.0
            if "vb_logits" in old_buf and "vb_logits" in p_new:
                # D049b-fix：vb 头高斯解析 KL(old‖new)，纳入联合目标
                # （old = 本轮更新开始策略的 forward 分布参数，含 vb_log_std）
                kl_vb = float(
                    kl_diag_gaussian_reverse(
                        old_buf["vb_logits"][mb],
                        old_buf["vb_log_std"][mb],
                        p_new["vb_logits"],
                        p_new["vb_log_std"],
                    )
                    .mean()
                    .item()
                )
            return {"joint": kl_z + kl_res + kl_vb, "z": kl_z, "res": kl_res,
                    "gate": kl_gate, "vb": kl_vb}

    def ksg_watch_step(
        self, obs_mb, mb, old_buf, lr_round, target=None, backtracks=None
    ) -> dict:
        """D048i B3：单 optimizer step 的看门执行（快照→step→KL→lr 减半回退）。

        lr_round = 本轮开始时各 param_group 的 lr（看门只临时改 lr，接受/拒绝
        后一律复位为该值——lr 缩放是单步重试手段，不跨步粘滞）。重试时参数+
        Adam 状态还原到 step 前快照，.grad 仍有效（它是在还原后参数处计算的），
        无需重算前向/反传。返回 dict：accepted/rejected/lr_scale/backtracks/
        joint/z/res/gate（KL 为最终尝试的达成值；NaN/Inf KL 恒判不通过）。
        """
        target = self.kl_step_target if target is None else target
        backtracks = self.kl_backtracks if backtracks is None else backtracks
        snap = self.ksg_snapshot()
        kl = {"joint": float("inf"), "z": 0.0, "res": 0.0, "gate": 0.0,
              "vb": 0.0}
        for attempt in range(backtracks + 1):
            scale = 0.5 ** attempt
            for g, lr0 in zip(self.optimizer.param_groups, lr_round):
                g["lr"] = lr0 * scale
            self.optimizer.step()
            kl = self.ksg_eval_kl(obs_mb, mb, old_buf)
            if kl["joint"] <= target:
                for g, lr0 in zip(self.optimizer.param_groups, lr_round):
                    g["lr"] = lr0
                return {
                    "accepted": True, "rejected": False,
                    "lr_scale": scale, "backtracks": attempt, **kl,
                }
            if attempt < backtracks:
                # 回退重试：参数 + Adam 状态还原后以减半 lr 重跑同一梯度
                self.ksg_restore(snap)
        # 回退预算耗尽：拒绝该步（还原快照、复位 lr），继续下一 minibatch
        self.ksg_restore(snap)
        for g, lr0 in zip(self.optimizer.param_groups, lr_round):
            g["lr"] = lr0
        return {
            "accepted": False, "rejected": True,
            "lr_scale": 0.5 ** backtracks, "backtracks": backtracks, **kl,
        }

    def update(
        self,
        rollout: dict,
        phase_labels: torch.Tensor | None = None,
        phase_warm_coef: float = 0.0,
    ) -> dict:
        """rollout: obs (T,N,D), phase/aux/gate, logp, value, reward, done, trunc."""
        T, N = rollout["obs"].shape[:2]
        # E49: rollout 带 "trunc_value"（复位前终末状态价值）时启用超时步自举
        trunc_values = rollout.get("trunc_value")
        adv = self.compute_gae(
            rollout["reward"], rollout["value"], rollout["done"], rollout["trunc"], rollout["last_value"],
            trunc_values=trunc_values,
        )
        returns = adv + rollout["value"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = rollout["obs"].reshape(T * N, -1)
        phase = (
            rollout["phase"].reshape(T * N, -1)
            if rollout.get("phase") is not None
            else None
        )
        aux = rollout["aux"].reshape(T * N, -1)
        logp_old = rollout["logp"].reshape(-1)
        adv_f = adv.reshape(-1)
        ret_f = returns.reshape(-1)
        val_f = rollout["value"].reshape(-1)
        # E49-C v2：KL 守卫 old 侧参照 = rollout 采样时的分布参数（训练侧仅在
        # --kl-guard 开启时分配填充；log_std 已在存储时 clone，与策略参数无共享
        # 存储）。展平顺序必须与 obs/logp 完全一致（[T,N,dim] -> [T*N,dim] 行主
        # 序），minibatch 索引 [mb] 的逐元素配对才成立，错配是静默错误。守卫开
        # 启而 rollout 缺 buffer = 调用方组装错误，宁可硬失败也不静默退回 v1 的
        # 步内参照口径。
        if self.kl_guard is not None:
            _old_keys = (
                "phase_mean_old",
                "phase_log_std_old",
                "aux_mean_old",
                "aux_log_std_old",
            )
            _missing = [k for k in _old_keys if rollout.get(k) is None]
            if _missing:
                raise ValueError(
                    "kl_guard enabled but rollout lacks old-distribution "
                    f"buffers: {_missing} (train_apt_isaac.py allocates/fills "
                    "them under --kl-guard)"
                )
            phase_mean_old = rollout["phase_mean_old"].reshape(T * N, -1)
            phase_log_std_old = rollout["phase_log_std_old"].reshape(T * N, -1)
            aux_mean_old = rollout["aux_mean_old"].reshape(T * N, -1)
            aux_log_std_old = rollout["aux_log_std_old"].reshape(T * N, -1)
        # E49-C：价值解释方差（return 被 critic 解释的比例，1 = 完美拟合）。
        # 无条件记录的纯日志量，不影响训练与 RNG 路径
        with torch.no_grad():
            expl_var = float(
                (1.0 - torch.var(ret_f - val_f) / (torch.var(ret_f) + 1e-8)).item()
            )
        gate = rollout.get("gate")
        if gate is not None:
            gate = gate.reshape(-1)
        # D049b-fix：vb 头的连续动作槽位（rollout 侧由 buf["vb_action"] 提供，
        # = act() 采样并送 env 执行的同一张量；旗标关的 run 无此键 → None →
        # 全部分支跳过，与旧行为逐位一致）
        vb = rollout.get("vb_action")
        if vb is not None:
            vb = vb.reshape(T * N, -1)
        # D051 伴生插桩（D050-② 内置，§5k；只读不改训练）：advantage 与
        # buffer vb_action 的逐维 Pearson 相关（3 维），判读 BC 预热后 PPO
        # 更新侧学习信号是否出现。adv 源 = 与 ratio 计算同一个 adv_f（Pearson
        # 对仿射不变，归一化前后同值）；std 取与 1/n 协方差同口径的有偏估计
        #（= np.corrcoef 标准口径，torch.std 默认无偏会差 sqrt((n-1)/n) 因子）；
        # NaN/退化（std=0 → 0/0）自然记 NaN。vb 头关闭（无 vb_action 键）→
        # None，消费方 .get
        adv_vb_corr = None
        if vb is not None:
            with torch.no_grad():
                _a = adv_f - adv_f.mean()
                _v = vb - vb.mean(dim=0, keepdim=True)
                _den = _a.pow(2).mean().sqrt() * _v.pow(2).mean(dim=0).sqrt()
                adv_vb_corr = [
                    float(c) for c in (_a.unsqueeze(1) * _v).mean(dim=0) / _den
                ]
        if phase_labels is not None:
            phase_labels = phase_labels.reshape(-1, phase_labels.shape[-1])

        # decaying latent exploration coefficient (paper: exploration -> exploitation)
        expl_coef = self.latent_expl_coef * max(0.0, 1.0 - self.it / max(1, self.max_iters))

        # E49：aux_executed=False（latent/token/to42）时 aux 项不进 log_prob/
        # entropy；decft 的 aux 是 29d 实际动作，恒参与（DecFtPolicy 无此属性，
        # getattr 默认 True）；latent_residual 的 aux 同为 29d 执行动作，恒参与
        aux_scored = getattr(self.policy, "aux_executed", True)

        # D048i：步级解析 KL 看门（kl_step_guard，与 kl_guard 互斥由 CLI 强制）。
        # B1 资格检查 + B2 本轮 old 分布缓存共用同一次轮开始 no_grad 前向——
        # forward_actor 无采样/无 dropout，不消耗 RNG，guard 开/关的 randperm
        # 序列逐位一致（关闭路径零行为变化的组成部分）
        ksg_on = getattr(self, "kl_step_guard", False)
        if ksg_on:
            lr_round = [float(g["lr"]) for g in self.optimizer.param_groups]
            with torch.no_grad():
                p_round0 = self.policy.forward_actor(obs)
                # B2：本轮 KL 的 old 端分布参数（= 本轮更新开始策略，不是
                # 每个 minibatch 步前策略）；整批 T*N，显存占用极小
                old_buf = {k: v.detach().clone() for k, v in p_round0.items()}
                # B1 资格检查：重算存量 action 联合 logp vs rollout 存储值
                # （分支口径经 ksg_joint_logp 与 post_update_kl 同构）
                dqual = self.ksg_joint_logp(
                    p_round0, obs, phase, aux, gate, aux_scored, vb=vb
                ) - logp_old
                qual_logp_maxdev = float(dqual.abs().max().item())
                qual_ratio_maxdev = float((dqual.exp() - 1.0).abs().max().item())
                # qual_kl_self：公式自检——同一组 μ/σ 对自身的解析联合 KL，
                # 应≈0。工单口径为首轮自检，这里每轮计算：成本近零（纯
                # elementwise 于已物化的 old_buf），且 hist 序列保持无 null，
                # 首轮语义不变
                kl_self = 0.0
                if "phase_mean" in old_buf:
                    kl_self = kl_self + kl_diag_gaussian_reverse(
                        old_buf["phase_mean"], old_buf["phase_log_std"],
                        old_buf["phase_mean"], old_buf["phase_log_std"],
                    ).mean()
                if "aux_mean" in old_buf:
                    kl_self = kl_self + kl_diag_gaussian_reverse(
                        old_buf["aux_mean"], old_buf["aux_log_std"],
                        old_buf["aux_mean"], old_buf["aux_log_std"],
                    ).mean()
                if "vb_logits" in old_buf:
                    # D049b-fix：vb 头已是连续高斯，纳入公式自检（同组 μ/σ
                    # 对自身解析 KL 应≈0，与 joint=z+res+vb 口径一致）
                    kl_self = kl_self + kl_diag_gaussian_reverse(
                        old_buf["vb_logits"], old_buf["vb_log_std"],
                        old_buf["vb_logits"], old_buf["vb_log_std"],
                    ).mean()
                qual_kl_self = float(kl_self)
            # B4 的 θ_start（B2 时刻快照，覆盖 optimizer 全部参数，含 decft
            # 双 param_group）
            _opt_params = [
                p for g in self.optimizer.param_groups for p in g["params"]
            ]
            theta_start = [p.detach().clone() for p in _opt_params]
            ksg_total = ksg_acc = ksg_rej = 0
            ksg_scales: list = []
            ksg_kl_joint: list = []
            ksg_kl_z: list = []
            ksg_kl_res: list = []
            ksg_kl_gate: list = []
            ksg_kl_vb: list = []

        losses = []
        # E49-C：KL 守卫的连续回滚计数与提前停止标志（作用域 = 整个 update()）
        roll_streak = 0
        stop = False
        # 真 epoch 循环：每个 epoch 重新洗牌（默认 num_epochs=1 = 历史单遍）
        for _ in range(self.num_epochs):
            idx = torch.randperm(T * N, device=self.device)
            for start in range(0, T * N, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]
                p = self.policy.forward_actor(obs[mb])
                if getattr(self.policy, "decoder_ft", False):
                    # E44: the decoder is the policy's mean network. Recompute mu
                    # from the STORED latent z (phase) so PPO gradients reach the
                    # decoder weights; reg keeps it near the official decoder.
                    aux_mean, dec_reg = self.policy.action_mean(phase[mb], obs[mb])
                    ad = Normal(aux_mean, p["aux_log_std"].exp())
                    lp = ad.log_prob(aux[mb]).sum(-1)
                else:
                    ad = Normal(p["aux_mean"], p["aux_log_std"].exp())
                    # 与 act() 同一约定：aux 未被执行时不进 log_prob
                    lp = ad.log_prob(aux[mb]).sum(-1) if aux_scored else 0.0
                if phase is not None:
                    pd = Normal(p["phase_mean"], p["phase_log_std"].exp())
                    lp = lp + pd.log_prob(phase[mb]).sum(-1)
                if gate is not None:
                    gd = Categorical(logits=p["gate_logits"])
                    lp = lp + gd.log_prob(gate[mb])
                if vb is not None:
                    # D049b-fix：与 act() 同一约定——buffer 的连续 vb_action
                    # 对 Normal(vb_logits, vb_log_std) 重算高斯 log_prob
                    # （z/res 的重算路径同款）
                    vbd = Normal(p["vb_logits"], p["vb_log_std"].exp())
                    lp = lp + vbd.log_prob(vb[mb]).sum(-1)
                logratio = lp - logp_old[mb]
                ratio = logratio.exp()
                # E49 训练健康指标：k3 估计的 approx_kl（逐点非负）、超出
                # clip 窗口的样本比例、phase 头当前 std（detached，进 losses）
                approx_kl = (torch.exp(logratio) - 1.0 - logratio).mean()
                clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                act_std = (
                    p["phase_log_std"].mean().exp() if phase is not None else 0.0
                )
                surr1 = ratio * adv_f[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_f[mb]
                ploss = -torch.min(surr1, surr2).mean()
                vloss = torch.nn.functional.mse_loss(self.policy.get_value(obs[mb]), ret_f[mb])
                # D048b：熵分解——res(aux) 头熵乘 res_ent_coef（0 = 残差头不享受
                # 熵 bonus，log_prob/信任域不受影响）；aux_scored=False 时 aux
                # 熵本就不存在，该系数无效果
                ent_aux = ad.entropy().sum(-1) if aux_scored else 0.0
                ent = self.res_ent_coef * ent_aux
                if phase is not None:
                    ent_z = pd.entropy().sum(-1)
                    ent = ent + ent_z
                else:
                    ent_z = 0.0
                if gate is not None:
                    ent = ent + gd.entropy()
                if vb is not None:
                    # D049b-fix：vb 高斯微分熵（同 z/res 公式）
                    ent = ent + vbd.entropy().sum(-1)
                loss = (
                    ploss
                    + self.value_coef * vloss
                    - self.entropy_coef * ent.mean()
                )
                if phase is not None:
                    if self.latent_prior_mean is not None:
                        kl = kl_normal(
                            p["phase_mean"], p["phase_log_std"], self.latent_prior_mean
                        )
                    else:
                        kl = kl_normal_std_normal(p["phase_mean"], p["phase_log_std"])
                    loss = (
                        loss
                        - expl_coef * pd.entropy().sum(-1).mean()
                        + self.latent_kl_coef * kl.mean()
                    )
                if phase_labels is not None:
                    loss = (
                        loss
                        + phase_warm_coef
                        * torch.nn.functional.mse_loss(p["phase_mean"], phase_labels[mb])
                    )
                if getattr(self.policy, "decoder_ft", False):
                    # E44: keep the fine-tuned decoder near the official one
                    loss = loss + self.decoder_reg_coef * dec_reg
                    if self.decoder_wreg_coef > 0.0:
                        # E44v2: direct weight-space anchor (||theta - theta_ref||^2)
                        wdrift = sum(
                            (p - rp).pow(2).sum()
                            for p, rp in zip(
                                self.policy.decoder.net.parameters(),
                                self.policy.decoder_ref.net.parameters(),
                            )
                        )
                        loss = loss + self.decoder_wreg_coef * wdrift
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nan_skip = 0.0
                kl_mb = 0.0
                rolled = 0.0
                if self.skip_nan and self._grads_nonfinite():
                    # E44: NaN guard -- drop this minibatch's update (keeps the
                    # fine-tuned decoder from exploding the run)
                    self.optimizer.zero_grad()
                    nan_skip = 1.0
                elif ksg_on:
                    # D048i B3：步级看门（覆盖所有 optimizer step，含第一次）。
                    # NaN skip 的 minibatch 没有发生 step，不进看门计数。
                    # kl_guard 与本分支互斥（CLI 强制），下方 kl_guard 原路径不动
                    w_step = self.ksg_watch_step(obs[mb], mb, old_buf, lr_round)
                    ksg_total += 1
                    ksg_scales.append(w_step["lr_scale"])
                    ksg_kl_gate.append(w_step["gate"])
                    ksg_kl_vb.append(w_step["vb"])
                    if w_step["accepted"]:
                        ksg_acc += 1
                        ksg_kl_joint.append(w_step["joint"])
                        ksg_kl_z.append(w_step["z"])
                        ksg_kl_res.append(w_step["res"])
                    else:
                        ksg_rej += 1
                else:
                    if self.kl_guard is not None:
                        # E49-C v2：step 前快照 policy 参数 **和** Adam 状态
                        # （回滚 = 两者都还原；v1 只还原参数，被拒步留下的
                        # exp_avg/exp_avg_sq 会污染后续接受步——旧注释"Adam
                        # 动量不回滚（常规做法）"作废）。快照单次使用，下个
                        # minibatch 重新快照。
                        snap_p = {
                            k: v.detach().clone()
                            for k, v in self.policy.state_dict().items()
                        }
                        snap_o = copy.deepcopy(self.optimizer.state_dict())
                    self.optimizer.step()
                    if self.kl_guard is not None:
                        with torch.no_grad():
                            # E49-C v2：old 分布 = rollout 采样时存储的分布参数
                            # （与策略参数无别名），new 分布 = 步后前向重算 →
                            # 每个候选步测的是"当前策略 vs 产生数据的策略"：
                            # 连续小步相对 rollout 参照的累计漂移单步即可见
                            # （v1 用步前 forward 当 old，单步前后对比，逐步小
                            # 步各自过阈但累计可走远），首 minibatch 无裁剪的洞
                            # 也天然被覆盖。解析对角高斯 KL，rsl_rl 口径
                            # KL(new‖old)。两点局限：① decoder_ft 模式 old 侧用
                            # forward 头近似（该模式实际 mean 走
                            # policy.action_mean，forward 的 aux_mean 是占位零
                            # 张量，E49 不用该模式）；② gate 头为离散分布、vb 头
                            # （D049b-fix 高斯）均不进本 KL 路径——vb 的更新约束
                            # 由 ksg 步级看门的 joint（=z+res+vb）覆盖，两守卫
                            # CLI 互斥。
                            p_new = self.policy.forward_actor(obs[mb])
                            kl_t = None
                            if phase is not None:
                                kl_t = kl_diag_gaussian(
                                    phase_mean_old[mb],
                                    phase_log_std_old[mb],
                                    p_new["phase_mean"],
                                    p_new["phase_log_std"],
                                )
                            if aux_scored:
                                kl_aux = kl_diag_gaussian(
                                    aux_mean_old[mb],
                                    aux_log_std_old[mb],
                                    p_new["aux_mean"],
                                    p_new["aux_log_std"],
                                )
                                kl_t = kl_aux if kl_t is None else kl_t + kl_aux
                            if kl_t is not None:
                                kl_mb = float(kl_t.mean().item())
                        if kl_mb > self.kl_guard:
                            # 超阈：回滚本步（参数 + Adam 状态）+ 缩小 lr
                            # （TRPO 回溯惯例）；load 再 deepcopy 一次防 torch
                            # load_state_dict 把快照张量别名进 optimizer.state
                            self.policy.load_state_dict(snap_p)
                            self.optimizer.load_state_dict(copy.deepcopy(snap_o))
                            self._lr_now = max(
                                1e-6, self._lr_now * self.kl_guard_shrink
                            )
                            self.optimizer.param_groups[0]["lr"] = self._lr_now
                            roll_streak += 1
                            rolled = 1.0
                            if roll_streak >= self.kl_guard_max_rolls:
                                # 连续回滚达上限：提前结束整个 epoch×minibatch 循环
                                stop = True
                        else:
                            roll_streak = 0
                            if self.kl_guard_grow > 1.0 and kl_mb < self.kl_guard / 2:
                                # 通过且 KL 小：lr 回升（镜像 rsl_rl 自适应，
                                # 上限钉在初始 lr）
                                self._lr_now = min(
                                    self._lr0, self._lr_now * self.kl_guard_grow
                                )
                                self.optimizer.param_groups[0]["lr"] = self._lr_now
                # 统计张量先 detach 存 GPU（切断 autograd 图，不占显存），epoch 末
                # 统一 .item() —— 原先每 minibatch ~6 次同步；各 key 的均值口径不变
                losses.append(
                    {
                        "loss": loss.detach(),
                        "ploss": ploss.detach(),
                        "vloss": vloss.detach(),
                        "ent": ent.detach().mean(),
                        "kl_prior": kl.detach().mean() if phase is not None else 0.0,
                        "approx_kl": approx_kl.detach(),
                        "clip_frac": clip_frac.detach(),
                        "act_std": (
                            act_std.detach()
                            if isinstance(act_std, torch.Tensor)
                            else act_std
                        ),
                        "ent_aux": (
                            ent_aux.detach().mean()
                            if isinstance(ent_aux, torch.Tensor)
                            else ent_aux
                        ),
                        "ent_z": (
                            ent_z.detach().mean()
                            if isinstance(ent_z, torch.Tensor)
                            else ent_z
                        ),
                        "aux_std": p["aux_log_std"].mean().exp().detach(),
                        "expl": expl_coef,
                        "dreg": (
                            dec_reg.detach()
                            if getattr(self.policy, "decoder_ft", False)
                            else 0.0
                        ),
                        "nan_skip": nan_skip,
                    }
                )
                if self.kl_guard is not None:
                    # E49-C：守卫量进 per-minibatch losses（kl_mb/kl_roll 会随下方
                    # agg 循环出便捷均值；明细/总数/lr 在 agg 循环外单列）
                    losses[-1]["kl_mb"] = kl_mb
                    losses[-1]["kl_roll"] = rolled
                if stop:
                    break
            if stop:
                # E49-C：断路同时提前结束 epoch 循环
                break
        self.it += 1
        for d in losses:
            d.setdefault("kl_prior", 0.0)
        # 张量 key 在 GPU 上 stack+mean 后每 key 只做一次 .item()；标量 key
        # （expl/nan_skip 等）仍走原 np.mean
        agg = {}
        for k in losses[0]:
            vals = [d[k] for d in losses]
            if isinstance(vals[0], torch.Tensor):
                agg[k] = float(torch.stack(vals).mean().item())
            else:
                agg[k] = float(np.mean(vals))
        if self.kl_guard is not None:
            # E49-C：不进上方均值循环的守卫量（losses[0] 中无这些键）
            agg["kl_mb_all"] = [float(d["kl_mb"]) for d in losses]
            agg["kl_rolls"] = int(sum(d["kl_roll"] for d in losses))
            agg["lr_now"] = float(self._lr_now)
        if ksg_on:
            # D048i B4：轮末聚合。全部新键独立命名，不与 kl_guard 四键
            # （kl_mb/kl_mb_all/kl_rolls/lr_now）或既有键重叠；analytic 三项
            # = 接受步的达成值（零接受步的轮记 0.0）；param_rel_move =
            # ‖θ_end−θ_start‖₂/‖θ_start‖₂（θ_start = B2 时刻快照）
            agg["qual_logp_maxdev"] = qual_logp_maxdev
            agg["qual_ratio_maxdev"] = qual_ratio_maxdev
            agg["qual_kl_self"] = qual_kl_self
            agg["kl_steps_total"] = int(ksg_total)
            agg["kl_steps_accepted"] = int(ksg_acc)
            agg["kl_steps_rejected"] = int(ksg_rej)
            agg["kl_lr_scale_min"] = (
                float(min(ksg_scales)) if ksg_scales else 1.0
            )
            agg["kl_lr_scale_mean"] = (
                float(np.mean(ksg_scales)) if ksg_scales else 1.0
            )
            agg["kl_analytic_joint_mean"] = (
                float(np.mean(ksg_kl_joint)) if ksg_kl_joint else 0.0
            )
            agg["kl_analytic_joint_max"] = (
                float(np.max(ksg_kl_joint)) if ksg_kl_joint else 0.0
            )
            agg["kl_analytic_z_mean"] = (
                float(np.mean(ksg_kl_z)) if ksg_kl_z else 0.0
            )
            agg["kl_analytic_res_mean"] = (
                float(np.mean(ksg_kl_res)) if ksg_kl_res else 0.0
            )
            agg["kl_gate_mean"] = (
                float(np.mean(ksg_kl_gate)) if ksg_kl_gate else 0.0
            )
            agg["kl_vb_mean"] = (
                float(np.mean(ksg_kl_vb)) if ksg_kl_vb else 0.0
            )
            with torch.no_grad():
                _num = torch.sqrt(sum(
                    (p - s).pow(2).sum()
                    for p, s in zip(_opt_params, theta_start)
                ))
                _den = torch.sqrt(sum(s.pow(2).sum() for s in theta_start))
                agg["param_rel_move"] = float((_num / (_den + 1e-12)).item())
        agg["expl_var"] = expl_var
        # D051：伴生插桩落账（vb 头关闭时为 None）
        agg["adv_vb_corr"] = adv_vb_corr
        # E49: 整轮更新结束后的统一 KL 测量——no_grad 下用更新后的 policy 对
        # 整批 rollout obs 重算联合 logp（与 minibatch 的 logp 同口径，含
        # aux_executed / decoder_ft 分支），k3 估计与 approx_kl 同族，但样本
        # = 整批 T*N、时机 = 全部 epoch 完成后（minibatch approx_kl 只反映
        # 更新中途的洗牌子集）。
        with torch.no_grad():
            p_all = self.policy.forward_actor(obs)
            if getattr(self.policy, "decoder_ft", False):
                aux_mean_all, _ = self.policy.action_mean(phase, obs)
                ad_all = Normal(aux_mean_all, p_all["aux_log_std"].exp())
                lp_all = ad_all.log_prob(aux).sum(-1)
            else:
                ad_all = Normal(p_all["aux_mean"], p_all["aux_log_std"].exp())
                lp_all = ad_all.log_prob(aux).sum(-1) if aux_scored else 0.0
            if phase is not None:
                pd_all = Normal(p_all["phase_mean"], p_all["phase_log_std"].exp())
                lp_all = lp_all + pd_all.log_prob(phase).sum(-1)
            if gate is not None:
                gd_all = Categorical(logits=p_all["gate_logits"])
                lp_all = lp_all + gd_all.log_prob(gate)
            if vb is not None:
                vbd_all = Normal(p_all["vb_logits"], p_all["vb_log_std"].exp())
                lp_all = lp_all + vbd_all.log_prob(vb).sum(-1)
            logratio_all = lp_all - logp_old
            agg["post_update_kl"] = float(
                (logratio_all.exp() - 1.0 - logratio_all).mean().item()
            )
        return agg
