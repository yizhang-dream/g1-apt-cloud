"""D065 C 臂 v2：decoder 迁进策略本体（in-graph）的 LoRA 策略。

**为什么要 in-graph**：v1 把 LoRA decoder 挂在 env 侧 ActionTerm，PPO 的 log_prob/熵只依赖
policy 自身输出 ⇒ 目标对 adapter 参数梯度恒为 0/None（v1 已实测确认：全冻结 policy 时
`loss.requires_grad=False`、`loss.backward()` 直接 RuntimeError）。ORCS/ViBe 的等价实现把
decoder 放进策略本体（`rsl_rl/models/sonic_base_model.py:316-340`：分布头在 decoder 之后），
梯度经 `log N(a; μ, σ)` 直达 decoder（E44 架构同理，见 refine-logs/tracker/E.md:159）。
本模块即该形态：**μ 路径 = token head → E49/D055 仿射 → 组装 994 → SonicLoRADecoder**。

架构（与 A/B 臂的对照关系）：
    A 臂：obs → actor MLP → μ(29)                     直出关节目标，无 decoder
    B 臂：obs → actor MLP → a(64) → [env 侧] token 仿射 → 994 → 冻结 decoder → q_des
    C 臂：obs → token head MLP → a(64) → **本模块内** token 仿射 → 994 → LoRA decoder → μ(29)
          （动作空间 29 维 = A 臂；decoder 在路径上且其 adapter 可适配 = 相对 A 的单变量）
    C vs E44 = 低秩（rank-16 零初始化 adapter）vs 全权重微调。

关键契约：
1. **动作分布 N(μ, σ)，σ 冻结**：`self.std` 是 `requires_grad=False` 的 Parameter（默认值 =
   init_noise_std，与 A/B 同起值），train 侧 `_prune_frozen_params` 把它从优化器 param_groups
   里摘掉（"σ 不进优化器"的机制）；log_prob/熵对 **29 维**（= 动作维，A 臂同口径）。
2. **token head 输入 = 官方 policy obs 切片**（`obs[:, :official_obs_dim]`）：因 env 侧
   policy 组末位追加了 930 维 `sonic_proprio`（decoder 的输入契约），head 的输入维与 A/B 臂
   actor 逐字相同（`num_actor_obs` 传 official_obs_dim 给基类）。
3. **994 组装逐字复刻 ActionTerm**：proprio 段不是本模块自己算的，而是 env 的
   `sonic_proprio_hist` 用 **`SonicActionCore` 同一份代码**（历史环形缓冲 + assemble）产出的
   930 维（raw 无噪声）；本模块只做 `cat([token, proprio])` 与 `decode_obs`（= 与 B 臂
   `wrap_sonic_torch_decoder` 同一运算序列）。等价断言见 `assert_matches_action_term`。
   该 obs 项须**构造期可调用**（obs manager 构造时会先调一次填维度，此时 env 的
   `episode_length_buf` 尚未创建）——无 buf 时跳过复位重填，见 `sonic_proprio_hist` docstring。
4. **两档可训集合**（`trainable`）：
   - `policy+adapter`（默认，from-scratch 平行对照）：head/critic/adapter 全可训，σ 冻结；
   - `adapter-only`（ORCS 忠实形态）：head 冻结（须 `--init-head-from` 提供合理初始化，
     否则随机冻结 head = 废臂），critic 与 adapter 仍可训（否则 PPO 的 loss 无梯度路径）。
5. **FSQ 走 STE**：policy 侧的 `decode_obs(..., ste=True)` 用 straight-through 写 round
   （前向与裸 round **逐位相同**，反向保留量化器上游梯度）。理由：裸 round 导数恒 0，会把
   token head 的梯度整体抹零（本机实测 head grad 全 0）——in-graph 训练里 head 就永远不学；
   ORCS 的 `fsq_quantize`（rsl_rl/models/sonic_base_model.py:90）同样是 STE 写法。
   adapter 在量化器**下游**，有无 STE 都能拿到梯度。

来源与许可：LoRA 数学与 decoder 装配见 sonic_decoder_lora.py（ORCS rsl_rl fork 的数学照抄，
BSD-3-Clause，ETH Zurich / NVIDIA；本模块不复制其文件，只照结构与调用形态）。
本模块**纯 torch 部分**（SonicLoRAMuPath）零 isaaclab/rsl_rl 依赖，本机 CPU 可单测；
`SonicLoRAPolicy` 的基类是 rsl_rl 的 ActorCritic（无 rsl_rl 时降级为 nn.Module，仅保证
本机可 import；实例化会显式报错）；`sonic_proprio_hist` 的 isaaclab import 在函数体内。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .sonic_decoder_lora import (  # 同目录 sibling（本模块只依赖纯 torch 的 LoRA 模块）
    DEFAULT_ALPHA,
    DEFAULT_RANK,
    DECODER_LAYERS,
    SonicLoRADecoder,
)

__all__ = [
    "DECODER_OBS_DIM",
    "PROPRIO_DIM",
    "TOKEN_DIM",
    "SonicLoRAMuPath",
    "SonicLoRAPolicy",
    "assert_matches_action_term",
    "build_decoder_from_onnx",
    "load_token_stats",
    "register_policy_class_in_rsl_rl",
    "sonic_proprio_hist",
]

TOKEN_DIM = 64
NUM_JOINTS = 29
HISTORY_LEN = 10
PROPRIO_DIM = HISTORY_LEN * 3 + 3 * HISTORY_LEN * NUM_JOINTS + HISTORY_LEN * 3  # 930
DECODER_OBS_DIM = TOKEN_DIM + PROPRIO_DIM  # 994
TOKEN_STD_FLOOR = 1e-3  # = SonicDecoderActionTermCfg.token_std_floor（token 仿射的 std 下限）
VALID_TOKEN_BOUNDS = ("none", "tanh")
POLICY_CLASS_NAME = "SonicLoRAPolicy"

# rsl_rl 基类（classic API ActorCritic）；本机无 rsl_rl 时降级为 nn.Module 以便 import 期可用
try:  # pragma: no cover - 服务器有 rsl_rl，本机走 except
    from rsl_rl.modules import ActorCritic as _ActorCriticBase

    HAS_RSL_RL = True
except ImportError:  # pragma: no cover - 本机（无训练 venv）
    _ActorCriticBase = nn.Module
    HAS_RSL_RL = False


# ---------------------------------------------------------------------------
# 常量加载 / decoder 构造（train 与 eval 共用，避免双份漂移）
# ---------------------------------------------------------------------------
def load_token_stats(path: str, device="cpu", std_floor: float = TOKEN_STD_FLOOR):
    """读官方 g1-mode token 统计 npz（mean/std 键），std 下限截断。

    与 `SonicDecoderActionTerm.__init__`（sonic_action_term.py:438-453）逐式同口径：
    `std = max(std, token_std_floor)`；维度必须是 (64,)。
    """
    stats = np.load(path)
    mean = torch.as_tensor(np.asarray(stats["mean"], dtype=np.float32), device=device)
    std = torch.as_tensor(
        np.maximum(np.asarray(stats["std"], dtype=np.float32), float(std_floor)), device=device
    )
    if mean.shape != (TOKEN_DIM,) or std.shape != (TOKEN_DIM,):
        raise ValueError(f"token_stats npz 的 mean/std 须为 ({TOKEN_DIM},)，收到 {tuple(mean.shape)}/{tuple(std.shape)}")
    return mean, std


def build_decoder_from_onnx(onnx_path: str, rank: int = DEFAULT_RANK, alpha: float = DEFAULT_ALPHA, device="cpu"):
    """从 ONNX 建 `SonicLoRADecoder`（7 层 + rank-16 零初始化 LoRA，构造时自检 C==B）。"""
    return SonicLoRADecoder.from_onnx(onnx_path, rank=int(rank), alpha=float(alpha), device=str(device))


# ---------------------------------------------------------------------------
# μ 路径（纯 torch：仿射 + 组装 + LoRA decoder）
# ---------------------------------------------------------------------------
class SonicLoRAMuPath(nn.Module):
    """token head 输出 → 仿射 token → 组装 994 → LoRA decoder → μ(29)。纯 torch，无 isaaclab。

    - `token_from_raw`：E49/D055 仿射 `token = mean + alpha * std * a`（token_bound="tanh" 时
      对 a 取 tanh），逐式照 `SonicActionCore.map_token`（sonic_action_term.py:205-215）。
    - `assemble_obs`：`cat([token, proprio])`，proprio 由 env 观测项给出（930 维，顺序 =
      sonic_action_term.py:24-34 的通道契约）。
    - `forward`：`decoder.decode_obs(assemble_obs(...))`——与 B 臂 ActionTerm 的
      `wrap_sonic_torch_decoder`（先切 token 片量化再 forward）**同一运算序列**。
    """

    def __init__(
        self,
        official_obs_dim: int,
        decoder: SonicLoRADecoder,
        token_mean: torch.Tensor,
        token_std: torch.Tensor,
        token_alpha: float = 1.0,
        token_bound: str = "tanh",
        proprio_dim: int = PROPRIO_DIM,
    ) -> None:
        super().__init__()
        if token_bound not in VALID_TOKEN_BOUNDS:
            raise ValueError(f"token_bound 须 ∈ {VALID_TOKEN_BOUNDS}，收到 {token_bound!r}")
        self.official_obs_dim = int(official_obs_dim)
        self.proprio_dim = int(proprio_dim)
        if self.official_obs_dim + self.proprio_dim <= 0:
            raise ValueError("official_obs_dim/proprio_dim 非法")
        self.token_alpha = float(token_alpha)
        self.token_bound = str(token_bound)
        # 常量 buffer：随 .to(device) 迁移，不进 state_dict（身份由信封记录）
        self.register_buffer("token_mean", torch.as_tensor(token_mean, dtype=torch.float32).reshape(TOKEN_DIM),
                             persistent=False)
        self.register_buffer("token_std", torch.as_tensor(token_std, dtype=torch.float32).reshape(TOKEN_DIM),
                             persistent=False)
        self.decoder = decoder

    # ---- token 仿射 / 组装 ----
    def token_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        """(N, 64) 策略原始输出 -> FSQ 前 token 坐标（逐式同 SonicActionCore.map_token）。"""
        if self.token_bound == "tanh":
            return self.token_mean + self.token_alpha * self.token_std * torch.tanh(raw)
        return self.token_mean + self.token_alpha * self.token_std * raw

    def split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """把 policy 组观测切成 (head 输入, decoder 本体 930)。"""
        if obs.dim() != 2 or int(obs.shape[-1]) < self.official_obs_dim + self.proprio_dim:
            raise ValueError(
                f"policy obs 须为 (N, >={self.official_obs_dim + self.proprio_dim})，收到 {tuple(obs.shape)}"
            )
        return obs[:, : self.official_obs_dim], obs[:, self.official_obs_dim: self.official_obs_dim + self.proprio_dim]

    def assemble_obs(self, head_out: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """(N,64) head 输出 + (N,·) policy obs -> (N,994) decoder 输入（token 未量化）。"""
        token = self.token_from_raw(head_out)
        _official, proprio = self.split_obs(obs)
        if int(token.shape[-1]) != TOKEN_DIM:
            raise ValueError(f"head 输出维须为 {TOKEN_DIM}，收到 {tuple(token.shape)}")
        return torch.cat([token, proprio], dim=-1)

    def forward(self, head_out: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """-> μ (N, 29)（归一化关节目标，SONIC/IsaacLab 序）。

        量化走 **STE**（`decode_obs(..., ste=True)`）：前向与 B 臂解码路径逐位相同，
        但反向保留「量化器上游」的梯度 —— 否则裸 round 的零导数会把 token head 的
        梯度整体抹零（实测：head grad 全 0），in-graph 训练就只剩 adapter 在学。
        """
        return self.decoder.decode_obs(self.assemble_obs(head_out, obs), ste=True)

    # ---- 诊断 / 身份（委托 decoder）----
    @property
    def lora_scale(self) -> float:
        return float(self.decoder.lora_scale)

    def set_lora_scale(self, scale: float) -> None:
        self.decoder.set_lora_scale(scale)

    def lora_scale_zero_(self) -> None:
        self.decoder.lora_scale_zero_()

    def lora_scale_restore_(self) -> None:
        self.decoder.lora_scale_restore_()

    def adapter_params(self) -> list[nn.Parameter]:
        return self.decoder.adapter_params()

    def base_parameters(self) -> list[nn.Parameter]:
        return self.decoder.base_parameters()

    def adapter_param_report(self) -> dict:
        return self.decoder.adapter_param_report()

    def adapter_delta_stats(self, envelope: float | None = None) -> dict:
        return self.decoder.adapter_delta_stats(envelope)

    def base_weight_md5(self) -> str:
        return self.decoder.base_weight_md5()

    def base_state_snapshot(self) -> dict:
        return self.decoder.base_state_snapshot()

    def base_state_maxdiff(self, snapshot: dict) -> float:
        return self.decoder.base_state_maxdiff(snapshot)

    def identity_envelope(self, extra: dict | None = None) -> dict:
        return self.decoder.identity_envelope(extra)

    def verify_lora_scale_zero(self) -> dict:
        return self.decoder.verify_lora_scale_zero()

    @property
    def construction_identity(self) -> dict:
        return self.decoder.construction_identity


# ---------------------------------------------------------------------------
# rsl_rl 策略（ActorCritic 变体）
# ---------------------------------------------------------------------------
class SonicLoRAPolicy(_ActorCriticBase):  # type: ignore[misc]
    """rsl_rl ActorCritic 变体：actor 干 = token head，μ 经 LoRA decoder，σ 冻结。

    构造参数（经 runner cfg 的 `policy` dict 透传，见 train/eval 侧 `policy_kwargs`）：
      - `num_actor_obs`（rsl_rl 传 policy 组总维 = official + 930）/`num_critic_obs`/`num_actions`（29）
      - `official_obs_dim`：head 输入维（= 官方 policy 八项之和；缺省按 num_actor_obs - proprio_dim 推）
      - `proprio_dim`：decoder 本体历史维（930）
      - `decoder`：`SonicLoRADecoder` 实例（**已挂好 adapter**，由调用方构造并进身份信封）
      - `token_mean/token_std/token_alpha/token_bound`：token 仿射常量（同 ActionTerm）
      - `sigma`：冻结的 σ 值（缺省 = init_noise_std，与 A/B 同起值）；`sigma_trainable` 可放开
      - `trainable`：`policy+adapter`（默认）/`adapter-only`（冻结 head）
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int = NUM_JOINTS,
        *,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        sigma: float | None = None,
        sigma_trainable: bool = False,
        official_obs_dim: int | None = None,
        proprio_dim: int = PROPRIO_DIM,
        decoder: SonicLoRADecoder | None = None,
        token_mean=None,
        token_std=None,
        token_alpha: float = 1.0,
        token_bound: str = "tanh",
        trainable: str = "policy+adapter",
        **kwargs,
    ) -> None:
        if not HAS_RSL_RL:
            raise RuntimeError(
                "SonicLoRAPolicy 需要 rsl_rl（rsl-rl-lib 2.3.3 经典 API ActorCritic）；"
                "本机（Windows）无训练 venv，请在服务器 .venv_isaac 内实例化。"
            )
        if decoder is None:
            raise ValueError("SonicLoRAPolicy 需要已构造的 LoRA decoder（见 build_decoder_from_onnx）")
        if token_mean is None or token_std is None:
            raise ValueError("SonicLoRAPolicy 需要 token_mean/token_std（见 load_token_stats）")
        official = int(official_obs_dim) if official_obs_dim is not None else int(num_actor_obs) - int(proprio_dim)
        if official <= 0:
            raise ValueError(f"official_obs_dim 推得 {official}（num_actor_obs={num_actor_obs}, proprio_dim={proprio_dim}）")
        # 基类：actor 干按 official 输入、64 输出（= B 臂 actor 结构逐字）；critic 同 A/B
        super().__init__(
            num_actor_obs=official,
            num_critic_obs=int(num_critic_obs),
            num_actions=TOKEN_DIM,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=float(init_noise_std),
        )
        self.official_obs_dim = official
        self.proprio_dim = int(proprio_dim)
        self.num_actions = int(num_actions)  # 动作维 = μ 维（29），与 A 臂同口径
        self.trainable = str(trainable)
        self.mu_path = SonicLoRAMuPath(
            official_obs_dim=official,
            decoder=decoder,
            token_mean=token_mean,
            token_std=token_std,
            token_alpha=token_alpha,
            token_bound=token_bound,
            proprio_dim=self.proprio_dim,
        )
        # σ：重建为 num_actions 维常量（冻结 ⇒ 不进优化器；见 train 侧 _prune_frozen_params）
        sigma0 = float(init_noise_std) if sigma is None else float(sigma)
        self.std = nn.Parameter(
            torch.full((self.num_actions,), sigma0), requires_grad=bool(sigma_trainable)
        )
        self.sigma_frozen = not bool(sigma_trainable)
        if self.trainable == "adapter-only":
            for p in self.actor.parameters():
                p.requires_grad_(False)

    # ---- μ / 分布 ----
    @staticmethod
    def _flat_obs(obs):
        """兼容 dict obs（IsaacLab wrapper 在有 critic 组时返回 {'policy','critic'}）。"""
        if isinstance(obs, dict):
            return obs["policy"]
        return obs

    def mu(self, obs) -> torch.Tensor:
        """μ(29) = LoRA decoder(token head(官方 obs 切片), 本体 930)。"""
        flat = self._flat_obs(obs)
        head_out = self.actor(flat[:, : self.official_obs_dim])
        return self.mu_path(head_out, flat)

    def update_distribution(self, observations) -> None:
        """N(μ, σ)（σ 冻结的常量向量，广播到 (N, 29)）。"""
        mu = self.mu(observations)
        std = self.std.to(mu.device).expand_as(mu)
        self.distribution = torch.distributions.Normal(mu, std)

    def act(self, observations, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations) -> torch.Tensor:
        """det 推理 = μ（29 维关节目标；注意与基类不同——基类返回 head 的 64 维输出）。"""
        return self.mu(observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """log N(a; μ, σ) 对 29 维求和（= E44 里梯度直达 decoder 的那条路径）。"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    # ---- 初始化 / 诊断 ----
    def load_head_from_ckpt(self, ckpt_path: str, torch_mod=torch) -> dict:
        """从 A/B 臂 ckpt 灌 token head（actor.* 键）；形状不符的键跳过并报告。

        - B 臂 ckpt：actor 首层输入 321（其 obs 里 actions 项 = 64 维 raw token）≠ token head
          的 286（actions 项 = 29 维关节目标）⇒ **首层**形状不符被跳过，其余键完整加载；
        - A 臂 ckpt：actor 首层 286 与 head 相同（末层 128→29 ≠ head 的 128→64）⇒
          **末层**被跳过（head 部分加载，须读报告确认）；
        - author v0 / 其他：按前缀 actor.* 尽力匹配。

        （原注释「B 臂全键形状相符」系笔误：B 臂 obs=321 与 head 输入 286 不同维，
        首层必跳过；跳过末层的是 A 臂。SF-4。）
        """
        payload = torch_mod.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        head_sd = {k[len("actor."):]: v for k, v in sd.items() if isinstance(k, str) and k.startswith("actor.")}
        if not head_sd:
            raise RuntimeError(f"ckpt 无 actor.* 键：{ckpt_path}（keys 示例 {list(sd)[:5]}）")
        own = self.actor.state_dict()
        matched = {k: v for k, v in head_sd.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
        skipped = sorted(set(head_sd) - set(matched))
        if not matched:
            raise RuntimeError(
                f"--init-head-from 与 token head 无任何同形状键（ckpt actor 键 {sorted(head_sd)[:6]}）"
            )
        self.actor.load_state_dict(matched, strict=False)
        return {
            "ckpt": str(ckpt_path),
            "n_loaded": len(matched),
            "loaded": sorted(matched),
            "skipped": skipped,
            "note": "形状不符的键被跳过（A 臂末层 29≠64；B 臂首层输入 321≠286）；"
                    "须读本报告确认 head 是否完整（adapter-only 档下 train 侧对非空 skipped 直接报错）",
        }

    def trainable_report(self) -> dict:
        """可训集合清单（进日志/身份信封；σ 冻结与 adapter-only 的 head 冻结在此可检查）。"""
        def _count(mod):
            ps = list(mod.parameters())
            return {"n_tensors": len(ps), "n_trainable": sum(1 for p in ps if p.requires_grad),
                    "n_elements": int(sum(p.numel() for p in ps)),
                    "n_trainable_elements": int(sum(p.numel() for p in ps if p.requires_grad))}

        return {
            "trainable_mode": self.trainable,
            "sigma": {"value": float(self.std.detach().reshape(-1)[0].item()), "frozen": bool(self.sigma_frozen),
                      "requires_grad": bool(self.std.requires_grad)},
            "head(actor)": _count(self.actor),
            "critic": _count(self.critic),
            "decoder_base": _count(self.mu_path.decoder.base_linears),
            "adapter": _count(nn.ModuleList([a for a in self.mu_path.decoder.adapters if a is not None])),
        }


def register_policy_class_in_rsl_rl(cls=SonicLoRAPolicy, name: str = POLICY_CLASS_NAME) -> str:
    """把策略类注入 rsl_rl 的类名解析命名空间，返回应写进 `policy.class_name` 的名字。

    rsl_rl 经典 API 的 runner 用 `eval(cfg["policy"]["class_name"])` 在其模块命名空间里解析类名，
    故把类挂到 `rsl_rl.runners.on_policy_runner`（首选）与 `rsl_rl.runners`/`rsl_rl.modules`
    （兜底）即可让 `class_name = "SonicLoRAPolicy"` 生效；**不覆盖** ActorCritic 原名，
    避免影响其他路径。调用方随后必须断言 runner 里的 policy 实例类型（防静默回落）。
    """
    targets = []
    try:  # pragma: no cover - 服务器有 rsl_rl
        import rsl_rl.runners.on_policy_runner as m1

        targets.append(m1)
    except ImportError:
        pass
    try:  # pragma: no cover
        import rsl_rl.runners as m2

        targets.append(m2)
    except ImportError:
        pass
    try:  # pragma: no cover
        import rsl_rl.modules as m3

        targets.append(m3)
    except ImportError:
        pass
    if not targets:
        raise RuntimeError("rsl_rl 不可导入——无法注册策略类（请在 isaaclab venv 内运行）")
    for mod in targets:
        setattr(mod, name, cls)
    return name


# ---------------------------------------------------------------------------
# env 侧观测项：decoder 的本体历史（930 维，raw 无噪声）
# ---------------------------------------------------------------------------
def sonic_proprio_hist(env, action_term_name: str = "joint_pos", asset_name: str = "robot") -> torch.Tensor:
    """policy 组末位的 decoder 本体观测项：(N, 930)，顺序/归一化 = B 臂 ActionTerm 逐字。

    实现要点（等价性论证见模块 docstring 与 train 侧 `assert_matches_action_term`）：
    1. 用 **`SonicActionCore` 同一份代码**（`push_state` + `assemble_obs`）维护 10 帧环形历史，
       core 挂在 `env._sonic_lora_core` 上（env 侧无 decoder action term，历史由本项负责）；
    2. 每控制步推入一帧：ang_vel=root_ang_vel_b、joint_pos=q-default、joint_vel=qvel、
       last_act=(q_des-default)/scale、gravity=projected_gravity_b（与
       `SonicDecoderActionTerm.process_actions` 的五通道逐字相同）；
    3. **复位处理**：`episode_length_buf == 0` 的 env（含 reset 时刻全体）先 `core.reset`
       重填站立帧——等价于 B 臂 ActionManager 调 `term.reset(env_ids)` 的时机；**并把
       这些行的 `last_act` 置 0**（SF-1：`processed_actions` 是上一局末残留，站立帧约定
       该通道为 0，同 B 臂 reset 帧语义）。
       **构造期容忍**：`ObservationManager._prepare_terms`（observation_manager.py:420）会在
       env 构造早期无条件调用本项一次填维度，此刻 `ManagerBasedRLEnv.episode_length_buf` 尚未
       创建（manager_based_rl_env.py:85）⇒ 无 buf 可查时跳过重填、只返回当前缓冲组装结果；
       该次脏帧无害（`env.reset()` 时 buf 全 0 ⇒ 全体走 `core.reset` 全量重填）；
    4. 时序：本项在 obs 计算时推入「本控制步末」状态，与 B 臂 ActionTerm 在下一控制步
       `process_actions` 推入的是**同一帧**（两步之间无物理推进）⇒ decoder 输入时序一致。
    5. 返回值 = `assemble_obs(占位 token)[:, 64:]`（占位 token 段被丢弃，避免重复实现拼接）。
    """
    try:  # 服务器执行根平铺 import / 仓库根包 import 双兼容
        from isaac.sonic_action_term import (
            SonicActionCore,
            _sonic_default_isaac,
            _sonic_scale_isaac,
            resolve_sonic_joint_ids,
        )
    except ImportError:  # pragma: no cover
        from apt_g1.isaac.sonic_action_term import (
            SonicActionCore,
            _sonic_default_isaac,
            _sonic_scale_isaac,
            resolve_sonic_joint_ids,
        )

    asset = env.scene[asset_name]
    core = getattr(env, "_sonic_lora_core", None)
    if core is None:
        core = SonicActionCore(
            num_envs=env.num_envs,
            device=env.device,
            # token 仿射常量本项不使用（decoder 在 policy 侧）；占位值仅为满足构造契约
            token_mean=np.zeros(TOKEN_DIM, dtype=np.float32),
            token_std=np.ones(TOKEN_DIM, dtype=np.float32),
            sonic_default=_sonic_default_isaac(),
            sonic_scale=_sonic_scale_isaac(),
            decoder=None,
        )
        ids, _identity = resolve_sonic_joint_ids(asset.joint_names)
        env._sonic_lora_core = core
        env._sonic_lora_joint_ids = torch.as_tensor(ids, dtype=torch.long, device=env.device)
    ids_t = env._sonic_lora_joint_ids

    # 复位 env 的站立帧重填（等价 ActionManager -> term.reset 的时机）。
    # 构造期容忍：obs 项在 env 构造期被调用时 episode_length_buf 还不存在（见 docstring 第 3 条），
    # 用 getattr 探测；为 None 则跳过重填（其后还有 env.reset() 的全量重填兜底）。
    ep = getattr(env, "episode_length_buf", None)
    reset_ids = None
    if ep is not None:
        reset_ids = (ep == 0).nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            core.reset(reset_ids)

    ang_vel = asset.data.root_ang_vel_b
    gravity = asset.data.projected_gravity_b
    joint_pos_rel = asset.data.joint_pos[:, ids_t] - core.sonic_default
    joint_vel = asset.data.joint_vel[:, ids_t]
    q_des_asset = env.action_manager.get_term(action_term_name).processed_actions
    last_act = core.last_act_from_q_des(q_des_asset[:, ids_t])  # 资产序 -> SONIC 序
    if reset_ids is not None and reset_ids.numel() > 0:
        # SF-1（复位边界混合帧）：core.reset 只重填历史与 q_des，**不清** action term 的
        # `processed_actions`（官方 JointAction.reset 同样不清）；若直接取它算 last_act，
        # 复位后首帧 = 站立状态 + 上一局末目标残留，与 B 臂该帧 last_act=0（站立帧约定，
        # sonic_action_term.py:297-309 的 reset 语义）不一致。只对复位行置零。
        # （last_act_from_q_des 返回新张量，就地改安全）
        last_act[reset_ids] = 0.0
    core.push_state(ang_vel, joint_pos_rel, joint_vel, last_act, gravity)
    zeros = torch.zeros(env.num_envs, TOKEN_DIM, device=env.device)
    return core.assemble_obs(zeros)[:, TOKEN_DIM:]


# ---------------------------------------------------------------------------
# 等价断言（train 侧运行期调用；selftest 用替身调用）
# ---------------------------------------------------------------------------
def assert_matches_action_term(mu_path: SonicLoRAMuPath, core, torch_mod=torch, batch: int = 4, seed: int = 0) -> dict:
    """断言 μ 路径的「token 仿射 + 994 组装」与 ActionTerm 的 `SonicActionCore` 逐位一致。

    做法：把同一份 (a, 五通道历史) 喂给两条路径，比较 (N, 994)：
      A) ActionTerm 路径：`core.map_token(a)` + `core.assemble_obs(...)`（core 的历史缓冲
         先被写成同一份 hist；token 常量须与 mu_path 相同——由调用方按同参构造 core）；
      B) μ 路径：`mu_path.assemble_obs(head_out=a, obs)`（obs 的 proprio 段 = 同一份 hist）。
    期望 `maxdiff == 0`（两条路径都只是 `mean + alpha*std*tanh(a)` 与 `cat([token, proprio])`）。
    """
    device = next(mu_path.parameters()).device
    g = torch_mod.Generator(device="cpu").manual_seed(int(seed))
    n = int(batch)
    raw = torch_mod.randn(n, TOKEN_DIM, generator=g).to(device)
    hist = (torch_mod.randn(n, PROPRIO_DIM, generator=g) * 0.1).to(device)
    # A) ActionTerm 路径：把 core 的 10 帧缓冲写成 hist，再走 map_token + assemble_obs
    with torch_mod.no_grad():
        core.hist_ang_vel.copy_(hist[:, 0:30].reshape(n, HISTORY_LEN, 3))
        core.hist_joint_pos.copy_(hist[:, 30:320].reshape(n, HISTORY_LEN, NUM_JOINTS))
        core.hist_joint_vel.copy_(hist[:, 320:610].reshape(n, HISTORY_LEN, NUM_JOINTS))
        core.hist_last_act.copy_(hist[:, 610:900].reshape(n, HISTORY_LEN, NUM_JOINTS))
        core.hist_gravity.copy_(hist[:, 900:930].reshape(n, HISTORY_LEN, 3))
        obs_a = core.assemble_obs(core.map_token(raw))
    # B) μ 路径：head 输出直接喂 raw（等价 head_out = a）
    obs_b_in = torch_mod.cat([torch_mod.zeros(n, mu_path.official_obs_dim, device=device), hist], dim=1)
    obs_b = mu_path.assemble_obs(raw, obs_b_in)
    maxdiff = float((obs_b - obs_a).abs().max().item())
    res = {
        "maxdiff_assemble": maxdiff,
        "bitwise_assemble": bool(torch_mod.equal(obs_b, obs_a)),
        "batch": n,
        "shape": list(obs_b.shape),
        "note": "A=ActionTerm(SonicActionCore.map_token+assemble_obs) / B=μ 路径 assemble_obs",
    }
    if getattr(core, "decoder", None) is not None:  # ActionTerm 侧有 decoder 时顺带比 decode
        with torch_mod.no_grad():
            out_a = core.decoder(obs_a)
            out_b = mu_path.decoder.decode_obs(obs_b)
        res["maxdiff_decode"] = float((out_b - out_a).abs().max().item())
    return res