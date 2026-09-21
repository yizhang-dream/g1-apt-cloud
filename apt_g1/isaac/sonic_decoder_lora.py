"""D065 C 臂模块：冻结 SONIC decoder 的 7 个 Linear 上挂 rank-16 零初始化 LoRA。

预注册=[refine-logs/ds/D065_DECODER_LORA_ARM.md](../../refine-logs/ds/D065_DECODER_LORA_ARM.md)
§3（C 臂规格：挂 decoder 全部 7 个 Linear、rank=16、alpha=1.0、down 随机初始化、
up 零初始化 ⇒ 构造时与 B 臂逐比特一致、只训 adapter、无感知条件化/无 augmentation 流）。

数学与结构逐式对照外部参考实现（本机克隆 tmp/orcs_rsl_rl，rsl_rl fork，BSD-3-Clause，
ETH Zurich / NVIDIA；本模块只抄数学，不抄文件）：

    Adapter.__init__        rsl_rl/modules/mlp_with_adapter.py:38-52
        low_rank = 0 < rank < min(in, out)；低秩=down(in→rank)+up(rank→out) 均无 bias；
        down 用 N(0, 1/sqrt(rank)) 初始化（:46）、up 零初始化（:47）、scale = alpha/rank（:48）；
        退化分支（rank<=0 或 rank>=min(in,out)）= 单个稠密 delta Linear、零初始化、
        scale = alpha（:50-52）——本臂 rank=16 远小于各层 min(in,out)（最小者末层 512→29
        的 29），按预注册 §3 不触发该分支；本模块仍实现它并在身份信封里逐层标注。
    Adapter.forward         rsl_rl/modules/mlp_with_adapter.py:54-58
        低秩 = scale * up(down(x))；退化 = scale * delta(x)。
    Adapter.delta_weight    rsl_rl/modules/mlp_with_adapter.py:60-67
        低秩 = scale * (up.weight @ down.weight)。
    MLPWithAdapter.forward  rsl_rl/modules/mlp_with_adapter.py:187-205
        adapter[i] 与 base[i] 吃**同一份运行激活**、相加后才在下一次迭代过激活：
            h = base[0](obs);            h = h + ad[0](adapter_obs)
            for i in 1..n-1:  h = act(h); out = base[i](h); out = out + ad[i](h); h = out
        本臂 adapter_obs = obs（无 augmentation 流，预注册 §3「无感知条件化」）。
    SonicWithAdapterModel   rsl_rl/models/sonic_adapter_model.py:29-120
        adapt_encoder=False / adapt_decoder=True —— encoder 与 FSQ 一律不动（:8-11 的理由：
        量化器上游的权重 delta 会被 round 抹掉），本模块同款：只挂 decoder。

与 `sonic_decoder_torch.SonicTorchDecoder` 的关系：本模块**重建**同一结构（把 base 的
7 个 Linear 权重逐位拷进自己的 ModuleList，不共享模块对象），FSQ 常量与
quantize_tokens/build_decoder_obs 公式照抄 sonic_decoder_torch.py:59-91，
因而可作为 decoder 的 drop-in 替换（`decode_obs` 与 sonic_action_term.py:129-145 的
`wrap_sonic_torch_decoder` 数值等价，构造时自检见 `decode_obs_equivalence`）。

对外接口（供 train/eval 接线）：
    SonicLoRADecoder(decoder, rank=16, alpha=1.0, device=...)
        —— 包住已构造的 SonicTorchDecoder（或任何 net=Sequential 的 decoder）
    SonicLoRADecoder.from_onnx(onnx_path, ...)     —— 独立从 ONNX 建（评测/导出用）
    adapter_params() / base_parameters()           —— 参数集合（进优化器用）
    adapter_state_dict() / load_adapter_state_dict() —— adapter 权重存取（ckpt 侧车）
    set_lora_scale() / lora_scale_zero_()          —— 遗忘检查②（置零 ⇒ 逐位回退到 B）
    merge_adapters()                               —— 低秩合并进 base 权重（导出用）
    base_weight_md5() / base_state_snapshot() / base_state_maxdiff()
                                                   —— 身份信封 + 训练后冻结断言
    identity_envelope()                            —— onnx md5 + rank/alpha + 层表 + 自检结果
    install_into_action_term(term, lora)           —— 换进 SonicDecoderActionTerm（鸭子类型；
                                                       **v1 遗留接线，v2 in-graph 下无调用点**）

遗忘检查两条（预注册 §4）在本模块内各有构造性实现：
    ① 零初始化构造性：__init__ 末尾对随机探针做 `self.forward(x) == base_net(x)` 逐位比对，
       结果落 `self.construction_identity`（train 侧启动即断言 bitwise=True）；
    ② `lora_scale_zero_()` 后逐位一致：`verify_lora_scale_zero()` 用同一探针复验，
       eval 侧 `--lora-scale 0` 走这条路径并断言 maxdiff==0。
    两条都不是「靠浮点加法恰好为 0」——envelope==0.0 时 forward 走**跳过 adapter 的快速路径**，
    与 base 的运算序列逐条相同，故逐位一致是构造性保证而非经验结论。

==============================================================================
已知限制（必读；接线方按此判读，不要在结论里越界）
**（v1 形态遗留说明，v2 已改 in-graph）**：以下第 1 条描述的是 **v1** 接线（本模块挂在
env 侧 ActionTerm）的梯度事实；D065 C 臂 **v2** 已把 decoder 迁进策略本体
（`sonic_lora_policy.SonicLoRAPolicy`，见该模块 docstring），梯度经 `log N(a;μ,σ)` 直达
adapter，第 1 条**不再适用于 v2**；第 2/3 条（scale 不进 state_dict、merge 非逐位等价）
与形态无关，仍有效。
==============================================================================
1. **梯度路径（结构性，v1 形态）**：v1 把本模块挂在 env 侧 ActionTerm（`sonic_action_term.py` 的
   `SonicActionDecoder`），而 rsl_rl 的 PPO 只在 **policy 模块内部**求梯度
   （rsl_rl 2.3.3：`PPO.optimizer = Adam(policy.parameters(), ...)`；log_prob/熵只依赖
   policy 自身输出）。decoder 在策略**之后**的确定性映射里（token→q_des），
   `SonicActionCore.compute_q_des` 又在 `torch.no_grad()` 下运行
   （sonic_action_term.py:285-291），故 PPO 目标对 adapter 参数**梯度恒为 0/None**。
   ORCS 的对应实现把 decoder 放进 policy 模型本体
   （rsl_rl/models/sonic_base_model.py:316-340 forward，分布头在 decoder 之后），
   梯度才成立。⇒ 本臂要真正训练 adapter，必须把 decoder 迁进 policy（or 等价的可微路径）；
   否则 up 零初始化 + 零梯度 = adapter 恒为 0，C 臂逐位等于 B 臂。
   train 脚本的 `LORA_GRAD` 诊断行就是为把这条限制变成可 grep 的运行期事实而设。
2. **scale 不进 state_dict**：`scale=alpha/rank` 是 Python 浮点属性（与 ORCS 一致，
   见 D065 §2b），故 adapter 的**有效** delta 需要 alpha/rank 一起存——本模块的
   `identity_envelope()`/`adapter_state_dict()` 两者都存。
3. **merge_adapters 是数学等价而非逐位等价**：合并后 GEMM 的累加序不同，浮点上
   与分开前向有 ~1e-6 量级差异（selftest 用容差断言并打印实测 maxdiff）。
"""

from __future__ import annotations

import hashlib
import torch
import torch.nn as nn

__all__ = [
    "DECODER_LAYERS",
    "LoRAAdapter",
    "SonicLoRADecoder",
    "install_into_action_term",
]

# SONIC decoder 结构常量（sonic_decoder_torch.py:31-58 的 7 个 Linear / 994→...→29）
DECODER_LAYERS = 7
TOKEN_DIM = 64
HISTORY_LEN = 10
NUM_JOINTS = 29
DECODER_OBS_DIM = TOKEN_DIM + HISTORY_LEN * 3 + 3 * HISTORY_LEN * NUM_JOINTS + HISTORY_LEN * 3  # 994

# FSQ 常量默认值（= sonic_decoder_torch.py:59-62；decoder 对象上有同名属性时以其为准）
_FSQ_OFFSET = 0.032237
_FSQ_SCALE = 15.515501
_FSQ_HALF = 0.5
_FSQ_DENOM = 16.0

# 默认 rank / alpha（D065 §3：rank=16、alpha=1.0 ⇒ scale=alpha/rank=1/16）
DEFAULT_RANK = 16
DEFAULT_ALPHA = 1.0

# adapter 输出与 base 输出之比的诊断里防止除零
_RATIO_EPS = 1e-12


# ---------------------------------------------------------------------------
# Layer 1: LoRAAdapter —— ORCS Adapter 的逐式照抄
# ---------------------------------------------------------------------------
class LoRAAdapter(nn.Module):
    """单层零初始化 LoRA adapter（数学 = ORCS `Adapter`，mlp_with_adapter.py:38-67）。

    低秩分支（0 < rank < min(in, out)）：``delta(x) = scale * up(down(x))``，
    ``scale = alpha / rank``；退化分支（rank<=0 或 rank>=min(in,out)）：
    ``delta(x) = scale * delta(x)``，``scale = alpha``（LoRA 惯例的除 rank 不再发生）。
    两分支的输出侧因子都零初始化 ⇒ 构造时 ``delta == 0``，被适配层逐位等于 base。

    `envelope` 是**外层包络**（默认 1.0），只在 forward 里乘进 delta，不写进权重：
    `lora_scale_zero_()` 把整棵 decoder 的包络置 0（遗忘检查②），可无损还原；
    它不进 state_dict（Python 浮点属性），由身份信封记录。
    """

    def __init__(self, in_features: int, out_features: int, rank: int = DEFAULT_RANK,
                 alpha: float = DEFAULT_ALPHA) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        # 判据逐字照 ORCS mlp_with_adapter.py:42
        self.low_rank = 0 < self.rank < min(self.in_features, self.out_features)
        if self.low_rank:
            self.down = nn.Linear(self.in_features, self.rank, bias=False)
            self.up = nn.Linear(self.rank, self.out_features, bias=False)
            # 初始化与 scale 逐条照 ORCS mlp_with_adapter.py:46-48
            nn.init.normal_(self.down.weight, std=1.0 / self.rank ** 0.5)
            nn.init.zeros_(self.up.weight)
            self.scale = self.alpha / self.rank
        else:
            self.delta = nn.Linear(self.in_features, self.out_features, bias=False)
            nn.init.zeros_(self.delta.weight)
            self.scale = self.alpha

    def forward(self, x: torch.Tensor, envelope: float = 1.0) -> torch.Tensor:
        """delta 输出（ORCS mlp_with_adapter.py:54-58 + 外层包络）。"""
        if self.low_rank:
            return envelope * self.scale * self.up(self.down(x))
        return envelope * self.scale * self.delta(x)

    @torch.no_grad()
    def delta_weight(self, envelope: float = 1.0) -> torch.Tensor:
        """有效权重 delta（out, in）（ORCS mlp_with_adapter.py:60-67 + 包络）。"""
        if self.low_rank:
            return envelope * self.scale * (self.up.weight @ self.down.weight)
        return envelope * self.scale * self.delta.weight

    def zero_output_(self) -> None:
        """把本层输出侧因子清零（up / delta 权重 = 0）⇒ delta 恒 0（不可还原的硬置零）。"""
        with torch.no_grad():
            if self.low_rank:
                self.up.weight.zero_()
            else:
                self.delta.weight.zero_()

    def config(self) -> dict:
        """逐层配置（进身份信封；含是否退化分支）。"""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "rank": self.rank,
            "alpha": self.alpha,
            "scale": self.scale,
            "low_rank": self.low_rank,
            "degenerate_full_rank": not self.low_rank,
            "min_in_out": min(self.in_features, self.out_features),
            "params": [n for n, _ in self.named_parameters()],
        }


# ---------------------------------------------------------------------------
# Layer 2: SonicLoRADecoder —— base 冻结 + 7 层 adapter
# ---------------------------------------------------------------------------
class SonicLoRADecoder(nn.Module):
    """冻结 SONIC decoder + 每层零初始化 LoRA 的 drop-in 替换（D065 C 臂）。

    构造参数
    --------
    decoder : 已构造的 decoder 对象（须有 ``net`` = ``nn.Sequential``，如
        ``SonicTorchDecoder``）。其 7 个 Linear 的权重被**逐位拷贝**进本模块自己的
        ModuleList（不共享模块对象，避免双父注册/误改原对象）；FSQ 常量若源对象有
        同名属性则继承（`sonic_decoder_torch.py:59-62`）。
    onnx_path : 不给 ``decoder`` 时从 ONNX 建（延迟 import sonic_decoder_torch，
        本机无 onnx 也能 import 本模块）。
    rank / alpha : LoRA 超参（默认 16 / 1.0；`layer_ranks` 可逐层覆写，None=该层不挂）。
    expected_layers : 期望的 Linear 层数（默认 7，即「挂 decoder 全部 7 个 Linear」）；
        None = 不校验（本机小模型单测用）。
    verify_identity : 构造末尾对随机探针做「零初始化 ⇒ 与 base 逐位一致」自检。

    冻结纪律：``base_linears`` 全部 ``requires_grad_(False)``；``adapter_params()``
    只返回 adapter 参数（即「只有 decoder adapter 可训」的可检查化）。
    """

    def __init__(
        self,
        decoder=None,
        *,
        onnx_path: str | None = None,
        device: str | torch.device | None = None,
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        layer_ranks=None,
        expected_layers: int | None = DECODER_LAYERS,
        verify_identity: bool = True,
    ) -> None:
        super().__init__()
        if decoder is None:
            if not onnx_path:
                raise ValueError("SonicLoRADecoder 需要 decoder 或 onnx_path 之一")
            decoder = self._build_from_onnx(onnx_path, device)
        net = getattr(decoder, "net", None)
        if net is None or not isinstance(net, nn.Sequential):
            raise TypeError(
                "decoder 须提供 nn.Sequential 的 .net（SonicTorchDecoder 契约）；"
                f"收到 {type(decoder).__name__}"
            )
        # 末层必须是 Linear（与 ORCS MLPWithAdapter._set_base 同纪律，
        # mlp_with_adapter.py:152-156）：尾随激活会被本模块的前向循环丢掉。
        if not isinstance(net[-1], nn.Linear):
            raise ValueError(
                f"base net 末层须为 Linear（ORCS mlp_with_adapter.py:152-156 同判据），"
                f"实际 {type(net[-1]).__name__}"
            )

        # ---- FSQ 常量继承（源对象没有则用 sonic_decoder_torch.py:59-62 的同值）----
        self.fsq_offset = float(getattr(decoder, "fsq_offset", _FSQ_OFFSET))
        self.fsq_scale = float(getattr(decoder, "fsq_scale", _FSQ_SCALE))
        self.fsq_half = float(getattr(decoder, "fsq_half", _FSQ_HALF))
        self.fsq_denom = float(getattr(decoder, "fsq_denom", _FSQ_DENOM))

        # ---- 抽 base 的 Linear / 激活（激活取同型新建实例，无状态）----
        linears = [m for m in net if isinstance(m, nn.Linear)]
        acts = [m for m in net if not isinstance(m, nn.Linear)]
        if not linears:
            raise ValueError("base net 里没有 Linear 层")
        if expected_layers is not None and len(linears) != int(expected_layers):
            raise ValueError(
                f"base net 的 Linear 层数 {len(linears)} != 期望 {expected_layers}"
                "（D065 §3 要求挂 decoder 全部 7 个 Linear；层数不符即契约漂移）"
            )
        if not acts:
            raise ValueError("base net 里没有激活层（decoder 是 MLP，应有 SiLU）")
        act_types = {type(a) for a in acts}
        if len(act_types) != 1:
            raise ValueError(f"base net 的激活层类型不唯一：{sorted(t.__name__ for t in act_types)}")
        self.activation_name = type(acts[0]).__name__
        self.act = type(acts[0])()  # 新建同型实例（无状态，与 base 的前向等价性由自检保证）

        self.base_linears = nn.ModuleList()
        for lin in linears:
            new = nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None)
            with torch.no_grad():
                new.weight.copy_(lin.weight)
                if lin.bias is not None:
                    new.bias.copy_(lin.bias)
            new.requires_grad_(False)  # 冻结范围的可检查化
            self.base_linears.append(new)

        # ---- 每层挂 adapter（layer_ranks 逐层覆写，None = 该层不挂）----
        self.rank = int(rank)
        self.alpha = float(alpha)
        if layer_ranks is None:
            ranks = [self.rank] * len(self.base_linears)
        else:
            ranks = list(layer_ranks)
            if len(ranks) != len(self.base_linears):
                raise ValueError(
                    f"layer_ranks 长度 {len(ranks)} != base 层数 {len(self.base_linears)}"
                )
        self.adapters = nn.ModuleList(
            LoRAAdapter(lin.in_features, lin.out_features, ranks[i], self.alpha)
            if ranks[i] is not None
            else None
            for i, lin in enumerate(self.base_linears)
        )
        self.layer_ranks = ranks

        self.obs_dim = int(self.base_linears[0].in_features)
        self.action_dim = int(self.base_linears[-1].out_features)

        # ---- 运行期开关 / 缓存 ----
        self._envelope = 1.0          # 外层 LoRA 包络（不进 state_dict）
        self.collect_stats = False    # True 时每次 forward 记录 base/adapter 输出比
        self.last_stats: dict | None = None

        # ---- 探针（遗忘检查①/②共用；persistent=False ⇒ 不进 state_dict，但随 .to() 迁移）----
        # 先在**源 net 自己的设备**上算 base 参考输出（避免调用方给的 device 与源 net 不一致
        # 时 device mismatch），再整体 .to(device)。
        probe_dev = next(net.parameters()).device
        g = torch.Generator(device="cpu").manual_seed(0)
        probe = torch.randn(4, self.obs_dim, generator=g).to(probe_dev)
        self.register_buffer("_probe_x", probe, persistent=False)
        with torch.no_grad():
            self.register_buffer("_probe_y_base", net(probe), persistent=False)

        if device is not None:
            self.to(device)
        self.eval()  # 与 SonicTorchDecoder 一致（无 dropout/BN，eval 不影响梯度）

        self.construction_identity: dict = {}
        if verify_identity:
            self.construction_identity = self.verify_construction_identity()

    # ------------------------------------------------------------ 构造入口
    @staticmethod
    def _build_from_onnx(onnx_path: str, device):
        """延迟 import sonic_decoder_torch 建 base（该模块 import onnx，本机不可用）。"""
        try:  # 服务器执行根平铺 import / 仓库根包 import 双兼容（仓内既有惯例）
            from isaac.sonic_decoder_torch import SonicTorchDecoder
        except ImportError:  # pragma: no cover（本机走此支，需 apt_g1 可导入）
            from apt_g1.isaac.sonic_decoder_torch import SonicTorchDecoder
        return SonicTorchDecoder(onnx_path, device=str(device) if device is not None else "cpu")

    @classmethod
    def from_onnx(cls, onnx_path: str, **kwargs) -> "SonicLoRADecoder":
        """从 ONNX 独立构造（评测/导出路径；onnx_path 一并进身份信封）。"""
        kwargs.setdefault("onnx_path", onnx_path)
        obj = cls(None, **kwargs)
        obj.onnx_path = str(onnx_path)
        return obj

    # ------------------------------------------------------------ token / obs
    def quantize_tokens(self, tokens: torch.Tensor, ste: bool = False) -> torch.Tensor:
        """FSQ 量化（逐式照 sonic_decoder_torch.py:66-71，常量取构造时继承值）。

        `ste=True`：用 straight-through 估计器写 round（`y + (round(y)-y).detach()`，
        与 ORCS `fsq_quantize` 同式，rsl_rl/models/sonic_base_model.py:90）——
        **前向数值与 ste=False 逐位相同**（差值精确、加法回原值），只是反向多一条恒等梯度。
        纯 round 的导数恒 0，会把量化器**上游**（in-graph 时的 token head）的梯度整体抹零；
        decoder 内部的 adapter 在量化器下游，不受此影响。
        """
        latent = tokens.reshape(-1, 2, 32)
        x = torch.tanh(latent + self.fsq_offset)
        bounded = self.fsq_scale * x - self.fsq_half
        quantized = torch.round(bounded)
        if ste:
            quantized = bounded + (quantized - bounded).detach()
        return (quantized / self.fsq_denom).reshape(-1, TOKEN_DIM)

    def build_decoder_obs(
        self,
        tokens: torch.Tensor,
        ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        last_actions: torch.Tensor,
        gravity_dir: torch.Tensor,
    ) -> torch.Tensor:
        """(N, 994) 组装（照 sonic_decoder_torch.py:73-91 的 parts 顺序）。"""
        parts = [
            tokens.reshape(-1, TOKEN_DIM),
            ang_vel.reshape(-1, HISTORY_LEN * 3),
            joint_pos.reshape(-1, HISTORY_LEN * NUM_JOINTS),
            joint_vel.reshape(-1, HISTORY_LEN * NUM_JOINTS),
            last_actions.reshape(-1, HISTORY_LEN * NUM_JOINTS),
            gravity_dir.reshape(-1, HISTORY_LEN * 3),
        ]
        return torch.cat(parts, dim=1)

    def decode(self, tokens, ang_vel, joint_pos, joint_vel, last_actions, gravity_dir):
        """token（FSQ 前坐标）+ 10 帧本体历史 -> (N, 29)（同 SonicTorchDecoder.decode）。"""
        tokens_q = self.quantize_tokens(tokens)
        obs = self.build_decoder_obs(tokens_q, ang_vel, joint_pos, joint_vel, last_actions, gravity_dir)
        return self.forward(obs)

    def decode_obs(self, obs: torch.Tensor, ste: bool = False) -> torch.Tensor:
        """拼好的 (N,994) -> (N,29)：切 token 片量化后前向。

        与 sonic_action_term.py:129-145 `wrap_sonic_torch_decoder` 数值等价（同一
        运算序列：obs[:, :64] → quantize_tokens → 拼回 → forward），是本模块作为
        ActionTerm decoder 注入点的入口；等价性在 train 侧安装时现场断言。
        `ste=True`（in-graph 训练路径用）：量化走 STE，前向逐位不变、反向保留上游梯度
        （见 `quantize_tokens` docstring）；推理/等价断言一律用 ste=False。
        """
        tokens_q = self.quantize_tokens(obs[:, :TOKEN_DIM], ste=ste)
        return self.forward(torch.cat([tokens_q, obs[:, TOKEN_DIM:]], dim=1))

    # ------------------------------------------------------------ 前向
    def forward(self, obs: torch.Tensor, stats: dict | None = None) -> torch.Tensor:
        """(N,994) -> (N,29)：base 与 adapter 共用同一激活、相加后过激活。

        循环结构逐字照 ORCS mlp_with_adapter.py:187-205；envelope==0.0 时走**跳过
        adapter 的快速路径**（遗忘检查②的逐位一致性由此成为构造性保证）。
        `stats`/`self.collect_stats` 打开时记录逐层 base/adapter 输出幅度与比值。
        """
        if obs.dim() != 2 or int(obs.shape[-1]) != self.obs_dim:
            raise ValueError(f"decoder obs 须为 (N, {self.obs_dim})，收到 {tuple(obs.shape)}")
        env = float(self._envelope)
        collect = stats is not None or self.collect_stats
        rec: dict | None = {} if collect else None

        h = self.base_linears[0](obs)
        ad = self.adapters[0]
        if ad is not None and env != 0.0:
            a_out = ad(obs, env)
            if rec is not None:
                rec["layer0"] = _output_pair_stats(h, a_out)
            h = h + a_out
        for i in range(1, len(self.base_linears)):
            h = self.act(h)
            base_out = self.base_linears[i](h)
            ad = self.adapters[i]
            if ad is not None and env != 0.0:
                a_out = ad(h, env)
                if rec is not None:
                    rec[f"layer{i}"] = _output_pair_stats(base_out, a_out)
                base_out = base_out + a_out
            h = base_out
        if rec is not None:
            rec["envelope"] = env
            self.last_stats = rec
            if stats is not None:
                stats.update(rec)
        return h

    # ------------------------------------------------------------ 参数集合
    def adapter_params(self) -> list[nn.Parameter]:
        """只含 adapter 参数的列表（进优化器的唯一对象；预注册 §3「只训 adapter」）。"""
        return [p for ad in self.adapters if ad is not None for p in ad.parameters()]

    def base_parameters(self) -> list[nn.Parameter]:
        """base（decoder 本体）参数列表（一律 requires_grad=False）。"""
        return list(self.base_linears.parameters())

    def non_adapter_params(self) -> list[nn.Parameter]:
        """非 adapter 参数（= base；供「冻结范围」检查）。"""
        return self.base_parameters()

    def adapter_param_report(self) -> dict:
        """adapter 参数清单（数量/元素数/requires_grad/是否有梯度）——进日志与身份信封。"""
        params = self.adapter_params()
        n_grad = sum(1 for p in params if p.grad is not None)
        return {
            "n_tensors": len(params),
            "n_elements": int(sum(p.numel() for p in params)),
            "n_requires_grad": sum(1 for p in params if p.requires_grad),
            "n_with_grad": n_grad,
            "names": [n for n, _ in self.named_parameters() if ".down." in n or ".up." in n or ".delta." in n],
            "base_n_tensors": sum(1 for _ in self.base_linears.parameters()),
            "base_n_requires_grad": sum(1 for p in self.base_parameters() if p.requires_grad),
        }

    def base_frozen(self) -> bool:
        """base 是否全部 requires_grad=False（冻结范围的可检查化）。"""
        return all(not p.requires_grad for p in self.base_parameters())

    # ------------------------------------------------------------ 存取
    def adapter_state_dict(self) -> dict:
        """adapter 权重（含退化分支的 delta）——ckpt 侧车/身份信封用。"""
        out = {}
        for i, ad in enumerate(self.adapters):
            if ad is None:
                continue
            for name, tensor in ad.state_dict().items():
                out[f"adapters.{i}.{name}"] = tensor.detach().clone()
        return out

    def load_adapter_state_dict(self, sd: dict, strict: bool = True) -> list[str]:
        """灌入 adapter 权重（返回未命中的键；strict 时未命中即报错）。"""
        want = {f"adapters.{i}.{n}" for i, ad in enumerate(self.adapters) if ad is not None
                for n in ad.state_dict()}
        have = set(sd)
        missing = sorted(want - have)
        extra = sorted(have - want)
        if strict and (missing or extra):
            raise ValueError(f"adapter state_dict 不匹配：missing={missing} extra={extra}")
        with torch.no_grad():
            for i, ad in enumerate(self.adapters):
                if ad is None:
                    continue
                for name in ad.state_dict():
                    key = f"adapters.{i}.{name}"
                    if key in sd:
                        ad.state_dict()[name].copy_(sd[key])
        return missing + extra

    # ------------------------------------------------------------ 包络（遗忘检查②）
    @property
    def lora_scale(self) -> float:
        """当前外层包络（1.0=正常，0.0=遗忘检查的置零态）。"""
        return float(self._envelope)

    def set_lora_scale(self, scale: float) -> None:
        """设置外层包络（乘进每层 delta；不改权重，可无损还原）。"""
        self._envelope = float(scale)

    def lora_scale_zero_(self) -> None:
        """置零包络 ⇒ forward 走跳过 adapter 的快速路径 ⇒ 逐位回到原 decoder。"""
        self._envelope = 0.0

    def lora_scale_restore_(self) -> None:
        """还原包络到 1.0（lora_scale_zero_ 的逆操作）。"""
        self._envelope = 1.0

    def verify_lora_scale_zero(self) -> dict:
        """遗忘检查②：置零后与 base 逐位一致（用构造期探针）。

        返回 {"maxdiff": float, "bitwise": bool, "probe_shape": [...], "scale_restored": float}；
        调用后自动还原包络到调用前值。
        """
        keep = self._envelope
        self._envelope = 0.0
        with torch.no_grad():
            out = self.forward(self._probe_x)
        self._envelope = keep
        diff = (out - self._probe_y_base).abs().max().item()
        return {
            "maxdiff": float(diff),
            "bitwise": bool(torch.equal(out, self._probe_y_base)),
            "probe_shape": list(self._probe_x.shape),
            "scale_restored": float(self._envelope),
        }

    def verify_construction_identity(self) -> dict:
        """遗忘检查①：零初始化 ⇒ 前向与无 adapter 的 base 逐位一致（构造期探针）。"""
        with torch.no_grad():
            out = self.forward(self._probe_x)
        diff = (out - self._probe_y_base).abs().max().item()
        return {
            "maxdiff": float(diff),
            "bitwise": bool(torch.equal(out, self._probe_y_base)),
            "probe_shape": list(self._probe_x.shape),
            "probe": "randn(4, obs_dim), manual_seed(0)",
            "lora_scale": float(self._envelope),
        }

    def verify_decode_obs_equivalence(self, n: int = 4, seed: int = 0) -> dict:
        """`decode_obs` 与 `quantize_tokens`+`forward` 两条路径的数值等价自检。

        = sonic_action_term.wrap_sonic_torch_decoder 的运算序列（切 token 片→量化→拼回→forward）。
        """
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        obs = torch.randn(int(n), self.obs_dim, generator=g).to(self._probe_x.device)
        with torch.no_grad():
            a = self.decode_obs(obs)
            b = self.forward(torch.cat([self.quantize_tokens(obs[:, :TOKEN_DIM]), obs[:, TOKEN_DIM:]], dim=1))
        return {"maxdiff": float((a - b).abs().max().item()), "bitwise": bool(torch.equal(a, b)),
                "shape": list(a.shape)}

    # ------------------------------------------------------------ 诊断/合并/身份
    def adapter_delta_stats(self, envelope: float | None = None) -> dict:
        """逐层 delta 权重范数 + 与 base 权重的相对幅度（权重空间的「适配器在干活」指标）。"""
        env = self._envelope if envelope is None else float(envelope)
        layers = []
        for i, (lin, ad) in enumerate(zip(self.base_linears, self.adapters)):
            if ad is None:
                layers.append({"layer": i, "adapter": None})
                continue
            dw = ad.delta_weight(env)
            w_norm = float(lin.weight.detach().norm().item())
            layers.append({
                "layer": i,
                "kind": "low_rank" if ad.low_rank else "full_rank",
                "scale": ad.scale,
                "delta_fro_norm": float(dw.norm().item()),
                "delta_abs_max": float(dw.abs().max().item()),
                "base_fro_norm": w_norm,
                "rel_fro": float(dw.norm().item() / (w_norm + _RATIO_EPS)),
            })
        return {"envelope": env, "layers": layers}

    @torch.no_grad()
    def merge_adapters(self) -> tuple[nn.Sequential, dict]:
        """把 delta 合并进 base 权重，返回新的 ``nn.Sequential``（导出用，不改本模块）。

        W_merged = W_base + envelope·scale·(up @ down)（数学等价于分开前向；GEMM 累加序不同，
        浮点上非逐位等价，selftest 以 1e-5 容差断言）。
        """
        mods: list[nn.Module] = []
        info_layers = []
        for i, (lin, ad) in enumerate(zip(self.base_linears, self.adapters)):
            merged = nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None)
            with torch.no_grad():
                w = lin.weight.detach().clone()
                if ad is not None:
                    w = w + ad.delta_weight(self._envelope)
                merged.weight.copy_(w)
                if lin.bias is not None:
                    merged.bias.copy_(lin.bias.detach())
            merged.requires_grad_(False)
            mods.append(merged)
            if i < len(self.base_linears) - 1:
                mods.append(type(self.act)())
            info_layers.append({
                "layer": i,
                "merged": ad is not None,
                "delta_fro_norm": float(ad.delta_weight(self._envelope).norm().item()) if ad is not None else 0.0,
            })
        net = nn.Sequential(*mods)
        net.eval()
        return net, {
            "envelope": float(self._envelope),
            "layers": info_layers,
            "activation": self.activation_name,
            "note": "W_merged = W_base + envelope·(alpha/rank)·up@down；数学等价、非逐位等价",
        }

    @torch.no_grad()
    def base_weight_md5(self) -> str:
        """base 权重（7 层 weight+bias，按序）字节流的 md5 —— 冻结范围的指纹。"""
        h = hashlib.md5()
        for lin in self.base_linears:
            h.update(lin.weight.detach().cpu().contiguous().numpy().tobytes())
            if lin.bias is not None:
                h.update(lin.bias.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    @torch.no_grad()
    def base_state_snapshot(self) -> dict:
        """base 权重快照（训练后断言「base 与初始逐位一致」用）。"""
        snap = {}
        for i, lin in enumerate(self.base_linears):
            snap[f"base.{i}.weight"] = lin.weight.detach().clone()
            if lin.bias is not None:
                snap[f"base.{i}.bias"] = lin.bias.detach().clone()
        return snap

    @torch.no_grad()
    def base_state_maxdiff(self, snapshot: dict) -> float:
        """当前 base 权重 vs 快照的最大绝对差（0.0 = 逐位冻结）。"""
        worst = 0.0
        for i, lin in enumerate(self.base_linears):
            key = f"base.{i}.weight"
            if key in snapshot:
                worst = max(worst, float((lin.weight.detach() - snapshot[key]).abs().max().item()))
            bkey = f"base.{i}.bias"
            if lin.bias is not None and bkey in snapshot:
                worst = max(worst, float((lin.bias.detach() - snapshot[bkey]).abs().max().item()))
        return worst

    def layer_table(self) -> list[dict]:
        """逐层形状表（身份信封；含是否挂了 adapter / 是否退化分支）。"""
        rows = []
        for i, (lin, ad) in enumerate(zip(self.base_linears, self.adapters)):
            row = {
                "layer": i,
                "in_features": int(lin.in_features),
                "out_features": int(lin.out_features),
                "adapter": None if ad is None else ad.config(),
            }
            rows.append(row)
        return rows

    def identity_envelope(self, extra: dict | None = None) -> dict:
        """身份信封：结构 + LoRA 配置 + base 指纹 + 两条遗忘检查结果。

        train/eval 侧再叠加 seed / git commit / onnx md5 等运行期身份（本模块只出
        「与本次构造绑定」的部分，避免把运行期上下文塞进模块）。
        """
        env = {
            "format": 1,
            "experiment": "D065",
            "arm": "C",
            "module": "apt_g1/isaac/sonic_decoder_lora.py",
            "class": type(self).__name__,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "activation": self.activation_name,
            "num_layers": len(self.base_linears),
            "rank": self.rank,
            "alpha": self.alpha,
            "layer_ranks": list(self.layer_ranks),
            "lora_scale": float(self._envelope),
            "degenerate_layers": [i for i, ad in enumerate(self.adapters)
                                  if ad is not None and not ad.low_rank],
            "layers": self.layer_table(),
            "base_weight_md5": self.base_weight_md5(),
            "base_frozen": self.base_frozen(),
            "adapter_params": self.adapter_param_report(),
            "construction_identity": self.construction_identity,
            "scale_zero_identity": self.verify_lora_scale_zero(),
            "decode_obs_equivalence": self.verify_decode_obs_equivalence(),
            "source": "ORCS rsl_rl/modules/mlp_with_adapter.py:38-205（数学照抄）",
        }
        if getattr(self, "onnx_path", None):
            env["onnx_path"] = str(self.onnx_path)
        if extra:
            env.update(extra)
        return env


def _output_pair_stats(base_out: torch.Tensor, adapter_out: torch.Tensor) -> dict:
    """单层 base/adapter 输出的幅度与比值（「适配器在干活」的运行期指标）。"""
    b = float(base_out.detach().abs().mean().item())
    a = float(adapter_out.detach().abs().mean().item())
    return {
        "base_abs_mean": b,
        "adapter_abs_mean": a,
        "ratio": a / (b + _RATIO_EPS),
        "adapter_abs_max": float(adapter_out.detach().abs().max().item()),
    }


# ---------------------------------------------------------------------------
# Layer 3: 接线到 ActionTerm（鸭子类型，不 import isaaclab）
# ---------------------------------------------------------------------------
def install_into_action_term(term, lora: SonicLoRADecoder) -> tuple[object, object]:
    """把 LoRA decoder 换进 ``SonicDecoderActionTerm``（返回被替换的旧对象）。

    **v1 遗留接线：v2（in-graph）形态下本函数无调用点**——v2 的 decoder 由 policy 持有
    （`sonic_lora_policy.SonicLoRAPolicy.mu_path.decoder`），env 侧不再有 decoder action term。
    保留本函数仅供 v1 复现/诊断（v1 挂法下 PPO 对 adapter 梯度恒 0，见模块 docstring
    「已知限制」第 1 条）。

    只依赖两个属性（鸭子类型，故本模块在无 isaaclab 的机器上仍可 import/单测）：
      - ``term._sonic_decoder``：SonicTorchDecoder 实例（替换为 lora）
      - ``term.core.decoder``：``(N,994)->(N,29)`` 可调用（替换为 lora.decode_obs）
    注意 ``core.decoder`` 在 term 构造时是 ``wrap_sonic_torch_decoder(旧 decoder)``
    的闭包——**必须一起换**，否则换了个寂寞（闭包仍指向旧 decoder）。
    """
    core = getattr(term, "core", None)
    if core is None or not hasattr(core, "decoder"):
        raise TypeError(
            f"{type(term).__name__} 无 .core.decoder（须为 SonicDecoderActionTerm）；"
            "LoRA 安装点契约不符即显式失败，不静默跳过"
        )
    old_decoder = getattr(term, "_sonic_decoder", None)
    old_callable = core.decoder
    term._sonic_decoder = lora
    core.decoder = lora.decode_obs
    return old_decoder, old_callable