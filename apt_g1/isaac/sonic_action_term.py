"""Frozen-SONIC-decoder ActionTerm for the Isaac Lab manager-based stack (D064).

立项 D064：RL 底座从自研 DirectRLEnv（apt_flat_env.py）迁到 Isaac Lab
manager-based velocity 栈 + rsl_rl；冻结 SONIC ONNX decoder 挂成自定义
ActionTerm，动作空间 = 64 维 FSQ token 直出（无 VAE，无 router/aux/res）。

两层结构：

1. ``SonicActionCore`` —— 纯 torch（零 isaaclab / onnx import，本机可单测）：
   - E49/D055 已验证的 token 仿射（token_mode 分支，apt_flat_env.py:1017-1036）
   - 10 帧环形历史（ang_vel 3 / joint_pos 29 / joint_vel 29 / last_act 29 /
     gravity 3 = 930 维；apt_flat_env.py:701-716 分配、889-898 push）
   - (N, 994) decoder obs 组装（sonic_decoder_torch.py:73-91 +
     apt_flat_env.py:869-887 的输入契约）
   - decode 注入式（core 只依赖一个 obs->action 可调用；真实 ONNX 解码器
     只在 cvgl 存在，本机单测用 stub）
   - SONIC 29 序 -> 资产关节序映射（resolve_sonic_joint_ids）

2. isaaclab 薄壳 —— ``SonicDecoderActionTermCfg`` / ``SonicDecoderActionTerm``
   （manager-based ActionTerm）。isaaclab / onnx / g1(资产 cfg) 的 import
   集中放在文件后部，保证 Layer 1 在无 isaaclab 的机器（本机 Windows）
   可独立 import。

Decoder obs 契约（994 维，通道顺序 = sonic_decoder_torch.py:83-91
build_decoder_obs 的 parts 顺序；帧序 oldest -> newest 沿 dim 1）::

    [  0:  64) token            （FSQ 量化前坐标，量化在 decoder 适配器内）
    [ 64:  94) ang_vel          10 x 3   （root_ang_vel_b，机体系）
    [ 94: 384) joint_pos        10 x 29  （q - sonic_default，相对量）
    [384: 674) joint_vel        10 x 29  （qvel，SONIC 序）
    [674: 964) last_act         10 x 29  （(q_des - default) / scale，
                                            被执行关节目标，非策略原始输出）
    [964: 994) gravity          10 x 3   （projected_gravity_b）

reset 站立帧 = gravity 每帧 [0, 0, -1]、其余通道 0（apt_flat_env.py:1720-1729）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

# SONIC 29 关节名（torch-only 模块，包 __init__ 均为空，本机可安全 import）。
# 这就是 SONIC 的 29 维顺序：apt_flat_env.py:509 `_body_names = G1_ISAACLab_ORDER`，
# 历史/解码器/动作全部按此序组织；不是 MuJoCo 序（MuJoCo 常量需经
# G1_MUJOCO_TO_ISAACLAB_DOF 重排，见 apt_flat_env.py:90-97）。
from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER

if TYPE_CHECKING:  # 仅注解用；运行时 import 在文件后部 isaaclab 薄壳段
    from isaaclab.envs import ManagerBasedEnv

# ---------------------------------------------------------------------------
# Layer 1: SonicActionCore —— 纯 torch，零 isaaclab / onnx import
# ---------------------------------------------------------------------------
SONIC_JOINT_NAMES = G1_ISAACLab_ORDER  # SONIC 29 序 = G1_ISAACLab_ORDER (joint_utils.py:10-40)
SONIC_NUM_JOINTS = 29
TOKEN_DIM = 64
HISTORY_LEN = 10
ANG_VEL_DIM = 3
GRAVITY_DIM = 3

# (N, 994) 各段宽度（sonic_decoder_torch.py:83-91 的 parts 逐段 reshape 宽度）
HIST_ANG_VEL_WIDTH = HISTORY_LEN * ANG_VEL_DIM        # 30
HIST_JOINT_WIDTH = HISTORY_LEN * SONIC_NUM_JOINTS     # 290
HIST_GRAVITY_WIDTH = HISTORY_LEN * GRAVITY_DIM        # 30
DECODER_OBS_DIM = (
    TOKEN_DIM
    + HIST_ANG_VEL_WIDTH
    + 3 * HIST_JOINT_WIDTH
    + HIST_GRAVITY_WIDTH
)  # 64 + 30 + 3*290 + 30 = 994

VALID_TOKEN_BOUNDS = ("none", "tanh")


def resolve_env_index(
    env_ids: Sequence[int] | torch.Tensor | slice | None,
    num_envs: int,
) -> tuple[Sequence[int] | torch.Tensor | slice, int]:
    """把 reset 钩子可能收到的 env_ids 形态解析成 (索引, 行数)。

    官方 ActionManager.reset 在 env_ids=None 时传 ``slice(None)``
    （tmp/isaac_ref/action_manager.py:306-316）；slice 不能过
    torch.as_tensor（TypeError: 'slice' object cannot be interpreted as an
    integer），必须原样作索引直通；仅 list/Tensor 走 as_tensor。
    返回第二项 = 该批行数 n（None / slice(None) -> num_envs；带界 slice 按
    range 长度），供站立帧 gravity 的 repeat 使用。
    """
    if env_ids is None:
        return slice(None), num_envs
    if isinstance(env_ids, slice):
        if env_ids.start is None and env_ids.stop is None:
            return env_ids, num_envs
        n = len(range(
            env_ids.start if env_ids.start is not None else 0,
            env_ids.stop if env_ids.stop is not None else num_envs,
            env_ids.step if env_ids.step is not None else 1,
        ))
        return env_ids, n
    t = torch.as_tensor(env_ids, dtype=torch.long)
    return t, int(t.numel())


def resolve_sonic_joint_ids(asset_joint_names: Sequence[str]) -> tuple[list[int], bool]:
    """SONIC 29 序 -> 资产关节序的索引映射。

    依据（与 apt_flat_env 同式）：
    - SONIC 29 序 = G1_ISAACLab_ORDER（apt_flat_env.py:509；joint_utils.py:9-40，
      其 docstring "G1 body joint names in IsaacLab order" 即按 gear_sonic 的
      IsaacLab 资产而设）。
    - 映射构造 = apt_flat_env.py:545-549
      ``[robot.joint_names.index(n) for n in G1_ISAACLab_ORDER]``。

    返回 ``(ids, identity)``：``ids[i]`` = SONIC 序第 i 个关节在资产关节表中
    的下标；``identity=True`` 表示资产关节序与 G1_ISAACLab_ORDER 完全同序
    （此时映射为直通 arange(29)，apply 可免索引 scatter）。gear_sonic 29-DOF
    g1 资产预期恒等，但资产 USD 解析序不在此静态保证，运行时必须校验并
    打日志（见 SonicDecoderActionTerm.__init__）。
    """
    names = list(asset_joint_names)
    ids = [names.index(n) for n in SONIC_JOINT_NAMES]
    identity = ids == list(range(len(SONIC_JOINT_NAMES)))
    return ids, identity


def wrap_sonic_torch_decoder(decoder: Callable) -> Callable[[torch.Tensor], torch.Tensor]:
    """把 SonicTorchDecoder 适配成 core 的 (N,994)->(N,29) 可调用契约。

    apt_flat_env 统一入口 ``self._decoder.decode(*self._decoder_obs_parts(tokens))``
    （apt_flat_env.py:1099）里，decode() 会先对 64 维 token 做 FSQ 量化再进
    MLP（sonic_decoder_torch.py:103-105）——E49 token 直出路径同样过量化
    （冻结 decoder 的一部分；"无 VAE" 指不做 VAE 解码，不是跳过 FSQ）。
    裸 ``forward(obs)`` 假设 token 已量化（sonic_decoder_torch.py:107-109）。
    本适配器对拼好的 (N,994) obs 切出 token 片 quantize_tokens 后再
    forward，与 decode(*parts) 数值等价。
    """

    def _decode_obs(obs: torch.Tensor) -> torch.Tensor:
        tokens_q = decoder.quantize_tokens(obs[:, :TOKEN_DIM])
        return decoder.forward(torch.cat([tokens_q, obs[:, TOKEN_DIM:]], dim=1))

    return _decode_obs


class SonicActionCore:
    """token 仿射 + 10 帧环形历史 + (N,994) 组装 + 注入式 decode 的纯 torch 核心。

    状态语义与 apt_flat_env 的 DirectRLEnv 实现逐条对应（见各类内注释的
    行号锚点）；decoder 通过构造参数注入，便于本机 stub 单测。
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        token_mean: torch.Tensor | np.ndarray,
        token_std: torch.Tensor | np.ndarray,
        sonic_default: torch.Tensor | np.ndarray,
        sonic_scale: torch.Tensor | np.ndarray,
        decoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        token_alpha: float = 1.0,
        token_bound: str = "none",
    ) -> None:
        if token_bound not in VALID_TOKEN_BOUNDS:
            raise ValueError(
                f"token_bound 须 ∈ {VALID_TOKEN_BOUNDS}（apt_flat_env.py:284 口径），"
                f"收到 {token_bound!r}"
            )
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.token_alpha = float(token_alpha)
        self.token_bound = token_bound
        self.token_mean = torch.as_tensor(token_mean, dtype=torch.float32, device=self.device).reshape(TOKEN_DIM)
        self.token_std = torch.as_tensor(token_std, dtype=torch.float32, device=self.device).reshape(TOKEN_DIM)
        self.sonic_default = torch.as_tensor(
            sonic_default, dtype=torch.float32, device=self.device
        ).reshape(SONIC_NUM_JOINTS)
        self.sonic_scale = torch.as_tensor(
            sonic_scale, dtype=torch.float32, device=self.device
        ).reshape(SONIC_NUM_JOINTS)
        self.decoder = decoder

        # ---- 10 帧环形历史（oldest -> newest 沿 dim 1；apt_flat_env.py:701-716）
        def _zeros(dim: int) -> torch.Tensor:
            return torch.zeros(self.num_envs, HISTORY_LEN, dim, dtype=torch.float32, device=self.device)

        self.hist_ang_vel = _zeros(ANG_VEL_DIM)        # (N, 10, 3)
        self.hist_joint_pos = _zeros(SONIC_NUM_JOINTS)  # (N, 10, 29) 存相对量 q - default
        self.hist_joint_vel = _zeros(SONIC_NUM_JOINTS)  # (N, 10, 29)
        self.hist_last_act = _zeros(SONIC_NUM_JOINTS)   # (N, 10, 29) 被执行关节目标（归一化）
        self.hist_gravity = _zeros(GRAVITY_DIM)         # (N, 10, 3)

        # 被执行关节目标缓存；初始 = 站立 default（apt_flat_env.py:765 分配、
        # 1790 reset 置 default 的合成态——新 env 未走任何控制步时 last_act=0）
        self.q_des = self.sonic_default.repeat(self.num_envs, 1).clone()

        # 初始即站立历史帧（等价 apt_flat_env 首个 _reset_idx 的填充，1720-1729）
        self.reset()

    # ------------------------------------------------------------ token 仿射
    def map_token(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """E49/D055 token 仿射：token = mean + alpha * std * a（token_bound 分支）。

        严格照 apt_flat_env.py:1017-1036 token_mode 分支：``"tanh"`` 时对 a 取
        tanh（限幅到 mean ± alpha*std，E49-A 抗漂移臂）；``"none"`` 为无界
        线性映射。a = 策略原始 64 维输出（obs 反馈口径是原始 a，与本映射
        解耦，apt_flat_env.py:1291-1300）。
        """
        if self.token_bound == "tanh":
            return self.token_mean + self.token_alpha * self.token_std * torch.tanh(raw_actions)
        return self.token_mean + self.token_alpha * self.token_std * raw_actions

    # ------------------------------------------------------------- obs 组装
    def assemble_obs(self, tokens: torch.Tensor) -> torch.Tensor:
        """拼 (N, 994) = token(64) + [ang_vel, joint_pos, joint_vel, last_act, gravity] 摊平。

        段序严格照 SonicTorchDecoder.build_decoder_obs 的 parts
        （sonic_decoder_torch.py:83-91）；每段 (N,10,d)->(N,10*d) 的 reshape
        即按帧 oldest->newest 摊平，与 apt_flat_env._decoder_obs_parts
        （apt_flat_env.py:869-887）直接传历史 (N,10,d) 张量、由
        build_decoder_obs 内 reshape 的路径数值一致。
        """
        parts = [
            tokens.reshape(-1, TOKEN_DIM),
            self.hist_ang_vel.reshape(-1, HIST_ANG_VEL_WIDTH),
            self.hist_joint_pos.reshape(-1, HIST_JOINT_WIDTH),
            self.hist_joint_vel.reshape(-1, HIST_JOINT_WIDTH),
            self.hist_last_act.reshape(-1, HIST_JOINT_WIDTH),
            self.hist_gravity.reshape(-1, HIST_GRAVITY_WIDTH),
        ]
        return torch.cat(parts, dim=1)

    # ----------------------------------------------------------- 历史维护
    def last_act_from_q_des(self, q_des: torch.Tensor) -> torch.Tensor:
        """被执行关节目标 -> 历史归一化动作 (q_des - default) / scale。

        apt_flat_env.py:1845（_post_step_history）：历史里的 last_act 通道是
        上一步被执行的关节目标（非策略原始输出）。
        """
        return (q_des - self.sonic_default) / self.sonic_scale

    @staticmethod
    def _push(buf: torch.Tensor, value: torch.Tensor) -> None:
        """环形左移 oldest->newest：丢弃最老帧，value 落到最新帧。

        apt_flat_env.py:890-892 原式（``buf[:, :-1] = buf[:, 1:].clone()``，
        clone 防同张量重叠视图赋值的未定义行为）。
        """
        buf[:, :-1] = buf[:, 1:].clone()
        buf[:, -1] = value.detach()

    def push_state(
        self,
        ang_vel: torch.Tensor,
        joint_pos_rel: torch.Tensor,
        joint_vel: torch.Tensor,
        last_act: torch.Tensor,
        gravity: torch.Tensor,
    ) -> None:
        """推入一个物理帧（五通道同序，apt_flat_env.py:894-898）。"""
        self._push(self.hist_ang_vel, ang_vel)
        self._push(self.hist_joint_pos, joint_pos_rel)
        self._push(self.hist_joint_vel, joint_vel)
        self._push(self.hist_last_act, last_act)
        self._push(self.hist_gravity, gravity)

    # ---------------------------------------------------------------- decode
    def compute_q_des(
        self, raw_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """策略 64 维原始输出 -> (q_des, tokens, obs)；q_des 缓存到 self.q_des。

        管线 = apt_flat_env token_mode 一步控制步：token 仿射（1026-1035）->
        组装 994 obs（_decoder_obs_parts）-> 冻结 decoder -> 归一化关节动作
        -> q_des = default + action * scale（1099-1102）。decoder 输出为
        SONIC 29 序归一化关节目标。全程 no_grad（冻结先验不建图，等价
        apt_flat_env.py:1306 的 .detach()）。
        """
        if self.decoder is None:
            raise RuntimeError("SonicActionCore.decoder 未注入（本机测试请传 stub）")
        with torch.no_grad():
            tokens = self.map_token(raw_actions)
            obs = self.assemble_obs(tokens)
            action = self.decoder(obs)
            q_des = self.sonic_default + action * self.sonic_scale
        self.q_des = q_des.detach()
        return self.q_des, tokens, obs

    # ----------------------------------------------------------------- reset
    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None = None
    ) -> None:
        """站立帧重填该批 env：gravity 每帧 [0,0,-1]、其余通道 0；q_des 回 default。

        apt_flat_env.py:1720-1729（历史填充）+ 1790（q_des 置 default）。
        env_ids 兼容官方 ActionManager.reset 的 slice(None) 形态
        （tmp/isaac_ref/action_manager.py:306-316，见 resolve_env_index）。
        """
        idx, n = resolve_env_index(env_ids, self.num_envs)
        if isinstance(idx, torch.Tensor):
            idx = idx.to(self.device)
        self.hist_ang_vel[idx] = 0.0
        self.hist_joint_pos[idx] = 0.0
        self.hist_joint_vel[idx] = 0.0
        self.hist_last_act[idx] = 0.0
        gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
        self.hist_gravity[idx] = gravity.repeat(n, 1)[:, None, :].expand(n, HISTORY_LEN, GRAVITY_DIM).clone()
        self.q_des[idx] = self.sonic_default


# ---------------------------------------------------------------------------
# Layer 2: isaaclab 薄壳 —— import 集中在后部，保证 Layer 1 可独立 import
# （本机无 isaaclab/onnx：以上代码 import 时不触发任何 isaaclab / onnx 依赖）
# ---------------------------------------------------------------------------
from isaaclab.managers import ActionTerm, ActionTermCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

try:  # 服务器执行根平铺 import / 仓库根包 import 双兼容（仓内既有惯例）
    from isaac.sonic_decoder_torch import SonicTorchDecoder
except ImportError:  # pragma: no cover（本机走此支，需 apt_g1 可导入）
    from apt_g1.isaac.sonic_decoder_torch import SonicTorchDecoder

# G1_MUJOCO_TO_ISAACLAB_DOF 所在模块顶部 import isaaclab（robots/__init__.py
# 还会 re-export g1），只能放在薄壳段——与 apt_flat_env.py:36 同一来源。
from gear_sonic.envs.manager_env.robots.g1 import G1_MUJOCO_TO_ISAACLAB_DOF  # noqa: E402

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


# SONIC 常量（照抄 apt_flat_env.py:55-78 的加载方式：MuJoCo 序常量 + 经
# G1_MUJOCO_TO_ISAACLAB_DOF 重排到 G1_ISAACLab_ORDER，apt_flat_env.py:90-97）。
SONIC_ACTION_SCALE_MUJOCO = np.array(
    [
        0.3506614664, 0.3506614664, 0.5475464652, 0.3506614664,
        0.4385773139, 0.4385773139, 0.3506614664, 0.3506614664,
        0.5475464652, 0.3506614664, 0.4385773139, 0.4385773139,
        0.5475464652, 0.4385773139, 0.4385773139, 0.4385773139,
        0.4385773139, 0.4385773139, 0.4385773139, 0.4385773139,
        0.0745008703, 0.0745008703, 0.4385773139, 0.4385773139,
        0.4385773139, 0.4385773139, 0.4385773139, 0.0745008703,
        0.0745008703,
    ],
    dtype=np.float32,
)

SONIC_DEFAULT_ANGLES_MUJOCO = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)


def _sonic_default_isaac() -> np.ndarray:
    """SONIC default angles in G1_ISAACLab_ORDER (29,)（apt_flat_env.py:90-92 同式）。"""
    return SONIC_DEFAULT_ANGLES_MUJOCO[G1_MUJOCO_TO_ISAACLAB_DOF].astype(np.float32)


def _sonic_scale_isaac() -> np.ndarray:
    """SONIC action scales in G1_ISAACLab_ORDER (29,)（apt_flat_env.py:95-97 同式）。"""
    return SONIC_ACTION_SCALE_MUJOCO[G1_MUJOCO_TO_ISAACLAB_DOF].astype(np.float32)


class SonicDecoderActionTerm(ActionTerm):
    """把冻结 SONIC decoder 挂成 manager-based ActionTerm（D064）。

    - 动作空间 64 维 = FSQ token 直出（无 VAE）；token 仿射照 E49 token_mode。
    - ``process_actions``（每控制步一次，action_manager.py:33-35）：先推
      上一步末物理状态 + 上一步 last_act 进历史（= DirectRLEnv 里
      _post_step_history 在物理步末推送的同一帧，apt_flat_env.py:1837-1846），
      再 token 仿射 -> 组装 994 -> decoder -> q_des 缓存。
    - ``apply_actions``（每仿真子步一次，action_manager.py:36-39）：按
      SONIC->asset 映射序 set_joint_position_target。
    - ``reset``（由 ActionManager.reset 调，action_manager.py:296-316）：
      站立帧重填该批 env + q_des 回 default。

    等价性注记（vs apt_flat_env DirectRLEnv 时序）：DirectRLEnv 在 step 末
    （物理已走完、超时 env 已被 auto-reset）执行 _post_step_history
    （apt_flat_env.py:1848-1850）；manager-based 的 reset 同样发生在上一控制
    步收尾、本控制步 process_actions 之前，故「process_actions 开头推送当前
    asset 状态」推到的就是同一帧（复位 env 推到站立帧，数值与
    apt_flat_env 一致）。
    """

    cfg: SonicDecoderActionTermCfg

    def __init__(self, cfg: SonicDecoderActionTermCfg, env: ManagerBasedEnv) -> None:
        # ActionTerm.__init__：self._env = env；self._asset = env.scene[asset_name]
        # （action_manager.py:42-57）
        super().__init__(cfg, env)
        if not self.cfg.token_stats:
            raise ValueError(
                "SonicDecoderActionTermCfg.token_stats 必填：token 直出路径需要 "
                "官方 g1-mode token 统计 npz（mean/std 键，apt_flat_env.py:276,616-625）"
            )
        if self.cfg.token_bound not in VALID_TOKEN_BOUNDS:
            raise ValueError(
                f"token_bound 须 ∈ {VALID_TOKEN_BOUNDS}，收到 {self.cfg.token_bound!r}"
            )

        # ---- 关节序映射：SONIC 29 序 -> 资产关节序（apt_flat_env.py:545-549 同式）。
        # find_joints(preserve_order=True) 保证返回下标跟随查询名序（即
        # G1_ISAACLab_ORDER 序），官方 JointAction 同款 API
        # （tmp/isaac_ref/joint_actions.py:61-63）。
        self._joint_ids, self._joint_names = self._asset.find_joints(
            G1_ISAACLab_ORDER, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints != SONIC_NUM_JOINTS:
            raise ValueError(
                f"资产需含全部 {SONIC_NUM_JOINTS} 个 SONIC 关节，解析到 {self._num_joints}："
                f"missing={sorted(set(G1_ISAACLab_ORDER) - set(self._joint_names))}"
            )
        self._joint_ids_t = torch.as_tensor(self._joint_ids, dtype=torch.long, device=self.device)
        _ids, identity = resolve_sonic_joint_ids(self._asset.joint_names)
        if list(_ids) != list(self._joint_ids):
            raise RuntimeError(
                "find_joints 与 resolve_sonic_joint_ids 的映射不一致（内部错误）"
            )
        print(
            f"[SonicDecoderActionTerm] SONIC->{self._asset.num_joints} 关节映射 "
            f"identity={identity}（G1_ISAACLab_ORDER vs asset.joint_names）；"
            f"ids={list(self._joint_ids)}"
        )

        # ---- 冻结 decoder（cvgl 的 ONNX 权重抽取版，动态 batch GPU MLP）
        self._sonic_decoder = SonicTorchDecoder(self.cfg.onnx_path, device=self.device)

        # ---- token 统计（npz mean/std，std 下限截断；apt_flat_env.py:616-625 同式）
        stats = np.load(self.cfg.token_stats)
        token_mean = torch.as_tensor(
            np.asarray(stats["mean"], dtype=np.float32), device=self.device
        )
        token_std = torch.as_tensor(
            np.maximum(
                np.asarray(stats["std"], dtype=np.float32), self.cfg.token_std_floor
            ),
            device=self.device,
        )
        if token_mean.shape != (TOKEN_DIM,) or token_std.shape != (TOKEN_DIM,):
            raise ValueError(
                f"token_stats npz 的 mean/std 须为 ({TOKEN_DIM},)，收到 "
                f"{tuple(token_mean.shape)} / {tuple(token_std.shape)}"
            )

        # ---- 纯 torch 核心（decoder 注入 = FSQ 量化 + MLP forward 适配）
        self.core = SonicActionCore(
            num_envs=self.num_envs,
            device=self.device,
            token_mean=token_mean,
            token_std=token_std,
            sonic_default=_sonic_default_isaac(),
            sonic_scale=_sonic_scale_isaac(),
            decoder=wrap_sonic_torch_decoder(self._sonic_decoder),
            token_alpha=self.cfg.token_alpha,
            token_bound=self.cfg.token_bound,
        )

        # raw/processed 缓冲（官方约定：raw = 策略输入原样；processed =
        # apply_actions 实际执行的量，tmp/isaac_ref/joint_actions.py:76-77,130-139）
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(
            self.num_envs, SONIC_NUM_JOINTS, device=self.device
        )

    # ------------------------------------------------------------ properties
    @property
    def action_dim(self) -> int:
        """64：策略动作 = 原始 64 维 token 坐标（无界，经仿射映射后进 decoder）。"""
        return TOKEN_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        """策略原始 64 维 token 坐标（官方语义，action_manager.py:75-79）。"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """(N, 29) 被执行关节目标 q_des（官方语义「处理后要施加的量」，
        joint_actions.py:158-160 中 apply 消费 processed_actions）。"""
        return self._processed_actions

    @property
    def sonic_q_des(self) -> torch.Tensor:
        """core 内缓存的被执行关节目标（与 processed_actions 同值，SONIC 29 序）。"""
        return self.core.q_des

    # ------------------------------------------------------------ operations
    def process_actions(self, actions: torch.Tensor) -> None:
        # 官方约定：先存 raw（joint_actions.py:132）
        self._raw_actions[:] = actions

        # 1) 推上一步末物理状态 + 上一步 last_act 进历史
        #    （= apt_flat_env.py:1837-1846 _post_step_history 同一帧：
        #    root_ang_vel_b / projected_gravity_b / joint_pos - default /
        #    joint_vel（经 SONIC->asset 映射索引）/ (q_des_prev - default)/scale）
        asset = self._asset
        ang_vel = asset.data.root_ang_vel_b
        gravity = asset.data.projected_gravity_b
        joint_pos_rel = asset.data.joint_pos[:, self._joint_ids_t] - self.core.sonic_default
        joint_vel = asset.data.joint_vel[:, self._joint_ids_t]
        last_act = self.core.last_act_from_q_des(self.core.q_des)
        self.core.push_state(ang_vel, joint_pos_rel, joint_vel, last_act, gravity)

        # 2) token 仿射 -> (N,994) 组装 -> 冻结 decoder -> q_des（core 内缓存）
        q_des, _tokens, _obs = self.core.compute_q_des(self._raw_actions)

        # processed = 本控制步要执行的 q_des（apply_actions 消费）
        self._processed_actions[:] = q_des

    def apply_actions(self) -> None:
        # 按 SONIC->asset 映射序写关节目标：joint_ids 形式与 apt_flat_env 的
        # scatter 等价（apt_flat_env.py:1331-1344 ``full[:, _body_idx] = q_des;
        # set_joint_position_target(full, joint_ids=None)``；官方
        # JointPositionAction 的 joint_ids 形式见 joint_actions.py:158-160）
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """ActionManager.reset 官方 hook（action_manager.py:296-316 会调用）。

        env_ids 兼容官方传参形态：None 时 ActionManager 传 ``slice(None)``
        （action_manager.py:306-316，slice 原样直通，见 resolve_env_index），
        部分 reset 传 list/Tensor。站立帧重填该批 env 历史 + q_des 回
        default（apt_flat_env.py:1720-1729, 1790）；raw 动作清零（官方
        JointAction.reset，joint_actions.py:141-142；processed 一并清零属
        防御性——下一控制步 process_actions 必然先于 apply 重写它）。
        """
        idx, _n = resolve_env_index(env_ids, self.num_envs)
        if isinstance(idx, torch.Tensor):
            idx = idx.to(self.device)
        self._raw_actions[idx] = 0.0
        self._processed_actions[idx] = 0.0
        self.core.reset(env_ids)  # core 侧同样经 resolve_env_index 解析

    def reset_idx(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """reset 的别名（派发口径用词；官方 hook 名是 ``reset``）。"""
        self.reset(env_ids)


@configclass
class SonicDecoderActionTermCfg(ActionTermCfg):
    """D064 token 直出 ActionTerm 的配置。

    class_type 指向本模块的 term（term 类须先于本 cfg 定义，configclass 需要
    类对象作默认值——官方 action_cfg.py 同款模式）。
    """

    class_type: type[ActionTerm] = SonicDecoderActionTerm
    # 官方 ActionTermCfg 基础字段（manager_term_cfg.py）：class_type/debug_vis/asset_name
    asset_name: str = "robot"
    debug_vis: bool = False

    # SONIC ONNX decoder 路径（相对执行根，apt_flat_env.py:207-209 同默认值）
    onnx_path: str = "gear_sonic_deploy/policy/release/model_decoder.onnx"
    # 官方 g1-mode token 统计 npz（mean/std 键；apt_flat_env.py:276,616-625）
    token_stats: str = ""
    # E49 token 仿射系数与限幅（apt_flat_env.py:275,284,277）
    token_alpha: float = 1.0
    token_bound: str = "none"  # "none"（无界）| "tanh"（mean ± alpha*std 限幅）
    token_std_floor: float = 1e-3
