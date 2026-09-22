"""D067 G2 输出端并行残差臂训练入口（三臂对照 + max-forward 奖励）。

预注册 = refine-logs/ds/DS_CONTINUOUS_EXECUTION_PLAN.md §5y（:496-506）。
立项口径（owner 2026-09-22 拍板第 4 项 + 目标改写）：
    a_t = decoder(token_t) + r(s_t)
  - token_t = author v0 ckpt（D061a）推理 token（--author-ckpt）；**推理期 ctx 命令槽
    钉最大速度档**（--intent-pin-max 默认开，假设 A2）。
  - decoder = 冻结 SONIC ONNX decoder（同 D064/D065）；其输入 = 994 维
    `[token(64), 本体历史(930)]`（sonic_action_term.py:24-34 通道契约）。
  - r = 轻量 MLP [256,128]+SiLU 输出 29 维关节目标残差，**末层零初始化** ⇒ 构造期
    与 decoder 臂输出逐位一致（残差恒等 0，不破坏 token 先验）。
  - PPO 只训 r（decoder/token 源全程冻结，梯度只达 r；critic 属值函数必要可训项，
    同 D065 adapter-only 口径，假设 A4）。
奖励（2026-09-22 owner 目标改写）：residual 臂默认 `--reward-mode max_forward`
  = 机体坐标前进速度（root_lin_vel_b[:, 0]）正向奖励 + 官方配方防摔/姿态/动作平滑正则，
  **无人工速度上限**，移除速度命令跟踪项（track_lin_vel_xy_exp / track_ang_vel_z_exp）。
  direct/decoder 臂默认保持 D064 原命令跟踪口径不动（保 D064 可比性）。

三臂（--arm direct|decoder|residual）：
  direct   : env 动作 = 官方 JointPositionAction（29 维），reward = 命令跟踪（D064 逐字）。
  decoder  : env 动作 = 冻结 SONIC decoder ActionTerm（64 维 token），reward = 命令跟踪（D064 逐字）。
  residual : env 动作 = direct 动作通路逐字（29 维，经 sibling `action="lora_policy"` 变体
             = direct 通路 + policy 组末位 930 维 decoder 本体历史观测）；policy obs 另追加
             64 维 `author_token`（env 侧冻结 author adapter 产出，观测常量，无梯度）；
             policy μ = 冻结 decoder(cat[token, proprio]) + r(官方 policy obs)，
             reward 默认 max_forward。

为什么 decoder 必须 in-graph（不在 env 侧 ActionTerm）：D065 v1 已实测——env 侧
  ActionTerm 的输出对 PPO 的 log_prob/熵无梯度（全冻结 policy 时 loss.requires_grad=False，
  backward 直接 RuntimeError），残差会退化成「逐位等于 decoder 臂」的假阴性。故残差臂与
  D065 C 臂同形态：decoder 在 policy 图内，μ 经分布头，梯度经 log N(a; μ, σ) 直达 r
  （见 train_g1_decoder_lora.py 模块 docstring 的完整论证）。

与 D064/D065 的关系（最小侵入，不改既有默认行为）：
  - 本文件是**新增入口**，不修改 train_g1_decoder.py（D064）任何默认值/行为。
  - env cfg 一律经 sibling `g1_velocity_decoder_env.make_env_cfg` 工厂获取（残差臂用既有
    `action="lora_policy"` 变体，即 direct 动作通路 + 930 维本体观测）；奖励/obs 需求全部
    在**本文件内**做 cfg 覆写（参考 eval_author_v0_decoder.py 的 cmd_cfg 覆写先例）；
    不改 g1_velocity_decoder_env.py，不动 sonic_action_term.py。
  - PPO 配方逐字段继承 D064 官方 G1 rough（保三臂可比性）。

分层（同 sonic_action_term.py / eval_author_v0_decoder.py 惯例）：
  Layer 1（纯 torch/numpy，零 isaaclab/onnx/rsl_rl 依赖，本机 Windows 可 import 可测）
    - 残差 MLP `ResidualActionMLP`（末层零初始化）
    - μ 路径 `ResidualMuPath`（冻结 decoder(decoder_obs) + r(r_obs)）
    - 奖励纯数学 `forward_vel_reward` / `max_forward_total_reward` + 配方表 `MAX_FORWARD_RECIPE`
    - 三臂配置表 `ARM_TABLE`、身份信封 `build_identity_envelope`
    - 契约件：`CTX_TERRAIN_MAP`/`ctx_terrain`（地形映射）、`forward_token_affine`（token
      前向仿射）、`load_token_stats`、`official_obs_slice`（r 输入切片守卫）、
      `assemble_decoder_obs`（decoder 输入组装）
    - `run_selftest()`：零初始化恒等 / 梯度隔离 / 奖励数学 / 三臂表 / 身份信封 /
      **前向仿射往返** / **residual vs decoder 臂 decoder 输入逐位一致** / 地形契约 /
      切片守卫
  Layer 2（isaac，import 集中在函数内；本机不 import）
    - `_import_heavy`（复用 train_g1_decoder._import_heavy，单份组装防漂移）
    - env cfg 构建 + max_forward 奖励覆写 + author_token 观测项注入
    - `ResidualDecoderPolicy` + rsl_rl 类名注册
    - runner 训练回路

用法（服务器 cvgl，cwd=仓根）：
    # 本机 CPU 自测（不 import isaaclab）
    python apt_g1/isaac/train_g1_decoder_residual.py --selftest     # -> D067_SELFTEST_PASS

    # 残差臂训练（author v0 为 token 源）
    python apt_g1/isaac/train_g1_decoder_residual.py --arm residual \\
        --terrain rough --num-envs 4096 --max-iterations 3000 \\
        --author-ckpt <d061a/ckpt_final.pt> --token-stats <g1_token_stats.npz>

    # direct / decoder 对照臂（reward 保持 D064 命令跟踪）
    python apt_g1/isaac/train_g1_decoder_residual.py --arm decoder \\
        --terrain rough --token-stats <g1_token_stats.npz>

契约映射（reviewer 验收 blocker 修复，2026-09-22）：
  M1 terrain 契约：CLI 地形名（plane/rough/stairs/stones/discrete，本文件 TERRAINS）
     **不得直传** `ctx_from_command`（后者只接受 eval_author_v0_decoder.TERRAIN_TYPES =
     plane/rough_paper/climbing_box，:193）。显式映射表 `CTX_TERRAIN_MAP`：
       plane -> plane；rough/stairs/stones/discrete -> rough_paper
     （本仓无 climbing_box env 地形，故不映射到 climbing_box；如需 climbing_box 语境，
     要另建对应 env 地形，不是本表的事。）映射在 `_token_obs_params`/`residual_token_obs`
     两处接入（`ctx_terrain()`，幂等：已是 ctx 地形则原样返回），并进身份信封
     `ctx_terrain`/`ctx_terrain_map` 字段。
  M2 token 前向仿射：`residual_token_obs` 里 `adapter.step` 返回的是**归一化动作 a**
     （= (tok-mean)/(alpha*std)，eval_author_v0_decoder.AuthorPolicyAdapter.step:638-642），
     不是 token 坐标；喂 decoder 分支前必须补**前向仿射**
       token = mean + alpha*std*tanh(a)（token_bound=="tanh"）/ mean + alpha*std*a（"none"）
     （权威式：sonic_action_term.py:205-215 map_token、sonic_lora_policy.py:157-161
     token_from_raw；与 adapter 的逆仿射互逆）。mean/std 与 decoder 分支**同一份
     `--token-stats` npz**（同一 std 下限 TOKEN_STD_FLOOR，与 author_token_stats /
     sonic_action_term.token_std_floor 同值）显式传入 `residual_token_obs` 并在运行期与
     adapter 内部统计做逐位一致性断言；alpha/bound 与 `_token_obs_params` 现值一致。
     该仿射身份进信封 `token_affine` 字段。
  M3 特权 critic（SF-1，**有意设计，不改行为**）：`author_token` 同时追加进 policy 与
     critic 两组，即 critic 可见的观测比 actor 多（特权 critic 不对称）；信封
     `privileged_critic` 字段声明。
  M4 plane 档奖励口径（SF-4）：max_forward 配方对**所有地形统一**施加官方 G1 rough
     正则（含 plane），与官方 flat 语义 `_apply_flat_semantics` 的正则**不同源** ⇒
     plane 档与官方 flat 的对照**非严格单变量**；信封 `reward_recipe_scope` 字段声明。

显式假设（指令含糊处的解释，均落成常量/打印，便于事后改判）：
  A1 token 经 policy obs 传递：author adapter 是**有状态自回归**件，若在 policy.forward
     里运行，PPO 的 evaluate 阶段会重跑并污染滚动窗 ⇒ 必须 env 侧每控制步算一次、把
     64 维 token 作为观测常量喂给 policy（`author_token` obs 项，本文件内新增）。
     decoder 是无状态 MLP，重跑安全。
  A2 intent-pin 值：`--intent-pin-value` 缺省 1.0（= 官方 G1 rough 命令 lin_vel_x 上界，
     rough_env_cfg.py:146-148 的 (0,1) 上界）。若语料速度带另有最大档，用
     `--intent-pin-value` 显式覆盖。
  A3 残差臂动作语义：env 侧 = direct 臂 JointPositionAction（scale=0.5、
     use_default_offset=True），policy μ（29 维）= 冻结 decoder 归一化输出 + r；与 D065 C 臂
     同口径（decoder 输出按归一化关节目标进分布，env 侧再 affine）。
  A4 critic 可训：PPO 值函数必须有梯度路径（同 D065 adapter-only），故 critic 与 r 可训，
     decoder/actor 占位/σ 冻结。
  A5 r 的输入 = 官方 policy obs 切片（`total - 930(proprio) - 64(token)`），即原 D064
     policy 八项（含 height_scan）——保持残差看到与 direct/decoder 臂相同的状态口径。
     **SF-2**：该「前缀切片」不再靠隐式假设——`_build_residual_policy_kwargs` 经
     `official_obs_slice` 按运行期 term 序推出宽度，并断言 `sonic_proprio`/`author_token`
     两 term 全部落在 official 段之后（起始=0 前缀契约），对不上即 fail loud。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # 以 `python apt_g1/isaac/train_g1_decoder_residual.py` 直跑时 sys.path[0] 是脚本
    # 目录而非仓根（train_g1_decoder.py:531-534 同款处理）。
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Layer 1: 契约常量 + 纯 torch 件（零 isaaclab / onnx / rsl_rl 依赖）
# ---------------------------------------------------------------------------
ARMS = ("direct", "decoder", "residual")
TERRAINS = ("plane", "rough", "stairs", "stones", "discrete")
SMOKE_NUM_ENVS = 8
SMOKE_ITERS = 2

TOKEN_DIM = 64
ACTION_DIM = 29              # 关节目标维（G1 29 DOF，SONIC/IsaacLab 序）
HISTORY_LEN = 10
PROPRIO_DIM = HISTORY_LEN * 3 + 3 * HISTORY_LEN * ACTION_DIM + HISTORY_LEN * 3  # 930
DECODER_OBS_DIM = TOKEN_DIM + PROPRIO_DIM  # 994（sonic_action_term.py:69-74）
RESIDUAL_HIDDEN = (256, 128)
RESIDUAL_ACTIVATION = "silu"

REWARD_MODES = ("auto", "track", "max_forward")
DEFAULT_REWARD_MODE = "auto"   # auto: residual -> max_forward；direct/decoder -> track
DEFAULT_FORWARD_WEIGHT = 1.0
DEFAULT_INTENT_PIN_VALUE = 1.0  # 官方 G1 rough 命令 lin_vel_x 上界（假设 A2）

# env 地形名 -> author ctx 地形名（ctx_from_command 只认 plane/rough_paper/climbing_box，
# 见 eval_author_v0_decoder.TERRAIN_TYPES:193）。本仓无 climbing_box env 地形，故不映射到
# 该档；rough/stairs/stones/discrete 全部归到语料里唯一的非平坦档 rough_paper。
# 未列出的地形名 -> `ctx_terrain` 显式报错（不静默回落，防新地形悄悄用错语境）。
CTX_TERRAIN_MAP = {
    "plane": "plane",
    "rough": "rough_paper",
    "stairs": "rough_paper",
    "stones": "rough_paper",
    "discrete": "rough_paper",
}
# token 统计 std 下限（与 eval_author_v0_decoder.author_token_stats floor 与
# sonic_action_term.SonicTokenCfg.token_std_floor 同值，保证两份统计逐位一致）。
TOKEN_STD_FLOOR = 1.0e-3

# max_forward 奖励配方（唯一事实源：Layer 2 覆写与 selftest 同读此表）。
# 正则项权重逐字取自官方 G1 rough 配方（tmp/isaac_ref/rough_env_cfg.py:20-100 +
# velocity_env_cfg.py:222-254 的 G1 post_init 覆写，= D064 继承的同一批正则）。
# 注意：本配方**对所有地形统一施加**（含 plane）——官方 flat 语义
# （_apply_flat_semantics）会改若干正则权重，但 max_forward 下三臂/跨地形用同一套
# 正则更利于归因（§5y「官方配方防摔/姿态/动作平滑正则」未按地形分档）。
MAX_FORWARD_RECIPE = {
    # 前进速度项（唯一正向任务项；无人工上限，见 forward_vel_reward）
    "forward_vel": {"weight": DEFAULT_FORWARD_WEIGHT, "unit": "m/s", "kind": "task"},
    # 官方防摔/姿态/动作平滑正则（sign: -1=罚，+1=塑形，0=关闭）
    "termination_penalty": {"weight": -200.0, "sign": -1, "kind": "safety"},
    "flat_orientation_l2": {"weight": -1.0, "sign": -1, "kind": "posture"},
    "lin_vel_z_l2": {"weight": 0.0, "sign": 0, "kind": "posture"},
    "ang_vel_xy_l2": {"weight": -0.05, "sign": -1, "kind": "posture"},
    "dof_torques_l2": {"weight": -1.5e-7, "sign": -1, "kind": "smooth"},
    "dof_acc_l2": {"weight": -1.25e-7, "sign": -1, "kind": "smooth"},
    "action_rate_l2": {"weight": -0.005, "sign": -1, "kind": "smooth"},
    "feet_air_time": {"weight": 0.25, "sign": 1, "kind": "gait"},
    "feet_slide": {"weight": -0.1, "sign": -1, "kind": "smooth"},
    "dof_pos_limits": {"weight": -1.0, "sign": -1, "kind": "safety"},
    "joint_deviation_hip": {"weight": -0.1, "sign": -1, "kind": "posture"},
    "joint_deviation_arms": {"weight": -0.1, "sign": -1, "kind": "posture"},
    "joint_deviation_torso": {"weight": -0.1, "sign": -1, "kind": "posture"},
}
# 移除的命令跟踪项（§5y：命令跟踪项移除）
MAX_FORWARD_REMOVED_TERMS = ("track_lin_vel_xy_exp", "track_ang_vel_z_exp")

# 三臂配置表（唯一事实源：CLI 校验 / 训练编排 / 身份信封 / selftest 同读）。
# env_action="lora_policy" 的 env 侧动作通路 = direct 臂逐字（JointPositionAction，
# g1_velocity_decoder_env.py:378-391），仅 policy obs 组末位多 930 维本体历史。
ARM_TABLE = {
    "direct": {
        "env_action": "direct",
        "action_dim": ACTION_DIM,
        "reward_mode": "track",
        "policy": "rsl_rl ActorCritic（D064 逐字）",
        "frozen_branch": False,
        "token_source": None,
    },
    "decoder": {
        "env_action": "decoder",
        "action_dim": TOKEN_DIM,
        "reward_mode": "track",
        "policy": "rsl_rl ActorCritic（D064 逐字）",
        "frozen_branch": False,
        "token_source": None,
    },
    "residual": {
        "env_action": "lora_policy",   # = direct 动作通路逐字 + 930 维本体观测
        "action_dim": ACTION_DIM,
        "reward_mode": "max_forward",
        "policy": "ResidualDecoderPolicy（decoder 分支冻结 + r 可训）",
        "frozen_branch": True,
        "token_source": "author_v0_ckpt",
    },
}


def resolve_reward_mode(arm: str, requested: str) -> str:
    """--reward-mode 解析：auto 时 residual->max_forward、direct/decoder->track。"""
    if requested not in REWARD_MODES:
        raise ValueError(f"reward_mode 须 ∈ {REWARD_MODES}，收到 {requested!r}")
    if requested != "auto":
        return requested
    return ARM_TABLE[arm]["reward_mode"]


# ------------------------------------------------- token 前向仿射 / terrain 契约
def forward_token_affine(
    raw: torch.Tensor,
    token_mean: torch.Tensor,
    token_std: torch.Tensor,
    token_alpha: float = 1.0,
    token_bound: str = "tanh",
) -> torch.Tensor:
    """归一化动作 a -> token 坐标（与 author adapter 的逆仿射互逆）。

    逐式同 `sonic_action_term.SonicActionCore.map_token`（:205-215）与
    `sonic_lora_policy.SonicLoRAMuPath.token_from_raw`（:157-161）：
        token_bound == "tanh": token = mean + alpha*std*tanh(a)
        否则                 : token = mean + alpha*std*a

    **为什么必须补这一步**：`AuthorPolicyAdapter.step` 返回的是归一化动作
    a = (tok-mean)/(alpha*std)（eval_author_v0_decoder.py:638-642），不是 token 坐标。
    把 a 直喂 decoder 分支的 FSQ 量化会破坏冻结 decoder 先验（reviewer 实测
    max|a-tok|=3.51）。
    """
    m = torch.as_tensor(token_mean, dtype=torch.float32, device=raw.device).reshape(-1)
    s = torch.as_tensor(token_std, dtype=torch.float32, device=raw.device).reshape(-1)
    if int(m.numel()) != TOKEN_DIM or int(s.numel()) != TOKEN_DIM:
        raise ValueError(f"token_mean/std 须为 ({TOKEN_DIM},)，收到 {int(m.numel())}/{int(s.numel())}")
    bound = str(token_bound)
    if bound not in ("none", "tanh"):
        raise ValueError(f"token_bound 须 ∈ ('none','tanh')，收到 {bound!r}")
    a = raw if bound != "tanh" else torch.tanh(raw)
    return m + float(token_alpha) * s * a


def ctx_terrain(terrain: str) -> str:
    """env 地形名 -> `ctx_from_command` 接受的地形名（显式映射，幂等）。

    映射表 = CTX_TERRAIN_MAP。已是 ctx 契约内地形名（eval_author_v0_decoder.
    TERRAIN_TYPES）时原样返回（幂等，便于在 `_token_obs_params` 与 `residual_token_obs`
    两处无脑接入）。未列出且不在 ctx 契约内的地形名 -> 显式 ValueError（不静默回落）。
    """
    name = str(terrain)
    if name in CTX_TERRAIN_MAP:
        return CTX_TERRAIN_MAP[name]
    if name in ("plane", "rough_paper", "climbing_box"):
        return name
    raise ValueError(
        f"地形 {name!r} 无 ctx 映射（CLI 允许 {TERRAINS}；ctx 契约允许 "
        f"plane/rough_paper/climbing_box，见 CTX_TERRAIN_MAP）"
    )


def official_obs_slice(
    policy_terms,
    term_dims,
    excluded: tuple[str, ...] = ("sonic_proprio", "author_token"),
) -> dict:
    """r 输入（official 段）的切片守卫（SF-2，纯函数，可本机单测）。

    契约：r 的输入 = policy obs 的**前缀** `flat[:, :official_obs_dim]`（假设 A5），
    其中 `official_obs_dim` = 运行期 policy 组总宽 − 被排除 term（默认
    `sonic_proprio`(930) + `author_token`(64)）的宽度之和。

    校验（任一条不满足即 RuntimeError，fail loud）：
      1. `policy_terms` 与 `term_dims` 一一对应；
      2. 被排除的 term 全部存在（缺位 = 接线失败，不能靠"宽度恰好凑上"蒙过去）；
      3. 被排除 term 的区间**全部落在 official 段之后**（即 official 段是前缀，
         起始 = 0）——否则 `flat[:, :official_obs_dim]` 会切进 proprio/token，
         与 `decoder_obs` 的 token/proprio 切片重叠，静默错喂。
    返回 {"start": 0, "width": official_width, "excluded": {term: width}, "policy_terms": [...]}.
    """
    names = [str(n) for n in policy_terms]
    if len(names) != len(term_dims):
        raise RuntimeError(f"policy 组 names/dims 错位：{len(names)} vs {len(term_dims)}")
    dims = [int(math.prod(int(x) for x in d)) for d in term_dims]
    total = int(sum(dims))
    spans: dict[str, tuple[int, int]] = {}
    pos = 0
    for n, w in zip(names, dims):
        spans[n] = (pos, pos + w)
        pos += w
    excl: dict[str, int] = {}
    for term in excluded:
        if term not in spans:
            raise RuntimeError(f"policy 组缺被排除 term {term!r}（现有 {names}）——r 输入切片契约不成立")
        excl[term] = int(spans[term][1] - spans[term][0])
    official_width = int(total - sum(excl.values()))
    if official_width <= 0:
        raise RuntimeError(f"official 段宽度推得 {official_width}（policy 总宽 {total}，排除 {excl}）")
    for term in excluded:
        start, _end = spans[term]
        if start < official_width:
            raise RuntimeError(
                f"{term!r} 区间起点 {start} < official 段宽度 {official_width}——"
                f"official 段不是 policy obs 前缀，r 输入切片会错喂（现有序 {names}）"
            )
    return {"start": 0, "width": official_width, "excluded": excl, "policy_terms": names}


def assemble_decoder_obs(
    flat: torch.Tensor,
    token_slice: tuple[int, int],
    proprio_slice: tuple[int, int],
) -> torch.Tensor:
    """policy obs -> (N,994) = cat[token(64), proprio(930)]（纯函数，供本机跨口径对照）。

    与 `ResidualDecoderPolicy.decoder_obs` 同式；抽成模块级纯函数是为了让本机 selftest
    能在无 rsl_rl 的环境下做「residual 臂 decoder 输入 vs decoder 臂 decoder 输入」的
    逐位对照（reviewer 要求的新增断言 ②）。
    """
    ts, tw = int(token_slice[0]), int(token_slice[1])
    ps, pw = int(proprio_slice[0]), int(proprio_slice[1])
    if tw != TOKEN_DIM or pw != PROPRIO_DIM:
        raise ValueError(f"token/proprio 宽度须为 {TOKEN_DIM}/{PROPRIO_DIM}，收到 {tw}/{pw}")
    if ts + tw > int(flat.shape[-1]) or ps + pw > int(flat.shape[-1]):
        raise ValueError(f"切片越界：obs 宽 {int(flat.shape[-1])}，token=({ts},{tw}) proprio=({ps},{pw})")
    return torch.cat([flat[:, ts : ts + tw], flat[:, ps : ps + pw]], dim=-1)


# ------------------------------------------------------------- 残差 MLP
def _make_activation(name: str) -> nn.Module:
    """激活名 -> 模块（本文件只用 silu；其余为显式可选项）。"""
    key = str(name).lower()
    if key in ("silu", "swish"):
        return nn.SiLU()
    if key == "relu":
        return nn.ReLU()
    if key == "elu":
        return nn.ELU()
    raise ValueError(f"未知激活 {name!r}（支持 silu/relu/elu）")


class ResidualActionMLP(nn.Module):
    """r(s_t)：轻量 MLP [256,128]+SiLU -> 29 维关节目标残差，**末层零初始化**。

    末层（weight 与 bias）全零 ⇒ r(s) ≡ 0（任意 s），故 a_t = decoder(...) + r(s_t)
    在构造期与 decoder 臂输出**逐位一致**（零初始化恒等，selftest 断言）。PPO 训练中
    末层从零开始学习，r 的初始贡献为 0，不破坏冻结 token 先验（§5y 设计）。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = ACTION_DIM,
        hidden: tuple[int, ...] = RESIDUAL_HIDDEN,
        activation: str = RESIDUAL_ACTIVATION,
    ) -> None:
        super().__init__()
        if int(obs_dim) <= 0:
            raise ValueError(f"obs_dim 须 > 0，收到 {obs_dim}")
        if int(action_dim) != ACTION_DIM:
            raise ValueError(f"残差输出维冻结为 {ACTION_DIM}，收到 {action_dim}")
        layers: list[nn.Module] = []
        last = int(obs_dim)
        for h in hidden:
            layers += [nn.Linear(last, int(h)), _make_activation(activation)]
            last = int(h)
        head = nn.Linear(last, int(action_dim))
        layers.append(head)
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(int(h) for h in hidden)
        self.activation = str(activation)
        # 末层零初始化（恒等起点）；其余层默认初始化由 nn.Linear 提供
        with torch.no_grad():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def is_zero(self, obs: torch.Tensor, atol: float = 0.0) -> bool:
        """r(obs) 是否恒为 0（零初始化自检用；atol=0 = 逐位）。"""
        with torch.no_grad():
            out = self.forward(obs)
        return bool((out.abs() <= atol).all())


def freeze_module(module: nn.Module) -> dict:
    """把模块整体冻结（requires_grad=False + eval()），返回参数计数报告。

    这是「decoder/token 源全冻结，梯度只达 r」的机制：冻结参数仍在模块里（可 save/
    load/诊断），但不进优化器（train 侧按 requires_grad 摘除，同 D065 口径）。
    """
    n_tensors = n_elements = 0
    for p in module.parameters():
        n_tensors += 1
        n_elements += int(p.numel())
        p.requires_grad_(False)
    module.eval()
    return {"n_tensors": n_tensors, "n_frozen": n_tensors, "n_elements": n_elements}


class ResidualMuPath(nn.Module):
    """μ = decoder(decoder_obs) + r(r_obs)（decoder 冻结分支 + 可训残差）。

    - `frozen_branch(decoder_obs)`：`torch.no_grad()` 下跑 decoder（冻结先验不建图）。
    - `forward(decoder_obs, r_obs)`：冻结分支输出 + 残差 MLP 输出（29 维）。

    梯度隔离（selftest 断言）：冻结后只有 `residual` 参数 requires_grad=True，
    μ 反传只在 residual 上产生 grad，decoder 参数 grad 恒 None。
    """

    def __init__(
        self,
        decoder: nn.Module,
        residual: ResidualActionMLP,
        *,
        freeze_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.residual = residual
        self.decoder_frozen = False
        if freeze_decoder:
            freeze_module(self.decoder)
            self.decoder_frozen = True

    def frozen_branch(self, decoder_obs: torch.Tensor) -> torch.Tensor:
        """冻结 decoder(decoder_obs) -> (N, 29)（no_grad，常量分支）。"""
        with torch.no_grad():
            return self.decoder(decoder_obs)

    def forward(self, decoder_obs: torch.Tensor, r_obs: torch.Tensor) -> torch.Tensor:
        return self.frozen_branch(decoder_obs) + self.residual(r_obs)

    def trainable_report(self) -> dict:
        def _count(mod: nn.Module) -> dict:
            ps = list(mod.parameters())
            return {
                "n_tensors": len(ps),
                "n_trainable": sum(1 for p in ps if p.requires_grad),
                "n_elements": int(sum(p.numel() for p in ps)),
                "n_trainable_elements": int(sum(p.numel() for p in ps if p.requires_grad)),
            }

        return {
            "decoder_frozen": bool(self.decoder_frozen),
            "decoder": _count(self.decoder),
            "residual": _count(self.residual),
        }

    def only_residual_trainable(self) -> bool:
        """除 residual 外无任何 requires_grad 参数（梯度隔离判据）。"""
        if any(p.requires_grad for p in self.decoder.parameters()):
            return False
        return any(p.requires_grad for p in self.residual.parameters())


# ------------------------------------------------------------- 奖励纯数学
def forward_vel_reward(
    vx_b: torch.Tensor, weight: float = DEFAULT_FORWARD_WEIGHT, positive_only: bool = True
) -> torch.Tensor:
    """机体坐标前进速度奖励（root_lin_vel_b[:, 0]）。

    - `positive_only=True`（默认）：`clamp(vx_b, min=0)` ⇒ 只奖励前进、不倒奖后退；
    - **无人工速度上限**：与 D048j `progress_bonus` 的 `clamp(vx, 0, 1)`（封顶 1 m/s）
      不同，本项不设上界（owner 2026-09-22 目标改写：无人工限速）。
    - `weight` 由调用侧传入（isaaclab RewardManager 亦会再乘 RewTerm.weight；此处
      纯数学版自含 weight 便于单测与直接求和）。
    """
    v = torch.as_tensor(vx_b, dtype=torch.float32)
    if positive_only:
        v = torch.clamp(v, min=0.0)
    return float(weight) * v


def max_forward_total_reward(
    vx_b: torch.Tensor,
    penalties: list[tuple[float, torch.Tensor]] | None = None,
    *,
    weight_forward: float = DEFAULT_FORWARD_WEIGHT,
    positive_only: bool = True,
) -> torch.Tensor:
    """前进项 + 正则项求和（`penalties` = [(weight, value), ...]，符号已含在 weight 内）。

    纯数学组合式，供 selftest 校验量纲/符号；Layer 2 的实际奖励由 isaaclab
    RewardManager 按 MAX_FORWARD_RECIPE 逐项加权求和（本函数与其同口径）。
    """
    out = forward_vel_reward(vx_b, weight_forward, positive_only)
    for w, v in (penalties or []):
        out = out + float(w) * torch.as_tensor(v, dtype=torch.float32)
    return out


# ------------------------------------------------------------- 身份信封
def _file_md5(path: str | os.PathLike) -> str | None:
    """文件 md5（复用 eval_g1_decoder._file_md5，避免双份实现漂移）。"""
    try:
        from isaac import eval_g1_decoder as egd  # noqa: PLC0415
    except ImportError:  # pragma: no cover（本机走此支）
        from apt_g1.isaac import eval_g1_decoder as egd
    return egd._file_md5(str(path))


def _git_head() -> str:
    try:
        from isaac import eval_g1_decoder as egd  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from apt_g1.isaac import eval_g1_decoder as egd
    return egd._git_head()


def _env_module_md5() -> dict:
    """冻结件/兄弟模块的实存文件 md5（身份信封；改动可追溯）。"""
    base = REPO_ROOT / "apt_g1" / "isaac"
    return {
        "train_g1_decoder_residual.py": _file_md5(Path(__file__).resolve()),
        "g1_velocity_decoder_env.py": _file_md5(base / "g1_velocity_decoder_env.py"),
        "sonic_action_term.py": _file_md5(base / "sonic_action_term.py"),
        "eval_author_v0_decoder.py": _file_md5(base / "eval_author_v0_decoder.py"),
        "train_g1_decoder.py": _file_md5(base / "train_g1_decoder.py"),
    }


def build_identity_envelope(
    arm: str,
    reward_mode: str,
    *,
    author_ckpt: str = "",
    onnx_path: str = "",
    token_stats: str = "",
    intent_pin: dict | None = None,
    terrain: str = "",
    ctx_terrain_name: str = "",
    extra: dict | None = None,
) -> dict:
    """身份信封：script/git + 冻结件 md5 + 三臂表项 + 奖励配方 + intent-pin + 契约声明。

    纯本机可构建（只做文件 md5 / git 调用，不 import isaaclab）。

    契约字段（reviewer blocker/should-fix 修复）：
      - `ctx_terrain`/`ctx_terrain_map`：env 地形名 -> author ctx 地形名（M1）；
      - `token_affine`：前向仿射式与 bound/alpha（M2）；
      - `privileged_critic`：critic 多看 token 的不对称声明（M3/SF-1）；
      - `reward_recipe_scope`：max_forward 对 plane 统一施加与官方 flat 不同源（M4/SF-4）。
    """
    if arm not in ARM_TABLE:
        raise ValueError(f"未知 arm {arm!r}（允许 {ARMS}）")
    env = {
        "format": "d067.residual.v2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "entry": "apt_g1/isaac/train_g1_decoder_residual.py",
        "script": str(Path(__file__).resolve()),
        "script_md5": _file_md5(Path(__file__).resolve()),
        "git_head": _git_head(),
        "arm": arm,
        "arm_table": dict(ARM_TABLE[arm]),
        "reward_mode": reward_mode,
        "reward_recipe": {k: dict(v) for k, v in MAX_FORWARD_RECIPE.items()}
        if reward_mode == "max_forward"
        else {"mode": "track", "note": "D064 命令跟踪口径逐字（direct/decoder 臂）"},
        "reward_removed_terms": list(MAX_FORWARD_REMOVED_TERMS) if reward_mode == "max_forward" else [],
        # SF-4：max_forward 正则对所有地形（含 plane）统一施加官方 G1 rough 配方，
        # 与官方 flat 语义 _apply_flat_semantics 的正则**不同源** ⇒ plane 档对照非严格单变量。
        "reward_recipe_scope": {
            "applied_to": "all_terrains",
            "regularizer_source": "official_g1_rough",
            "note": (
                "plane 档对照非严格单变量：官方 flat 语义(_apply_flat_semantics)会改若干正则权重，"
                "max_forward 对 plane 仍施加 rough 配方（§5y 未按地形分档）"
            ),
        },
        "residual": {
            "hidden": list(RESIDUAL_HIDDEN),
            "activation": RESIDUAL_ACTIVATION,
            "last_layer_zero_init": True,
            "obs_dim_note": "r 的输入 = 官方 policy obs 切片（total - 930 proprio - 64 token）",
        },
        "decoder_input": {"token_dim": TOKEN_DIM, "proprio_dim": PROPRIO_DIM, "obs_dim": DECODER_OBS_DIM},
        "token_source": {
            "author_ckpt": str(author_ckpt) if author_ckpt else "",
            "ckpt_md5": _file_md5(author_ckpt) if author_ckpt and os.path.isfile(str(author_ckpt)) else None,
        },
        # M1：地形契约映射（env 地形名 -> ctx_from_command 接受名）
        "ctx_terrain": str(ctx_terrain_name or (ctx_terrain(terrain) if terrain else "")),
        "env_terrain": str(terrain),
        "ctx_terrain_map": dict(CTX_TERRAIN_MAP),
        # M2：前向仿射声明（token = mean + alpha*std*(tanh(a)|a)）
        "token_affine": {
            "formula": "token = mean + alpha*std*(tanh(a) if token_bound=='tanh' else a)",
            "token_bound": "cli --token-bound（缺省 tanh）",
            "token_alpha": "cli --token-alpha（缺省 1.0）",
            "token_stats": str(token_stats),
            "stats_source": "与 decoder 分支同一份 --token-stats npz（load_token_stats, floor=1e-3）",
            "authoritative": ["sonic_action_term.py:205-215", "sonic_lora_policy.py:157-161"],
        },
        # M3/SF-1：特权 critic 不对称声明
        "privileged_critic": {
            "author_token_in_critic": True,
            "note": "author_token 同时进 policy 与 critic 两组：critic 可比 actor 多看该 token（有意设计，不改行为）",
        },
        "decoder": {
            "onnx_path": str(onnx_path) if onnx_path else "factory-default(make_env_cfg, SONIC_DECODER_ONNX)",
            "onnx_md5": _file_md5(onnx_path) if onnx_path and os.path.isfile(str(onnx_path)) else None,
        },
        "token_stats": {
            "path": str(token_stats) if token_stats else "",
            "md5": _file_md5(token_stats) if token_stats and os.path.isfile(str(token_stats)) else None,
        },
        "intent_pin": dict(intent_pin) if intent_pin else None,
        "env_module_md5": _env_module_md5(),
    }
    if extra:
        env.update(extra)
    return env


# ------------------------------------------------------------- selftest
def _check(cond: bool, msg: str) -> None:
    """自测断言：不满足即 RuntimeError（带原因，避免裸 assert 被 -O 剥掉）。"""
    if not cond:
        raise RuntimeError(f"selftest check failed: {msg}")


class _MockDecoder(nn.Module):
    """本机替身 decoder（(N,994)->(N,29)）：确定性线性映射，模拟冻结 ONNX MLP。"""

    def __init__(self, in_dim: int = DECODER_OBS_DIM, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, action_dim)
        with torch.no_grad():
            # 非平凡权重，保证输出非零（恒等性断言需要 decoder 输出 != 0）
            self.lin.weight.uniform_(-0.5, 0.5)
            self.lin.bias.uniform_(-0.1, 0.1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.lin(obs)


def selftest_zero_init_identity(seed: int = 0) -> dict:
    """零初始化恒等性：mock decoder 输出 + 零 r -> μ 与 decoder 输出逐位一致。"""
    g = torch.Generator().manual_seed(seed)
    n, r_obs_dim = 8, 321
    decoder = _MockDecoder()
    residual = ResidualActionMLP(r_obs_dim)
    mu_path = ResidualMuPath(decoder, residual, freeze_decoder=True)

    decoder_obs = torch.randn(n, DECODER_OBS_DIM, generator=g)
    r_obs = torch.randn(n, r_obs_dim, generator=g)

    # 1) 残差恒为零（逐位，任意 r_obs）
    _check(residual.is_zero(r_obs, atol=0.0), "末层零初始化后 r(r_obs) 非零")
    _check(bool((residual(r_obs) == 0).all()), "r(r_obs) 非逐位零")

    # 2) μ == decoder(decoder_obs) 逐位
    with torch.no_grad():
        q = decoder(decoder_obs)
        mu = mu_path(decoder_obs, r_obs)
    _check(torch.equal(mu, q), f"μ 与 decoder 输出非逐位一致（maxdiff={float((mu - q).abs().max())}）")
    _check(bool(torch.isfinite(mu).all()), "μ 含非有限值")

    # 3) 反向验证：把 r 末层设为非零后 μ 应偏离（防「恒等」被静默成因其他原因）
    with torch.no_grad():
        head = residual.net[-1]
        head.weight.fill_(0.01)
    mu2 = mu_path(decoder_obs, r_obs)
    _check(not torch.equal(mu2, q), "非零 r 下 μ 仍与 decoder 逐位一致（恒等断言失效）")

    return {"residual_is_zero": True, "mu_bitwise_equal_decoder": True, "shape": list(mu.shape)}


def selftest_token_affine_roundtrip() -> dict:
    """新增断言 ①（M2）：前向仿射 a -> tok -> a' 往返（逐位）。

    构造：mean=0、std=2 的整数次幂 ⇒ `mean + std*a` 与 `(tok-mean)/std` 都是精确
    float32 运算（二进制可表示），故逐位往返成立。这直接验证 `forward_token_affine`
    与 author adapter 的逆仿射（eval_author_v0_decoder.py:638-642）**互逆**——即
    `residual_token_obs` 补的前向仿射确实把 a 还原回 token 坐标（reviewer BLOCKER-2）。
    同时做反向验证：不做前向仿射（直用 a）时与 token 坐标显著不同（复现
    reviewer 的 max|a-tok| 现象，防「往返」因平凡原因通过）。
    """
    g = torch.Generator().manual_seed(11)
    n = 4096
    token_mean = torch.zeros(TOKEN_DIM, dtype=torch.float32)
    token_std = torch.pow(torch.tensor(2.0), (torch.arange(TOKEN_DIM) % 7 - 3).float())
    alpha = 1.0

    for bound in ("none", "tanh"):
        if bound == "none":
            # a 任意 -> tok = std*a -> a' = tok/std（精确往返）
            a = torch.randn(n, TOKEN_DIM, generator=g, dtype=torch.float32) * 2.0
        else:
            # tanh 分支：tok = std*tanh(a) 落在 (-std, std)；a' = atanh(tok/std) 还原。
            # tanh/atanh 非精确互逆（float32 饱和区损失有效位），故取中等幅值 + 容差。
            a = torch.randn(n, TOKEN_DIM, generator=g, dtype=torch.float32) * 0.5
        tok = forward_token_affine(a, token_mean, token_std, alpha, bound)
        # 逆仿射（逐式同 AuthorPolicyAdapter.step:638-642）
        z = (tok - token_mean) / (alpha * token_std)
        if bound == "tanh":
            z = torch.atanh(z.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))
        a_back = z
        if bound == "none":
            _check(torch.equal(a, a_back), f"none 分支 a->tok->a' 非逐位往返（maxdiff={float((a - a_back).abs().max())}）")
        else:
            _check(
                float((a - a_back).abs().max()) <= 1.0e-5,
                f"tanh 分支 a->tok->a' 往返偏差过大（maxdiff={float((a - a_back).abs().max())}）",
            )
        # 反向验证：漏前向仿射（tok 误用 a）时偏离显著
        _check(
            float((tok - a).abs().max()) > 0.1,
            f"{bound} 分支 tok 与 a 过于接近（前向仿射可能是恒等，断言失去区分度）",
        )
    return {
        "affine_roundtrip_bitwise_none": True,
        "affine_roundtrip_tanh_ok": True,
        "tanh_tolerance": 1.0e-5,
        "token_dim": TOKEN_DIM,
    }


def selftest_residual_vs_decoder_arm_decoder_input(seed: int = 0) -> dict:
    """新增断言 ②（M2 跨口径对照）：零残差时 residual 臂 μ 的 decoder 输入 ==
    decoder 臂构造的 decoder 输入，逐位一致。

    现有 `μ == decoder(dec_obs)` 自洽检查抓不到「decoder 输入本身口径错」——它只证明
    μ 等于用同一个错 dec_obs 喂 decoder 的结果。本断言**跨口径对照**：residual 臂
    `assemble_decoder_obs(policy_obs, token_slot, proprio_slot)` 的输出，必须与
    「decoder 臂语义」构造的输入（token 来自 author adapter 的 a 经**前向仿射**、
    proprio 取自 policy obs 的 930 段）逐位相等。若 residual_token_obs 漏了前向仿射
    （用 a 当 tok），本断言即失败。

    用 mock token_stats（零 mean + 2 的幂 std）保证仿射精确可逆；mock adapter.step
    返回一个已知 a，再手工走「前向仿射 -> assemble」重建 decoder 输入。
    """
    g = torch.Generator().manual_seed(seed)
    n = 8
    token_mean = torch.zeros(TOKEN_DIM, dtype=torch.float32)
    token_std = torch.pow(torch.tensor(2.0), (torch.arange(TOKEN_DIM) % 7 - 3).float())
    alpha, bound = 1.0, "none"

    official_dim = 321
    total = official_dim + PROPRIO_DIM + TOKEN_DIM  # official 段在前的真实 term 序
    token_start = official_dim + PROPRIO_DIM
    proprio_start = official_dim
    policy_obs = torch.randn(n, total, generator=g, dtype=torch.float32)

    # mock adapter.step 的返回 = 归一化动作 a（故意取远离 0 的值，放大小口径差异）
    a = torch.randn(n, TOKEN_DIM, generator=g, dtype=torch.float32) * 3.0

    # --- decoder 臂语义构造的 decoder 输入：token 坐标经前向仿射 + proprio 段 ---
    tok = forward_token_affine(a, token_mean, token_std, alpha, bound)
    dec_obs_expected = torch.cat(
        [tok, policy_obs[:, proprio_start : proprio_start + PROPRIO_DIM]], dim=-1
    )

    # --- residual 臂路径：把 token 写回 policy obs 的 token 槽，再走 assemble ---
    policy_obs_r = policy_obs.clone()
    policy_obs_r[:, token_start : token_start + TOKEN_DIM] = tok
    dec_obs_residual = assemble_decoder_obs(
        policy_obs_r, (token_start, TOKEN_DIM), (proprio_start, PROPRIO_DIM)
    )
    _check(
        torch.equal(dec_obs_residual, dec_obs_expected),
        f"residual 臂 decoder 输入 != decoder 臂语义输入（maxdiff={float((dec_obs_residual - dec_obs_expected).abs().max())}）",
    )

    # 反例：漏前向仿射（token 槽写 a 而非 tok）时两者必须不等（证明本断言有区分度）
    policy_obs_bad = policy_obs.clone()
    policy_obs_bad[:, token_start : token_start + TOKEN_DIM] = a
    dec_obs_bad = assemble_decoder_obs(
        policy_obs_bad, (token_start, TOKEN_DIM), (proprio_start, PROPRIO_DIM)
    )
    _check(
        not torch.equal(dec_obs_bad, dec_obs_expected),
        "漏前向仿射的 decoder 输入与正确输入逐位相同（断言失去区分度）",
    )

    # --- 端到端：residual 臂 μ（零残差）== decoder 臂 decoder(正确 decoder 输入) ---
    decoder = _MockDecoder()
    mu_path = ResidualMuPath(decoder, ResidualActionMLP(official_dim), freeze_decoder=True)
    r_obs = policy_obs_r[:, :official_dim]
    with torch.no_grad():
        mu = mu_path(dec_obs_residual, r_obs)
        q = decoder(dec_obs_expected)
    _check(torch.equal(mu, q), f"零残差 μ != decoder(decoder 臂输入)（maxdiff={float((mu - q).abs().max())}）")
    return {
        "decoder_input_bitwise_equal": True,
        "affine_missing_arm_differs": True,
        "official_dim": official_dim,
    }


def selftest_gradient_isolation(seed: int = 0) -> dict:
    """梯度隔离：只 r 参数 requires_grad；μ 反传只在 r 上产生 grad。"""
    g = torch.Generator().manual_seed(seed)
    n, r_obs_dim = 6, 321
    decoder = _MockDecoder()
    residual = ResidualActionMLP(r_obs_dim)
    mu_path = ResidualMuPath(decoder, residual, freeze_decoder=True)

    # 1) 参数 requires_grad 状态
    _check(not any(p.requires_grad for p in mu_path.decoder.parameters()), "decoder 参数未冻结")
    _check(all(p.requires_grad for p in mu_path.residual.parameters()), "residual 参数未全部可训")
    _check(mu_path.only_residual_trainable(), "only_residual_trainable 判据不通过")

    # 2) 反传：只有 residual 拿到 grad，decoder grad 恒 None
    decoder_obs = torch.randn(n, DECODER_OBS_DIM, generator=g)
    r_obs = torch.randn(n, r_obs_dim, generator=g)
    mu_path.zero_grad(set_to_none=True)
    mu = mu_path(decoder_obs, r_obs)
    _check(mu.requires_grad, "μ 不带梯度（残差未接入计算图）")
    loss = (mu ** 2).mean()
    loss.backward()
    dec_grads = [p.grad for p in mu_path.decoder.parameters()]
    res_grads = [p.grad for p in mu_path.residual.parameters()]
    _check(all(g_ is None for g_ in dec_grads), f"decoder 参数拿到梯度（{dec_grads}）")
    _check(all(g_ is not None for g_ in res_grads), "residual 参数未拿到梯度")
    _check(any(float(g_.abs().max()) > 0 for g_ in res_grads), "residual 梯度恒为 0")

    return {
        "decoder_frozen": True,
        "only_residual_trainable": True,
        "decoder_grad_none": True,
        "residual_grad_nonzero": True,
        "trainable_report": mu_path.trainable_report(),
    }


def selftest_reward_math() -> dict:
    """奖励纯数学单测：前进项（符号/无上限/只前进）+ 正则项（符号/量纲）。"""
    # --- 前进项符号 ---
    vx = torch.tensor([-0.5, 0.0, 0.3, 1.0, 5.0])
    r = forward_vel_reward(vx, weight=1.0, positive_only=True)
    _check(bool((r >= 0).all()), "positive_only 下出现负奖励")
    _check(float(r[0]) == 0.0, "后退速度未置零（positive_only）")
    _check(abs(float(r[2]) - 0.3) < 1e-6 and abs(float(r[3]) - 1.0) < 1e-6, "前进项量纲错（应为 1:1 m/s）")
    # 无人工上限：vx=5 的奖励 > vx=1 的奖励（对照 D048j clamp(0,1) 封顶）
    _check(float(r[4]) > float(r[3]), "前进项存在人工上限（vx=5 未高于 vx=1）")
    _check(abs(float(r[4]) - 5.0) < 1e-6, "前进项被限幅（应无上限）")
    # weight 线性缩放
    r2 = forward_vel_reward(vx, weight=2.0, positive_only=True)
    _check(torch.allclose(r2, 2.0 * r), "weight 未线性缩放")
    # positive_only=False 时保留后退负值（对照口径）
    r3 = forward_vel_reward(vx, weight=1.0, positive_only=False)
    _check(float(r3[0]) < 0, "positive_only=False 时后退应保留负值")

    # --- 正则项符号/量纲（配方表） ---
    for name, spec in MAX_FORWARD_RECIPE.items():
        w = float(spec["weight"])
        if spec["kind"] == "task":
            _check(w > 0, f"任务项 {name} 权重应 > 0，实为 {w}")
        elif spec.get("sign", 0) == 0:
            _check(w == 0.0, f"关闭项 {name} 权重应为 0，实为 {w}")
        elif spec["sign"] < 0:
            _check(w < 0, f"罚项 {name} 权重应 < 0，实为 {w}")
        else:
            _check(w > 0, f"塑形项 {name} 权重应 > 0，实为 {w}")
    _check(all(t not in MAX_FORWARD_RECIPE for t in MAX_FORWARD_REMOVED_TERMS),
           "被移除的命令跟踪项仍在配方表内")

    # --- 组合式：罚项使总奖励下降；总奖励形状/有限性 ---
    base = max_forward_total_reward(vx, weight_forward=1.0)
    penalized = max_forward_total_reward(
        vx,
        penalties=[(float(MAX_FORWARD_RECIPE["action_rate_l2"]["weight"]), torch.full_like(vx, 1.0))],
        weight_forward=1.0,
    )
    _check(bool((penalized <= base + 1e-6).all()), "罚项未使总奖励下降")
    _check(penalized.shape == vx.shape and bool(torch.isfinite(penalized).all()), "总奖励形状/有限性异常")

    return {
        "forward_positive_only": True,
        "forward_no_upper_cap": True,
        "forward_unit_mps": True,
        "recipe_signs_ok": True,
        "removed_terms_ok": True,
    }


def selftest_arm_table() -> dict:
    """三臂配置表自检（动作维/奖励默认/冻结分支/token 源）。"""
    _check(tuple(ARM_TABLE) == ARMS, f"ARM_TABLE 键 {tuple(ARM_TABLE)} != {ARMS}")
    for arm, spec in ARM_TABLE.items():
        _check(spec["action_dim"] in (ACTION_DIM, TOKEN_DIM), f"{arm} action_dim 非法")
        _check(spec["env_action"] in ("direct", "decoder", "lora_policy"), f"{arm} env_action 非法")
    _check(ARM_TABLE["direct"]["action_dim"] == ACTION_DIM, "direct 臂应 29 维")
    _check(ARM_TABLE["decoder"]["action_dim"] == TOKEN_DIM, "decoder 臂应 64 维")
    _check(ARM_TABLE["residual"]["action_dim"] == ACTION_DIM, "residual 臂应 29 维")
    _check(ARM_TABLE["residual"]["frozen_branch"] is True, "residual 臂应标 frozen_branch")
    _check(ARM_TABLE["residual"]["token_source"] == "author_v0_ckpt", "residual 臂 token 源应为 author v0")
    _check(ARM_TABLE["residual"]["env_action"] == "lora_policy", "residual 臂 env 侧应为 direct 通路变体")
    # reward auto 解析
    _check(resolve_reward_mode("residual", "auto") == "max_forward", "residual auto 应 -> max_forward")
    _check(resolve_reward_mode("direct", "auto") == "track", "direct auto 应 -> track")
    _check(resolve_reward_mode("decoder", "auto") == "track", "decoder auto 应 -> track")
    _check(resolve_reward_mode("direct", "max_forward") == "max_forward", "显式 max_forward 未被采纳")
    return {"arms": list(ARMS), "auto_reward_resolved": {a: resolve_reward_mode(a, "auto") for a in ARMS}}


def selftest_identity_envelope() -> dict:
    """身份信封自检（本机可构建，含三臂表项、奖励配方与新增契约字段）。"""
    env = build_identity_envelope("residual", "max_forward", intent_pin={"intent_pin_max": True, "value": 1.0})
    _check(env["arm"] == "residual", "信封 arm 字段错")
    _check(env["arm_table"]["frozen_branch"] is True, "信封 arm_table 未带冻结标记")
    _check(env["reward_mode"] == "max_forward", "信封 reward_mode 错")
    _check("forward_vel" in env["reward_recipe"], "信封奖励配方缺前进项")
    _check(set(MAX_FORWARD_REMOVED_TERMS) <= set(env["reward_removed_terms"]), "信封未记录被移除项")
    _check(env["script_md5"] and len(env["script_md5"]) == 32, "信封 script_md5 非法")
    _check(isinstance(env["env_module_md5"], dict) and env["env_module_md5"], "信封 env_module_md5 为空")
    # 新增契约字段（M1/M2/M3/M4）
    _check("ctx_terrain_map" in env and env["ctx_terrain_map"] == CTX_TERRAIN_MAP, "信封缺 ctx_terrain_map")
    _check("ctx_terrain" in env, "信封缺 ctx_terrain 字段")
    _check("token_affine" in env and "formula" in env["token_affine"], "信封缺 token_affine 字段")
    _check(env["privileged_critic"]["author_token_in_critic"] is True, "信封缺 privileged_critic 声明")
    _check(env["reward_recipe_scope"]["applied_to"] == "all_terrains", "信封缺 reward_recipe_scope 声明")
    # JSON 可序列化（写盘前提）
    json.dumps(env, ensure_ascii=False)
    return {"envelope_keys": sorted(env)}


def selftest_ctx_terrain_contract() -> dict:
    """M1 地形契约自检：CLI 地形名 -> ctx 契约名映射全覆盖且无非法值。"""
    allowed = ("plane", "rough_paper", "climbing_box")
    for t in TERRAINS:
        got = ctx_terrain(t)
        _check(got in allowed, f"{t} -> {got} 不在 ctx 契约 {allowed}")
    _check(ctx_terrain("plane") == "plane", "plane 应映射到 plane")
    for t in ("rough", "stairs", "stones", "discrete"):
        _check(ctx_terrain(t) == "rough_paper", f"{t} 应映射到 rough_paper，实为 {ctx_terrain(t)}")
    # 幂等：ctx 契约名原样返回
    _check(ctx_terrain("rough_paper") == "rough_paper", "ctx_terrain 非幂等（rough_paper）")
    # 未知名 fail loud
    try:
        ctx_terrain("mars")
        raise RuntimeError("未知地形未 fail loud")
    except ValueError:
        pass
    return {"terrains": list(TERRAINS), "mapped": {t: ctx_terrain(t) for t in TERRAINS}}


def selftest_official_obs_slice_guard() -> dict:
    """SF-2 切片守卫自检：正确 term 序通过；前缀假设被破坏时 fail loud。"""
    # 正确序：official 段在前，proprio/token 在后
    terms = ("base_lin_vel", "base_ang_vel", "joint_pos", "sonic_proprio", "author_token")
    dims = ((3,), (3,), (29,), (930,), (64,))
    off = official_obs_slice(terms, dims)
    _check(off["start"] == 0, "official 段起点应为 0")
    _check(off["width"] == 3 + 3 + 29, f"official 段宽度错：{off['width']}")
    _check(off["excluded"] == {"sonic_proprio": 930, "author_token": 64}, f"排除宽度错：{off['excluded']}")
    # 前缀假设被破坏（proprio 排在 official 段之前）-> fail loud
    bad_terms = ("sonic_proprio", "base_lin_vel", "author_token")
    bad_dims = ((930,), (3,), (64,))
    try:
        official_obs_slice(bad_terms, bad_dims)
        raise RuntimeError("前缀假设被破坏时未 fail loud")
    except RuntimeError:
        pass
    # 缺被排除 term -> fail loud
    try:
        official_obs_slice(("base_lin_vel",), ((3,),))
        raise RuntimeError("缺 author_token 时未 fail loud")
    except RuntimeError:
        pass
    return {"official_width": off["width"], "excluded": off["excluded"], "fail_loud_ok": True}


def run_selftest() -> None:
    """本机 CPU 全量自测（不 import 任何 isaac 模块）。"""
    torch.manual_seed(0)
    res = {
        "zero_init_identity": selftest_zero_init_identity(),
        "gradient_isolation": selftest_gradient_isolation(),
        "reward_math": selftest_reward_math(),
        "arm_table": selftest_arm_table(),
        "identity_envelope": selftest_identity_envelope(),
        "token_affine_roundtrip": selftest_token_affine_roundtrip(),
        "residual_vs_decoder_arm_decoder_input": selftest_residual_vs_decoder_arm_decoder_input(),
        "ctx_terrain_contract": selftest_ctx_terrain_contract(),
        "official_obs_slice_guard": selftest_official_obs_slice_guard(),
    }
    print("[d067] selftest 明细: " + json.dumps(res, ensure_ascii=False), flush=True)
    print("D067_SELFTEST_PASS", flush=True)


# ---------------------------------------------------------------------------
# Layer 2: isaac（以下 import 都在函数体内；模块 import 期零 isaaclab 依赖）
# ---------------------------------------------------------------------------
def _import_train_g1_decoder():
    try:
        from isaac import train_g1_decoder as td
    except ImportError:  # pragma: no cover（本机走此支）
        from apt_g1.isaac import train_g1_decoder as td
    return td


def _import_heavy() -> SimpleNamespace:
    """App 起来后的重型 import：复用 train_g1_decoder._import_heavy 并补齐 D067 需要件。

    复用父本单份组装（torch/np/gym/ManagerBasedRLEnv/RslRl*Cfg/EnvWrapper/
    OnPolicyRunner/mdp/file_md5/make_env_cfg/SonicDecoderActionTerm[Cfg]），
    追加本文件所需：ObsTerm/RewTerm/SceneEntityCfg（obs/奖励覆写）、AuthorPolicyAdapter
    相关件（author token 源）、SonicTorchDecoder（残差臂 policy 内冻结 decoder）。
    """
    td = _import_train_g1_decoder()
    hv = td._import_heavy()
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import SceneEntityCfg

    try:  # 服务器执行根平铺 import / 仓库根包 import 双兼容
        from isaac.g1_velocity_decoder_env import SONIC_DECODER_ONNX  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from apt_g1.isaac.g1_velocity_decoder_env import SONIC_DECODER_ONNX  # type: ignore[no-redef]

    try:
        from isaac.eval_author_v0_decoder import (  # noqa: PLC0415
            AuthorPolicyAdapter,
            TOKEN_SCALE_DEFAULT,
            build_state,
            ctx_from_command,
            load_author_ckpt,
        )
    except ImportError:  # pragma: no cover
        from apt_g1.isaac.eval_author_v0_decoder import (  # type: ignore[no-redef]
            AuthorPolicyAdapter,
            TOKEN_SCALE_DEFAULT,
            build_state,
            ctx_from_command,
            load_author_ckpt,
        )
    try:
        from isaac.sonic_decoder_torch import SonicTorchDecoder  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from apt_g1.isaac.sonic_decoder_torch import SonicTorchDecoder  # type: ignore[no-redef]

    hv.ObsTerm = ObsTerm
    hv.RewTerm = RewTerm
    hv.SceneEntityCfg = SceneEntityCfg
    hv.SONIC_DECODER_ONNX = SONIC_DECODER_ONNX
    hv.AuthorPolicyAdapter = AuthorPolicyAdapter
    hv.TOKEN_SCALE_DEFAULT = TOKEN_SCALE_DEFAULT
    hv.build_state = build_state
    hv.ctx_from_command = ctx_from_command
    hv.load_author_ckpt = load_author_ckpt
    hv.SonicTorchDecoder = SonicTorchDecoder
    return hv


# ------------------------------------------------------------- 冻结 decoder 分支
def make_frozen_decoder_branch(decoder, token_dim: int = TOKEN_DIM) -> nn.Module:
    """把 SonicTorchDecoder 包成 (N,994)->(N,29) 的 nn.Module 分支。

    运算序列与 `sonic_action_term.wrap_sonic_torch_decoder` 逐字相同：先对 token 片做
    FSQ 量化，再 forward 全 994 维（冻结 decoder 的一部分；「无 VAE」指不做 VAE 解码）。
    """

    class _FrozenSonicDecoderBranch(nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.decoder = inner
            self.token_dim = int(token_dim)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            tokens_q = self.decoder.quantize_tokens(obs[:, : self.token_dim])
            return self.decoder.forward(torch.cat([tokens_q, obs[:, self.token_dim :]], dim=1))

    return _FrozenSonicDecoderBranch(decoder)


# ------------------------------------------------------------- max_forward 奖励覆写
def _max_forward_reward_func(env, asset_cfg, positive_only: bool = True):
    """isaaclab RewardTerm func：root_lin_vel_b[:,0] 正向项（weight 由 RewTerm 施加）。

    返回 raw 前向速度（positive_only 时 clamp(min=0)），RewardManager 再乘 RewTerm.weight。
    与 Layer 1 的 forward_vel_reward 同口径（此处的 weight 交给 manager）。
    """
    vx_b = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0]
    if positive_only:
        vx_b = torch.clamp(vx_b, min=0.0)
    return vx_b


def _apply_max_forward_recipe(cfg, hv, weight_forward: float, positive_only: bool = True) -> dict:
    """把 max_forward 奖励配方覆写到 cfg.rewards（D067 残差臂专用）。

    做法（对照 §5y「奖励=机体坐标前进速度最大化 + 官方正则、命令跟踪项移除」）：
      1. 移除命令跟踪项（track_lin_vel_xy_exp / track_ang_vel_z_exp -> None）；
      2. 正则项权重按 MAX_FORWARD_RECIPE 逐项覆写（防摔/姿态/动作平滑）；
      3. 新增 `max_forward_vel` 项（func = _max_forward_reward_func，weight = --forward-weight）。
    对 cfg.rewards.* 用直接属性访问（不做 hasattr 静默跳过；官方 term 名漂移首启即报错，
    同 g1_velocity_decoder_env._adapt_g1_sonic_asset 纪律）。
    """
    rewards = cfg.rewards
    removed = []
    for term in MAX_FORWARD_REMOVED_TERMS:
        if getattr(rewards, term, None) is not None:
            setattr(rewards, term, None)
            removed.append(term)
    reg_written = {}
    for name, spec in MAX_FORWARD_RECIPE.items():
        if spec["kind"] == "task":
            continue
        term_cfg = getattr(rewards, name, None)
        if term_cfg is None:
            # 配方表内的项却不存在/为 None = 官方 term 名漂移，显式失败（不静默漏适配）
            raise RuntimeError(
                f"rewards.{name} 不存在/为 None（官方 term 名漂移？配方表 {list(MAX_FORWARD_RECIPE)}）"
            )
        term_cfg.weight = float(spec["weight"])
        reg_written[name] = float(spec["weight"])

    # 新增前进项（dynamic attr：RewardManager 按 cfg.__dict__ 枚举 term，同 obs manager 惯例）
    func = (lambda env, asset_cfg: _max_forward_reward_func(env, asset_cfg, positive_only))
    setattr(
        rewards,
        "max_forward_vel",
        hv.RewTerm(func=func, weight=float(weight_forward), params={"asset_cfg": hv.SceneEntityCfg("robot")}),
    )
    return {"removed": removed, "regularizers": reg_written, "forward_weight": float(weight_forward)}


# ------------------------------------------------------------- author_token 观测项
def load_token_stats(path: str | os.PathLike, *, floor: float = TOKEN_STD_FLOOR) -> tuple[torch.Tensor, torch.Tensor]:
    """读 `--token-stats` npz 的 mean/std（float32，CPU，std 施加 floor）。

    与 `eval_author_v0_decoder.author_token_stats`（:489-498）与
    `sonic_action_term.SonicDecoderActionTerm`（:443-453，floor=token_std_floor）
    **同式同 floor**，故 `residual_token_obs` 的前向仿射与 author adapter 的逆仿射
    消费的是同一份统计（token 值域契约 M2）。
    """
    stats = np.load(str(path))
    mean = torch.as_tensor(np.asarray(stats["mean"], dtype=np.float32), device="cpu").reshape(-1)
    std = torch.as_tensor(np.asarray(stats["std"], dtype=np.float32), device="cpu").reshape(-1)
    if mean.shape != (TOKEN_DIM,) or std.shape != (TOKEN_DIM,):
        raise ValueError(f"token_stats 的 mean/std 须为 ({TOKEN_DIM},)，收到 {tuple(mean.shape)}/{tuple(std.shape)}")
    return mean, torch.clamp(std, min=float(floor))


def residual_token_obs(
    env,
    ckpt_path: str,
    token_stats: str,
    token_alpha: float,
    token_bound: str,
    intent_pin_max: bool,
    intent_pin_value: float,
    terrain: str = "plane",
    *,
    token_mean: torch.Tensor | None = None,
    token_std: torch.Tensor | None = None,
    author_loader,
    adapter_factory,
    ctx_builder,
    ctx_builder_batch,
    state_builder,
):
    """policy 观测项：(N, 64) 冻结 author adapter 产出的 **token 坐标**（观测常量，无梯度）。

    设计依据（假设 A1）：author adapter 是**有状态自回归**件——若放进 policy.forward，
    PPO 的 evaluate 阶段会重跑并污染滚动窗。故必须 env 侧每控制步算一次、作为观测
    喂给 policy。token 走 obs ⇒ 与 rollout buffer 同步存储，evaluate 阶段读的是同一值。

    **值域契约（M2）**：`adapter.step` 返回的是归一化动作 a = (tok-mean)/(alpha*std)
    （eval_author_v0_decoder.AuthorPolicyAdapter.step:638-642），**不是** token 坐标。
    喂 decoder 分支（会做 FSQ 量化）前必须补前向仿射
    `tok = mean + alpha*std*(tanh(a) if token_bound=='tanh' else a)`
    （`forward_token_affine`；权威式 sonic_action_term.py:205-215 / sonic_lora_policy.py:157-161）。
    `token_mean`/`token_std` 由调用侧从**与 decoder 分支同一份** `--token-stats` npz 显式
    传入（`_token_obs_params` 经 `load_token_stats`）；装载 adapter 时与 adapter 内部统计
    做**逐位一致性断言**（不同源即 fail loud）。

    地形契约（M1）：`terrain` 先经 `ctx_terrain` 映射到 ctx 契约名
    （rough/stairs/stones/discrete -> rough_paper），再进 ctx 构造，杜绝
    `ctx_from_command` 首帧 ValueError。

    依赖经 params 显式注入（`author_loader`/`adapter_factory`/`ctx_builder`/
    `ctx_builder_batch`/`state_builder`，见 `_attach_token_obs`）——避免模块级全局转发名，
    便于单测替换。

    实现要点：
      - 懒建 `AuthorPolicyAdapter`（冻结）挂 `env._d067_token_adapter`；
      - **每控制步只推一帧**：以 `episode_length_buf` 快照去重（ObservationManager 可能
        在一步内被多次 compute，若无去重会多推帧污染自回归窗）；
      - state 由 `robot.data` 直取（**不走 obs**——避免本项在 obs 计算中递归调用
        observation_manager）：joint_pos（SONIC 序，绝对）+ root_quat_wxyz + env 局部
        trans，与 eval_author_v0_decoder._run_cond_main 同口径（build_state 内做
        Isaac->MuJoCo perm 重排）；
      - ctx 命令槽：`intent_pin_max=True` 钉 `intent_pin_value`（默认 1.0，假设 A2），
        否则读 env 当前速度命令 vx（逐 env）；
      - 返回 token.detach()（不建图，梯度不达 token 源）。
    构造期容忍：ObservationManager 在 env 构造早期会无条件调本项一次填维度，此时
    `episode_length_buf` 可能尚未创建——用 getattr 探测，缺件时返回零 token 占位
    （其后 env.reset() 后的正式计算覆盖）。
    """
    terrain_ctx = ctx_terrain(terrain)
    adapter = getattr(env, "_d067_token_adapter", None)
    if adapter is None:
        if token_mean is None or token_std is None:
            raise ValueError(
                "residual_token_obs 需要显式 token_mean/token_std（与 decoder 分支同一份 "
                "--token-stats npz；见 `_token_obs_params`）"
            )
        model, model_cfg = author_loader(ckpt_path, device=str(env.device))
        token_mean_t = torch.as_tensor(token_mean, dtype=torch.float32, device=env.device).reshape(-1)
        token_std_t = torch.as_tensor(token_std, dtype=torch.float32, device=env.device).reshape(-1)
        adapter = adapter_factory(
            model,
            model_cfg,
            token_mean_t,
            token_std_t,
            num_envs=env.num_envs,
            device=str(env.device),
            token_alpha=token_alpha,
            token_bound=token_bound,
        )
        # 同一份统计守卫：adapter 内部统计（含 floor）须与传入的前向仿射统计逐位一致
        a_mean = getattr(adapter, "token_mean", None)
        a_std = getattr(adapter, "token_std", None)
        if a_mean is not None and a_std is not None:
            am = torch.as_tensor(a_mean, dtype=torch.float32, device=token_mean_t.device).reshape(-1)
            as_ = torch.as_tensor(a_std, dtype=torch.float32, device=token_std_t.device).reshape(-1)
            if not (torch.equal(am, token_mean_t) and torch.equal(as_, token_std_t)):
                raise RuntimeError(
                    "adapter 内部 token 统计与传入的前向仿射统计不一致（必须同源同一份 npz）："
                    f"mean maxdiff={float((am - token_mean_t).abs().max())} "
                    f"std maxdiff={float((as_ - token_std_t).abs().max())}"
                )
        env._d067_token_adapter = adapter
        env._d067_token_cache = None
        env._d067_token_step_key = None
        env._d067_token_pin_max = bool(intent_pin_max)
        env._d067_token_pin_value = float(intent_pin_value)
        env._d067_token_terrain = str(terrain_ctx)
        env._d067_token_mean = token_mean_t
        env._d067_token_std = token_std_t
        env._d067_token_alpha = float(token_alpha)
        env._d067_token_bound = str(token_bound)
        print(
            f"[d067] author token 源已装载：ckpt={ckpt_path} token_stats={token_stats} "
            f"vocab={model_cfg.get('vocab')} code_min={model_cfg.get('code_min')} "
            f"intent_pin_max={intent_pin_max} intent_pin_value={intent_pin_value} "
            f"token_bound={token_bound} terrain={terrain!r}->ctx_terrain={terrain_ctx!r}",
            flush=True,
        )

    ep = getattr(env, "episode_length_buf", None)
    if ep is None:
        # 构造期兜底：返回零 token 占位（维度正确；reset 后正式计算覆盖）
        return torch.zeros(env.num_envs, TOKEN_DIM, device=env.device)

    # 每控制步去重（同一步内多次 compute 只推一帧）
    key = ep.detach().clone()
    prev_key = getattr(env, "_d067_token_step_key", None)
    if prev_key is not None and torch.equal(prev_key, key):
        cached = getattr(env, "_d067_token_cache", None)
        if cached is not None:
            return cached

    reset_ids = (ep == 0).nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        adapter.reset(reset_ids)

    robot = env.scene["robot"]
    joint_ids_t = getattr(env, "_d067_joint_ids_t", None)
    if joint_ids_t is None:
        try:
            from isaac.sonic_action_term import resolve_sonic_joint_ids  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            from apt_g1.isaac.sonic_action_term import resolve_sonic_joint_ids  # type: ignore[no-redef]

        ids, _identity = resolve_sonic_joint_ids(robot.joint_names)
        joint_ids_t = torch.as_tensor(ids, dtype=torch.long, device=env.device)
        env._d067_joint_ids_t = joint_ids_t

    # state = joint_pos（SONIC 序，绝对）+ root quat_wxyz + env 局部 trans
    jp = robot.data.joint_pos[:, joint_ids_t]
    quat = robot.data.root_quat_w
    trans = robot.data.root_pos_w - env.scene.env_origins
    st = state_builder(jp, quat, trans, jp_absolute=True)

    if env._d067_token_pin_max:
        ctx = ctx_builder(env._d067_token_pin_value, env._d067_token_terrain)
    else:
        cmd = env.command_manager.get_term("base_velocity").vel_command_b[:, 0]
        ctx = ctx_builder_batch(cmd, env._d067_token_terrain)
    with torch.no_grad():
        raw = adapter.step(st, ctx=ctx)
        # M2：adapter.step 返回归一化动作 a（逆仿射产物），补前向仿射 -> token 坐标，
        # 否则 decoder 分支的 FSQ 量化吃到 a 而非 tok（先验破坏）。
        tok = forward_token_affine(
            raw,
            env._d067_token_mean,
            env._d067_token_std,
            env._d067_token_alpha,
            env._d067_token_bound,
        )
    tok = tok.detach()
    env._d067_token_cache = tok
    env._d067_token_step_key = key
    return tok


def _token_obs_params(hv, cli, terrain: str) -> tuple[dict, str]:
    """residual_token_obs 的 params（含注入的可调用件，避免模块级全局转发名）。

    token_mean/std 由**与 decoder 分支同一份** `--token-stats` npz 读出（M2 值域契约）；
    terrain 在此先经 `ctx_terrain` 映射（M1 地形契约，幂等），再进 ctx 构造器。
    返回 (params, ctx_terrain_name)。
    """
    if not cli.token_stats:
        raise RuntimeError("residual_token_obs 需要 --token-stats（前向仿射 mean/std 与 decoder 分支同源）")
    token_mean, token_std = load_token_stats(cli.token_stats)
    ctx_terrain_name = ctx_terrain(terrain)
    return {
        "ckpt_path": cli.author_ckpt,
        "token_stats": cli.token_stats,
        "token_alpha": cli.token_alpha if cli.token_alpha is not None else 1.0,
        "token_bound": cli.token_bound or "tanh",
        "intent_pin_max": bool(cli.intent_pin_max),
        "intent_pin_value": float(cli.intent_pin_value),
        "terrain": terrain,
        "token_mean": token_mean,
        "token_std": token_std,
        "author_loader": hv.load_author_ckpt,
        "adapter_factory": lambda model, cfg, mean, std, **kw: hv.AuthorPolicyAdapter(
            model,
            code_min=int(cfg["code_min"]),
            vocab=int(cfg["vocab"]),
            token_mean=mean,
            token_std=std,
            token_scale=hv.TOKEN_SCALE_DEFAULT,
            **kw,
        ),
        "ctx_builder": lambda pin_value, terr: hv.ctx_from_command(pin_value, 0.0, 0.0, terrain=terr),
        "ctx_builder_batch": lambda vx, terr: np.stack(
            [hv.ctx_from_command(float(v), 0.0, 0.0, terrain=terr) for v in np.asarray(vx).reshape(-1)]
        ),
        "state_builder": hv.build_state,
    }, ctx_terrain_name


# ------------------------------------------------------------- rsl_rl 残差策略
try:  # pragma: no cover - 服务器有 rsl_rl；本机走 except（降级 nn.Module 仅供 import）
    from rsl_rl.modules import ActorCritic as _ActorCriticBase

    HAS_RSL_RL = True
except ImportError:  # pragma: no cover - 本机（无训练 venv）
    _ActorCriticBase = nn.Module  # type: ignore[assignment,misc]
    HAS_RSL_RL = False

RESIDUAL_POLICY_CLASS_NAME = "ResidualDecoderPolicy"


class ResidualDecoderPolicy(_ActorCriticBase):  # type: ignore[misc]
    """rsl_rl ActorCritic 变体：μ = 冻结 decoder(cat[token, proprio]) + r(官方 obs)。

    - 动作分布 N(μ, σ)，动作维 29（= 关节目标，与 direct 臂同口径）。
    - `token_slice` = policy obs 里 64 维 `author_token` 的切片；`proprio_slice` = 930 维
      `sonic_proprio` 的切片；二者由 train 侧按 observation_manager 运行期 term 序算出
      （不硬编码位置）。
    - `official_obs_dim` = policy obs 总维 - 930 - 64 = 原 D064 policy 八项（含 height_scan）；
      r 的输入即该切片（假设 A5）。
    - 冻结分支在 `ResidualMuPath` 内（decoder no_grad）；**只有 residual 与 critic 可训**，
      σ 默认冻结（`sigma_trainable=False`，§5y「PPO 只训 r」）；基类 actor 干被冻结且不参与
      前向（本臂无独立 actor MLP，μ 由冻结分支 + r 构成）。
    - 可训集合摘除由 train 侧 `_prune_frozen_params` 落地（同 D065 口径）。
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int = ACTION_DIM,
        *,
        decoder: nn.Module | None = None,
        token_slice: tuple[int, int] | None = None,
        proprio_slice: tuple[int, int] | None = None,
        official_obs_dim: int | None = None,
        residual_hidden: tuple[int, ...] = RESIDUAL_HIDDEN,
        activation: str = RESIDUAL_ACTIVATION,
        init_noise_std: float = 1.0,
        sigma: float | None = None,
        sigma_trainable: bool = False,
        **kwargs,
    ) -> None:
        if not HAS_RSL_RL:
            raise RuntimeError(
                "ResidualDecoderPolicy 需要 rsl_rl（rsl-rl-lib 2.3.3 经典 API ActorCritic）；"
                "本机（Windows）无训练 venv，请在服务器 .venv_isaac 内实例化。"
            )
        if decoder is None:
            raise ValueError("ResidualDecoderPolicy 需要冻结 decoder（见 _build_residual_policy_kwargs）")
        if token_slice is None or proprio_slice is None:
            raise ValueError("ResidualDecoderPolicy 需要 token_slice/proprio_slice（obs 切片）")
        if int(num_actions) != ACTION_DIM:
            raise ValueError(f"动作维冻结为 {ACTION_DIM}，收到 {num_actions}")
        super().__init__(
            num_actor_obs=int(num_actor_obs),
            num_critic_obs=int(num_critic_obs),
            num_actions=ACTION_DIM,
            actor_hidden_dims=[512, 256, 128],  # 占位（本臂不用基类 actor 干）
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            init_noise_std=float(init_noise_std),
        )
        # 基类 actor 干不参与本臂前向：冻结（不进优化器，防污染可训集合）
        for p in self.actor.parameters():
            p.requires_grad_(False)
        self.actor_frozen = True

        self.token_start, self.token_width = int(token_slice[0]), int(token_slice[1])
        self.proprio_start, self.proprio_width = int(proprio_slice[0]), int(proprio_slice[1])
        if self.token_width != TOKEN_DIM:
            raise ValueError(f"token_slice 宽度须为 {TOKEN_DIM}，收到 {self.token_width}")
        if self.proprio_width != PROPRIO_DIM:
            raise ValueError(f"proprio_slice 宽度须为 {PROPRIO_DIM}，收到 {self.proprio_width}")
        if int(self.token_width) + int(self.proprio_width) != DECODER_OBS_DIM:
            raise ValueError("token+proprio 宽度应等于 decoder 输入维 994")

        # r 输入维：缺省 = 官方 policy obs 切片（total - proprio - token），假设 A5
        if official_obs_dim is None:
            official_obs_dim = int(num_actor_obs) - self.proprio_width - self.token_width
        self.official_obs_dim = int(official_obs_dim)
        if self.official_obs_dim <= 0:
            raise ValueError(f"official_obs_dim 推得 {self.official_obs_dim}（num_actor_obs={num_actor_obs}）")
        self.mu_path = ResidualMuPath(
            decoder, ResidualActionMLP(self.official_obs_dim, ACTION_DIM, residual_hidden, activation)
        )

        sigma0 = float(init_noise_std) if sigma is None else float(sigma)
        self.std = nn.Parameter(torch.full((ACTION_DIM,), sigma0), requires_grad=bool(sigma_trainable))
        self.sigma_frozen = not bool(sigma_trainable)

    @staticmethod
    def _flat_obs(obs):
        if isinstance(obs, dict):
            return obs["policy"]
        return obs

    def decoder_obs(self, flat: torch.Tensor) -> torch.Tensor:
        """从 policy obs 切出 (N,994) = cat[author_token(64), sonic_proprio(930)]。"""
        return assemble_decoder_obs(
            flat,
            (self.token_start, self.token_width),
            (self.proprio_start, self.proprio_width),
        )

    def mu(self, obs) -> torch.Tensor:
        """μ(29) = decoder(cat[token, proprio]) + r(官方 obs 切片)。"""
        flat = self._flat_obs(obs)
        dec_obs = self.decoder_obs(flat)
        r_obs = flat[:, : self.official_obs_dim]
        return self.mu_path(dec_obs, r_obs)

    def update_distribution(self, observations) -> None:
        mu = self.mu(observations)
        std = self.std.to(mu.device).expand_as(mu)
        self.distribution = torch.distributions.Normal(mu, std)

    def act(self, observations, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations) -> torch.Tensor:
        return self.mu(observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    def trainable_report(self) -> dict:
        def _count(mod):
            ps = list(mod.parameters())
            return {
                "n_tensors": len(ps),
                "n_trainable": sum(1 for p in ps if p.requires_grad),
                "n_elements": int(sum(p.numel() for p in ps)),
                "n_trainable_elements": int(sum(p.numel() for p in ps if p.requires_grad)),
            }

        return {
            "sigma": {"value": float(self.std.detach().reshape(-1)[0].item()), "frozen": bool(self.sigma_frozen)},
            "actor_placeholder": _count(self.actor),
            "critic": _count(self.critic),
            "decoder": _count(self.mu_path.decoder),
            "residual": _count(self.mu_path.residual),
            "only_residual_trainable": self.mu_path.only_residual_trainable(),
        }


def register_residual_policy_in_rsl_rl(cls=ResidualDecoderPolicy, name: str = RESIDUAL_POLICY_CLASS_NAME) -> str:
    """把残差策略类注入 rsl_rl 类名解析命名空间（同 sonic_lora_policy 手法）。

    rsl_rl 经典 API 用 `eval(cfg["policy"]["class_name"])` 在 runner 模块命名空间解析；
    挂到 rsl_rl.runners.on_policy_runner / rsl_rl.runners / rsl_rl.modules 即可生效。
    调用方随后必须断言 runner 内 policy 实例类型（防静默回落成 ActorCritic 把残差臂训成 direct 臂）。
    """
    import importlib

    targets = []
    for mod_name in ("rsl_rl.runners.on_policy_runner", "rsl_rl.runners", "rsl_rl.modules"):
        try:  # pragma: no cover - 服务器有 rsl_rl
            targets.append(importlib.import_module(mod_name))
        except ImportError:
            continue
    if not targets:
        raise RuntimeError("rsl_rl 不可导入——无法注册残差策略类（请在 isaaclab venv 内运行）")
    for mod in targets:
        setattr(mod, name, cls)
    return name


# ------------------------------------------------------------- env / policy 装配
def _policy_obs_slot(env, torch_mod, term_name: str, width_expected: int) -> dict:
    """定位 policy obs 里某 term 的切片（运行期 term 序，不硬编码位置）。

    与 eval_author_v0_decoder._policy_obs_cmd_slot 同款：按 observation_manager
    active_terms/group_obs_term_dim 逐 term 维度累加得 start/width；term 缺位即显式失败。
    """
    om = env.observation_manager
    names = [str(n) for n in om.active_terms["policy"]]
    dims = list(om.group_obs_term_dim["policy"])
    if len(names) != len(dims):
        raise RuntimeError(f"policy 组 names/dims 错位：{len(names)} vs {len(dims)}")
    if term_name not in names:
        raise RuntimeError(f"policy 组无 {term_name!r} 观测项（现有 {names}）——残差臂接线失败")
    idx = names.index(term_name)
    start = 0
    for term_dims in dims[:idx]:
        start += int(torch_mod.tensor(list(term_dims)).prod())
    width = int(torch_mod.tensor(list(dims[idx])).prod())
    if width != int(width_expected):
        raise RuntimeError(f"{term_name} 宽度 {width} != {width_expected}")
    return {"term": term_name, "index": int(idx), "start": int(start), "width": int(width), "policy_terms": names}


def _attach_token_obs(cfg, hv, cli, terrain: str) -> dict:
    """把 64 维 `author_token` 观测项追加到 cfg.observations.policy 末位（残差臂）。

    返回地形契约/仿射信息（进身份信封）：`ctx_terrain` = 实际进 ctx 的地形名
    （经 CTX_TERRAIN_MAP 映射），`token_stats` = 前向仿射统计路径。
    """
    # configclass 的 policy 组是 dataclass 实例；直接 setattr 新 term（obs manager 按
    # __dict__ 枚举，见 tmp/isaac_ref/observation_manager.py:391-393）。字段序 = 追加序。
    params, ctx_terrain_name = _token_obs_params(hv, cli, terrain)
    setattr(cfg.observations.policy, "author_token", hv.ObsTerm(func=residual_token_obs, params=params))
    # critic 特权组同步追加（**有意的不对称**：critic 可比 actor 多看 token，SF-1 声明；
    # policy/critic 同 func 同参，故 critic 看到的 token 与 actor 逐位一致，仅多这一项）
    if getattr(cfg.observations, "critic", None) is not None:
        setattr(cfg.observations.critic, "author_token", hv.ObsTerm(func=residual_token_obs, params=params))
    return {
        "ctx_terrain": ctx_terrain_name,
        "env_terrain": str(terrain),
        "token_stats": cli.token_stats,
        "token_bound": params["token_bound"],
        "token_alpha": params["token_alpha"],
        "privileged_critic": True,
    }


def _build_residual_env_cfg(hv, cli, device: str, suffix: str):
    """残差臂 env cfg：direct 动作通路（29 维）+ max_forward 奖励 + author_token 观测。

    env 侧走 sibling 既有 `action="lora_policy"` 变体（= direct 臂动作通路逐字 + policy 组
    末位 930 维 decoder 本体历史观测），残差臂的唯一 env 侧增量 = 追加 `author_token` 观测项。
    """
    td = _import_train_g1_decoder()
    cfg, how = td._build_env_cfg(
        hv.make_env_cfg, hv.mdp, hv.SonicDecoderActionTermCfg, cli.terrain, "lora_policy", cli, device, suffix
    )
    recipe = _apply_max_forward_recipe(cfg, hv, cli.forward_weight)
    token_info = _attach_token_obs(cfg, hv, cli, cli.terrain)
    print(
        f"[d067] residual env cfg via {how} + max_forward 奖励 + author_token 观测：{recipe}；"
        f"token_info={json.dumps(token_info, ensure_ascii=False)}",
        flush=True,
    )
    return cfg, how, recipe, token_info


def _build_residual_policy_kwargs(hv, env, cli, device: str) -> tuple[dict, dict]:
    """构造残差臂 policy kwargs：冻结 decoder 分支 + token/proprio 切片（运行期定位）。

    SF-2 守卫：`official_obs_dim`（r 的输入宽度）由 `official_obs_slice` 按**运行期
    term 序**推出（policy 总宽 − sonic_proprio − author_token），并断言 official 段是
    policy obs 前缀（起始=0）；不再让 `flat[:, :official_obs_dim]` 的隐式前缀假设裸奔。
    """
    onnx_path = cli.onnx_path or hv.SONIC_DECODER_ONNX
    decoder = hv.SonicTorchDecoder(onnx_path, device=device)
    branch = make_frozen_decoder_branch(decoder)
    for p in branch.parameters():
        p.requires_grad_(False)
    branch.eval()
    token_slot = _policy_obs_slot(env, hv.torch, "author_token", TOKEN_DIM)
    proprio_slot = _policy_obs_slot(env, hv.torch, "sonic_proprio", PROPRIO_DIM)
    om = env.observation_manager
    off = official_obs_slice(om.active_terms["policy"], om.group_obs_term_dim["policy"])
    kwargs = {
        "decoder": branch,
        "token_slice": (token_slot["start"], token_slot["width"]),
        "proprio_slice": (proprio_slot["start"], proprio_slot["width"]),
        "official_obs_dim": int(off["width"]),
        "sigma_trainable": bool(cli.sigma_trainable),
        "sigma": float(cli.sigma) if cli.sigma is not None else None,
    }
    info = {
        "token_slot": token_slot,
        "proprio_slot": proprio_slot,
        "official_slice": off,
        "onnx_path": onnx_path,
        "decoder_frozen": True,
    }
    return kwargs, info


def _prune_frozen_params(optimizer) -> dict:
    """把 requires_grad=False 的参数从优化器 param_groups 摘掉（同 D065 手法）。

    落地「PPO 只训 r（+critic）」：decoder/actor 占位/σ（冻结时）摘除，只剩 residual
    与 critic。**摘除须先于 runner.load**（resume 时 optimizer state 按已摘除的组存）。
    """
    removed, kept = [], 0
    for grp in optimizer.param_groups:
        keep = [p for p in grp["params"] if p.requires_grad]
        removed.extend([p for p in grp["params"] if not p.requires_grad])
        grp["params"] = keep
        kept += len(keep)
    return {"n_removed_tensors": len(removed), "n_kept_tensors": kept}


def _resolve_ppo_and_policy(runner) -> tuple[object, object, object]:
    """从 rsl_rl runner 取 (alg, optimizer, policy)（属性名兼容查找，缺即显式失败）。"""
    alg = getattr(runner, "alg", None) or getattr(runner, "algorithm", None)
    if alg is None:
        raise RuntimeError(f"runner 上找不到 alg/algorithm（{type(runner).__name__}）")
    optimizer = getattr(alg, "optimizer", None) or getattr(alg, "optim", None)
    if optimizer is None:
        raise RuntimeError(f"PPO 上找不到 optimizer/optim（{type(alg).__name__}）")
    policy = getattr(alg, "policy", None)
    if policy is None:
        raise RuntimeError(f"PPO 上找不到 policy（{type(alg).__name__}）")
    return alg, optimizer, policy


def _make_runner_cfg_dict(hv, cli, out_dir: Path, device: str, policy_kwargs: dict | None) -> dict:
    """rsl_rl runner cfg dict：PPO 配方逐字段继承 D064 官方 G1 rough（保可比性）。

    residual 臂改 policy.class_name + 透传 policy_kwargs（照 D065 的
    `runner_cfg.to_dict()` + `dict["policy"].update(...)` 手法）；direct/decoder 臂
    = D064 逐字（默认 ActorCritic）。
    """
    RslRlPpoActorCriticCfg = hv.RslRlPpoActorCriticCfg
    RslRlPpoAlgorithmCfg = hv.RslRlPpoAlgorithmCfg
    RslRlOnPolicyRunnerCfg = hv.RslRlOnPolicyRunnerCfg

    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.seed = cli.seed
    runner_cfg.device = device
    runner_cfg.num_steps_per_env = 24
    runner_cfg.max_iterations = cli.max_iterations
    runner_cfg.save_interval = 50
    runner_cfg.experiment_name = out_dir.name
    runner_cfg.empirical_normalization = False
    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    runner_cfg.algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    runner_cfg.run_name = f"{cli.arm}_{cli.terrain}"
    cfg_dict = runner_cfg.to_dict()
    if cli.arm == "residual":
        cfg_dict["policy"]["class_name"] = register_residual_policy_in_rsl_rl()
        cfg_dict["policy"].update(policy_kwargs or {})
    return cfg_dict


def _default_out_dir(arm: str, terrain: str, seed: int) -> Path:
    return REPO_ROOT / "apt_g1" / "outputs" / f"d067_{arm}_{terrain}_s{seed}"


# ------------------------------------------------------------- 训练主回路
def _run(cli, out_dir: Path, launcher_args) -> None:
    """App 起来后的主逻辑（三臂共用；residual 臂额外接线冻结分支 + 残差策略）。"""
    hv = _import_heavy()
    td = _import_train_g1_decoder()
    torch_mod = hv.torch
    device = getattr(launcher_args, "device", "cuda:0")
    reward_mode = resolve_reward_mode(cli.arm, cli.reward_mode)

    torch_mod.manual_seed(cli.seed)
    hv.np.random.seed(cli.seed)
    print(
        f"[CFG] out={out_dir} arm={cli.arm} terrain={cli.terrain} reward_mode={reward_mode} "
        f"num_envs={cli.num_envs} iters={cli.max_iterations} seed={cli.seed} device={device}",
        flush=True,
    )

    # ---- 身份信封（含三臂表项 + 奖励配方 + intent-pin + 地形/仿射契约声明） ----
    intent_pin = {
        "intent_pin_max": bool(cli.intent_pin_max),
        "value": float(cli.intent_pin_value) if cli.arm == "residual" else None,
    }
    envelope = build_identity_envelope(
        cli.arm,
        reward_mode,
        author_ckpt=cli.author_ckpt,
        onnx_path=cli.onnx_path,
        token_stats=cli.token_stats,
        intent_pin=intent_pin,
        terrain=cli.terrain,
    )
    (out_dir / "run.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[d067] run.json -> {out_dir / 'run.json'}", flush=True)

    # ---- env cfg ----
    token_info = None
    if cli.arm == "residual":
        if not cli.author_ckpt:
            raise RuntimeError("--arm residual 需要 --author-ckpt（token 源 = D061a author v0 ckpt）")
        cfg, cfg_how, recipe, token_info = _build_residual_env_cfg(hv, cli, device, suffix="main")
    else:
        cfg, cfg_how = td._build_env_cfg(
            hv.make_env_cfg, hv.mdp, hv.SonicDecoderActionTermCfg, cli.terrain, cli.arm, cli, device, suffix="main"
        )
        recipe = None
        if reward_mode == "max_forward":
            recipe = _apply_max_forward_recipe(cfg, hv, cli.forward_weight)
    print(f"[d067] env cfg via {cfg_how}（reward_mode={reward_mode} recipe={recipe}）", flush=True)
    if token_info is not None:
        # 地形契约/仿射统计进信封（reviewer M1/M2 要求）
        envelope["ctx_terrain"] = token_info["ctx_terrain"]
        envelope["env_terrain"] = token_info["env_terrain"]
        envelope["token_affine"]["token_bound"] = token_info["token_bound"]
        envelope["token_affine"]["token_alpha"] = token_info["token_alpha"]
        envelope["token_affine"]["token_stats"] = token_info["token_stats"]
        (out_dir / "run.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")

    # asymmetric 退化判定（同 D064）
    has_critic_group = getattr(cfg.observations, "critic", None) is not None
    asym_ok, rsl_ver = td._rsl_rl_supports_dict_obs()
    if has_critic_group and not asym_ok:
        cfg.observations.critic = None
        has_critic_group = False
        print(f"[WARN] rsl_rl {rsl_ver} 不支持 dict obs，退化单组（丢 critic）", flush=True)

    env = hv.ManagerBasedRLEnv(cfg=cfg)
    obs, _ = env.reset()
    if not (isinstance(obs, dict) and "policy" in obs):
        raise RuntimeError(f"env obs 应为含 'policy' 的 dict，实际 {type(obs)}")
    policy_obs = obs["policy"]
    obs_dim_policy = int(policy_obs.shape[-1])
    obs_dim_critic = int(obs["critic"].shape[-1]) if (isinstance(obs, dict) and obs.get("critic") is not None) else 0
    action_dim = int(env.action_manager.total_action_dim)
    print(f"[d067] obs(policy)={obs_dim_policy} obs(critic)={obs_dim_critic or '-'} action={action_dim}", flush=True)

    # ---- policy 接线（residual：冻结 decoder 分支 + token/proprio 切片） ----
    policy_kwargs = None
    wiring = None
    if cli.arm == "residual":
        # 动作通路守卫：残差臂必须落在 direct 动作通路（29 维 JointPositionAction）。
        # 若 make_env_cfg 签名回退把 lora_policy 变体拿成别的动作通路，此处即显式失败
        # （否则 μ(29) 与 env 期望动作维错位，训练静默崩）。
        _check(action_dim == ACTION_DIM, f"residual 臂 env 动作维应 {ACTION_DIM}，实 {action_dim}")
        policy_kwargs, wiring = _build_residual_policy_kwargs(hv, env, cli, device)
        print(f"[d067] residual 接线 {json.dumps(wiring, ensure_ascii=False)}", flush=True)
        # 身份信封补记实际接线（env 运行期才能确定）
        envelope["residual_wiring"] = wiring
        envelope["obs_dim_policy"] = obs_dim_policy
        envelope["obs_dim_critic"] = obs_dim_critic
        envelope["action_dim"] = action_dim
        (out_dir / "run.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")

    runner_cfg_dict = _make_runner_cfg_dict(hv, cli, out_dir, device, policy_kwargs)

    if cli.smoke:
        _smoke(cli=cli, out_dir=out_dir, env=env, policy_obs=policy_obs, obs_dim_policy=obs_dim_policy,
               action_dim=action_dim, hv=hv, runner_cfg_dict=runner_cfg_dict, device=device)
        return

    # ---- 正式训练 ----
    wrapped = hv.EnvWrapper(env)
    runner = hv.OnPolicyRunner(wrapped, runner_cfg_dict, log_dir=str(out_dir), device=device)
    if cli.arm == "residual":
        _alg, optimizer, policy = _resolve_ppo_and_policy(runner)
        if not isinstance(policy, ResidualDecoderPolicy):
            raise RuntimeError(
                f"runner 内 policy 类型 {type(policy).__name__} != ResidualDecoderPolicy"
                "（策略类注册失败，拒绝把残差臂训成 direct 臂）"
            )
        prune = _prune_frozen_params(optimizer)
        print(f"[d067] 可训集合（只 r + critic）：{prune}；{policy.trainable_report()}", flush=True)
    if cli.resume:
        if not cli.ckpt:
            raise RuntimeError("--resume 需要 --ckpt <path>")
        runner.load(str(Path(cli.ckpt).resolve()))
        print(f"[d067] resumed from {cli.ckpt}", flush=True)
    runner.learn(num_learning_iterations=runner_cfg_dict["max_iterations"])
    final_ckpt = out_dir / "model_final.pt"
    runner.save(str(final_ckpt))
    print(f"[d067] done. final ckpt -> {final_ckpt}", flush=True)
    env.close()


def _smoke(*, cli, out_dir, env, policy_obs, obs_dim_policy, action_dim, hv, runner_cfg_dict, device) -> None:
    """D067 冒烟：三臂 obs/action 形状 + residual 零初始化恒等 + learn x2 + ckpt 落盘。"""
    n = cli.num_envs
    _check(n == SMOKE_NUM_ENVS, f"smoke num_envs 应为 {SMOKE_NUM_ENVS}")
    _check(policy_obs.ndim == 2 and policy_obs.shape[0] == n, "policy obs 形状异常")
    print(f"[d067] smoke a) obs={obs_dim_policy} action={action_dim} PASS", flush=True)

    if cli.arm == "residual":
        token_slot = _policy_obs_slot(env, hv.torch, "author_token", TOKEN_DIM)
        proprio_slot = _policy_obs_slot(env, hv.torch, "sonic_proprio", PROPRIO_DIM)
        _check(action_dim == ACTION_DIM, f"residual 臂 action 维应 {ACTION_DIM}，实 {action_dim}")
        print(f"[d067] smoke b) residual 槽 token={token_slot} proprio={proprio_slot} PASS", flush=True)

    wrapped = hv.EnvWrapper(env)
    runner = hv.OnPolicyRunner(wrapped, runner_cfg_dict, log_dir=str(out_dir), device=device)
    if cli.arm == "residual":
        _alg, optimizer, policy = _resolve_ppo_and_policy(runner)
        _check(isinstance(policy, ResidualDecoderPolicy), "runner policy 非 ResidualDecoderPolicy")
        # 零初始化恒等：μ == 冻结 decoder(cat[token,proprio]) 逐位
        dec_obs = policy.decoder_obs(policy._flat_obs(policy_obs))
        _check(policy.mu_path.residual.is_zero(policy_obs[:, : policy.official_obs_dim], atol=0.0),
               "残差非零初始化")
        with hv.torch.no_grad():
            _check(hv.torch.equal(policy.mu(policy_obs), policy.mu_path.decoder(dec_obs)),
                   "μ 与冻结 decoder 输出非逐位一致")
        prune = _prune_frozen_params(optimizer)
        _check(policy.mu_path.only_residual_trainable(), "梯度隔离不通过（除 r 外仍有可训参数）")
        print(f"[d067] smoke b2) 零初始化恒等 + 梯度隔离 PASS {prune}", flush=True)
    runner.learn(num_learning_iterations=SMOKE_ITERS)
    ckpt = out_dir / "model_smoke.pt"
    runner.save(str(ckpt))
    _check(ckpt.exists() and ckpt.stat().st_size > 0, "ckpt 未落盘")
    print(f"[d067] smoke c) learn x{SMOKE_ITERS} PASS -> {ckpt}", flush=True)
    env.close()
    print(
        f"SMOKE PASS: arm={cli.arm} reward_mode={resolve_reward_mode(cli.arm, cli.reward_mode)} "
        f"obs={obs_dim_policy} action={action_dim} learn={SMOKE_ITERS}it ckpt={ckpt.name}",
        flush=True,
    )


# ------------------------------------------------------------- CLI
def build_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="D067 G2 输出端并行残差臂训练入口（三臂对照 + max-forward 奖励）"
    )
    ap.add_argument("--arm", choices=ARMS, default="residual",
                    help="direct/decoder（D064 逐字对照臂）/residual（a=decoder(token)+r(s)）")
    ap.add_argument("--terrain", choices=TERRAINS, default="rough")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iterations", type=int, default=3000)
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--reward-mode", choices=REWARD_MODES, default=DEFAULT_REWARD_MODE,
                    help="auto：residual->max_forward、direct/decoder->track（D064 可比）")
    ap.add_argument("--forward-weight", type=float, default=DEFAULT_FORWARD_WEIGHT,
                    help="max_forward 前进速度项权重")
    ap.add_argument("--author-ckpt", default="", help="D061a author v0 ckpt（residual 臂 token 源，必填）")
    ap.add_argument("--intent-pin-max", action=argparse.BooleanOptionalAction, default=True,
                    help="推理期把 author ctx 命令槽钉最大速度档（默认开）")
    ap.add_argument("--intent-pin-value", type=float, default=DEFAULT_INTENT_PIN_VALUE,
                    help="钉档值（默认 1.0 = 官方 G1 rough lin_vel_x 上界，见假设 A2）")
    ap.add_argument("--token-stats", default="", help="官方 g1-mode token 统计 npz（decoder 仿射/author 逆仿射）")
    ap.add_argument("--onnx-path", default="", help="SONIC ONNX decoder（残差臂冻结分支；缺省工厂默认）")
    ap.add_argument("--token-alpha", type=float, default=None)
    ap.add_argument("--token-bound", choices=("none", "tanh"), default=None)
    ap.add_argument("--sigma", type=float, default=None, help="残差臂 σ 值（缺省=init_noise_std=1.0）")
    ap.add_argument("--sigma-trainable", action=argparse.BooleanOptionalAction, default=False,
                    help="残差臂 σ 是否可训（默认关 = PPO 只训 r）")
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--smoke", action="store_true", help="冒烟：num_envs=8、iters=2")
    ap.add_argument("--selftest", action="store_true", help="本机纯 torch 自测（不 import isaaclab）")
    return ap


def main() -> None:
    cli = build_args().parse_args()

    # ---- 纯本机分支：绝不 import isaac ----
    if cli.selftest:
        run_selftest()
        return

    if cli.resume and not cli.ckpt:
        build_args().error("--resume 需要 --ckpt <path>")
    if cli.arm == "residual" and not cli.author_ckpt:
        build_args().error("--arm residual 需要 --author-ckpt <D061a ckpt>（token 源）")
    if cli.arm == "residual" and not cli.token_stats:
        build_args().error("--arm residual 需要 --token-stats <npz>（author 逆仿射 token 统计）")
    if cli.arm == "decoder" and not cli.token_stats:
        build_args().error("--arm decoder 需要 --token-stats <npz>（token 仿射统计）")

    if cli.smoke:
        cli.num_envs = SMOKE_NUM_ENVS
        cli.max_iterations = SMOKE_ITERS

    out_dir = Path(cli.output_dir) if cli.output_dir else _default_out_dir(cli.arm, cli.terrain, cli.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # AppLauncher 启动链（app 起来之前不 import isaaclab 栈）
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = cli.num_envs
    launcher_args.headless = cli.headless
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    try:
        _run(cli, out_dir, launcher_args)
    except Exception as exc:  # noqa: BLE001
        if cli.smoke:
            print(f"SMOKE FAIL: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()