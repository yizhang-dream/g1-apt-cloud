"""D065 G0 对齐冒烟门：ORCS/ViBe 发布 ckpt 的 decoder(+LoRA adapter) 与我方 SONIC
decoder 逐层对齐比对 + Dodge 任务 adapter 前向冒烟（纯 torch，零仿真、零训练、
CPU 可跑；只需 torch/numpy/stdlib + 本仓 `isaac/sonic_decoder_torch.py`）。

背景（ORCS_SURVEY.md §2/§4-A 第一步）：ViBe/ORCS（arXiv 2609.09918）把 rank-16
LoRA 挂在**与我们同一份** SONIC `sonic_release` 解码器上；survey 在服务器上已实测
「中间层 1–5 逐位相等、首层列置换、末层行置换，置换 = G1 IsaacLab↔MuJoCo 关节序」，
但**没做端到端数值冒烟**。本脚本把那条结论落成可重复的三门：

  g0a 权重对齐   7 层 base 权重逐层 maxdiff + 置换复原（首层列置换 / 末层行置换），
                 并判定两种置换假设（A=仅 proprio 部分被置换、token 列恒等；
                 B=含 token 列）哪种成立。
  g0b metadata   ckpt 顶层非张量键（iter/infos/cfg 等）全量打印 + adapter
                 rank/alpha 尽力提取（alpha 不落 state_dict，找不到就标 infer）。
  g0c 前向冒烟   按 ORCS 结构（base 线性 + scale·up(down(x)) 逐层相加）走一遍
                 Dodge 21 维 augmentation 流，检查 finite/形状；另做「我方 decoder
                 （按置换喂输入/反置换读输出）≈ 对方 base（adapter 置零）」的
                 端到端数值交叉验证。

三个门各自独立 PASS/FAIL，写 JSON + stdout 人话判读。

------------------------------------------------------------------------------
实现口径（逐条对照上游源码，均为「照抄实现」，运行时**不 import orcs/rsl_rl**）
------------------------------------------------------------------------------
- Adapter（`lok-i/rsl_rl` `rsl_rl/modules/mlp_with_adapter.py::Adapter`）：
  低秩分支 `down: (in→rank)`、`up: (rank→out)`，`up` 零初始化，`scale = alpha/rank`；
  `rank <= 0` 或 `rank >= min(in,out)` 时**静默退化为全秩稠密 delta**（单 Linear，
  权重零初始化），此时 `scale = alpha`（LoRA 惯例的除 rank 不再发生）。前向
  `scale * up(down(x))` 或 `scale * delta(x)`。
- 逐层前向（同文件 `MLPWithAdapter.forward`）：**adapter 与 base 线性层并联、吃同一
  份运行激活**，且第 i 层 adapter（i>0）的输入 = 第 i-1 层 base 输出之和（激活前）：
      h = base[0](obs);            h += adapters[0](adapter_obs)
      for i in 1..6:  h = SiLU(h); h += adapters[i](h) ... 即 out = base[i](h) + ad[i](h)
  注意不同实现顺序（先激活再各自走 base/adapter）会算错——原实现是 base 与 adapter
  共用同一个输入 h，二者相加后**下一次迭代开头**才过激活。
- 层名映射：base 是 `nn.Sequential([Linear, Act]*N + Linear)`（`rsl_rl/modules/mlp.py`
  的 MLP 即 nn.Sequential 子类），故 ckpt 键为
      `decoder.base.{0,2,4,6,8,10,12}.{weight,bias}`  ↔ 我方 7 个 Linear（同序）；
      `decoder.adapters.{0..6}.{down,up}.weight`（低秩）或 `.delta.weight`（全秩）；
      `adapter_normalizer.{_mean,_std,_var,count}`（EmpiricalNormalization，
      前向 `(x - mean) / (std + eps)`，eps=1e-2）。
  键前缀（`prefix`）自适应：扫描 state_dict 找 `decoder.base.<i>.weight` 模式定位。
- 首层输入 994 维的列置换：token 列 0–63 恒等，proprio 五通道
  ang_vel(10×3) / joint_pos(10×29) / joint_vel(10×29) / last_act(10×29) /
  gravity(10×3)（`sonic_decoder_torch.py:83-91` 的 parts 顺序），关节类通道
  逐帧内按同一关节置换重排；末层 29 维输出行置换为**同一**关节置换。
  本脚本 data-driven 复原置换（先按字节精确匹配、失败再走 L2 容差匹配），
  再与 `G1_MUJOCO_TO_ISAACLAB_DOF`（硬编码副本，来源见常量处注释）比对方向。
- Dodge augmentation 流 21 维（`orcs/tasks/dodge/observation_cfgs.py:115-131`
  `augmentation_group` 里 `_grp({**ball_state_terms(), **robot_root_state_terms(),
  **robot_motion_cmd_terms})` 的 dict 插入序 = 拼接序）：
      [0:3)   ball_pos_b       球相对机器人根、yaw 系（dodge/mdp/observations.py:24）
      [3:6)   ball_vel_b       球相对速度（同文件 :33）
      [6:9)   robot_root_pos_env            根在 env 系位置（core/mdp/observations.py:39）
      [9:12)  robot_root_lin_vel_b          机体系线速度（同上 base_lin_vel）
      [12:15) robot_root_ang_vel_b          机体系角速度
      [15:18) robot_root_lin_vel_cmd        参考锚系线速度命令（:54）
      [18:21) robot_root_ang_vel_cmd        参考锚系角速度命令（:62）
  站立常量参考：dodge 的参考动作是「保持站立」，命令恒零（observations_cfgs
  docstring：「the reference is a held stand」+「constant zero here」）；球在两次投掷
  之间被**钉在 env 原点上方 park_z=8.0 m、速度清零**（dodge/mdp/events.py:258/283-291），
  故 ball_pos_b ≈ (0, 0, park_z - 根高)、ball_vel_b = 0。根高本身用本脚本引入的近似常量
  `DODGE_ROOT_HEIGHT_M = 0.793`（只影响这一个分量；属显式假设，见 KNOWN_LIMITATIONS）。

用法（服务器 .venv_isaac，cwd=仓根）::

    python apt_g1/probe/probe_orcs_align_smoke.py \
        --ckpt /path/to/Orcs-Dodge-AdaptSonic/model_7500.pt \
        --task dodge --out apt_g1/outputs/d065/g0_smoke.json

    # 本机（无 onnx/ckpt，仅自检门逻辑）：
    python apt_g1/probe/probe_orcs_align_smoke.py --selftest

`--ckpt` 与 `--selftest` 互斥；`--out` 默认 `outputs/d065/g0_smoke.json`，其中以
`outputs/` 开头的相对路径解析到 **`apt_g1/outputs/`**（仓内 probe 产物家、已
gitignore；`--out` 给绝对路径则直通，其他相对路径按当前工作目录）。

身份信封：ckpt sha256/字节数、ONNX md5+sha256、全部张量键形状清单、本脚本 sha256、
关节置换常量摘要，全部进 JSON。

许可提醒：ORCS 发布 ckpt 内含 NVIDIA SONIC 权重，受 **NVIDIA Open Model License**
约束（内部对照可用；**不得随公开仓分发**）。本脚本只读不复制权重，也不把权重写进
任何产物——JSON 里只有形状与统计量。

已知限制见常量区 `KNOWN_LIMITATIONS`（同时回写进 JSON notes）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from types import ModuleType

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/apt_g1/probe
_APT_G1 = os.path.dirname(_HERE)                             # <repo>/apt_g1
_REPO_ROOT = os.path.dirname(_APT_G1)                        # <repo>
for _p in (_REPO_ROOT, _APT_G1):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXPERIMENT = "D065"
GATE_NAME = "G0"
SCRIPT_TAG = "probe_orcs_align_smoke"

N_JOINTS = 29
HISTORY_LEN = 10
TOKEN_DIM = 64
ANG_VEL_DIM = 3
GRAVITY_DIM = 3
SONIC_JOINT_WIDTH = HISTORY_LEN * N_JOINTS        # 290
SONIC_VEC3_WIDTH = HISTORY_LEN * ANG_VEL_DIM      # 30
DECODER_OBS_DIM = TOKEN_DIM + SONIC_VEC3_WIDTH + 3 * SONIC_JOINT_WIDTH + SONIC_VEC3_WIDTH  # 994
AUG_DIM_DODGE = 21
EXPECTED_NUM_LAYERS = 7

# (通道名, 起始, 每帧宽度, 是否关节类通道) —— 我方 994 契约分段
# （sonic_decoder_torch.py:83-91 build_decoder_obs / sonic_action_term.py:25-34）
PROPRIO_CHANNELS = [
    ("ang_vel", TOKEN_DIM, ANG_VEL_DIM, False),
    ("joint_pos", TOKEN_DIM + SONIC_VEC3_WIDTH, N_JOINTS, True),
    ("joint_vel", TOKEN_DIM + SONIC_VEC3_WIDTH + SONIC_JOINT_WIDTH, N_JOINTS, True),
    ("last_act", TOKEN_DIM + SONIC_VEC3_WIDTH + 2 * SONIC_JOINT_WIDTH, N_JOINTS, True),
    ("gravity", TOKEN_DIM + SONIC_VEC3_WIDTH + 3 * SONIC_JOINT_WIDTH, GRAVITY_DIM, False),
]

# G1_MUJOCO_TO_ISAACLAB_DOF 的硬编码副本（语义：perm[i] = SONIC/IsaacLab 序第 i 个关节
# 在 MuJoCo 序里的下标；用法同 sonic_action_term.py:365 SONIC_DEFAULT_ANGLES_MUJOCO[perm]）。
# 权威源 gear_sonic/envs/manager_env/robots/g1.py（该模块顶部 import isaaclab，本机不可导入）；
# 同一硬编码副本已见于 eval_author_v0_decoder.py:208-218（那边叫 A1 假设）。
G1_MUJOCO_TO_ISAACLAB_DOF = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17,
    24, 18, 25, 19, 26, 20, 27, 21, 28,
)
M2I = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF, dtype=np.int64)
M2I_INV = np.argsort(M2I).astype(np.int64)   # 反向：MuJoCo 序 -> SONIC/IsaacLab 序

# 判读阈值
TOL_WEIGHT = 1e-5      # 层权重逐元素绝对差上限（survey 实测为 0；给浮点搬运留余量）
TOL_FORWARD = 1e-3     # 交叉验证：我方 decoder vs 对方 base(adapter=0) 输出绝对差上限
MATCH_ATOL = 1e-6      # 置换复原的列/行匹配容差（优先走字节精确匹配）

# ORCS Dodge 站立常量参考（见模块 docstring 引用处）
DODGE_PARK_Z_M = 8.0        # dodge/mdp/events.py:258 park_z
DODGE_ROOT_HEIGHT_M = 0.793  # G1 站立盆骨高近似常量（本脚本引入；只影响 aug 的 ball_pos_b z 分量）
AUG_NORMALIZER_EPS = 1e-2   # rsl_rl/modules/normalization.py EmpiricalNormalization.eps

LICENSE_NOTE = (
    "ORCS 发布 ckpt 内含 NVIDIA SONIC 权重，受 NVIDIA Open Model License 约束："
    "仅作内部对照/对齐比读，不得随公开仓分发；本脚本只读、只写形状与统计量。"
)

KNOWN_LIMITATIONS = [
    "alpha 不进 state_dict（rsl_rl 侧为 Python 属性），只能从 ckpt 非张量 metadata 里找；"
    "找不到时 JSON 标 alpha_source=not_in_state_dict，rank 从 down.weight 形状推。"
    "默认 rank=16/alpha=1.0 是 ORCS 源码默认值（rl.py:102-103），不是本脚本实测。",
    "Dodge aug 21 维分段顺序按 observation_cfgs.py 的 dict 插入序；若 mjlab "
    "ObservationManager 实际按字母序或其他序拼接，则分段语义错位（形状仍对、门不受影响，"
    "但 aug 值的物理解读会偏）。",
    "站立常量参考中 ball_pos_b 的 z 分量依赖 DODGE_ROOT_HEIGHT_M=0.793 这个近似常量，"
    "且假设机器人站在 env 原点正下方；这只是冒烟输入，不是 dodge 的动力学仿真。",
    "g0c 只用 ckpt 里 adapter_normalizer 的统计量；若该键缺失则按恒等归一化并在 JSON 注明。",
    "置换方向（m2i vs inv_m2i）两者都接受并如实报告；本脚本不做物理落地验证"
    "（对齐结论仍需 G1 级别闭环/权限实测确认）。",
    "v1 只支持 dodge（PerLoco/UOLM 需要地形扫描与物体状态，离线造不出）。",
    "--selftest 的输入/输出维用真实 994/29（置换逻辑依赖它们），但中间层宽度取小值"
    "（32/32/16/16/8/8）以省时间——门逻辑只读张量形状，与真实宽度无关。",
]


# ---------------------------------------------------------------------------
# Layer 1: 通用小工具
# ---------------------------------------------------------------------------
def file_digests(path: str) -> dict:
    """单次读取算 md5 + sha256 + 字节数。"""
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    total = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            md5.update(chunk)
            sha.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest(), "bytes": total}


def text_digests(text: str) -> dict:
    payload = text.encode("utf-8")
    return {
        "md5": hashlib.md5(payload).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def jsonable(obj, depth: int = 0, max_items: int = 60, max_str: int = 300):
    """把 ckpt 里的任意对象转成可 JSON 化的结构（有界深度/条数）。"""
    if depth > 6:
        return repr(obj)[:max_str]
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else repr(obj)
    if isinstance(obj, torch.Tensor):
        return {"__tensor__": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {
            str(k): jsonable(v, depth + 1, max_items, max_str)
            for k, v in list(obj.items())[:max_items]
        }
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v, depth + 1, max_items, max_str) for v in list(obj)[:max_items]]
    return repr(obj)[:max_str]


def tensor_shape_map(state_dict: dict) -> dict:
    """全部张量键 -> 形状（身份信封用；字典序）。"""
    out = {}
    for key in sorted(state_dict):
        val = state_dict[key]
        if isinstance(val, torch.Tensor):
            out[key] = [int(x) for x in val.shape]
    return out


def joint_perm_digest() -> dict:
    """关节置换常量摘要（写进身份信封，可与权威源核对）。"""
    payload = json.dumps({"g1_mujoco_to_isaaclab_dof": [int(x) for x in M2I]},
                         separators=(",", ":"))
    digest = text_digests(payload)
    return {
        "g1_mujoco_to_isaaclab_dof": [int(x) for x in M2I],
        "semantics": "perm[i] = SONIC/IsaacLab 序第 i 个关节在 MuJoCo 序中的下标",
        "md5": digest["md5"],
    }


# ---------------------------------------------------------------------------
# Layer 2: 置换构造 / 复原 / 结构分析
# ---------------------------------------------------------------------------
def input_perm_from_joint_map(jmap: np.ndarray) -> np.ndarray:
    """我方 994 维输入 idx -> 对方 994 维输入 idx 的置换。

    token 列、ang_vel、gravity 块恒等；joint_pos/joint_vel/last_act 三块逐帧内
    按 jmap 重排（jmap[i] = 我方第 i 个关节在对方序里的下标）。
    """
    jmap = np.asarray(jmap, dtype=np.int64)
    if jmap.shape != (N_JOINTS,) or sorted(jmap.tolist()) != list(range(N_JOINTS)):
        raise ValueError(f"jmap 必须是 0..{N_JOINTS - 1} 的排列，得到 {jmap.tolist()}")
    perm = np.arange(DECODER_OBS_DIM, dtype=np.int64)
    for start, width, is_joint in ((c[1], c[2], c[3]) for c in PROPRIO_CHANNELS):
        if not is_joint:
            continue
        for f in range(HISTORY_LEN):
            lo = start + f * width
            perm[lo:lo + width] = lo + jmap
    return perm


def output_perm_from_joint_map(jmap: np.ndarray) -> np.ndarray:
    """我方 29 维输出行 idx -> 对方 29 维输出行 idx（同一关节映射）。"""
    return np.asarray(jmap, dtype=np.int64).copy()


def is_permutation(arr: np.ndarray) -> bool:
    a = np.asarray(arr)
    return a.ndim == 1 and sorted(a.tolist()) == list(range(len(a)))


def analyze_input_perm(perm: np.ndarray) -> dict:
    """把复原出的列置换拆成「假设 A/B + 逐通道块结构 + 关节图」。"""
    perm = np.asarray(perm, dtype=np.int64)
    out: dict = {
        "is_bijection": bool(is_permutation(perm)) if perm.shape == (DECODER_OBS_DIM,) else False,
        "token_identity": None,
        "hypothesis": None,
        "per_channel": {},
        "joint_map": None,
        "frames_consistent": None,
        "channels_consistent": None,
        "structure_ok": False,
    }
    if not out["is_bijection"]:
        return out
    out["token_identity"] = bool(np.array_equal(perm[:TOKEN_DIM], np.arange(TOKEN_DIM)))
    out["hypothesis"] = "A" if out["token_identity"] else "B"
    if not out["token_identity"]:
        out["hypothesis_note"] = (
            "token 列非恒等：置换不只是 proprio 部分（假设 B），D065 前提"
            "「唯一实质差异是关节序」不成立"
        )
    joint_maps: dict[str, np.ndarray] = {}
    for name, start, width, is_joint in PROPRIO_CHANNELS:
        if not is_joint:
            ident = bool(np.array_equal(perm[start:start + HISTORY_LEN * width],
                                        np.arange(start, start + HISTORY_LEN * width)))
            out["per_channel"][name] = {"identity": ident}
            continue
        frames = []
        ok = True
        for f in range(HISTORY_LEN):
            lo = start + f * width
            block = perm[lo:lo + width] - lo
            if not is_permutation(block):
                ok = False
                break
            frames.append(block)
        if not ok:
            out["per_channel"][name] = {"identity": False, "per_frame_perm": None}
            out["structure_ok"] = False
            out["structure_note"] = (
                f"{name} 块的逐帧重排不是合法排列（0..28）——置换不是「proprio 逐帧内关节重排」"
            )
            out["joint_map"] = None
            return out
        same = all(np.array_equal(frames[0], fr) for fr in frames)
        out["per_channel"][name] = {
            "identity": bool(np.array_equal(frames[0], np.arange(N_JOINTS))),
            "per_frame_perm_same": bool(same),
            "per_frame_perm": [int(x) for x in frames[0]],
        }
        if not same:
            out["structure_ok"] = False
            out["structure_note"] = f"{name} 的 10 帧重排互不相同（帧间不一致）"
            out["joint_map"] = None
            return out
        joint_maps[name] = frames[0]
    out["frames_consistent"] = all(
        c.get("per_frame_perm_same", c.get("identity", False))
        for c in out["per_channel"].values()
    )
    jms = list(joint_maps.values())
    if not jms:
        out["channels_consistent"] = False
        return out
    out["channels_consistent"] = all(np.array_equal(jms[0], jm) for jm in jms)
    out["joint_map"] = [int(x) for x in jms[0]]
    non_joint_identity = all(
        c["identity"] for n, c in out["per_channel"].items() if n not in joint_maps
    )
    out["structure_ok"] = bool(
        out["token_identity"] and out["channels_consistent"] and non_joint_identity
    )
    return out


def joint_map_direction(jmap) -> str:
    """复原出的关节图与 G1_MUJOCO_TO_ISAACLAB_DOF 的方向比对。"""
    if jmap is None:
        return "n/a"
    arr = np.asarray(jmap, dtype=np.int64)
    if arr.shape != (N_JOINTS,):
        return "n/a"
    if np.array_equal(arr, M2I):
        return "matches_g1_mujoco_to_isaaclab_dof"
    if np.array_equal(arr, M2I_INV):
        return "matches_inverse_of_g1_mujoco_to_isaaclab_dof"
    return "neither"


def match_vectors(our: torch.Tensor, their: torch.Tensor, atol: float) -> tuple[np.ndarray, dict]:
    """给两组同形状向量逐条配对，返回 perm（our i -> their j，-1 = 未配上）。

    先按 float32 字节精确匹配（survey 的逐位相等口径），失败再退到 L2 距离容差匹配。
    """
    our = our.detach().to(torch.float32).contiguous()
    their = their.detach().to(torch.float32).contiguous()
    n = int(our.shape[0])
    info = {"method": "exact_bytes", "n_matched": 0, "bijective": False,
            "max_residual": None, "n_duplicate_hits": 0}
    if their.shape != our.shape or n == 0:
        info["method"] = "shape_mismatch"
        return np.full(n, -1, dtype=np.int64), info

    our_np = our.numpy()
    their_np = their.numpy()
    index: dict[bytes, list[int]] = {}
    for j in range(n):
        index.setdefault(their_np[j].tobytes(), []).append(j)
    perm = np.full(n, -1, dtype=np.int64)
    used: set[int] = set()
    dup = 0
    for i in range(n):
        cands = index.get(our_np[i].tobytes())
        if not cands:
            continue
        pick = next((c for c in cands if c not in used), None)
        if pick is None:
            dup += 1
            continue
        perm[i] = pick
        used.add(pick)
    info["n_matched"] = int((perm >= 0).sum())
    info["n_duplicate_hits"] = dup
    if info["n_matched"] == n and is_permutation(perm):
        info["bijective"] = True
        info["max_residual"] = 0.0
        return perm, info

    # 容差匹配兜底
    info["method"] = "l2_tolerant"
    dist = torch.cdist(our, their)
    best = dist.argmin(dim=1)
    resid = dist.gather(1, best.view(-1, 1)).view(-1)
    best_np = best.numpy().astype(np.int64)
    resid_np = resid.numpy()
    perm2 = np.where(resid_np <= atol, best_np, -1)
    info["n_matched"] = int((perm2 >= 0).sum())
    info["max_residual"] = float(resid_np.max()) if n else None
    info["bijective"] = bool(is_permutation(perm2))
    if not info["bijective"]:
        info["exact_attempt"] = {
            "n_matched": int((perm >= 0).sum()),
            "n_duplicate_hits": dup,
        }
    return perm2, info


def recover_input_perm(W_our: torch.Tensor, W_their: torch.Tensor, atol: float):
    """首层：(out,in) 权重，按列（输入维）配对。"""
    return match_vectors(W_our.t().contiguous(), W_their.t().contiguous(), atol)


def recover_output_perm(W_our: torch.Tensor, W_their: torch.Tensor, atol: float):
    """末层：(out,in) 权重，按行（输出维）配对。"""
    return match_vectors(W_our, W_their, atol)


# ---------------------------------------------------------------------------
# Layer 3: ckpt 解析
# ---------------------------------------------------------------------------
def _permissive_pickle_module() -> ModuleType:
    """给 torch.load 的兜底 pickle module：未知类退化成哑元（cmp.py 同款思路）。

    只在常规 torch.load 失败时使用（例如 ckpt 里带了未安装依赖类的 metadata）。
    """
    import pickle

    real_find_class = pickle.Unpickler.find_class

    class _Stub:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self._state = state

    class _PermissiveUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return real_find_class(self, module, name)
            except Exception:
                return type(name, (_Stub,), {"__module__": module})

    mod = ModuleType("permissive_pickle")
    mod.Unpickler = _PermissiveUnpickler
    mod.load = pickle.load
    mod.loads = pickle.loads
    mod.dump = pickle.dump
    mod.dumps = pickle.dumps
    mod.Pickler = pickle.Pickler
    return mod


def load_ckpt(path: str) -> tuple[object, str]:
    """torch.load(map_location=cpu)；失败再试容错 pickle。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False), "torch.load(weights_only=False)"
    except Exception as exc:  # noqa: BLE001 - 兜底路径要吞掉任意反序列化异常
        first_err = f"{type(exc).__name__}: {exc}"
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False,
                         pickle_module=_permissive_pickle_module())
        return raw, f"torch.load(permissive_pickle)  [首次失败: {first_err}]"
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"[FATAL] ckpt 反序列化失败：{path}\n  常规 torch.load: {first_err}\n"
            f"  容错 pickle: {type(exc).__name__}: {exc}"
        )


BASE_KEY_PATTERNS = (
    # 首选：SonicWithAdapterModel 里 decoder 是 MLPWithAdapter，base 是 nn.Sequential
    r"^(?P<prefix>.*?)decoder\.base\.(?P<idx>\d+)\.(?P<kind>weight|bias)$",
    # 兜底：actor 直接就是 MLPWithAdapter（无 `decoder.` 包装）
    r"^(?P<prefix>.*?)(?:^|\.)base\.(?P<idx>\d+)\.(?P<kind>weight|bias)$",
)
ADAPTER_KEY_PATTERNS = (
    r"^(?P<prefix>.*?)decoder\.adapters\.(?P<idx>\d+)\.(?P<kind>down|up|delta)\.weight$",
    r"^(?P<prefix>.*?)(?:^|\.)adapters\.(?P<idx>\d+)\.(?P<kind>down|up|delta)\.weight$",
)
SD_PROBE_PATTERN = r"(?:^|\.)(?:decoder\.)?base\.\d+\.weight$"


def _count_decoder_base_keys(sd: dict) -> int:
    """state_dict 里像 decoder.base 的键数（容器定位用，宽于正式解析）。"""
    import re

    probe = re.compile(SD_PROBE_PATTERN)
    return sum(1 for k in sd if isinstance(k, str) and probe.search(k))


def find_state_dict_container(raw) -> tuple[str, dict, list[str]]:
    """在 ckpt 顶层容器里定位 actor 权重 dict（前缀自适应）。"""
    notes: list[str] = []
    candidates: list[tuple[str, dict, int]] = []
    if isinstance(raw, dict):
        top_hits = _count_decoder_base_keys(raw)
        if top_hits:
            candidates.append(("<top-level>", raw, top_hits))
        for key, val in raw.items():
            if isinstance(val, dict):
                hits = _count_decoder_base_keys(val)
                if hits:
                    candidates.append((str(key), val, hits))
    if not candidates:
        raise SystemExit(
            "[FATAL] 在 ckpt 里找不到含 'decoder.base.' 的权重 dict——"
            f"顶层键：{list(raw.keys()) if isinstance(raw, dict) else type(raw)}"
        )
    candidates.sort(key=lambda c: -c[2])
    name, sd, hits = candidates[0]
    if len(candidates) > 1:
        notes.append(f"候选容器 {[c[0] for c in candidates]}，取 decoder.base 命中最多者 '{name}'（{hits} 键）")
    return name, sd, notes


def parse_decoder_state_dict(sd: dict) -> dict:
    """抽出 base 7 层 / adapters / adapter_normalizer / 键前缀（多套键模式依次尝试）。"""
    import re

    base_regexes = [re.compile(p) for p in BASE_KEY_PATTERNS]
    ad_regexes = [re.compile(p) for p in ADAPTER_KEY_PATTERNS]

    def collect(regexes: list) -> tuple[dict, set, int]:
        """依次试每套键模式，返回 (entries_by_idx, prefixes, 命中的模式序号或 -1)。"""
        for pi, rx in enumerate(regexes):
            entries: dict[int, dict] = {}
            prefixes: set[str] = set()
            for key, val in sd.items():
                if not isinstance(key, str) or not isinstance(val, torch.Tensor):
                    continue
                m = rx.match(key)
                if not m:
                    continue
                prefixes.add(m.group("prefix"))
                entries.setdefault(int(m.group("idx")), {})[m.group("kind")] = val
            if entries:
                return entries, prefixes, pi
        return {}, set(), -1

    base_entries, base_prefixes, base_pi = collect(base_regexes)
    ad_entries, ad_prefixes, ad_pi = collect(ad_regexes)
    if not base_entries:
        raise SystemExit(
            "[FATAL] state_dict 里没有任何 decoder.base.<i>.{weight,bias} 键"
            f"（顶层样本键：{sorted(k for k in sd if isinstance(k, str))[:8]}）"
        )
    prefixes = base_prefixes | ad_prefixes
    prefix = sorted(prefixes)[0] if prefixes else ""
    idxs = sorted(base_entries)
    if len(idxs) != EXPECTED_NUM_LAYERS:
        raise SystemExit(
            f"[FATAL] decoder.base 层数 {len(idxs)} != 期望 {EXPECTED_NUM_LAYERS}（键序号 {idxs}）"
        )
    base = []
    for i in idxs:
        ent = base_entries[i]
        if "weight" not in ent or "bias" not in ent:
            raise SystemExit(f"[FATAL] decoder.base.{i} 缺 weight/bias")
        base.append({"index": int(i), "weight": ent["weight"], "bias": ent["bias"]})
    adapters = []
    for k in range(len(base)):
        ent = ad_entries.get(k)
        if not ent:
            adapters.append(None)
            continue
        if "delta" in ent:
            w = ent["delta"]
            adapters.append({
                "kind": "full_rank",
                "delta": w,
                "rank": int(min(w.shape)),
                "in_features": int(w.shape[1]),
                "out_features": int(w.shape[0]),
            })
        elif "down" in ent and "up" in ent:
            down, up = ent["down"], ent["up"]
            adapters.append({
                "kind": "low_rank",
                "down": down,
                "up": up,
                "rank": int(down.shape[0]),
                "in_features": int(down.shape[1]),
                "out_features": int(up.shape[0]),
            })
        else:
            adapters.append({"kind": "incomplete", "keys": sorted(ent)})
    normalizer = {"mean": None, "std": None, "var": None, "count": None, "keys": []}
    for key, val in sd.items():
        if not isinstance(key, str) or "adapter_normalizer" not in key:
            continue
        normalizer["keys"].append(key)
        if not isinstance(val, torch.Tensor):
            continue
        if key.endswith("._mean"):
            normalizer["mean"] = val
        elif key.endswith("._std"):
            normalizer["std"] = val
        elif key.endswith("._var"):
            normalizer["var"] = val
        elif key.endswith("count"):
            normalizer["count"] = val
    normalizer["keys"].sort()
    encoder_keys = sorted(k for k in sd if isinstance(k, str) and "encoder." in k)
    return {
        "prefix": prefix,
        "base": base,
        "adapters": adapters,
        "normalizer": normalizer,
        "encoder_keys": encoder_keys,
        "n_keys": len(sd),
        "base_key_pattern": BASE_KEY_PATTERNS[base_pi] if base_pi >= 0 else None,
        "adapter_key_pattern": ADAPTER_KEY_PATTERNS[ad_pi] if ad_pi >= 0 else None,
        "n_adapter_layers": sum(1 for a in adapters if a is not None),
    }


def extract_our_layers(net: torch.nn.Module) -> tuple[list[dict], list[str]]:
    """从我方 decoder（Sequential(Linear,SiLU,...)）抽出 7 个 Linear 与激活名单。"""
    layers: list[dict] = []
    acts: list[str] = []
    for mod in net:
        if isinstance(mod, torch.nn.Linear):
            layers.append({
                "weight": mod.weight.detach(),
                "bias": mod.bias.detach() if mod.bias is not None else None,
            })
        else:
            acts.append(type(mod).__name__)
    return layers, acts


def build_mlp_from_layers(layers: list[dict], activation: str = "SiLU") -> torch.nn.Sequential:
    """按 [(W,b)] 造 Sequential(Linear, Act, ..., Linear)（selftest 用）。"""
    act_cls = getattr(torch.nn, activation)
    mods: list[torch.nn.Module] = []
    for i, ent in enumerate(layers):
        w, b = ent["weight"], ent["bias"]
        lin = torch.nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=b is not None)
        with torch.no_grad():
            lin.weight.copy_(w)
            if b is not None:
                lin.bias.copy_(b)
        mods.append(lin)
        if i < len(layers) - 1:
            mods.append(act_cls())
    net = torch.nn.Sequential(*mods)
    net.eval()
    return net


def resolve_out_path(out: str) -> str:
    """--out 解析：绝对路径直通；以 `outputs/` 开头的相对路径落到 `apt_g1/outputs/`
    （仓内 probe 产物家，已 gitignore），其余相对路径按当前工作目录。"""
    if os.path.isabs(out):
        return out
    parts = out.replace("\\", "/").split("/")
    if parts and parts[0] == "outputs":
        return os.path.join(_APT_G1, *parts)
    return os.path.abspath(out)


def resolve_onnx_path(explicit: str | None) -> tuple[str, str]:
    """--onnx 显式优先；否则 apt_g1/model_decoder.onnx -> gear_sonic_deploy/... 依次探测。"""
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"[FATAL] --onnx 不存在：{explicit}")
        return os.path.abspath(explicit), "explicit(--onnx)"
    cands = [
        (os.path.join(_APT_G1, "model_decoder.onnx"), "default(apt_g1/model_decoder.onnx)"),
        (os.path.join(_REPO_ROOT, "gear_sonic_deploy", "policy", "release", "model_decoder.onnx"),
         "fallback(gear_sonic_deploy/policy/release/model_decoder.onnx)"),
    ]
    for path, src in cands:
        if os.path.isfile(path):
            return os.path.abspath(path), src
    raise SystemExit(
        "[FATAL] 找不到 SONIC decoder ONNX，候选：\n  " +
        "\n  ".join(p for p, _ in cands) + "\n用 --onnx 显式指定。"
    )


def load_our_decoder_net(onnx_path: str) -> torch.nn.Sequential:
    """复用 sonic_decoder_torch 的 ONNX 抽权重路径（该模块 import onnx，故延迟 import）。"""
    try:
        from isaac.sonic_decoder_torch import SonicTorchDecoder  # noqa: PLC0415
    except ImportError as exc:  # 本机无 onnx 时给可行动报错
        raise SystemExit(
            f"[FATAL] 导入 isaac.sonic_decoder_torch 失败（该模块需要 onnx）：{exc}\n"
            "  本机（Windows）无 onnx 属预期——真模式请在服务器 .venv_isaac 跑，"
            "本机只跑 --selftest。"
        ) from exc
    decoder = SonicTorchDecoder(onnx_path, device="cpu")
    return decoder.net


# ---------------------------------------------------------------------------
# Layer 4: ORCS 结构前向（照抄 rsl_rl，运行时不 import rsl_rl/orcs）
# ---------------------------------------------------------------------------
class AdapterLite(torch.nn.Module):
    """rsl_rl `Adapter` 的照抄实现（低秩 + 全秩退化两分支）。

    来源：lok-i/rsl_rl `rsl_rl/modules/mlp_with_adapter.py::Adapter`
    （BSD-3-Clause，ETH Zurich / NVIDIA 版权；本脚本只抄数学，不抄文件）。
    低秩：down (in→rank)、up (rank→out)，up 零初始化，scale = alpha / rank；
    全秩（rank<=0 或 rank>=min(in,out)）：单 Linear delta，零初始化，scale = alpha。
    """

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.low_rank = 0 < rank < min(in_features, out_features)
        if self.low_rank:
            self.down = torch.nn.Linear(in_features, rank, bias=False)
            self.up = torch.nn.Linear(rank, out_features, bias=False)
            self.scale = float(alpha) / float(rank)
        else:
            self.delta = torch.nn.Linear(in_features, out_features, bias=False)
            self.scale = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_rank:
            return self.scale * self.up(self.down(x))
        return self.scale * self.delta(x)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        if self.low_rank:
            return self.scale * (self.up.weight @ self.down.weight)
        return self.scale * self.delta.weight


class OrcsAdaptedDecoder(torch.nn.Module):
    """MLPWithAdapter 结构的照抄（base 冻结线性 + 逐层零初始化 LoRA）。

    前向顺序严格照 `MLPWithAdapter.forward`：adapter[i] 与 base[i] 吃**同一个**输入 h
    （i=0 时 h=base[0](obs)，adapter 吃 adapter_obs），相加后才在下一次迭代过激活。
    """

    def __init__(self, base_layers: list[dict], adapters: list["AdapterLite | None"],
                 activation: str = "SiLU") -> None:
        super().__init__()
        self.base = torch.nn.ModuleList()
        for ent in base_layers:
            w, b = ent["weight"], ent["bias"]
            lin = torch.nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=b is not None)
            with torch.no_grad():
                lin.weight.copy_(w)
                if b is not None:
                    lin.bias.copy_(b)
            lin.requires_grad_(False)
            self.base.append(lin)
        self.adapters = torch.nn.ModuleList([a for a in adapters])
        self.act = getattr(torch.nn, activation)()
        self.eval()

    def forward(self, obs: torch.Tensor, adapter_obs: torch.Tensor | None = None) -> torch.Tensor:
        if adapter_obs is None:
            adapter_obs = obs
        h = self.base[0](obs)
        if self.adapters[0] is not None:
            h = h + self.adapters[0](adapter_obs)
        for i in range(1, len(self.base)):
            h = self.act(h)
            out = self.base[i](h)
            if self.adapters[i] is not None:
                out = out + self.adapters[i](h)
            h = out
        return h


def build_adapter_modules(parsed: dict, alpha: float = 1.0) -> list["AdapterLite | None"]:
    """把 ckpt 里的 adapter 张量灌进 AdapterLite（scale 用 alpha；alpha 未知时给 1.0 并注明）。"""
    mods: list[AdapterLite | None] = []
    for ad in parsed["adapters"]:
        if ad is None:
            mods.append(None)
            continue
        rank = int(ad.get("rank", 0))
        if ad["kind"] == "low_rank":
            mod = AdapterLite(int(ad["in_features"]), int(ad["out_features"]), rank, alpha)
            with torch.no_grad():
                mod.down.weight.copy_(ad["down"])
                mod.up.weight.copy_(ad["up"])
        elif ad["kind"] == "full_rank":
            mod = AdapterLite(int(ad["in_features"]), int(ad["out_features"]), -1, alpha)
            with torch.no_grad():
                mod.delta.weight.copy_(ad["delta"])
        else:
            mods.append(None)
            continue
        mods.append(mod)
    return mods


# ---------------------------------------------------------------------------
# Layer 5: 冒烟输入（Dodge aug 流 / 本体感受窗）
# ---------------------------------------------------------------------------
def build_dodge_aug(root_height: float = DODGE_ROOT_HEIGHT_M,
                    park_z: float = DODGE_PARK_Z_M) -> tuple[np.ndarray, list[str]]:
    """Dodge 站立常量参考的 21 维 aug 向量（分段语义见模块 docstring）。

    返回 (aug(21,), 分段说明)。球 park 在 env 原点上方 park_z、速度清零；机器人在
    env 原点站立（root xy ≈ 0、z = root_height），命令恒零。
    """
    segments = [
        ("ball_pos_b", 3), ("ball_vel_b", 3),
        ("robot_root_pos_env", 3), ("robot_root_lin_vel_b", 3), ("robot_root_ang_vel_b", 3),
        ("robot_root_lin_vel_cmd", 3), ("robot_root_ang_vel_cmd", 3),
    ]
    aug = np.zeros(AUG_DIM_DODGE, dtype=np.float32)
    aug[0:3] = np.array([0.0, 0.0, park_z - root_height], dtype=np.float32)  # 球在头顶 park
    # 其余 18 维（球速度 / 根位置 xy / 根速度 / 命令）在站立参考下恒零
    names = []
    off = 0
    for name, width in segments:
        names.append(f"[{off}:{off + width}) {name}")
        off += width
    return aug, names


def build_smoke_obs(batch: int, seed: int) -> tuple[torch.Tensor, dict]:
    """造 (N,994) 本体感受窗 + 已量化 token（我方 SONIC/IsaacLab 序）。

    通道顺序 = sonic_decoder_torch.py:83-91：token64 | ang_vel(10x3) | joint_pos(10x29)
    | joint_vel(10x29) | last_act(10x29) | gravity(10x3)。帧序 oldest->newest。
    非 token 段围绕「站立帧」（sonic_action_term.py:35：gravity 每帧 [0,0,-1]、其余 0）
    加小噪声；token 取 FSQ 32 级网格 k/16（levels=32，sonic_base_model.fsq_quantize 口径）。
    """
    g = torch.Generator().manual_seed(int(seed))
    tokens = torch.randint(-16, 17, (batch, TOKEN_DIM), generator=g).float() / 16.0
    ang_vel = 0.05 * torch.randn(batch, SONIC_VEC3_WIDTH, generator=g)
    joint_pos = 0.02 * torch.randn(batch, SONIC_JOINT_WIDTH, generator=g)
    joint_vel = 0.10 * torch.randn(batch, SONIC_JOINT_WIDTH, generator=g)
    last_act = 0.05 * torch.randn(batch, SONIC_JOINT_WIDTH, generator=g)
    gravity = torch.zeros(batch, SONIC_VEC3_WIDTH)
    gravity.view(batch, HISTORY_LEN, GRAVITY_DIM)[:, :, 2] = -1.0
    gravity = gravity + 0.01 * torch.randn(batch, SONIC_VEC3_WIDTH, generator=g)
    obs = torch.cat([tokens, ang_vel, joint_pos, joint_vel, last_act, gravity], dim=1)
    meta = {
        "batch": int(batch),
        "seed": int(seed),
        "dim": int(obs.shape[1]),
        "layout": {
            "token": [0, TOKEN_DIM],
            "ang_vel": [TOKEN_DIM, TOKEN_DIM + SONIC_VEC3_WIDTH],
            "joint_pos": [TOKEN_DIM + SONIC_VEC3_WIDTH,
                          TOKEN_DIM + SONIC_VEC3_WIDTH + SONIC_JOINT_WIDTH],
            "joint_vel": [TOKEN_DIM + SONIC_VEC3_WIDTH + SONIC_JOINT_WIDTH,
                          TOKEN_DIM + SONIC_VEC3_WIDTH + 2 * SONIC_JOINT_WIDTH],
            "last_act": [TOKEN_DIM + SONIC_VEC3_WIDTH + 2 * SONIC_JOINT_WIDTH,
                         TOKEN_DIM + SONIC_VEC3_WIDTH + 3 * SONIC_JOINT_WIDTH],
            "gravity": [TOKEN_DIM + SONIC_VEC3_WIDTH + 3 * SONIC_JOINT_WIDTH, DECODER_OBS_DIM],
        },
        "token_grid": "FSQ levels=32 -> k/16",
        "standing_frame": "gravity [0,0,-1]/帧、其余通道 0 + 小噪声（sonic_action_term.py:35）",
    }
    return obs, meta


def normalize_aug(aug: torch.Tensor, normalizer: dict) -> tuple[torch.Tensor, dict]:
    """按 ckpt 的 adapter_normalizer 归一化（EmpiricalNormalization: (x-mean)/(std+eps)）。"""
    mean, std = normalizer.get("mean"), normalizer.get("std")
    if mean is None or std is None:
        return aug, {"applied": False, "reason": "ckpt 缺 adapter_normalizer._mean/_std，按恒等处理"}
    m = mean.reshape(-1).to(torch.float32)
    s = std.reshape(-1).to(torch.float32)
    if int(m.numel()) != int(aug.shape[-1]):
        return aug, {
            "applied": False,
            "reason": f"normalizer 维度 {int(m.numel())} != aug 维度 {int(aug.shape[-1])}",
        }
    return (aug - m) / (s + AUG_NORMALIZER_EPS), {
        "applied": True,
        "eps": AUG_NORMALIZER_EPS,
        "mean": [round(float(x), 6) for x in m.tolist()],
        "std": [round(float(x), 6) for x in s.tolist()],
        "count": int(normalizer["count"].reshape(-1)[0]) if normalizer.get("count") is not None else None,
    }


# ---------------------------------------------------------------------------
# Layer 6: 三门
# ---------------------------------------------------------------------------
def gate_g0a(our_layers: list[dict], parsed: dict, atol: float = TOL_WEIGHT,
             match_atol: float = MATCH_ATOL) -> dict:
    """权重对齐门：逐层 maxdiff + 首层列置换 + 末层行置换复原。"""
    theirs = parsed["base"]
    n = len(theirs)
    res: dict = {
        "gate": "g0a",
        "prefix": parsed["prefix"],
        "num_layers_ours": len(our_layers),
        "num_layers_theirs": n,
        "tol": atol,
        "layers": [],
        "input_perm": None,
        "output_perm": None,
        "notes": [],
        "verdict": "FAIL",
    }
    if len(our_layers) != n:
        res["notes"].append(f"层数不等（我方 {len(our_layers)} vs ckpt {n}）")
        return res

    ours = our_layers[-1]
    last = theirs[-1]
    if tuple(ours["weight"].shape) != tuple(last["weight"].shape):
        res["notes"].append(
            f"末层形状不等：我方 {tuple(ours['weight'].shape)} vs ckpt {tuple(last['weight'].shape)}"
        )
        return res

    # 末层行置换
    o_perm, o_info = recover_output_perm(ours["weight"], last["weight"], match_atol)
    if o_info["bijective"]:
        their_rows = last["weight"][torch.as_tensor(o_perm, dtype=torch.long), :]
        o_maxdiff = float((their_rows - ours["weight"]).abs().max())
        o_bitwise = bool(torch.equal(their_rows, ours["weight"]))
        b_theirs = last["bias"][torch.as_tensor(o_perm, dtype=torch.long)]
        ob_maxdiff = float((b_theirs - ours["bias"]).abs().max())
        ob_bitwise = bool(torch.equal(b_theirs, ours["bias"]))
    else:
        o_maxdiff, o_bitwise, ob_maxdiff, ob_bitwise = None, False, None, False
    res["output_perm"] = {
        "recover": o_info,
        "perm_our_to_their": [int(x) for x in o_perm],
        "maxdiff_weight_after_perm": o_maxdiff,
        "bitwise_weight": o_bitwise,
        "maxdiff_bias_after_perm": ob_maxdiff,
        "bitwise_bias": ob_bitwise,
        "is_bijection": bool(is_permutation(o_perm)) if (o_perm >= 0).all() else False,
        "joint_map_direction": joint_map_direction(o_perm if o_info["bijective"] else None),
    }

    # 首层列置换
    ours0 = our_layers[0]
    first = theirs[0]
    if tuple(ours0["weight"].shape) != tuple(first["weight"].shape):
        res["notes"].append(
            f"首层形状不等：我方 {tuple(ours0['weight'].shape)} vs ckpt {tuple(first['weight'].shape)}"
        )
        return res
    i_perm, i_info = recover_input_perm(ours0["weight"], first["weight"], match_atol)
    res["input_perm"] = {
        "recover": i_info,
        "perm_our_to_their": [int(x) for x in i_perm],
        "is_bijection": bool(i_info["bijective"]),
    }
    if i_info["bijective"]:
        cols = first["weight"][:, torch.as_tensor(i_perm, dtype=torch.long)]
        i_maxdiff = float((cols - ours0["weight"]).abs().max())
        i_bitwise = bool(torch.equal(cols, ours0["weight"]))
        res["input_perm"].update({
            "maxdiff_weight_after_perm": i_maxdiff,
            "bitwise_weight": i_bitwise,
            "maxdiff_bias": float((first["bias"] - ours0["bias"]).abs().max()),
            "bitwise_bias": bool(torch.equal(first["bias"], ours0["bias"])),
        })
        analysis = analyze_input_perm(i_perm)
        res["input_perm"]["analysis"] = analysis
        res["input_perm"]["joint_map_direction"] = joint_map_direction(analysis.get("joint_map"))
    else:
        res["input_perm"].update({
            "maxdiff_weight_after_perm": None,
            "bitwise_weight": False,
            "analysis": None,
        })
        res["notes"].append("首层列置换未能复原（匹配非双射）——对齐不成立")

    # 中间层直接比
    for k in range(1, n - 1):
        a, b = our_layers[k], theirs[k]
        if tuple(a["weight"].shape) != tuple(b["weight"].shape):
            res["layers"].append({"layer": k, "shape_mismatch": True,
                                  "ours": list(a["weight"].shape), "theirs": list(b["weight"].shape)})
            res["notes"].append(f"第 {k} 层形状不等，已跳过")
            continue
        w_diff = float((b["weight"] - a["weight"]).abs().max())
        b_diff = float((b["bias"] - a["bias"]).abs().max())
        res["layers"].append({
            "layer": k,
            "ckpt_key_index": theirs[k]["index"],
            "weight_shape": list(b["weight"].shape),
            "maxdiff_weight": w_diff,
            "maxdiff_bias": b_diff,
            "bitwise": bool(torch.equal(b["weight"], a["weight"])
                            and torch.equal(b["bias"], a["bias"])),
        })

    # 汇总判定
    checks: list[tuple[str, bool]] = []
    if res["input_perm"] and res["input_perm"].get("maxdiff_weight_after_perm") is not None:
        checks.append(("首层列置换后 maxdiff<=tol",
                       res["input_perm"]["maxdiff_weight_after_perm"] <= atol))
        checks.append(("首层 bias maxdiff<=tol", res["input_perm"]["maxdiff_bias"] <= atol))
        checks.append(("首层列置换为双射", bool(res["input_perm"]["is_bijection"])))
        ana = res["input_perm"].get("analysis") or {}
        checks.append(("首层置换假设 A（token 列恒等 + proprio 块结构）", bool(ana.get("structure_ok"))))
        checks.append(("首层关节图方向可解释（m2i 或其逆）",
                       res["input_perm"].get("joint_map_direction", "neither").startswith("matches_")))
    else:
        checks.append(("首层列置换可复原", False))
    if res["output_perm"] and res["output_perm"].get("maxdiff_weight_after_perm") is not None:
        checks.append(("末层行置换后 maxdiff<=tol",
                       res["output_perm"]["maxdiff_weight_after_perm"] <= atol))
        checks.append(("末层 bias 行置换后 maxdiff<=tol",
                       res["output_perm"]["maxdiff_bias_after_perm"] <= atol))
        checks.append(("末层行置换为双射", bool(res["output_perm"]["is_bijection"])))
    else:
        checks.append(("末层行置换可复原", False))
    for ent in res["layers"]:
        if ent.get("shape_mismatch"):
            checks.append((f"第 {ent['layer']} 层形状一致", False))
            continue
        checks.append((f"第 {ent['layer']} 层 maxdiff<=tol",
                       ent["maxdiff_weight"] <= atol and ent["maxdiff_bias"] <= atol))
    # 入/出关节图一致性（两条独立复原路径应给出同一关节映射）
    jm_in = (res["input_perm"].get("analysis") or {}).get("joint_map") if res["input_perm"] else None
    jm_out = res["output_perm"]["perm_our_to_their"] if res["output_perm"] else None
    if jm_in is not None and jm_out is not None and res["output_perm"]["is_bijection"]:
        checks.append(("入/出关节图一致", [int(x) for x in jm_in] == [int(x) for x in jm_out]))
    res["checks"] = [{"name": name, "ok": bool(ok)} for name, ok in checks]
    res["checks_failed"] = [c["name"] for c in res["checks"] if not c["ok"]]
    res["verdict"] = "PASS" if all(ok for _, ok in checks) else "FAIL"
    return res


def gate_g0b(raw, sd: dict, parsed: dict, container: str) -> dict:
    """metadata 门（信息性）：顶层非张量键 + adapter rank/alpha 尽力提取。"""
    res: dict = {
        "gate": "g0b",
        "state_dict_container": container,
        "top_level": [],
        "runtime": {},
        "decoder": {},
        "rank_alpha_hits": [],
        "alpha": None,
        "alpha_source": "not_in_state_dict",
        "rank_per_layer": [None if a is None else a.get("rank") for a in parsed["adapters"]],
        "notes": [],
        "verdict": "PASS",
    }
    if isinstance(raw, dict):
        for key, val in raw.items():
            ent = {"key": str(key)}
            if isinstance(val, torch.Tensor):
                ent.update({"kind": "tensor", "shape": list(val.shape), "dtype": str(val.dtype)})
            elif isinstance(val, dict):
                ent.update({"kind": "dict", "n_keys": len(val),
                            "keys": [str(k) for k in list(val)[:40]]})
            elif isinstance(val, (int, float, str, bool)) or val is None:
                ent.update({"kind": "scalar", "value": val if not isinstance(val, float)
                            or math.isfinite(val) else repr(val)})
            else:
                ent.update({"kind": type(val).__name__, "preview": repr(val)[:200]})
            res["top_level"].append(ent)
    for key in ("iter", "infos", "current_learning_iteration"):
        if isinstance(raw, dict) and key in raw:
            res["runtime"][key] = jsonable(raw[key])
    kinds = []
    for i, ad in enumerate(parsed["adapters"]):
        kinds.append({
            "layer": i,
            "kind": None if ad is None else ad.get("kind"),
            "rank": None if ad is None else ad.get("rank"),
            "in_features": None if ad is None else ad.get("in_features"),
            "out_features": None if ad is None else ad.get("out_features"),
        })
    res["decoder"] = {
        "prefix": parsed["prefix"],
        "base_key_pattern": parsed.get("base_key_pattern"),
        "adapter_key_pattern": parsed.get("adapter_key_pattern"),
        "n_adapter_layers": parsed.get("n_adapter_layers"),
        "n_base_layers": len(parsed["base"]),
        "base_layer_shapes": [[int(x) for x in e["weight"].shape] for e in parsed["base"]],
        "base_layer_key_index": [int(e["index"]) for e in parsed["base"]],
        "adapter_layers": kinds,
        "n_encoder_keys": len(parsed["encoder_keys"]),
        "normalizer_keys": parsed["normalizer"]["keys"],
        "normalizer_dim": (int(parsed["normalizer"]["mean"].reshape(-1).numel())
                           if parsed["normalizer"]["mean"] is not None else None),
        "normalizer_count": (int(parsed["normalizer"]["count"].reshape(-1)[0])
                             if parsed["normalizer"]["count"] is not None else None),
    }

    # 递归搜非张量 metadata 里的 rank/alpha
    def walk(obj, path: str, depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(obj, dict):
            keys = {str(k) for k in obj}
            if {"rank", "alpha"} & keys:
                hit = {"path": path or "<root>"}
                for k in ("rank", "alpha", "scale", "adapter_obs_group", "class_name"):
                    if k in obj:
                        hit[k] = jsonable(obj[k], depth=2)
                res["rank_alpha_hits"].append(hit)
            for k, v in list(obj.items())[:80]:
                if isinstance(v, (dict, list, tuple)):
                    walk(v, f"{path}.{k}" if path else str(k), depth + 1)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(list(obj)[:20]):
                if isinstance(v, (dict, list, tuple)):
                    walk(v, f"{path}[{i}]", depth + 1)

    if isinstance(raw, dict):
        walk(raw, "")
    if res["rank_alpha_hits"]:
        hit = res["rank_alpha_hits"][0]
        res["alpha"] = hit.get("alpha")
        res["alpha_source"] = f"metadata:{hit['path']}"
        res["notes"].append(
            f"非张量 metadata 命中 rank/alpha：{hit['path']} -> "
            f"rank={hit.get('rank')} alpha={hit.get('alpha')}"
        )
    else:
        res["alpha_source"] = "not_in_state_dict"
        res["notes"].append(
            "ckpt 非张量 metadata 里没有 rank/alpha（rsl_rl 的 scale 是 Python 属性、不进 "
            "state_dict）：rank 已从 down.weight 形状推；alpha 未定，按 ORCS 源码默认 "
            "rank=16/alpha=1.0（rl.py:102-103）只作参考，实际 scale=alpha/rank 无法从 "
            "ckpt 判定——除非 rank>=min(in,out) 走全秩分支（那时 scale=alpha）"
        )
    if any(a is None for a in parsed["adapters"]):
        res["notes"].append("有层没有 adapter（decoder 部分层未挂）——与 `adapt_decoder=True` 默认不符")
    ranks = [a.get("rank") for a in parsed["adapters"] if a]
    if ranks and len(set(ranks)) > 1:
        res["notes"].append(f"各层 rank 不一致：{ranks}")
    return res


def gate_g0c(parsed: dict, our_net: torch.nn.Module, g0a: dict, batch: int, seed: int,
             alpha: float = 1.0, alpha_source: str = "default") -> dict:
    """前向冒烟门：Dodge aug + 合成本体窗过 ORCS 结构；另做我方 decoder 交叉验证。"""
    res: dict = {
        "gate": "g0c",
        "task": "dodge",
        "alpha_used": float(alpha),
        "alpha_source": alpha_source,
        "aug": {},
        "obs": {},
        "orcs_forward": {},
        "adapter_delta": {},
        "cross_check": {},
        "errors": [],
        "notes": [],
        "verdict": "FAIL",
    }
    aug_np, aug_names = build_dodge_aug()
    res["aug"] = {
        "dim": int(aug_np.shape[0]),
        "expected_dim": AUG_DIM_DODGE,
        "order": aug_names,
        "values": [float(x) for x in aug_np],
        "source": "站立常量参考（park_z=8.0, 根高 0.793, 命令恒零）",
    }
    try:
        adapters = build_adapter_modules(parsed, float(alpha))
        net = OrcsAdaptedDecoder(parsed["base"], adapters)
    except Exception as exc:  # noqa: BLE001 - 冒烟门自己兜异常，别让门崩溃
        res["errors"].append(f"建模块失败：{type(exc).__name__}: {exc}")
        return res
    res["adapter_scales"] = [None if a is None else a.scale for a in adapters]
    n_adapters = int(parsed.get("n_adapter_layers",
                                sum(1 for a in parsed["adapters"] if a is not None)))
    res["n_adapter_layers"] = n_adapters
    if n_adapters == 0:
        res["errors"].append(
            "ckpt 里没有任何 adapter 键（0 层）——无法验证 ORCS 的 LoRA 结构前向："
            "该 ckpt 可能是未适配的纯基座，或 adapter 键名不匹配本脚本的模式"
            "（见 g0b.decoder.adapter_key_pattern）。"
        )
        res["verdict_reason"] = "缺 adapter 键"
        return res
    if n_adapters != len(parsed["base"]):
        res["notes"].append(
            f"只有 {n_adapters}/{len(parsed['base'])} 层有 adapter（ORCS 默认挂全部 7 层）"
        )

    aug_dim_ckpt = None
    if parsed["adapters"] and parsed["adapters"][0] is not None:
        aug_dim_ckpt = parsed["adapters"][0].get("in_features")
    res["aug"]["ckpt_adapter0_in_features"] = aug_dim_ckpt
    if aug_dim_ckpt is not None and int(aug_dim_ckpt) != AUG_DIM_DODGE:
        res["errors"].append(
            f"ckpt 第 0 层 adapter 输入宽 {aug_dim_ckpt} != dodge aug 维 {AUG_DIM_DODGE}"
            "（该 ckpt 可能不是 Dodge 任务的 adapter，维度不符必然 shape error）"
        )
        res["verdict"] = "FAIL"
        res["verdict_reason"] = "aug 维度与 ckpt adapter 不匹配"
        return res

    try:
        obs, obs_meta = build_smoke_obs(batch, seed)
        res["obs"] = obs_meta
        aug_raw = torch.from_numpy(aug_np).view(1, -1).repeat(batch, 1)
        aug_norm, norm_meta = normalize_aug(aug_raw, parsed["normalizer"])
        res["aug"]["normalizer"] = norm_meta
        with torch.no_grad():
            out = net(obs, aug_norm)
        finite = bool(torch.isfinite(out).all())
        res["orcs_forward"] = {
            "shape": [int(x) for x in out.shape],
            "expected_shape": [int(batch), N_JOINTS],
            "finite": finite,
            "abs_max": float(out.abs().max()),
            "abs_mean": float(out.abs().mean()),
            "note": "归一化关节目标（29 维，对方 MuJoCo 序），非物理角度",
        }
        if list(out.shape) != [int(batch), N_JOINTS]:
            res["errors"].append(f"输出形状 {list(out.shape)} != {(batch, N_JOINTS)}")
        if not finite:
            res["errors"].append("输出含非有限值")
    except Exception as exc:  # noqa: BLE001
        res["errors"].append(f"ORCS 结构前向失败：{type(exc).__name__}: {exc}")
        return res

    # adapter 幅度（信息性）：delta weight Frobenius 范数
    deltas = []
    for i, ad in enumerate(adapters):
        if ad is None:
            deltas.append(None)
            continue
        with torch.no_grad():
            dw = ad.delta_weight()
        deltas.append({
            "layer": i,
            "kind": "low_rank" if ad.low_rank else "full_rank",
            "scale": ad.scale,
            "delta_fro_norm": float(dw.norm()),
            "delta_abs_max": float(dw.abs().max()),
        })
    res["adapter_delta"] = {"per_layer": deltas,
                            "note": "delta = scale·(up@down) 或 scale·delta；零初始化适配器应全 0"}

    # 交叉验证：我方 decoder(按置换喂输入/反置换读输出) == 对方 base(adapter 关)
    in_perm = None
    out_perm = None
    if g0a.get("input_perm") and g0a["input_perm"].get("is_bijection"):
        in_perm = np.asarray(g0a["input_perm"]["perm_our_to_their"], dtype=np.int64)
    if g0a.get("output_perm") and g0a["output_perm"].get("is_bijection"):
        out_perm = np.asarray(g0a["output_perm"]["perm_our_to_their"], dtype=np.int64)
    perm_source = "g0a_recovered"
    if in_perm is None:
        in_perm = input_perm_from_joint_map(M2I)
        perm_source = "constant_G1_MUJOCO_TO_ISAACLAB_DOF(g0a 未复原，退常量)"
    if out_perm is None:
        out_perm = output_perm_from_joint_map(M2I)
    try:
        base_only = OrcsAdaptedDecoder(parsed["base"], [None] * len(parsed["base"]))
        obs_their = obs[:, torch.as_tensor(np.argsort(in_perm), dtype=torch.long)]
        with torch.no_grad():
            their_out = base_only(obs_their)
            our_out = our_net(obs)
        our_via_their = their_out[:, torch.as_tensor(out_perm, dtype=torch.long)]
        maxdiff = float((our_via_their - our_out).abs().max())
        res["cross_check"] = {
            "enabled": True,
            "perm_source": perm_source,
            "maxdiff": maxdiff,
            "tol": TOL_FORWARD,
            "abs_max_ours": float(our_out.abs().max()),
            "note": "我方 decoder(输入按列置换、输出按行置换) vs ckpt base 线性 7 层(adapter 关)",
            "ok": bool(maxdiff <= TOL_FORWARD),
        }
        if maxdiff > TOL_FORWARD:
            res["errors"].append(f"交叉验证 maxdiff={maxdiff:.3e} > tol={TOL_FORWARD:g}")
    except Exception as exc:  # noqa: BLE001
        res["cross_check"] = {"enabled": True, "ok": False,
                              "error": f"{type(exc).__name__}: {exc}"}
        res["errors"].append(f"交叉验证失败：{type(exc).__name__}: {exc}")

    ok = not res["errors"]
    res["verdict"] = "PASS" if ok else "FAIL"
    if ok:
        res["notes"].append(
            "前向冒烟通过：Dodge 21 维 aug + (N,994) 本体窗过 ORCS 结构（base+adapter 逐层相加）"
            "输出 (N,29) 有限；且我方 decoder 与对方 base 端到端数值一致"
        )
    return res


# ---------------------------------------------------------------------------
# Layer 7: selftest（合成 ckpt + 已知置换；正向应 PASS / 故意错配应 FAIL）
# ---------------------------------------------------------------------------
def build_synthetic_case(seed: int = 0,
                         hidden: tuple[int, ...] = (32, 32, 16, 16, 8, 8),
                         rank: int = 4,
                         jmap: np.ndarray | None = None,
                         corrupt_middle: float = 0.0,
                         layer0_random_col_perm: bool = False,
                         layer0_corrupt_col: bool = False,
                         aug_dim: int = AUG_DIM_DODGE,
                         include_alpha_meta: bool = True,
                         zero_adapters: bool = False,
                         bare_keys: bool = False,
                         full_rank_adapters: bool = False):
    """造一个形状同构的合成 ckpt dict + 我方 decoder（返回 raw, our_net, our_layers, 说明）。"""
    jmap = M2I if jmap is None else np.asarray(jmap, dtype=np.int64)
    g = torch.Generator().manual_seed(int(seed))

    def rnd(*shape, scale=0.15):
        return scale * torch.randn(*shape, generator=g)

    dims = [DECODER_OBS_DIM, *hidden, N_JOINTS]
    our_layers = []
    for i in range(len(dims) - 1):
        our_layers.append({
            "weight": rnd(dims[i + 1], dims[i]),
            "bias": rnd(dims[i + 1], scale=0.05),
        })
    our_net = build_mlp_from_layers(our_layers)

    in_perm = input_perm_from_joint_map(jmap)
    out_perm = output_perm_from_joint_map(jmap)
    if layer0_random_col_perm:
        rng = np.random.default_rng(seed + 1234)
        in_perm = rng.permutation(DECODER_OBS_DIM)
    inv_in = np.argsort(in_perm)
    inv_out = np.argsort(out_perm)

    theirs = []
    for i, ent in enumerate(our_layers):
        w = ent["weight"].clone()
        b = ent["bias"].clone()
        if i == 0:
            w = w[:, torch.as_tensor(inv_in, dtype=torch.long)]      # their 列 = 我方列按置换
            if layer0_corrupt_col:
                w[:, 5] = w[:, 5] + 3.0                              # 制造不可复原
        elif i == len(our_layers) - 1:
            w = w[torch.as_tensor(inv_out, dtype=torch.long), :]     # their 行 = 我方行按置换
            b = b[torch.as_tensor(inv_out, dtype=torch.long)]
        elif corrupt_middle and i == len(our_layers) // 2:
            w = w + corrupt_middle
        theirs.append({"weight": w, "bias": b})

    # bare_keys=True 模拟「actor 直接是 MLPWithAdapter」的键名（无 `decoder.` 包装），
    # 用于验证 BASE/ADAPTER_KEY_PATTERNS 的兜底分支。
    pre = "" if bare_keys else "decoder."
    sd: dict[str, torch.Tensor] = {}
    for k, ent in enumerate(theirs):
        sd[f"{pre}base.{2 * k}.weight"] = ent["weight"]
        sd[f"{pre}base.{2 * k}.bias"] = ent["bias"]
    for k, ent in enumerate(our_layers):
        in_f = int(ent["weight"].shape[1])
        out_f = int(ent["weight"].shape[0])
        if k == 0:
            in_f = aug_dim
        if full_rank_adapters:
            # 全秩退化分支（rsl_rl: rank<=0 或 rank>=min(in,out)）：单 Linear delta、scale=alpha
            sd[f"{pre}adapters.{k}.delta.weight"] = torch.zeros(out_f, in_f) if zero_adapters                 else rnd(out_f, in_f, scale=0.02)
            continue
        down = rnd(rank, in_f, scale=1.0 / math.sqrt(rank))
        up = torch.zeros(out_f, rank)
        if not zero_adapters:
            up = rnd(out_f, rank, scale=0.02)
        sd[f"{pre}adapters.{k}.down.weight"] = down
        sd[f"{pre}adapters.{k}.up.weight"] = up
    sd["adapter_normalizer._mean"] = torch.zeros(1, aug_dim)
    sd["adapter_normalizer._std"] = 0.5 * torch.ones(1, aug_dim)
    sd["adapter_normalizer.count"] = torch.tensor(4096, dtype=torch.long)
    sd["encoder.0.weight"] = rnd(16, 40)
    sd["std"] = 0.5 * torch.ones(N_JOINTS)

    raw: dict = {"actor_state_dict": sd, "iter": 7500,
                 "infos": {"run_name": "selftest"},
                 "model_kwargs": {"class_name": "rsl_rl.models.SonicWithAdapterModel"}}
    if include_alpha_meta:
        raw["infos"]["actor_cfg"] = {"rank": rank, "alpha": 1.0,
                                     "adapter_obs_group": "augmentation"}
    return raw, our_net, our_layers, {
        "jmap": [int(x) for x in jmap],
        "in_perm_head": [int(x) for x in in_perm[:8]],
        "hidden": list(hidden),
        "rank": rank,
        "aug_dim": aug_dim,
    }


def run_selftest(seed: int = 0, verbose: bool = True) -> dict:
    """合成数据双向自检：应 PASS 的两条路径 + 应 FAIL 的两条错配路径。"""
    cases: list[dict] = []
    failures: list[str] = []

    def check(cond: bool, label: str) -> None:
        if not cond:
            failures.append(label)
        if verbose:
            print(f"  [{'ok ' if cond else 'FAIL'}] {label}")

    # --- 用例 1：形状同构 + 已知置换 -> g0a/g0b/g0c 全应 PASS ---
    raw, our_net, our_layers, meta = build_synthetic_case(seed=seed)
    container, sd, notes = find_state_dict_container(raw)
    parsed = parse_decoder_state_dict(sd)
    g0a = gate_g0a(our_layers, parsed)
    g0b = gate_g0b(raw, sd, parsed, container)
    g0c = gate_g0c(parsed, our_net, g0a, batch=3, seed=seed, alpha=1.0)

    cases.append({"case": "s1_synthetic_should_pass", "expect": {"g0a": "PASS", "g0c": "PASS"},
                  "got": {"g0a": g0a["verdict"], "g0b": g0b["verdict"], "g0c": g0c["verdict"]},
                  "meta": meta})
    if verbose:
        print(f"用例 1（合成、已知置换、应 PASS）：g0a={g0a['verdict']} g0b={g0b['verdict']} "
              f"g0c={g0c['verdict']}")
    check(g0a["verdict"] == "PASS", "用例1 g0a 应为 PASS")
    check(g0b["verdict"] == "PASS", "用例1 g0b 应为 PASS")
    check(g0c["verdict"] == "PASS", "用例1 g0c 应为 PASS")
    check((g0a.get("input_perm") or {}).get("analysis", {}).get("hypothesis") == "A",
          "用例1 首层置换假设应判为 A（token 列恒等）")
    check((g0a.get("input_perm") or {}).get("joint_map_direction") ==
          "matches_g1_mujoco_to_isaaclab_dof", "用例1 关节图应命中 G1_MUJOCO_TO_ISAACLAB_DOF")
    check((g0a.get("output_perm") or {}).get("joint_map_direction") ==
          "matches_g1_mujoco_to_isaaclab_dof", "用例1 末层行置换应命中同一常量")
    check(bool((g0c.get("cross_check") or {}).get("ok")), "用例1 交叉验证应 PASS")
    check(g0b.get("alpha_source", "").startswith("metadata:"), "用例1 alpha 应从 metadata 提取")
    check(g0b.get("rank_per_layer", [None])[0] == meta["rank"], "用例1 rank 应从 down 形状推出")

    # --- 用例 2：中间层被扰动 -> g0a 应 FAIL ---
    raw2, _net2, layers2, _meta2 = build_synthetic_case(seed=seed + 1, corrupt_middle=1e-2)
    _c, sd2, _n = find_state_dict_container(raw2)
    parsed2 = parse_decoder_state_dict(sd2)
    g0a2 = gate_g0a(layers2, parsed2)
    cases.append({"case": "s2_middle_layer_corrupt_should_fail", "expect": {"g0a": "FAIL"},
                  "got": {"g0a": g0a2["verdict"]}})
    if verbose:
        print(f"用例 2（中间层 +1e-2，应 FAIL）：g0a={g0a2['verdict']}")
    check(g0a2["verdict"] == "FAIL", "用例2 g0a 应为 FAIL（中间层错配）")
    big = [e for e in g0a2["layers"] if e.get("maxdiff_weight", 0) > TOL_WEIGHT]
    check(len(big) >= 1, "用例2 应至少报出一层 maxdiff 超阈")

    # --- 用例 3：置换语义错（token 列也乱序 = 假设 B）-> g0a 应 FAIL ---
    raw3, _net3, layers3, _m3 = build_synthetic_case(seed=seed + 2, layer0_random_col_perm=True)
    _c, sd3, _n = find_state_dict_container(raw3)
    parsed3 = parse_decoder_state_dict(sd3)
    g0a3 = gate_g0a(layers3, parsed3)
    hyp3 = ((g0a3.get("input_perm") or {}).get("analysis") or {}).get("hypothesis")
    cases.append({"case": "s3_random_col_perm_should_fail", "expect": {"g0a": "FAIL"},
                  "got": {"g0a": g0a3["verdict"], "hypothesis": hyp3}})
    if verbose:
        print(f"用例 3（首层随机列置换 = 假设 B，应 FAIL）：g0a={g0a3['verdict']} 假设={hyp3}")
    check(g0a3["verdict"] == "FAIL", "用例3 g0a 应为 FAIL（置换前提不成立）")
    check(hyp3 == "B", "用例3 应判定为假设 B（token 列非恒等）")

    # --- 用例 4：首层某列不可复原 -> g0a 应 FAIL ---
    raw4, _net4, layers4, _m4 = build_synthetic_case(seed=seed + 3, layer0_corrupt_col=True)
    _c, sd4, _n = find_state_dict_container(raw4)
    parsed4 = parse_decoder_state_dict(sd4)
    g0a4 = gate_g0a(layers4, parsed4)
    cases.append({"case": "s4_unrecoverable_column_should_fail", "expect": {"g0a": "FAIL"},
                  "got": {"g0a": g0a4["verdict"]}})
    if verbose:
        print(f"用例 4（首层一列被替换、不可复原，应 FAIL）：g0a={g0a4['verdict']}")
    check(g0a4["verdict"] == "FAIL", "用例4 g0a 应为 FAIL（置换不可复原）")

    # --- 用例 5：aug 维度与 ckpt adapter 不符 -> g0c 应 FAIL（而非抛异常）---
    raw5, net5, layers5, _m5 = build_synthetic_case(seed=seed + 4, aug_dim=20)
    _c, sd5, _n = find_state_dict_container(raw5)
    parsed5 = parse_decoder_state_dict(sd5)
    g0a5 = gate_g0a(layers5, parsed5)
    g0c5 = gate_g0c(parsed5, net5, g0a5, batch=2, seed=seed, alpha=1.0)
    cases.append({"case": "s5_aug_dim_mismatch_should_fail", "expect": {"g0c": "FAIL"},
                  "got": {"g0c": g0c5["verdict"], "errors": g0c5["errors"]}})
    if verbose:
        print(f"用例 5（ckpt adapter 输入宽 20 != dodge 21，应 FAIL）：g0c={g0c5['verdict']} "
              f"errors={len(g0c5['errors'])}")
    check(g0c5["verdict"] == "FAIL", "用例5 g0c 应为 FAIL（aug 维度不符）")
    check(len(g0c5["errors"]) >= 1, "用例5 应给出可读错误信息（shape 不符）")

    # --- 用例 6：adapter 全宽 21 但注入错的 aug 张量形状 -> 门内捕获掉异常 ---
    raw6, net6, layers6, _m6 = build_synthetic_case(seed=seed + 5, aug_dim=AUG_DIM_DODGE)
    _c, sd6, _n = find_state_dict_container(raw6)
    parsed6 = parse_decoder_state_dict(sd6)
    # 人为把第 0 层 adapter 的输入宽度改小（模拟接线错），验证门失败而不是崩
    parsed6["adapters"][0]["in_features"] = 20
    g0a6 = gate_g0a(layers6, parsed6)
    g0c6 = gate_g0c(parsed6, net6, g0a6, batch=2, seed=seed, alpha=1.0)
    cases.append({"case": "s6_wrong_wiring_should_fail", "expect": {"g0c": "FAIL"},
                  "got": {"g0c": g0c6["verdict"], "errors": g0c6["errors"]}})
    if verbose:
        print(f"用例 6（adapter 输入宽被改成 20，应 FAIL）：g0c={g0c6['verdict']}")
    check(g0c6["verdict"] == "FAIL", "用例6 g0c 应为 FAIL（接线错）")
    check(len(g0c6["errors"]) >= 1, "用例6 应给出可读错误信息")

    # --- 用例 7：键名无 `decoder.` 包装（兜底模式）-> 仍应 PASS ---
    raw7, net7, layers7, _m7 = build_synthetic_case(seed=seed + 6, bare_keys=True)
    _c7, sd7, _n7 = find_state_dict_container(raw7)
    parsed7 = parse_decoder_state_dict(sd7)
    g0a7 = gate_g0a(layers7, parsed7)
    g0c7 = gate_g0c(parsed7, net7, g0a7, batch=2, seed=seed, alpha=1.0)
    cases.append({"case": "s7_bare_key_names_should_pass",
                  "expect": {"g0a": "PASS", "g0c": "PASS"},
                  "got": {"g0a": g0a7["verdict"], "g0c": g0c7["verdict"],
                          "base_key_pattern": parsed7.get("base_key_pattern")}})
    if verbose:
        print(f"用例 7（键名 base.N/adapters.N，无 decoder. 前缀，应 PASS）："
              f"g0a={g0a7['verdict']} g0c={g0c7['verdict']}")
    check(g0a7["verdict"] == "PASS" and g0c7["verdict"] == "PASS",
          "用例7 兜底键模式应仍能解析并 PASS")
    check(parsed7.get("n_adapter_layers") == 7, "用例7 应解析到 7 层 adapter 键")

    # --- 用例 8：全秩退化 adapter（delta.weight，rank>=min(in,out)）-> PASS 且 scale=alpha ---
    raw8, net8, layers8, _m8 = build_synthetic_case(seed=seed + 7, full_rank_adapters=True)
    _c8, sd8, _n8 = find_state_dict_container(raw8)
    parsed8 = parse_decoder_state_dict(sd8)
    g0a8 = gate_g0a(layers8, parsed8)
    g0c8 = gate_g0c(parsed8, net8, g0a8, batch=2, seed=seed, alpha=3.0)
    kinds8 = [a.get("kind") for a in parsed8["adapters"] if a]
    cases.append({"case": "s8_full_rank_adapter_should_pass",
                  "expect": {"g0a": "PASS", "g0c": "PASS", "kinds": "full_rank x7", "scale": 3.0},
                  "got": {"g0a": g0a8["verdict"], "g0c": g0c8["verdict"],
                          "kinds": sorted(set(kinds8)), "scale": g0c8.get("adapter_scales", [None])[0]}})
    if verbose:
        print(f"用例 8（全秩退化 delta 分支，应 PASS 且 scale=alpha）：g0a={g0a8['verdict']} "
              f"g0c={g0c8['verdict']} kinds={sorted(set(kinds8))}")
    check(g0a8["verdict"] == "PASS" and g0c8["verdict"] == "PASS", "用例8 全秩分支应 PASS")
    check(set(kinds8) == {"full_rank"}, "用例8 应识别为 full_rank（delta.weight）")
    check(abs(float(g0c8["adapter_scales"][0]) - 3.0) < 1e-12,
          "用例8 全秩 scale 应 = alpha（不除 rank）")

    verdict = "PASS" if not failures else "FAIL"
    if verbose:
        print(f"\nselftest: {verdict}（{len(cases)} 用例，{len(failures)} 项断言失败）")
        for f in failures:
            print(f"  - 断言失败：{f}")
    return {"verdict": verdict, "cases": cases, "failed_checks": failures,
            "detail": {c["case"]: c for c in cases}}


# ---------------------------------------------------------------------------
# Layer 8: 真模式主流程
# ---------------------------------------------------------------------------
def run_real(args) -> dict:
    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"[FATAL] --ckpt 不存在：{ckpt_path}")
    if args.task != "dodge":
        raise SystemExit(f"[FATAL] v1 只支持 --task dodge（收到 {args.task}）")
    onnx_path, onnx_src = resolve_onnx_path(args.onnx)

    print(f"[{EXPERIMENT}-{GATE_NAME}] ckpt = {ckpt_path}")
    raw, load_method = load_ckpt(ckpt_path)
    container, sd, find_notes = find_state_dict_container(raw)
    parsed = parse_decoder_state_dict(sd)
    our_net = load_our_decoder_net(onnx_path)
    our_layers, our_acts = extract_our_layers(our_net)

    identity = {
        "experiment": EXPERIMENT,
        "gate": GATE_NAME,
        "script": SCRIPT_TAG,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt_path": ckpt_path,
        "ckpt": file_digests(ckpt_path),
        "ckpt_load_method": load_method,
        "onnx_path": onnx_path,
        "onnx_path_source": onnx_src,
        "onnx": file_digests(onnx_path),
        "state_dict_container": container,
        "n_state_keys": parsed["n_keys"],
        "tensor_shapes": tensor_shape_map(sd),
        "our_decoder_layers": [list(l["weight"].shape) for l in our_layers],
        "our_decoder_acts": our_acts,
        "joint_perm_digest": joint_perm_digest(),
        "host": {"python": sys.version.split()[0], "torch": torch.__version__,
                 "numpy": np.__version__, "platform": sys.platform},
        "license_note": LICENSE_NOTE,
    }

    g0a = gate_g0a(our_layers, parsed)
    g0b = gate_g0b(raw, sd, parsed, container)
    alpha_meta = g0b.get("alpha")
    if isinstance(alpha_meta, (int, float)) and math.isfinite(float(alpha_meta)):
        alpha_val, alpha_src = float(alpha_meta), g0b["alpha_source"]
    elif args.alpha is not None:
        alpha_val, alpha_src = float(args.alpha), "cli(--alpha)"
    else:
        alpha_val, alpha_src = 1.0, ("default(rsl_rl Adapter alpha=1.0；rank 取 down 形状 -> "
                                     "低秩 scale=alpha/rank)")
    g0c = gate_g0c(parsed, our_net, g0a, batch=args.batch, seed=args.seed,
                   alpha=alpha_val, alpha_source=alpha_src)
    notes = list(find_notes) + list(KNOWN_LIMITATIONS)

    overall = all(g["verdict"] == "PASS" for g in (g0a, g0b, g0c))
    report = {
        "experiment": EXPERIMENT,
        "gate": GATE_NAME,
        "mode": "real",
        "identity": identity,
        "g0a": g0a,
        "g0b": g0b,
        "g0c": g0c,
        "verdict": {
            "g0a": g0a["verdict"], "g0b": g0b["verdict"], "g0c": g0c["verdict"],
            "overall": "PASS" if overall else "FAIL",
        },
        "notes": notes,
        "license_note": LICENSE_NOTE,
    }
    print_summary(report)
    return report


def print_summary(report: dict) -> None:
    """stdout 人话判读（每门一行 + 总判，便于 grep）。"""
    g0a, g0b, g0c = report["g0a"], report["g0b"], report["g0c"]
    print("\n=== D065 G0 对齐冒烟判读 ===")
    lay = ", ".join(
        f"L{e['layer']}:{e.get('maxdiff_weight', float('nan')):.3g}"
        for e in g0a["layers"] if not e.get("shape_mismatch")
    )
    ip = g0a.get("input_perm") or {}
    ana = (ip.get("analysis") or {})
    print(f"[G0A {g0a['verdict']}] 权重对齐：中间层 maxdiff({lay})；"
          f"首层列置换={'复原' if ip.get('is_bijection') else '失败'}"
          f"(maxdiff={ip.get('maxdiff_weight_after_perm')}, 假设={ana.get('hypothesis')}, "
          f"方向={ip.get('joint_map_direction')})；"
          f"末层行置换={g0a['output_perm']['joint_map_direction']}"
          f"(maxdiff={(g0a.get('output_perm') or {}).get('maxdiff_weight_after_perm')})")
    for name, ok in ((c["name"], c["ok"]) for c in g0a.get("checks", [])):
        if not ok:
            print(f"       └ 未过项：{name}")
    top_keys = [e["key"] for e in g0b["top_level"]]
    top_show = ", ".join(top_keys[:8]) + (" …" if len(top_keys) > 8 else "")
    print(f"[G0B {g0b['verdict']}] metadata：容器='{g0b['state_dict_container']}'，"
          f"rank/层={g0b['rank_per_layer']}，alpha={g0b['alpha']}（{g0b['alpha_source']}），"
          f"顶层非张量键={len(top_keys)} 个：{top_show}")
    fw = g0c.get("orcs_forward") or {}
    cc = g0c.get("cross_check") or {}
    print(f"[G0C {g0c['verdict']}] 前向冒烟：aug={g0c['aug'].get('dim')} 维，"
          f"输出 shape={fw.get('shape')} finite={fw.get('finite')}；"
          f"交叉验证 maxdiff={cc.get('maxdiff')}（tol={cc.get('tol')}）")
    for err in g0c.get("errors", []):
        print(f"       └ 错误：{err}")
    ov = report["verdict"]["overall"]
    print(f"[{EXPERIMENT}-{GATE_NAME}] OVERALL {ov}")
    print(f"D065 G0 SMOKE {ov}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="D065 G0 对齐冒烟门：ORCS/ViBe ckpt decoder(+LoRA) vs 我方 SONIC decoder"
                    "（纯 torch，CPU）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt", type=str, default=None,
                    help="ORCS 发布 ckpt 路径（real 模式必填；--selftest 时忽略）")
    ap.add_argument("--task", type=str, default="dodge", choices=["dodge"],
                    help="任务（v1 只支持 dodge）")
    ap.add_argument("--selftest", action="store_true",
                    help="合成 ckpt 自检门逻辑（不读任何外部文件，含应 PASS/应 FAIL 两类用例）")
    ap.add_argument("--onnx", type=str, default=None,
                    help="SONIC decoder ONNX；默认按 apt_g1/model_decoder.onnx -> "
                         "gear_sonic_deploy/policy/release/model_decoder.onnx 依次探测")
    ap.add_argument("--out", type=str, default="outputs/d065/g0_smoke.json",
                    help="JSON 产物路径；默认 outputs/d065/g0_smoke.json —— 以 outputs/ 开头的"
                         "相对路径落在 apt_g1/outputs/ 下（仓内 probe 产物家，gitignore）")
    ap.add_argument("--batch", type=int, default=4, help="冒烟 batch（默认 4）")
    ap.add_argument("--seed", type=int, default=0, help="冒烟随机种子（默认 0）")
    ap.add_argument("--alpha", type=float, default=None,
                    help="LoRA alpha 覆盖值（默认：ckpt metadata 有就用它，否则取 rsl_rl "
                         "默认 1.0；rank 从 down.weight 形状推，低秩 scale=alpha/rank）")
    args = ap.parse_args(argv)

    if args.selftest and args.ckpt:
        print("[WARN] --selftest 与 --ckpt 同时给出：走 selftest，不读 ckpt")
    if not args.selftest and not args.ckpt:
        ap.error("real 模式需要 --ckpt（或加 --selftest 跑自检）")

    if args.selftest:
        print(f"[{EXPERIMENT}-{GATE_NAME}] selftest（合成 ckpt，无外部文件读取）")
        st = run_selftest(seed=args.seed)
        report = {
            "experiment": EXPERIMENT, "gate": GATE_NAME, "mode": "selftest",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "identity": {
                "host": {"python": sys.version.split()[0], "torch": torch.__version__,
                         "numpy": np.__version__, "platform": sys.platform},
                "joint_perm_digest": joint_perm_digest(),
                "script_self_sha256": file_digests(os.path.abspath(__file__))["sha256"],
            },
            "selftest": st,
            "verdict": {"overall": st["verdict"]},
            "notes": KNOWN_LIMITATIONS,
            "license_note": LICENSE_NOTE,
        }
        print(f"D065 G0 SELFTEST {st['verdict']}")
        exit_code = 0 if st["verdict"] == "PASS" else 1
    else:
        report = run_real(args)
        exit_code = 0 if report["verdict"]["overall"] == "PASS" else 1

    out_path = resolve_out_path(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\nJSON 落盘：{out_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
