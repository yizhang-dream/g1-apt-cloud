"""E61/b AXIS-A: closed-loop rollout of the author-v0 token model on MuJoCo G1 flat ground.

自回归用法：每 50 Hz 控制拍前，author 模型吃「W=200 帧滑窗 state36 + 13 维 ctx + 滑窗 token 码」，
输出下一帧 token 码；码 -> value=code/TOKEN_SCALE(16) -> `env.step({"token": ..., "aux": zeros(12)})`，
由 env 内部的冻结 SONIC 解码器把 token 还原成 29 关节目标。指标 per run：
survived / fall_step / n_steps / realized_vx（dx 与 dur 并存）/ h_min / 命令 nominal。

============================================================================
【考古 1】decoder 契约
============================================================================
- `SonicOnnxDecoder`：`apt_g1/sonic/sonic_wrapper.py:23`，输入 994 维；
  `build_decoder_obs` 定义在 `apt_g1/sonic/sonic_wrapper.py:71`；前向 `decode` 在
  `apt_g1/sonic/sonic_wrapper.py:93`（torch 侧/导出侧同源）。
- 994 通道表（`apt_g1/isaac/sonic_action_term.py:24-33`）：
  token 64 + ang_vel 10x3 + joint_pos 10x29 + joint_vel 10x29 + last_act 10x29 + gravity 10x3。
- 本脚本沿用 `planner_sonic.py` 的构造写法：用 `NoQuantDecoder`
  （`apt_g1/_archive/eval_distill.py:12`）包住 `model_decoder.onnx` 的 CPU session，
  跳过 FSQ 量化、直接把喂进来的 token 值当量化值用——这正是「author 出的码直驱动
  decoder」所需的路径。注意：脚本本身不拼 994 维 obs，obs 由 env 内部历史缓冲拼
  （`apt_g1/envs/mujoco_g1_flat_env.py:343` `_get_sonic_history`）。
- token 值语义：语料以 int16 存码，value = code / token_scale(16)；decoder 吃 value。
  零值码 = clip(0 - code_min, 0, vocab - 1)。

============================================================================
【考古 2】state36 拼法出处
============================================================================
`state36 = jp_mj(29) + root_quat_wxyz(4) + root_trans_m(3)`，
出处 `apt_g1/build/d060_windows.py:106`（窗口切片脚本，语料构造口径）。
- jp 是 **MuJoCo 关节序**（与 isaaclab 序之间靠 `G1_MUJOCO_TO_ISAACLAB_DOF` 互换，
  见 `envs/mujoco_g1_flat_env.py` 顶部常量与 `_get_sonic_history`）。
- 本脚本直接取 `env.data.qpos[env.body_qpos_adr]`（29，MuJoCo 序）+ `qpos[3:7]`（wxyz）
  + `qpos[0:3]`（m，世界系）。

============================================================================
【考古 3】env 接口
============================================================================
- `MujocoG1FlatEnv` 定义 `apt_g1/envs/mujoco_g1_flat_env.py:102`；
  构造签名含 `sim_dt=0.005`、`control_decimation=4`（=> 50 Hz 控制）、
  `episode_length_s`、`stand_only`、`use_elastic_band` 等。
- `reset` : `apt_g1/envs/mujoco_g1_flat_env.py:260`（无参，返回 obs）。
- `_get_sonic_history` : `apt_g1/envs/mujoco_g1_flat_env.py:343`（内部 5 路历史）。
- `step(action{token,aux})` : `apt_g1/envs/mujoco_g1_flat_env.py:495`，
  返回 `(obs, reward, terminated, info)`；termination 含 `qpos[2] < 0.2` 等条件。
- env 内部 `step()` 在 terminated 时会**自动调 reset()**；本脚本用子类覆盖
  `_check_termination()` 恒返回 False 抑制自动 reset，从而能读到真实的
  h_min / fall_step（否则摔倒瞬间状态被 reset 冲掉）。fall 判定由脚本自己按
  `qpos[2] < FALL_H` 复刻，口径与 env 一致。

============================================================================
【ctx 13 维布局】= cmd4 + ht4 + onehot3 + hp2  （语义已核对，非假设）
============================================================================
出处：`train_author_v0.py:147-161` `ctx_from_record` + `build/d060_windows.py` 的
`make_command` 口径（TERRAIN_TYPES / TERRAIN_DEFAULT_PARAMS / COMMAND_FIELDS 同源）。
- cmd4 = [target_vel, movement_direction, mode, height]，序 = COMMAND_FIELDS；
  无真值的字段一律哨兵 -1.0（`make_command` 的硬口径：缺省 0 不许被当成真值）。
- ht4     = 四个命令字段各自的 has_truth 标志（有真值 1.0 / 无真值 0.0），
            按命令矩阵实际有无真值填，**不是** height/历史占位。
- onehot3 = **terrain** 三分类 one-hot，序 = TERRAIN_TYPES
            （plane -> [1,0,0]；rough_paper -> [0,1,0]；climbing_box -> [0,0,1]）；
            **与 mode 无关**（旧版按 mode 填是错的）。平地 MuJoCo 闭环恒 = plane。
- hp2     = terrain 的 (height, noise)，取 TERRAIN_DEFAULT_PARAMS 被
            `cmd["terrain_params"]` 覆盖后的值；平地 plane -> (0.0, 0.0)；
            **不是** sin/cos 相位。
- 若 ckpt 里带 `default_ctx` / `ctx_example`，优先直接用其值（绕开本节构造）。
- 本脚本绝不训练、绝不改 ckpt；只做前向。

============================================================================
【风险清单】
============================================================================
R1 命令档位：语料只用 (0.6, SLOW_WALK=1) 与 (1.0, RUN=3) 两档（in-distribution），
   默认矩阵就这两档；**0.3 档已被弃用（OOD，勿再作为默认命令）**。
R2 off-by-one：第 k 拍喂给 env 的 token 是「预测的第 k 帧 token」（用前 k-1 帧
   state/token 推得），即控制 token = 预测的下一帧，存在一拍假设；见 `_rollout_one_seed`
   docstring。若服务器发现系统性滞后，可改成先记录再喂（一行改动）。
R3 算力：author 每 50 Hz 拍都要前向一次（20 s = 1000 次 + 冷启动），房间 CPU 上
   `--no-cuda` 会非常慢；服务器跑建议 cuda（`--device cuda`），本机仅跑 selftest/dry-run。
R4 ctx 后 9 维（ht4/onehot3/hp2）语义已按 `train_author_v0.ctx_from_record` +
   `d060_windows.make_command` 逐数值对齐（见上方布局节；selftest 内有与
   `ctx_from_record` 的同条目一致性断言）。命令 `terrain` / `terrain_params`
   由调用方在 cmd dict 里给，缺省 terrain="plane"（平地闭环口径）。
R5 token 维数/码 shape：author 输出 logits 的尾部形状未确证（见 `codes_from_logits`），
   脚本对 (...,token_dim,vocab)、(...,token_dim*vocab)、(...,vocab) 三种都做了兼容；
   若退化成单码广播，信封里 `single_code_broadcast=true` 会显式标出。
R6 本机无 MuJoCo/onnx：`--selftest` / `--dry-run` 走 mock + CPU torch，不验证物理
   正确性，只验证「延迟 import + 自回归数据流 + 模型契约 + 指标口径」。selftest 需要
   本机 torch（CPU 即可），mujoco/onnxruntime 仍不触发。

Usage（服务器）:
    python sim/rollout_author_v0_mj.py --ckpt outputs/d061_author_v0/ckpt_final.pt \
        --command-vx 0.6,1.0 --seeds 0,1 --dur-s 20 --out-json outputs/d061b_axis_a.json
本机:
    python sim/rollout_author_v0_mj.py --selftest
    python sim/rollout_author_v0_mj.py --dry-run

D062-R1 v1 电池（--v1-ckpt，D062-R1 四臂 anchor/chunk/priv/both ckpt_final.pt）:
    python sim/rollout_author_v0_mj.py --v1-ckpt outputs/d062_r1/both/ckpt_final.pt \
        --command-vx 0.6,1.0 --seeds 0,1 --dur-s 20 --out-json outputs/d062_r1_axis_a.json
    走 train_author_v1.load_author_v1 正规加载（服务器从 sync 克隆包根跑；
    不复用本脚本的 v0 加载副本）；逐帧滑窗自回归口径与 v0 相同——只走 v1 主干
    encode+head，chunk/priv 头不参与；结果 JSON 增记 model_src（可 --model-src
    覆盖标签）与 v1 身份信封（ckpt md5 / variant / model_cfg）。不给 --v1-ckpt
    时 v0 路径行为不变。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# 常量（契约相关，勿随手改）
# ---------------------------------------------------------------------------
AUTHOR_STATE_DIM = 36          # jp29 + quat wxyz4 + trans m3  (d060_windows.py:106)
AUTHOR_CTX_DIM = 13            # cmd4 + ht4 + onehot3 + hp2
WINDOW = 200                   # 滑窗帧数 W
TOKEN_SCALE = 16.0             # value = code / TOKEN_SCALE
CTX_SENTINEL = -1.0            # 未指定命令字段的哨兵（D033，与 d060_windows.SENTINEL 同值）
CTRL_HZ = 50.0                 # sim_dt 0.005 * decimation 4 -> 50 Hz
FALL_H = 0.2                   # 与 env._check_termination 的 qpos[2] 阈值一致
AUX_DIM = 12
DEFAULT_REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
DEC_ONNX_REL = "gear_sonic_deploy/policy/release/model_decoder.onnx"
DEFAULT_CKPT = "outputs/d061_author_v0/ckpt_final.pt"

# ctx 语义常量（train_author_v0.py:79-84 / build/d060_windows.py 同源硬编码副本，
# 本脚本自包含，不 import 那些模块）
COMMAND_FIELDS = ("target_vel", "movement_direction", "mode", "height")
TERRAIN_TYPES = ("plane", "rough_paper", "climbing_box")
TERRAIN_DEFAULT_PARAMS = {"plane": {}, "rough_paper": {"noise": 0.04},
                          "climbing_box": {"height": 0.5}}
# ctx 信封里的语义串（与 build_ctx 实现同源，dry-run / 生产信封共用防漂移）
CTX_LAYOUT_SEMANTIC = (
    "cmd4(target_vel,movement_direction,mode,height; 无真值=-1 哨兵) + "
    "ht4(四字段 has_truth 标志) + "
    "onehot3(terrain: plane/rough_paper/climbing_box) + "
    "hp2(terrain height,noise; plane=(0,0))"
)

# 命令矩阵：语料 in-distribution 两档（vx -> SONIC mode）
#   1 = SLOW_WALK, 3 = RUN
#   mode 语义出处：eval_distill.py:17-23 SCEN（('slow_walk', dict(mode=1, ...))、
#   ('idle' mode=0) / ('walk' mode=2)）；3=RUN 见 gen_planner_scripts_d060.py:106
#   MODE_IDS = {"SLOW_WALK": 1, "WALK": 2, "RUN": 3}。
COMMAND_MODE_TABLE = {0.6: 1, 1.0: 3}
DEFAULT_COMMAND_MODE = 1

METRIC_DEFS_ZH = {
    "survived": "整段跑满 dur_s 且未触发摔倒判据（qpos[2] < 0.2）",
    "fall_step": "首次摔倒的控制拍序号（50 Hz），未摔倒为 null",
    "n_steps": "实际执行的控制拍数（50 Hz）",
    "dx_m": "root x 位移（米，世界系，末值减初值）",
    "dur_s": "实际时长（秒）= n_steps / 50",
    "realized_vx": "实测前进速度 dx_m / dur_s（米/秒）",
    "h_min": "整段 root 高度最小值（米）",
    "cmd": "命令 nominal：target_vel / movement_direction / mode / height",
}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if x.strip() != ""]


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if x.strip() != ""]


def mode_for_vx(vx: float) -> int:
    """0.6 -> SLOW_WALK(1), 1.0 -> RUN(3)，其他档回落到 SLOW_WALK 并标注。"""
    key = round(float(vx), 6)
    for k, v in COMMAND_MODE_TABLE.items():
        if abs(k - key) < 1e-6:
            return v
    return DEFAULT_COMMAND_MODE


def build_run_matrix(command_vx: list[float], seeds: list[int], dur_s: float) -> list[dict]:
    """命令矩阵行（平地主线：terrain 恒 plane -> ctx onehot3=[1,0,0] / hp2=(0,0)）。"""
    rows = []
    for vx in command_vx:
        for sd in seeds:
            rows.append(
                {
                    "target_vel": float(vx),
                    "movement_direction": 0.0,
                    "mode": int(mode_for_vx(vx)),
                    "height": CTX_SENTINEL,
                    "terrain": TERRAIN_TYPES[0],   # plane（平地 MuJoCo 闭环）
                    "terrain_params": {},
                    "seed": int(sd),
                    "dur_s": float(dur_s),
                    "in_distribution": bool(
                        any(abs(k - float(vx)) < 1e-6 for k in COMMAND_MODE_TABLE)
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# ctx 构造（见顶部 ctx 13 维布局；语义 = train_author_v0.ctx_from_record 同口径）
# ---------------------------------------------------------------------------
def build_ctx(cmd: dict, step: int = 0, n_steps: int = 0) -> np.ndarray:
    """cmd(4) + has_truth(4) + terrain one-hot(3) + (height,noise)(2) = 13。

    与 `train_author_v0.ctx_from_record` 逐数值同口径（selftest 有同条目断言）：
    - cmd 4 维序 = COMMAND_FIELDS；`has_truth` 未给时按「值 != -1 哨兵」推断，
      给了就按它裁（无真值字段强制回哨兵，= d060_windows.make_command 的硬口径）。
    - terrain 取 `cmd["terrain"]`，缺省 "plane"（平地闭环）；非法值报错不静默。
    - terrain_params 覆盖 TERRAIN_DEFAULT_PARAMS 缺省后取 (height, noise)。
    step / n_steps 仅为调用点签名兼容保留，自本次语义修复起不再参与计算
    （旧版在此填 sin/cos 相位，已被 terrain (height,noise) 取代）。
    """
    vals = [float(cmd.get(k, CTX_SENTINEL)) for k in COMMAND_FIELDS]
    ht = cmd.get("has_truth")
    if ht is None:
        flags = [v != CTX_SENTINEL for v in vals]
    else:
        flags = [k in ht for k in COMMAND_FIELDS]
    vals = [v if f else CTX_SENTINEL for v, f in zip(vals, flags)]  # make_command:167 口径

    ttype = cmd.get("terrain", TERRAIN_TYPES[0])
    if ttype not in TERRAIN_TYPES:
        raise ValueError(f"cmd terrain 非法 {ttype!r}（允许 {TERRAIN_TYPES}）")
    onehot = [0.0, 0.0, 0.0]
    onehot[TERRAIN_TYPES.index(ttype)] = 1.0
    p = dict(TERRAIN_DEFAULT_PARAMS[ttype])
    p.update(cmd.get("terrain_params") or {})
    hp = [float(p.get("height", 0.0)), float(p.get("noise", 0.0))]

    return np.asarray(vals + [1.0 if f else 0.0 for f in flags] + onehot + hp,
                      dtype=np.float32)


def zero_value_code(code_min: int, vocab: int) -> int:
    """零值码：value=0 对应的码，clip 进 [0, vocab-1]。"""
    return int(min(max(0 - int(code_min), 0), int(vocab) - 1))


def codes_to_token(codes: np.ndarray, token_dim: int, code_min: int) -> np.ndarray:
    """模型 argmax 索引 -> decoder 吃的 token 值。

    模型输入/目标是「码 − code_min」（`train_author_v0.py:363-364, 482-484`），故
    argmax 输出的是**索引**，码 = 索引 + code_min，value = 码 / TOKEN_SCALE。
    同一模型轴 B 口径见 `eval_author_v0_decoder.py:632`
    `tok = (idx + code_min) / token_scale`。
    """
    codes = np.asarray(codes, dtype=np.float32).reshape(-1) + np.float32(int(code_min))
    if codes.size == 1 and token_dim != 1:
        codes = np.tile(codes, int(token_dim))
    if codes.size != int(token_dim):
        raise ValueError(f"codes size {codes.size} != token_dim {token_dim}")
    return (codes / np.float32(TOKEN_SCALE)).astype(np.float32)


# ---------------------------------------------------------------------------
# author 模型：加载 + 前向适配
# ---------------------------------------------------------------------------
def _script_local_paths() -> list[str]:
    """按脚本自身位置推出三个候选根：脚本目录 / 其父目录（apt_g1）/ 仓根。

    服务器平铺布局（脚本由 PYTHONPATH 解析）与本地仓内布局（apt_g1/sim/…）都能覆盖；
    只在 import 前补 sys.path，不改环境变量。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
    return [p for p in cands if p and os.path.isdir(p)]


def _import_train_author():
    """服务器上脚本平铺在 GR00T-WholeBodyControl 下跑，也可能带 apt_g1 包前缀。"""
    import importlib

    for p in reversed(_script_local_paths()):  # 仓根优先（apt_g1.* 与 training.* 都通）
        if p not in sys.path:
            sys.path.insert(0, p)
    errs = []
    for mod in (
        "train_author_v0",
        "training.train_author_v0",
        "apt_g1.training.train_author_v0",
        "apt_g1.train_author_v0",
    ):
        try:
            return importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{mod}: {e}")
    raise ImportError("无法 import train_author_v0；候选全部失败:\n  " + "\n  ".join(errs))


def _import_train_author_v1():
    """同 _import_train_author，但目标模块为 train_author_v1（D062-R1 v1 ckpt 用）。

    服务器平铺执行根（train_author_v1.py 在 sys.path 顶层）与本地仓内布局
    （apt_g1/training/ 包，从 sync 克隆包根跑）都覆盖。
    """
    import importlib

    for p in reversed(_script_local_paths()):  # 仓根优先（apt_g1.* 与 training.* 都通）
        if p not in sys.path:
            sys.path.insert(0, p)
    errs = []
    for mod in (
        "train_author_v1",
        "training.train_author_v1",
        "apt_g1.training.train_author_v1",
        "apt_g1.train_author_v1",
    ):
        try:
            return importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{mod}: {e}")
    raise ImportError("无法 import train_author_v1；候选全部失败:\n  " + "\n  ".join(errs))


def _filter_kwargs(cls_or_fn, cfg: dict) -> dict:
    import inspect

    try:
        params = inspect.signature(cls_or_fn).parameters
    except (TypeError, ValueError):
        return dict(cfg)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(cfg)
    return {k: v for k, v in cfg.items() if k in params}


def _resolve_forward(model):
    """按 forward 形参名把 (states, tokens, intent, ctx) 映射到调用方式。

    名字里含 ctx/context/cond/cmd -> ctx；含 tok/code/act -> tokens；
    含 intent -> intent（与 tokens 同槽实参：部署期 intent 槽 = 生成码，与轴 C
    autoregressive_rollout 的两槽同回填口径一致）；含 state/obs/prop/hist -> states。
    无法识别时按位置 (states, tokens, intent, ctx) 兜底（intent 紧跟 tokens，
    ctx 仍居末位；见 R5，属未确证假设）。

    真实签名 `AuthorV0Transformer.forward(tokens_idx, state, intent_idx, ctx_feat)`
    （train_author_v0.py:416）四槽全部按关键词命中，不走兜底。
    """
    import inspect

    try:
        names = [
            p.name
            for p in inspect.signature(model.forward).parameters.values()
            if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
        ]
    except (TypeError, ValueError):
        names = []

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for n in names:
        ln = n.lower()
        if any(k in ln for k in ("ctx", "context", "cond", "cmd", "command")) and "ctx" not in used:
            mapping[n] = "ctx"
        elif any(k in ln for k in ("tok", "code", "act")) and "tokens" not in used:
            mapping[n] = "tokens"
        elif "intent" in ln and "intent" not in used:
            mapping[n] = "intent"
        elif any(
            k in ln for k in ("state", "obs", "prop", "hist", "window")
        ) and "states" not in used:
            mapping[n] = "states"
        used.add(mapping.get(n, ""))
    ordered = [k for k in ("states", "tokens", "intent", "ctx") if k not in mapping.values()]
    idx = 0
    for n in names:
        if n not in mapping and idx < len(ordered):
            mapping[n] = ordered[idx]
            idx += 1
    unbound = [n for n in names if n not in mapping]
    if unbound:  # 静默丢参会在 model() 处炸成难读的 TypeError，这里 fail loud
        raise ValueError(
            f"forward 形参 {unbound} 无法映射到 (states/tokens/intent/ctx)；"
            f"完整形参 {names}，已映射 {mapping}。请扩关键词表或改调用点。")

    def call(states: np.ndarray, tokens: np.ndarray, ctx: np.ndarray):
        if not names:
            # 形参表不可自省：按上方兜底序 (states, tokens, intent=tokens, ctx) 位置调用
            return model(states, tokens, tokens, ctx)
        kwargs = {}
        for n in names:
            kind = mapping.get(n)
            if kind == "states":
                kwargs[n] = states
            elif kind == "tokens":
                kwargs[n] = tokens
            elif kind == "intent":
                kwargs[n] = tokens  # 部署期 intent 槽与 tokens 同源（见 docstring）
            elif kind == "ctx":
                kwargs[n] = ctx
        return model(**kwargs)

    return call, mapping


def codes_from_logits(logits, token_dim: int, vocab: int) -> tuple[np.ndarray, bool]:
    """从 (..., vocab) / (..., token_dim, vocab) / (..., token_dim*vocab) 取最后一拍的码。

    返回 (codes[token_dim] int64, single_code_broadcast)。
    """
    arr = logits
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 0:
        raise ValueError("logits 维度为 0，无法解析")
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]  # 去掉 batch 维
    if arr.ndim >= 2 and arr.shape[-1] == token_dim * vocab:
        arr = arr.reshape(arr.shape[:-1] + (token_dim, vocab))
    last = arr[-1] if arr.ndim >= 2 else arr  # 取最后一拍（batch 已去）
    if last.ndim >= 2 and last.shape[-1] == vocab:
        codes = np.argmax(last, axis=-1).reshape(-1)
    elif last.ndim == 1 and last.shape[0] == vocab:
        codes = np.argmax(last).reshape(1)
    else:
        # 已是码（或无法识别）：尽量当码用
        codes = np.asarray(last).reshape(-1).astype(np.int64)
    if codes.size == 1 and token_dim != 1:
        return codes, True
    if codes.size != token_dim:
        raise ValueError(f"从 logits {arr.shape} 解析出的码数 {codes.size} != token_dim {token_dim}")
    return codes, False


class AuthorRunner:
    """author-v0 的自回归包装：维护 W=200 帧滑窗，每拍吐下一帧 token 值（token_dim,）。

    窗口逐帧存的是**一整帧 (token_dim,) 码**（= `train_author_v0.WindowDataset`
    的 `token_stream[j]` 形状 (WIN, TOKEN_DIM)）：真实 `AuthorV0Transformer.forward`
    要求 tokens_idx/intent_idx 为 (b, t, K) 码张量、state 为 (b, t, 36)，故
    `_batches` 组出 (1, W, token_dim) / (1, W, 36) / (1, 13)。若模型退化成单码输出
    （见 R5），该单码广播成整帧后入窗，信封以 `single_code_broadcast` 标注。
    """

    def __init__(self, model, cfg: dict, device: str = "cpu", default_ctx=None):
        self.model = model
        self.cfg = dict(cfg)
        self.device = device
        self.vocab = int(cfg["vocab"])
        self.code_min = int(cfg.get("code_min", 0))
        self.state_dim = int(cfg.get("state_dim", AUTHOR_STATE_DIM))
        self.win = int(cfg.get("win", WINDOW))
        self.token_dim = int(cfg["token_dim"])
        self.default_ctx = None if default_ctx is None else np.asarray(default_ctx, np.float32)
        self.ctx_source = "built" if default_ctx is None else "ckpt"
        self.single_code_broadcast = False
        self._forward, self._fwd_map = _resolve_forward(model)
        self._zero_code = zero_value_code(self.code_min, self.vocab)
        self.reset()

    # -- 窗口 ---------------------------------------------------------------
    def _zero_frame(self) -> np.ndarray:
        return np.full(self.token_dim, self._zero_code, np.int64)

    def reset(self, s0: np.ndarray | None = None):
        s0 = np.zeros(self.state_dim, np.float32) if s0 is None else np.asarray(s0, np.float32)
        self._states = deque([s0.copy() for _ in range(self.win)], maxlen=self.win)
        self._codes = deque([self._zero_frame() for _ in range(self.win)], maxlen=self.win)

    def _batches(self, ctx: np.ndarray):
        states = np.stack(self._states, axis=0)[None].astype(np.float32)   # (1,W,36)
        codes = np.stack(self._codes, axis=0)[None].astype(np.int64)      # (1,W,token_dim)
        import torch

        st = torch.from_numpy(states).to(self.device)
        ck = torch.from_numpy(codes).to(self.device)
        cx = torch.from_numpy(np.asarray(ctx, np.float32)[None]).to(self.device)
        return st, ck, cx

    def _run_model(self, states, tokens, ctx):
        import torch

        with torch.no_grad():
            out = self._forward(states, tokens, ctx)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def _as_frame(self, codes: np.ndarray) -> np.ndarray:
        """码 -> 一整帧 (token_dim,) 索引（单码广播时补齐；越界码 clip 进 vocab）。"""
        c = np.asarray(codes).reshape(-1).astype(np.int64)
        if c.size == 0:
            return self._zero_frame()
        if c.size == 1 and self.token_dim != 1:
            c = np.tile(c, self.token_dim)
        if c.size != self.token_dim:
            raise ValueError(f"码数 {c.size} != token_dim {self.token_dim}，无法组成整帧")
        return np.clip(c, 0, self.vocab - 1)

    def step(self, state36: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        """把当前 state36 推进窗口，返回下一帧 token 值（token_dim,）。"""
        self._states.append(np.asarray(state36, np.float32).reshape(-1))
        st, ck, cx = self._batches(ctx)
        logits = self._run_model(st, ck, cx)
        codes, single = codes_from_logits(logits, self.token_dim, self.vocab)
        self.single_code_broadcast = bool(single)
        self._codes.append(self._as_frame(codes))
        return codes_to_token(codes, self.token_dim, self.code_min)

    def push_realized_code(self, codes: np.ndarray):
        """可选：用实际喂出去的码回填（当前闭环与预测码一致，故默认不调）。"""
        if np.asarray(codes).size:
            self._codes[-1] = self._as_frame(codes)


# ---------------------------------------------------------------------------
# env / decoder 构造（延迟 import，工厂可注入）
# ---------------------------------------------------------------------------
def _ensure_sys_path(repo: str) -> None:
    """照 planner_sonic.py:68-71 的先例补齐 sys.path，兼容平铺/包两种写法。"""
    parent = os.path.dirname(os.path.abspath(repo))
    for p in (repo, os.path.join(repo, "apt_g1"), parent):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def _import_noquant_decoder():
    import importlib

    errs = []
    for mod in ("eval_distill", "eval.eval_distill", "apt_g1.eval.eval_distill",
                "apt_g1._archive.eval_distill"):
        try:
            return importlib.import_module(mod).NoQuantDecoder
        except Exception as e:  # noqa: BLE001
            errs.append(f"{mod}: {e}")
    raise ImportError("无法 import NoQuantDecoder；候选全部失败:\n  " + "\n  ".join(errs))


def _no_auto_reset_env_cls():
    """MujocoG1FlatEnv 的子类：覆盖 _check_termination 抑制 step 内自动 reset。"""
    import importlib

    base = None
    errs = []
    for mod in ("envs.mujoco_g1_flat_env", "apt_g1.envs.mujoco_g1_flat_env"):
        try:
            base = importlib.import_module(mod).MujocoG1FlatEnv
            break
        except Exception as e:  # noqa: BLE001
            errs.append(f"{mod}: {e}")
    if base is None:
        raise ImportError("无法 import MujocoG1FlatEnv；候选全部失败:\n  " + "\n  ".join(errs))

    class _NoAutoResetEnv(base):  # type: ignore[misc, valid-type]
        """fall 判据由调用方复刻（FALL_H），避免摔倒瞬间被 reset 冲掉 h_min/fall_step。"""

        def _check_termination(self):  # noqa: D102
            return False

    _NoAutoResetEnv.__name__ = "NoAutoResetMujocoG1FlatEnv"
    return _NoAutoResetEnv


def make_mujoco_factory(repo: str, ckpt_dec_onnx: str):
    """返回 factory(seed, cmd) -> env（每次调用一个干净 env）。"""

    def factory(seed: int, cmd: dict):
        import mujoco  # noqa: F401  (延迟 import：本机 selftest/dry-run 不触发)

        _ensure_sys_path(repo)
        np.random.seed(int(seed))
        NoQuantDecoder = _import_noquant_decoder()
        EnvCls = _no_auto_reset_env_cls()
        env = EnvCls(
            NoQuantDecoder(ckpt_dec_onnx),
            repo,
            use_elastic_band=False,
            stand_only=True,
        )
        return env

    return factory


# ---------------------------------------------------------------------------
# 闭环
# ---------------------------------------------------------------------------
def state36_from_env(env) -> np.ndarray:
    """jp_mj(29) + root_quat_wxyz(4) + root_trans_m(3)。出处 d060_windows.py:106。"""
    qpos = np.asarray(env.data.qpos)
    jp = qpos[np.asarray(env.body_qpos_adr)]
    q = np.concatenate([np.asarray(jp, np.float32).reshape(-1),
                        np.asarray(qpos[3:7], np.float32).reshape(-1),
                        np.asarray(qpos[0:3], np.float32).reshape(-1)])
    if q.size != AUTHOR_STATE_DIM:
        raise ValueError(f"state36 拼出 {q.size} 维，期望 {AUTHOR_STATE_DIM}")
    return q


def rollout_one_seed(env, runner: AuthorRunner, cmd: dict, dur_s: float, seed: int) -> dict:
    """单 episode 闭环。

    off-by-one 假设（R2）：第 k 拍（k=1..N）执行前先做一次 author 前向，喂给
    env.step 的是「预测的第 k 帧 token」，其条件窗是前 k-1 拍已观测到的
    state36 与已执行的 token 码——即控制 token 永远是"下一帧"，本身不含任何
    lookahead；若服务器实测发现系统性滞后一拍，把 `env.step` 与 `runner.step`
    顺序对调即可（本函数内已用注释标出）。
    """
    env.reset()
    s0 = state36_from_env(env)
    runner.reset(s0)
    n_steps = int(round(float(dur_s) * CTRL_HZ))

    x0 = float(env.data.qpos[0])
    h_min = float(env.data.qpos[2])
    fall_step = None
    tokens_log: list[int] = []
    steps_done = 0

    for k in range(n_steps):
        ctx = runner.default_ctx if runner.default_ctx is not None else build_ctx(cmd, k, n_steps)
        tok = runner.step(state36_from_env(env), ctx)   # <- 预测下一帧 token
        # 仅记录首维；round(tok*16) = argmax 索引 + code_min = 语料口径的 FSQ 码值
        tokens_log.append(int(round(float(tok.reshape(-1)[0]) * TOKEN_SCALE)))
        _, _, _, _ = env.step({"token": tok,
                               "aux": np.zeros(AUX_DIM, dtype=np.float32)})
        steps_done += 1
        h = float(env.data.qpos[2])
        h_min = min(h_min, h)
        if fall_step is None and h < FALL_H:
            fall_step = k + 1
            break
        if not np.all(np.isfinite(np.asarray(env.data.qpos, np.float64))):
            fall_step = k + 1
            break

    x1 = float(env.data.qpos[0])
    dx = x1 - x0
    dur = steps_done / CTRL_HZ
    return {
        "survived": bool(fall_step is None),
        "fall_step": fall_step,
        "n_steps": int(steps_done),
        "dx_m": round(float(dx), 4),
        "dur_s": round(float(dur), 4),
        "realized_vx": round(float(dx / dur), 4) if dur > 0 else None,
        "h_min": round(float(h_min), 4),
        "h_end": round(float(env.data.qpos[2]), 4),
        "x_end": round(x1, 4),
        "tokens_head": tokens_log[:5],
        "seed": int(seed),
        "cmd": dict(cmd),
    }


def summarize(per_run: list[dict]) -> dict:
    out: dict = {}
    for r in per_run:
        key = f"vx{float(r['cmd']['target_vel']):.2f}_mode{int(r['cmd']['mode'])}"
        bucket = out.setdefault(key, {"n_runs": 0, "n_survived": 0, "vx_sum": 0.0, "vx_n": 0})
        bucket["n_runs"] += 1
        bucket["n_survived"] += 1 if r["survived"] else 0
        if r["realized_vx"] is not None:
            bucket["vx_sum"] += float(r["realized_vx"])
            bucket["vx_n"] += 1
    for key, b in out.items():
        b["survival_rate"] = round(b["n_survived"] / b["n_runs"], 4) if b["n_runs"] else None
        b["realized_vx_mean"] = round(b["vx_sum"] / b["vx_n"], 4) if b["vx_n"] else None
        b["realized_vx_n"] = b["vx_n"]
        del b["vx_sum"]
        del b["vx_n"]
    return out


# ---------------------------------------------------------------------------
# selftest / dry-run（本机可跑：selftest 需 CPU torch；mujoco/onnx 一律不触发）
# ---------------------------------------------------------------------------
class _MockEnv:
    """极简 mock env：走 x 方向，可按 script 在指定拍"摔倒"。"""

    def __init__(self, vx: float = 0.5, fall_at: int | None = None):
        self.data = type("D", (), {})()
        self.data.qpos = np.zeros(36, dtype=np.float64)
        self.data.qpos[2] = 0.76
        self.data.qpos[3] = 1.0
        self.body_qpos_adr = np.arange(7, 36)
        self.vx = float(vx)
        self.fall_at = fall_at
        self.n = 0
        self.seen_tokens: list[np.ndarray] = []

    def reset(self):
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = 0.76
        self.data.qpos[3] = 1.0
        self.n = 0
        return np.zeros(8, np.float32)

    def step(self, action):
        tok = np.asarray(action["token"], np.float32)
        assert tok.shape == (4,), f"mock 期望 token_dim=4, 得到 {tok.shape}"
        assert np.asarray(action["aux"]).shape == (AUX_DIM,)
        self.seen_tokens.append(tok.copy())
        self.data.qpos[0] += self.vx / CTRL_HZ
        self.n += 1
        if self.fall_at is not None and self.n >= self.fall_at:
            self.data.qpos[2] = 0.05
        return np.zeros(8, np.float32), 0.0, False, {}


class _MockAuthor:
    """假 author：不依赖 torch，产出可预测的码，用来验证滑窗/码->token/指标口径。

    窗口形状与 `AuthorRunner` 同契约：逐帧存整帧 (token_dim,) 码。
    """

    def __init__(self, token_dim: int = 4, vocab: int = 32, code_min: int = 0):
        self.token_dim = token_dim
        self.vocab = vocab
        self.code_min = code_min
        self.default_ctx = None
        self.single_code_broadcast = False
        self.reset()

    def reset(self, s0=None):
        self._zero = zero_value_code(self.code_min, self.vocab)
        self._states = deque([np.zeros(AUTHOR_STATE_DIM, np.float32) for _ in range(WINDOW)],
                             maxlen=WINDOW)
        self._codes = deque([np.full(self.token_dim, self._zero, np.int64)
                             for _ in range(WINDOW)], maxlen=WINDOW)

    def step(self, state36: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        assert np.asarray(state36).shape == (AUTHOR_STATE_DIM,), state36.shape
        assert np.asarray(ctx).shape == (AUTHOR_CTX_DIM,), ctx.shape
        self._states.append(np.asarray(state36, np.float32))
        codes = np.full((self.token_dim,), (self.code_min + 3) % self.vocab, np.int64)
        tok = codes_to_token(codes, self.token_dim, self.code_min)
        assert np.allclose(tok, (codes[0] + self.code_min) / TOKEN_SCALE)
        self._codes.append(codes)
        return tok


def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    print("[selftest] 延迟 import 检查（不应触发 mujoco/onnx/torch）")
    check("mujoco 未 import", "mujoco" not in sys.modules)
    check("torch 未 import", "torch" not in sys.modules)
    check("onnxruntime 未 import", "onnxruntime" not in sys.modules)

    print("[selftest] 码/ctx/矩阵工具")
    check("zero_value_code(code_min=0) == 0", zero_value_code(0, 32) == 0)
    check("zero_value_code(code_min=-16) == 16", zero_value_code(-16, 32) == 16)
    check("zero_value_code clip 上界", zero_value_code(-64, 32) == 31)
    cmd = {"target_vel": 0.6, "movement_direction": 0.0, "mode": 1, "height": -1.0}
    ctx = build_ctx(cmd, 0, 10)
    check("ctx 维度 13", ctx.shape == (AUTHOR_CTX_DIM,))
    check("ctx cmd4 = [0.6, 0.0, 1.0, -1.0]", list(np.round(ctx[:4], 3)) == [0.6, 0.0, 1.0, -1.0])
    check("ctx ht4 = has_truth [1,1,1,0]（height 无真值）", list(ctx[4:8]) == [1.0, 1.0, 1.0, 0.0])
    check("ctx onehot3 = terrain plane [1,0,0]", list(ctx[8:11]) == [1.0, 0.0, 0.0])
    check("ctx hp2 = plane (0.0, 0.0)", list(ctx[11:14]) == [0.0, 0.0])
    check("ctx 无相位残留（step/n_steps 不改数值）",
          np.array_equal(ctx, build_ctx(cmd, 37, 37)))
    check("onehot3 与 mode 无关（mode=3 仍 [1,0,0]）",
          list(build_ctx(dict(cmd, mode=3), 0, 10)[8:11]) == [1.0, 0.0, 0.0])
    box = build_ctx({"target_vel": 0.6, "terrain": "climbing_box"}, 0, 10)
    check("ctx climbing_box -> onehot [0,0,1] / hp (0.5, 0.0)",
          list(box[8:11]) == [0.0, 0.0, 1.0] and np.allclose(box[11:14], [0.5, 0.0]))
    rp = build_ctx({"target_vel": 0.6, "terrain": "rough_paper",
                    "terrain_params": {"noise": 0.07}}, 0, 10)
    check("ctx rough_paper -> onehot [0,1,0] / noise 覆盖缺省 0.07",
          list(rp[8:11]) == [0.0, 1.0, 0.0] and np.allclose(rp[11:14], [0.0, 0.07]))
    red = build_ctx({"target_vel": 0.6, "height": 0.7, "has_truth": ["target_vel"]}, 0, 10)
    check("has_truth 不含 height -> 值强制回哨兵（make_command 口径）",
          list(red[:4]) == [0.6, -1.0, -1.0, -1.0] and list(red[4:8]) == [1.0, 0.0, 0.0, 0.0])
    try:
        build_ctx({"target_vel": 0.6, "terrain": "moon"}, 0, 10)
        raised = False
    except ValueError:
        raised = True
    check("非法 terrain fail loud（不静默）", raised)
    check("mode_for_vx(0.6)=1", mode_for_vx(0.6) == 1)
    check("mode_for_vx(1.0)=3", mode_for_vx(1.0) == 3)
    m = build_run_matrix([0.6, 1.0], [0, 1], 20.0)
    check("run 矩阵 2vx x 2seed = 4", len(m) == 4)
    check("矩阵全部 in-distribution", all(r["in_distribution"] for r in m))

    print("[selftest] codes_from_logits 三种尾部形状")
    vocab, tdim = 32, 4
    lg = np.zeros((1, WINDOW, tdim, vocab), np.float32)
    lg[0, -1, :, 5] = 10.0
    c, single = codes_from_logits(lg, tdim, vocab)
    check("(...,token_dim,vocab) -> 码 [5,5,5,5]", list(c) == [5] * 4 and not single)
    lg2 = np.zeros((1, WINDOW, tdim * vocab), np.float32)
    lg2[0, -1, 2 * vocab + 7] = 10.0
    c2, _ = codes_from_logits(lg2, tdim, vocab)
    check("(...,token_dim*vocab) reshape 正确", list(c2) == [0, 0, 7, 0])
    lg3 = np.zeros((1, WINDOW, vocab), np.float32)
    lg3[0, -1, 9] = 10.0
    c3, s3 = codes_from_logits(lg3, tdim, vocab)
    check("(...,vocab) 单码广播且打标", list(c3) == [9] and s3)
    check("codes_to_token = (index+code_min)/TOKEN_SCALE（code_min=-14）",
          np.allclose(codes_to_token(np.array([16, 0, 8]), 3, -14),
                      np.array([(16 - 14) / 16, (0 - 14) / 16, (8 - 14) / 16], np.float32)))
    check("codes_to_token code_min=0 时退化为 index/16",
          np.allclose(codes_to_token(np.array([16, 0, 8]), 3, 0),
                      np.array([1.0, 0.0, 0.5], np.float32)))
    check("code_min=-14, index=14 -> value 0.0（冷启动零值码）",
          np.allclose(codes_to_token(np.array([14]), 1, -14), np.array([0.0], np.float32)))
    check("code_min=-14, index=0 -> value (0-14)/16 = -0.875",
          np.allclose(codes_to_token(np.array([0]), 1, -14), np.array([-0.875], np.float32)))

    print("[selftest] 冷启动窗口 + 闭环指标（mock env / mock author）")
    runner = _MockAuthor(token_dim=4, vocab=32)
    check("冷启动 token 槽=整帧零值码",
          len(runner._codes) == WINDOW
          and all(int(np.asarray(f).max()) == 0 and np.asarray(f).shape == (4,)
                  for f in runner._codes))
    check("冷启动 state 槽=0", all(float(np.abs(s).max()) == 0.0 for s in runner._states))
    cmd = {"target_vel": 0.5, "movement_direction": 0.0, "mode": 1, "height": -1.0}
    env = _MockEnv(vx=0.5)
    r = rollout_one_seed(env, runner, cmd, dur_s=1.0, seed=0)
    check("n_steps = 1s*50Hz = 50", r["n_steps"] == 50)
    check("mock vx=0.5 回读 realized_vx ~0.5", abs(r["realized_vx"] - 0.5) < 1e-6)
    check("dx_m ~0.5m", abs(r["dx_m"] - 0.5) < 1e-6)
    check("未摔倒 survived=True / fall_step=None", r["survived"] and r["fall_step"] is None)
    check("h_min 记录到 0.76", abs(r["h_min"] - 0.76) < 1e-9)
    check("env 收到 50 个 token", len(env.seen_tokens) == 50)

    env_f = _MockEnv(vx=0.5, fall_at=30)
    r_f = rollout_one_seed(env_f, _MockAuthor(), cmd, dur_s=1.0, seed=1)
    check("摔倒截断：fall_step=30", r_f["fall_step"] == 30)
    check("摔倒不自动 reset：n_steps=30", r_f["n_steps"] == 30)
    check("摔倒 survived=False", r_f["survived"] is False)
    check("摔倒 h_min<0.2", r_f["h_min"] < 0.2)

    s = summarize([r, {**r, "cmd": {"target_vel": 1.0, "mode": 3}},
                   {**r_f, "cmd": {"target_vel": 1.0, "mode": 3}}])
    check("summary 分档 key", set(s.keys()) == {"vx0.50_mode1", "vx1.00_mode3"})
    check("summary vx0.5 survival_rate=1.0", s["vx0.50_mode1"]["survival_rate"] == 1.0)
    check("summary vx1.0 survival_rate=0.5", s["vx1.00_mode3"]["survival_rate"] == 0.5)
    check("metric_defs 覆盖 realized_vx", "realized_vx" in METRIC_DEFS_ZH)

    print("[selftest] ctx 与 train_author_v0.ctx_from_record 同条目逐数值一致（SHOULD-FIX-2）")
    TA = _import_train_author()
    check("train_author_v0 已加载（torch 延后到此处才 import）",
          TA.__name__.split(".")[-1] == "train_author_v0")
    ctx_cases = [
        # (rollout cmd, manifest 窗记录) —— 同一命令的两种载体，ctx 必须逐位相同
        ({"target_vel": 0.6, "movement_direction": 0.0, "mode": 1, "height": -1.0},
         {"stem": "c0", "target_vel": 0.6, "movement_direction": 0.0, "mode": 1.0,
          "height": -1.0, "has_truth": ["target_vel", "movement_direction", "mode"],
          "terrain": "plane", "terrain_params": {}}),
        ({"target_vel": 1.0, "movement_direction": 0.0, "mode": 3, "height": 0.5,
          "terrain": "climbing_box", "terrain_params": {"height": 0.5}},
         {"stem": "c1", "target_vel": 1.0, "movement_direction": 0.0, "mode": 3.0,
          "height": 0.5, "has_truth": ["target_vel", "movement_direction", "mode", "height"],
          "terrain": "climbing_box", "terrain_params": {"height": 0.5}}),
        ({"target_vel": 0.6, "terrain": "rough_paper"},
         {"stem": "c2", "target_vel": 0.6, "movement_direction": -1.0, "mode": -1.0,
          "height": -1.0, "has_truth": ["target_vel"], "terrain": "rough_paper",
          "terrain_params": {}}),
        ({"target_vel": 0.6, "height": 0.7, "has_truth": ["target_vel"]},
         {"stem": "c3", "target_vel": 0.6, "movement_direction": -1.0, "mode": -1.0,
          "height": -1.0, "has_truth": ["target_vel"], "terrain": "plane",
          "terrain_params": {}}),
    ]
    for i, (cmd_i, rec_i) in enumerate(ctx_cases):
        ref = TA.ctx_from_record(rec_i)
        mine = build_ctx(cmd_i, 0, 10)
        check(f"case{i} build_ctx == ctx_from_record（dtype/shape/数值）",
              mine.shape == ref.shape == (AUTHOR_CTX_DIM,) and mine.dtype == ref.dtype
              and np.array_equal(mine, ref))

    print("[selftest] 真实 AuthorV0Transformer 走通 AuthorRunner.step（BLOCKER-1 回归）")
    import shutil
    import tempfile

    import torch

    model_kw = dict(vocab=27, d_model=32, n_layer=1, n_head=2, ffn=64, grad_ckpt=False)
    real = TA.AuthorV0Transformer(**model_kw).eval()
    _, fmap = _resolve_forward(real)
    check("forward 四槽关键词全命中（intent_idx -> intent）",
          fmap == {"tokens_idx": "tokens", "state": "states",
                   "intent_idx": "intent", "ctx_feat": "ctx"})
    tmp = tempfile.mkdtemp(prefix="d061b_selftest_")
    try:
        ckpt_path = os.path.join(tmp, "ckpt_final.pt")
        torch.save({"format": "d061a.v0", "step": 0, "model": real.state_dict(),
                    "model_cfg": dict(model_kw, state_dim=AUTHOR_STATE_DIM, win=WINDOW,
                                      token_dim=64, code_min=-14)}, ckpt_path)
        loaded, cfg, dctx = _load_author(ckpt_path, "cpu")
        check("_load_author 读回小模型（cfg.vocab=27 / default_ctx=None）",
              cfg["vocab"] == 27 and cfg["token_dim"] == 64 and dctx is None)
        runner_real = AuthorRunner(loaded, cfg, device="cpu")
        check("AuthorRunner 冷启动窗口 = W",
              len(runner_real._states) == WINDOW and len(runner_real._codes) == WINDOW)
        tok = runner_real.step(np.zeros(AUTHOR_STATE_DIM, np.float32), build_ctx(cmd, 0, 10))
        check("真实模型 step 返回 (token_dim,) 有限 token",
              tok.shape == (64,) and bool(np.all(np.isfinite(tok))))
        check("token 值 = 码/16（对齐 TOKEN_SCALE）",
              bool(np.allclose(tok, np.round(tok * TOKEN_SCALE) / TOKEN_SCALE)))
        check("真实模型未退化成单码广播", runner_real.single_code_broadcast is False)
        last_frame = np.asarray(runner_real._codes[-1]).reshape(-1)
        check("整帧码已回填进窗口（自回归闭环 / 形状 token_dim）",
              last_frame.shape == (cfg["token_dim"],)
              and bool(np.all((last_frame >= 0) & (last_frame < cfg["vocab"]))))
        tok2 = runner_real.step(np.ones(AUTHOR_STATE_DIM, np.float32),
                                build_ctx(cmd, 1, 10))
        check("第 2 拍仍可前向（滑窗 + intent 两槽同源）",
              tok2.shape == (64,) and bool(np.all(np.isfinite(tok2))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("[selftest] D062-R1 v1 适配层（--v1-ckpt：load_author_v1 → v0 四槽逐帧适配）")
    try:
        T1 = _import_train_author_v1()
    except ImportError as e:  # v1 电池环境才要求该模块；缺则跳过不判负
        T1 = None
        print(f"  [skip] train_author_v1 不可导入（{str(e).splitlines()[0]}）")
    if T1 is not None:
        v1_kw = dict(d_model=32, n_layer=1, n_head=2, ffn=64, grad_ckpt=False,
                     use_chunk=True, use_priv=True, chunk_k=4, chunk_prefix=4,
                     chunk_d_model=32, chunk_layers=1, chunk_n_head=2,
                     chunk_ffn=64, priv_hidden=32, priv_extra=0)
        v1_model = T1.AuthorV1Transformer(27, **v1_kw).eval()
        tmp_v1 = tempfile.mkdtemp(prefix="d061b_selftest_v1_")
        try:
            v1_path = os.path.join(tmp_v1, "ckpt_final.pt")
            v1_args = argparse.Namespace(
                d_model=32, n_layer=1, n_head=2, ffn=64, grad_ckpt=False,
                chunk_k=4, chunk_prefix=4, chunk_d_model=32, chunk_layers=1,
                chunk_n_head=2, chunk_ffn=64, priv_hidden=32,
                priv_extra_cols=None, variant="both")
            T1.save_ckpt(v1_path, v1_model, 0, v1_args,
                         {"vocab": 27, "code_min": -14})
            m1, cfg1, dctx1 = _load_author_v1(v1_path, "cpu")
            check("v1 ckpt 经 load_author_v1 读回（variant=both/vocab=27/"
                  "default_ctx=None）",
                  cfg1.get("variant") == "both" and int(cfg1["vocab"]) == 27
                  and dctx1 is None)
            runner_v1 = AuthorRunner(m1, cfg1, device="cpu")
            tok1 = runner_v1.step(np.zeros(AUTHOR_STATE_DIM, np.float32),
                                  build_ctx(cmd, 0, 10))
            check("v1 适配层 step → (token_dim,)=64 有限 token",
                  tok1.shape == (64,) and bool(np.all(np.isfinite(tok1))))
            check("v1 适配层无单码广播（主干逐帧头完整）",
                  runner_v1.single_code_broadcast is False)
        finally:
            shutil.rmtree(tmp_v1, ignore_errors=True)

    print("AXIS_A_SELFTEST_" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def dry_run(args) -> int:
    matrix = build_run_matrix(parse_floats(args.command_vx), parse_ints(args.seeds), args.dur_s)
    using_v1 = bool(getattr(args, "v1_ckpt", None))
    plan = {
        "mode": "dry-run",
        "ckpt": os.path.abspath(args.v1_ckpt) if using_v1 else args.ckpt,
        "ckpt_exists": os.path.exists(args.v1_ckpt if using_v1 else args.ckpt),
        "author_module": ("train_author_v1.load_author_v1（--v1-ckpt）" if using_v1
                          else "train_author_v0 (延迟 import)"),
        "author_cls": ("AuthorV1Transformer→v0 四槽逐帧适配（encode+head，"
                       "chunk/priv 头不参与）" if using_v1
                       else "AuthorV0Transformer"),
        "repo": args.repo,
        "decoder_onnx": os.path.join(args.repo, DEC_ONNX_REL),
        "device": args.device,
        "ctrl_hz": CTRL_HZ,
        "window": WINDOW,
        "token_scale": TOKEN_SCALE,
        "ctx_dim": AUTHOR_CTX_DIM,
        "ctx_layout": CTX_LAYOUT_SEMANTIC,
        "aux_dim": AUX_DIM,
        "fall_h": FALL_H,
        "out_json": args.out_json,
        "n_runs": len(matrix),
        "run_matrix": matrix,
        "load_plan": [
            ("1) import train_author_v1.load_author_v1 -> 逐帧 encode+head 适配"
             if using_v1 else
             "1) import train_author_v0 -> AuthorV0Transformer(**ckpt['model_cfg']) -> load_state_dict"),
            "2) import eval_distill.NoQuantDecoder(repo/" + DEC_ONNX_REL + ")",
            "3) import envs.mujoco_g1_flat_env.MujocoG1FlatEnv (子类抑制自动 reset)",
            "4) 每 run: env.reset() -> 冷启动滑窗填空 -> 50Hz 自回归 token -> env.step",
        ],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print("[dry-run] 未 import mujoco/onnx/torch；按上面矩阵准备服务器执行。")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _load_author(ckpt_path: str, device: str, repo: str = DEFAULT_REPO):
    import torch

    _ensure_sys_path(repo)
    TA = _import_train_author()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = None
    for k in ("model_cfg", "cfg", "config"):
        if isinstance(ckpt, dict) and isinstance(ckpt.get(k), dict):
            cfg = dict(ckpt[k])
            break
    if cfg is None:
        raise KeyError("ckpt 里找不到 model_cfg；现有 key: "
                       + str(list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)))
    model = TA.AuthorV0Transformer(**_filter_kwargs(TA.AuthorV0Transformer, cfg))
    state = None
    for k in ("model", "state_dict", "model_state_dict", "ckpt"):
        if isinstance(ckpt, dict) and ckpt.get(k) is not None:
            state = ckpt[k]
            break
    if state is None:
        raise KeyError("ckpt 里找不到权重；现有 key: " + str(list(ckpt.keys())))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    default_ctx = None
    for k in ("default_ctx", "ctx_example", "ctx"):
        if isinstance(ckpt, dict) and ckpt.get(k) is not None:
            cand = np.asarray(ckpt[k], np.float32).reshape(-1)
            if cand.size == AUTHOR_CTX_DIM:
                default_ctx = cand
                break
    return model, cfg, default_ctx


def _v1_frame_adapter(v1_model, token_dim: int):
    """AuthorV1Transformer → v0 式四槽逐帧前向适配（惰性建类：模块级不 import torch）。

    v1 的 forward 吃 batch dict 且会连带跑 chunk/priv 头；逐帧主路径只需要主干部
    （encode + head，与 v0 AuthorV0Transformer.forward 逐字同构，语义不变），故适配成
    forward(tokens_idx, state, intent_idx, ctx_feat) -> (b, t, token_dim, vocab)。
    该签名使 _resolve_forward 的关键词映射与 v0 完全一致，AuthorRunner 零改动；
    chunk/priv 头不参与逐帧路径（--use-chunk-head 类流式推理不在本脚本范围）。
    """
    import torch

    class _V1FrameAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = v1_model
            self.token_dim = int(token_dim)

        def forward(self, tokens_idx, state, intent_idx, ctx_feat):
            b, t, _k = tokens_idx.shape
            frames, _ctx_repr = self.model.encode(
                tokens_idx, state, intent_idx, ctx_feat)
            return self.model.head(frames).view(
                b, t, self.token_dim, int(self.model.vocab))

    return _V1FrameAdapter()


def _load_author_v1(ckpt_path: str, device: str, repo: str = DEFAULT_REPO):
    """D062-R1 v1 ckpt 正规加载：training/train_author_v1.load_author_v1 → (model, cfg)。

    返回 (v0 四槽逐帧适配模型, cfg, default_ctx)。default_ctx 恒 None：v1 save_ckpt
    不写 ctx 示例，ctx 仍由本脚本 build_ctx 构造（与 v0 口径一致）。cfg 含
    vocab/code_min/state_dim/win/token_dim（AuthorRunner 所需键全在）与
    variant/d_model 等身份字段，直接透传。
    """
    _ensure_sys_path(repo)
    T1 = _import_train_author_v1()
    model, cfg = T1.load_author_v1(ckpt_path, device=device)
    adapter = _v1_frame_adapter(model, int(cfg.get("token_dim", 64)))
    return adapter, dict(cfg), None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E61/b AXIS-A: author-v0 自回归 token 闭环 (MuJoCo)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--v1-ckpt", default=None,
                    help="D062-R1 v1 ckpt（走 train_author_v1.load_author_v1 正规路径；"
                         "给了则忽略 --ckpt，逐帧滑窗口径与 v0 可比，chunk/priv 头不参与）")
    ap.add_argument("--model-src", default=None,
                    help="模型来源标签（记录进结果 JSON 的 model_src；缺省：给了 "
                         "--v1-ckpt 即 load_author_v1，否则不记该字段）")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--command-vx", default="0.6,1.0", help="命令档位（0.6=SLOW_WALK, 1.0=RUN）")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--dur-s", type=float, default=20.0)
    ap.add_argument("--out-json", default="outputs/d061b_axis_a.json")
    ap.add_argument("--device", default="cpu", help="author 推理设备（服务器建议 cuda）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.dry_run:
        return dry_run(args)

    matrix = build_run_matrix(parse_floats(args.command_vx), parse_ints(args.seeds), args.dur_s)
    t0 = time.time()
    if args.v1_ckpt:
        model, cfg, default_ctx = _load_author_v1(args.v1_ckpt, args.device, args.repo)
    else:
        model, cfg, default_ctx = _load_author(args.ckpt, args.device, args.repo)
    runner = AuthorRunner(model, cfg, device=args.device, default_ctx=default_ctx)
    factory = make_mujoco_factory(args.repo, os.path.join(args.repo, DEC_ONNX_REL))

    per_run = []
    for row in matrix:
        cmd = {k: row[k] for k in ("target_vel", "movement_direction", "mode", "height",
                                   "terrain", "terrain_params")}
        env = factory(row["seed"], cmd)
        r = rollout_one_seed(env, runner, cmd, row["dur_s"], row["seed"])
        per_run.append(r)
        print(f"[run] vx={cmd['target_vel']} mode={cmd['mode']} seed={row['seed']} -> "
              f"survived={r['survived']} fall_step={r['fall_step']} "
              f"realized_vx={r['realized_vx']} h_min={r['h_min']}")

    model_ckpt = args.v1_ckpt if args.v1_ckpt else args.ckpt
    envelope = {
        "script": os.path.abspath(__file__),
        "script_md5": md5_file(os.path.abspath(__file__)),
        "ckpt": os.path.abspath(model_ckpt),
        "ckpt_md5": md5_file(model_ckpt) if os.path.exists(model_ckpt) else None,
        "model_cfg": cfg,
        "device": args.device,
        "repo": args.repo,
        "decoder_onnx": os.path.join(args.repo, DEC_ONNX_REL),
        "ctrl_hz": CTRL_HZ,
        "window": WINDOW,
        "token_scale": TOKEN_SCALE,
        "ctx_dim": AUTHOR_CTX_DIM,
        "ctx_layout": CTX_LAYOUT_SEMANTIC,
        "ctx_source": runner.ctx_source,
        "single_code_broadcast": runner.single_code_broadcast,
        "fall_h": FALL_H,
        "aux_dim": AUX_DIM,
        "run_matrix": matrix,
        "metric_defs": METRIC_DEFS_ZH,
        "assumptions": [
            "R2 off-by-one: env.step 喂的是预测的下一帧 token",
            "ctx 后 9 维（ht4/onehot3/hp2）语义已对齐 train_author_v0.ctx_from_record"
            "（ht4=命令 has_truth / onehot3=terrain / hp2=terrain height,noise）；"
            "terrain 缺省 plane，可用 cmd['terrain']/cmd['terrain_params'] 覆盖",
            "intent 槽与 tokens 槽同源回填（部署期 intent = 生成码，"
            "轴 C autoregressive_rollout 口径）",
            "状态拼接 jp_mj+quat_wxyz+trans_m（d060_windows.py:106）",
            "抑制 env 自动 reset，fall 判据由脚本复刻 qpos[2] < 0.2",
        ],
        "wall_time_s": round(time.time() - t0, 2),
        "per_run": per_run,
        "summary": summarize(per_run),
    }
    if args.v1_ckpt:  # D062-R1 v1 身份信封（不给 --v1-ckpt 时 v0 信封字段零增改）
        envelope["model_src"] = args.model_src or "load_author_v1"
        envelope["v1"] = {
            "ckpt": os.path.abspath(args.v1_ckpt),
            "ckpt_md5": md5_file(args.v1_ckpt),
            "model_src": "load_author_v1",
            "variant": cfg.get("variant"),
            "model_cfg": dict(cfg),
            "frame_path": "AuthorV1Transformer.encode+head（逐帧主干；"
                          "chunk/priv 头不参与）",
        }
    out = args.out_json
    if not os.path.isabs(out):
        out = os.path.join(os.getcwd(), out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(envelope, f, ensure_ascii=False, indent=2)
    print(f"[done] {len(per_run)} runs, json -> {out}")
    print(json.dumps(envelope["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
