"""D060 §5t 统一窗格式脊柱（四源语料共用的格式定义 + builder + 落盘/对账 + 划分）。

预注册 = refine-logs/ds/DS_CONTINUOUS_EXECUTION_PLAN.md §5t（窗格式冻结，本模块是
该格式的**单一实现**；改字段 = 改判据，须报 owner，不得在别处另开一份口径）。

冻结格式（4s 窗 @50Hz = 200 帧；每窗五个字段）
------------------------------------------------
  command        {target_vel 标量, movement_direction, mode, height}
                 源 2/3 无命令真值 → target_vel 用 **speedA 锚定标签**（窗内实测根速
                 中位，见 spec);其余三项填 -1 哨兵（D033 约定）+ has_truth 标记；
                 **禁止均值校准**（D048n/p 教训：speedA 严格单调可学，校准式塌缩）。
  state          (200, 36) float32 = jp_mj 29 + root quat_wxyz 4 + root trans_m 3 每帧。
                 **offline 语料只有运动学**——不是闭环 1762 维 obs（闭环 obs 与扰动
                 状态是源 4 的职责；源 4 用 state_from_kinematics() 落入同一布局）。
                 速度信息靠窗内因果差分；jv（npz 里有 jv_isaac 29 维）是否并入 state
                 留 D061 训练侧决定，暂不进格式。
  terrain_desc   {type, params}：type ∈ {plane, rough_paper, climbing_box}
                 （源 2=climbing_box{height≈0.5}、源 3=plane、源 4=rough_paper）。
  intent_tokens  (200, 64) int16。源 2/3 无配对改写数据 → intent = token_stream 本身
                 （自条件，intent_source="self_corpus"；v2 计划 ②③：改写能力来自地形/
                 命令条件化 + 源 1/4 + 后训，不来自伪配对）。源 1 传 planner 剧本、
                 源 4 传闭环回写时可以显式给 intent_tokens 并记 intent_source。
  token_stream   (200, 64) int16 = FSQ **码**（value = code / token_scale）。
                 源 npz 的 `tokens` 是反量化后的 f32（D060 实测全语料落在 1/16 网格、
                 码域 ≈[-14,12]），落盘前经 token_grid_scale 自动探测 scale 后整数化，
                 scale 记在每窗 meta/分片 meta/manifest（无损可逆，不静默取整）。
                 非网格值（真连续量）→ 直接报错。

落盘与对账
----------
分片 npz（默认每片 128 窗，轴 0 = 窗序）+ `manifest.json` 逐窗对账（D045 惯例）：
每窗记 stem / 帧源 / 命令标签 / terrain / intent 来源 / 帧区间 / token_scale，段级记
actor 与 train|heldout、窗数/覆盖帧/未覆盖帧。stride<win 时窗相互重叠，对账须把
**覆盖并集**与**访问帧**分开：`windows×200 = frames_covered_unique + overlap_redundancy`
且 `frames = frames_covered_unique + frames_uncovered`（manifest.reconciliation.ok）。

state 维数（owner 2026-09-17 裁决，canonical）
---------------------------------------------
state = **36 维** = jp_mj 29 + root quat_wxyz 4 + root trans_m 3（派发文本里写的
"(200,86)" 是笔误：该组成合计 36）。**不留保留位**——50 个零维无信息量且会误导后人，
本方案已撤销（旧 86 维分片作废重切，不做向后读兼容）。

演员防泄漏划分（一次性冻结，源 2/源 3 通用）
--------------------------------------------
`split` 模式读 seed_metadata_v004.parquet 的全量 actor_uid（522 名），按演员
parquet 行数三分位分层抽 15%（≈78 名）为 held-out，seed=0 落盘
`data/ds_bones/actor_split_d060.json`（含 522 名逐名 train/heldout + 抽样方法 + seed，
可复现）。stem 内 `__A\\d+` 提演员（`_M` 镜像段同演员，一并被剔除）。

用法（本仓 = apt_g1/build/，lab-ts 部署根 = ~/ros2_data/apt_g1 平铺顶层；两条
sys.path 都自洽——本模块 numpy-only、无仓内 import，可直接拷到执行根跑）
  python d060_windows.py --selftest
  python d060_windows.py split  --parquet data/ds_bones/seed_metadata_v004.parquet \
      --out data/ds_bones/actor_split_d060.json
  python d060_windows.py pick   --select-json <D059 select_d059.json> \
      --split-json data/ds_bones/actor_split_d060.json --top-per-band 20 \
      --out-list data/ds_bones/g1_d060_climb/d060_list.txt
  python d060_windows.py window --npz-dirs data/ds_bones/g1_b3p/npz \
      data/ds_bones/g1_b4lite/npz ... --terrain plane \
      --split-json data/ds_bones/actor_split_d060.json \
      --out-dir data/ds_bones/d060_windows_src3
  python d060_windows.py window --npz-dirs data/ds_bones/g1_d060_climb/npz \
      --terrain climbing_box --box-height 0.5 --gate-json <gate_d059.json> ...

给源 1 / 源 4（另一条 agent 线）的最小 API：
  from d060_windows import (WIN, STRIDE, FPS, TOKEN_DIM, STATE_DIM,
                            make_command, make_terrain, build_window,
                            windows_from_arrays, state_from_kinematics,
                            write_shards, load_windows)
  w = build_window(token_stream=tok, state=state, command=make_command(0.4),
                   terrain_desc=make_terrain("rough_paper", noise=0.04),
                   intent_tokens=None)          # None = 自条件（源 2/3 口径）
  ws = windows_from_arrays(tok, jp_mj, quat, trans, terrain_desc=...,
                           command=None)        # None = 逐窗 speedA 标签
  recs = write_shards(ws, out_dir, shard_size=128, split_tags=["train"]*len(ws))
  for w in load_windows(".../manifest.json", split="train", limit=8): ...
注意 token 侧：源 npz 的 tokens 是 1/16 网格上的反量化值，本模块自动整数化并记
token_scale（解码 value = code/token_scale）；源 1/4 若已有整数码，直接传即可
（scale=1），若传别的网格值同样自动探测。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np

# ------------------------------------------------------------------ 冻结常量
FPS = 50.0            # 全语料编码帧率（D043/D044：CSV 120Hz → 50Hz 重采样）
WIN = 200             # 4s 窗 @50Hz（§5t 冻结）
STRIDE = 100          # 默认 50% 重叠；CLI 可调（§5t：stride 默认可调）
TOKEN_DIM = 64        # FSQ token 维
N_JOINT = 29          # MuJoCo 序关节
# owner 2026-09-17 裁决：state canonical = 36 维（jp 29 + quat 4 + trans 3），无保留位
STATE_DIM = N_JOINT + 4 + 3
SENTINEL = -1.0       # D033 无真值哨兵
TOKEN_DTYPE = np.int16
STATE_LAYOUT = "jp_mj(29) + root_quat_wxyz(4) + root_trans_m(3)"
TERRAIN_TYPES = ("plane", "rough_paper", "climbing_box")
TERRAIN_DEFAULT_PARAMS = {"plane": {}, "rough_paper": {"noise": 0.04},
                          "climbing_box": {"height": 0.5}}
COMMAND_FIELDS = ("target_vel", "movement_direction", "mode", "height")
INTENT_SOURCE_SELF = "self_corpus"    # intent = token_stream（源 2/3）
INTENT_SOURCE_PLANNER = "planner_script"   # 源 1
INTENT_SOURCE_CLOSED_LOOP = "closed_loop"  # 源 4
FORMAT_VERSION = "d060.1"
ACTOR_RE = re.compile(r"__A(\d+)")
MIRROR_RE = re.compile(r"_M$")  # D044/D045 镜像副本段命名（与母段同 actor_uid）

# 源 npz 键（转换器 D044 产物；源 4 若换键名用 --key-* 覆盖）
KEY_TOKENS = "tokens"
KEY_JP = "jp_mj"
KEY_QUAT = "quat_wxyz"
KEY_TRANS = "trans_m"


# ------------------------------------------------------------------ 小工具
def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def frame_speed_bwd(trans, fps=FPS, axes="xy"):
    """逐帧速度：后向差分 v[t]=||t[t]-t[t-1]||×fps，v[0]=v[1]（D048n 逐字同构）。

    axes="xy" = 水平面（speedA 口径）；"xyz" = 全 3D 范数。
    """
    t = np.asarray(trans, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] < 2:
        raise ValueError(f"trans 需 (n,>=2)，得 {t.shape}")
    t = t if axes == "xyz" else t[:, :2]
    v = np.zeros(len(t), dtype=np.float64)
    if len(t) >= 2:
        v[1:] = np.linalg.norm(np.diff(t, axis=0), axis=1) * float(fps)
        v[0] = v[1]  # 镜像 E39 walk_phase_rate / D048n frame_speed_bwd
    return v


def window_speedA(trans_seg, i0, i1, fps=FPS):
    """speedA 锚定标签 = 窗内实测根速中位（§5t）。

    trans_m XY 后向差分 ×50Hz，差分在**整段**上做（v[0]=v[1] 镜像只作用于段首），
    再取窗帧 [i0, i1) 的中位——不在窗切片上重算差分：窗起点 i0>0 时若对切片重算会
    人为把该窗首帧镜像成第二帧速度。两种口径（整段差分 vs 窗切片重算）的中位**多数窗
    等价、并非全等**——reviewer 实测 60 窗中 4 窗不同、max |Δ| 8.5e-4（差异来自窗首
    镜像值恰好跨过中位数）。本模块固定用整段差分口径并在 manifest speed_label 记明。
    **只用锚定标签，禁止均值校准**（D048n/p：校准式塌缩）。
    """
    v = frame_speed_bwd(trans_seg, fps=fps, axes="xy")
    seg = v[int(i0):int(i1)]
    if seg.size == 0:
        return float("nan")
    return float(np.median(seg))


def make_command(target_vel, movement_direction=SENTINEL, mode=SENTINEL,
                 height=SENTINEL, has_truth=None):
    """command dict（-1 哨兵，D033）。has_truth 缺省按值推断（≠ 哨兵即 True）。"""
    cmd = {"target_vel": float(target_vel),
           "movement_direction": float(movement_direction),
           "mode": float(mode), "height": float(height)}
    if has_truth is None:
        has_truth = {k: (cmd[k] != float(SENTINEL)) for k in COMMAND_FIELDS}
    else:
        has_truth = {k: bool(has_truth[k]) for k in COMMAND_FIELDS}
    for k in COMMAND_FIELDS:
        if not has_truth[k]:
            cmd[k] = float(SENTINEL)  # 无真值一律哨兵，防"缺省 0"被当成真值
    return {"cmd": cmd, "has_truth": has_truth}


def command_vector(command):
    """command dict → float32 (4,) 向量，序 = COMMAND_FIELDS。"""
    cmd = command["cmd"] if "cmd" in command else command
    return np.asarray([float(cmd[k]) for k in COMMAND_FIELDS], dtype=np.float32)


def command_has_truth_vector(command):
    ht = command.get("has_truth") if isinstance(command, dict) else None
    if ht is None:
        ht = {k: (float(command["cmd" if "cmd" in command else command][k])
                  != float(SENTINEL)) for k in COMMAND_FIELDS}
    return np.asarray([bool(ht[k]) for k in COMMAND_FIELDS], dtype=np.bool_)


def make_terrain(terrain_type, **params):
    """terrain_desc dict（特权描述）。params 未给时取该类型缺省（plane={}、
    rough_paper{noise:0.04}、climbing_box{height:0.5}）。"""
    if terrain_type not in TERRAIN_TYPES:
        raise ValueError(f"terrain type 未知: {terrain_type!r}，允许 {TERRAIN_TYPES}")
    p = dict(TERRAIN_DEFAULT_PARAMS[terrain_type])
    p.update({k: (float(v) if isinstance(v, (int, float)) else v)
              for k, v in params.items()})
    return {"type": terrain_type, "params": p}


def terrain_from_args(terrain_type, box_height=None, noise=None):
    kw = {}
    if box_height is not None:
        kw["height"] = box_height
    if noise is not None:
        kw["noise"] = noise
    return make_terrain(terrain_type, **kw)


def slice_starts(n_frames, win=WIN, stride=STRIDE):
    """滑窗起点（只取完整窗，drop-last；窗帘未满不产窗 = 格式契约 200 帧硬约束）。"""
    if win <= 0 or stride <= 0:
        raise ValueError("win/stride 必须为正")
    n = int(n_frames)
    if n < win:
        return []
    return list(range(0, n - win + 1, stride))


def parse_actor(stem):
    """stem 内 `__A\\d+` 提演员（`_M` 镜像段与母段同 uid，正则同覆盖）。"""
    m = ACTOR_RE.search(str(stem))
    return f"A{m.group(1)}" if m else None


def is_mirror_stem(stem):
    """`_M` 后缀 = BONES-SEED 镜像副本段（§5t 预注册：镜像 `_M` 全排除出训练）。

    与 is_heldout 相互独立：held-out 是**演员级**身份排除，本判定是**段级**结构排除
    （train 演员的 `_M` 副本也一律不进训练窗）。window/pick 模式默认丢弃并计数；
    确需保留时 --keep-mirror（manifest 记 segments_mirror_kept）。
    """
    return bool(MIRROR_RE.search(str(stem)))


def covered_unique(starts, win, n_frames):
    """窗区间并集长度（stride<win 时窗相互重叠，需与"覆盖帧数"区分开对账）。"""
    if not starts:
        return 0
    total, cur_s, cur_e = 0, int(starts[0]), int(starts[0]) + int(win)
    for s in list(starts)[1:]:
        s, e = int(s), int(s) + int(win)
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return min(total, int(n_frames))


def token_grid_scale(values, max_scale=1024, tol=1e-6):
    """探测 token 值所在的 2 的幂网格（源 npz 的 tokens 是**反量化后的 FSQ 码**，
    不是整数索引——D060 实测全语料 = 1/16 网格、码域 ≈ [-14, 12]）。

    返回最小的 s ∈ {1,2,4,…,max_scale} 使 values×s 全为整数（tol 内）；无解返回
    None（真连续量 → 调用方须报错，不得静默取整）。
    """
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return 1
    if a.dtype.kind in "iu" or not a.any():
        return 1
    for s in [1 << k for k in range(int(np.log2(max_scale)) + 1)]:
        if np.max(np.abs(a * s - np.rint(a * s))) <= tol:
            return s
    return None


def to_token_int16(tokens, name="token_stream", scale=None):
    """token → int16 **码**（FSQ 码 = 反量化值 × token_scale，无损且可逆）。

    源 npz 存的是反量化后的 f32（1/16 网格）；落盘前按 token_scale 转整数码，
    scale 由 token_grid_scale 自动探测（或调用方显式给）。解码：value = code/scale。
    非网格值（真连续量）→ ValueError，不静默取整。
    """
    a = np.asarray(tokens)
    if a.dtype.kind in "iu":
        if a.size and (np.min(a) < -32768 or np.max(a) > 32767):
            raise ValueError(f"{name} 超 int16 域")
        return (a.astype(TOKEN_DTYPE, copy=False) if a.dtype != TOKEN_DTYPE else a), 1
    f = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(f)):
        raise ValueError(f"{name} 含非有限值")
    s = int(scale) if scale else token_grid_scale(f)
    if s is None:
        raise ValueError(f"{name} 不在任何 2 的幂网格上（FSQ 码应可整数化）："
                         f"max|frac|={np.max(np.abs(f * 16 - np.rint(f * 16))):.3e}")
    codes = np.rint(f * s)
    if codes.size and (codes.min() < -32768 or codes.max() > 32767):
        raise ValueError(f"{name} 超 int16 域（scale={s}）")
    return codes.astype(TOKEN_DTYPE), s


def state_from_kinematics(jp_mj, quat_wxyz, trans_m):
    """运动学三件 → (n, 36) float32 状态窗矩阵（§5t state 布局的唯一装配点）。

    注意：这是 **offline 运动学状态**，不是闭环 1762 维 obs（源 4 的闭环 obs 若要用
    同一 state 槽，应据其记录的运动学重建本布局）。
    owner 2026-09-17 裁决：36 维 = jp_mj 29 + root quat_wxyz 4 + root trans_m 3，
    **无保留位**（派发文本里的 86 是笔误）；jv 是否并入留给 D061 训练侧决定。
    """
    jp = np.asarray(jp_mj, dtype=np.float32)
    q = np.asarray(quat_wxyz, dtype=np.float32)
    tr = np.asarray(trans_m, dtype=np.float32)
    n = len(jp)
    for a, name, dim in ((jp, "jp_mj", N_JOINT), (q, "quat_wxyz", 4), (tr, "trans_m", 3)):
        if a.ndim != 2 or a.shape[1] != dim or len(a) != n:
            raise ValueError(f"{name} 形状需 ({n},{dim})，得 {a.shape}")
    st = np.concatenate([jp, q, tr], axis=1)
    if st.shape[1] != STATE_DIM:
        raise ValueError(f"state 维需 {STATE_DIM}，得 {st.shape[1]}")
    if not np.all(np.isfinite(st)):
        raise ValueError("state 含非有限值")
    return st


# ------------------------------------------------------------------ 窗构造
def build_window(token_stream, state, command, terrain_desc, intent_tokens=None,
                 meta=None, token_scale=None):
    """§5t 冻结窗的**纯函数**构造器（源 1–4 共用）。

    入参：token_stream/state 已按窗切好（(200,64)/(200,36)）；command = make_command()
    产物；terrain_desc = make_terrain() 产物；intent_tokens=None → 自条件
    （intent = token_stream，intent_source="self_corpus"）；token_scale=None →
    自动探测 token 网格（源 2/3 npz 实测 16；整数码输入则 1）。
    出参：{"token_stream","intent_tokens","state","command","terrain_desc","meta"}，
    meta 记 token_scale（解码 value = code/scale）/state_layout/intent_source/speedA。
    """
    tok, detected = to_token_int16(token_stream, "token_stream", scale=token_scale)
    scale = int(token_scale) if token_scale else detected
    st = np.asarray(state, dtype=np.float32)
    if tok.ndim != 2 or tok.shape[1] != TOKEN_DIM:
        raise ValueError(f"token_stream 形状需 (w,{TOKEN_DIM})，得 {tok.shape}")
    if st.ndim != 2 or st.shape[1] != STATE_DIM:
        raise ValueError(f"state 形状需 (w,{STATE_DIM})，得 {st.shape}")
    if len(st) != len(tok):
        raise ValueError(f"state/token_stream 帧数不一致：{len(st)} vs {len(tok)}")
    if not np.all(np.isfinite(st)):
        raise ValueError("state 含非有限值")
    if "cmd" not in command:
        raise ValueError("command 需为 make_command() 产物（含 cmd/has_truth）")
    if terrain_desc.get("type") not in TERRAIN_TYPES:
        raise ValueError(f"terrain type 非法：{terrain_desc.get('type')!r}")
    if intent_tokens is None:
        intent = tok.copy()
        intent_source = INTENT_SOURCE_SELF
    else:
        intent, _ = to_token_int16(intent_tokens, "intent_tokens", scale=scale)
        if intent.shape != tok.shape:
            raise ValueError(f"intent_tokens 形状需同 token_stream {tok.shape}，"
                             f"得 {intent.shape}")
        intent_source = (meta or {}).get("intent_source", "explicit")
    m = {"format": FORMAT_VERSION, "win": int(tok.shape[0]),
         "token_dim": TOKEN_DIM, "state_dim": int(st.shape[1]),
         "token_scale": scale,
         "token_decode_rule": "value = code / token_scale（FSQ 反量化值）",
         "state_layout": STATE_LAYOUT,
         "intent_source": intent_source}
    if meta:
        m.update(meta)
    return {"token_stream": tok, "intent_tokens": intent, "state": st,
            "command": {"cmd": dict(command["cmd"]),
                        "has_truth": dict(command["has_truth"])},
            "terrain_desc": {"type": terrain_desc["type"],
                             "params": dict(terrain_desc.get("params", {}))},
            "meta": m}


def window_contract_violations(w, win=WIN, state_dim=None):
    """字段契约断言（消费者测试/落盘前共用），返回违规列表（空 = 合规）。

    state_dim=None → 取 canonical STATE_DIM（36）。"""
    bad = []
    for k in ("token_stream", "intent_tokens", "state", "command", "terrain_desc",
              "meta"):
        if k not in w:
            bad.append(f"缺字段 {k}")
    if bad:
        return bad
    tok, it, st = w["token_stream"], w["intent_tokens"], w["state"]
    if tok.shape != (win, TOKEN_DIM) or tok.dtype != TOKEN_DTYPE:
        bad.append(f"token_stream 契约违例 {tok.shape}/{tok.dtype}")
    if it.shape != (win, TOKEN_DIM) or it.dtype != TOKEN_DTYPE:
        bad.append(f"intent_tokens 契约违例 {it.shape}/{it.dtype}")
    want = STATE_DIM if state_dim is None else int(state_dim)
    if st.shape != (win, want) or st.dtype != np.float32:
        bad.append(f"state 契约违例 {st.shape}/{st.dtype}（期望 (200,{want})）")
    if w["meta"].get("intent_source") == INTENT_SOURCE_SELF and not np.array_equal(tok, it):
        bad.append("intent_source=self_corpus 但 intent != token_stream（自条件违约）")
    if not np.all(np.isfinite(st)):
        bad.append("state 含非有限值")
    if w["terrain_desc"]["type"] not in TERRAIN_TYPES:
        bad.append(f"terrain type 非法 {w['terrain_desc']['type']!r}")
    return bad


def windows_from_arrays(tokens, jp_mj, quat_wxyz, trans_m, *, terrain_desc,
                        command=None, intent_tokens=None, win=WIN, stride=STRIDE,
                        fps=FPS, stem="", intent_source=None, meta_extra=None):
    """整段数组 → 窗 list（源 1/2/3/4 的统一切窗入口）。

    command=None → 逐窗 speedA 锚定标签（源 2/3）；给固定 command（源 1 planner
    名义命令）则所有窗共用。intent_tokens=None → 自条件。
    """
    tok, tok_scale = to_token_int16(tokens, "tokens")
    n = len(tok)
    for a, name, dim in ((jp_mj, "jp_mj", N_JOINT), (quat_wxyz, "quat_wxyz", 4),
                         (trans_m, "trans_m", 3)):
        if len(a) != n:
            raise ValueError(f"{name} 长度 {len(a)} != tokens {n}")
    state = state_from_kinematics(jp_mj, quat_wxyz, trans_m)
    starts = slice_starts(n, win=win, stride=stride)
    out = []
    for k, i0 in enumerate(starts):
        i1 = i0 + win
        if command is None:
            cmd = make_command(window_speedA(trans_m, i0, i1, fps=fps))
        else:
            cmd = command
        m = {"stem": stem, "window_index": k, "frame_start": i0, "frame_end": i1,
             "n_segment_frames": n, "fps": float(fps), "win": int(win),
             "stride": int(stride)}
        if intent_source:
            m["intent_source"] = intent_source
        if meta_extra:
            m.update(meta_extra)
        w = build_window(tok[i0:i1], state[i0:i1], cmd, terrain_desc,
                         None if intent_tokens is None else intent_tokens[i0:i1],
                         meta=m, token_scale=tok_scale)
        if command is None:
            w["meta"]["speedA"] = cmd["cmd"]["target_vel"]
        out.append(w)
    return out


# ------------------------------------------------------------------ 划分表
def load_actor_split(path):
    """读 actor_split json → {heldout:set, train:set, doc:dict}（缺失即 fast-fail）。"""
    if path in (None, "", "none", "None"):
        return {"heldout": set(), "train": set(), "doc": None}
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    held = set(doc.get("heldout") or [])
    train = set(doc.get("train") or [])
    if not held:
        actors = doc.get("actors") or {}
        held = {a for a, v in actors.items()
                if (v.get("split") if isinstance(v, dict) else v) == "heldout"}
        train = {a for a, v in actors.items()
                 if (v.get("split") if isinstance(v, dict) else v) == "train"}
    return {"heldout": held, "train": train, "doc": doc, "path": os.path.abspath(path)}


def is_heldout(stem, split):
    """按 stem 的演员判是否 held-out（无 `__A\\d+` 的 stem 无法归属 → 非 held-out）。"""
    a = parse_actor(stem)
    return bool(a) and a in split["heldout"]


def stratified_holdout(actor_rows, seed=0, frac=0.15, n_strata=3):
    """演员行数三分位分层抽 held-out（精确到 round(frac×N) 名，seed 固定可复现）。

    actor_rows: {actor: 行数}. 返回 (heldout_sorted, train_sorted, strata_info,
    actor_stratum：{actor: 层号})。抽样在每一层内对【字典序排序后的名单】用
    np.random.default_rng(seed+层号) 置换取前 k_i——不依赖 set 迭代序/哈希随机化，
    跨机可复现。
    """
    actors = sorted(actor_rows)
    n = len(actors)
    if n == 0:
        return [], [], {}, {}
    rows = np.asarray([actor_rows[a] for a in actors], dtype=np.int64)
    # 分层变量 = 演员 parquet 行数（活动量代理）；等频分位（tie 归下层，确定性）
    qs = [np.percentile(rows, 100.0 * i / n_strata) for i in range(1, n_strata)]
    stratum = np.zeros(n, dtype=np.int64)
    for i, q in enumerate(qs):
        stratum += (rows > q).astype(np.int64)
    target = int(round(frac * n))
    per = [int(np.floor(frac * (stratum == s).sum())) for s in range(n_strata)]
    rem = target - sum(per)
    if rem > 0:  # 余数按小数部分大者补（确定性：并列按层号）
        frac_parts = sorted(range(n_strata),
                            key=lambda s: (-(frac * (stratum == s).sum() - per[s]), s))
        for s in frac_parts[:rem]:
            if per[s] < (stratum == s).sum():
                per[s] += 1
    held, train, info = [], [], {}
    actor_stratum = {}
    for s in range(n_strata):
        idx = [i for i in range(n) if stratum[i] == s]
        for i in idx:
            actor_stratum[actors[i]] = s
        rng = np.random.default_rng(seed + s)  # 逐层独立流，seed 主键=seed
        perm = rng.permutation(len(idx))
        k = per[s]
        pick = {idx[p] for p in perm[:k]}
        info[f"stratum{s}"] = {
            "rows_range": [int(rows[idx].min()), int(rows[idx].max())],
            "n_actors": len(idx), "n_heldout": k,
            "heldout": sorted(actors[i] for i in idx if i in pick)}
        for i in idx:
            (held if i in pick else train).append(actors[i])
    return sorted(held), sorted(train), info, actor_stratum


# ------------------------------------------------------------------ 段枚举 / 落盘
def collect_segments(npz_dirs, limit=None, dedupe_by_stem=True, verbose=True):
    """npz 目录集 → 段 list（dedupe：同 stem 多目录出现时保留首个 = canonical 目录
    优先；D060 recon 实测 b4lite 各 gate 子目录与 npz/ 的同名段 tokens 逐字节相同）。"""
    segs, seen = [], {}
    for d in npz_dirs:
        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        if not files:
            print(f"[collect] 警告：{d} 下无 npz")
        for p in files:
            stem = os.path.basename(p)[:-4]
            if dedupe_by_stem and stem in seen:
                seen[stem]["duplicate_dirs"].append(d)
                continue
            rec = {"stem": stem, "npz_path": os.path.abspath(p), "npz_dir": d,
                   "duplicate_dirs": []}
            segs.append(rec)
            seen[stem] = rec
    segs.sort(key=lambda r: (r["npz_dir"], r["stem"]))
    if limit:
        segs = segs[:int(limit)]
    if verbose:
        n_dup = sum(len(r["duplicate_dirs"]) for r in segs)
        print(f"[collect] 段 {len(segs)}（去重丢弃同 stem 副本 {n_dup}）")
    return segs


def read_segment_arrays(npz_path, key_tokens=KEY_TOKENS, key_jp=KEY_JP,
                        key_quat=KEY_QUAT, key_trans=KEY_TRANS):
    """读段 npz 的四件（tokens/jp_mj/quat_wxyz/trans_m）+ meta 字符串（可缺）。"""
    with np.load(npz_path, allow_pickle=False) as z:
        miss = [k for k in (key_tokens, key_jp, key_quat, key_trans) if k not in z]
        if miss:
            raise KeyError(f"{os.path.basename(npz_path)} 缺键 {miss}（有 "
                           f"{sorted(z.files)}）")
        out = {"tokens": z[key_tokens], "jp_mj": z[key_jp],
               "quat_wxyz": z[key_quat], "trans_m": z[key_trans]}
        out["meta"] = str(z["meta"]) if "meta" in z.files else None
    return out


def write_shards(windows, out_dir, shard_size=128, prefix="w", split_tags=None,
                 extra_manifest=None):
    """窗 list → 分片 npz（轴 0 = 窗序）+ 逐窗记录（供 manifest）。

    返回 (shard_records, window_records)。窗 npz 键见模块 docstring；intent/token
    存 int16**码**（源值经 token_scale 整数化，可逆；scale 随窗记），state 存 float32。
    """
    if not windows:
        return [], []
    os.makedirs(out_dir, exist_ok=True)
    w0 = windows[0]["state"].shape[0]
    for w in windows:
        v = window_contract_violations(w, win=w0)
        if v:
            raise ValueError(f"窗契约违例：{v}")
    tags = list(split_tags) if split_tags is not None else ["train"] * len(windows)
    if len(tags) != len(windows):
        raise ValueError("split_tags 长度须等于窗数")
    shards, wrecs = [], []
    for s0 in range(0, len(windows), int(shard_size)):
        chunk = windows[s0:s0 + int(shard_size)]
        tagset = sorted(set(tags[s0:s0 + int(shard_size)]))
        name = f"{prefix}_{'+'.join(tagset)}_{s0:06d}-{s0 + len(chunk) - 1:06d}.npz"
        path = os.path.join(out_dir, name)
        np.savez_compressed(
            path,
            token_stream=np.stack([w["token_stream"] for w in chunk]).astype(TOKEN_DTYPE),
            intent_tokens=np.stack([w["intent_tokens"] for w in chunk]).astype(TOKEN_DTYPE),
            state=np.stack([w["state"] for w in chunk]).astype(np.float32),
            command=np.stack([command_vector(w["command"]) for w in chunk]),
            command_has_truth=np.stack([command_has_truth_vector(w["command"])
                                        for w in chunk]),
            terrain_type=np.asarray([w["terrain_desc"]["type"] for w in chunk]),
            terrain_params_json=np.asarray(
                [json.dumps(w["terrain_desc"].get("params", {}), ensure_ascii=False,
                            sort_keys=True) for w in chunk]),
            meta_json=np.asarray([json.dumps(w["meta"], ensure_ascii=False,
                                             sort_keys=True) for w in chunk]),
            shard_meta_json=np.asarray(json.dumps(
                {"format": FORMAT_VERSION, "n_windows": len(chunk), "win": w0,
                 "token_dim": TOKEN_DIM, "state_dim": int(chunk[0]["state"].shape[1]),
                 "token_scale": chunk[0]["meta"].get("token_scale"),
                 "token_decode_rule": "value = code / token_scale",
                 "state_layout": chunk[0]["meta"].get("state_layout"),
                 "index_offset": s0,
                 **(extra_manifest or {})}, ensure_ascii=False)))
        shards.append({"path": os.path.abspath(path), "name": name, "n": len(chunk),
                       "first_index": s0, "last_index": s0 + len(chunk) - 1,
                       "md5": md5_file(path), "bytes": os.path.getsize(path),
                       "splits": tagset})
        for j, w in enumerate(chunk):
            m = w["meta"]
            cmd = w["command"]
            wrecs.append({
                "stem": m.get("stem", ""), "shard": name, "index": j,
                "global_index": s0 + j, "i0": m.get("frame_start"),
                "i1": m.get("frame_end"), "split": tags[s0 + j],
                "speedA": m.get("speedA"),
                "target_vel": cmd["cmd"]["target_vel"],
                "movement_direction": cmd["cmd"]["movement_direction"],
                "mode": cmd["cmd"]["mode"], "height": cmd["cmd"]["height"],
                "has_truth": [k for k in COMMAND_FIELDS if cmd["has_truth"][k]],
                "terrain": w["terrain_desc"]["type"],
                "terrain_params": w["terrain_desc"].get("params", {}),
                "intent_source": m.get("intent_source"),
                "token_scale": m.get("token_scale"),
            })
    return shards, wrecs


def load_windows(manifest_path, split=None, limit=None, shard_dir=None):
    """loader（numpy-only）：读 manifest.json（或 manifest dict）→ 逐窗 dict。

    产出 dict：token_stream/intent_tokens (200,64) int16、state (200,36) f32、
    command {cmd,has_truth}、terrain_desc {type,params}、meta(dict，含 stem/i0/i1)、
    + 段级 stem/split/speedA。split=None → 不过滤；"train"/"heldout" → 按窗过滤。
    分片解析：先 <manifest 目录>/npz/<name>（可搬移）→ 再 manifest 里记的绝对路径
    （原地）；两处都没有即 fast-fail（不静默跳过）。
    """
    doc = manifest_path
    if isinstance(manifest_path, str):
        p = manifest_path
        if os.path.isdir(p):
            p = os.path.join(p, "manifest.json")
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        base = os.path.dirname(os.path.abspath(p))
    else:
        base = os.getcwd()
    if shard_dir is None:
        shard_dir = os.path.join(base, "npz")
    abs_by_name = {s.get("name"): s.get("path") for s in doc.get("shards", [])}
    handles, n_out = {}, 0
    try:
        for rec in doc.get("windows", []):
            if split and rec.get("split") != split:
                continue
            name = rec["shard"]
            if name not in handles:
                cand = [os.path.join(shard_dir, name), abs_by_name.get(name)]
                real = next((c for c in cand if c and os.path.isfile(c)), None)
                if real is None:
                    raise FileNotFoundError(f"分片缺失：{name}（找过 {cand}）")
                handles[name] = np.load(real, allow_pickle=False)
            z = handles[name]
            i = int(rec["index"])
            cmd = {k: float(rec.get(k, SENTINEL)) for k in COMMAND_FIELDS}
            yield {
                "token_stream": z["token_stream"][i],
                "intent_tokens": z["intent_tokens"][i],
                "state": z["state"][i],
                "command": {"cmd": cmd,
                            "has_truth": {k: (k in (rec.get("has_truth") or []))
                                          for k in COMMAND_FIELDS}},
                "terrain_desc": {"type": str(z["terrain_type"][i]),
                                 "params": json.loads(str(z["terrain_params_json"][i]))},
                "meta": json.loads(str(z["meta_json"][i])),
                "stem": rec.get("stem"), "split": rec.get("split"),
                "speedA": rec.get("speedA"),
            }
            n_out += 1
            if limit and n_out >= int(limit):
                return
    finally:
        for z in handles.values():
            try:
                z.close()
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------------------ 模式：window
def build_segment_windows(seg, terrain_desc, split, *, win=WIN, stride=STRIDE,
                          fps=FPS, strict_actor=True, **kw):
    """单段 → (windows, seg_record, err)。段记录含 actor/train|heldout/窗数/帧数。"""
    rec = dict(seg)
    rec.update({"actor": parse_actor(seg["stem"]), "terrain": terrain_desc["type"]})
    if split is not None:
        rec["split"] = "heldout" if is_heldout(seg["stem"], split) else "train"
    else:
        rec["split"] = "train"
    if strict_actor and rec["actor"] is None:
        rec["actor_unparsed"] = True
    try:
        a = read_segment_arrays(seg["npz_path"], **kw)
        n = len(a["tokens"])
        rec["frames"] = int(n)
        rec["npz_md5"] = md5_file(seg["npz_path"])
        v = frame_speed_bwd(a["trans_m"], fps=fps, axes="xy")
        rec["speed_frames"] = {"median": float(np.median(v)),
                               "q1": float(np.percentile(v, 25)),
                               "q3": float(np.percentile(v, 75))} if n else None
        # 四元数单位性观察项（非门）
        qn = np.linalg.norm(np.asarray(a["quat_wxyz"], dtype=np.float64), axis=1)
        rec["quat_norm_mean"] = float(np.mean(qn)) if n else float("nan")
        ws = windows_from_arrays(a["tokens"], a["jp_mj"], a["quat_wxyz"], a["trans_m"],
                                 terrain_desc=terrain_desc, win=win, stride=stride,
                                 fps=fps, stem=seg["stem"],
                                 meta_extra={"npz_path": seg["npz_path"],
                                             "npz_md5": rec["npz_md5"]})
        rec["windows"] = len(ws)
        rec["frames_covered_unique"] = covered_unique(
            [w["meta"]["frame_start"] for w in ws], win, n)
        rec["frames_uncovered"] = int(n) - rec["frames_covered_unique"]
        rec["window_frames_visits"] = len(ws) * int(win)
        rec["overlap_redundancy"] = rec["window_frames_visits"] - \
            rec["frames_covered_unique"]
        if ws:
            rec["frame_start_first"] = ws[0]["meta"]["frame_start"]
            rec["frame_end_last"] = ws[-1]["meta"]["frame_end"]
            sp = np.asarray([w["meta"]["speedA"] for w in ws])
            rec["speedA_windows"] = {"min": float(sp.min()), "median": float(np.median(sp)),
                                     "max": float(sp.max())}
        return ws, rec, None
    except Exception as exc:  # noqa: BLE001 —— 失败段如实记 error，不猜
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["frames"] = rec.get("frames")
        return [], rec, rec["error"]


def gate_status_map(gate_json, verbose=True):
    """D059 gate JSON → {stem: "PASS"|"FAIL"|"ERROR"|"UNKNOWN"}。

    逐段判据认键顺序（D059 envelope 产物的真实键 = hard_ok，其次给兼容别名）：
      error → ERROR；hard_ok / gate_pass / pass / ok → PASS|FAIL；一个都没有 → UNKNOWN
    （UNKNOWN 不静默当 PASS：window 模式计数并默认保留，manifest 记
    counts.segments_gate_unknown，便于判读门产物版本漂移）。
    """
    if gate_json in (None, "", "none", "None"):
        return {}
    with open(gate_json, encoding="utf-8") as f:
        d = json.load(f)
    out, unknown = {}, 0
    for r in (d.get("rows") or d.get("segments") or []):
        stem = r.get("stem")
        if stem is None:
            continue
        if "error" in r:
            out[stem] = "ERROR"
            continue
        for k in ("hard_ok", "gate_pass", "pass", "ok"):
            if k in r:
                out[stem] = "PASS" if bool(r[k]) else "FAIL"
                break
        else:
            out[stem] = "UNKNOWN"
            unknown += 1
    for e in (d.get("errors") or []):
        if isinstance(e, dict) and e.get("stem"):
            out[e["stem"]] = "ERROR"
    if verbose:
        n_pass = sum(1 for v in out.values() if v == "PASS")
        n_fail = sum(1 for v in out.values() if v == "FAIL")
        print(f"[gate] 读 {os.path.basename(gate_json)}：{len(out)} 段 "
              f"(PASS {n_pass} / FAIL {n_fail} / ERROR "
              f"{sum(1 for v in out.values() if v == 'ERROR')} / UNKNOWN {unknown})"
              f" mode={d.get('gate_mode')}")
    return out


def run_window_mode(args):
    split = load_actor_split(args.split_json)
    terrain = terrain_from_args(args.terrain, box_height=args.box_height,
                               noise=args.noise)
    gates = gate_status_map(args.gate_json)
    t0 = time.time()
    segs = collect_segments(args.npz_dirs, limit=args.limit)
    all_w, srecs = [], []
    counts = {"segments_found": len(segs), "segments_used": 0,
              "segments_skipped_heldout": 0, "segments_skipped_mirror": 0,
              "segments_mirror_kept": 0, "segments_skipped_gate": 0,
              "segments_gate_unknown": 0,
              "segments_skipped_actor_unparsed": 0, "segments_skipped_short": 0,
              "segments_error": 0}
    skipped = {"heldout": [], "mirror": [], "gate": [], "actor_unparsed": [],
               "short": [], "error": []}
    split_tags = []
    for s in segs:
        stem = s["stem"]
        actor = parse_actor(stem)
        g = gates.get(stem)
        if not args.include_heldout and split["heldout"] and is_heldout(stem, split):
            counts["segments_skipped_heldout"] += 1
            skipped["heldout"].append({"stem": stem, "actor": actor,
                                       "npz_path": s["npz_path"],
                                       "note": "G5 评测专用，不进训练窗"})
            continue
        if is_mirror_stem(stem) and not args.keep_mirror:
            # §5t：镜像 _M 全排除出训练（段级结构排除，与上面演员级 held-out 互斥计数）
            counts["segments_skipped_mirror"] += 1
            skipped["mirror"].append({"stem": stem, "actor": actor,
                                      "npz_path": s["npz_path"],
                                      "note": "镜像副本（_M），§5t 预注册排除出训练"})
            continue
        if is_mirror_stem(stem) and args.keep_mirror:
            counts["segments_mirror_kept"] += 1
        if g in ("FAIL", "ERROR"):
            counts["segments_skipped_gate"] += 1
            skipped["gate"].append({"stem": stem, "gate": g,
                                    "npz_path": s["npz_path"]})
            continue
        if g == "UNKNOWN":
            counts["segments_gate_unknown"] += 1
        if actor is None and args.drop_unparsed_actor:
            counts["segments_skipped_actor_unparsed"] += 1
            skipped["actor_unparsed"].append({"stem": stem})
            continue
        ws, rec, err = build_segment_windows(
            s, terrain, split, win=args.win, stride=args.stride, fps=args.fps,
            key_tokens=args.key_tokens, key_jp=args.key_jp,
            key_quat=args.key_quat, key_trans=args.key_trans)
        rec["gate_status"] = g or "n/a"
        if err:
            counts["segments_error"] += 1
            skipped["error"].append({"stem": stem, "npz_path": s["npz_path"],
                                     "error": err})
            srecs.append(rec)
            continue
        if not ws:
            counts["segments_skipped_short"] += 1
            skipped["short"].append({"stem": stem, "frames": rec.get("frames"),
                                     "win": args.win})
            srecs.append(rec)
            continue
        counts["segments_used"] += 1
        tag = rec["split"]
        all_w.extend(ws)
        split_tags.extend([tag] * len(ws))
        srecs.append(rec)

    shards, wrecs = write_shards(all_w, args.npz_out_dir, shard_size=args.shard_size,
                                 prefix=args.shard_prefix, split_tags=split_tags,
                                 extra_manifest={"terrain": terrain,
                                                 "stride": int(args.stride),
                                                 "fps": float(args.fps)})

    tok_min = int(min(int(t.min()) for t in (w["token_stream"] for w in all_w))) \
        if all_w else None
    tok_max = int(max(int(t.max()) for t in (w["token_stream"] for w in all_w))) \
        if all_w else None
    sp = np.asarray([w["meta"]["speedA"] for w in all_w
                     if w["meta"].get("speedA") is not None], dtype=np.float64)

    def _frames_of(recs):
        return int(sum(int(r["frames"]) for r in recs
                       if isinstance(r.get("frames"), int)))

    win_recs = [r for r in srecs if r.get("windows")]
    short_recs = [r for r in srecs if not r.get("windows") and "error" not in r
                  and isinstance(r.get("frames"), int)]
    err_recs = [r for r in srecs if "error" in r and isinstance(r.get("frames"), int)]
    # 帧数三拆（windowed=进窗段的段帧；too_short=零窗过短段帧；error=读失败段帧）
    frames_windowed = _frames_of(win_recs)
    frames_too_short = _frames_of(short_recs)
    frames_error = _frames_of(err_recs)
    covered = int(sum(int(r.get("frames_covered_unique") or 0) for r in srecs))
    visits = int(sum(int(r.get("window_frames_visits") or 0) for r in srecs))
    doc = {
        "exp": "D060", "gate": "G1 格式冒烟（统一窗格式脊柱）",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": os.path.abspath(__file__),
        "script_md5": md5_file(os.path.abspath(__file__)),
        "format": {"format_version": FORMAT_VERSION, "fps": float(args.fps),
                   "win": int(args.win), "stride": int(args.stride),
                   "token_dim": TOKEN_DIM,
                   "state_dim": STATE_DIM,
                   "state_layout": STATE_LAYOUT,
                   "state_dim_note": ("owner 2026-09-17 裁决：state canonical = 36 维"
                                      "（jp 29 + quat 4 + trans 3），无保留位——派发文本"
                                      "里的 86 是笔误，50 零维保留位方案撤销；jv 是否并入"
                                      "留给 D061 训练侧决定，暂不进格式"),
                   "token_dtype": "int16", "state_dtype": "float32",
                   "token_scale": (all_w[0]["meta"].get("token_scale") if all_w
                                   else None),
                   "token_decode_rule": ("token_stream 存 int16 码；反量化值 = "
                                         "code / token_scale（源 npz 存的是 1/16 "
                                         "网格上的反量化 FSQ 值，非整数索引）"),
                   "speed_label": ("speedA 锚定标签 = 窗内实测根速中位"
                                   "（trans_m XY 后向差分 ×fps；差分在整段上做、"
                                   "中位取窗帧；禁止均值校准）"),
                   "sentinel": SENTINEL,
                   "command_fields": list(COMMAND_FIELDS),
                   "intent_policy": ("源 2/3 无配对改写数据 → intent=token_stream"
                                     "（自条件，intent_source=self_corpus）")},
        "inputs": {"npz_dirs": [os.path.abspath(d) for d in args.npz_dirs],
                   "split_json": split.get("path"),
                   "split_seed": (split["doc"] or {}).get("seed"),
                   "split_heldout_actors": len(split["heldout"]),
                   "gate_json": (os.path.abspath(args.gate_json)
                                 if gates else None),
                   "terrain": terrain, "include_heldout": bool(args.include_heldout),
                   "keep_mirror": bool(args.keep_mirror),
                   "mirror_rule": ("镜像 `_M` 段默认排除（§5t 预注册；"
                                   f"段级 `_M$` 判定 is_mirror_stem，本次剔 "
                                   f"{counts['segments_skipped_mirror']} 段），"
                                   "与 held-out（演员级）互斥计数"),
                   "key_map": {"tokens": args.key_tokens, "jp_mj": args.key_jp,
                               "quat_wxyz": args.key_quat, "trans_m": args.key_trans}},
        "counts": dict(counts, **{
            "windows": len(all_w),
            "window_frames": int(len(all_w) * int(args.win)),
            "windows_train": int(sum(1 for t in split_tags if t == "train")),
            "windows_heldout": int(sum(1 for t in split_tags if t == "heldout")),
            "shards": len(shards),
            "shard_bytes": int(sum(s["bytes"] for s in shards)),
            "frames_windowed": frames_windowed,
            "frames_too_short": frames_too_short,
            "frames_error": frames_error,
            "frames_covered_unique": covered,
            "frames_uncovered": frames_windowed - covered,
            "window_frame_visits": visits,
            "overlap_redundancy": visits - covered,
            "token_min": tok_min, "token_max": tok_max,
            "token_scale": (all_w[0]["meta"].get("token_scale") if all_w else None),
            "elapsed_s": round(time.time() - t0, 2),
        }),
        "speedA_dist": ({"n": int(sp.size), "min": float(sp.min()),
                         "q1": float(np.percentile(sp, 25)),
                         "median": float(np.median(sp)),
                         "q3": float(np.percentile(sp, 75)),
                         "max": float(sp.max()),
                         "tertile_edges": [float(np.percentile(sp, 100 / 3)),
                                           float(np.percentile(sp, 200 / 3))]}
                        if sp.size else None),
        "heldout_segments_excluded": skipped["heldout"],
        "skipped": {k: v for k, v in skipped.items() if v},
        "segments": srecs,
        "shards": shards,
        "windows": wrecs,
        "reconciliation": {
            "rule": "段级：windows×win = frames_covered_unique + overlap_redundancy，"
                    "frames = frames_covered_unique + frames_uncovered（stride<win 时"
                    "窗重叠，覆盖帧与访问帧必须分开对账）；全局：manifest 窗数 = "
                    "分片窗数之和",
            "windows_from_shards": int(sum(s["n"] for s in shards)),
            "windows_in_manifest": len(wrecs),
            "frames_windowed": frames_windowed, "frames_covered_unique": covered,
            "frames_uncovered": frames_windowed - covered,
            "frames_too_short": frames_too_short, "frames_error": frames_error,
            "overlap_redundancy": visits - covered,
            "ok": bool(sum(s["n"] for s in shards) == len(wrecs)
                       and all((r.get("frames") is None)
                               or ((r.get("window_frames_visits") or 0)
                                   == (r.get("frames_covered_unique") or 0)
                                   + (r.get("overlap_redundancy") or 0))
                               for r in srecs)),
        },
    }
    out_manifest = args.manifest_out or os.path.join(args.out_dir, "manifest.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_manifest)), exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print(f"[window] 段 用 {counts['segments_used']}/{counts['segments_found']}"
          f"（heldout 剔 {counts['segments_skipped_heldout']}、镜像 剔 "
          f"{counts['segments_skipped_mirror']}、gate 剔 "
          f"{counts['segments_skipped_gate']}、过短 {counts['segments_skipped_short']}、"
          f"错 {counts['segments_error']}）")
    print(f"[window] 窗 {len(all_w)}（train {doc['counts']['windows_train']}"
          f"/heldout {doc['counts']['windows_heldout']}）帧 {doc['counts']['window_frames']}"
          f" 分片 {len(shards)} terrain={terrain['type']} "
          f"耗时 {doc['counts']['elapsed_s']}s")
    if sp.size:
        e = doc["speedA_dist"]["tertile_edges"]
        print(f"[window] speedA 分位 min {sp.min():.3f} / q1 "
              f"{doc['speedA_dist']['q1']:.3f} / med {doc['speedA_dist']['median']:.3f}"
              f" / q3 {doc['speedA_dist']['q3']:.3f} / max {sp.max():.3f} | "
              f"三分位界 {e[0]:.3f}, {e[1]:.3f} m/s")
    print(f"[window] -> {out_manifest}")
    return doc


# ------------------------------------------------------------------ 模式：split
def run_split_mode(args):
    """演员防泄漏划分（一次性冻结）：pyarrow 读 parquet（延迟 import，本模块核心
    保持 numpy-only）。"""
    import pyarrow.parquet as pq  # noqa: PLC0415 —— 仅本模式需要
    t = pq.read_table(args.parquet, columns=["actor_uid", "is_mirror"])
    actors = t.column("actor_uid").to_pylist()
    rows = {}
    for a in actors:
        rows[a] = rows.get(a, 0) + 1
    held, train, info, actor_stratum = stratified_holdout(
        rows, seed=args.seed, frac=args.frac, n_strata=args.strata)
    doc = {
        "exp": "D060", "purpose": ("§5t 防泄漏划分（演员级，一次性冻结）：held-out "
                                   "演员的全部段只进 G5 评测，不进任何训练窗；"
                                   "源 2 选料与源 3 过滤共用本表"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_parquet": os.path.abspath(args.parquet),
        "source_parquet_md5": md5_file(args.parquet),
        "seed": int(args.seed), "frac": float(args.frac), "n_strata": int(args.strata),
        "method": (f"按演员 parquet 行数等频 {args.strata} 分层，层内对字典序名单用 "
                   f"numpy.random.default_rng(seed+层号) 置换取前 k_i（目标 "
                   f"round({args.frac}×522)={int(round(args.frac * len(rows)))} 名，"
                   "余数按小数部分大者补）；分层变量=演员活动量（行数），"
                   "不依赖 set 迭代序 → 跨机可复现"),
        "actor_stem_rule": "stem 内 `__A\\d+`（`_M` 镜像段与母段同 uid，同样剔除）",
        "counts": {"n_actors": len(rows), "n_heldout": len(held),
                   "n_train": len(train),
                   "frac_realized": round(len(held) / max(len(rows), 1), 4),
                   "n_rows": int(sum(rows.values()))},
        "strata": info,
        "heldout": held, "train": train,
        "actors": {a: {"split": ("heldout" if a in set(held) else "train"),
                       "rows": int(rows[a]),
                       "stratum": int(actor_stratum[a])}
                   for a in sorted(rows)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print(f"[split] 演员 {len(rows)} → heldout {len(held)}（{doc['counts']['frac_realized']}"
          f"）train {len(train)} seed={args.seed}")
    for s, v in info.items():
        print(f"[split]   {s}: 行数 [{v['rows_range'][0]},{v['rows_range'][1]}] "
              f"演员 {v['n_actors']} heldout {v['n_heldout']}")
    print(f"[split] -> {os.path.abspath(args.out)}")
    return doc


# ------------------------------------------------------------------ 模式：pick
def run_pick_mode(args):
    """D059 选样 JSON → 源 2 转换清单（排 held-out 演员、排镜像、按档内长度取头部保窗数）。

    选样器按 v_med 速度档均匀取样（G0-③ oracle 协议口径，不按签名滤）；本模式在
    其 selected 上叠加 D060 的三条约束：① 演员不在 held-out 划分表；② stem 非 `_M`
    镜像副本（§5t：镜像全排除出训练）；③ 段长足够多产窗（窗长 200@50Hz 需源行数
    ≥ 480）。档内按 n_frames_src 降序取 --top-per-band。
    """
    with open(args.select_json, encoding="utf-8") as f:
        sel = json.load(f)
    split = load_actor_split(args.split_json)
    rows = sel.get("selected") or []
    if not rows:
        raise SystemExit(f"{args.select_json} 无 selected 条目")
    keep, drop_held, drop_short, drop_mirror = [], [], [], []
    per_band = {}
    for r in rows:
        band = str(r.get("band_name", r.get("band", "?")))
        n_src = int(r.get("n_frames_src") or 0)
        if is_heldout(r["stem"], split):
            drop_held.append({"stem": r["stem"], "actor": r.get("actor_uid"),
                              "band": band})
            continue
        if is_mirror_stem(r["stem"]):
            drop_mirror.append({"stem": r["stem"], "actor": r.get("actor_uid"),
                                "band": band})
            continue
        if n_src < args.min_frames:
            drop_short.append({"stem": r["stem"], "n_frames_src": n_src, "band": band})
            continue
        per_band.setdefault(band, []).append(r)
    for band, rs in per_band.items():
        rs.sort(key=lambda r: (-int(r.get("n_frames_src") or 0), r["stem"]))
        keep.extend(rs[:args.top_per_band])
    keep.sort(key=lambda r: (str(r.get("band_name", r.get("band"))), r["stem"]))
    list_path = args.out_list
    os.makedirs(os.path.dirname(os.path.abspath(list_path)) or ".", exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(r["csv_path"] + "\n")
    est = {}
    for r in keep:
        n_enc = int(int(r["n_frames_src"]) // 2.4)
        est[r["stem"]] = {"n_frames_src": int(r["n_frames_src"]), "n_frames_enc_est":
                          n_enc, "windows_est": len(slice_starts(n_enc, args.win,
                                                                 args.stride))}
    doc = {"exp": "D060", "mode": "pick", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "script": os.path.abspath(__file__),
           "select_json": os.path.abspath(args.select_json),
           "split_json": split.get("path"),
           "rules": {"drop_heldout_actors": True, "drop_mirror_stems": True,
                     "min_frames_src": args.min_frames,
                     "top_per_band": args.top_per_band, "win": args.win,
                     "stride": args.stride,
                     "fps_enc": 50.0, "fps_src": 120.0},
           "counts": {"selected_in": len(rows), "kept": len(keep),
                      "dropped_heldout": len(drop_held),
                      "dropped_mirror": len(drop_mirror),
                      "dropped_short": len(drop_short),
                      "per_band_kept": {b: len([k for k in keep
                                                if str(k.get("band_name",
                                                             k.get("band"))) == b])
                                        for b in sorted(per_band)}},
           "est_windows_total": int(sum(v["windows_est"] for v in est.values())),
           "dropped_heldout": drop_held, "dropped_mirror": drop_mirror,
           "dropped_short": drop_short,
           "kept": [{"stem": r["stem"], "actor_uid": r.get("actor_uid"),
                     "band_name": r.get("band_name"), "v_med": r.get("v_med"),
                     "csv_path": r["csv_path"], **est[r["stem"]]} for r in keep],
           "out_list": os.path.abspath(list_path)}
    out_json = args.out_json or (os.path.splitext(list_path)[0] + ".pick.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print(f"[pick] 输入选中 {len(rows)} → 保留 {len(keep)}（heldout 剔 "
          f"{len(drop_held)}、镜像 剔 {len(drop_mirror)}、过短(<{args.min_frames} "
          f"源帧) 剔 {len(drop_short)}）预计窗数 {doc['est_windows_total']}")
    print(f"[pick] 分档保留 {doc['counts']['per_band_kept']}")
    print(f"[pick] -> {os.path.abspath(list_path)}\n[pick] -> {out_json}")
    return doc


# ------------------------------------------------------------------ selftest
def run_selftest():
    tmp = tempfile.mkdtemp(prefix="d060_selftest_")
    n_ok = n_bad = 0

    def chk(name, cond, extra=""):
        nonlocal n_ok, n_bad
        if cond:
            n_ok += 1
            print(f"  [ok]   {name} {extra}")
        else:
            n_bad += 1
            print(f"  [FAIL] {name} {extra}")

    def synth_segment(n=520, seed=0, speed=0.30, stem="synth_walk_001__A901"):
        """合成段：XY 直线匀速（y 向有 1e-4 级抖动 → speed 只近似恒定）。"""
        rng = np.random.default_rng(seed)
        step = speed / FPS  # m/frame 沿 +x
        trans = np.stack([np.arange(n) * step,
                          np.zeros(n) + rng.normal(0, 1e-4, n),
                          np.zeros(n) + 0.75], axis=1)
        jp = rng.normal(0, 0.2, (n, N_JOINT))
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        quat += rng.normal(0, 1e-3, (n, 4))
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        tok = (rng.integers(-11, 12, (n, TOKEN_DIM)) / 16.0).astype(np.float32)
        return tok, jp, quat, trans, stem

    def exact_segment(n=520, speeds=(0.2, 0.6), stem="synth_exact_001__A903"):
        """合成段：y/z 无抖动、速度分段严格恒定（用于解析对拍）。"""
        steps = np.where(np.arange(n) < n // 2, speeds[0] / FPS, speeds[1] / FPS)
        trans = np.stack([np.cumsum(steps), np.zeros(n), np.full(n, 0.75)], axis=1)
        jp = np.zeros((n, N_JOINT))
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        tok = np.zeros((n, TOKEN_DIM), dtype=np.float32)
        return tok, jp, quat, trans, stem

    try:
        print("[selftest 1] 格式契约（build_window 纯函数）")
        tok, jp, quat, trans, stem = synth_segment()
        st = state_from_kinematics(jp, quat, trans)
        chk("state_from_kinematics 形状/布局（canonical 36 维）",
            st.shape == (520, STATE_DIM) and STATE_DIM == 36
            and st.dtype == np.float32, f"{st.shape}/{st.dtype}")
        chk("state 列 29..32 = quat、33..35 = trans（无保留位）",
            np.allclose(st[0, 29:33], quat[0], atol=1e-6)
            and np.allclose(st[0, 33:36], trans[0], atol=1e-4)
            and st.shape[1] == N_JOINT + 4 + 3)
        w = build_window(tok[:WIN], st[:WIN], make_command(0.4),
                         make_terrain("climbing_box", height=0.5), None)
        chk("窗字段齐全", not window_contract_violations(w))
        chk("intent 自条件 = token_stream 逐元素相同",
            np.array_equal(w["intent_tokens"], w["token_stream"])
            and w["meta"]["intent_source"] == INTENT_SOURCE_SELF)
        chk("token dtype int16（1/16 网格反量化值 → 整数码，scale=16）",
            w["token_stream"].dtype == np.int16 and w["meta"]["token_scale"] == 16,
            f"scale={w['meta']['token_scale']}")
        chk("码可无损还原反量化值（code/scale == 源值 ×1e-6 内）",
            np.allclose(w["token_stream"][0] / w["meta"]["token_scale"], tok[0],
                        atol=1e-6))
        chk("token_grid_scale 探测：1/16 网格→16、1/4→4、整数→1",
            token_grid_scale(np.full(8, 0.3125)) == 16
            and token_grid_scale(np.full(8, 0.25)) == 4
            and token_grid_scale(np.arange(8.0)) == 1)
        chk("terrain 参数化 climbing_box{height:0.5}",
            w["terrain_desc"] == {"type": "climbing_box", "params": {"height": 0.5}})
        chk("command 哨兵 -1（mode/height/direction 无真值）",
            np.allclose(command_vector(w["command"]), [0.4, -1.0, -1.0, -1.0],
                        atol=1e-6)
            and command_has_truth_vector(w["command"]).tolist()
            == [True, False, False, False])
        raised = False
        try:
            build_window(np.full((WIN, TOKEN_DIM), 1.0 / 3.0, np.float32), st[:WIN],
                         make_command(0.4), make_terrain("plane"))
        except ValueError:
            raised = True
        chk("非 2 的幂网格 token（真连续量）触发 ValueError（不静默取整）", raised)
        raised = False
        try:
            make_terrain("stairs")
        except ValueError:
            raised = True
        chk("未知 terrain 类型触发 ValueError", raised)
        raised = False
        try:
            build_window(tok[:WIN], st[:WIN - 1], make_command(0.4),
                         make_terrain("plane"))
        except ValueError:
            raised = True
        chk("state/token 帧数不齐触发 ValueError", raised)
        chk("无真值字段强制回哨兵（防缺省 0 冒充真值）",
            make_command(0.4, has_truth={"target_vel": True, "movement_direction":
                                         False, "mode": False, "height": False})
            ["cmd"]["mode"] == SENTINEL)

        print("[selftest 2] speedA 计算对拍")
        v = frame_speed_bwd(trans, fps=FPS, axes="xy")
        chk("近似匀速段 frame_speed_bwd ≈ 0.30（y 抖动 1e-4 级）",
            np.allclose(v, 0.30, atol=1e-3), f"min {v.min():.4f} max {v.max():.4f}")
        chk("v[0] 镜像 v[1]", v[0] == v[1])
        got = window_speedA(trans, 0, WIN)
        chk("窗内 speedA = 中位（首窗，±1e-3）", abs(got - 0.30) < 1e-3, f"{got:.6f}")
        # 对拍 1：解析值（严格分段恒定速度，无抖动）
        tok_x, jp_x, q_x, tr_x, _ = exact_segment()
        vx_exact = frame_speed_bwd(tr_x, FPS, "xy")
        chk("严格恒定段帧速 = 解析值（0.2 前半 / 0.6 后半）",
            abs(vx_exact[1] - 0.2) < 1e-12 and abs(vx_exact[-1] - 0.6) < 1e-12,
            f"v[1]={vx_exact[1]:.6f} v[-1]={vx_exact[-1]:.6f}")
        chk("窗 [0,200) speedA = 0.2（解析）",
            abs(window_speedA(tr_x, 0, 200) - 0.2) < 1e-12,
            f"{window_speedA(tr_x, 0, 200):.6f}")
        chk("窗 [320,520) speedA = 0.6（解析，全在快段）",
            abs(window_speedA(tr_x, 320, 520) - 0.6) < 1e-12,
            f"{window_speedA(tr_x, 320, 520):.6f}")
        # 对拍 2：测试内独立实现（不复用模块函数）逐窗比对
        d = np.linalg.norm(np.diff(tr_x[:, :2], axis=0), axis=1) * FPS
        v_indep = np.concatenate([[d[0]], d])  # v[0]=v[1] 镜像，独立重写一遍
        ok_med = True
        for i0 in slice_starts(len(tr_x)):
            ref = float(np.median(v_indep[i0:i0 + 200]))
            if abs(window_speedA(tr_x, i0, i0 + 200) - ref) > 1e-12:
                ok_med = False
        chk("逐窗 speedA = 独立实现的帧速中位（全窗扫一遍）", ok_med)
        # 对拍 3：窗内切片重算差分的中位（本 fixture 无抖动 → 与整段口径全等；
        #   真实语料两者只**多数窗**等价，reviewer 实测 60 窗中 4 个不同、max 8.5e-4）
        got2 = window_speedA(trans, 300, 300 + WIN)
        chk("本 fixture 下切窗重算差分与整段差分中位一致（镜像只影响首帧；"
            "真实语料仅多数窗等价）",
            abs(float(np.median(frame_speed_bwd(trans[300:500], FPS))) - got2) < 1e-9)
        vxyz = frame_speed_bwd(trans, fps=FPS, axes="xyz")
        chk("axes='xyz' 含 z 抖动 > xy 口径", float(vxyz.max()) >= float(v.max()))

        print("[selftest 3] 滑窗边界")
        chk("n=199 < win → 0 窗", slice_starts(199) == [])
        chk("n=200 → [0]", slice_starts(200) == [0])
        chk("n=300 stride=100 → [0,100]", slice_starts(300) == [0, 100])
        chk("n=399 stride=100 → [0,100]", slice_starts(399) == [0, 100])
        chk("n=400 stride=100 → [0,100,200]", slice_starts(400) == [0, 100, 200])
        chk("stride 可调（stride=200）", slice_starts(700, 200, 200) == [0, 200, 400])
        chk("drop-last：末窗必须完整（尾帧 < win 不产窗）",
            all(s + 200 <= 399 for s in slice_starts(399)))
        ws = windows_from_arrays(tok, jp, quat, trans, terrain_desc=make_terrain("plane"),
                                 stem=stem, stride=100)
        chk("整段切窗数 = slice_starts 长度", len(ws) == len(slice_starts(len(tok))),
            f"{len(ws)}")
        chk("逐窗 speedA 标签 ≈ 0.30（±1e-3）",
            all(abs(w["meta"]["speedA"] - 0.30) < 1e-3 for w in ws))
        ws_x = windows_from_arrays(tok_x, jp_x, q_x, tr_x,
                                   terrain_desc=make_terrain("plane"), stem="exact")
        chk("严格段逐窗 speedA 落在 {0.2, 0.6} 解析值上",
            all(min(abs(w["meta"]["speedA"] - 0.2), abs(w["meta"]["speedA"] - 0.6))
                < 1e-12 for w in ws_x), f"n={len(ws_x)}")
        chk("窗帧区间连续（i1 = i0+200）",
            all(w["meta"]["frame_end"] - w["meta"]["frame_start"] == 200 for w in ws))
        chk("覆盖帧并集：重叠取并、断裂不虚增、空窗表为 0",
            covered_unique(slice_starts(520, 200, 100), 200, 520) == 500
            and covered_unique([0, 300], 200, 520) == 400
            and covered_unique([], 200, 520) == 0)

        print("[selftest 4] 演员解析 + 划分过滤")
        gj = os.path.join(tmp, "gate.json")
        with open(gj, "w", encoding="utf-8") as f:
            json.dump({"gate_mode": "envelope",
                       "rows": [{"stem": "s_pass", "hard_ok": True},
                                {"stem": "s_fail", "hard_ok": False},
                                {"stem": "s_err", "error": "boom"},
                                {"stem": "s_unknown", "whatever": 1}],
                       "errors": []}, f)
        gm = gate_status_map(gj)
        chk("gate_status_map 认 hard_ok 真键 + ERROR/UNKNOWN 不静默当 PASS",
            gm == {"s_pass": "PASS", "s_fail": "FAIL", "s_err": "ERROR",
                   "s_unknown": "UNKNOWN"}, f"{gm}")
        chk("parse_actor 常规 stem", parse_actor("come_up_50cm_box_R_004__A356") == "A356")
        chk("parse_actor 镜像 stem", parse_actor("come_up_50cm_box_R_004__A356_M") == "A356")
        chk("parse_actor 无演员 stem → None", parse_actor("Jump_002") is None)
        chk("is_mirror_stem：`_M$` 命中、母段与普通段不命中",
            is_mirror_stem("come_up_50cm_box_R_004__A356_M")
            and not is_mirror_stem("come_up_50cm_box_R_004__A356")
            and not is_mirror_stem("Jump_002__A017")
            and not is_mirror_stem("Neutral_stoop_down_001__A057"))
        split_path = os.path.join(tmp, "actor_split.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump({"seed": 0, "heldout": ["A356", "A017"],
                       "train": ["A901"],
                       "actors": {"A356": {"split": "heldout"}, "A901": {"split": "train"}}},
                      f)
        sp = load_actor_split(split_path)
        chk("load_actor_split 读 heldout/train", sp["heldout"] == {"A356", "A017"})
        chk("is_heldout 命中（含 _M 镜像）",
            is_heldout("x__A356", sp) and is_heldout("x__A356_M", sp)
            and not is_heldout("x__A901", sp) and not is_heldout("no_actor", sp))
        held, train_, info, _ = stratified_holdout(
            {f"A{i:03d}": 10 * (i % 5) + 1 for i in range(522)}, seed=0, frac=0.15)
        chk("分层抽样总数 = round(0.15×522) = 78", len(held) == 78, f"{len(held)}")
        chk("train + heldout = 全体且不相交",
            len(set(held) | set(train_)) == 522 and not (set(held) & set(train_)))
        h2, t2, _, as2 = stratified_holdout(
            {f"A{i:03d}": 10 * (i % 5) + 1 for i in range(522)}, seed=0, frac=0.15)
        chk("同 seed 复现逐名一致", h2 == held and t2 == train_)
        chk("层号登记覆盖全体", len(as2) == 522)
        h3, _, _, _ = stratified_holdout(
            {f"A{i:03d}": 10 * (i % 5) + 1 for i in range(522)}, seed=1, frac=0.15)
        chk("换 seed 结果改变（抽样确实用 seed）", h3 != held)

        print("[selftest 5] 落盘 + 对账 + loader 回读")
        npz_dir = os.path.join(tmp, "src")
        os.makedirs(npz_dir, exist_ok=True)
        stems = ["synth_001__A901", "synth_002__A356", "synth_003__A902_M"]
        for i, s in enumerate(stems):
            tok_i, jp_i, q_i, tr_i, _ = synth_segment(n=460, seed=i, speed=0.2 + 0.1 * i,
                                                      stem=s)
            np.savez(os.path.join(npz_dir, s + ".npz"), tokens=tok_i, jp_mj=jp_i,
                     quat_wxyz=q_i, trans_m=tr_i, meta=json.dumps({"stem": s}))
        segs = collect_segments([npz_dir], verbose=False)
        chk("段枚举", len(segs) == 3, f"{len(segs)}")
        all_w, tags, keep_stems, skip_mirror = [], [], [], []
        for s in segs:
            if is_heldout(s["stem"], sp):
                continue
            if is_mirror_stem(s["stem"]):   # §5t：镜像 _M 全排除出训练
                skip_mirror.append(s["stem"])
                continue
            wsi, rec, err = build_segment_windows(s, make_terrain("plane"), sp)
            chk(f"段 {s['stem']} 无 error", err is None, f"{err or ''}")
            all_w.extend(wsi)
            tags.extend([rec["split"]] * len(wsi))
            keep_stems.append(s["stem"])
        chk("held-out 演员段被剔除（A356 段不进窗）",
            "synth_002__A356" not in keep_stems and len(keep_stems) == 1)
        chk("镜像 _M 段被剔除且不进窗（§5t：`_M$` 段级排除）",
            "synth_003__A902_M" not in keep_stems
            and skip_mirror == ["synth_003__A902_M"]
            and all(is_mirror_stem(s) is False for s in keep_stems), f"{skip_mirror}")
        out_dir = os.path.join(tmp, "out")
        shards, wrecs = write_shards(all_w, os.path.join(out_dir, "npz"),
                                     shard_size=2, split_tags=tags)
        chk("分片数 = ceil(窗数/2)", len(shards) == int(np.ceil(len(all_w) / 2)),
            f"{len(shards)} shards / {len(all_w)} windows")
        doc = {"windows": wrecs, "shards": shards}
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        back = list(load_windows(os.path.join(out_dir, "manifest.json"), limit=2))
        chk("loader 回读 2 窗", len(back) == 2)
        chk("loader token 逐元素一致（int16 码）",
            np.array_equal(back[0]["token_stream"], all_w[0]["token_stream"]))
        chk("loader 回读 meta 带 token_scale=16（解码规则随窗走）",
            back[0]["meta"].get("token_scale") == 16)
        chk("loader state 逐元素一致",
            np.array_equal(back[0]["state"], all_w[0]["state"]))
        chk("loader command dict 还原（含哨兵与 has_truth）",
            np.allclose(command_vector(back[0]["command"]),
                        command_vector(all_w[0]["command"]))
            and command_has_truth_vector(back[0]["command"]).tolist()
            == command_has_truth_vector(all_w[0]["command"]).tolist())
        chk("loader 回读窗契约合规",
            not window_contract_violations(
                {k: back[0][k] for k in ("token_stream", "intent_tokens", "state",
                                         "command", "terrain_desc")}
                | {"meta": back[0]["meta"]}))
        chk("manifest 窗记录数 = 写出窗数", len(wrecs) == len(all_w))
        chk("分片窗计数对账 = manifest",
            sum(s["n"] for s in shards) == len(wrecs))
        ntr = sum(1 for t in tags if t == "train")
        back_tr = list(load_windows(os.path.join(out_dir, "manifest.json"),
                                    split="train"))
        chk("split=train 过滤后窗数正确（held-out 窗 0）", len(back_tr) == ntr,
            f"{len(back_tr)}")
        chk("no windows leak for A356 / for _M mirror (heldout/heldout tag 不存在)",
            all(r["split"] == "train" for r in wrecs)
            and all(not is_mirror_stem(r["stem"]) for r in wrecs))

        print("[selftest 6] manifest 对账字段（段级 窗×200 = 覆盖并集 + 冗余；"
              "frames = 覆盖并集 + 未覆盖）")
        for s in segs:
            wsi, rec, err = build_segment_windows(s, make_terrain("plane"), sp)
            if err:
                continue
            chk(f"段 {s['stem']} 帧数对账",
                rec["window_frames_visits"]
                == rec["frames_covered_unique"] + rec["overlap_redundancy"]
                and rec["frames"] == rec["frames_covered_unique"]
                + rec["frames_uncovered"],
                f"{rec['windows']}×200={rec['window_frames_visits']} = "
                f"{rec['frames_covered_unique']}+{rec['overlap_redundancy']}；"
                f"{rec['frames']}={rec['frames_covered_unique']}"
                f"+{rec['frames_uncovered']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n[selftest] 通过 {n_ok} / 失败 {n_bad}")
    return n_bad == 0


# ------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(
        description="D060 §5t 统一窗格式脊柱（格式/builder/落盘/划分/loader）")
    ap.add_argument("cmd", nargs="?", default=None,
                    choices=["window", "split", "pick", "selftest"],
                    help="子命令（省略且给 --selftest 时等同 selftest）")
    ap.add_argument("--selftest", action="store_true", help="合成 fixture 自检")
    # 通用
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--win", type=int, default=WIN)
    ap.add_argument("--stride", type=int, default=STRIDE)
    # window
    ap.add_argument("--npz-dirs", nargs="+", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--npz-out-dir", default=None, help="分片 npz 目录（默认 <out-dir>/npz）")
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument("--terrain", default="plane", choices=list(TERRAIN_TYPES))
    ap.add_argument("--box-height", type=float, default=None)
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--include-heldout", action="store_true",
                    help="也产 held-out 窗（tag=heldout，供 G5 评测；默认剔除）")
    ap.add_argument("--drop-unparsed-actor", action="store_true",
                    help="stem 无 `__A\\d+` 的段直接丢（默认保留并记 actor_unparsed）")
    ap.add_argument("--keep-mirror", action="store_true",
                    help="保留 `_M` 镜像段（§5t 默认全排除；保留时 manifest 记 "
                         "counts.segments_mirror_kept）")
    ap.add_argument("--gate-json", default=None, help="D059 gate JSON：FAIL/ERROR 段跳过")
    ap.add_argument("--shard-size", type=int, default=128)
    ap.add_argument("--shard-prefix", default="w")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 段（调试用）")
    ap.add_argument("--key-tokens", default=KEY_TOKENS)
    ap.add_argument("--key-jp", default=KEY_JP)
    ap.add_argument("--key-quat", default=KEY_QUAT)
    ap.add_argument("--key-trans", default=KEY_TRANS)
    # split
    ap.add_argument("--parquet", default="data/ds_bones/seed_metadata_v004.parquet")
    ap.add_argument("--out", default="data/ds_bones/actor_split_d060.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--strata", type=int, default=3)
    # pick
    ap.add_argument("--select-json", default=None)
    ap.add_argument("--out-list", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--top-per-band", type=int, default=20)
    ap.add_argument("--min-frames", type=int, default=600,
                    help="源行数下限（窗长 200@50Hz 需源帧 ≥ 480 才有一窗）")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    cmd = args.cmd
    if args.selftest and cmd is None:
        cmd = "selftest"
    if cmd == "selftest" or args.selftest:
        return 0 if run_selftest() else 1
    if cmd == "split":
        run_split_mode(args)
        return 0
    if cmd == "pick":
        if not (args.select_json and args.out_list):
            raise SystemExit("pick 需 --select-json 与 --out-list")
        run_pick_mode(args)
        return 0
    if cmd == "window":
        if not args.npz_dirs:
            raise SystemExit("window 需 --npz-dirs")
        if not args.out_dir:
            args.out_dir = os.path.join(args.npz_dirs[0], "d060_windows")
        if not args.npz_out_dir:
            args.npz_out_dir = os.path.join(args.out_dir, "npz")
        run_window_mode(args)
        return 0
    raise SystemExit("用法：d060_windows.py {window|split|pick|selftest} …（--help）")


if __name__ == "__main__":
    sys.exit(main())
