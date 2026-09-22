"""D062=A3 作者 v1（≤300M）联合升级训练：chunk 头 + 特权辅助头 + 四跑消融。

预注册 = refine-logs/ds/DS_CONTINUOUS_EXECUTION_PLAN.md §5x（:467-490）：
①规模 ≤300M 硬门（实测打印，超则减层重配，D061 同款纪律）；②chunk 头（**自回归版，
非并行一次性输出**）：一次前向输出 K 帧 token，K 起步 8、消融扫 {4,8,20}（对齐 40 步
chunk 整除），训练 = 逐帧 teacher-forcing + chunk 内自回归 rollout 约束（rollout 段
CE 累积，权重 ramp），推理 = 滑窗 + clean-prefix 缓存（上一 chunk 末尾**已执行段**
作当前前缀，对应 ω-0 RTC 机制）；③特权辅助预测头：ctx 表征上挂轻量头预测未来 K 步
可导出监督量（未来根速度 = state trans3 逐帧差分、未来关节速度 = jp29 差分、地形
标量参数 = one-hot + params 回归），部署时该头不参与推理（推理环零改动）；
④四跑消融 CLI --variant anchor|chunk|priv|both。

四跑 ↔ §5x 命名对照（CLI 值 = 短名，元数据同时记预注册名）
--------------------------------------------------------
    anchor → v1-锚：300M 纯放大（同 v0 配方仅变规模；回应 09-18 未决③「先补规模」）
    chunk  → v1a  ：锚 + 仅 chunk 头
    both   → v1b  ：锚 + chunk 头 + 特权头（**主配置**）
    priv   → v1c  ：锚 + 仅特权头
四跑由外部脚本/命令行分别起（本脚本一次只跑一个 variant，不内部循环）。

与 v0（apt_g1/training/train_author_v0.py）的关系
-------------------------------------------------
**数据接口逐字兼容**：语料定位/索引（装配 manifest → 四源逐窗记录）、ctx 13 维口径
（command4+has_truth4+terrain one-hot3+height/noise2）、码平移 code_min 非负索引、
state 36 维（jp29+quat4+trans3）、窗 200×64/200×36、split=='train' 硬断言——全部沿
v0（本脚本同样**自包含**：不 import 任何 apt_g1/d060 模块，常量是 d060_windows 的
硬编码副本，可平铺到执行根直接跑）。因此**同一语料目录 v0/v1 都可读**。
v1 增量三件：chunk 头、特权头、四跑 variant 开关；训练配方沿 v0（adafactor 固定 lr +
warmup + cosine，warmup 步数按新规模放大到 2,000）。

模型（骨架源自 v0 AuthorV0Transformer = consumer_stub TinyCausalWindowTransformer 同构）
----------------------------------------------------------------------------------------
- 主干：帧嵌入 = state_proj(state) + 64 token 维共享 embedding 取均值 + intent 同构；
  command/terrain 上下文投成双前缀帧（n_ctx=2）；可学位置表；torch.triu 因果 mask；
  逐层 nn.TransformerEncoderLayer（pre-LN/gelu）+ torch.utils.checkpoint 梯度检查点；
  head = Linear(d_model, 64*V) 出 (b, t, 64, V)。
- ctx 表征 = ln_f 后两个前缀帧的均值池化（(b, d_model)）——chunk 头与特权头都挂它。
- chunk 头（自回归）：ctx 表征投影进 d_chunk + 已执行前缀帧嵌入（64 维共享表取均值）
  + 步位置表，过 n 层因果小 transformer；**逐帧前向**取末位出 (b, 64, V)，把自身
  预测（softmax 软嵌入，可微）回灌作下一步输入 → 一次调用输出 K 帧。训练只算
  rollout 段 CE（对真值帧 P..P+K-1 累积）；推理走 chunk_generate（硬 argmax 回灌）。
- 特权头：ctx 表征 → Linear+GELU(priv_hidden) → 四/五个轻量输出头：
  root_vel (b,K,3)、joint_vel (b,K,29)、terrain_type (b,3 分类)、terrain_param (b,2 回归)，
  以及 --priv-extra-cols 指定的**真特权列**（elevation/接触；缺省空，语料补录后即插，
  逐帧形状 (WIN,d) 或 (WIN,) → 监督目标取帧 [1,K+1]）。部署时该头丢弃。
- 规模默认：d_model 1280 / n_layer 18 / n_head 10 / ffn 2560（≈268.5M 主干，实测打印；
  超 --max-params 300,000,000 则减一层重配）；chunk 头默认 d_chunk 512 × 2 层
  （≈5.7M）→ both ≈274M，四 variant 均 ≤300M（--print-params-only 四口径实打印）。

硬纪律（沿 v0，逐条可查）
------------------------
- heldout 演员级防泄漏：只索引 split=='train' 窗，索引后断言 heldout 零进入，
  meta 落 leak_audit={train_windows, heldout_excluded, heldout_entered:0}。
- val 窗级 5% seed0（rng 独立于语料 seed，不碰 heldout）。
- 码域 V=29（G4 装配实读四源并集 [-15,13]）与 199 目标位（logits[:,:-1] vs tok[:,1:]）。
- 身份信封：git commit / 四源 manifest md5 / variant / 参数量（写 meta.json）。
- **步数预算 = 收敛判据**（尾部 5 eval 点相对变化 <3% 且过两条触发守卫即停，
  --converge-stop 默认开；epochs 与 --max-steps 只是上界，不硬编码「必须跑满 N 步」）。
  两条守卫（D063 1B，防 D062 首轮四臂误停于 warmup 终点）：① 相位守卫=尾窗须全部
  落在 warmup 后的余弦下降段；② 降势守卫=窗口须整体走平（净降 ≤ 单步噪声带），拦
  「缓慢匀速下降被 max-min<3% 误当平台」。触发状态写入 meta.converge_guard。
- 门判读沿用 D061a 四条（NaN/尾部收敛/top1≥0.185/泄漏=0），打印 D062_GATE_*。
- --intent-pin-max：**仅推理期**把 ctx 命令槽（target_vel）钉到语料速度带最大档
  （速度带 = train 窗 speedA 三分位，最大档取值 = 上三分位中位），训练不受影响。

用法
----
    python train_author_v1.py --selftest                 # 本机 CPU 全链路（SELFTEST_PASS）
    python train_author_v1.py --variant both             # 服务器主配置（v1b）
    python train_author_v1.py --variant anchor --out-dir outputs/d062_v1_anchor
    python train_author_v1.py --print-params-only        # 四 variant 参数量实打印 + 预算核
    python train_author_v1.py --chunk-k 20 --variant chunk   # 消融扫 K∈{4,8,20}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------ 冻结常量
# （d060_windows.py 硬编码副本：本脚本自包含，服务器平铺布局不依赖 apt_g1 包）
WIN = 200                 # 4s 窗 @50Hz（§5t 冻结）
TOKEN_DIM = 64           # FSQ token 维
STATE_DIM = 36           # jp29 + quat4 + trans3（owner 2026-09-17 canonical 裁决）
FPS = 50.0               # 编码帧率（差分求速度用）
SENTINEL = -1.0          # D033 无真值哨兵
COMMAND_FIELDS = ("target_vel", "movement_direction", "mode", "height")
TERRAIN_TYPES = ("plane", "rough_paper", "climbing_box")
TERRAIN_DEFAULT_PARAMS = {"plane": {}, "rough_paper": {"noise": 0.04},
                          "climbing_box": {"height": 0.5}}
# 上下文向量 = command(4) + has_truth(4) + terrain one-hot(3) + (height,noise)(2)
CTX_DIM = 4 * 2 + 3 + 2
# state 布局（canonical 36 维）——特权头目标取用
JP_SLICE = slice(0, 29)          # jp_mj 29
TRANS_SLICE = slice(33, 36)      # root trans_m 3
CTX_TERRAIN_ONEHOT = slice(8, 11)
CTX_TERRAIN_PARAMS = slice(11, 13)

# 四跑消融（§5x 预注册矩阵）→ (use_chunk, use_priv)
VARIANT_HEADS = {"anchor": (False, False), "chunk": (True, False),
                 "priv": (False, True), "both": (True, True)}
VARIANT_PREREG_NAME = {"anchor": "v1-锚", "chunk": "v1a", "priv": "v1c",
                       "both": "v1b(主配置)"}
CHUNK_K_SCAN = (4, 8, 20)        # §5x 消融扫（整除 40 步 chunk）

# 判据数字（沿 D061a 冻结，勿改）
MIN_LR = 1e-5            # cosine 终点
GRAD_CLIP = 1.0
GATE_TAIL_N = 5          # 门②尾部 eval 点数
GATE_TAIL_REL = 0.03     # 门②/收敛判据：尾部相对变化上限
GATE_TOP1_MIN = 0.185    # 门③ val top1 下限
SPEED_EVERY = 50         # [SPEED] 行间隔（step）
CHUNK_MAX_SEQ = 128      # chunk 头位置表容量（1 + 前缀 + K 上限，超出 fail loud）

# 模型构造 kwarg 白名单（ckpt 的 model_cfg 按此重建模型，eval 侧 load_author_v1 用）
MODEL_KWARG_KEYS = ("d_model", "n_layer", "n_head", "ffn", "grad_ckpt",
                    "use_chunk", "use_priv", "chunk_k", "chunk_prefix",
                    "chunk_d_model", "chunk_layers", "chunk_n_head", "chunk_ffn",
                    "priv_hidden", "priv_extra")


# ------------------------------------------------------------------ 小工具
def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def md5_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def git_rev():
    """当前 git 提交（非 git 环境如实返回 None，不炸）。"""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(
            os.path.abspath(__file__)), capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001 —— 环境探测兜底
        return None


def write_json(path, obj):
    """UTF-8 无 BOM、LF 行尾落 JSON。"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def parse_extra_cols(spec):
    """--priv-extra-cols 'elevation_patch,contact' → ['elevation_patch','contact']（空→[]）。"""
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return [str(x).strip() for x in spec if str(x).strip()]
    return [p.strip() for p in str(spec).split(",") if p.strip()]


def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("d062")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh_stream = open(os.path.join(out_dir, "train.log"), "w",
                     encoding="utf-8", newline="\n")
    fh = logging.StreamHandler(fh_stream)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ------------------------------------------------------------------ 上下文向量（沿 v0 口径）
def ctx_from_record(rec):
    """manifest 逐窗记录 → 13 维上下文（command4+has_truth4+onehot3+height/noise2）。"""
    cmd = [float(rec.get(k, SENTINEL)) for k in COMMAND_FIELDS]
    ht = [1.0 if k in (rec.get("has_truth") or []) else 0.0 for k in COMMAND_FIELDS]
    ttype = rec.get("terrain")
    if ttype not in TERRAIN_TYPES:
        raise ValueError(f"窗 {rec.get('stem')} terrain 非法 {ttype!r}"
                         f"（允许 {TERRAIN_TYPES}）")
    onehot = [0.0, 0.0, 0.0]
    onehot[TERRAIN_TYPES.index(ttype)] = 1.0
    # climbing_box→height 0.5、rough_paper→noise、plane→0（记录值覆盖缺省）
    p = dict(TERRAIN_DEFAULT_PARAMS[ttype])
    p.update(rec.get("terrain_params") or {})
    hp = [float(p.get("height", 0.0)), float(p.get("noise", 0.0))]
    return np.asarray(cmd + ht + onehot + hp, dtype=np.float32)


# ------------------------------------------------------------------ 速度带 / intent-pin-max
def compute_speed_bands(speeds):
    """train 窗 speedA 序列 → 三分位速度带（§5t speedA 锚定标签法口径，禁均值校准）。

    返回 {"n", "tertile_lo", "tertile_hi", "bands":[低/中/高 档代表值],
          "counts":[三档窗数], "max_band_value"}；speedA 缺失/全同值时按窗 ctx
    target_vel 兜底由调用侧负责（本函数只吃数值序列）。
    档代表值 = 该档内 speedA 中位（锚定口径，不用均值）。
    """
    v = np.asarray([x for x in speeds if x is not None and np.isfinite(x)],
                   dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "tertile_lo": None, "tertile_hi": None, "bands": [],
                "counts": [0, 0, 0], "max_band_value": None}
    lo = float(np.quantile(v, 1.0 / 3.0))
    hi = float(np.quantile(v, 2.0 / 3.0))
    masks = [v <= lo, (v > lo) & (v <= hi), v > hi]
    bands, counts = [], []
    for m in masks:
        seg = v[m]
        counts.append(int(seg.size))
        bands.append(float(np.median(seg)) if seg.size else None)
    max_band = None
    for b in reversed(bands):  # 最大档：优先取上三分位；空则退中/低档
        if b is not None:
            max_band = b
            break
    return {"n": int(v.size), "tertile_lo": lo, "tertile_hi": hi, "bands": bands,
            "counts": counts, "max_band_value": max_band}


def pin_ctx_max_band(ctx_feat, band_value):
    """推理期把 ctx 命令槽钉到语料速度带最大档（--intent-pin-max）。

    **只用于推理**（训练不受影响）：target_vel 槽（idx 0）置 band_value，
    对应 has_truth 槽（idx 4）置 1.0（有真值）。返回新数组，**不原地改**输入。
    """
    if band_value is None:
        raise ValueError("intent-pin-max 需要语料速度带最大档值（speed_bands 为空）")
    out = np.array(ctx_feat, dtype=np.float32, copy=True)
    out[..., 0] = float(band_value)
    out[..., 4] = 1.0
    return out


def infer_ctx(corpus, ctx_feat, intent_pin_max=False):
    """推理侧 ctx 出口（eval 侧统一调它）：--intent-pin-max 时钉最大速度档。"""
    if not intent_pin_max:
        return np.asarray(ctx_feat, dtype=np.float32)
    return pin_ctx_max_band(ctx_feat, (corpus.get("speed_bands") or {}).get(
        "max_band_value"))


# ------------------------------------------------------------------ 路径解析（沿 v0）
def _reroot(abs_path, old_root, data_root):
    """绝对路径按装配 data_root 前缀重定位到 --data-root 下（服务器→本地搬移用）。"""
    if not (abs_path and old_root and data_root and os.path.isabs(abs_path)):
        return None
    old_root = old_root.rstrip(os.sep) + os.sep
    if abs_path.startswith(old_root):
        return os.path.join(data_root, os.path.relpath(abs_path, old_root))
    return None


def resolve_source_manifest(mpath, data_root, asm_root, sid):
    cands = [mpath]
    if data_root:
        cands.append(_reroot(mpath, asm_root, data_root))
        cands.append(os.path.join(data_root, os.path.basename(mpath)))
        cands.append(os.path.join(data_root, sid, os.path.basename(mpath)))
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"源 {sid} manifest 缺失：{mpath}（找过 {cands}）")


def resolve_shard_path(sdir, name, recorded, data_root, asm_root):
    """分片定位：相对源目录（npz/ 同级）或 manifest 记的绝对路径；--data-root 重定位。"""
    cands = [os.path.join(sdir, "npz", name), recorded, os.path.join(sdir, name)]
    if data_root:
        cands.append(os.path.join(data_root, name))
        cands.append(os.path.join(data_root, "npz", name))
        cands.append(_reroot(recorded, asm_root, data_root))
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"分片缺失：{name}（找过 {[c for c in cands if c]}）")


# ------------------------------------------------------------------ 语料索引（v0 接口兼容）
def build_corpus_index(assembly_manifest, data_root=None, exclude_src4=False,
                       limit_windows=None, logger=None):
    """装配 manifest → 四源逐窗记录 → train 窗索引（**与 v0 同一接口/同一目录可读**）。

    只索引 split=='train'；heldout 全量排除并计数（leak_audit）。返回 dict 含
    entries（shard 绝对路径 + index + 预计算 ctx）、code_min/vocab、leak_audit、
    逐源统计、四源 manifest md5（身份信封）与 **speed_bands**（v1 新增：train 窗
    speedA 三分位速度带，供 --intent-pin-max 推理期钉档）。
    """
    def log(msg, *a):
        if logger is not None:
            logger.info(msg, *a)
        else:
            print(msg % a if a else msg)

    if not os.path.isfile(assembly_manifest):
        raise SystemExit(f"[corpus] 装配 manifest 不存在：{assembly_manifest}")
    with open(assembly_manifest, encoding="utf-8") as f:
        asm = json.load(f)
    srcs = asm.get("sources") or {}
    if not srcs:
        raise SystemExit("[corpus] 装配 manifest 无 sources（四源引用清单为空）")
    asm_root = asm.get("data_root")

    entries, md5s, src_stats = [], {}, {}
    code_min = code_max = None
    heldout_excluded = other_excluded = 0
    for sid in sorted(srcs):
        blk = srcs[sid] or {}
        if exclude_src4 and str(sid).startswith("src4"):
            log("[corpus] --exclude-src4：跳过源 %s", sid)
            continue
        mpath = resolve_source_manifest(blk.get("manifest_path"), data_root,
                                        asm_root, sid)
        with open(mpath, encoding="utf-8") as f:
            sdoc = json.load(f)
        md5s[sid] = md5_file(mpath)
        counts = sdoc.get("counts") or blk.get("counts") or {}
        tmin, tmax = counts.get("token_min"), counts.get("token_max")
        if tmin is None or tmax is None:
            raise SystemExit(f"[corpus] 源 {sid} 无 token_min/token_max，无法定 vocab")
        code_min = int(tmin) if code_min is None else min(code_min, int(tmin))
        code_max = int(tmax) if code_max is None else max(code_max, int(tmax))

        shard_paths = {s.get("name"): s.get("path")
                       for s in sdoc.get("shards", [])}
        sdir = os.path.dirname(os.path.abspath(mpath))
        n_src_train = n_src_held = 0
        first_entry = None
        for rec in sdoc.get("windows") or []:
            split = rec.get("split")
            if split == "heldout":
                heldout_excluded += 1
                n_src_held += 1
                continue
            if split != "train":
                other_excluded += 1
                continue
            name = rec.get("shard")
            resolved = resolve_shard_path(sdir, name, shard_paths.get(name),
                                          data_root, asm_root)
            e = {"source": sid, "shard": resolved, "index": int(rec["index"]),
                 "split": "train", "ctx": ctx_from_record(rec),
                 # speedA：§5t 锚定标签（无则 None，速度带兜底到 ctx target_vel）
                 "speedA": (None if rec.get("speedA") is None
                            else float(rec["speedA"]))}
            entries.append(e)
            n_src_train += 1
            if first_entry is None:
                first_entry = e
        # 单源首窗契约抽查：形状 200×64 / 200×36 硬断言（fail loud 不静默）
        if first_entry is not None:
            with np.load(first_entry["shard"], allow_pickle=False) as z:
                t = z["token_stream"][first_entry["index"]]
                s = z["state"][first_entry["index"]]
            assert t.shape == (WIN, TOKEN_DIM), f"{sid} token 形状 {t.shape}"
            assert s.shape == (WIN, STATE_DIM), f"{sid} state 形状 {s.shape}"
        src_stats[sid] = {"manifest": os.path.abspath(mpath),
                          "manifest_md5": md5s[sid], "windows_train": n_src_train,
                          "windows_heldout": n_src_held}
        log("[corpus] 源 %s：train %d / heldout %d（manifest %s）",
            sid, n_src_train, n_src_held, md5s[sid])

    # 防泄漏断言：heldout 零进入训练索引（leak 断言——索引里只允许 split=='train'）
    heldout_entered = sum(1 for e in entries if e.get("split") == "heldout")
    assert heldout_entered == 0, \
        f"leak 断言失败：{heldout_entered} 个 heldout 窗进入训练索引（泄漏）"
    bad = [e for e in entries if e["split"] != "train"]
    assert not bad, f"leak 断言失败：{len(bad)} 个非 train 窗进入训练索引（heldout 泄漏）"

    n_before_limit = len(entries)
    if limit_windows:
        entries = entries[:int(limit_windows)]
    if not entries:
        raise SystemExit("[corpus] train 窗索引为空（检查 manifest/split/--exclude-src4）")
    vocab = int(code_max - code_min + 1)
    leak_audit = {
        "train_windows": len(entries),
        "heldout_excluded": heldout_excluded,
        "heldout_entered": heldout_entered,  # == 0（上方断言）
        "corpus_has_heldout": bool(heldout_excluded > 0),
        "other_split_excluded": other_excluded,
        "train_before_limit": n_before_limit,
    }
    # 速度带（v1）：train 窗 speedA 三分位；speedA 全缺时退用 ctx target_vel 真值窗
    speeds = [e["speedA"] for e in entries if e["speedA"] is not None]
    band_src = "speedA(§5t 锚定标签，整段差分窗内中位)"
    if not speeds:
        speeds = [float(e["ctx"][0]) for e in entries
                  if float(e["ctx"][0]) != SENTINEL]
        band_src = "ctx.target_vel 兜底（语料未记 speedA）"
    speed_bands = compute_speed_bands(speeds)
    speed_bands["source"] = band_src
    log("[corpus] 索引完成：train %d（limit 前 %d）/ heldout 排除 %d / 码域 [%d,%d] V=%d / "
        "速度带三档 %s（最大档 %s，源=%s）",
        len(entries), n_before_limit, heldout_excluded, code_min, code_max, vocab,
        speed_bands["bands"], speed_bands["max_band_value"], band_src)
    return {"assembly": os.path.abspath(assembly_manifest), "entries": entries,
            "code_min": code_min, "code_max": code_max, "vocab": vocab,
            "leak_audit": leak_audit, "source_stats": src_stats,
            "md5s": md5s, "speed_bands": speed_bands,
            "corpus_md5_combined": md5_text(
                "".join(f"{k}:{v}\n" for k, v in sorted(md5s.items())))}


def split_train_val(entries, val_frac, val_seed):
    """窗级 val 切分（rng 独立于语料 seed；heldout 不参与——它已不在 entries 里）。"""
    n = len(entries)
    rng = np.random.default_rng(int(val_seed))
    perm = rng.permutation(n)
    val_n = int(round(n * float(val_frac)))
    if n >= 2:
        val_n = max(val_n, 1)
        val_n = min(val_n, n - 1)
    else:
        val_n = 0
    val_idx = {int(i) for i in perm[:val_n]}
    train_pool = [e for i, e in enumerate(entries) if i not in val_idx]
    val_pool = [e for i, e in enumerate(entries) if i in val_idx]
    return train_pool, val_pool


# ------------------------------------------------------------------ Dataset
class WindowDataset(Dataset):
    """(shard,index) 定位窗 → 批元素（码平移非负索引；ctx 预计算自 manifest 记录）。

    分片解压结果按分片缓存（npz 每次键访问都整片解压，逐窗裸读代价不可接受）；
    缓存随 worker 进程各自持有（pickle 时清空，NpzFile 不可序列化）。
    extra_cols = 特权真值列（--priv-extra-cols；elevation/接触等语料补录后即插），
    缺失即 fail loud（不静默全零）。
    """

    def __init__(self, entries, code_min, vocab, extra_cols=()):
        self.entries = entries
        self.code_min = int(code_min)
        self.vocab = int(vocab)
        self.extra_cols = tuple(extra_cols)
        self._cache = {}

    def __len__(self):
        return len(self.entries)

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_cache"] = {}
        return st

    def _shard(self, path):
        z = self._cache.get(path)
        if z is None:
            keys = ("token_stream", "intent_tokens", "state") + self.extra_cols
            with np.load(path, allow_pickle=False) as npz:
                z = {k: npz[k] for k in keys}
            self._cache[path] = z
        return z

    def _extra_frame(self, z, j):
        """特权真值列 → (WIN, D) 逐帧矩阵（D=0 表示未启用）。"""
        if not self.extra_cols:
            return np.zeros((WIN, 0), dtype=np.float32)
        parts = []
        for name in self.extra_cols:
            if name not in z:
                raise KeyError(
                    f"特权列 {name!r} 不在分片 npz（--priv-extra-cols 需语料补录后"
                    f"才可用；已要求 {list(self.extra_cols)}）")
            a = np.asarray(z[name][j], dtype=np.float32)
            if a.ndim == 0:
                raise ValueError(f"特权列 {name!r} 形状 {a.shape} 非逐帧"
                                 f"（需 (WIN,) 或 (WIN,d)）")
            if a.shape[0] != WIN:
                raise ValueError(f"特权列 {name!r} 首轴 {a.shape[0]} != WIN {WIN}")
            parts.append(a.reshape(WIN, -1))
        return np.concatenate(parts, axis=1)

    def __getitem__(self, i):
        e = self.entries[i]
        z = self._shard(e["shard"])
        j = e["index"]
        return {
            "tok": z["token_stream"][j].astype(np.int64) - self.code_min,
            "intent": z["intent_tokens"][j].astype(np.int64) - self.code_min,
            "state": z["state"][j].astype(np.float32),
            "ctx": e["ctx"].astype(np.float32),
            "priv_extra_frame": self._extra_frame(z, j),
        }


def peek_extra_width(entries, extra_cols):
    """从数据侧实测真特权列总宽（模型 priv_extra 唯一事实源：列数≠列宽，
    elevation_patch=81/contact=4 等宽度只有分片知道；空列/空池→0）。"""
    cols = tuple(extra_cols)
    if not cols:
        return 0
    if not entries:
        raise ValueError("peek_extra_width：列 %r 已指定但窗口池为空" % (cols,))
    e = entries[0]
    with np.load(e["shard"], allow_pickle=False) as npz:
        total = 0
        for name in cols:
            if name not in npz:
                raise KeyError(
                    f"特权列 {name!r} 不在分片 npz（--priv-extra-cols 需语料补录后"
                    f"才可用；已要求 {list(cols)}）")
            a = np.asarray(npz[name][e["index"]], dtype=np.float32)
            if a.ndim == 0:
                raise ValueError(f"特权列 {name!r} 形状 {a.shape} 非逐帧")
            total += int(a.reshape(WIN, -1).shape[1])
    return total


def make_loaders(train_pool, val_pool, corpus, args, device):
    workers = int(args.num_workers)
    if workers > 0 and os.name == "nt":
        print("[data] Windows 检出：num_workers 回退 0（spawn 兼容）")
        workers = 0
    pin = device.type == "cuda"
    g = torch.Generator().manual_seed(int(args.seed))
    extra_cols = parse_extra_cols(args.priv_extra_cols)
    train_ds = WindowDataset(train_pool, corpus["code_min"], corpus["vocab"],
                             extra_cols)
    val_ds = WindowDataset(val_pool, corpus["code_min"], corpus["vocab"], extra_cols)
    train_ld = DataLoader(train_ds, batch_size=int(args.batch_windows), shuffle=True,
                          num_workers=workers, generator=g, pin_memory=pin,
                          drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=int(args.batch_windows), shuffle=False,
                        num_workers=workers, pin_memory=pin, drop_last=False)
    return train_ld, val_ld


# ------------------------------------------------------------------ 模型：chunk 头
class ChunkHead(nn.Module):
    """自回归 chunk 头：ctx 表征 + 干净前缀帧 → 逐帧前向输出 K 帧 token（非并行一次性）。

    序列 = [ctx_proj(ctx)] + [帧嵌入(前缀帧/已生成帧)] + 步位置表；因果 mask；
    末位出 (b, 64, V)。训练 rollout：把自身预测的 **softmax 软嵌入**回灌（可微，
    梯度经共享帧嵌入表回传）；推理 chunk_generate 用硬 argmax 回灌。
    """

    def __init__(self, vocab, d_ctx, d_chunk=512, n_layer=2, n_head=8, ffn=1024):
        super().__init__()
        self.vocab = int(vocab)
        self.d_chunk = int(d_chunk)
        self.ctx_proj = nn.Linear(int(d_ctx), self.d_chunk)
        self.frame_emb = nn.Embedding(self.vocab, self.d_chunk)  # 64 token 维共享
        self.step_pos = nn.Parameter(torch.zeros(1, CHUNK_MAX_SEQ, self.d_chunk))
        layer = nn.TransformerEncoderLayer(
            self.d_chunk, int(n_head), int(ffn), dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layer),
                                             enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(self.d_chunk)
        self.head = nn.Linear(self.d_chunk, TOKEN_DIM * self.vocab)

    def frames_emb(self, idx):
        """(b, L, 64) 码索引 → (b, L, d_chunk)（64 token 维取均值，与主干同构）。"""
        return self.frame_emb(idx).mean(dim=2)

    def _run(self, seq):
        """(b, L, d_chunk) → (b, L, d_chunk)（因果）。"""
        L = int(seq.shape[1])
        if L > self.step_pos.shape[1]:
            raise ValueError(f"chunk 序列长 {L} 超位置表 {self.step_pos.shape[1]}"
                             f"（--chunk-prefix + --chunk-k 过大）")
        x = seq + self.step_pos[:, :L]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device),
                          diagonal=1)
        for lyr in self.encoder.layers:
            x = lyr(x, mask)
        return self.ln_f(x)

    def logits_next_emb(self, ctx_vec, prev_emb):
        """ctx_vec (b,d_ctx) + 已嵌入前缀帧 (b,P,d_chunk) → (b,64,V) 下一帧 logits。"""
        seq = [self.ctx_proj(ctx_vec).unsqueeze(1)]
        if prev_emb is not None and prev_emb.shape[1] > 0:
            seq.append(prev_emb)
        x = self._run(torch.cat(seq, dim=1))
        return self.head(x[:, -1]).view(-1, TOKEN_DIM, self.vocab)

    def logits_next(self, ctx_vec, prev_idx):
        """ctx_vec (b,d_ctx) + 前缀**码索引** (b,P,64) → (b,64,V)（推理路径用）。"""
        prev_emb = (None if prev_idx is None or prev_idx.shape[1] == 0
                    else self.frames_emb(prev_idx.long()))
        return self.logits_next_emb(ctx_vec, prev_emb)

    def rollout(self, ctx_vec, prefix_idx, true_idx):
        """自回归 rollout K 帧（软嵌入回灌）→ {"logits": (b,K,64,V), "step_ce": [...]}。

        prefix_idx (b,P,64) = clean 前缀（训练期取真值帧，与推理期已执行段同语义）；
        true_idx (b,K,64) = 目标真值帧（CE 只在 rollout 段累积，权重由训练侧 ramp）。
        **统一口径**：prev 始终是 d_chunk 空间的帧嵌入（前缀先过 frames_emb，生成帧
        用软嵌入），不混原始码索引。
        """
        k = int(true_idx.shape[1])
        prev = (None if prefix_idx is None or prefix_idx.shape[1] == 0
                else self.frames_emb(prefix_idx.long()))      # (b,P,d_chunk)
        logits, step_ce = [], []
        for j in range(k):
            lg = self.logits_next_emb(ctx_vec, prev)  # (b,64,V)
            logits.append(lg)
            step_ce.append(F.cross_entropy(lg.float().reshape(-1, self.vocab),
                                           true_idx[:, j].reshape(-1)))
            probs = F.softmax(lg.float(), dim=-1)             # (b,64,V)
            soft = torch.einsum("btv,vd->btd", probs,
                                self.frame_emb.weight.float())  # (b,64,d_chunk)
            soft = soft.mean(dim=1, keepdim=True)               # (b,1,d_chunk)
            prev = soft if prev is None else torch.cat([prev, soft], dim=1)
        return {"logits": torch.stack(logits, dim=1),
                "step_ce": step_ce,
                "loss_ce": torch.stack(step_ce).mean()}


# ------------------------------------------------------------------ 模型：特权头
class PrivilegedHead(nn.Module):
    """ctx 表征 → 未来 K 步特权量（部署丢弃；推理环零改动）。

    输出：root_vel (b,K,3)、joint_vel (b,K,29)、terrain_type (b,3 分类)、
    terrain_param (b,2 回归)、extra (b,K,n_extra，--priv-extra-cols 真特权列即插)。
    """

    def __init__(self, d_ctx, k, hidden=256, n_joint=29, n_extra=0):
        super().__init__()
        self.k = int(k)
        self.n_extra = int(n_extra)
        self.trunk = nn.Sequential(nn.Linear(int(d_ctx), int(hidden)), nn.GELU())
        self.root_vel = nn.Linear(int(hidden), self.k * 3)
        self.joint_vel = nn.Linear(int(hidden), self.k * int(n_joint))
        self.terrain_type = nn.Linear(int(hidden), len(TERRAIN_TYPES))
        self.terrain_param = nn.Linear(int(hidden), 2)
        self.extra = (nn.Linear(int(hidden), self.k * self.n_extra)
                      if self.n_extra > 0 else None)

    def forward(self, ctx_vec):
        h = self.trunk(ctx_vec)
        out = {"root_vel": self.root_vel(h).view(-1, self.k, 3),
               "joint_vel": self.joint_vel(h).view(-1, self.k, 29),
               "terrain_type": self.terrain_type(h),
               "terrain_param": self.terrain_param(h),
               "extra": (None if self.extra is None
                         else self.extra(h).view(-1, self.k, self.n_extra))}
        return out


def priv_targets(state, ctx_feat, k, extra_frame=None, fps=FPS):
    """未来 K 步特权监督量（**全部可从语料导出**，见模块 docstring）。

    root_vel[j] = (trans[j+1]-trans[j])*fps、joint_vel[j] = (jp[j+1]-jp[j])*fps
    （j=0..K-1，即帧 1..K 的差分，全是「未来」量）；terrain_type/param 取自 ctx；
    extra_frame (b,WIN,D) 真特权列 → 目标 = 帧 [1, K+1)。
    """
    trans = state[:, :, TRANS_SLICE]
    jp = state[:, :, JP_SLICE]
    tgt = {"root_vel": (trans[:, 1:k + 1] - trans[:, :k]) * float(fps),
           "joint_vel": (jp[:, 1:k + 1] - jp[:, :k]) * float(fps),
           "terrain_type": ctx_feat[:, CTX_TERRAIN_ONEHOT].argmax(dim=-1),
           "terrain_param": ctx_feat[:, CTX_TERRAIN_PARAMS],
           "extra": None}
    if extra_frame is not None and int(extra_frame.shape[-1]) > 0:
        tgt["extra"] = extra_frame[:, 1:k + 1, :]
    return tgt


def priv_loss(out, tgt):
    """特权辅助损失 = MSE(根速)+MSE(关节速)+CE(地形型)+MSE(地形参)（+MSE 真特权列）。"""
    loss = (F.mse_loss(out["root_vel"].float(), tgt["root_vel"])
            + F.mse_loss(out["joint_vel"].float(), tgt["joint_vel"])
            + F.cross_entropy(out["terrain_type"].float(), tgt["terrain_type"])
            + F.mse_loss(out["terrain_param"].float(), tgt["terrain_param"]))
    if out.get("extra") is not None and tgt.get("extra") is not None:
        loss = loss + F.mse_loss(out["extra"].float(), tgt["extra"].float())
    return loss


# ------------------------------------------------------------------ 模型：主干
class AuthorV1Transformer(nn.Module):
    """统一窗 → 每帧 64×V logits（+ 可选 chunk 头 / 特权头）。

    主干与 v0 AuthorV0Transformer 逐字同构（consumer_stub 同构放大：双前缀上下文帧 +
    可学位置表 + causal mask + 逐层梯度检查点）；增量 = ctx 表征池化出口（两个前缀帧
    均值）+ chunk 头 + 特权头，由 variant 开关决定挂哪个（anchor 两个都不挂 = 纯放大）。
    """

    def __init__(self, vocab, state_dim=STATE_DIM, d_model=1280, n_layer=18,
                 n_head=10, ffn=2560, grad_ckpt=True,
                 use_chunk=True, use_priv=False, chunk_k=8, chunk_prefix=8,
                 chunk_d_model=512, chunk_layers=2, chunk_n_head=8, chunk_ffn=1024,
                 priv_hidden=256, priv_extra=0):
        super().__init__()
        self.vocab = int(vocab)
        self.n_ctx = 2
        self.grad_ckpt = bool(grad_ckpt)
        self.use_chunk = bool(use_chunk)
        self.use_priv = bool(use_priv)
        self.chunk_k = int(chunk_k)
        self.chunk_prefix = int(chunk_prefix)
        self.priv_extra = int(priv_extra)
        if not (0 <= self.chunk_prefix) or self.chunk_prefix + self.chunk_k > WIN:
            raise ValueError(f"chunk_prefix {self.chunk_prefix} + chunk_k "
                             f"{self.chunk_k} 超窗长 {WIN}")
        self.tok_emb = nn.Embedding(self.vocab, d_model)
        self.intent_emb = nn.Embedding(self.vocab, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.cmd_proj = nn.Linear(CTX_DIM, d_model)
        self.pos = nn.Parameter(torch.zeros(1, WIN + self.n_ctx, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, ffn, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layer),
                                             enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, TOKEN_DIM * self.vocab)
        self.chunk_head = (ChunkHead(self.vocab, d_model, chunk_d_model,
                                     chunk_layers, chunk_n_head, chunk_ffn)
                           if self.use_chunk else None)
        self.priv_head = (PrivilegedHead(d_model, self.chunk_k, priv_hidden,
                                         n_extra=self.priv_extra)
                          if self.use_priv else None)

    def encode(self, tokens_idx, state, intent_idx, ctx_feat):
        """主干前向 → (逐帧隐藏 (b,t,d), ctx 表征 (b,d))。"""
        b, t, _k = tokens_idx.shape
        # 帧嵌入 = 状态投影 + 64 个 token 维码嵌入均值（维间共享表）+ intent 同构
        x = self.state_proj(state)
        x = x + self.tok_emb(tokens_idx).mean(dim=2)
        x = x + self.intent_emb(intent_idx).mean(dim=2)
        ctx = self.cmd_proj(ctx_feat).unsqueeze(1).expand(b, self.n_ctx, -1)
        x = torch.cat([ctx, x], dim=1) + self.pos[:, :t + self.n_ctx]
        total = t + self.n_ctx
        mask = torch.triu(torch.ones(total, total, dtype=torch.bool,
                                     device=x.device), diagonal=1)
        for lyr in self.encoder.layers:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x = torch_checkpoint(lyr, x, mask, use_reentrant=False)
            else:
                x = lyr(x, mask)
        h = self.ln_f(x)
        return h[:, self.n_ctx:], h[:, :self.n_ctx].mean(dim=1)

    def forward(self, batch):
        """batch dict → {"logits","ctx_repr"[,"chunk"][,"priv"]}。"""
        frames, ctx_repr = self.encode(batch["tok"], batch["state"],
                                       batch["intent"], batch["ctx"])
        out = {"logits": self.head(frames).view(
            batch["tok"].shape[0], batch["tok"].shape[1], TOKEN_DIM, self.vocab),
            "ctx_repr": ctx_repr, "chunk": None, "priv": None}
        if self.chunk_head is not None:
            p, k = self.chunk_prefix, self.chunk_k
            out["chunk"] = self.chunk_head.rollout(
                ctx_repr, batch["tok"][:, :p], batch["tok"][:, p:p + k])
        if self.priv_head is not None:
            out["priv"] = self.priv_head(ctx_repr)
        return out


def model_kwargs_from_args(args, variant, extra_width=0):
    """CLI args + variant → AuthorV1Transformer kwargs（ckpt 重建同用；
    extra_width=真特权列总宽，须由数据侧 peek_extra_width 实测传入）。"""
    use_chunk, use_priv = VARIANT_HEADS[variant]
    return {"d_model": int(args.d_model), "n_layer": int(args.n_layer),
            "n_head": int(args.n_head), "ffn": int(args.ffn),
            "grad_ckpt": bool(args.grad_ckpt),
            "use_chunk": use_chunk, "use_priv": use_priv,
            "chunk_k": int(args.chunk_k), "chunk_prefix": int(args.chunk_prefix),
            "chunk_d_model": int(args.chunk_d_model),
            "chunk_layers": int(args.chunk_layers),
            "chunk_n_head": int(args.chunk_n_head), "chunk_ffn": int(args.chunk_ffn),
            "priv_hidden": int(args.priv_hidden),
            "priv_extra": int(extra_width)}


def build_model(vocab, args, variant=None, extra_width=0):
    return AuthorV1Transformer(vocab, **model_kwargs_from_args(
        args, variant or args.variant, extra_width))


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def variant_param_table(args, vocab, max_params, extra_width=0):
    """四 variant 参数量实测表（逐个建-删，峰值显存只按单模型算）。"""
    rows = {}
    for v in ("anchor", "chunk", "priv", "both"):
        m = build_model(vocab, args, variant=v, extra_width=extra_width)
        n = count_params(m)
        rows[v] = {"n_params": n, "n_params_m": round(n / 1e6, 3),
                   "budget_ok": bool(n <= int(max_params)),
                   "prereg_name": VARIANT_PREREG_NAME[v]}
        del m
    return rows


# ------------------------------------------------------------------ 推理接口（eval 侧调用）
def make_chunk_cache(ctx_feat, prefix_idx=None):
    """滑窗推理缓存：ctx 表征入口 + **已执行（clean）前缀段**。

    prefix_idx = 上一 chunk 末尾**已执行段**的码索引 (b, P, 64)；本 chunk 从该干净
    前缀继续自回归生成（ω-0 RTC 语义：前缀来自实际执行，不用模型自己的预测）。
    """
    return {"ctx": torch.as_tensor(ctx_feat), "frames": (
        None if prefix_idx is None else torch.as_tensor(prefix_idx).long())}


def extend_executed(cache, executed_idx):
    """把**实际执行**的段追加进 clean 前缀（推理环回写；不原地改 cache）。"""
    ex = torch.as_tensor(executed_idx).long()
    prev = cache.get("frames")
    frames = ex if prev is None or prev.shape[1] == 0 else \
        torch.cat([prev, ex], dim=1)
    return {"ctx": cache["ctx"], "frames": frames}


@torch.no_grad()
def chunk_generate(model, ctx_vec, cache=None, k=None):
    """推理期自回归生成 K 帧（硬 argmax 回灌）+ clean-prefix 缓存。

    入参：ctx_vec (b,d_model) = **主干 ctx 表征**（model.encode 的第二返回值，
    与训练期 chunk 头入口同口径；流式推理每个滑窗重算一次）；
    --intent-pin-max 的钉档发生在 encode 之前的 13 维原始 ctx 上
    （pin_ctx_max_band），不在本函数内。cache = make_chunk_cache 产物（可带
    已执行前缀）；k 缺省用模型 chunk_k。返回 (生成码 (b,K,64) long, 新 cache)
    —— 新 cache 的 frames 仍只含**已执行段**（生成段待执行侧回写：extend_executed）。
    """
    if model.chunk_head is None:
        raise RuntimeError("该 variant 无 chunk 头（--variant chunk|both 才有）")
    model.eval()
    ctx = torch.as_tensor(ctx_vec, dtype=torch.float32,
                          device=next(model.parameters()).device)
    if ctx.shape[-1] != model.cmd_proj.out_features:
        raise ValueError(
            f"ctx_vec 末维 {int(ctx.shape[-1])} ≠ d_model "
            f"{int(model.cmd_proj.out_features)}：应传 model.encode 的 ctx 表征"
            f"（训练期 chunk 头同口径），不是 {CTX_DIM} 维原始 ctx")
    if cache is None:
        cache = {"ctx": None, "frames": None}
    prev = cache.get("frames")
    prev = None if prev is None else prev.to(ctx.device)
    k = int(model.chunk_k if k is None else k)
    gen = []
    for _ in range(k):
        lg = model.chunk_head.logits_next(ctx, prev)      # (b,64,V)
        idx = lg.argmax(dim=-1)                            # (b,64)
        gen.append(idx)
        prev = (idx.unsqueeze(1) if prev is None or prev.shape[1] == 0
                else torch.cat([prev, idx.unsqueeze(1)], dim=1))
    return torch.stack(gen, dim=1), {"ctx": cache["ctx"], "frames": cache.get("frames")}


def load_author_v1(ckpt_path, device="cpu"):
    """eval 侧加载（返回 (model, model_cfg)）；ckpt 是自产文件故 weights_only=False。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt["model_cfg"])
    kw = {k: cfg[k] for k in MODEL_KWARG_KEYS}
    model = AuthorV1Transformer(cfg["vocab"], **kw)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


# ------------------------------------------------------------------ 训练件
def resolve_amp(amp_arg, device):
    """→ (autocast dtype | None, 是否需要 GradScaler)。cpu/off 一律回退 fp32。"""
    if amp_arg == "off" or device.type != "cuda":
        return None, False
    if amp_arg == "bf16":
        return torch.bfloat16, False
    if amp_arg == "fp16":
        return torch.float16, True
    # auto：bf16 优先，否则 fp16 + GradScaler
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def make_lr_lambda(total_steps, warmup_steps, base_lr, min_lr):
    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        denom = max(1, total_steps - warmup_steps)
        prog = min(1.0, max(0.0, (step - warmup_steps) / denom))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return (min_lr + (base_lr - min_lr) * cos) / base_lr
    return fn


def rollout_weight_at(step, target, ramp_steps):
    """rollout 权重线性 ramp（0 → target）：step 步已完成时的有效权重。"""
    if ramp_steps <= 0:
        return float(target)
    return float(target) * min(1.0, max(0.0, float(step) / float(ramp_steps)))


def _autocast_ctx(device, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def step_losses(model, batch, vocab, device, amp_dtype, rollout_weight=0.0,
                priv_weight=0.0):
    """一步前向 → (总 loss, top1_acc, logits_shape, 分项 dict)。

    总 loss = 逐帧 teacher-forcing CE（199 目标位，主任务）
              + rollout_weight × chunk 内自回归 rollout CE（chunk 头存在时）
              + priv_weight × 特权辅助损失（特权头存在时）；
    CE 在 fp32 上算（bf16 稳定性）。
    """
    with _autocast_ctx(device, amp_dtype):
        out = model(batch)
    lg = out["logits"][:, :-1].float()
    tgt = batch["tok"][:, 1:]
    main = F.cross_entropy(lg.reshape(-1, vocab), tgt.reshape(-1))
    with torch.no_grad():
        acc = float((lg.argmax(-1) == tgt).float().mean())
    parts = {"main": float(main.detach()), "rollout": None, "priv": None}
    total = main
    if out.get("chunk") is not None:
        r = out["chunk"]["loss_ce"]
        parts["rollout"] = float(r.detach())
        total = total + float(rollout_weight) * r
    if out.get("priv") is not None:
        p = priv_loss(out["priv"], priv_targets(batch["state"], batch["ctx"],
                                                model.chunk_k,
                                                batch.get("priv_extra_frame")))
        parts["priv"] = float(p.detach())
        total = total + float(priv_weight) * p
    return total, acc, list(out["logits"].shape), parts


@torch.no_grad()
def evaluate(model, loader, vocab, device, amp_dtype):
    """val 主指标（按 token 位加权平均）+ 分项诊断（rollout/priv 头存在时）。"""
    model.eval()
    tot = {"loss": 0.0, "hit": 0.0, "n": 0, "rollout": 0.0, "rollout_n": 0,
           "priv": 0.0, "priv_n": 0}
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with _autocast_ctx(device, amp_dtype):
            out = model(batch)
        lg = out["logits"][:, :-1].float()
        tgt = batch["tok"][:, 1:]
        loss = F.cross_entropy(lg.reshape(-1, vocab), tgt.reshape(-1),
                               reduction="sum")
        tot["loss"] += float(loss)
        tot["hit"] += float((lg.argmax(-1) == tgt).sum())
        tot["n"] += int(tgt.numel())
        if out.get("chunk") is not None:
            tot["rollout"] += float(out["chunk"]["loss_ce"].detach())
            tot["rollout_n"] += 1
        if out.get("priv") is not None:
            tot["priv"] += float(priv_loss(out["priv"], priv_targets(
                batch["state"], batch["ctx"], model.chunk_k,
                batch.get("priv_extra_frame"))).detach())
            tot["priv_n"] += 1
    model.train()
    if tot["n"] == 0:
        return {"val_loss": float("nan"), "val_top1": float("nan"),
                "val_rollout": None, "val_priv": None}
    return {"val_loss": tot["loss"] / tot["n"], "val_top1": tot["hit"] / tot["n"],
            "val_rollout": (tot["rollout"] / tot["rollout_n"]
                            if tot["rollout_n"] else None),
            "val_priv": (tot["priv"] / tot["priv_n"] if tot["priv_n"] else None)}


def save_ckpt(path, model, step, args, corpus, extra=None):
    variant = args.variant
    # priv_extra 取**实际模型宽度**（--priv-extra-cols 经数据侧 peek 实测后落到
    # model.priv_extra）；不补传会记 0，load_author_v1 重建时 priv_head.extra
    # 缺表 → unexpected keys 加载失败（带真特权列的 ckpt 不可回读）。
    cfg = {"vocab": corpus["vocab"], "code_min": corpus["code_min"],
           "variant": variant, "state_dim": STATE_DIM, "win": WIN,
           "token_dim": TOKEN_DIM, "exp": "D062",
           **model_kwargs_from_args(args, variant,
                                    extra_width=int(getattr(model, "priv_extra", 0)))}
    torch.save({
        "format": "d062.v1", "step": int(step), "variant": variant,
        "model": model.state_dict(), "model_cfg": cfg,
        "priv_extra_cols": parse_extra_cols(args.priv_extra_cols),
        **(extra or {}),
    }, path)


def build_adafactor(params, lr):
    """Adafactor（裁定口径：固定 lr、无 relative_step/无 scale_parameter/无 warmup_init）。

    torch≥2.12 重写 Adafactor：新实现本身就是固定-lr 语义，旧三参已从签名移除；
    老 torch 仍显式传三参。返回 (optimizer, variant) 供 meta 记录。
    """
    import inspect
    sig = inspect.signature(torch.optim.Adafactor.__init__).parameters
    if "relative_step" in sig:
        return torch.optim.Adafactor(
            params, lr=float(lr), relative_step=False, scale_parameter=False,
            warmup_init=False), "legacy(explicit fixed-lr flags)"
    return torch.optim.Adafactor(params, lr=float(lr)), "torch>=2.12 fixed-lr rewrite"


def converge_guard(eval_points, warmup_steps, tail_n=GATE_TAIL_N,
                   rel=GATE_TAIL_REL):
    """收敛判据（=门②口径）叠加两条触发守卫，返回判读明细 dict。

    判据本体：尾部 tail_n 个 eval 点相对变化 < rel 且尾均 < 全序列首点。
    两条守卫（D063 1B 触发，防 D062 首轮四臂误停：min_steps=warmup=2000 →
    warmup 一结束的首个 eval 点即取得触发资格，停时 LR 仍在峰值、top1 单调爬升）：
      ① 相位守卫 tail_in_descent：尾部滑窗**全部**落在 warmup 后的余弦下降段
         （窗口首 eval 点 step > warmup_steps）——把触发资格推离 warmup 终点。
      ② 降势守卫 net_flat：窗口须整体走平（|末点-首点| ≤ 单步噪声带），拦
         「缓慢匀速下降被 max-min<3% 误当平台」。

    返回 {"ok": bool, ...诊断字段}；ok=True 才可触发停止。诊断字段供 meta/
    train.log 追溯（warmup_done / tail_in_descent / tail_flat / net_flat /
    net_drop / noise_band / tail_rel_change / first_tail_step / …）。
    """
    pts = [(e.get("step"), float(e["val_loss"])) for e in eval_points
           if e.get("val_loss") is not None
           and math.isfinite(float(e["val_loss"]))]
    g = {"ok": False, "warmup_done": False, "tail_in_descent": False,
         "tail_flat": False, "net_flat": False, "net_drop": None,
         "noise_band": None, "tail_rel_change": None,
         "first_tail_step": None, "last_tail_step": None,
         "warmup_steps": int(warmup_steps), "reason": ""}
    if len(pts) < int(tail_n):
        g["reason"] = "eval 点不足（%d<%d）" % (len(pts), int(tail_n))
        return g
    tail = pts[-int(tail_n):]
    vals = [v for _s, v in tail]
    steps = [s for s, _v in tail]
    g["first_tail_step"], g["last_tail_step"] = steps[0], steps[-1]
    if any(s is None for s in steps):
        g["reason"] = "eval 点缺 step（相位守卫无法判读）"
        return g
    if min(vals) <= 0:
        g["reason"] = "val_loss 非正"
        return g
    # ① 相位守卫：尾窗全部越过 warmup 终点（严格大于，排除仍在 LR 峰值的边界点）
    g["warmup_done"] = bool(steps[-1] >= int(warmup_steps))
    g["tail_in_descent"] = bool(steps[0] > int(warmup_steps))
    # 判据本体（尾部相对变化 + 尾均 < 首点）
    g["tail_rel_change"] = (max(vals) - min(vals)) / min(vals)
    g["tail_flat"] = bool(g["tail_rel_change"] < float(rel)
                          and (sum(vals) / len(vals)) < pts[0][1])
    # ② 降势守卫：|末点-首点| 落在单步噪声带（尾窗相邻差分均值）内才算走平
    diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
    noise = sum(diffs) / len(diffs) if diffs else 0.0
    g["noise_band"] = noise
    g["net_drop"] = abs(vals[-1] - vals[0])
    g["net_flat"] = bool(g["net_drop"] <= noise)
    g["ok"] = bool(g["tail_flat"] and g["tail_in_descent"] and g["net_flat"])
    if not g["ok"]:
        if not g["tail_in_descent"]:
            g["reason"] = ("相位守卫：尾窗首点 step=%s 未越过 warmup=%d"
                           % (steps[0], int(warmup_steps)))
        elif not g["net_flat"]:
            g["reason"] = ("降势守卫：净降 %.5f > 单步噪声带 %.5f"
                           "（匀速下降非平台）" % (g["net_drop"], noise))
        else:
            g["reason"] = "判据本体未满足（尾部未走平或尾均未低于首点）"
    return g


def tail_converged(eval_points, tail_n=GATE_TAIL_N, rel=GATE_TAIL_REL,
                   warmup_steps=0):
    """收敛判据布尔包装（含相位/降势守卫；warmup_steps=0 时相位守卫恒过）。

    训练循环用它决定**步数预算**（不硬编码步数）：收敛即停；判读明细见
    converge_guard（触发时写入 meta 的 converge_guard 诊断字段）。
    """
    return converge_guard(eval_points, warmup_steps=warmup_steps,
                          tail_n=tail_n, rel=rel)["ok"]


def train_model(args, corpus, model, device, logger, train_pool, val_pool,
                judge_after=True):
    """训练循环：AMP + Adafactor + warmup/cosine + 周期 eval/ckpt + 早停 + [SPEED]
    + **收敛判据停止**（尾部 eval 变化 <3% 即停；epochs/--max-steps 仅为上界）。

    返回 hist dict；judge_after=True 时训练结束自动判门（打印 D062_GATE_* + gate.json）。
    """
    vocab = corpus["vocab"]
    variant = args.variant
    use_chunk, use_priv = VARIANT_HEADS[variant]
    train_ld, val_ld = make_loaders(train_pool, val_pool, corpus, args, device)
    amp_dtype, use_scaler = resolve_amp(args.amp, device)
    amp_record = {"arg": args.amp, "device": device.type,
                  "dtype": None if amp_dtype is None else str(amp_dtype),
                  "grad_scaler": bool(use_scaler),
                  "note": "auto=cuda bf16（不支持则 fp16+GradScaler），cpu/off 回退 fp32"}
    scaler = None
    if use_scaler:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):  # 旧版 torch 回退
            scaler = torch.cuda.amp.GradScaler()

    steps_per_epoch = max(1, math.ceil(len(train_pool) / int(args.batch_windows)))
    epoch_steps = steps_per_epoch * int(args.epochs)
    total_steps = (min(epoch_steps, int(args.max_steps))
                   if args.max_steps else epoch_steps)
    eval_every = max(1, int(round(steps_per_epoch * float(args.eval_every_frac))))
    ckpt_every = max(1, int(round(steps_per_epoch * float(args.ckpt_every_frac))))
    ramp_steps = (int(args.rollout_ramp_steps) if args.rollout_ramp_steps
                  else int(args.warmup_steps))
    min_steps = max(int(args.warmup_steps), GATE_TAIL_N * eval_every)
    logger.info("[train] variant=%s（%s）device=%s amp=%s grad_ckpt=%s | "
                "train %d 窗 / val %d 窗 | batch %d → %d 步/epoch × %d epoch"
                "（上界 %d 步；收敛判据=尾部 %d eval 点相对变化 <%.0f%% 且过相位/"
                "降势守卫即停，min_steps=%d）| eval 每 %d 步 / ckpt 每 %d 步 | chunk_k=%d "
                "prefix=%d rollout_w=%.3f ramp=%d | priv_w=%.3f extra_cols=%s",
                variant, VARIANT_PREREG_NAME[variant], device, amp_dtype,
                model.grad_ckpt, len(train_pool), len(val_pool),
                args.batch_windows, steps_per_epoch, args.epochs, total_steps,
                GATE_TAIL_N, GATE_TAIL_REL * 100, min_steps, eval_every,
                ckpt_every, args.chunk_k, args.chunk_prefix, args.rollout_weight,
                ramp_steps, args.priv_weight, parse_extra_cols(args.priv_extra_cols))

    opt, adafactor_variant = build_adafactor(model.parameters(), args.lr)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, [make_lr_lambda(total_steps, int(args.warmup_steps),
                             float(args.lr), MIN_LR)])

    hist = {"total_steps": total_steps, "steps_done": 0, "train_losses": [],
            "rollout_losses": [], "rollout_weights": [], "priv_losses": [],
            "eval_points": [], "curves": [], "early_stop": False,
            "completed_epochs": False, "nan_seen": False, "logits_shape": None,
            "steps_per_epoch": steps_per_epoch, "eval_every": eval_every,
            "ckpt_every": ckpt_every, "best_val_loss": None,
            "adafactor_variant": adafactor_variant, "amp": amp_record,
            "variant": variant, "prereg_name": VARIANT_PREREG_NAME[variant],
            "use_chunk": use_chunk, "use_priv": use_priv,
            "chunk_k": int(args.chunk_k), "chunk_prefix": int(args.chunk_prefix),
            "priv_extra_dim": int(model.priv_extra), "converged": False,
            "converge_stop": bool(args.converge_stop),
            "converge_guard": None,   # 最新一次 eval 的收敛守卫判读明细（可追溯）
            "steps_budget_source": ("收敛判据（尾部 %d eval 点相对变化<%.0f%% 且过"
                                    "相位/降势守卫即停）；"
                                    "epochs=%d/--max-steps=%s 仅为上界"
                                    % (GATE_TAIL_N, GATE_TAIL_REL * 100,
                                       args.epochs, args.max_steps)),
            "rollout_weight_target": float(args.rollout_weight),
            "rollout_ramp_steps": ramp_steps,
            "priv_weight": float(args.priv_weight)}
    best_val, bad_evals = None, 0
    recent_losses = []
    t0 = time.time()
    last_speed_t, windows_seen = t0, 0
    model.train()
    stop = False
    for _epoch in range(int(args.epochs)):
        if stop:
            break
        for batch in train_ld:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if hist["steps_done"] == 0:
                assert int(batch["tok"].min()) >= 0 and \
                    int(batch["tok"].max()) < vocab, \
                    "码索引越界（code_min/vocab 与实际码域不符）"
            rw = rollout_weight_at(hist["steps_done"], args.rollout_weight,
                                   ramp_steps)
            loss, _acc, shape, parts = step_losses(
                model, batch, vocab, device, amp_dtype, rollout_weight=rw,
                priv_weight=args.priv_weight)
            if hist["logits_shape"] is None:
                hist["logits_shape"] = shape
            floss = float(loss.detach())
            if not math.isfinite(floss):
                hist["nan_seen"] = True
                logger.error("[train] step %d loss 非有限（%s）→ 停（门①判负）",
                             hist["steps_done"] + 1, floss)
                stop = True
                break
            opt.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            sch.step()
            hist["steps_done"] += 1
            hist["train_losses"].append(floss)
            recent_losses.append(floss)
            if use_chunk:
                hist["rollout_losses"].append(parts["rollout"])
                hist["rollout_weights"].append(rw)
            if use_priv:
                hist["priv_losses"].append(parts["priv"])
            windows_seen += int(batch["tok"].shape[0])

            if hist["steps_done"] % SPEED_EVERY == 0:
                now = time.time()
                dt = max(1e-9, now - last_speed_t)
                avg_s = (now - t0) / hist["steps_done"]
                eta_m = avg_s * (total_steps - hist["steps_done"]) / 60.0
                logger.info("[SPEED] step %d/%d windows=%d rate=%.1f w/s "
                            "elapsed=%.0fs ETA=%.1fm",
                            hist["steps_done"], total_steps, windows_seen,
                            windows_seen / dt, now - t0, eta_m)
                last_speed_t = now

            if hist["steps_done"] % eval_every == 0 or \
                    hist["steps_done"] == total_steps:
                ev = evaluate(model, val_ld, vocab, device, amp_dtype)
                ev["step"] = hist["steps_done"]
                hist["eval_points"].append(ev)
                hist["curves"].append([
                    hist["steps_done"],
                    float(np.mean(recent_losses)) if recent_losses else float("nan"),
                    float(ev["val_loss"]), float(ev["val_top1"]),
                    float(sch.get_last_lr()[0]), round(time.time() - t0, 2)])
                recent_losses = []
                logger.info("[eval] step %d/%d val_loss=%.4f val_top1=%.4f "
                            "rollout=%s priv=%s (lr=%.2e)", hist["steps_done"],
                            total_steps, ev["val_loss"], ev["val_top1"],
                            _fmt(ev["val_rollout"]), _fmt(ev["val_priv"]),
                            sch.get_last_lr()[0])
                # 早停：严格小于最优才算改善，连续 patience 个 eval 点无改善即停
                if best_val is None or ev["val_loss"] < best_val:
                    best_val = float(ev["val_loss"])
                    hist["best_val_loss"] = best_val
                    bad_evals = 0
                else:
                    bad_evals += 1
                    if bad_evals >= int(args.patience):
                        hist["early_stop"] = True
                        logger.info("[early_stop] 连续 %d 个 eval 点无改善"
                                    "（best=%.4f）→ 停", bad_evals, best_val)
                        stop = True
                # 收敛判据（步数预算来源，非硬编码）：尾部变化 <3% 且过两条守卫即停
                cg = converge_guard(hist["eval_points"], int(args.warmup_steps))
                hist["converge_guard"] = cg  # 逐 eval 点覆盖存最新判读（可追溯）
                if args.converge_stop and not stop and \
                        hist["steps_done"] >= min_steps and cg["ok"]:
                    hist["converged"] = True
                    logger.info("[converge] 尾部 %d eval 点相对变化 %.2f%%<%.0f%% "
                                "（val_loss=%.4f）→ 步数预算达成，停（非硬编码步数）"
                                "｜守卫 warmup_done=%s tail_in_descent=%s "
                                "net_flat=%s（净降%.5f≤噪声带%.5f）",
                                GATE_TAIL_N, (cg["tail_rel_change"] or 0.0) * 100,
                                GATE_TAIL_REL * 100, ev["val_loss"],
                                cg["warmup_done"], cg["tail_in_descent"],
                                cg["net_flat"], cg["net_drop"] or 0.0,
                                cg["noise_band"] or 0.0)
                    stop = True
                elif args.converge_stop and not stop and \
                        hist["steps_done"] >= min_steps and cg["tail_flat"]:
                    # 近失（本体判平但守卫拦下）：只在此时打诊断，避免逐点刷屏
                    logger.info("[converge] 近失拦下（%s）｜tail_in_descent=%s "
                                "net_flat=%s net_drop=%s noise_band=%s",
                                cg["reason"], cg["tail_in_descent"],
                                cg["net_flat"], cg["net_drop"], cg["noise_band"])
            if hist["steps_done"] % ckpt_every == 0:
                p = os.path.join(args.out_dir,
                                 f"ckpt_step{hist['steps_done']:07d}.pt")
                save_ckpt(p, model, hist["steps_done"], args, corpus)
                logger.info("[ckpt] %s", p)
            if stop or hist["steps_done"] >= total_steps:
                break
    hist["completed_epochs"] = (not hist["early_stop"]) and (not hist["nan_seen"]) \
        and (not hist["converged"]) and hist["steps_done"] >= total_steps
    if not hist["eval_points"]:  # 极短跑兜底：至少一个 eval 点
        ev = evaluate(model, val_ld, vocab, device, amp_dtype)
        ev["step"] = hist["steps_done"]
        hist["eval_points"].append(ev)
        hist["curves"].append([hist["steps_done"],
                               float(np.mean(hist["train_losses"][-eval_every:]))
                               if hist["train_losses"] else float("nan"),
                               float(ev["val_loss"]), float(ev["val_top1"]),
                               float(sch.get_last_lr()[0]), round(time.time() - t0, 2)])
    hist["final_val_loss"] = hist["eval_points"][-1]["val_loss"]
    hist["final_val_top1"] = hist["eval_points"][-1]["val_top1"]
    hist["final_val_rollout"] = hist["eval_points"][-1].get("val_rollout")
    hist["final_val_priv"] = hist["eval_points"][-1].get("val_priv")
    hist["elapsed_s"] = round(time.time() - t0, 2)
    final_path = os.path.join(args.out_dir, "ckpt_final.pt")
    save_ckpt(final_path, model, hist["steps_done"], args, corpus, extra={
        "final_val_loss": hist["final_val_loss"],
        "final_val_top1": hist["final_val_top1"]})
    logger.info("[ckpt] final %s（%d 步，%.1fs）", final_path, hist["steps_done"],
                hist["elapsed_s"])
    if judge_after:
        gate = judge_gate(hist, corpus["leak_audit"], logger=logger)
        write_json(os.path.join(args.out_dir, "gate.json"), gate)
    return hist


def _fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


# ------------------------------------------------------------------ 门判读（沿 D061a 四条）
def judge_gate(hist, leak_audit, logger=None):
    """D062 四条门自动判读（判据数字冻结=沿 D061a，见模块常量）。"""
    def log(msg, *a):
        if logger is not None:
            logger.info(msg, *a)
        else:
            print(msg % a if a else msg)

    tl = hist.get("train_losses") or []
    ev = hist.get("eval_points") or []
    finite = all(math.isfinite(x) for x in tl) and all(
        math.isfinite(e["val_loss"]) and math.isfinite(e["val_top1"]) for e in ev)
    c1 = bool(finite and (hist.get("completed_epochs") or hist.get("early_stop")
                          or hist.get("converged")))

    tail = [e["val_loss"] for e in ev[-GATE_TAIL_N:]]
    if ev and tail and min(tail) > 0:
        rel = (max(tail) - min(tail)) / min(tail)
        c2 = bool(rel < GATE_TAIL_REL
                  and (sum(tail) / len(tail)) < ev[0]["val_loss"])
    else:
        rel = float("inf")
        c2 = False
    last_top1 = ev[-1]["val_top1"] if ev else float("nan")
    c3 = bool(ev and math.isfinite(last_top1) and last_top1 >= GATE_TOP1_MIN)
    entered = int(leak_audit.get("heldout_entered", 0))
    c4 = bool(entered == 0)

    items = {
        "c1_finite_complete": {
            "rule": "无 NaN/Inf 且完成 epoch 预算（或 early_stop / 收敛判据停止）",
            "pass": c1, "loss_finite": finite,
            "completed_epochs": bool(hist.get("completed_epochs")),
            "early_stop": bool(hist.get("early_stop")),
            "converged": bool(hist.get("converged")),
            "nan_seen": bool(hist.get("nan_seen"))},
        "c2_val_loss_converged": {
            "rule": f"尾部 {GATE_TAIL_N} eval 点 (max-min)/min<{GATE_TAIL_REL} "
                    f"且尾部均值 < 首 eval 点",
            "pass": c2, "tail_rel_change": rel if math.isfinite(rel) else None,
            "tail_mean": (sum(tail) / len(tail)) if tail else None,
            "first_val_loss": ev[0]["val_loss"] if ev else None,
            "n_eval_points": len(ev)},
        "c3_val_top1": {
            "rule": f"val top1 ≥ {GATE_TOP1_MIN}（5×均匀基线 1/29 从严）",
            "pass": c3, "final_val_top1": last_top1},
        "c4_leak_audit": {
            "rule": "leak_audit heldout_entered==0（训练索引 heldout 零进入）",
            "pass": c4, "heldout_entered": entered, "leak_audit": leak_audit},
    }
    ok = c1 and c2 and c3 and c4
    marker = "D062_GATE_PASS" if ok else "D062_GATE_FAIL"
    log(f"[gate] {marker}")
    for k, v in items.items():
        log("[gate]   %s: %s（%s）", k, "PASS" if v["pass"] else "FAIL", v["rule"])
    return {"marker": marker, "pass": ok, "criteria": items,
            "variant": hist.get("variant"),
            "prereg_name": hist.get("prereg_name"),
            "final_val_loss": hist.get("final_val_loss"),
            "final_val_top1": hist.get("final_val_top1"),
            "judged_at": now_str()}


# ------------------------------------------------------------------ meta
def write_run_meta(out_dir, args, corpus, hist, n_params, gate, device,
                   param_table=None):
    meta = {
        "exp": "D062", "role": "作者 v1（≤300M）A3 联合升级训练（chunk 头+特权头）",
        "created_at": now_str(), "cwd": os.getcwd(),
        "identity": {   # 身份信封：git commit / 语料 md5 / variant / 参数量
            "script": os.path.abspath(__file__),
            "script_md5": md5_file(__file__),
            "git_rev": git_rev(),
            "hostname": socket.gethostname(),
            "torch": torch.__version__, "numpy": np.__version__,
            "variant": args.variant,
            "prereg_name": VARIANT_PREREG_NAME.get(args.variant),
            "n_params": n_params,
            "corpus_md5_per_source": corpus["md5s"],
            "corpus_md5_combined": corpus["corpus_md5_combined"],
        },
        "args": vars(args),
        "model": {"variant": args.variant, "use_chunk": hist.get("use_chunk"),
                  "use_priv": hist.get("use_priv"),
                  "d_model": args.d_model, "n_layer": args.n_layer,
                  "n_head": args.n_head, "ffn": args.ffn,
                  "n_params": n_params, "max_params": int(args.max_params),
                  "grad_ckpt": bool(args.grad_ckpt),
                  "vocab": corpus["vocab"],
                  "code_range": [corpus["code_min"], corpus["code_max"]],
                  "param_budget_ok": n_params <= int(args.max_params),
                  "chunk": {"k": int(args.chunk_k),
                            "prefix": int(args.chunk_prefix),
                            "d_model": int(args.chunk_d_model),
                            "n_layer": int(args.chunk_layers)},
                  "priv": {"hidden": int(args.priv_hidden),
                           "extra_cols": parse_extra_cols(args.priv_extra_cols),
                           "deploy_use": False},
                  "param_table_all_variants": param_table},
        "corpus": {"assembly": corpus["assembly"],
                   "source_stats": corpus["source_stats"],
                   "exclude_src4": bool(args.exclude_src4),
                   "limit_windows": args.limit_windows,
                   "speed_bands": corpus["speed_bands"]},
        "leak_audit": corpus["leak_audit"],
        "splits": {"train_windows": hist.get("n_train"),
                   "val_windows": hist.get("n_val"),
                   "val_frac": args.val_frac, "val_seed": args.val_seed},
        "intent_pin": {"intent_pin_max": bool(args.intent_pin_max),
                       "note": "仅推理期钉 ctx 命令槽到语料速度带最大档；训练不受影响",
                       "max_band_value": (corpus.get("speed_bands") or {}).get(
                           "max_band_value")},
        "curves": hist["curves"],
        "eval_points": hist["eval_points"],
        "summary": {k: hist.get(k) for k in (
            "steps_done", "total_steps", "early_stop", "completed_epochs",
            "converged", "converge_stop", "converge_guard", "steps_budget_source",
            "nan_seen",
            "logits_shape", "final_val_loss", "final_val_top1",
            "final_val_rollout", "final_val_priv", "best_val_loss", "elapsed_s",
            "amp", "adafactor_variant", "rollout_weight_target",
            "rollout_ramp_steps", "priv_weight", "priv_extra_dim")},
        "gate": None if gate is None else {"marker": gate["marker"],
                                           "pass": gate["pass"]},
        "elapsed_s": hist.get("elapsed_s"),
    }
    write_json(os.path.join(out_dir, "meta.json"), meta)
    return meta


# ------------------------------------------------------------------ selftest
def make_selftest_corpus(root, seed=0, n_train_per_src=12, n_held_per_src=1):
    """内存造 2 源合成微语料（码域 [-14,12]，terrain 两型）+ heldout 泄漏探针。

    与 v0 selftest **同一装配/源 manifest + npz 分片布局**（键名/字段/形状全同）——
    这本身就是「v1 与 v0 数据接口兼容（同一语料目录可读）」的活证据；额外多写一个
    逐帧真特权列 elevation_probe (WIN,3) 供 --priv-extra-cols 即插接口演练。
    """
    rng = np.random.default_rng(seed)
    code_min, code_max = -14, 12
    layout = (("srcA_test", "climbing_box", {"height": 0.5}),
              ("srcB_test", "rough_paper", {"noise": 0.04}))
    srcs = {}
    for sid, terr, tparams in layout:
        sdir = os.path.join(root, sid)
        os.makedirs(os.path.join(sdir, "npz"), exist_ok=True)
        n_train, n_held = int(n_train_per_src), int(n_held_per_src)
        n = n_train + n_held
        toks = rng.integers(code_min, code_max + 1,
                            (n, WIN, TOKEN_DIM)).astype(np.int16)
        state = rng.normal(0, 0.3, (n, WIN, STATE_DIM)).astype(np.float32)
        cmd = np.full((n, len(COMMAND_FIELDS)), SENTINEL, dtype=np.float32)
        ht = np.zeros((n, len(COMMAND_FIELDS)), dtype=bool)
        cmd[:, 0] = np.round(rng.uniform(0.0, 1.0, n), 3)
        ht[:, 0] = True
        for i in range(n):
            if rng.random() < 0.5:  # 部分窗带方向真值
                cmd[i, 1] = float(np.round(rng.uniform(-3.14, 3.14), 3))
                ht[i, 1] = True
        splits = ["train"] * n_train + ["heldout"] * n_held
        name = f"{sid}_all_000000-{n - 1:06d}.npz"
        path = os.path.join(sdir, "npz", name)
        np.savez_compressed(
            path, token_stream=toks, intent_tokens=toks.copy(), state=state,
            command=cmd, command_has_truth=ht,
            terrain_type=np.asarray([terr] * n),
            terrain_params_json=np.asarray(
                [json.dumps(tparams, sort_keys=True)] * n),
            # v1 新接口演练：逐帧真特权列（elevation/接触补录后同形即插）
            elevation_probe=rng.normal(0, 0.1, (n, WIN, 3)).astype(np.float32),
            meta_json=np.asarray([json.dumps(
                {"stem": f"{sid}_w{i:03d}", "token_scale": 16}, sort_keys=True)
                for i in range(n)]))
        recs = []
        for i in range(n):
            recs.append({
                "stem": f"{sid}_w{i:03d}", "shard": name, "index": i,
                "split": splits[i],
                "speedA": float(np.round(rng.uniform(0.02, 0.3), 3)),
                "target_vel": float(cmd[i, 0]),
                "movement_direction": float(cmd[i, 1]),
                "mode": SENTINEL, "height": SENTINEL,
                "has_truth": [COMMAND_FIELDS[j] for j in range(len(COMMAND_FIELDS))
                              if ht[i, j]],
                "terrain": terr, "terrain_params": dict(tparams),
                "intent_source": "self_corpus", "token_scale": 16})
        sman = {
            "exp": "D062_SELFTEST", "created_at": now_str(),
            "format": {"win": WIN, "token_dim": TOKEN_DIM, "state_dim": STATE_DIM,
                       "format_version": "d060.1"},
            "counts": {"windows": n, "windows_train": n_train,
                       "windows_heldout": n_held, "token_min": code_min,
                       "token_max": code_max, "token_scale": 16},
            "shards": [{"name": name, "path": os.path.abspath(path), "n": n,
                        "md5": md5_file(path), "bytes": os.path.getsize(path)}],
            "windows": recs}
        write_json(os.path.join(sdir, "manifest.json"), sman)
        srcs[sid] = {"source_id": sid, "label": sid,
                     "manifest_path": os.path.abspath(
                         os.path.join(sdir, "manifest.json")),
                     "counts": dict(sman["counts"])}
    apath = os.path.join(root, "assembly_manifest.json")
    write_json(apath, {"exp": "D062_SELFTEST", "created_at": now_str(),
                       "data_root": os.path.abspath(root), "sources": srcs})
    return apath


def run_selftest():
    """本机 CPU 硬验收：合成语料 → 四 variant 各跑数十步 → chunk/priv/推理/pin 全检。"""
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="d062_selftest_")
    try:
        asm = make_selftest_corpus(tmp, seed=0, n_train_per_src=12, n_held_per_src=1)
        out_dir = os.path.join(tmp, "out")
        base = ["--assembly-manifest", asm, "--data-root", tmp, "--out-dir", out_dir,
                "--d-model", "32", "--n-layer", "2", "--n-head", "2", "--ffn", "64",
                "--chunk-d-model", "32", "--chunk-layers", "1", "--chunk-n-head", "2",
                "--chunk-ffn", "64", "--chunk-k", "4", "--chunk-prefix", "2",
                "--batch-windows", "4", "--epochs", "4", "--eval-every-frac", "1.0",
                "--ckpt-every-frac", "1.0", "--lr", "0.02", "--warmup-steps", "1",
                "--rollout-weight", "0.1", "--rollout-ramp-steps", "2",
                "--priv-weight", "0.05", "--num-workers", "0", "--device", "cpu",
                "--seed", "0"]
        args0 = parse_args(base + ["--variant", "both"])
        logger = setup_logging(out_dir)
        logger.info("[selftest] 合成微语料（2 源×13 窗，含 2 个 heldout 泄漏探针）→ %s",
                    tmp)

        # ---- 数据接口（v0 同款布局；v0/v1 同一目录可读的活证据）
        corpus = build_corpus_index(args0.assembly_manifest, data_root=args0.data_root,
                                    exclude_src4=args0.exclude_src4,
                                    limit_windows=args0.limit_windows, logger=logger)
        la = corpus["leak_audit"]
        assert la["heldout_excluded"] == 2 and la["train_windows"] == 24 \
            and la["heldout_entered"] == 0, f"leak 断言/计数异常：{la}"
        assert corpus["vocab"] == 27 and corpus["code_min"] == -14, \
            f"码域契约异常：{corpus['code_min']}..{corpus['code_max']}"
        assert (WIN, TOKEN_DIM, STATE_DIM, CTX_DIM) == (200, 64, 36, 13), \
            "冻结格式常量被改（v0 兼容性破坏）"
        assert 199 == WIN - 1, "目标位数（199）契约异常"

        # ---- 速度带 + intent-pin-max（推理期钉档）
        bands = corpus["speed_bands"]
        assert bands["n"] == 24 and len(bands["bands"]) == 3 \
            and all(b is not None for b in bands["bands"]), f"速度带异常：{bands}"
        assert bands["max_band_value"] == max(bands["bands"]), "最大档≠上三分位中位"
        raw_ctx = corpus["entries"][0]["ctx"].copy()
        pinned = pin_ctx_max_band(raw_ctx, bands["max_band_value"])
        # 钉档值经 float32 落地，比较用 isclose（0.2725 的 float32 往返非逐位相等）
        assert np.isclose(float(pinned[0]), float(bands["max_band_value"]),
                          rtol=1e-6, atol=1e-7) and float(pinned[4]) == 1.0, \
            "intent-pin-max 未钉 ctx 命令槽/has_truth"
        assert not np.allclose(pinned, raw_ctx), "intent-pin-max 未改变 ctx（值重合）"
        assert np.allclose(raw_ctx, corpus["entries"][0]["ctx"]), \
            "pin_ctx_max_band 原地改了输入（训练侧 ctx 被污染）"
        assert np.allclose(infer_ctx(corpus, raw_ctx, False), raw_ctx) and \
            np.isclose(float(infer_ctx(corpus, raw_ctx, True)[0]),
                       float(bands["max_band_value"]), rtol=1e-6, atol=1e-7), \
            "infer_ctx 的 intent-pin 分支异常"
        logger.info("[selftest] 速度带三档 %s / 最大档 %s（intent-pin-max 钉档验证通过）",
                    bands["bands"], bands["max_band_value"])

        train_pool, val_pool = split_train_val(corpus["entries"], args0.val_frac,
                                               args0.val_seed)
        assert len(train_pool) + len(val_pool) == 24, "窗池守恒断言失败"
        device = torch.device("cpu")

        # ---- 参数量表（四 variant 实测打印；小配置只验纪律，真规模由 --print-params-only 打印）
        table = variant_param_table(
            args0, corpus["vocab"], args0.max_params,
            extra_width=peek_extra_width(train_pool,
                                         parse_extra_cols(args0.priv_extra_cols)))
        for v, row in table.items():
            print(f"[selftest][params] variant={v:<6s} prereg={row['prereg_name']:<10s} "
                  f"n_params={row['n_params']}（{row['n_params_m']}M）"
                  f"budget_ok={row['budget_ok']}")
        assert all(row["budget_ok"] for row in table.values()), "小配置超预算"

        # ---- 四 variant 各跑数十步（全链路：建模型 → 训练 → eval → ckpt）
        results = {}
        for variant in ("anchor", "chunk", "priv", "both"):
            args = parse_args(base + ["--variant", variant])
            torch.manual_seed(args.seed)
            model = build_model(corpus["vocab"], args,
                                extra_width=peek_extra_width(
                                    train_pool,
                                    parse_extra_cols(args.priv_extra_cols)))
            n_params = count_params(model)
            logger.info("[selftest] variant=%s n_params=%d", variant, n_params)
            assert n_params <= int(args.max_params), "小配置超参预算"
            before = {k: v.detach().clone() for k, v in model.named_parameters()}
            hist = train_model(args, corpus, model, device, logger, train_pool,
                               val_pool, judge_after=False)
            hist["n_train"], hist["n_val"] = len(train_pool), len(val_pool)
            assert hist["variant"] == variant and hist["logits_shape"][1:] == \
                [WIN, TOKEN_DIM, corpus["vocab"]], \
                f"variant/logits 契约违例 {variant} {hist['logits_shape']}"
            assert all(math.isfinite(x) for x in hist["train_losses"]), \
                f"{variant} loss 非有限"
            assert all(math.isfinite(e["val_loss"]) and math.isfinite(e["val_top1"])
                       for e in hist["eval_points"]), f"{variant} eval 非有限"
            delta = max(float((v.detach() - before[k]).abs().max())
                        for k, v in model.named_parameters())
            assert delta > 0.0, f"{variant} 参数零更新（param_delta={delta}）"
            assert hist["steps_done"] == 24, \
                f"{variant} 步数契约违例 {hist['steps_done']}（4 epoch × 6 步）"
            assert hist["steps_budget_source"] and not hist["converged"] \
                and not hist["early_stop"], f"{variant} 预算来源/停止标志异常"
            if variant in ("chunk", "both"):
                assert hist["rollout_losses"] and \
                    all(x is not None and math.isfinite(x) and x > 0
                        for x in hist["rollout_losses"]), \
                    f"{variant} chunk rollout 段 loss 非有限/非正"
                assert max(hist["rollout_weights"]) > 0, f"{variant} rollout 权重未 ramp 起"
                assert hist["eval_points"][-1]["val_rollout"] is not None, \
                    f"{variant} val rollout 未记"
            else:
                assert hist["rollout_losses"] == [], f"{variant} 不该有 rollout loss"
            if variant in ("priv", "both"):
                assert hist["priv_losses"] and all(
                    x is not None and math.isfinite(x) for x in hist["priv_losses"]), \
                    f"{variant} 特权损失非有限"
            else:
                assert hist["priv_losses"] == [], f"{variant} 不该有特权损失"
            results[variant] = (model, hist, n_params)
            logger.info("[selftest] %s 完成：steps=%d final_val_loss=%.4f "
                        "rollout=%s priv=%s param_delta=%.3e", variant,
                        hist["steps_done"], hist["final_val_loss"],
                        _fmt(hist["final_val_rollout"]), _fmt(hist["final_val_priv"]),
                        delta)

        # ---- 特权头输出维度（含 --priv-extra-cols 真特权列即插）
        args_x = parse_args(base + ["--variant", "both", "--priv-extra-cols",
                                    "elevation_probe"])
        model_x = build_model(corpus["vocab"], args_x,
                              extra_width=peek_extra_width(
                                  train_pool,
                                  parse_extra_cols(args_x.priv_extra_cols)))
        assert model_x.priv_extra == 3, "priv_extra 维度未按 --priv-extra-cols 生效"
        tr_ld, _va_ld = make_loaders(train_pool, val_pool, corpus, args_x, device)
        batch = next(iter(tr_ld))
        assert batch["priv_extra_frame"].shape[-1] == 3, \
            f"真特权列未随批进入：{tuple(batch['priv_extra_frame'].shape)}"
        out = model_x(batch)
        b, k = batch["tok"].shape[0], args_x.chunk_k
        assert tuple(out["priv"]["root_vel"].shape) == (b, k, 3), "root_vel 维度错"
        assert tuple(out["priv"]["joint_vel"].shape) == (b, k, 29), "joint_vel 维度错"
        assert tuple(out["priv"]["terrain_type"].shape) == (b, 3), "terrain_type 维度错"
        assert tuple(out["priv"]["terrain_param"].shape) == (b, 2), "terrain_param 维度错"
        assert tuple(out["priv"]["extra"].shape) == (b, k, 3), "extra 维度错"
        pl = priv_loss(out["priv"], priv_targets(batch["state"], batch["ctx"], k,
                                                 batch["priv_extra_frame"]))
        assert math.isfinite(float(pl)), "特权损失非有限"
        logger.info("[selftest] 特权头维度 ok（root_vel %s/joint_vel %s/terrain %s/%s/"
                    "extra %s）priv_loss=%.4f", tuple(out["priv"]["root_vel"].shape),
                    tuple(out["priv"]["joint_vel"].shape),
                    tuple(out["priv"]["terrain_type"].shape),
                    tuple(out["priv"]["terrain_param"].shape),
                    tuple(out["priv"]["extra"].shape), float(pl))
        # 缺列 fail loud（不静默全零）
        args_miss = parse_args(base + ["--variant", "both", "--priv-extra-cols",
                                       "contact_missing"])
        m_ld, _ = make_loaders(train_pool, val_pool, corpus, args_miss, device)
        try:
            next(iter(m_ld))
            raise AssertionError("缺失真特权列未 fail loud")
        except (KeyError, ValueError) as e:
            logger.info("[selftest] 缺列 fail loud ok（%s）", type(e).__name__)

        # ---- chunk 推理接口：滑窗 + clean-prefix 缓存
        model_both = results["both"][0]
        ctx_feat = batch["ctx"]
        with torch.no_grad():
            _h, ctx_vec = model_both.encode(batch["tok"], batch["state"],
                                            batch["intent"], batch["ctx"])
        pre = batch["tok"][:, :3]
        cache = make_chunk_cache(ctx_feat, pre)
        assert torch.equal(cache["frames"], pre.long()), "clean 前缀未被逐字保留"
        gen, cache2 = chunk_generate(model_both, ctx_vec, cache, k=4)
        assert tuple(gen.shape) == (b, 4, TOKEN_DIM) and gen.dtype == torch.int64, \
            f"chunk 生成形状/类型异常 {tuple(gen.shape)}/{gen.dtype}"
        assert int(gen.min()) >= 0 and int(gen.max()) < corpus["vocab"], \
            "生成码越界（码域/非负索引）"
        assert torch.equal(cache2["frames"], pre.long()), \
            "生成后 cache 的 frames 被写成了生成段（应只含已执行段）"
        cache3 = extend_executed(cache, gen[:, :2])
        assert tuple(cache3["frames"].shape) == (b, 5, TOKEN_DIM) and \
            torch.equal(cache3["frames"][:, -2:], gen[:, :2]), \
            "extend_executed 未把已执行段追加进 clean 前缀"
        gen_a, _ = chunk_generate(model_both, ctx_vec, cache, k=4)
        assert torch.equal(gen, gen_a), "chunk 推理不确定（同输入两次生成不同）"
        # intent-pin-max 推理生效：钉档后主干输出必变
        out_a = model_both({**batch, "ctx": batch["ctx"]})
        pinned_b = pin_ctx_max_band(batch["ctx"].numpy(), bands["max_band_value"])
        out_b = model_both({**batch, "ctx": torch.as_tensor(pinned_b)})
        diff = float((out_a["logits"] - out_b["logits"]).abs().max())
        assert diff > 0.0, "intent-pin-max 钉档未影响模型输出"
        logger.info("[selftest] chunk 推理接口 ok（生成 %s / clean 前缀保留 / "
                    "pin 后 logits maxdiff=%.3e）", tuple(gen.shape), diff)

        # ---- 收敛判据（步数预算来源）helper 直测：本体 + 两条触发守卫
        def _pt(step, loss):
            return {"step": step, "val_loss": loss, "val_top1": 0.2}
        # 下降段平台（尾窗步 40..80 全越过 warmup=5，尾部走平）→ True
        conv = [_pt(10, 3.50), _pt(20, 3.00), _pt(30, 2.50), _pt(40, 2.000),
                _pt(50, 2.001), _pt(60, 2.000), _pt(70, 2.001), _pt(80, 2.000)]
        assert tail_converged(conv, warmup_steps=5), \
            "收敛判据对下降段平台判 False"
        cg = converge_guard(conv, warmup_steps=5)
        assert cg["ok"] and cg["warmup_done"] and cg["tail_in_descent"] \
            and cg["net_flat"], f"下降段平台守卫误判：{cg}"
        # ① 相位守卫：同一平台序列但 warmup 未结束（尾窗首点 step≤warmup）→ False
        assert not tail_converged(conv, warmup_steps=40), \
            "相位守卫失效：warmup 内（=40）仍判收敛"
        assert not converge_guard(conv, warmup_steps=40)["tail_in_descent"], \
            "相位守卫未把 tail_in_descent 置 False"
        # ② 降势守卫：下降段但匀速下降（净降 0.02 > 单步噪声 0.005）→ False
        drift = [_pt(10, 3.50), _pt(20, 3.00), _pt(30, 2.50), _pt(40, 1.100),
                 _pt(50, 1.095), _pt(60, 1.090), _pt(70, 1.085), _pt(80, 1.080)]
        gd = converge_guard(drift, warmup_steps=5)
        assert gd["tail_flat"] and not gd["net_flat"] and not gd["ok"], \
            f"降势守卫失效：匀速下降被当平台 {gd}"
        assert not tail_converged(drift, warmup_steps=5), \
            "降势守卫失效：下降段匀速下降仍判收敛"
        # 本体反例：震荡超相对变化上限 → False；eval 点不足 → False
        noisy = [dict(e, val_loss=1.0 + 0.5 * (i % 2)) for i, e in enumerate(conv)]
        assert not tail_converged(noisy, warmup_steps=5), \
            "收敛判据对震荡序列判 True"
        assert not tail_converged(conv[:3], warmup_steps=5), \
            "收敛判据在 eval 点不足时判 True"
        logger.info("[selftest] 收敛判据 helper ok（下降段平台 True / 相位守卫 "
                    "warmup 内 False / 降势守卫匀速下降 False / 震荡与点数不足 "
                    "False）")

        # ---- gate 判读 mock（真实代码路径：四条全过 + 失败分支）
        mock_evals, loss = [], 3.50
        for i in range(8):
            mock_evals.append({"step": (i + 1) * 10, "val_loss": round(loss, 4),
                               "val_top1": 0.20, "val_rollout": None,
                               "val_priv": None})
            loss = loss - (0.05 if i < 3 else 0.001)
        mock_hist = {"train_losses": [3.6] * 80, "eval_points": mock_evals,
                     "completed_epochs": True, "early_stop": False,
                     "converged": False, "nan_seen": False,
                     "final_val_loss": mock_evals[-1]["val_loss"],
                     "final_val_top1": 0.20, "variant": "both",
                     "prereg_name": "v1b(主配置)"}
        gate = judge_gate(mock_hist, la, logger=logger)
        write_json(os.path.join(out_dir, "gate.json"), gate)
        assert gate["pass"] and gate["marker"] == "D062_GATE_PASS", \
            f"gate mock 判读异常：{gate['marker']}"
        gate_conv = judge_gate(dict(mock_hist, converged=True), la, logger=None)
        assert gate_conv["pass"], "收敛停止应算门①通过（步数预算=收敛判据）"
        gate_bad = judge_gate(dict(mock_hist, eval_points=[
            dict(e, val_top1=0.10) for e in mock_evals]), la, logger=None)
        assert (not gate_bad["pass"]) and gate_bad["marker"] == "D062_GATE_FAIL", \
            "gate 失败分支判读异常"

        # ---- 落盘 + 身份信封 + eval 侧加载
        _, hist_both, n_both = results["both"]
        write_run_meta(out_dir, args0, corpus, hist_both, n_both, gate, device,
                       param_table=table)
        assert os.path.isfile(os.path.join(out_dir, "meta.json")), "meta.json 未落盘"
        assert os.path.isfile(os.path.join(out_dir, "ckpt_final.pt")), "final ckpt 未落盘"
        with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        for key in ("variant", "n_params", "git_rev", "corpus_md5_combined",
                    "corpus_md5_per_source"):
            assert key in meta["identity"], f"身份信封缺 {key}"
        assert meta["identity"]["variant"] == "both" and \
            meta["identity"]["n_params"] == n_both, "身份信封 variant/参数量不一致"
        assert meta["model"]["param_table_all_variants"]["both"]["n_params"] == n_both
        assert meta["intent_pin"]["max_band_value"] == bands["max_band_value"]
        model_rt, cfg_rt = load_author_v1(os.path.join(out_dir, "ckpt_final.pt"),
                                          device="cpu")
        assert cfg_rt["variant"] == "both" and model_rt.use_chunk and model_rt.use_priv
        with torch.no_grad():
            out_rt = model_rt({**batch, "ctx": batch["ctx"]})
        assert tuple(out_rt["logits"].shape) == (b, WIN, TOKEN_DIM, corpus["vocab"]), \
            "ckpt 重建模型输出形状异常"

        # ---- 真特权列 ckpt 往返（防回归：save_ckpt 的 model_cfg.priv_extra 必须
        # 记**实际模型宽度**，否则 load_author_v1 重建出 priv_head.extra=None 与
        # 权重表不匹配 → unexpected keys 加载失败）
        x_path = os.path.join(out_dir, "ckpt_extra_roundtrip.pt")
        save_ckpt(x_path, model_x, 0, args_x, corpus)
        model_xr, cfg_xr = load_author_v1(x_path, device="cpu")
        assert cfg_xr["priv_extra"] == 3 and model_xr.priv_extra == 3, \
            (f"带 --priv-extra-cols 的 ckpt 往返后 priv_extra≠3"
             f"（cfg={cfg_xr['priv_extra']} model={model_xr.priv_extra}）")
        with torch.no_grad():
            out_xr = model_xr({**batch, "ctx": batch["ctx"]})
        assert tuple(out_xr["priv"]["extra"].shape) == (b, k, 3), \
            "真特权列 ckpt 重建模型 extra 头维度异常"
        logger.info("[selftest] 真特权列 ckpt 往返 ok（priv_extra=3 逐字保留，"
                    "extra 头输出 %s）", tuple(out_xr["priv"]["extra"].shape))

        logger.info("[selftest] 全绿（%.1fs）：四 variant 各 %d 步 / chunk rollout 有限 / "
                    "特权头维度对 / 参数量打印 / heldout 零进入 / intent-pin-max 生效 / "
                    "推理缓存接口 / 收敛判据 / 身份信封 / ckpt 重建",
                    time.time() - t0, 24)
        print("SELFTEST_PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="D062=A3 作者 v1（≤300M）联合升级训练：chunk 头（自回归）+ 特权辅助头 "
                    "+ 四跑消融（自包含，服务器平铺布局直接跑；数据接口与 v0 兼容）")
    ap.add_argument("--assembly-manifest",
                    default="data/d060/g1_assembly_full/assembly_manifest.json",
                    help="四源装配 manifest（只引用不合并；与 v0 同一口径）")
    ap.add_argument("--data-root", default="data",
                    help="数据重定位根（manifest 记的服务器绝对路径按此重根）")
    ap.add_argument("--exclude-src4", action="store_true",
                    help="源 id 过滤：排除 src4（闭环 q_des state 口径）")
    ap.add_argument("--limit-windows", type=int, default=None,
                    help="冒烟截断：只取前 K 个 train 窗")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-seed", type=int, default=0,
                    help="窗级 val 切分 seed（独立于语料 seed；heldout 不参与）")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--batch-windows", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30,
                    help="**上界**（步数预算=收敛判据，见 --converge-stop）")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="硬步数上限（缺省 = epochs × 步/epoch）")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=2000,
                    help="warmup 步数（按新规模放大：v0 1000/4,300 步 ≈23% 预热）")
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"],
                    help="auto=cuda bf16（不支持则 fp16+GradScaler），cpu/off 回退 fp32")
    ap.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true",
                    default=True, help="逐层梯度检查点（默认开）")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--d-model", type=int, default=1280)
    ap.add_argument("--n-layer", type=int, default=18,
                    help="默认 18（≈268.5M 主干）；超 --max-params 则减层重配")
    ap.add_argument("--n-head", type=int, default=10)
    ap.add_argument("--ffn", type=int, default=2560)
    ap.add_argument("--max-params", type=int, default=300000000,
                    help="参数量硬预算 ≤300M（§5x；超出报错退出）")
    # --- v1 增量：chunk 头
    ap.add_argument("--variant", default="both",
                    choices=["anchor", "chunk", "priv", "both"],
                    help="消融矩阵：anchor=v1-锚纯放大 / chunk=v1a 仅 chunk 头 / "
                         "priv=v1c 仅特权头 / both=v1b 双头主配置")
    ap.add_argument("--chunk-k", type=int, default=8,
                    help="chunk 头一次前向输出的帧数 K（§5x 消融扫 {4,8,20}，整除 40）")
    ap.add_argument("--chunk-prefix", type=int, default=8,
                    help="chunk 头的 clean 前缀帧数（推理期=上一 chunk 已执行段长度）")
    ap.add_argument("--chunk-d-model", type=int, default=512)
    ap.add_argument("--chunk-layers", type=int, default=2)
    ap.add_argument("--chunk-n-head", type=int, default=8)
    ap.add_argument("--chunk-ffn", type=int, default=1024)
    ap.add_argument("--rollout-weight", type=float, default=0.1,
                    help="chunk 内自回归 rollout CE 权重（线性 ramp 到该值）")
    ap.add_argument("--rollout-ramp-steps", type=int, default=None,
                    help="rollout 权重 ramp 步数（缺省=--warmup-steps）")
    # --- v1 增量：特权头
    ap.add_argument("--priv-weight", type=float, default=0.05,
                    help="特权辅助损失权重（部署丢弃该头）")
    ap.add_argument("--priv-hidden", type=int, default=256)
    ap.add_argument("--priv-extra-cols", default="",
                    help="真特权列（elevation/接触；逗号分隔，缺省空。语料补录后即插，"
                         "缺失 fail loud 不静默全零）")
    # --- v1 增量：09-22 目标改写适配
    ap.add_argument("--intent-pin-max", action="store_true",
                    help="推理期把 ctx 命令槽钉到语料速度带最大档（训练不受影响）")
    # --- 收敛判据 / 沙箱 / 运行
    ap.add_argument("--converge-stop", dest="converge_stop", action="store_true",
                    default=True, help="收敛判据停止（默认开；步数预算=判据非硬编码）")
    ap.add_argument("--no-converge-stop", dest="converge_stop",
                    action="store_false")
    ap.add_argument("--eval-every-frac", type=float, default=0.1,
                    help="每 epoch 的该比例步数做一次 eval")
    ap.add_argument("--ckpt-every-frac", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=10,
                    help="早停：连续 N 个 eval 点无改善")
    ap.add_argument("--out-dir", default="outputs/d062_author_v1")
    ap.add_argument("--force", action="store_true",
                    help="覆盖已含 meta.json 的 out-dir（默认拒跑防覆盖生产 ckpt）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true",
                    help="本机 CPU 自测（合成微语料四 variant 全链路，SELFTEST_PASS）")
    ap.add_argument("--print-params-only", action="store_true",
                    help="只构建模型打印**四 variant** 参数量（实测）并核 --max-params "
                         "预算后退出，不读数据不训练")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        return run_selftest()

    # 预算/格式前置校验（fail loud，不静默降级）
    if args.chunk_k not in CHUNK_K_SCAN:
        print(f"[warn] --chunk-k {args.chunk_k} 不在 §5x 消融扫描集 {CHUNK_K_SCAN} 内")
    if 40 % int(args.chunk_k) != 0:
        print(f"[warn] --chunk-k {args.chunk_k} 不整除 40 步 chunk（§5x 建议 {CHUNK_K_SCAN}）")
    if int(args.chunk_prefix) + int(args.chunk_k) > WIN:
        raise SystemExit(f"[args] --chunk-prefix {args.chunk_prefix} + --chunk-k "
                         f"{args.chunk_k} 超窗长 {WIN}")
    if args.intent_pin_max:
        print("[intent-pin] 已开 --intent-pin-max（**仅推理期**钉 ctx 命令槽到语料"
              "速度带最大档；训练不受影响）")

    if args.print_params_only:
        # 四 variant 参数量实打印（V=29 与 V=27 双口径；V=29 = G4 装配实读四源并集
        # [-15,13]，src3 官方库贡献 -15 与 13）；均须 ≤ --max-params 才 rc 0。
        # 本分支不读数据，extra 宽度只能按 0 计——指定了真特权列就 fail loud
        # （真宽度只有数据侧 peek_extra_width 知道，静默按 0 打印会误导预算核）。
        if parse_extra_cols(args.priv_extra_cols):
            raise SystemExit(
                f"[params] --print-params-only 与 --priv-extra-cols "
                f"{parse_extra_cols(args.priv_extra_cols)} 不可同用：该分支不读数据，"
                f"extra 宽度按 extra=0 口径，真宽度走数据侧（peek_extra_width）；"
                f"请去掉 --priv-extra-cols 或用训练路径核预算")
        all_ok = True
        for vocab in (29, 27):
            table = variant_param_table(args, vocab, args.max_params)
            for v, row in table.items():
                ok = row["budget_ok"]
                all_ok = all_ok and ok
                print(f"[params] vocab={vocab} variant={v:<6s} "
                      f"prereg={row['prereg_name']:<10s} d_model={args.d_model} "
                      f"n_layer={args.n_layer} n_head={args.n_head} ffn={args.ffn} "
                      f"chunk={args.chunk_layers}x{args.chunk_d_model} k={args.chunk_k} "
                      f"grad_ckpt={args.grad_ckpt} → n_params={row['n_params']}"
                      f"（{row['n_params_m']}M）max_params={args.max_params} "
                      f"budget_ok={ok}")
        if not all_ok:
            raise SystemExit(f"[params] 有 variant/口径超出 --max-params "
                             f"{args.max_params} 预算（减 --n-layer/--ffn/--d-model 重配）")
        return 0

    # 沙箱纪律：--out-dir 已含 meta.json 且无 --force 时拒跑（防覆盖生产 ckpt/曲线）
    if os.path.isfile(os.path.join(args.out_dir, "meta.json")) and not args.force:
        raise SystemExit(
            f"[sandbox] 拒跑：{args.out_dir}/meta.json 已存在（防覆盖生产 "
            f"ckpt/曲线）；确认覆盖请加 --force")

    logger = setup_logging(args.out_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    logger.info("[d062] %s | torch %s | device %s | git %s", now_str(),
                torch.__version__, device, git_rev())

    corpus = build_corpus_index(args.assembly_manifest, data_root=args.data_root,
                                exclude_src4=args.exclude_src4,
                                limit_windows=args.limit_windows, logger=logger)
    train_pool, val_pool = split_train_val(corpus["entries"], args.val_frac,
                                           args.val_seed)
    logger.info("[data] train %d / val %d（val_frac=%.2f val_seed=%d）",
                len(train_pool), len(val_pool), args.val_frac, args.val_seed)

    model = build_model(corpus["vocab"], args,
                        extra_width=peek_extra_width(
                            train_pool,
                            parse_extra_cols(args.priv_extra_cols))).to(device)
    n_params = count_params(model)
    logger.info("[model] variant=%s（%s）n_params=%d（%.2fM）d_model=%d n_layer=%d "
                "n_head=%d ffn=%d V=%d | chunk=%s k=%d prefix=%d | priv=%s extra=%s",
                args.variant, VARIANT_PREREG_NAME[args.variant], n_params,
                n_params / 1e6, args.d_model, args.n_layer, args.n_head, args.ffn,
                corpus["vocab"], model.use_chunk, args.chunk_k, args.chunk_prefix,
                model.use_priv, parse_extra_cols(args.priv_extra_cols))
    if n_params > int(args.max_params):
        raise SystemExit(
            f"[model] 参数量 {n_params} 超出 --max-params {args.max_params} 预算，"
            f"拒跑（减 --n-layer/--ffn/--d-model 重配后重跑）")

    hist = train_model(args, corpus, model, device, logger, train_pool, val_pool,
                       judge_after=True)
    hist["n_train"], hist["n_val"] = len(train_pool), len(val_pool)
    gate = None
    gp = os.path.join(args.out_dir, "gate.json")
    if os.path.isfile(gp):
        with open(gp, encoding="utf-8") as f:
            gate = json.load(f)
    # 四 variant 参数量表（构建一次，随 meta 落盘：--print-params-only 同源口径，
    # extra 宽度取本跑实际模型宽度 model.priv_extra）
    param_table = variant_param_table(args, corpus["vocab"], args.max_params,
                                      extra_width=int(model.priv_extra))
    write_run_meta(args.out_dir, args, corpus, hist, n_params, gate, device,
                   param_table=param_table)
    logger.info("[d062] 完成：variant=%s %s | meta.json/train.log/gate.json/ckpt* → %s",
                args.variant, (gate or {}).get("marker", "?"), args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())