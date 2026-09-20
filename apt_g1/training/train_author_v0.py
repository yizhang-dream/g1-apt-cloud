"""D061a 作者 v0（≤80M）BC 训练：四源统一窗语料 → causal transformer 逐帧 next-token。

计划 = refine-logs/ds/DS_TERRAIN_AUTHOR_PLAN.md §5 G2 门/§3⑦ 模型阶梯第一级 +
执行计划（DS_CONTINUOUS_EXECUTION_PLAN.md）§5v 预注册（D061a，80M 级 BC 管线
验证；D060 G5 产量门 PASS 后的发射资格）。数据 = D060 四源统一窗语料
（assembly manifest 只引用不合并，分片留各源目录）。本脚本**自包含**：不 import 任何
apt_g1/d060 模块（地形/命令常量 = d060_windows 的硬编码副本），可部署到执行根平铺
顶层直接 `python train_author_v0.py` 跑。

模型（骨架源自 apt_g1/build/d060_consumer_stub.py 的 TinyCausalWindowTransformer）
---------------------------------------------------------------------------
- 输入每窗：token_stream (200,64) int16 码、intent_tokens (200,64) int16、
  state (200,36) f32（jp29+quat4+trans3 canonical）、command 4 元 + has_truth 4 bool、
  terrain one-hot 3 + height/noise 2 → 上下文向量 CTX_DIM=13（consumer_stub
  window_to_tensors 同口径；码平移 code_min 非负索引，vocab = 全源码域并集跨度）。
- 帧嵌入 = state_proj(state) + 64 个 token 维共享 embedding 取均值 + intent 同构
  （两张独立共享表）；command/terrain 上下文向量投成双前缀帧（n_ctx=2）；可学位置表
  (1,202,d_model)；torch.triu 因果 mask；head=Linear(d_model, 64*V)。
- 默认 d_model=768 / n_layer=11 / n_head=12 / ffn=3072（CLI 可调），逐层
  nn.TransformerEncoderLayer 用 torch.utils.checkpoint 梯度检查点（--grad-ckpt
  默认开）。**80M 预算裁定**：cmd_proj 取单层 Linear(CTX_DIM,d_model)（stub 原为
  Linear+GELU+Linear 两层 ≈+0.59M 参数，会把默认配置顶穿 80M）；码域 = G4 装配
  实读四源并集 [-15,13] V=29（src3 官方库贡献 -15 与 13），双口径实测 V=27 →
  79,532,736 / V=29 → 79,634,240，均 ≤ --max-params 80,000,000 硬断言
  （--print-params-only 双报独立复核；真实跑 V 由装配 manifest 动态取）。
- 目标 = 逐帧 teacher-forcing next-token CE（logits[:,:-1] vs tok[:,1:]），同步记
  next_token_top1_acc（64 维独立头 top1 平均命中）。

训练配方（判据数字已裁定，勿改）：Adafactor(lr=1e-3, relative_step=False,
scale_parameter=False, warmup_init=False) + warmup 1000 步线性 + cosine 至 1e-5，
batch 64 窗，--epochs 30，AMP auto（cuda bf16 / fp16+GradScaler / cpu 回退 fp32），
grad clip 1.0，--eval-every-frac 0.1，--ckpt-every-frac 0.5，早停 patience=10 eval 点。

防泄漏：只索引 split=='train' 窗（索引后断言 heldout 零进入），meta 落
leak_audit={train_windows, heldout_excluded, heldout_entered:0}。val 切分 = 窗级、
从 train 池 rng 抽（--val-seed 独立于语料 seed，不碰 heldout）。

门判读（训练结束后自动判 + 打印 + 落 gate.json）：①无 NaN/Inf 且完成 epochs（或
early_stop）；②val loss 尾部 5 eval 点 (max-min)/min<0.03 且尾部均值 < 首 eval 点；
③val top1 ≥0.185；④leak_audit heldout_entered==0（训练索引 heldout 零进入）。
全过打印 D061A_GATE_PASS，否则 D061A_GATE_FAIL + 各条明细。

用法
----
    python train_author_v0.py --selftest        # 本机 CPU 冒烟（SELFTEST_PASS，rc 0）
    python train_author_v0.py                    # 服务器全量（默认参，读装配 manifest）
    python train_author_v0.py --limit-windows 64 # 冒烟截断
    python train_author_v0.py --print-params-only  # 只构建打印参数量（V=27/V=29 双口径）
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
SENTINEL = -1.0          # D033 无真值哨兵
COMMAND_FIELDS = ("target_vel", "movement_direction", "mode", "height")
TERRAIN_TYPES = ("plane", "rough_paper", "climbing_box")
TERRAIN_DEFAULT_PARAMS = {"plane": {}, "rough_paper": {"noise": 0.04},
                          "climbing_box": {"height": 0.5}}
# 上下文向量 = command(4) + has_truth(4) + terrain one-hot(3) + (height,noise)(2)
CTX_DIM = 4 * 2 + 3 + 2

# 判据数字（已裁定，勿改）
MIN_LR = 1e-5            # cosine 终点
GRAD_CLIP = 1.0
GATE_TAIL_N = 5          # 门②尾部 eval 点数
GATE_TAIL_REL = 0.03     # 门②尾部相对变化上限
GATE_TOP1_MIN = 0.185    # 门③ val top1 下限
SPEED_EVERY = 50         # [SPEED] 行间隔（step）


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


def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("d061a")
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


# ------------------------------------------------------------------ 上下文向量
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


# ------------------------------------------------------------------ 路径解析
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


# ------------------------------------------------------------------ 语料索引
def build_corpus_index(assembly_manifest, data_root=None, exclude_src4=False,
                       limit_windows=None, logger=None):
    """装配 manifest → 四源逐窗记录 → train 窗索引。

    只索引 split=='train'；heldout 全量排除并计数（leak_audit）。返回 dict 含
    entries（shard 绝对路径 + index + 预计算 ctx）、code_min/vocab、leak_audit、
    逐源统计与四源 manifest md5（身份信封）。
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
                 "split": "train", "ctx": ctx_from_record(rec)}
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
    log("[corpus] 索引完成：train %d（limit 前 %d）/ heldout 排除 %d / 码域 [%d,%d] V=%d",
        len(entries), n_before_limit, heldout_excluded, code_min, code_max, vocab)
    return {"assembly": os.path.abspath(assembly_manifest), "entries": entries,
            "code_min": code_min, "code_max": code_max, "vocab": vocab,
            "leak_audit": leak_audit, "source_stats": src_stats,
            "md5s": md5s,
            "corpus_md5_combined": md5_text(
                "".join(f"{k}:{v}\n" for k, v in sorted(md5s.items())))}


def split_train_val(entries, val_frac, val_seed):
    """窗级 val 切分（rng 独立于语料 seed）。"""
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
    """

    def __init__(self, entries, code_min, vocab):
        self.entries = entries
        self.code_min = int(code_min)
        self.vocab = int(vocab)
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
            with np.load(path, allow_pickle=False) as npz:
                z = {k: npz[k] for k in ("token_stream", "intent_tokens", "state")}
            self._cache[path] = z
        return z

    def __getitem__(self, i):
        e = self.entries[i]
        z = self._shard(e["shard"])
        j = e["index"]
        return {
            "tok": z["token_stream"][j].astype(np.int64) - self.code_min,
            "intent": z["intent_tokens"][j].astype(np.int64) - self.code_min,
            "state": z["state"][j].astype(np.float32),
            "ctx": e["ctx"].astype(np.float32),
        }


def make_loaders(train_pool, val_pool, corpus, args, device):
    workers = int(args.num_workers)
    if workers > 0 and os.name == "nt":
        print("[data] Windows 检出：num_workers 回退 0（spawn 兼容）")
        workers = 0
    pin = device.type == "cuda"
    g = torch.Generator().manual_seed(int(args.seed))
    train_ds = WindowDataset(train_pool, corpus["code_min"], corpus["vocab"])
    val_ds = WindowDataset(val_pool, corpus["code_min"], corpus["vocab"])
    train_ld = DataLoader(train_ds, batch_size=int(args.batch_windows), shuffle=True,
                          num_workers=workers, generator=g, pin_memory=pin,
                          drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=int(args.batch_windows), shuffle=False,
                        num_workers=workers, pin_memory=pin, drop_last=False)
    return train_ld, val_ld


# ------------------------------------------------------------------ 模型
class AuthorV0Transformer(nn.Module):
    """统一窗 → 每帧 64×V logits。

    骨架源自 apt_g1/build/d060_consumer_stub.py TinyCausalWindowTransformer：
    command/terrain 双前缀上下文帧 + 逐帧 next-token 头。差异 = 编码器用
    nn.TransformerEncoderLayer（pre-LN/gelu，逐层 torch.utils.checkpoint 梯度检查点）
    与 cmd_proj 单层投影（80M 预算裁定，见模块 docstring）。
    """

    def __init__(self, vocab, state_dim=STATE_DIM, d_model=768, n_layer=11,
                 n_head=12, ffn=3072, grad_ckpt=True):
        super().__init__()
        self.vocab = int(vocab)
        self.n_ctx = 2
        self.grad_ckpt = bool(grad_ckpt)
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

    def forward(self, tokens_idx, state, intent_idx, ctx_feat):
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
        x = self.ln_f(x)[:, self.n_ctx:]
        return self.head(x).view(b, t, TOKEN_DIM, self.vocab)


def build_model(vocab, args):
    return AuthorV0Transformer(vocab, d_model=args.d_model, n_layer=args.n_layer,
                               n_head=args.n_head, ffn=args.ffn,
                               grad_ckpt=args.grad_ckpt)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


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


def _autocast_ctx(device, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def step_loss_acc(model, batch, vocab, device, amp_dtype):
    """一步前向 → (loss, top1_acc, logits_shape)。CE 在 fp32 上算（bf16 稳定性）。"""
    with _autocast_ctx(device, amp_dtype):
        logits = model(batch["tok"], batch["state"], batch["intent"], batch["ctx"])
    lg = logits[:, :-1].float()
    tgt = batch["tok"][:, 1:]
    loss = F.cross_entropy(lg.reshape(-1, vocab), tgt.reshape(-1))
    with torch.no_grad():
        acc = float((lg.argmax(-1) == tgt).float().mean())
    return loss, acc, list(logits.shape)


@torch.no_grad()
def evaluate(model, loader, vocab, device, amp_dtype):
    """val loss + top1（按 token 位加权平均）。"""
    model.eval()
    tot_loss = tot_hit = n_pos = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with _autocast_ctx(device, amp_dtype):
            logits = model(batch["tok"], batch["state"], batch["intent"],
                           batch["ctx"])
        lg = logits[:, :-1].float()
        tgt = batch["tok"][:, 1:]
        loss = F.cross_entropy(lg.reshape(-1, vocab), tgt.reshape(-1),
                               reduction="sum")
        tot_loss += float(loss)
        tot_hit += float((lg.argmax(-1) == tgt).sum())
        n_pos += int(tgt.numel())
    model.train()
    if n_pos == 0:
        return float("nan"), float("nan")
    return tot_loss / n_pos, tot_hit / n_pos


def save_ckpt(path, model, step, args, corpus, extra=None):
    torch.save({
        "format": "d061a.v0", "step": int(step),
        "model": model.state_dict(),
        "model_cfg": {"d_model": args.d_model, "n_layer": args.n_layer,
                      "n_head": args.n_head, "ffn": args.ffn,
                      "vocab": corpus["vocab"], "code_min": corpus["code_min"],
                      "state_dim": STATE_DIM, "win": WIN, "token_dim": TOKEN_DIM},
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


def train_model(args, corpus, model, device, logger, train_pool, val_pool,
                judge_after=True):
    """完整训练循环：AMP + Adafactor + warmup/cosine + 周期 eval/ckpt + 早停 + [SPEED]。

    返回 hist dict（曲线/eval 点/early_stop/completed_epochs/nan_seen/形状记录）；
    judge_after=True 时训练结束自动判门（打印 D061A_GATE_* 并落 gate.json）。
    """
    vocab = corpus["vocab"]
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
    total_steps = steps_per_epoch * int(args.epochs)
    eval_every = max(1, int(round(steps_per_epoch * float(args.eval_every_frac))))
    ckpt_every = max(1, int(round(steps_per_epoch * float(args.ckpt_every_frac))))
    logger.info("[train] device=%s amp=%s grad_ckpt=%s | train %d 窗 / val %d 窗 | "
                "batch %d → %d 步/epoch × %d epoch = %d 步 | eval 每 %d 步 / ckpt 每 %d 步",
                device, amp_dtype, model.grad_ckpt, len(train_pool), len(val_pool),
                args.batch_windows, steps_per_epoch, args.epochs, total_steps,
                eval_every, ckpt_every)

    opt, adafactor_variant = build_adafactor(model.parameters(), args.lr)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, [make_lr_lambda(total_steps, int(args.warmup_steps),
                             float(args.lr), MIN_LR)])

    hist = {"total_steps": total_steps, "steps_done": 0, "train_losses": [],
            "eval_points": [], "curves": [], "early_stop": False,
            "completed_epochs": False, "nan_seen": False, "logits_shape": None,
            "steps_per_epoch": steps_per_epoch, "eval_every": eval_every,
            "ckpt_every": ckpt_every, "best_val_loss": None,
            "adafactor_variant": adafactor_variant, "amp": amp_record}
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
            loss, _acc, shape = step_loss_acc(model, batch, vocab, device, amp_dtype)
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
                val_loss, val_top1 = evaluate(model, val_ld, vocab, device,
                                              amp_dtype)
                hist["eval_points"].append({"step": hist["steps_done"],
                                            "val_loss": float(val_loss),
                                            "val_top1": float(val_top1)})
                hist["curves"].append([
                    hist["steps_done"],
                    float(np.mean(recent_losses)) if recent_losses else float("nan"),
                    float(val_loss), float(val_top1),
                    float(sch.get_last_lr()[0]), round(time.time() - t0, 2)])
                recent_losses = []
                logger.info("[eval] step %d/%d val_loss=%.4f val_top1=%.4f "
                            "(lr=%.2e)", hist["steps_done"], total_steps,
                            val_loss, val_top1, sch.get_last_lr()[0])
                # 早停：严格小于最优才算改善，连续 patience 个 eval 点无改善即停
                if best_val is None or val_loss < best_val:
                    best_val = float(val_loss)
                    hist["best_val_loss"] = best_val
                    bad_evals = 0
                else:
                    bad_evals += 1
                    if bad_evals >= int(args.patience):
                        hist["early_stop"] = True
                        logger.info("[early_stop] 连续 %d 个 eval 点无改善"
                                    "（best=%.4f）→ 停", bad_evals, best_val)
                        stop = True
            if hist["steps_done"] % ckpt_every == 0:
                p = os.path.join(args.out_dir,
                                 f"ckpt_step{hist['steps_done']:07d}.pt")
                save_ckpt(p, model, hist["steps_done"], args, corpus)
                logger.info("[ckpt] %s", p)
            if stop or hist["steps_done"] >= total_steps:
                break
    hist["completed_epochs"] = (not hist["early_stop"]) and (not hist["nan_seen"]) \
        and hist["steps_done"] >= total_steps
    if not hist["eval_points"]:  # 极短跑兜底：至少一个 eval 点
        val_loss, val_top1 = evaluate(model, val_ld, vocab, device, amp_dtype)
        hist["eval_points"].append({"step": hist["steps_done"],
                                    "val_loss": float(val_loss),
                                    "val_top1": float(val_top1)})
        hist["curves"].append([hist["steps_done"],
                               float(np.mean(hist["train_losses"][-eval_every:]))
                               if hist["train_losses"] else float("nan"),
                               float(val_loss), float(val_top1),
                               float(sch.get_last_lr()[0]), round(time.time() - t0, 2)])
    hist["final_val_loss"] = hist["eval_points"][-1]["val_loss"]
    hist["final_val_top1"] = hist["eval_points"][-1]["val_top1"]
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


# ------------------------------------------------------------------ 门判读
def judge_gate(hist, leak_audit, logger=None):
    """D061a 四条门自动判读（判据数字冻结，见模块常量）。"""
    def log(msg, *a):
        if logger is not None:
            logger.info(msg, *a)
        else:
            print(msg % a if a else msg)

    tl = hist.get("train_losses") or []
    ev = hist.get("eval_points") or []
    finite = all(math.isfinite(x) for x in tl) and all(
        math.isfinite(e["val_loss"]) and math.isfinite(e["val_top1"]) for e in ev)
    c1 = bool(finite and (hist.get("completed_epochs") or hist.get("early_stop")))

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
            "rule": "无 NaN/Inf 且完成 epochs（或 early_stop 触发）",
            "pass": c1, "loss_finite": finite,
            "completed_epochs": bool(hist.get("completed_epochs")),
            "early_stop": bool(hist.get("early_stop")),
            "nan_seen": bool(hist.get("nan_seen"))},
        "c2_val_loss_converged": {
            "rule": f"尾部 {GATE_TAIL_N} eval 点 (max-min)/min<{GATE_TAIL_REL} "
                    f"且尾部均值 < 首 eval 点",
            "pass": c2, "tail_rel_change": rel if math.isfinite(rel) else None,
            "tail_mean": (sum(tail) / len(tail)) if tail else None,
            "first_val_loss": ev[0]["val_loss"] if ev else None,
            "n_eval_points": len(ev)},
        "c3_val_top1": {
            "rule": f"val top1 ≥ {GATE_TOP1_MIN}",
            "pass": c3, "final_val_top1": last_top1},
        "c4_leak_audit": {
            "rule": "leak_audit heldout_entered==0（训练索引 heldout 零进入）",
            "pass": c4, "heldout_entered": entered, "leak_audit": leak_audit},
    }
    ok = c1 and c2 and c3 and c4
    marker = "D061A_GATE_PASS" if ok else "D061A_GATE_FAIL"
    log(f"[gate] {marker}")
    for k, v in items.items():
        log("[gate]   %s: %s（%s）", k, "PASS" if v["pass"] else "FAIL", v["rule"])
    return {"marker": marker, "pass": ok, "criteria": items,
            "final_val_loss": hist.get("final_val_loss"),
            "final_val_top1": hist.get("final_val_top1"),
            "judged_at": now_str()}


# ------------------------------------------------------------------ meta
def write_run_meta(out_dir, args, corpus, hist, n_params, gate, device):
    meta = {
        "exp": "D061A", "role": "作者 v0（≤80M）BC 训练", "created_at": now_str(),
        "cwd": os.getcwd(),
        "identity": {
            "script": os.path.abspath(__file__),
            "script_md5": md5_file(__file__),
            "git_rev": git_rev(),
            "hostname": socket.gethostname(),
            "torch": torch.__version__, "numpy": np.__version__,
            "corpus_md5_per_source": corpus["md5s"],
            "corpus_md5_combined": corpus["corpus_md5_combined"],
        },
        "args": vars(args),
        "model": {"d_model": args.d_model, "n_layer": args.n_layer,
                  "n_head": args.n_head, "ffn": args.ffn,
                  "n_params": n_params, "max_params": int(args.max_params),
                  "grad_ckpt": bool(args.grad_ckpt),
                  "vocab": corpus["vocab"],
                  "code_range": [corpus["code_min"], corpus["code_max"]],
                  "param_budget_ok": n_params <= int(args.max_params)},
        "corpus": {"assembly": corpus["assembly"],
                   "source_stats": corpus["source_stats"],
                   "exclude_src4": bool(args.exclude_src4),
                   "limit_windows": args.limit_windows},
        "leak_audit": corpus["leak_audit"],
        "splits": {"train_windows": hist.get("n_train"),
                   "val_windows": hist.get("n_val"),
                   "val_frac": args.val_frac, "val_seed": args.val_seed},
        "curves": hist["curves"],
        "eval_points": hist["eval_points"],
        "summary": {k: hist.get(k) for k in (
            "steps_done", "total_steps", "early_stop", "completed_epochs",
            "nan_seen", "logits_shape", "final_val_loss", "final_val_top1",
            "best_val_loss", "elapsed_s", "amp", "adafactor_variant")},
        "gate": None if gate is None else {
            "marker": gate["marker"], "pass": gate["pass"]},
        "elapsed_s": hist.get("elapsed_s"),
    }
    write_json(os.path.join(out_dir, "meta.json"), meta)
    return meta


# ------------------------------------------------------------------ selftest
def make_selftest_corpus(root, seed=0):
    """内存造 2 源×6 窗合成微语料（码域 [-14,12]，terrain 两型）+2 个 heldout 泄漏探针。

    写成真实装配/源 manifest + npz 分片布局（临时目录），让索引/Dataset/训练全链路
    走真实代码路径。
    """
    rng = np.random.default_rng(seed)
    code_min, code_max = -14, 12
    layout = (("srcA_test", "climbing_box", {"height": 0.5}, 5, 1),
              ("srcB_test", "rough_paper", {"noise": 0.04}, 5, 1))
    srcs = {}
    for sid, terr, tparams, n_train, n_held in layout:
        sdir = os.path.join(root, sid)
        os.makedirs(os.path.join(sdir, "npz"), exist_ok=True)
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
            "exp": "D061A_SELFTEST", "created_at": now_str(),
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
    write_json(apath, {"exp": "D061A_SELFTEST", "created_at": now_str(),
                       "data_root": os.path.abspath(root), "sources": srcs})
    return apath


def run_selftest():
    """本机 CPU 硬验收：合成微语料 → Dataset → 小模型 → 3 步训练 + 1 eval + gate mock。"""
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="d061a_selftest_")
    try:
        asm = make_selftest_corpus(tmp, seed=0)
        out_dir = os.path.join(tmp, "out")
        argv = ["--assembly-manifest", asm, "--data-root", tmp, "--out-dir", out_dir,
                "--d-model", "64", "--n-layer", "2", "--n-head", "2", "--ffn", "128",
                "--batch-windows", "4", "--epochs", "1", "--eval-every-frac", "1.0",
                "--ckpt-every-frac", "1.0", "--lr", "0.02", "--warmup-steps", "1",
                "--num-workers", "0", "--device", "cpu", "--seed", "0"]
        args = parse_args(argv)
        logger = setup_logging(out_dir)
        logger.info("[selftest] 合成微语料（2 源×6 窗，含 2 个 heldout 泄漏探针）→ %s",
                    tmp)

        corpus = build_corpus_index(args.assembly_manifest, data_root=args.data_root,
                                    exclude_src4=args.exclude_src4,
                                    limit_windows=args.limit_windows, logger=logger)
        la = corpus["leak_audit"]
        assert la["heldout_excluded"] == 2 and la["train_windows"] == 10 \
            and la["heldout_entered"] == 0, f"leak 断言/计数异常：{la}"
        assert corpus["vocab"] == 27 and corpus["code_min"] == -14, \
            f"码域契约异常：{corpus['code_min']}..{corpus['code_max']}"

        train_pool, val_pool = split_train_val(corpus["entries"], args.val_frac,
                                               args.val_seed)
        assert len(train_pool) + len(val_pool) == 10, "窗池守恒断言失败"

        device = torch.device("cpu")
        torch.manual_seed(args.seed)
        model = build_model(corpus["vocab"], args)
        n_params = count_params(model)
        logger.info("[selftest] 小配置参数量 %d（max_params=%d）", n_params,
                    args.max_params)
        assert n_params <= int(args.max_params), "小配置超参预算（不应发生）"

        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        hist = train_model(args, corpus, model, device, logger, train_pool,
                           val_pool, judge_after=False)
        hist["n_train"], hist["n_val"] = len(train_pool), len(val_pool)

        # 形状契约 / 有限性 / 参数更新断言
        assert hist["logits_shape"][1:] == [WIN, TOKEN_DIM, corpus["vocab"]], \
            f"logits 形状契约违例 {hist['logits_shape']}"
        assert all(math.isfinite(x) for x in hist["train_losses"]), "loss 非有限"
        delta = max(float((v.detach() - before[k]).abs().max())
                    for k, v in model.named_parameters())
        assert delta > 0.0, f"参数零更新（param_delta={delta}）"
        assert hist["steps_done"] == 3 and len(hist["eval_points"]) >= 1, \
            f"selftest 步数/eval 契约违例 {hist['steps_done']}"

        # gate 判读 mock（真实代码路径：收敛曲线 → 四条全过）
        mock_evals, loss = [], 3.50
        for i in range(8):
            mock_evals.append({"step": (i + 1) * 10, "val_loss": round(loss, 4),
                               "val_top1": 0.20})
            loss = loss - (0.05 if i < 3 else 0.001)
        mock_hist = {"train_losses": [3.6] * 80, "eval_points": mock_evals,
                     "completed_epochs": True, "early_stop": False,
                     "nan_seen": False, "final_val_loss": mock_evals[-1]["val_loss"],
                     "final_val_top1": 0.20}
        gate = judge_gate(mock_hist, la, logger=logger)
        write_json(os.path.join(out_dir, "gate.json"), gate)
        assert gate["pass"] and gate["marker"] == "D061A_GATE_PASS", \
            f"gate mock 判读异常：{gate['marker']}"
        bad_evals = [dict(e, val_top1=0.10) for e in mock_evals]
        gate_bad = judge_gate(dict(mock_hist, eval_points=bad_evals), la,
                              logger=None)
        assert (not gate_bad["pass"]) and gate_bad["marker"] == "D061A_GATE_FAIL", \
            "gate 失败分支判读异常"

        write_run_meta(out_dir, args, corpus, hist, n_params, gate, device)
        assert os.path.isfile(os.path.join(out_dir, "meta.json")), "meta.json 未落盘"
        assert os.path.isfile(os.path.join(out_dir, "ckpt_final.pt")), "final ckpt 未落盘"
        logger.info("[selftest] loss 有限 / param_delta=%.3e / 形状契约 ok / "
                    "gate mock ok / meta+ckpt 落盘（%.1fs）", delta,
                    time.time() - t0)
        print("SELFTEST_PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="D061a 作者 v0（≤80M）BC 训练：四源统一窗语料 next-token causal "
                    "transformer（自包含，服务器平铺布局直接跑）")
    ap.add_argument("--assembly-manifest",
                    default="data/d060/g1_assembly_full/assembly_manifest.json",
                    help="四源装配 manifest（只引用不合并）")
    ap.add_argument("--data-root", default="data",
                    help="数据重定位根（manifest 记的服务器绝对路径按此重根）")
    ap.add_argument("--exclude-src4", action="store_true",
                    help="源 id 过滤：排除 src4（闭环 q_des state 口径）")
    ap.add_argument("--limit-windows", type=int, default=None,
                    help="冒烟截断：只取前 K 个 train 窗")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-seed", type=int, default=0,
                    help="窗级 val 切分 seed（独立于语料 seed）")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--batch-windows", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"],
                    help="auto=cuda bf16（不支持则 fp16+GradScaler），cpu/off 回退 fp32")
    ap.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true",
                    default=True, help="逐层梯度检查点（默认开）")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-layer", type=int, default=11)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--ffn", type=int, default=3072)
    ap.add_argument("--max-params", type=int, default=80000000,
                    help="参数量硬预算（超出报错退出）")
    ap.add_argument("--eval-every-frac", type=float, default=0.1,
                    help="每 epoch 的该比例步数做一次 eval")
    ap.add_argument("--ckpt-every-frac", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=10,
                    help="早停：连续 N 个 eval 点无改善")
    ap.add_argument("--out-dir", default="outputs/d061a_author_v0")
    ap.add_argument("--force", action="store_true",
                    help="覆盖已含 meta.json 的 out-dir（默认拒跑防覆盖生产 ckpt）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true",
                    help="本机 CPU 自测（合成微语料，SELFTEST_PASS / rc 0）")
    ap.add_argument("--print-params-only", action="store_true",
                    help="只构建模型打印参数量（V=27 与 V=29 双口径；V=29 = G4 装配"
                         "实读四源并集 [-15,13]，src3 官方库贡献 -15 与 13）并核 "
                         "--max-params 预算后退出，不读数据不训练")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        return run_selftest()

    if args.print_params_only:
        # 参数量独立复核双口径：V=27（旧口径）与 V=29（G4 装配实读四源并集
        # [-15,13]，src3 官方库贡献 -15 与 13）；均须 ≤ --max-params 才 rc 0。
        # 真实跑按装配 manifest 的 token_min/max 动态定 vocab（meta 如实记）。
        all_ok = True
        for vocab in (27, 29):
            model = build_model(vocab, args)
            n_params = count_params(model)
            budget_ok = n_params <= int(args.max_params)
            all_ok = all_ok and budget_ok
            print(f"[params] vocab={vocab} d_model={args.d_model} "
                  f"n_layer={args.n_layer} n_head={args.n_head} ffn={args.ffn} "
                  f"grad_ckpt={args.grad_ckpt} → n_params={n_params}"
                  f"（{n_params / 1e6:.3f}M）max_params={args.max_params} "
                  f"budget_ok={budget_ok}")
        if not all_ok:
            raise SystemExit(f"[params] 有口径超出 --max-params {args.max_params} "
                             f"预算（减 --n-layer/--ffn/--d-model 重配）")
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
    logger.info("[d061a] %s | torch %s | device %s | git %s", now_str(),
                torch.__version__, device, git_rev())

    corpus = build_corpus_index(args.assembly_manifest, data_root=args.data_root,
                                exclude_src4=args.exclude_src4,
                                limit_windows=args.limit_windows, logger=logger)
    train_pool, val_pool = split_train_val(corpus["entries"], args.val_frac,
                                           args.val_seed)
    logger.info("[data] train %d / val %d（val_frac=%.2f val_seed=%d）",
                len(train_pool), len(val_pool), args.val_frac, args.val_seed)

    model = build_model(corpus["vocab"], args).to(device)
    n_params = count_params(model)
    logger.info("[model] n_params=%d（%.2fM）d_model=%d n_layer=%d n_head=%d "
                "ffn=%d V=%d", n_params, n_params / 1e6, args.d_model,
                args.n_layer, args.n_head, args.ffn, corpus["vocab"])
    if n_params > int(args.max_params):
        raise SystemExit(
            f"[model] 参数量 {n_params} 超出 --max-params {args.max_params} 预算，"
            f"拒跑（调 --n-layer/--ffn/--d-model 或提高预算后重跑）")

    hist = train_model(args, corpus, model, device, logger, train_pool, val_pool,
                       judge_after=True)
    hist["n_train"], hist["n_val"] = len(train_pool), len(val_pool)
    gate = None
    gp = os.path.join(args.out_dir, "gate.json")
    if os.path.isfile(gp):
        with open(gp, encoding="utf-8") as f:
            gate = json.load(f)
    write_run_meta(args.out_dir, args, corpus, hist, n_params, gate, device)
    logger.info("[d061a] 完成：%s | meta.json/train.log/gate.json/ckpt* → %s",
                (gate or {}).get("marker", "?"), args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
