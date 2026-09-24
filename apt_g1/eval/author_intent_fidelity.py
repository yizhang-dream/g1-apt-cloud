"""D061b 轴C：作者模型 intent 重写保真度（teacher-forced 前 K 帧 + 自回归生成区）。

背景（DS 计划 §5v 轴C 判据）
---------------------------------------------------------------------------
D061a 训练口径是"逐帧 teacher-forcing next-token CE"；但部署时模型必须自己把
生成码喂回下一帧（intent 槽重写 = 闭环的核心动作）。本脚本专门量化这条闭环：
前 `--prefix-frames` 帧给真值（token 与 intent 两槽都给），K..199 帧自回归生成，
每步前向 0..k-1 帧、取 `logits[:,-1]` argmax 作为第 k 帧的码，**token 与 intent
两槽同时回填生成码**（只回填 intent 会从 token 槽泄漏真值，指标虚高）。

因果掩码下"每步只前向 0..k-1"与整窗前向等价（掩码保证不看到未来），故自回归
结果 = 逐步重算的精确 rollout（代价 O(WIN-K) 次前向，非近似）。

指标（per source + overall，定义见输出 JSON 的 `metric_defs`）
---------------------------------------------------------------------------
- token_top1_acc：自回归生成区逐码 top1 命中率（主指标）。
- token_top1_acc_full_window / _last100：单次整窗 teacher-forcing 口径。
- frame_exact_match(_full_window)：帧级 64 维全中比例。
- intent_rewrite_distance(_norm)：生成区 intent 槽生成码 vs 真值码的逐帧汉明距离
  均值（及 /64 归一化）。
- n_windows / n_frames / n_codes：计数。
- token_value_mae：码→物理值（value = code / token_scale=16）逐码绝对误差均值。

复用
---------------------------------------------------------------------------
窗读取与模型**复用 `apt_g1/training/train_author_v0.py`**（importlib 按文件加载，
不污染 sys.path）：`build_corpus_index` / `split_train_val` / `WindowDataset` /
`AuthorV0Transformer` / `write_json` / `md5_file` / 常量。若该模块不可加载，脚本
内置兜底（模型类 + 索引 + 窗读取的等价副本，见 `_make_shim`），并在输出 JSON 的
`env.module_fallback` 注明。

模型配置一律以 ckpt 的 `model_cfg` 为准（meta.json 顶层无 args.model 段）。

v1 支持（D062-R1 电池）：`--v1-ckpt` 指向 D062 v1 ckpt（anchor/chunk/priv/both）时，
走 `train_author_v1.load_author_v1` 正规加载（执行根平铺坑——从 sync 克隆包根跑），
逐帧路径经 `_V1FrameAdapter` 只走主干 encode+head（与 v0 forward 同形状同语义，
chunk/priv 头不参与），指标口径与 v0 完全一致；结果 JSON 增记 `v1` 身份信封
（ckpt md5 / variant / model_cfg / model_src=load_author_v1）。

CLI 速览：`python apt_g1/eval/author_intent_fidelity.py --selftest`
（本机 CPU 自测）/ `--selftest-errors`（三条报错路径）/ 正常评测见 --help。
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 模块级默认常量（train_author_v0 同名常量不存在时的兜底；实读仍以该模块为准）
WIN = 200
TOKEN_DIM = 64
STATE_DIM = 36
CTX_DIM = 13
SENTINEL = -1.0
TOKEN_SCALE = 16
N_CTX = 2

# 语料装配口径：源 4（q_des 闭环记录）是否入池。评测主判据用"未入池"的训练分布，
# 该值同时写进输出 config/env（C3），便于判读时确认源 4 是否参与。
EXCLUDE_SRC4 = False

CMD4 = ("target_vel", "movement_direction", "mode", "height")
TERRAINS = ("plane", "rough_paper", "climbing_box")
TERRAIN_DEFAULT_PARAMS = {"plane": {}, "rough_paper": {"noise": 0.04},
                          "climbing_box": {"height": 0.5}}

MODEL_CFG_KEYS = ("d_model", "n_layer", "n_head", "ffn", "vocab", "code_min",
                  "state_dim", "win", "token_dim")


# ------------------------------------------------------------------ 小工具
def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    """UTF-8 无 BOM、LF 行尾落 JSON。"""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def default_train_module_path():
    return os.path.abspath(os.path.join(script_dir(), "..", "training",
                                        "train_author_v0.py"))


def load_train_module(path=None, module_name="train_author_v0_axis_c"):
    """按文件加载 train_author_v0（无副作用，main 有 guard）。→ (mod | None, path, err)"""
    p = os.path.abspath(path) if path else default_train_module_path()
    if not os.path.isfile(p):
        return None, p, f"文件不存在：{p}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod, p, None
    except Exception as e:  # noqa: BLE001 —— 兜底而非崩溃（回退副本）
        return None, p, f"{type(e).__name__}: {e}"


def default_v1_train_module_path():
    return os.path.abspath(os.path.join(script_dir(), "..", "training",
                                        "train_author_v1.py"))


def load_v1_module(path=None):
    """按文件加载 train_author_v1（D062-R1 v1 ckpt 用；口径同 load_train_module）。

    候选序：显式 path → 仓库内默认位置（apt_g1/training/）→ 服务器平铺执行根
    （cwd/train_author_v1.py）→ sys.path 包导入（training.* / apt_g1.*，覆盖从
    sync 克隆包根跑的布局）。返回 (mod | None, path, err)。
    """
    cands = []
    if path:
        cands.append(os.path.abspath(path))
    cands.append(default_v1_train_module_path())
    cands.append(os.path.abspath("train_author_v1.py"))
    errs = []
    for p in cands:
        if not os.path.isfile(p):
            errs.append(f"{p}: 文件不存在")
            continue
        mod, pp, err = load_train_module(p, module_name="train_author_v1_axis_c")
        if mod is not None:
            return mod, pp, None
        errs.append(f"{pp}: {err}")
    last_path = cands[-1] if cands else default_v1_train_module_path()
    for mod_name in ("training.train_author_v1", "apt_g1.training.train_author_v1",
                     "apt_g1.train_author_v1", "train_author_v1"):
        try:
            return importlib.import_module(mod_name), mod_name, None
        except Exception as e:  # noqa: BLE001
            errs.append(f"{mod_name}: {type(e).__name__}: {e}")
    return None, last_path, "所有候选失败：" + "；".join(errs)


# ------------------------------------------------------------------ 兜底副本
class _FallbackModel(nn.Module):
    """train_author_v0.AuthorV0Transformer 的等价副本（仅 import 失败时启用）。

    结构必须与上游一致（selftest 里有逐数值等价断言守护），改动请同步上游。
    """

    def __init__(self, vocab, state_dim=STATE_DIM, d_model=768, n_layer=11,
                 n_head=12, ffn=3072, grad_ckpt=True):
        super().__init__()
        self.vocab = int(vocab)
        self.n_ctx = N_CTX
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
        # 与上游 train_author_v0.AuthorV0Transformer.forward 同步（2026-09-24 起
        # 均值走 embedding_bag，逐位一致由本 selftest torch.equal 守护）
        x = self.state_proj(state)
        x = x + F.embedding_bag(tokens_idx.reshape(-1, TOKEN_DIM),
                                self.tok_emb.weight, mode="mean").view(b, t, -1)
        x = x + F.embedding_bag(intent_idx.reshape(-1, TOKEN_DIM),
                                self.intent_emb.weight, mode="mean").view(b, t, -1)
        ctx = self.cmd_proj(ctx_feat).unsqueeze(1).expand(b, self.n_ctx, -1)
        x = torch.cat([ctx, x], dim=1) + self.pos[:, :t + self.n_ctx]
        total = t + self.n_ctx
        mask = torch.triu(torch.ones(total, total, dtype=torch.bool,
                                     device=x.device), diagonal=1)
        for lyr in self.encoder.layers:
            x = lyr(x, mask)
        x = self.ln_f(x)[:, self.n_ctx:]
        return self.head(x).view(b, t, TOKEN_DIM, self.vocab)


def _fallback_ctx_from_record(rec):
    cmd = [float(rec.get(k, SENTINEL)) for k in CMD4]
    ht = [1.0 if k in (rec.get("has_truth") or []) else 0.0 for k in CMD4]
    ttype = rec.get("terrain")
    if ttype not in TERRAINS:
        raise ValueError(f"窗 {rec.get('stem')} terrain 非法 {ttype!r}")
    onehot = [0.0, 0.0, 0.0]
    onehot[TERRAINS.index(ttype)] = 1.0
    p = dict(TERRAIN_DEFAULT_PARAMS[ttype])
    p.update(rec.get("terrain_params") or {})
    hp = [float(p.get("height", 0.0)), float(p.get("noise", 0.0))]
    return np.asarray(cmd + ht + onehot + hp, dtype=np.float32)


def _fallback_resolve_shard(sdir, name, recorded, data_root, asm_root):
    cands = [os.path.join(sdir, "npz", name), recorded, os.path.join(sdir, name)]
    if data_root:
        cands.append(os.path.join(data_root, name))
        cands.append(os.path.join(data_root, "npz", name))
        if recorded and asm_root and os.path.isabs(recorded):
            old = asm_root.rstrip(os.sep) + os.sep
            if recorded.startswith(old):
                cands.append(os.path.join(
                    data_root, os.path.relpath(recorded, old)))
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"分片缺失：{name}（找过 {[c for c in cands if c]}）")


def _fallback_build_index(assembly_manifest, data_root=None, exclude_src4=False,
                          limit_windows=None, logger=None):
    """train_author_v0.build_corpus_index 的等价精简副本（仅 import 失败时启用）。

    同样只索引 split=='train'，ctx 预计算口径一致。
    """
    def log(msg, *a):
        if logger is not None:
            logger.info(msg, *a)

    if not os.path.isfile(assembly_manifest):
        raise SystemExit(f"[corpus] 装配 manifest 不存在：{assembly_manifest}")
    with open(assembly_manifest, encoding="utf-8") as f:
        asm = json.load(f)
    srcs = asm.get("sources") or {}
    if not srcs:
        raise SystemExit("[corpus] 装配 manifest 无 sources")
    asm_root = asm.get("data_root")
    entries, md5s, src_stats = [], {}, {}
    code_min = code_max = None
    heldout_excluded = other_excluded = 0
    for sid in sorted(srcs):
        blk = srcs[sid] or {}
        if exclude_src4 and str(sid).startswith("src4"):
            continue
        mpath = blk.get("manifest_path")
        if not (mpath and os.path.isfile(mpath)) and data_root:
            cand = os.path.join(data_root, sid, os.path.basename(mpath or ""))
            mpath = cand if os.path.isfile(cand) else mpath
        if not (mpath and os.path.isfile(mpath)):
            raise SystemExit(f"[corpus] 源 {sid} manifest 缺失：{mpath}")
        mpath = os.path.abspath(mpath)
        with open(mpath, encoding="utf-8") as f:
            sdoc = json.load(f)
        md5s[sid] = md5_file(mpath)
        counts = sdoc.get("counts") or blk.get("counts") or {}
        tmin, tmax = counts.get("token_min"), counts.get("token_max")
        if tmin is None or tmax is None:
            raise SystemExit(f"[corpus] 源 {sid} 无 token_min/token_max")
        code_min = int(tmin) if code_min is None else min(code_min, int(tmin))
        code_max = int(tmax) if code_max is None else max(code_max, int(tmax))
        shard_paths = {s.get("name"): s.get("path")
                       for s in sdoc.get("shards", [])}
        sdir = os.path.dirname(mpath)
        n_src_train = n_src_held = 0
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
            entries.append({
                "source": sid,
                "shard": _fallback_resolve_shard(sdir, name,
                                                 shard_paths.get(name),
                                                 data_root, asm_root),
                "index": int(rec["index"]), "split": "train",
                "ctx": _fallback_ctx_from_record(rec)})
            n_src_train += 1
        src_stats[sid] = {"manifest": mpath, "manifest_md5": md5s[sid],
                          "windows_train": n_src_train,
                          "windows_heldout": n_src_held}
        log("[corpus] 源 %s：train %d / heldout %d", sid, n_src_train, n_src_held)
    if not entries:
        raise SystemExit("[corpus] train 窗索引为空")
    n_before_limit = len(entries)
    if limit_windows:
        entries = entries[:int(limit_windows)]
    vocab = int(code_max - code_min + 1)
    leak_audit = {"train_windows": len(entries),
                  "heldout_excluded": heldout_excluded, "heldout_entered": 0,
                  "corpus_has_heldout": bool(heldout_excluded > 0),
                  "other_split_excluded": other_excluded,
                  "train_before_limit": n_before_limit}
    log("[corpus] fallback 索引：train %d / heldout 排除 %d", len(entries),
        heldout_excluded)
    return {"assembly": os.path.abspath(assembly_manifest), "entries": entries,
            "code_min": code_min, "code_max": code_max, "vocab": vocab,
            "leak_audit": leak_audit, "source_stats": src_stats, "md5s": md5s,
            "corpus_md5_combined": None}


def _fallback_split_train_val(entries, val_frac, val_seed):
    """train_author_v0.split_train_val 的等价副本（窗级 rng 切分）。"""
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


def _fallback_load_window(entry, code_min):
    """单窗读取（兜底：无分片缓存，仅供 fallback 索引路径的最小可用）。"""
    with np.load(entry["shard"], allow_pickle=False) as z:
        tok = z["token_stream"][entry["index"]].astype(np.int64) - int(code_min)
        it = z["intent_tokens"][entry["index"]].astype(np.int64) - int(code_min)
        st = z["state"][entry["index"]].astype(np.float32)
    return {"tok": tok, "intent": it, "state": st,
            "ctx": np.asarray(entry["ctx"], dtype=np.float32)}


class _FallbackWindowDataset:
    """WindowDataset 的等价副本（兜底；分片按片缓存）。"""

    def __init__(self, entries, code_min, vocab):
        self.entries = entries
        self.code_min = int(code_min)
        self.vocab = int(vocab)
        self._cache = {}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        z = self._cache.get(e["shard"])
        if z is None:
            with np.load(e["shard"], allow_pickle=False) as npz:
                z = {k: npz[k] for k in ("token_stream", "intent_tokens",
                                         "state")}
            self._cache[e["shard"]] = z
        j = e["index"]
        return {"tok": z["token_stream"][j].astype(np.int64) - self.code_min,
                "intent": z["intent_tokens"][j].astype(np.int64) - self.code_min,
                "state": z["state"][j].astype(np.float32),
                "ctx": np.asarray(e["ctx"], dtype=np.float32)}


def _make_shim():
    """train_author_v0 不可加载时的最小兜底命名空间（副本已在上方注明）。"""
    class _Shim:
        WIN = WIN
        TOKEN_DIM = TOKEN_DIM
        STATE_DIM = STATE_DIM
        CTX_DIM = CTX_DIM
        SENTINEL = SENTINEL
        AuthorV0Transformer = _FallbackModel
        WindowDataset = _FallbackWindowDataset
        build_corpus_index = staticmethod(_fallback_build_index)
        split_train_val = staticmethod(_fallback_split_train_val)
        write_json = staticmethod(write_json)
        md5_file = staticmethod(md5_file)
        make_selftest_corpus = None
    return _Shim()


# ------------------------------------------------------------------ ckpt / 模型
class _V1FrameAdapter(nn.Module):
    """AuthorV1Transformer → v0 式四槽逐帧前向适配（D062-R1 --v1-ckpt 用）。

    v1 forward 吃 batch dict 且会连带跑 chunk/priv 头；逐帧主路径只需要主干部
    （encode + head，与 v0 AuthorV0Transformer.forward 逐字同构，语义不变），故
    适配成 forward(tokens_idx, state, intent_idx, ctx_feat) -> (b,t,token_dim,vocab)，
    teacher_forced_predictions / autoregressive_rollout 等调用点零改动；
    chunk/priv 头不参与逐帧路径。
    """

    def __init__(self, v1_model, token_dim):
        super().__init__()
        self.model = v1_model
        self.token_dim = int(token_dim)

    def forward(self, tokens_idx, state, intent_idx, ctx_feat):
        b, t, _k = tokens_idx.shape
        frames, _ctx_repr = self.model.encode(tokens_idx, state, intent_idx,
                                              ctx_feat)
        return self.model.head(frames).view(b, t, self.token_dim,
                                            int(self.model.vocab))


def load_ckpt(path):
    if not os.path.isfile(path):
        raise SystemExit(
            f"[fidelity] ckpt 不存在：{path}"
            "（用 --ckpt 指定 D061a 产物，默认 outputs/d061_author_v0/ckpt_final.pt）")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # 老 torch 无 weights_only 形参
        return torch.load(path, map_location="cpu")


def read_model_cfg(ckpt, ckpt_path):
    cfg = ckpt.get("model_cfg")
    if not isinstance(cfg, dict) or not cfg:
        raise SystemExit(
            f"[fidelity] ckpt 无 model_cfg 段：{ckpt_path}"
            f"（顶层键 {sorted(ckpt.keys())}；D061a save_ckpt 必写 model_cfg）")
    missing = [k for k in MODEL_CFG_KEYS if k not in cfg]
    if missing:
        raise SystemExit(
            f"[fidelity] ckpt model_cfg 缺键 {missing}：{ckpt_path}"
            f"（需含 {list(MODEL_CFG_KEYS)}；实有 {sorted(cfg.keys())}）")
    if "model" not in ckpt:
        raise SystemExit(f"[fidelity] ckpt 无 model 权重段：{ckpt_path}")
    return cfg


def build_model_from_cfg(cfg, mod, win_len, token_dim):
    if int(cfg["win"]) != int(win_len):
        raise SystemExit(
            f"[fidelity] ckpt model_cfg.win={cfg['win']} 与语料窗长 {win_len} 不符"
            "（位置表按 WIN 固定，暂不支持异长窗）")
    if int(cfg["token_dim"]) != int(token_dim):
        raise SystemExit(
            f"[fidelity] ckpt model_cfg.token_dim={cfg['token_dim']} 与 {token_dim} 不符")
    model = mod.AuthorV0Transformer(
        int(cfg["vocab"]), state_dim=int(cfg["state_dim"]),
        d_model=int(cfg["d_model"]), n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]), ffn=int(cfg["ffn"]), grad_ckpt=False)
    return model


# ------------------------------------------------------------------ 推理件
@torch.no_grad()
def teacher_forced_predictions(model, tok, intent, state, ctx):
    """整窗 teacher-forcing 前向 → 对齐到帧的预测 (b,t,64)；帧 0 无意义（填 -1）。"""
    logits = model(tok, state, intent, ctx)
    pred = logits.argmax(-1)  # (b,t,64)：位置 j 预测帧 j+1
    out = torch.full_like(tok, -1)
    out[:, 1:] = pred[:, :-1]
    return out


@torch.no_grad()
def autoregressive_rollout(model, tok, intent, state, ctx, prefix_frames,
                           progress=None):
    """前 K 帧真值、K..T-1 自回归（token/intent 两槽同时回填生成码）。

    返回 (pred_tok, pred_intent)：仅帧 K..T-1 有值（其余填 -1）。因果掩码下
    每步只前向 0..k-1 与整窗前向等价，这里逐步重算以保证零未来信息泄漏。
    """
    b, t, d = tok.shape
    k0 = int(prefix_frames)
    tok_used = tok.clone()
    intent_used = intent.clone()
    pred_tok = torch.full_like(tok, -1)
    pred_intent = torch.full_like(tok, -1)
    for k in range(k0, t):
        logits = model(tok_used[:, :k], state[:, :k], intent_used[:, :k], ctx)
        codes = logits[:, -1].argmax(-1)  # (b,64)：第 k 帧生成码
        pred_tok[:, k] = codes
        pred_intent[:, k] = codes
        tok_used[:, k] = codes      # 两槽同时回填（只回填 intent 会从 token 槽泄漏真值）
        intent_used[:, k] = codes
    if progress is not None:
        progress()
    return pred_tok, pred_intent


def new_acc():
    return {"gen_hit": 0, "gen_tot": 0, "tf_hit": 0, "tf_tot": 0,
            "l100_hit": 0, "l100_tot": 0,
            "fem": 0.0, "fem_n": 0, "fem_tf": 0.0, "fem_tf_n": 0,
            "hd": 0.0, "hd_n": 0, "vmae": 0.0, "vmae_n": 0,
            "n_windows": 0, "n_frames": 0}


def acc_update(acc, pred_gen, true_tok, true_intent, pred_tf, prefix_frames,
               token_scale):
    """把一批窗的结果累加进 acc（传入张量已按分组切好，帧维 = 整窗 T）。"""
    k = int(prefix_frames)
    t = true_tok.shape[1]
    pg = pred_gen[:, k:]
    tg = true_tok[:, k:]
    acc["gen_hit"] += int((pg == tg).sum())
    acc["gen_tot"] += int(pg.numel())
    acc["fem"] += float((pg == tg).all(-1).sum())
    acc["fem_n"] += int(pg.shape[0] * pg.shape[1])
    acc["hd"] += float((pred_gen[:, k:] != true_intent[:, k:]).sum(-1).sum())
    acc["hd_n"] += int(pg.shape[0] * pg.shape[1])
    acc["vmae"] += float((pg - tg).abs().sum()) / float(token_scale)
    acc["vmae_n"] += int(pg.numel())
    tfp = pred_tf[:, 1:]
    tft = true_tok[:, 1:]
    acc["tf_hit"] += int((tfp == tft).sum())
    acc["tf_tot"] += int(tfp.numel())
    acc["fem_tf"] += float((tfp == tft).all(-1).sum())
    acc["fem_tf_n"] += int(tfp.shape[0] * tfp.shape[1])
    # tfp/tft 的列 j 对应帧 j+1（帧 1..T-1）；最后 100 帧 = 帧 T-100..T-1
    l100 = max(0, (t - 1) - 100)
    acc["l100_hit"] += int((tfp[:, l100:] == tft[:, l100:]).sum())
    acc["l100_tot"] += int(tfp[:, l100:].numel())
    acc["n_windows"] += int(pg.shape[0])
    acc["n_frames"] += int(pg.shape[0] * pg.shape[1])


def acc_finalize(acc):
    def r(num, den):
        return float(num) / float(den) if den else float("nan")

    out = {
        "token_top1_acc": r(acc["gen_hit"], acc["gen_tot"]),
        "token_top1_acc_full_window": r(acc["tf_hit"], acc["tf_tot"]),
        "token_top1_acc_last100": r(acc["l100_hit"], acc["l100_tot"]),
        "frame_exact_match": r(acc["fem"], acc["fem_n"]),
        "frame_exact_match_full_window": r(acc["fem_tf"], acc["fem_tf_n"]),
        "intent_rewrite_distance": r(acc["hd"], acc["hd_n"]),
        "token_value_mae": r(acc["vmae"], acc["vmae_n"]),
        "n_windows": int(acc["n_windows"]),
        "n_frames": int(acc["n_frames"]),
        "n_codes": int(acc["gen_tot"]),
    }
    out["intent_rewrite_distance_norm"] = (
        out["intent_rewrite_distance"] / float(TOKEN_DIM)
        if out["intent_rewrite_distance"] == out["intent_rewrite_distance"]
        else float("nan"))
    return out


METRIC_DEFS = {
    "token_top1_acc": "【主轴】自回归生成区（帧 K..WIN-1，K=--prefix-frames）内 64 个 token 维各自 argmax 命中真值码的比例，分母 = 生成区帧数×64（前 K 帧给真值上下文，K 起 token/intent 两槽同时回填生成码）。",
    "token_top1_acc_full_window": "单次整窗 teacher-forcing 前向（token/intent 两槽全真值）下，帧 1..WIN-1 的逐码 top1 命中率（真值前缀上界口径，不受 K 影响）。",
    "token_top1_acc_last100": "teachers-forcing 整窗口径下最后 min(100, WIN-1) 帧的逐码 top1 命中率（默认 K=100 时其帧区间与生成区一致，可与主轴对读）。",
    "token_value_mae": "生成区码→物理值（value = code / token_scale，token_scale=16）后与真值逐码绝对误差均值（诊断用辅助量）。",
    "frame_exact_match": "生成区内 64 维全部命中的帧占比（帧级全中，比逐码更严）。",
    "frame_exact_match_full_window": "teacher-forcing 整窗口径（帧 1..WIN-1）的 64 维全中帧占比。",
    "intent_rewrite_distance": "生成区 intent 槽被回填生成码后，与真值 intent 码的逐帧汉明距离（每帧 0..64 个码不同）对帧取平均——量化 intent 重写保真度。",
    "intent_rewrite_distance_norm": "intent_rewrite_distance / 64（归一化到 [0,1]）。",
    "n_windows": "该分组（per source 或 overall）参与评测的 val 窗数。",
    "n_frames": "该分组生成区帧数（n_windows × (WIN-K)）。",
    "n_codes": "该分组生成区码数（n_frames × 64，等于 token_top1_acc 的分母）。",
}


# ------------------------------------------------------------------ 主评测
def resolve_paths(args):
    sd = script_dir()
    ckpt_arg = getattr(args, "v1_ckpt", None) or args.ckpt  # --v1-ckpt 优先
    ckpt = os.path.abspath(ckpt_arg) if ckpt_arg else os.path.abspath(
        os.path.join(sd, "..", "outputs", "d061_author_v0", "ckpt_final.pt"))
    meta = args.meta
    if meta is None:
        cand = os.path.join(os.path.dirname(ckpt), "meta.json")
        meta = cand if os.path.isfile(cand) else None
    if meta is not None:
        meta = os.path.abspath(meta)
    asm = os.path.abspath(args.assembly_manifest) if args.assembly_manifest \
        else os.path.abspath(os.path.join(
            sd, "..", "data", "d060", "g1_assembly_full",
            "assembly_manifest.json"))
    return ckpt, meta, asm


def run_eval(args, progress_every=20):
    t0 = time.time()
    ckpt_path, meta_path, asm_path = resolve_paths(args)
    mod, mod_path, mod_err = load_train_module(args.train_module)
    fallback = mod is None
    notes = []
    if fallback:
        notes.append(f"train_author_v0 加载失败（{mod_err}）→ 启用内置副本"
                     f"（模型类/索引/窗读取）；上游路径 {mod_path}")
        mod = _make_shim()
    win_len = int(getattr(mod, "WIN", WIN))
    token_dim = int(getattr(mod, "TOKEN_DIM", TOKEN_DIM))

    v1_variant = None
    v1_module_path = None
    if getattr(args, "v1_ckpt", None):
        # D062-R1 v1 路径：load_author_v1 正规加载（执行根平铺坑——从 sync 克隆包根跑）
        ckpt_path = os.path.abspath(args.v1_ckpt)
        if not os.path.isfile(ckpt_path):
            raise SystemExit(f"[fidelity] --v1-ckpt 不存在：{ckpt_path}")
        t1, v1_module_path, t1_err = load_v1_module()
        if t1 is None:
            raise SystemExit(f"[fidelity] train_author_v1 不可加载：{v1_module_path}"
                             f"（{t1_err}）")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("[fidelity] --device cuda 但 torch.cuda 不可用")
        v1_model, cfg = t1.load_author_v1(ckpt_path, device=str(device))
        if int(cfg.get("win", 0)) != int(win_len) or \
                int(cfg["token_dim"]) != int(token_dim):
            raise SystemExit(
                f"[fidelity] v1 ckpt win/token_dim={cfg.get('win')}/"
                f"{cfg.get('token_dim')} 与语料 {win_len}/{token_dim} 不符")
        model = _V1FrameAdapter(v1_model, int(cfg["token_dim"]))
        v1_variant = cfg.get("variant")
        notes.append(f"--v1-ckpt：经 train_author_v1.load_author_v1 加载"
                     f"（variant={v1_variant}）；逐帧路径走主干 encode+head，"
                     f"chunk/priv 头不参与")
        ckpt = {}  # v1 权重已由 load_author_v1 严格加载；不再持有原始 ckpt dict
    else:
        ckpt = load_ckpt(ckpt_path)
        cfg = read_model_cfg(ckpt, ckpt_path)
        model = build_model_from_cfg(cfg, mod, win_len, token_dim)
        miss, unexp = model.load_state_dict(ckpt["model"], strict=False)
        if miss or unexp:
            raise SystemExit(f"[fidelity] ckpt 权重与模型不匹配：缺 {list(miss)} / "
                             f"多 {list(unexp)}（检查 model_cfg 与权重是否同源）")
    if int(cfg["win"]) < int(args.prefix_frames) + 1:
        raise SystemExit(f"[fidelity] --prefix-frames={args.prefix_frames} 过大："
                         f"窗长 {cfg['win']}（需 ≤ win-1）")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[fidelity] --device cuda 但 torch.cuda 不可用")
    model.to(device).eval()

    code_min = int(cfg["code_min"])
    vocab = int(cfg["vocab"])
    logger = None
    corpus = mod.build_corpus_index(asm_path, data_root=args.data_root,
                                    exclude_src4=EXCLUDE_SRC4,
                                    limit_windows=None, logger=logger)
    _, val_pool = mod.split_train_val(corpus["entries"], args.val_frac,
                                      args.val_seed)
    if args.limit_val_windows is not None:
        val_pool = val_pool[:int(args.limit_val_windows)]
    if not val_pool:
        raise SystemExit(
            f"[fidelity] val 池为空：train 窗 {len(corpus['entries'])} 个，"
            f"--val-frac={args.val_frac} / --limit-val-windows="
            f"{args.limit_val_windows} 下没有任何 val 窗"
            "（val 由 split_train_val 从 train 池按 val_seed 抽；"
            "--limit-val-windows 0 = 显式空池）")
    srcs_in_val = sorted({e["source"] for e in val_pool})
    print(f"[fidelity] ckpt={ckpt_path} code_min={code_min} V={vocab} "
          f"win={cfg['win']} K={args.prefix_frames} val 窗 {len(val_pool)} "
          f"源 {srcs_in_val} device={device}")

    ds = mod.WindowDataset(val_pool, code_min, vocab)
    accs = {"__overall__": new_acc()}
    for sid in srcs_in_val:
        accs[sid] = new_acc()

    batch = int(args.batch_windows)
    done = 0
    for i0 in range(0, len(val_pool), batch):
        entries = val_pool[i0:i0 + batch]
        items = [ds[j] for j in range(i0, min(i0 + batch, len(val_pool)))]
        tok = torch.from_numpy(np.stack([it["tok"] for it in items])).to(device)
        intent = torch.from_numpy(
            np.stack([it["intent"] for it in items])).to(device)
        state = torch.from_numpy(
            np.stack([it["state"] for it in items])).to(device)
        ctx = torch.from_numpy(
            np.stack([it["ctx"] for it in items])).to(device)
        tf_pred = teacher_forced_predictions(model, tok, intent, state, ctx)
        gen_tok, gen_intent = autoregressive_rollout(
            model, tok, intent, state, ctx, args.prefix_frames)
        src_list = [e["source"] for e in entries]
        for sid in ["__overall__"] + sorted(set(src_list)):
            if sid == "__overall__":
                m = torch.ones(len(entries), dtype=torch.bool)
            else:
                m = torch.tensor([s == sid for s in src_list],
                                 device=gen_tok.device)
            acc_update(accs[sid], gen_tok[m], tok[m], intent[m], tf_pred[m],
                       args.prefix_frames, TOKEN_SCALE)
        done += len(entries)
        if progress_every and (done % progress_every == 0 or done >= len(val_pool)):
            print(f"[fidelity] 进度 {done}/{len(val_pool)} 窗"
                  f"（{time.time() - t0:.0f}s）")

    overall = acc_finalize(accs["__overall__"])
    per_source = {sid: acc_finalize(a) for sid, a in accs.items()
                  if sid != "__overall__"}
    for sid in sorted(per_source):
        m = per_source[sid]
        print(f"[fidelity] {sid}: n={m['n_windows']} "
              f"gen_top1={m['token_top1_acc']:.4f} "
              f"tf_top1={m['token_top1_acc_full_window']:.4f} "
              f"fem={m['frame_exact_match']:.4f} "
              f"intent_hd={m['intent_rewrite_distance']:.2f}")
    print(f"[fidelity] OVERALL: n={overall['n_windows']} "
          f"gen_top1={overall['token_top1_acc']:.4f} "
          f"tf_top1={overall['token_top1_acc_full_window']:.4f} "
          f"last100={overall['token_top1_acc_last100']:.4f} "
          f"fem={overall['frame_exact_match']:.4f} "
          f"intent_hd={overall['intent_rewrite_distance']:.2f} "
          f"(norm {overall['intent_rewrite_distance_norm']:.4f})")

    meta_payload = None
    if meta_path:
        if not os.path.isfile(meta_path):
            raise SystemExit(f"[fidelity] --meta 指定的文件不存在：{meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            meta_doc = json.load(f)
        meta_payload = {
            "path": meta_path, "md5": md5_file(meta_path),
            "model": meta_doc.get("model"), "args": meta_doc.get("args"),
            "gate": meta_doc.get("gate"), "leak_audit": meta_doc.get("leak_audit"),
        }
        mvocab = (meta_doc.get("model") or {}).get("vocab") \
            or (meta_doc.get("args") or {}).get("vocab")
        if mvocab is not None and int(mvocab) != vocab:
            notes.append(f"meta 记的 vocab={mvocab} 与 ckpt model_cfg.vocab="
                         f"{vocab} 不一致（以 ckpt 为准）")
    else:
        notes.append("未找到 meta.json（--meta 缺省且 ckpt 同目录无此文件），"
                     "身份信封只含 ckpt/脚本/语料 md5")

    payload = {
        "exp": "D061B_INTENT_FIDELITY",
        "created_at": now_str(),
        "elapsed_s": round(time.time() - t0, 1),
        "metric_defs": METRIC_DEFS,
        "overall": overall,
        "per_source": per_source,
        "env": {
            "ckpt": ckpt_path, "ckpt_md5": md5_file(ckpt_path),
            "ckpt_step": ckpt.get("step"), "ckpt_format": ckpt.get("format"),
            "meta": meta_payload,
            "script": os.path.abspath(__file__),
            "script_md5": md5_file(os.path.abspath(__file__)),
            "train_module": mod_path,
            "train_module_md5": md5_file(mod_path) if os.path.isfile(mod_path)
            else None,
            "module_fallback": fallback,
            "assembly_manifest": asm_path,
            "assembly_md5": md5_file(asm_path) if os.path.isfile(asm_path)
            else None,
            "corpus_md5_combined": corpus.get("corpus_md5_combined"),
            "source_stats": corpus.get("source_stats"),
            "leak_audit": corpus.get("leak_audit"),
            "exclude_src4": EXCLUDE_SRC4,
            "torch": torch.__version__, "device": str(device),
        },
        "config": {
            "prefix_frames": int(args.prefix_frames),
            "win": int(cfg["win"]), "token_dim": int(token_dim),
            "token_scale": TOKEN_SCALE, "code_min": code_min, "vocab": vocab,
            "model_cfg": {k: cfg[k] for k in MODEL_CFG_KEYS},
            "val_frac": float(args.val_frac), "val_seed": int(args.val_seed),
            "limit_val_windows": args.limit_val_windows,
            "batch_windows": int(args.batch_windows),
            "data_root": args.data_root,
            "exclude_src4": EXCLUDE_SRC4,
            "rollout": "teacher-forced 前 K 帧 + K..WIN-1 逐帧自回归；每步前向 "
                       "0..k-1（因果掩码等价整窗）；token 与 intent 两槽同时"
                       "回填生成码；state/ctx 只用真值（无其它回填）",
        },
        "notes": notes,
    }
    if v1_variant is not None:  # D062-R1 v1 身份信封（v0 路径不新增任何键）
        payload["v1"] = {
            "model_src": "load_author_v1",
            "ckpt": ckpt_path,
            "ckpt_md5": md5_file(ckpt_path),
            "variant": v1_variant,
            "model_cfg": dict(cfg),
            "frame_path": "主干 encode+head（逐帧路径；chunk/priv 头不参与）",
            "train_module_v1": v1_module_path,
        }
    out = os.path.abspath(args.out_json)
    mod.write_json(out, payload)
    print(f"[fidelity] 落盘 {out}（{payload['elapsed_s']}s）")
    return 0


# ------------------------------------------------------------------ selftest
def _selftest_ckpt(path, cfg, mod, seed=0):
    torch.manual_seed(seed)
    model = mod.AuthorV0Transformer(
        int(cfg["vocab"]), state_dim=int(cfg["state_dim"]),
        d_model=int(cfg["d_model"]), n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]), ffn=int(cfg["ffn"]), grad_ckpt=False)
    torch.save({"format": "d061a.v0", "step": 0, "model": model.state_dict(),
                "model_cfg": dict(cfg)}, path)
    return model


def _mini_args(**kw):
    base = dict(ckpt=None, meta=None, assembly_manifest=None, data_root=None,
                prefix_frames=WIN // 2, val_frac=0.05, val_seed=0,
                limit_val_windows=None, device="cpu",
                out_json=os.path.join("tmp", "d061b_selftest.json"),
                selftest=False, selftest_errors=False, batch_windows=4,
                train_module=None)
    base.update(kw)
    return argparse.Namespace(**base)


def run_selftest():
    t0 = time.time()
    mod, mod_path, mod_err = load_train_module()
    if mod is None:
        print(f"[selftest] FAIL：train_author_v0 不可加载（{mod_path}）：{mod_err}",
              file=sys.stderr)
        return 1
    tmp = tempfile.mkdtemp(prefix="d061b_selftest_")
    asm = mod.make_selftest_corpus(tmp, seed=0)
    corpus = mod.build_corpus_index(asm, data_root=None, exclude_src4=False,
                                   limit_windows=None, logger=None)
    cfg = {"d_model": 32, "n_layer": 1, "n_head": 4, "ffn": 64,
           "vocab": corpus["vocab"], "code_min": corpus["code_min"],
           "state_dim": STATE_DIM, "win": WIN, "token_dim": TOKEN_DIM}
    ckpt = os.path.join(tmp, "ckpt_final.pt")
    _selftest_ckpt(ckpt, cfg, mod, seed=0)
    mod.write_json(os.path.join(tmp, "meta.json"),
                   {"exp": "D061B_SELFTEST", "args": {"val_frac": 0.05},
                    "model": {"vocab": corpus["vocab"], "n_params": None},
                    "gate": {"pass": None}, "leak_audit": corpus["leak_audit"]})
    out_json = os.path.join(tmp, "d061b_intent_fidelity.json")
    # 本机 torch CPU 极慢（实测 1 层/32 维的 200 帧前向 ~6.6s），自测把生成区压到
    # 10 帧（K=190）以保证自测可跑完；正式 CLI 默认 K=WIN//2=100。
    st_k = 190
    args = _mini_args(ckpt=ckpt, assembly_manifest=asm, out_json=out_json,
                      train_module=mod_path, batch_windows=2,
                      prefix_frames=st_k)
    rc = run_eval(args)
    assert rc == 0, f"run_eval 返回 {rc}"

    with open(out_json, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["overall"]["n_windows"] == 1, \
        f"val 窗数应 1（10 train 窗 × 0.05）：{doc['overall']['n_windows']}"
    assert sum(m["n_windows"] for m in doc["per_source"].values()) == 1
    assert 0.0 <= doc["overall"]["token_top1_acc"] <= 1.0
    assert 0.0 <= doc["overall"]["frame_exact_match"] <= 1.0
    assert doc["overall"]["n_frames"] == 1 * (WIN - st_k)
    assert doc["overall"]["n_codes"] == doc["overall"]["n_frames"] * TOKEN_DIM
    assert doc["overall"]["intent_rewrite_distance_norm"] == \
        doc["overall"]["intent_rewrite_distance"] / 64
    assert len(doc["metric_defs"]) >= 8 and all(
        isinstance(v, str) and v for v in doc["metric_defs"].values())
    assert doc["env"]["ckpt_md5"] == md5_file(ckpt)
    assert doc["env"]["module_fallback"] is False
    assert doc["env"]["meta"]["path"].endswith("meta.json")
    assert doc["config"]["prefix_frames"] == st_k
    assert doc["config"]["prefix_frames"] == args.prefix_frames
    with open(out_json, "rb") as f:
        raw = f.read()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r\n" not in raw, \
        "输出 JSON 必须 LF 无 BOM"
    print(f"[selftest] 输出 JSON 校验通过（{len(raw)}B, LF/无BOM, "
          f"metric_defs {len(doc['metric_defs'])} 条）")

    # 兜底副本等价性：同权重同输入 → 逐数值 allclose
    ref = mod.AuthorV0Transformer(cfg["vocab"], state_dim=STATE_DIM,
                                  d_model=cfg["d_model"], n_layer=cfg["n_layer"],
                                  n_head=cfg["n_head"], ffn=cfg["ffn"],
                                  grad_ckpt=False).eval()
    fb = _FallbackModel(cfg["vocab"], state_dim=STATE_DIM, d_model=cfg["d_model"],
                        n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                        ffn=cfg["ffn"], grad_ckpt=False).eval()
    fb.load_state_dict(ref.state_dict())
    torch.manual_seed(1)
    it_ = torch.randint(0, cfg["vocab"], (2, 6, TOKEN_DIM))
    st_ = torch.randn(2, 6, STATE_DIM)
    cx_ = torch.randn(2, CTX_DIM)
    with torch.no_grad():
        y_ref = ref(it_, st_, it_, cx_)
        y_fb = fb(it_, st_, it_, cx_)
    assert torch.equal(y_ref, y_fb), "兜底模型副本与上游 AuthorV0Transformer 不等价"
    print("[selftest] 兜底副本 forward 与上游逐数值一致")

    # 兜底索引副本：同装配 manifest → 同窗数/同码域
    shim = _make_shim()
    c2 = shim.build_corpus_index(asm, data_root=None)
    assert len(c2["entries"]) == len(corpus["entries"]), \
        f"兜底索引窗数 {len(c2['entries'])} != {len(corpus['entries'])}"
    assert (c2["code_min"], c2["vocab"]) == (corpus["code_min"], corpus["vocab"])
    tp, vp = shim.split_train_val(corpus["entries"], 0.05, 0)
    tp_r, vp_r = mod.split_train_val(corpus["entries"], 0.05, 0)
    assert len(vp) == len(vp_r) and len(tp) == len(tp_r)
    print(f"[selftest] 兜底索引/切分副本一致（train {len(tp)} / val {len(vp)}）")

    # 自回归不回填 state/ctx 真值以外的东西：同窗两次调用结果确定
    ds = mod.WindowDataset(vp_r, corpus["code_min"], corpus["vocab"])
    it0 = ds[0]
    tt = torch.from_numpy(it0["tok"][None])
    ii = torch.from_numpy(it0["intent"][None])
    ss = torch.from_numpy(it0["state"][None])
    cc = torch.from_numpy(it0["ctx"][None])
    with torch.no_grad():
        g1, _ = autoregressive_rollout(ref, tt, ii, ss, cc, 195)
        g2, _ = autoregressive_rollout(ref, tt, ii, ss, cc, 195)
    assert torch.equal(g1, g2), "自回归 rollout 非确定"
    assert bool((g1[:, :195] == -1).all()), "生成区之外不应有预测（无泄漏回填）"
    # 泄漏硬测：生成区（≥K）的真值给垃圾 → rollout 结果必须完全不变
    tt_bad = tt.clone()
    ii_bad = ii.clone()
    tt_bad[:, 195:] = (tt[:, 195:] + 1) % cfg["vocab"]
    ii_bad[:, 195:] = (ii[:, 195:] + 1) % cfg["vocab"]
    with torch.no_grad():
        g3, gc3 = autoregressive_rollout(ref, tt_bad, ii_bad, ss, cc, 195)
    assert torch.equal(g1, g3), \
        "rollout 读取了生成区真值（token 槽泄漏：改真值改变了生成结果）"
    assert bool((g1[:, 195:] == gc3[:, 195:]).all()), "token/intent 两槽回填不一致"
    # 正向断言（C2）：intent 槽确被 forward 消费——把**前缀区** intent 整体置乱后，
    # 生成区结果必须变化（前缀 intent 每步都进前向，若模型忽略它则此断言失败）。
    # 注意与上面泄漏硬测相反：这里改的是被消费的前缀区，不是被回填的生成区。
    ii_scr = ((ii[:, :195] + 1) % cfg["vocab"])
    ii_bad2 = ii.clone()
    ii_bad2[:, :195] = ii_scr
    with torch.no_grad():
        g4, _ = autoregressive_rollout(ref, tt, ii_bad2, ss, cc, 195)
    assert not torch.equal(g1[:, 195:], g4[:, 195:]), \
        "intent 槽未被 forward 消费（前缀区 intent 整体置乱后生成区结果不变）"
    print("[selftest] intent 槽消费正向断言 OK（前缀区 intent 置乱 → 生成区结果改变）")
    print(f"[selftest] rollout 确定性 OK；生成区帧 {WIN - 195}；"
          f"生成区真值改垃圾不改变结果（token/intent 两槽同回填，无泄漏）"
          f"（自测总耗时 {time.time() - t0:.1f}s）")
    print("AXIS_C_SELFTEST_PASS")
    return 0


def run_selftest_errors():
    """三条报错路径各造一次：ckpt 缺失 / 配置键缺失 / val 池空。"""
    mod, mod_path, mod_err = load_train_module()
    if mod is None:
        print(f"[selftest-errors] FAIL：train_author_v0 不可加载：{mod_err}",
              file=sys.stderr)
        return 1
    tmp = tempfile.mkdtemp(prefix="d061b_errors_")
    asm = mod.make_selftest_corpus(tmp, seed=0)
    corpus = mod.build_corpus_index(asm, data_root=None, logger=None)
    cfg = {"d_model": 32, "n_layer": 1, "n_head": 4, "ffn": 64,
           "vocab": corpus["vocab"], "code_min": corpus["code_min"],
           "state_dim": STATE_DIM, "win": WIN, "token_dim": TOKEN_DIM}
    ok_ckpt = os.path.join(tmp, "ckpt_final.pt")
    good = _selftest_ckpt(ok_ckpt, cfg, mod, seed=0)
    bad_cfg = {k: v for k, v in cfg.items() if k != "n_head"}
    bad_ckpt = os.path.join(tmp, "ckpt_missing_key.pt")
    torch.save({"format": "d061a.v0", "step": 0, "model": good.state_dict(),
                "model_cfg": bad_cfg}, bad_ckpt)

    cases = [
        ("ckpt 缺失", _mini_args(ckpt=os.path.join(tmp, "nope.pt"),
                                 assembly_manifest=asm,
                                 out_json=os.path.join(tmp, "o1.json")),
         "ckpt 不存在"),
        ("model_cfg 键缺失", _mini_args(ckpt=bad_ckpt, assembly_manifest=asm,
                                        out_json=os.path.join(tmp, "o2.json")),
         "缺键"),
        ("val 池空", _mini_args(ckpt=ok_ckpt, assembly_manifest=asm,
                                limit_val_windows=0,
                                out_json=os.path.join(tmp, "o3.json")),
         "val 池为空"),
    ]
    failures = []
    for i, (name, args, kw) in enumerate(cases, 1):
        try:
            run_eval(args)
            msg = ""
        except SystemExit as e:
            msg = str(e.code if getattr(e, "code", None) is not None else e)
        except Exception as e:  # noqa: BLE001 —— 非 SystemExit 也算未按契约报错
            msg = f"{type(e).__name__}: {e}"
        hit = bool(msg) and kw in msg
        print(f"[selftest-errors] {i}/3 {name} → {'OK' if hit else 'FAIL'}；"
              f"错误信息：{msg.splitlines()[0] if msg else '(无明确报错)'}")
        if not hit:
            failures.append(name)
    if failures:
        print(f"[selftest-errors] FAIL：{failures}", file=sys.stderr)
        return 1
    print("AXIS_C_SELFTEST_ERRORS_PASS")
    return 0


# ------------------------------------------------------------------ CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="D061b 轴C：作者模型 intent 重写保真度（teacher-forced 前缀 + "
                    "自回归生成区）")
    p.add_argument("--ckpt", default=None,
                   help="D061a ckpt（默认 apt_g1/outputs/d061_author_v0/"
                        "ckpt_final.pt）；模型配置以 ckpt 的 model_cfg 为准；"
                        "与 --v1-ckpt 二选一（v1 优先）")
    p.add_argument("--v1-ckpt", default=None,
                   help="D062-R1 v1 ckpt（train_author_v1.load_author_v1 正规加载；"
                        "逐帧指标口径与 v0 一致，chunk/priv 头不参与）")
    p.add_argument("--meta", default=None,
                   help="meta.json（缺省自动找 ckpt 同目录；缺则身份信封只记 ckpt）")
    p.add_argument("--assembly-manifest", default=None,
                   help="装配 manifest（默认 apt_g1/data/d060/g1_assembly_full/"
                        "assembly_manifest.json）")
    p.add_argument("--data-root", default="data",
                   help="数据根（分片重定位用，默认 data）")
    p.add_argument("--prefix-frames", type=int, default=WIN // 2,
                   help="前 K 帧 teacher-forced（token/intent 两槽真值），默认 "
                        "WIN//2=100")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--val-seed", type=int, default=0)
    p.add_argument("--limit-val-windows", type=int, default=None)
    p.add_argument("--batch-windows", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-json", default="outputs/d061b_intent_fidelity.json")
    p.add_argument("--train-module", default=None,
                   help="train_author_v0.py 路径（缺省取仓库内默认位置）")
    p.add_argument("--selftest", action="store_true",
                   help="本机 CPU 自测（合成微语料 + 小模型，末行 AXIS_C_SELFTEST_PASS）")
    p.add_argument("--selftest-errors", action="store_true",
                   help="三条报错路径自测（ckpt 缺失/model_cfg 键缺失/val 池空）")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        return run_selftest()
    if args.selftest_errors:
        return run_selftest_errors()
    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
