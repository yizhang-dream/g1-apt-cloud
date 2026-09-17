"""D060 G1 消费者测试：统一窗语料的 stub causal transformer（forward + 一步 SGD）。

预注册 = refine-logs/ds/DS_CONTINUOUS_EXECUTION_PLAN.md §5t G1：四源各 ≥100 窗 +
**统一 loader + 消费者测试**（D058 消费者测试先例）。本脚本是"这批窗真的能训"的
最小实证：把 d060_windows.py 产出的窗按 manifest 喂进一个小 causal transformer，
前向 + 一步 SGD，断言形状/数值契约 —— 不训练、不产出模型，只证数据通路与梯度通路。

模型（~1–3M 参数，CPU 可跑）
---------------------------
输入（每窗）：token_stream (200,64) int16 码、state (200,36) f32（jp 29 + quat 4 +
trans 3，owner 2026-09-17 裁决的 canonical 口径，无保留位）、intent_tokens
(200,64) int16 码、command (4,)（-1 哨兵 + has_truth 掩码）、terrain（类型 one-hot +
参数）。序列 = [命令上下文帧, 地形上下文帧] + 200 个状态帧，因果注意力；每帧输出
64×V logits（V = 码域跨度）。**目标 = 下一帧的 token 码**（teacher forcing：
第 t 帧输出预测 token_stream[t+1]）——attention 因果 + intent 与 token 同流（源 2/3
自条件），所以这是真实的一步预测任务而非恒等复制。

用法（本机 torch-CPU；也适用于服务器有 torch 的 venv）
    python d060_consumer_stub.py --manifest <windows/manifest.json> --split train \
        --limit 16 --steps 1 --seed 0
    python d060_consumer_stub.py --synthetic            # 无需任何数据文件的自检
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

TOKEN_DIM = 64
# 命令/地形上下文向量 = command(4) + has_truth(4) + terrain one-hot(3) + 参数(height,noise)
CTX_DIM = 4 * 2 + 3 + 2


def import_d060():
    """import 窗格式脊柱（三种 sys.path 布局：执行根平铺 / 本仓 build 分域 / 包导入）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.getcwd()):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    last = None
    for name in ("d060_windows", "build.d060_windows", "apt_g1.build.d060_windows"):
        try:
            return importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 —— 逐候选回退
            last = exc
    raise ImportError(f"d060_windows 不可 import（最后错误 {type(last).__name__}: {last}）")


W = import_d060()


# ------------------------------------------------------------------ 契约检查
def contract_report(manifest_path, split=None, limit=None):
    """逐个窗跑 §5t 字段契约断言；返回 (n_checked, violations, token_range)。"""
    bad, n = [], 0
    lo, hi = None, None
    for w in W.load_windows(manifest_path, split=split, limit=limit):
        n += 1
        v = W.window_contract_violations(w, win=w["token_stream"].shape[0])
        if not w.get("stem"):
            v.append("stem 缺失")
        if w["command"]["cmd"]["target_vel"] == W.SENTINEL and \
                w["command"]["has_truth"]["target_vel"]:
            v.append("target_vel 哨兵与 has_truth 矛盾")
        t = w["token_stream"]
        lo = int(t.min()) if lo is None else min(lo, int(t.min()))
        hi = int(t.max()) if hi is None else max(hi, int(t.max()))
        if w["terrain_desc"]["type"] not in W.TERRAIN_TYPES:
            v.append(f"terrain 非法 {w['terrain_desc']['type']}")
        if v:
            bad.append({"stem": w.get("stem"), "violations": v})
    return n, bad, (lo, hi)


# ------------------------------------------------------------------ 小模型
def build_model(torch, nn, vocab, state_dim, d_model=160, n_layer=4, n_head=4, ffn=512):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True,
                                              dropout=0.0)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(),
                                     nn.Linear(ffn, d_model))

        def forward(self, x, mask):
            y = self.ln1(x)
            a, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
            x = x + a
            return x + self.mlp(self.ln2(x))

    class TinyCausalWindowTransformer(nn.Module):
        """统一窗 → 64×V logits/帧。command/terrain 作为前缀上下文帧注入。"""

        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab, d_model)
            self.intent_emb = nn.Embedding(vocab, d_model)
            self.state_proj = nn.Linear(state_dim, d_model)
            self.cmd_proj = nn.Sequential(nn.Linear(CTX_DIM, d_model),
                                          nn.GELU(), nn.Linear(d_model, d_model))
            self.pos = nn.Parameter(torch.zeros(1, W.WIN + 2, d_model))
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, TOKEN_DIM * vocab)
            self.vocab = vocab
            self.n_ctx = 2

        def forward(self, tokens_idx, state, intent_idx, ctx_feat):
            b, t, k = tokens_idx.shape
            pos = self.pos[:, :t + self.n_ctx]
            # 帧嵌入 = 状态投影 + 64 个 token 维码嵌入的均值（维间共享表）+ intent 同构
            x = self.state_proj(state)
            x = x + self.tok_emb(tokens_idx).mean(dim=2)
            x = x + self.intent_emb(intent_idx).mean(dim=2)
            ctx = self.cmd_proj(ctx_feat).unsqueeze(1).expand(b, self.n_ctx, -1)
            x = torch.cat([ctx, x], dim=1) + pos
            m = torch.triu(torch.ones(t + self.n_ctx, t + self.n_ctx, dtype=torch.bool,
                                      device=x.device), diagonal=1)
            for blk in self.blocks:
                x = blk(x, m)
            x = self.ln_f(x)[:, self.n_ctx:]
            return self.head(x).view(b, t, TOKEN_DIM, self.vocab)

    return TinyCausalWindowTransformer()


def window_to_tensors(windows, torch, vocab, code_min):
    """窗 dict list → 张量批（token/intent 码平移为非负索引；-1 哨兵保留）。"""
    tok = np.stack([w["token_stream"] for w in windows]).astype(np.int64)
    it = np.stack([w["intent_tokens"] for w in windows]).astype(np.int64)
    st = np.stack([w["state"] for w in windows]).astype(np.float32)
    cmdf = []
    for w in windows:
        c = W.command_vector(w["command"])
        ht = W.command_has_truth_vector(w["command"]).astype(np.float32)
        terr = np.zeros(len(W.TERRAIN_TYPES), dtype=np.float32)
        terr[W.TERRAIN_TYPES.index(w["terrain_desc"]["type"])] = 1.0
        p = w["terrain_desc"]["params"]
        terr = np.concatenate([terr, np.asarray(
            [float(p.get("height", 0.0)), float(p.get("noise", 0.0))], dtype=np.float32)])
        cmdf.append(np.concatenate([c, ht, terr]))
    ctx = np.stack(cmdf).astype(np.float32)
    return (torch.from_numpy(tok - code_min), torch.from_numpy(st),
            torch.from_numpy(it - code_min), torch.from_numpy(ctx))


def run_smoke(manifest_path, split, limit, steps, seed, lr, device, synthetic=False,
              quiet=False):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    t0 = time.time()
    out = {"script": os.path.abspath(__file__), "manifest": manifest_path,
           "split": split, "limit": limit, "steps": steps, "seed": seed,
           "lr": lr, "device": device, "synthetic": bool(synthetic)}
    if synthetic:
        rng = np.random.default_rng(seed)
        n, code_min, vocab = limit or 8, -14, 28
        wins = []
        for i in range(n):
            tok = rng.integers(0, vocab, (W.WIN, TOKEN_DIM)).astype(np.int16) + code_min
            st = rng.normal(0, 0.3, (W.WIN, W.STATE_DIM)).astype(np.float32)
            wins.append({"token_stream": tok, "intent_tokens": tok.copy(), "state": st,
                         "command": W.make_command(0.4),
                         "terrain_desc": W.make_terrain(
                             W.TERRAIN_TYPES[i % 3], height=0.5, noise=0.04),
                         "meta": {"stem": f"synth_{i:03d}", "token_scale": 16},
                         "stem": f"synth_{i:03d}", "split": "train", "speedA": 0.4})
        n_checked, bad, rng_codes = len(wins), [], (code_min, code_min + vocab - 1)
    else:
        doc = json.load(open(manifest_path, encoding="utf-8"))
        code_min = int(doc["counts"]["token_min"])
        code_max = int(doc["counts"]["token_max"])
        vocab = code_max - code_min + 1
        wins = list(W.load_windows(manifest_path, split=split, limit=limit))
        n_checked, bad, rng_codes = contract_report(manifest_path, split=split,
                                                   limit=limit)
        if not wins:
            raise SystemExit("没有可用的窗（manifest/split/limit 组合为空）")
    out.update({"n_windows": len(wins), "code_min": code_min, "vocab": vocab,
                "contract_checked": n_checked, "contract_violations": bad,
                "contract_ok": not bad,
                "token_src_range": [int(rng_codes[0]), int(rng_codes[1])]})
    tok, st, it, ctx = window_to_tensors(wins, torch, vocab, code_min)
    out["batch_shapes"] = {"token_stream": list(tok.shape), "state": list(st.shape),
                           "intent_tokens": list(it.shape), "ctx_feat": list(ctx.shape)}
    assert tok.shape[1:] == (W.WIN, TOKEN_DIM), f"token 形状契约违例 {tok.shape}"
    assert st.shape[1:] == (W.WIN, W.STATE_DIM), f"state 形状契约违例 {st.shape}"
    assert ctx.shape[1] == CTX_DIM, f"ctx 形状 {ctx.shape} != (*,{CTX_DIM})"
    model = build_model(torch, nn, vocab, st.shape[-1]).to(device)
    out["n_params"] = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    model.train()
    losses = []
    if steps < 1:
        raise SystemExit("--steps 需 ≥1")
    for s in range(steps):
        logits = model(tok, st, it, ctx)
        # 下一帧预测：第 t 帧输出 vs token_stream[t+1]
        lg = logits[:, :-1]
        tgt = tok[:, 1:]
        loss = nn.functional.cross_entropy(
            lg.reshape(-1, vocab), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # 返回的是裁剪前总范数（阈值 1e9 实际不裁剪，只为取数）
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
        opt.step()
        with torch.no_grad():
            acc = float((lg.argmax(-1) == tgt).float().mean())
        losses.append({"step": s, "loss": float(loss.detach()), "grad_norm": gnorm,
                       "next_token_top1_acc": acc})
    delta = max(float((v.detach() - before[k]).abs().max())
                for k, v in model.named_parameters())
    finite = all(np.isfinite(l["loss"]) for l in losses)
    out.update({"logits_shape": list(logits.shape), "losses": losses,
                "loss_first": losses[0]["loss"], "loss_last": losses[-1]["loss"],
                "max_param_delta_after_step": delta,
                "loss_finite": finite, "grads_finite": np.isfinite(
                    losses[-1]["grad_norm"]).item(),
                "params_changed": delta > 0.0,
                "elapsed_s": round(time.time() - t0, 2)})
    ok = (finite and out["params_changed"] and out["contract_ok"]
          and losses[-1]["grad_norm"] > 0.0)
    out["verdict"] = "PASS" if ok else "FAIL"
    if not quiet:
        print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="D060 G1 消费者测试（stub causal transformer）")
    ap.add_argument("--manifest", default=None, help="<windows>/manifest.json")
    ap.add_argument("--split", default="train", choices=[None, "train", "heldout"])
    ap.add_argument("--limit", type=int, default=16, help="取前 N 窗（默认 16）")
    ap.add_argument("--steps", type=int, default=1, help="SGD 步数（G1 口径 = 1）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--synthetic", action="store_true", help="无需数据文件的自检")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)
    if not args.synthetic and not args.manifest:
        raise SystemExit("需 --manifest <windows/manifest.json> 或 --synthetic")
    out = run_smoke(args.manifest, args.split, args.limit, args.steps, args.seed,
                    args.lr, args.device, synthetic=args.synthetic)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"[consumer] -> {args.out_json}")
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
