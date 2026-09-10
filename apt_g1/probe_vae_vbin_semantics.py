"""D048m 离线探针：E39 VAE 速度档（vb）条件语义——token 层步态节奏排序判别
（纯 torch 解码侧，零仿真，CPU 可跑）。

背景（D048l 闭环反转）：D048l 在 apt_flat_env 评测里用 force_vbin 强制速度档
闭环跑，实测 b1 -> 0.61 m/s、b2 -> 0.45 m/s——与名义档位方向相反（vb 越大应越快，
E39 的 vb 本就是 walk 相位变化率三分位：bin0 慢 / bin2 快）。本探针判别这个
反转发生在 **VAE 条件层** 还是 **闭环执行层**。

方法：z 固定（先验采样 torch.randn，种子固定），walk-clock 相位 0 -> 2pi 均匀扫
n_steps，对 vb in {0,1,2} 各解码一整条 token 流（db 固定前向档 4，bin4=+x fwd，
见 DirSpeedPhaseTokenVAE docstring 与 env 的 atan2 分桶公式），同 z 同相位对齐。
对每条流测：
  1) rate = mean|wrap(dphi)|——E39 训练标签的生成量 walk_phase_rate
     （train_token_vae_e39.py :121-135：token 帧均值中心化 -> 协方差 -> eigh
     top-2 -> 投影 -> arctan2 相位 -> wrap 到 [-pi,pi) 差分 -> |dphi|；本探针
     去 mode 掩码、全帧当 walk）。rate 即「解码 token 流的步态相位变化率」，
     是 vb 标签（E39 v-bin 三分位）的直接生成量。
  2) token 均值范数（||mean_t tok_t||_2，流的整体幅度刻度）。
  3) 档间 token L2 距离 d(vb0,vb1)/d(vb1,vb2)/d(vb0,vb2)（逐相位距离的流级
     平均），并报其相对 token 均值范数的比例（vb 条件是否实质性移动 token）。

判读分叉（预注册，threshold=10%）：
  monotone_up  rate(vb2)>rate(vb1)>rate(vb0) 且相邻相对差均 >10%
               -> VAE 条件层排序正常，D048l 的 b1/b2 反转应归因闭环执行层
               （策略 z 分布 / residual 通道 / PD 跟踪等）。
  inverted     rate 严格降序（相邻相对差均 <-10%）
               -> VAE 条件层自身反转：vb 越大解码出的步态节奏反而越慢，
               直接解释闭环反转，无需执行层背锅。
  flat         相邻相对差绝对值均 <=10%
               -> vb 条件对解码节奏几乎无影响，反转不能归因条件层的节奏排序。
  mixed        其余（非单调/不一致），看 per_z 明细。

接口镜像（以代码为准，逐字对照见 run_probe 内注释）：
  - 类：apt_g1.isaac.token_window_vae.DirSpeedPhaseTokenVAE（env 同款 decode-only
    类；apt_flat_env.py :456-477：构造 (n_vbins=3, n_dbins=8) -> .to(device) ->
    load_state_dict(torch.load(vae_path, map_location=device), strict=False)
    （ckpt 另带 encoder/mu/logvar，decoder 才是必需）-> vae.eval()）。本探针
    从 state_dict 形状反推各维（canonical e39 = 16/64/10/256/3/8，与 env 默认
    一致），对 fork/变体 ckpt 更稳。
  - 调用：apt_flat_env.py :699-718 逐字镜像 decode(phase, sc, vb, db)——注意
    形参名陷阱：decode 第 1 参名义 z 实际吃的是**策略潜变量**（_pre_physics_step
    :988-989 的 actions[:, :16]），第 2 参名义 phase 实际吃的是 walk clock 的
    sc = torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)；vb/db 是 long。
  - 相位语义：env 的 _latent_phase 从 0 起每控制步固定步进（pca.npz 的 rate，
    latent_cmd_phase_rate=False 时），mod 2pi；decode 只通过 (sin, cos) 感知
    相位，故探针在 [0, 2pi) 均匀取 n_steps 个相位即等价覆盖时钟可达全集。

用法（服务器，cwd=仓根）：python apt_g1/probe_vae_vbin_semantics.py
    [--vae /home/cvgluser/ros2_data/apt_g1/outputs/token_vae_e39/vae.pt]
    [--n-z 32] [--n-steps 200] [--seed 0] [--device cpu]
    [--out outputs/probe_vae_vbin_semantics.json]
    [--tokens-npz path/to/token.npy.npz]  # 可选：encode 真实 token 取 mu 作 z 族
产物：--out JSON（args/vae_md5/code_md5/per_z/aggregate/verdict）+ stdout 人类
可读表。依赖：torch + numpy + stdlib。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

# 仓根入 sys.path（脚本位于 <repo>/apt_g1/ 下），保证 apt_g1.* 可导入
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from apt_g1.isaac.token_window_vae import DirSpeedPhaseTokenVAE as DecodeVAE  # noqa: E402

DB_FORWARD = 4  # db 固定前向档：DirSpeedPhaseTokenVAE docstring「bin 4 = +x forward」，
                # env 分桶公式 atan2(cmd_vy, cmd_vx)=0 -> floor(pi/(2pi)*8)=4，已代码确认
REL_THRESHOLD = 0.10  # 相邻档 rate 相对差判升降 / token 距离实质性判定的门槛


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_dims_from_state_dict(sd: dict) -> dict:
    """从 vae.pt state_dict 形状反推 VAE 构造维（canonical e39 与 env 默认一致）。

    decoder 输入 = latent + phase_dim(2) + speed_embed(8) + dir_embed(8)。
    """
    w0 = sd["decoder.0.weight"]           # (hidden, latent + 2 + 8 + 8)
    se = sd["speed_embed.weight"]         # (n_vbins, 8)
    de = sd["dir_embed.weight"]           # (n_dbins, 8)
    hidden, cond_in = int(w0.shape[0]), int(w0.shape[1])
    dims = {
        "hidden_dim": hidden,
        "latent_dim": cond_in - 2 - int(se.shape[1]) - int(de.shape[1]),
        "n_vbins": int(se.shape[0]),
        "n_dbins": int(de.shape[0]),
        "token_dim": int(sd["decoder.4.weight"].shape[0]),
        "phase_dim": 2,
    }
    # e39 canonical ckpt 带 encoder（env 用 strict=False 丢弃）；有则反推 window
    if "encoder.0.weight" in sd:
        dims["window"] = int(sd["encoder.0.weight"].shape[1]) // dims["token_dim"]
    else:
        dims["window"] = 10
    return dims


def walk_phase_rate_unmasked(tok: np.ndarray):
    """E39 walk_phase_rate（train_token_vae_e39.py :121-135）去 mode 掩码版。

    全帧当 walk；算法逐行保持一致（float32 口径、eigh 升序特征值取
    V[:, ::-1][:, :2]、wrap 到 [-pi,pi) 差分、rate[0]=rate[1]）。返回
    (pmean, V2, rate, phi)；rate 标量口径 = mean|diff(phi)|（与 pca.npz 的
    rate 一致），由调用方对 phi 取 diff 均值。
    """
    walk = tok
    pmean = walk.mean(0)
    c = walk - pmean
    cov = c.T @ c / len(c)
    ev, V = np.linalg.eigh(cov)
    V2 = V[:, ::-1][:, :2].astype(np.float32)
    proj = (tok - pmean) @ V2
    phi = np.arctan2(proj[:, 1], proj[:, 0]).astype(np.float32)
    rate = np.zeros(len(tok), dtype=np.float32)
    dphi = np.diff(phi)
    dphi = np.mod(dphi + np.pi, 2.0 * np.pi) - np.pi
    rate[1:] = np.abs(dphi)
    rate[0] = rate[1]
    return pmean.astype(np.float32), V2, rate, phi


def load_vae(path: str, device: str):
    """镜像 env 加载（apt_flat_env.py :456-477）：

        vae = DirSpeedPhaseTokenVAE(n_vbins=3, n_dbins=8).to(device)
        vae.load_state_dict(torch.load(path, map_location=device), strict=False)
        vae.eval()

    差异仅一处：构造维从 ckpt 形状反推（canonical e39 时与 env 默认全等）。
    """
    sd = torch.load(path, map_location=device)
    dims = infer_dims_from_state_dict(sd)
    vae = DecodeVAE(
        token_dim=dims["token_dim"], window=dims["window"],
        latent_dim=dims["latent_dim"], hidden_dim=dims["hidden_dim"],
        phase_dim=dims["phase_dim"], n_vbins=dims["n_vbins"],
        n_dbins=dims["n_dbins"],
    ).to(device)
    vae.load_state_dict(sd, strict=False)  # ckpt 另带 encoder；只需 decoder
    vae.eval()
    return vae, dims


def build_z_family(vae, dims, args, device):
    """z 族：默认先验 torch.randn(n_z, latent)（CPU 生成器播种，跨设备确定性）；
    给 --tokens-npz 时 encode 真实 token 窗取 mu 作 z 族（需 ckpt 带 encoder，
    e39 canonical 有——env 的 decode-only 类不带 encoder，故此处临时用训练侧
    同构类 DirSpeedPhaseTokenVAE（train_token_vae_e39.py，decoder/嵌入参数名
    与 token_window_vae 版逐字相同，state_dict 完全兼容）加载同一 ckpt）。"""
    n_z = args.n_z
    if not args.tokens_npz:
        g = torch.Generator().manual_seed(args.seed)
        return torch.randn(n_z, dims["latent_dim"], generator=g,
                           dtype=torch.float32).to(device), "prior_randn"
    from apt_g1.train_token_vae_e39 import (DirSpeedPhaseTokenVAE as FullVAE,
                                            build_windows)
    arr = np.load(args.tokens_npz)
    tok = (arr["token"] if "token" in arr.files else arr[arr.files[0]]).astype(np.float32)
    full = FullVAE(token_dim=dims["token_dim"], window=dims["window"],
                   latent_dim=dims["latent_dim"], hidden_dim=dims["hidden_dim"],
                   phase_dim=dims["phase_dim"], n_vbins=dims["n_vbins"],
                   n_dbins=dims["n_dbins"]).to(device)
    full.load_state_dict(torch.load(args.vae, map_location=device), strict=False)
    full.eval()
    x = torch.from_numpy(build_windows(tok, dims["window"])).to(device)
    with torch.no_grad():
        mu = full.encode(x)[0]  # (n_frames, latent)
    if len(mu) > n_z:  # 均匀下采样到 n_z
        idx = np.linspace(0, len(mu) - 1, n_z).astype(np.int64)
        mu = mu[torch.from_numpy(idx).to(device)]
    return mu.detach().float(), f"encoded_mu_from {args.tokens_npz} (n_frames={len(tok)})"


def run_probe(vae, dims, z_family, args, device):
    """同 z / 同相位扫 vb in {0,1,2} 解码 + 度量。decode 调用逐字镜像
    apt_flat_env.py :695-718（latent_mode 分支，逐字对照）：

        with torch.no_grad():
            phi = self._latent_phase
            sc = torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)
            ...
            vb = torch.bucketize(cmd_v, edges).clamp(0, n - 1)  # force_vbin>=0 时整档覆写
            ang = torch.atan2(self._commands[:, 1], self._commands[:, 0])
            db = torch.floor((ang + math.pi) / (2.0 * math.pi) * 8).long() % 8
            tokens = self._vae.decode(phase, sc, vb, db).detach()

    对照：decode 第 1 参 phase = 策略 z（本探针 = 先验/编码 z，形状 (N, latent)）；
    第 2 参 sc = walk clock (sin, cos)，本探针 phi 在 [0, 2pi) 均匀 n_steps 个
    （env 时钟固定步进 mod 2pi，decode 只见 (sin,cos)，集合等价）；vb 整条流
    常量（= D048l force_vbin 语义，覆写自然分桶）；db 常量 4（+x fwd，公式见上）。
    detach/dtype/device 同 env：no_grad + .detach()、float32 条件、long 档位。
    """
    n_steps = args.n_steps
    # env 相位语义镜像：标量 walk clock phi，mod 2pi；sc = stack([sin, cos], dim=1)
    phi = (torch.arange(n_steps, dtype=torch.float32, device=device)
           * (math.tau / n_steps))  # [0, 2pi)，端点不重复（env 的 % tau 亦不可达 2pi）
    sc = torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)  # (n_steps, 2) float32
    db_t = torch.full((n_steps,), DB_FORWARD, dtype=torch.long, device=device)

    per_z, streams = [], {}
    for zi in range(len(z_family)):
        z_row = z_family[zi:zi + 1].expand(n_steps, -1)  # (n_steps, latent)
        rec = {"zi": zi, "z_norm": float(z_family[zi].norm().item()),
               "rate": {}, "token_mean_norm": {}, "dist": {}}
        for vb in range(dims["n_vbins"]):
            vb_t = torch.full((n_steps,), vb, dtype=torch.long, device=device)
            with torch.no_grad():
                # ^^^ env 逐字镜像：tokens = self._vae.decode(phase, sc, vb, db).detach()
                tokens = vae.decode(z_row, sc, vb_t, db_t).detach()
            tok = tokens.cpu().numpy().astype(np.float32)  # (n_steps, token_dim)
            streams[(zi, vb)] = tok
            _, _, _, phi_est = walk_phase_rate_unmasked(tok)
            rec["rate"][str(vb)] = float(np.abs(np.diff(phi_est)).mean())
            rec["token_mean_norm"][str(vb)] = float(np.linalg.norm(tok.mean(0)))
        for a, b in ((0, 1), (1, 2), (0, 2)):
            diff = streams[(zi, a)] - streams[(zi, b)]
            rec["dist"][f"d{a}{b}"] = float(np.linalg.norm(diff, axis=1).mean())
        scale = float(np.mean([rec["token_mean_norm"][str(v)]
                               for v in range(dims["n_vbins"])]))
        rec["dist_rel"] = {k: (v / scale if scale > 0 else float("nan"))
                           for k, v in rec["dist"].items()}
        per_z.append(rec)
    return per_z


def aggregate(per_z, dims):
    def ms(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "n": int(len(vals))}

    def rel(next_, prev_):
        return (next_ - prev_) / prev_ if abs(prev_) > 1e-12 else float("nan")

    r = {v: ms([pz["rate"][str(v)] for pz in per_z]) for v in range(dims["n_vbins"])}
    rel01 = rel(r[1]["mean"], r[0]["mean"])
    rel12 = rel(r[2]["mean"], r[1]["mean"])
    dist = {f"d{a}{b}": ms([pz["dist"][f"d{a}{b}"] for pz in per_z])
            for a, b in ((0, 1), (1, 2), (0, 2))}
    dist_rel = {f"d{a}{b}": ms([pz["dist_rel"][f"d{a}{b}"] for pz in per_z])
                for a, b in ((0, 1), (1, 2), (0, 2))}
    tnorm = {v: ms([pz["token_mean_norm"][str(v)] for pz in per_z])
             for v in range(dims["n_vbins"])}
    return {"rate_vb": {str(v): r[v] for v in r},
            "rate_rel_diff": {"vb1_vs_vb0": float(rel01), "vb2_vs_vb1": float(rel12)},
            "token_mean_norm_vb": {str(v): tnorm[v] for v in tnorm},
            "token_dist": dist, "token_dist_rel": dist_rel,
            "rel_threshold": REL_THRESHOLD,
            "rate_rel01": rel01, "rate_rel12": rel12}


def classify_rate_ordering(rel01: float, rel12: float, thr: float) -> str:
    """预注册判读：rate(vb2)>rate(vb1)>rate(vb0)（相邻相对差 >10% 才算升降）。"""
    def cls(x):
        return "up" if x > thr else ("down" if x < -thr else "flat")
    c01, c12 = cls(rel01), cls(rel12)
    if c01 == "up" and c12 == "up":
        return "monotone_up"
    if c01 == "down" and c12 == "down":
        return "inverted"
    if c01 == "flat" and c12 == "flat":
        return "flat"
    return "mixed"


VERDICT_TEXT = {
    "monotone_up": "解码节奏随 vb 单调升 -> VAE 条件层排序正常；D048l 的 b1->0.61/"
                   "b2->0.45 闭环反转应归因闭环执行层（策略 z 分布 / residual / 跟踪）。",
    "inverted": "解码节奏随 vb 反转（降序）-> VAE 条件层自身把档位节奏写反，"
                "可直接解释 D048l 闭环反转。",
    "flat": "vb 条件对解码节奏无实质影响（|相邻相对差|<=10%）-> 反转不能归因 "
            "VAE 条件层的节奏排序；档位差异可能主要走 z 分布或执行层。",
    "mixed": "排序非单调/不一致 -> 看 per_z 明细与逐 z 分布再判。",
}


def main():
    ap = argparse.ArgumentParser(
        description="D048m: E39 VAE v-bin conditional semantics probe (decode-only, zero sim)")
    ap.add_argument("--vae", type=str,
                    default="/home/cvgluser/ros2_data/apt_g1/outputs/token_vae_e39/vae.pt",
                    help="E39 vae.pt 路径（state_dict，含 encoder，decoder 为必需）")
    ap.add_argument("--n-z", dest="n_z", type=int, default=32,
                    help="z 族大小（先验采样数 / npz 编码后下采样目标数）")
    ap.add_argument("--n-steps", dest="n_steps", type=int, default=200,
                    help="walk clock 相位扫描步数（[0, 2pi) 均匀）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="outputs/probe_vae_vbin_semantics.json")
    ap.add_argument("--tokens-npz", dest="tokens_npz", type=str, default=None,
                    help="可选：真实 token npz（键 'token' 或首个二维数组），"
                         "给了则 encode 取 mu 作 z 族（替代先验 randn）")
    args = ap.parse_args()
    if args.n_steps < 4:
        ap.error("--n-steps must be >= 4")
    if args.n_z < 2:
        ap.error("--n-z must be >= 2 (std over z)")

    if not os.path.exists(args.vae):
        sys.exit(f"[D048m] vae.pt not found: {args.vae}")
    device = args.device
    vae, dims = load_vae(args.vae, device)
    z_family, z_source = build_z_family(vae, dims, args, device)
    print(f"[D048m] vae={args.vae} md5={md5_file(args.vae)}")
    print(f"[D048m] dims={dims} z_source={z_source} z={tuple(z_family.shape)} "
          f"n_steps={args.n_steps} db={DB_FORWARD} (+x fwd) device={device}")

    per_z = run_probe(vae, dims, z_family, args, device)
    agg = aggregate(per_z, dims)
    verdict = classify_rate_ordering(agg["rate_rel01"], agg["rate_rel12"], REL_THRESHOLD)

    # token 距离实质性：vb 条件是否实质性移动 token（相对流自身幅度刻度，>10%）
    rel_d02 = agg["token_dist_rel"]["d02"]["mean"]
    token_moves = ("substantial" if rel_d02 > REL_THRESHOLD else "marginal")

    out = {
        "probe": "D048m probe_vae_vbin_semantics",
        "args": {**vars(args), "db_forward": DB_FORWARD,
                 "rel_threshold": REL_THRESHOLD},
        "vae_md5": md5_file(args.vae),
        "code_md5": md5_file(os.path.abspath(__file__)),
        "vae_dims": dims,
        "z_source": z_source,
        "metric": "walk_phase_rate_unmasked (E39 train_token_vae_e39.walk_phase_rate, "
                  "mode 掩码移除); rate 标量 = mean|wrap(dphi)| (与 pca.npz rate 同口径)",
        "per_z": per_z,
        "aggregate": agg,
        "verdict": {
            "rate_ordering": verdict,
            "interpretation": VERDICT_TEXT[verdict],
            "rate_rel_diff": {"vb1_vs_vb0": agg["rate_rel_diff"]["vb1_vs_vb0"],
                              "vb2_vs_vb1": agg["rate_rel_diff"]["vb2_vs_vb1"]},
            "rule": "rate(vb2)>rate(vb1)>rate(vb0), 相邻相对差 >10% 才算升降",
            "vb_condition_moves_tokens": token_moves,
            "token_dist_rel_d02_mean": float(rel_d02),
        },
    }
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[D048m] saved {args.out}")

    # ---- 人类可读表 ----
    print("\n== D048m: E39 VAE v-bin decode semantics (zero sim) ==")
    print(f"{'vb':>2} | {'rate mean±std (rad/frame)':>28} | {'token mean norm':>16}")
    for v in range(dims["n_vbins"]):
        rr, tn = agg["rate_vb"][str(v)], agg["token_mean_norm_vb"][str(v)]
        print(f"{v:>2} | {rr['mean']:>14.6f} ± {rr['std']:.6f} | {tn['mean']:>16.4f}")
    print(f"adjacent rel diff: vb1-vs-vb0 = {agg['rate_rel01']*100:+.1f}%   "
          f"vb2-vs-vb1 = {agg['rate_rel12']*100:+.1f}%   (threshold ±{REL_THRESHOLD*100:.0f}%)")
    print("token dist (stream-mean L2, same z & phase-aligned):")
    for k in ("d01", "d12", "d02"):
        d, dr = agg["token_dist"][k], agg["token_dist_rel"][k]
        print(f"  {k}: {d['mean']:.4f} ± {d['std']:.4f}   rel-to-norm {dr['mean']*100:.1f}%")
    print(f"VERDICT: {verdict}")
    print(f"  {VERDICT_TEXT[verdict]}")


if __name__ == "__main__":
    main()
