"""D046b 交叉 z 探针 v3.1（单轴交叉转向矩阵，owner 评审三轮修正版）——判别 v2.1
VAE 条件通路「真琴键 vs 贴纸」。

背景：v2.1 探针（probe_vae_v2.py）显示 locomotion 窗 z + 条件 IDLE 解码不向 IDLE
质心转向（0.004）；而此前「条件=WALK 输出像 WALK」的高对角可能被 z 驱动混杂。
本探针把 z 驱动与条件驱动分开：**每次只交叉一个条件轴**，其余条件轴保持窗口
自身真值——z 从条件值 i 的训练窗取（确定性 mu），被测轴条件置 j 解码，输出对
train 前向按真值条件分组的 token 质心做最近质心判定 → 转向矩阵
M[i][j] = P(命中 j)。若输出跟条件走，off-diag 高；若跟 z 走，off-diag 高的
P(命中 i)。

v2 修正（owner 代码级评审 2026-09-07 五项发现，v1 数值矩阵仍可用、机制结论作废）：
  ① z_follow 实现错：v1 取对角均值（=完美跟条件时反而 1.0）。预注册定义是
     off-diagonal 格的 P(pred==i)（输出仍命中 z 源类）。v2 每格保存完整命中
     向量，z_follow = off-diag P(pred==i) 均值。v1 台账 z/c=1.53 作废。
  ② mode 输出名错位：v1 把矩阵位置当 embed_idx 查名（实际参与标签
     [0,2,3,4,5,7,8,9]），argmax 目标名全体错位（如 WALK→WALK 0.724 被写成
     预测 SLOW_WALK）。v2 一律经 values[t] 映射，出报告前打印映射核对。
  ③ 质心未校准 + 基线缺失：v1 质心取模型自重建均值。v2 改用真实 train 窗末
     token（y，与 decode 输出同空间）按真值分组均值做质心；新增同条件（对角）
     识别率=可辨识性校准；新增正确基线
     P(命中 j | z_i, cond_i)（同 z 不改条件=自条件前向），逐轴报
     mean delta = 交叉 off-diag − 基线 off-diag。
  ④ (mode,UL) 训练支持分层：交叉 mode 时 UL 保持窗口真值，而训练集 sym 全来自
     IDLE、零 (IDLE,none)——大量格子是训练未见组合，IDLE 归因混入 OOD。
     v2 从 train 窗标签构建 (mode,UL) 支持集 S，mode/ul 轴每格按交叉后组合
     是否 ∈S 拆 trained-combo / unseen-combo 两套矩阵；IDLE 归因只允许引用
     trained-combo 格；支持矩阵本身作为产物输出。
  ⑤ （门脚本侧，不在本脚本）D047 path ratio 时间窗 diff，见
     b3p_gate_isaac.py playback/hold instrumentation。
  ⑥ [v3] 训练支持集升为 (mode,UL,vb,db) 四元组合（owner 评审二轮 P1：R1 的
     (mode,UL) 二元分层低估 OOD——IDLE 进入列过二元检查的 5,263 窗仅 465 窗
     （8.8%）四元有训练支持，phase/z 分布尚未核查）。四轴交叉格一律按四元
     支持分层，有效窗数随矩阵落盘；trained 覆盖过薄时判「不可识别」，不得
     定位到 decoder 条件响应。phase 为连续量不入组合键，作为覆盖局限登记。
  ⑦ [v3] 真 token 段级留出识别校准（owner 评审二轮 P1：diag_identification
     测的是重建输出，混表示误差与读出误差；方向轴连训练真值自身都识别不好
     ——db 真值宏识别 ~0.22）。同一最近质心读出对真实 train 窗末 token 报
     resub + 段级留出（seg_id 奇偶两半，同段窗不跨半）宏识别率/混淆矩阵；
     留出宏识别 < 2*chance -> verdict=readout_unidentifiable（该轴指标不可
     判读，低交叉命中不可归因）。执行侧保真验证（朝向/速度/qMAE）另号做。
  ⑧ [v3] 判定措辞降级（owner 评审二轮 P2：机器输出不得再出「架构问题证实」；
     z 的均值/方差也不能证明其含任务相关信息）：verdict 分支名统一
     *_on_readout / readout_unidentifiable；z_diag 解读收窄为「未观察到所测
     top-8 投影上的方差塌缩」，不构成 z 含任务相关信息的证明。
  ⑨ [v3.1] 留出分组改按源原始段（owner 评审三轮 P1：v3 按过采样后段序号奇偶
     分半，而构建器过采样按 mode 预算整段/截段复制原始片段——v21 材料
     99 原始段展开 215 段、39 个原始段跨两侧（服务器实证，复制 token 逐位
     一致），同源窗相关性抬高留出识别）。v3.1 按源原始段下标（stem 升序）
     奇偶分半，副本继承源段组别，同源窗必不跨半；身份来源=构建器落盘
     segment_source.npy（build_b4lite_vae_inputs_v2 v3.1 起新材料），现役
     材料无该文件则从 build_meta 的 oversample_plan 确定性重放（copies 计数
     +逐段帧数+总帧数三重 checksum，不符即报错）；另附 stem_hash /
     stem_halves 两个固定对照分组作敏感性波动报告（预注册门只认主分组）。
     源段分组仍是片段级而非演员独立终评，仅消除同片段泄漏。
  ⑩ [v3.1] 留出识别门加覆盖检查（owner 评审三轮 P2：v3 缺类时静默缩小分类
     任务（feas_t 只留两侧皆有窗的类）、随机线仍按原类数 K——8 类仅 2 类
     可留出且 token 全同会以宏识别 0.5 ≥ 2/8 误放行）。任一目标类无双侧窗
     → verdict=readout_unidentifiable（留出覆盖不足，缺类单列），缩小任务
     的宏识别只作描述、不得对完整任务随机线放行。

预注册判读量（owner 指令 2026-09-07；z_follow 按 ① 修正实现；v3 增补第一道门）：
  cond_follow = 逐轴 off-diagonal (i≠j) 的 P(命中 j) 均值
  z_follow    = 逐轴 off-diagonal (i≠j) 的 P(命中 i) 均值
  baseline    = off-diagonal (i≠j) 的 P(命中 j | z_i, cond_i) 均值（v2 增补）
  chance      = 1/k
  判定（逐轴，按序，v3.1 修订）：任一目标类无双侧留出窗 ->
  readout_unidentifiable（覆盖不足）；elif 真 token 源段留出宏识别 <
  2*chance -> readout_unidentifiable（该轴指标不可判读，不进入下列三选一）；
  elif cond_follow >= 2*chance -> cond_available_on_readout；
  elif z_follow >= 3*cond_follow -> z_dominant_on_readout；else ->
  partial_on_readout。措辞只允许到「该指标下响应强弱」，不做机制实锤断言
  （owner 评审口径；v1「架构问题证实」措辞撤回）。
样本门槛：矩阵行（z 组）>= --min-win（默认 150）训练窗（respect segment_bounds
+ keep mask）；低于门槛的条件值不入阵、单独登记。

J9 复测：mode 矩阵 IDLE 行（z_IDLE + 条件 j -> 能否被命令离开 hub）与 IDLE 列
（z_i + 条件 IDLE -> 能否被命令进入 hub）直接回答「hub 能否被命令」；列与
v2.1 探针 dev-z 版 0.004 直接对照。v2 起行/列另拆 trained/unseen-combo。

诊断：IDLE z vs locomotion z 分布（train mu 全体按 locomotion 子集 PCA 的前
8 主成分投影：逐 PC 标准化均值位移 |Δmean|/std_loco 与 std 比）——排除
「IDLE z 本身无信息/塌缩」的平凡解释。

注意：mode 轴按训练窗实况取值——v2.1 中 SLOW_WALK(1)/INJURED_WALK(6) 为 T-fam
零训练窗，指令所列 9 座中 INJURED 不满足「z 从训练窗口取」，实做 8×8（剩余
mode 全部 >=150 窗门槛运行时强制校验并登记计数）。

Usage (CVGL det 容器或 lab-ts venv_isaac，纯 torch 前向，分钟级):
  python probe_vae_cross.py \
    --run-dir /home/cvgluser/ros2_data/g1_b4lite_probe/vae_v21/run1 \
    --inputs-dir /home/cvgluser/ros2_data/g1_b4lite_probe/vae_inputs_v21
产出 <run-dir>/../probe_cross_v3_1/{metrics_d046b_v3_1.json, summary_d046b_v3_1.txt}
（v1/v2/v3 产物在 probe_cross/、probe_cross_v2/、probe_cross_v3/ 不覆盖，append-only
对照；v3 留出数字系副本跨侧泄漏口径，以其 stem_hash 敏感性/R3 重跑为准）
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from train_token_vae_e39_v2 import (DirSpeedPhaseTokenVAE, build_windows_bounded,
                                    per_frame_labels, phi_rate_from)
try:                      # 服务器执行根平铺 import / 仓库根包 import 双兼容
    from holdout_ident import (derive_segment_sources, holdout_splits,
                               nearest_centroid_holdout)
except ImportError:
    from apt_g1.holdout_ident import (derive_segment_sources, holdout_splits,
                                      nearest_centroid_holdout)

UL_NAMES = {0: "none", 1: "sym", 2: "asym"}


def load_split(d: str, use_ul: bool):
    tok = np.load(os.path.join(d, "token.npy")).astype(np.float32)
    bnd = np.load(os.path.join(d, "segment_bounds.npy"))
    mode = np.load(os.path.join(d, "mode_id.npy")).astype(np.int64)
    ang = np.load(os.path.join(d, "angle_bin.npy")).astype(np.int64)
    ul = (np.load(os.path.join(d, "ul.npy")).astype(np.int64)
          if use_ul else np.zeros(len(tok), dtype=np.int64))
    return tok, bnd, mode, ang, ul


def fmt_matrix(M: np.ndarray, row_names, col_names) -> str:
    w = 7
    head = "cond_j-> ".ljust(9) + "".join(str(c)[:6].rjust(w) for c in col_names)
    lines = [head]
    for r, row in zip(row_names, M):
        cells = "".join((f"{v:7.3f}" if v is not None and np.isfinite(v)
                         else "    --").rjust(w) for v in row)
        lines.append(str(r)[:8].ljust(9) + cells)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    base = f"{home}/ros2_data/apt_g1/data/ds_bones/g1_b4lite"
    ap.add_argument("--run-dir", default=f"{base}/vae_v21/run1")
    ap.add_argument("--inputs-dir", default=f"{base}/vae_inputs_v21")
    ap.add_argument("--snapshot", default="vae_ep100.pt",
                    help="与 probe_vae_v2 v2.1 评测同快照，保持可比")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--min-win", type=int, default=150)
    ap.add_argument("--n-pc", type=int, default=8)
    ap.add_argument("--out-dir", default=None,
                    help="缺省 <run-dir>/../probe_cross_v3_1")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={dev}", flush=True)

    # ---- 模型加载（与 probe_vae_v2.py 同口径） ----
    meta = json.load(open(os.path.join(args.run_dir, "meta.json")))
    use_ul = bool(meta["use_ul"])
    model = DirSpeedPhaseTokenVAE(window=meta["window"], latent_dim=meta["latent_dim"],
                                  hidden_dim=meta["hidden"], n_vbins=3, n_dbins=8,
                                  n_modes=int(meta["n_modes"]), use_ul=use_ul).to(dev)
    sd_path = os.path.join(args.run_dir, args.snapshot)
    model.load_state_dict(torch.load(sd_path, map_location=dev))
    model.eval()
    build_meta = json.load(open(os.path.join(args.inputs_dir, "build_meta.json")))
    mode_names = {t["embed_idx"]: t["mode_name"] for t in build_meta["mode_table"]}
    walk_idx = meta["walk_embed_idx"]

    # ---- train 窗口 + 逐窗条件（复制 probe_vae_v2.py 的 train 侧口径） ----
    tok, bnd, mode, ang, ul = load_split(args.inputs_dir, use_ul)
    pca = np.load(os.path.join(args.run_dir, "pca.npz"))
    pmean, V2 = pca["pmean"], pca["V2"]
    base_mode = ((mode == walk_idx).astype(int) * 2 if walk_idx is not None
                 else np.asarray(mode))
    rate, phi = phi_rate_from(tok, pmean, V2)
    edges = np.quantile(rate[base_mode == 2], [1 / 3, 2 / 3])
    vb = np.clip(np.digitize(rate, edges), 0, 2).astype(np.int64)
    phase2 = np.stack([np.sin(phi), np.cos(phi)], 1).astype(np.float32)

    W = meta["window"]
    x, tok_y = build_windows_bounded(tok, bnd, W)  # tok_y=窗末真 token（decode 目标空间）
    mb = per_frame_labels(mode, bnd, W)
    ub = per_frame_labels(ul, bnd, W)
    pb = per_frame_labels(phase2, bnd, W)
    vbw = per_frame_labels(vb, bnd, W)
    dbw = per_frame_labels(ang, bnd, W)
    def window_seg_ids(bnd: np.ndarray, W: int) -> np.ndarray:
        """窗→段 id（build_windows_bounded 枚举序：每段 n-W+1 窗，按段序拼接）。"""
        ids = []
        for si, (a, b) in enumerate(bnd):
            n = int(b - a)
            if n >= W:
                ids.extend([si] * (n - W + 1))
        return np.asarray(ids, dtype=np.int64)

    seg_ids = window_seg_ids(bnd, W)
    km = os.path.join(args.inputs_dir, "window_keep_mask.npy")
    if os.path.isfile(km):
        m = np.load(km)
        x, tok_y, pb, vbw, dbw, mb, ub = (v[m] for v in
                                          (x, tok_y, pb, vbw, dbw, mb, ub))
        seg_ids = seg_ids[m]
    N = len(x)
    assert len(seg_ids) == N, "window_seg_ids 与窗枚举错位"
    print(f"[data] train windows N={N}", flush=True)

    # ---- [v3.1⑨] 过采样身份：每个过采样后段 -> 源原始段下标（留出按源段分组，
    # 过采样副本继承源段组别；v3 按过采样后段序号奇偶，副本跨两侧，已废弃） ----
    src_path = os.path.join(args.inputs_dir, "segment_source.npy")
    if os.path.isfile(src_path):
        src_idx = np.load(src_path).astype(np.int64)
        if len(src_idx) != len(bnd):
            raise RuntimeError("segment_source.npy 与 segment_bounds 段数不符")
        orig_stems = [d["stem"] for d in
                      sorted(build_meta["segment_detail"]["train"],
                             key=lambda d: d["stem"])]
        provenance = "segment_source.npy(构建器落盘)"
    else:
        src_idx, orig_stems = derive_segment_sources(bnd, build_meta)
        provenance = "oversample_plan_replay(三重checksum通过)"
    w_src = src_idx[seg_ids]
    splits_map = holdout_splits(w_src, orig_stems)
    half_primary = splits_map["stem_parity"]
    sens_splits = {k: splits_map[k] for k in ("stem_hash", "stem_halves")}
    print(f"[ident-src] 身份={provenance}；过采样后段 {len(src_idx)} = 原始段 "
          f"{len(orig_stems)} + 副本 {len(src_idx) - len(orig_stems)}；主分组="
          f"源段奇偶（同源窗不跨半），对照分组={list(sens_splits)}", flush=True)

    def batches(*arrays):
        for i in range(0, N, args.batch):
            yield [torch.from_numpy(np.ascontiguousarray(a[i:i + args.batch])).to(dev)
                   for a in arrays]

    # ---- 全窗 mu（z 源）+ 自条件前向 rec（v2 基线源：同 z 不改条件） ----
    mus, recs = [], []
    with torch.no_grad():
        for bx, bp, bv, bd, bm, bu in batches(x, pb, vbw, dbw, mb, ub):
            mu, _ = model.encode(bx)
            mus.append(mu.cpu().numpy())
            recs.append(model.decode(mu, bp, bv, bd, bm, bu).cpu().numpy())
    mus = np.concatenate(mus)
    recs = np.concatenate(recs)
    assert np.isfinite(mus).all() and np.isfinite(recs).all(), "NaN/Inf in forward"
    mu_t = torch.from_numpy(mus).float().to(dev)

    # ---- [修正④→⑥] 训练支持集：二元 (mode,UL) 仅存档打印，主判升四元
    # (mode,UL,vb,db)（owner 评审二轮 P1：解码同时消费 vb/db/phase，二元支持
    # 高估训练覆盖；phase 连续不入键，作为覆盖局限登记） ----
    n_modes_total = int(meta["n_modes"])
    sup_counts = np.zeros((n_modes_total, 3), dtype=np.int64)
    np.add.at(sup_counts, (mb, ub), 1)
    sup_ok = sup_counts > 0
    sup4_counts = np.zeros((n_modes_total, 3, 3, 8), dtype=np.int64)
    np.add.at(sup4_counts, (mb, ub, vbw, dbw), 1)
    sup4_ok = sup4_counts > 0
    win4_ok = sup4_ok[mb, ub, vbw, dbw]
    # IDLE 进入列覆盖核查（复现 owner 复算口径：mode≠0 且过二元检查的窗，
    # 其中四元有训练支持的比例——R1「trained-combo 维持」结论的覆盖前提）
    idle_entry_bin = (mb != 0) & sup_ok[0, ub]
    idle_entry_4 = idle_entry_bin & sup4_ok[0, ub, vbw, dbw]
    idle_entry_support = {
        "definition": "mode 交叉至 IDLE（进入列）off-diag 窗：过 (mode,UL) 二元"
                      "支持检查 vs 四元 (mode,UL,vb,db) 有训练支持",
        "n_binary_pass": int(idle_entry_bin.sum()),
        "n_4way_pass": int(idle_entry_4.sum()),
        "share_4way_over_binary": round(
            float(idle_entry_4.sum() / max(int(idle_entry_bin.sum()), 1)), 4),
    }
    sup_rows = [m for m in range(n_modes_total) if sup_counts[m].any()]
    sup_row_names = [mode_names.get(m, str(m)) for m in sup_rows]
    print("\n[support] (mode,UL) train-window 支持矩阵（行=mode 列=UL，0=缺席；"
          "仅存档，主判用四元）：", flush=True)
    print("mode\\UL    ".ljust(11) + "".join(UL_NAMES[u].rjust(9) for u in range(3)),
          flush=True)
    for m, rn_ in zip(sup_rows, sup_row_names):
        print(rn_[:10].ljust(11) + "".join(str(int(sup_counts[m, u])).rjust(9)
                                           for u in range(3)), flush=True)
    print(f"[support4] 四元组合支持：{int(sup4_ok.sum())} 组合 / 覆盖窗 "
          f"{int(win4_ok.sum())}/{N}；IDLE 进入列二元过检 "
          f"{int(idle_entry_bin.sum())} 窗中四元支持 {int(idle_entry_4.sum())}"
          f"（{idle_entry_support['share_4way_over_binary']:.1%}）", flush=True)

    def combo_trained_mask(axis: str, idx: np.ndarray, vj: int):
        """[修正⑥] 四元支持：交叉 vj 后 (mode,UL,vb,db) 是否 ∈ 训练窗组合集。"""
        if axis == "mode":
            return sup4_ok[vj, ub[idx], vbw[idx], dbw[idx]]
        if axis == "ul":
            return sup4_ok[mb[idx], vj, vbw[idx], dbw[idx]]
        if axis == "vb":
            return sup4_ok[mb[idx], ub[idx], vj, dbw[idx]]
        if axis == "db":
            return sup4_ok[mb[idx], ub[idx], vbw[idx], vj]
        return None

    # ---- 轴定义：mode / ul / vb / db（各轴独立矩阵） ----
    def axis_values(labels: np.ndarray) -> tuple[list[int], dict[int, str]]:
        cnt = {int(v): int((labels == v).sum()) for v in np.unique(labels)}
        keep = sorted(v for v, c in cnt.items() if c >= args.min_win)
        dropped = {v: c for v, c in cnt.items() if c < args.min_win}
        return keep, dropped

    axes = {
        "mode": (mb, {v: mode_names.get(v, str(v)) for v in range(n_modes_total)}),
        "ul": (ub, UL_NAMES),
        "vb": (vbw, {0: "vb0", 1: "vb1", 2: "vb2"}),
        "db": (dbw, {v: f"db{v}" for v in range(8)}),
    }
    results: dict[str, dict] = {}
    for axis, (labels, names) in axes.items():
        values, dropped = axis_values(labels)
        K = len(values)
        if K < 2:
            # 单值轴无 off-diag（readout 会 nan），登记后跳过
            results[axis] = {
                "values": values,
                "row_names": [names.get(v, str(v)) for v in values],
                "row_train_windows": [int((labels == v).sum()) for v in values],
                "below_min_win_dropped": {},
                "verdict": "k1_skipped(单值轴，无 off-diag 可判读)",
            }
            print(f"\n[axis={axis}] k={K}<2 -> skipped(单值轴)", flush=True)
            continue
        # [修正②] 行列名一律经 values[t] 映射；argmax 目标名查 col_names（位置->名）
        row_names = [names.get(v, str(v)) for v in values]
        col_names = list(row_names)
        print(f"\n[axis={axis}] values 核对（矩阵位置->标签值->名）: "
              f"{[(t, v, row_names[t]) for t, v in enumerate(values)]}", flush=True)
        counts = [int((labels == v).sum()) for v in values]
        # [修正③] 质心 = 真实 train 窗末 token（y，decode 目标空间）按真值分组均值
        cents = np.stack([tok_y[labels == v].mean(0) for v in values])
        cent_t = torch.from_numpy(cents).float().to(dev)

        # [修正⑦→v3.1⑨⑩] 真 token 可辨识性校准（同一最近质心读出，对真实
        # train 窗末 token）：resub=全体窗对全窗质心；holdout=源原始段分组留出
        # （过采样副本继承源段组别，同源窗必不跨半——v3 按过采样后段序号奇偶，
        # 副本跨两侧抬高识别，owner 评审三轮 P1），质心=留入半；覆盖门=任一
        # 目标类无双侧窗即不可估计（⑩，不得静默缩小任务后放行）；stem_hash /
        # stem_halves 固定对照分组只作敏感性波动报告。留出宏识别是该读出在
        # 真值上的保守上限：连真值都识别不好的轴，decode 侧低命中不可归因于
        # 条件通路（owner 评审二轮 P1-2）。源段分组仍是片段级而非演员独立终评。
        chance = 1.0 / K
        ident = nearest_centroid_holdout(tok_y, labels, values, half_primary,
                                         row_names)
        holdout_macro = ident["holdout_macro_recall"]
        resub_macro = ident["resub_macro_recall"]
        ident_ok = ident["readout_identifiable"]
        sensitivity = {}
        for nm, hm in sens_splits.items():
            r = nearest_centroid_holdout(tok_y, labels, values, hm, row_names)
            sensitivity[nm] = {"holdout_macro_recall": r["holdout_macro_recall"],
                               "coverage": r["coverage"]}
        real_ident = {
            "readout": "nearest-centroid（与主矩阵同读出）对真实 train 窗末 token",
            "holdout_split": "[v3.1] 源原始段分组（stem 升序奇偶，过采样副本继承"
                             "源段组别，同源窗不跨半）；质心=留入半；v3 按过采样"
                             "后段序号奇偶（副本跨侧）已废弃",
            "source_provenance": provenance,
            "sensitivity_fixed_splits": sensitivity,
            "resub_macro_recall": resub_macro,
            "resub_per_class": ident["resub_per_class"],
            "holdout_macro_recall": holdout_macro,
            "holdout_per_class": ident["holdout_per_class"],
            "holdout_confusion_rowtrue": ident["holdout_confusion_rowtrue"],
            "coverage": ident["coverage"],
            "classes_missing_holdout": ident["classes_missing_holdout"],
            "reduced_chance": ident["reduced_chance"],
            "n_source_segments_per_class": {
                row_names[t]: int(len(np.unique(w_src[labels == values[t]])))
                for t in range(K)},
            "readout_identifiable": ident_ok,
            "not_identifiable_reason": ident["reason"],
        }
        print(f"[ident] 真 token 校准(源段留出): resub_macro={resub_macro} "
              f"holdout_macro={holdout_macro}（2*chance={round(2 * chance, 3)}，"
              f"coverage={ident['coverage']}）-> "
              f"{'identifiable' if ident_ok else 'NOT identifiable: ' + str(ident['reason'])}",
              flush=True)

        # [修正③] 基线：同 z 不改条件（自条件前向 recs）对同一质心集的命中分布
        base = np.zeros((K, K), dtype=np.float64)   # base[i,j]=P(pred==j | z_i, cond_i)
        with torch.no_grad():
            for i, vi in enumerate(values):
                idx = np.where(labels == vi)[0]
                rr = torch.from_numpy(recs[idx]).float().to(dev)
                pred_b = torch.cdist(rr, cent_t).argmin(1).cpu().numpy()
                base[i] = np.bincount(pred_b, minlength=K) / len(idx)

        M = np.zeros((K, K), dtype=np.float64)      # M[i,j]=P(pred==j | z_i, cond_j)
        hit_i = np.zeros((K, K), dtype=np.float64)  # hit_i[i,j]=P(pred==i)（z 源类命中）
        top = np.zeros((K, K), dtype=np.int64)      # 每格最近质心 argmax（断裂归因用）
        top_p = np.zeros((K, K), dtype=np.float64)  # 每格最近质心概率
        stratified = True   # [修正⑥] 四轴全部分层（四元支持口径）
        sub_names = ("trained", "unseen") if stratified else ()
        # 分层存格：hit 向量 + 窗数；空格 = None
        s_hit = {s: [[None] * K for _ in range(K)] for s in sub_names}
        s_n = {s: np.zeros((K, K), dtype=np.int64) for s in sub_names}
        with torch.no_grad():
            for i, vi in enumerate(values):
                idx = np.where(labels == vi)[0]
                mu_i = mu_t[idx]
                ph_i = torch.from_numpy(pb[idx]).float().to(dev)
                cv = torch.from_numpy(vbw[idx]).to(dev)
                cd = torch.from_numpy(dbw[idx]).to(dev)
                cm = torch.from_numpy(mb[idx]).to(dev)
                cu = torch.from_numpy(ub[idx]).to(dev)
                for j, vj in enumerate(values):
                    if axis == "vb":
                        out = model.decode(mu_i, ph_i,
                                           torch.full_like(cv, vj), cd, cm, cu)
                    elif axis == "db":
                        out = model.decode(mu_i, ph_i, cv,
                                           torch.full_like(cd, vj), cm, cu)
                    elif axis == "mode":
                        out = model.decode(mu_i, ph_i, cv, cd,
                                           torch.full_like(cm, vj), cu)
                    elif axis == "ul":
                        out = model.decode(mu_i, ph_i, cv, cd, cm,
                                           torch.full_like(cu, vj))
                    else:
                        raise ValueError(axis)
                    d = torch.cdist(out, cent_t)
                    pred = d.argmin(1)
                    hv = (torch.bincount(pred, minlength=K).float()
                          / float(len(idx))).cpu().numpy()  # [修正①] 完整命中向量
                    M[i, j] = float(hv[j])
                    hit_i[i, j] = float(hv[i])
                    top[i, j] = int(hv.argmax())
                    top_p[i, j] = float(hv.max())
                    if stratified:
                        tm = combo_trained_mask(axis, idx, vj)
                        for s, sm in (("trained", tm), ("unseen", ~tm)):
                            if sm.any():
                                sm_t = torch.from_numpy(sm).to(dev)
                                bcs = torch.bincount(pred[sm_t], minlength=K).float()
                                s_hit[s][i][j] = (bcs / float(sm.sum())).cpu().numpy()
                                s_n[s][i, j] = int(sm.sum())
        off = ~np.eye(K, dtype=bool)
        cond_follow = float(M[off].mean())
        z_follow = float(hit_i[off].mean())      # [修正①] off-diag P(pred==i)
        base_follow = float(base[off].mean())    # [修正③] 基线 off-diag
        mean_delta = cond_follow - base_follow   # [修正③] 交叉 − 基线
        diag_ident = float(np.diag(M).mean())    # [修正③] 同条件可辨识率
        # [修正⑦⑧→v3.1⑩] 第一道门=真 token 源段留出识别（缺类判覆盖不足；
        # 读出不可辨识则该轴指标不可判读）；措辞只到「该指标下响应强弱」
        #（v1「架构问题证实」撤回）
        if ident["coverage"] != "full":
            verdict = (f"readout_unidentifiable(留出覆盖不足："
                       f"{','.join(ident['classes_missing_holdout'])} 类无双侧窗，"
                       f"识别任务不可估计；缩小任务宏识别={holdout_macro} 只作描述)")
        elif not ident_ok:
            verdict = (f"readout_unidentifiable(真token源段留出宏识别={holdout_macro} "
                       f"< 2*chance={round(2 * chance, 3)}，该轴指标不可判读；"
                       f"低交叉命中不可归因)")
        elif cond_follow >= 2 * chance:
            verdict = "cond_available_on_readout(该指标下条件响应>=2*chance)"
        elif z_follow >= 3 * cond_follow:
            verdict = "z_dominant_on_readout(该指标下 z 侧命中占优；不作机制实锤断言)"
        else:
            verdict = "partial_on_readout(该指标下部分可控)"
        # 断裂对归因：off-diag 中条件命中所指质心不是输出 argmax 的格（名经 values 映射）
        broken = [[row_names[i], col_names[j], round(float(M[i, j]), 4),
                   col_names[int(top[i, j])], round(float(top_p[i, j]), 4)]
                  for i in range(K) for j in range(K)
                  if i != j and int(top[i, j]) != j]

        def strat_pack(s: str) -> dict | None:
            if not stratified:
                return None
            mj = [[(float(s_hit[s][i][j][j]) if s_hit[s][i][j] is not None else None)
                   for j in range(K)] for i in range(K)]
            mi = [[(float(s_hit[s][i][j][i]) if s_hit[s][i][j] is not None else None)
                   for j in range(K)] for i in range(K)]
            tg = [[(col_names[int(s_hit[s][i][j].argmax())]
                    if s_hit[s][i][j] is not None else None)
                   for j in range(K)] for i in range(K)]
            tp = [[(float(s_hit[s][i][j].max()) if s_hit[s][i][j] is not None else None)
                   for j in range(K)] for i in range(K)]
            offv_j = [mj[i][j] for i in range(K) for j in range(K)
                      if i != j and mj[i][j] is not None]
            offv_i = [mi[i][j] for i in range(K) for j in range(K)
                      if i != j and mi[i][j] is not None]
            return {
                "cell_n": s_n[s].tolist(),
                "n_windows_total": int(s_n[s].sum()),
                "matrix_hit_j": mj,
                "matrix_hit_i": mi,
                "argmax_target": tg,
                "argmax_prob": tp,
                "cond_follow_offdiag_nonempty": (round(float(np.mean(offv_j)), 4)
                                                 if offv_j else None),
                "z_follow_offdiag_nonempty": (round(float(np.mean(offv_i)), 4)
                                              if offv_i else None),
                "n_nonempty_offdiag_cells": len(offv_j),
            }

        combo_strat = None
        if stratified:
            unseen_share = float((s_n["unseen"][off].sum()
                                  / max(int(s_n["trained"][off].sum()
                                            + s_n["unseen"][off].sum()), 1)))
            combo_strat = {
                "combo_definition": (
                    "trained: 交叉后 (mode,UL,vb,db) 四元 ∈ train 窗组合支持集"
                    "（phase 连续不入键）；unseen: ∉S；空格=None。IDLE 归因只允许"
                    "引用 trained 格，trained 覆盖过薄（n_windows_total 小）时判"
                    "不可识别" if axis == "mode"
                    else "trained: 交叉后 (mode,UL,vb,db) 四元 ∈ train 窗组合支持"
                         "集（phase 连续不入键）；unseen: ∉S；空格=None"),
                "trained": strat_pack("trained"),
                "unseen": strat_pack("unseen"),
                "offdiag_window_share_unseen": round(unseen_share, 4),
            }

        results[axis] = {
            "values": values, "row_names": row_names, "row_train_windows": counts,
            "below_min_win_dropped": {names.get(v, str(v)): c for v, c in dropped.items()},
            "matrix_hit_j": M.tolist(),
            "matrix_hit_i_zsource": hit_i.tolist(),
            "matrix_argmax_target": [[col_names[int(t)] for t in row] for row in top],
            "matrix_argmax_prob": top_p.tolist(),
            "baseline_hit_j_samecond": base.tolist(),
            "matrix_delta_hit_j_minus_baseline": (M - base).tolist(),
            "cond_follow": round(cond_follow, 4),
            "z_follow": round(z_follow, 4),
            "baseline_follow_offdiag": round(base_follow, 4),
            "mean_delta_offdiag_cross_minus_baseline": round(mean_delta, 4),
            "diag_identification": round(diag_ident, 4),
            "real_token_identification": real_ident,
            "chance": round(chance, 4),
            "z_over_cond_ratio": round(z_follow / max(cond_follow, 1e-9), 2),
            "cond_over_chance": round(cond_follow / chance, 2),
            "cond_over_baseline": round(cond_follow / max(base_follow, 1e-9), 2),
            "verdict": verdict,
            "offdiag_broken_cells_top1_not_j": broken,
            "combo_stratified": combo_strat,
        }
        print(f"\n[axis={axis}] k={K} chance={chance:.3f} "
              f"cond_follow={cond_follow:.4f} z_follow={z_follow:.4f} "
              f"baseline={base_follow:.4f} mean_delta={mean_delta:+.4f} "
              f"diag_ident={diag_ident:.4f} -> {verdict}", flush=True)
        print("M_hit_j (P(pred==j | z_i, cond_j)):", flush=True)
        print(fmt_matrix(M, row_names, col_names), flush=True)
        print("baseline P(pred==j | z_i, cond_i):", flush=True)
        print(fmt_matrix(base, row_names, col_names), flush=True)
        if stratified:
            for s in sub_names:
                print(f"stratified[{s}] M_hit_j:", flush=True)
                print(fmt_matrix(np.array([[np.nan if v is None else v for v in r]
                                           for r in results[axis]["combo_stratified"][s]["matrix_hit_j"]],
                                          dtype=np.float64), row_names, col_names),
                      flush=True)

    # ---- J9 复测：IDLE 行/列（mode 矩阵；含 trained/unseen 分层） ----
    j9 = None
    mres = results["mode"]
    if 0 in mres["values"]:
        r = mres["values"].index(0)
        Mm = np.asarray(mres["matrix_hit_j"])
        rn = mres["row_names"]
        K = len(rn)
        col_off = [float(Mm[i, r]) for i in range(K) if i != r]
        row_off = [float(Mm[r, j]) for j in range(K) if j != r]
        j9 = {
            "idle_embed_idx": 0,
            "idle_row_P_hit_j": {rn[j]: round(float(Mm[r, j]), 4)
                                 for j in range(K)},          # 命令离开 hub
            "idle_row_offdiag_mean": round(float(np.mean(row_off)), 4),
            "idle_col_P_hit_IDLE": {rn[i]: round(float(Mm[i, r]), 4)
                                    for i in range(K)},        # 命令进入 hub
            "idle_col_offdiag_mean": round(float(np.mean(col_off)), 4),
            "reference_probe_v21_devz_idle_hit": 0.004,
            "chance": mres["chance"],
            "note": "行=z_IDLE 条件 j（能否被命令离开）；列=z_i 条件 IDLE"
                    "（能否被命令进入；与 v2.1 dev-z 版 0.004 对照）；"
                    "分层解读只引用 trained 格（v3 起四元支持口径，覆盖过薄判"
                    "不可识别）",
        }
        st = mres.get("combo_stratified")
        if st:
            mj_t = st["trained"]["matrix_hit_j"]
            mj_u = st["unseen"]["matrix_hit_j"]
            n_t = st["trained"]["cell_n"]
            for key, mat in (("trained", mj_t), ("unseen", mj_u)):
                rowv = [mat[r][j] for j in range(K) if j != r and mat[r][j] is not None]
                colv = [mat[i][r] for i in range(K) if i != r and mat[i][r] is not None]
                j9[f"idle_row_offdiag_{key}"] = (round(float(np.mean(rowv)), 4)
                                                 if rowv else None)
                j9[f"idle_col_offdiag_{key}"] = (round(float(np.mean(colv)), 4)
                                                 if colv else None)
                j9[f"idle_row_nonempty_offdiag_cells_{key}"] = len(rowv)
                j9[f"idle_col_nonempty_offdiag_cells_{key}"] = len(colv)
        print(f"\n[J9] IDLE 行(离开) off-diag mean={j9['idle_row_offdiag_mean']}; "
              f"IDLE 列(进入) off-diag mean={j9['idle_col_offdiag_mean']}", flush=True)
        if st:
            print(f"[J9 stratified] 行 trained={j9.get('idle_row_offdiag_trained')}"
                  f"(n_cells={j9.get('idle_row_nonempty_offdiag_cells_trained')}) "
                  f"unseen={j9.get('idle_row_offdiag_unseen')}"
                  f"(n_cells={j9.get('idle_row_nonempty_offdiag_cells_unseen')}); "
                  f"列 trained={j9.get('idle_col_offdiag_trained')}"
                  f"(n_cells={j9.get('idle_col_nonempty_offdiag_cells_trained')}) "
                  f"unseen={j9.get('idle_col_offdiag_unseen')}"
                  f"(n_cells={j9.get('idle_col_nonempty_offdiag_cells_unseen')})",
                  flush=True)

    # ---- 诊断：IDLE z vs locomotion z 分布 ----
    idle_mask = mb == 0
    z_diag = None
    if idle_mask.any() and (~idle_mask).any():
        mu_loco = mus[~idle_mask]
        mu_idle = mus[idle_mask]
        mu_c = mu_loco - mu_loco.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(mu_c, full_matrices=False)
        Vp = Vt[:args.n_pc].T  # (16, n_pc)
        pl, pi_ = mu_c @ Vp, (mu_idle - mu_loco.mean(0, keepdims=True)) @ Vp
        sd_l = pl.std(0)
        sd_i = pi_.std(0)
        shift = np.abs(pi_.mean(0) - pl.mean(0)) / np.maximum(sd_l, 1e-9)
        z_diag = {
            "n_idle": int(idle_mask.sum()), "n_loco": int((~idle_mask).sum()),
            "pca_basis": "locomotion 子集 mu（train 全 locomotion 窗）",
            "n_pc": args.n_pc,
            "per_pc_std_mean_shift": [round(float(s), 3) for s in shift],
            "per_pc_std_loco": [round(float(s), 4) for s in sd_l],
            "per_pc_std_idle": [round(float(s), 4) for s in sd_i],
            "std_ratio_idle_over_loco_mean": round(float(sd_i.mean() / sd_l.mean()), 4),
            "mean_abs_shift_top8": round(float(shift.mean()), 3),
            "interpretation_hint": "仅必要条件口径：std_ratio 不塌缩且 shift 有限 ->"
                                   "「未观察到所测 top-8 投影上的方差塌缩」（不构成"
                                   " z 含任务相关信息的证明）；若塌缩 -> IDLE z "
                                   "无信息的平凡解释成立",
        }
        print(f"\n[zdiag] mean|shift| top{args.n_pc}={z_diag['mean_abs_shift_top8']}; "
              f"std_ratio(idle/loco)={z_diag['std_ratio_idle_over_loco_mean']}",
              flush=True)

    out_dir = args.out_dir or os.path.join(args.run_dir, "..", "probe_cross_v3_1")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    overall = {}
    for a in results:
        ra = results[a]
        if "cond_follow" not in ra:      # k1_skipped 单值轴
            overall[a] = {"verdict": ra["verdict"]}
            continue
        overall[a] = {"cond_follow": ra["cond_follow"],
                      "z_follow": ra["z_follow"],
                      "baseline_follow_offdiag": ra["baseline_follow_offdiag"],
                      "mean_delta_offdiag": ra["mean_delta_offdiag_cross_minus_baseline"],
                      "diag_identification": ra["diag_identification"],
                      "chance": ra["chance"],
                      "verdict": ra["verdict"]}
    out = {
        "_meta": {"script": "apt_g1/probe_vae_cross.py", "experiment": "D046b",
                  "version": "v3.1 (owner 评审三轮修正版 2026-09-07)",
                  "snapshot": sd_path, "device": str(dev),
                  "train_windows": int(N), "min_win": args.min_win,
                  "design": "单轴交叉：z 取条件值 i 的训练窗 mu，被测轴置 j，"
                            "其余轴保持窗口真值；质心=train 窗末真 token（y，与"
                            "decode 输出同空间）按真值分组"
                            "均值（v2 修正③）；基线=同 z 自条件前向命中分布",
                  "corrections": {
                      "z_follow": "off-diag P(pred==i)（v1 误用对角均值，作废 z/c=1.53）",
                      "name_mapping": "行/列/argmax 目标名一律经 values[t] 映射"
                                      "（v1 位置当标签，mode 轴全体错位）",
                      "centroid": "train 窗末真 token（y=decode 目标空间；v1 为模型"
                                  "自重建均值，未校准）",
                      "baseline": "P(命中 j | z_i, cond_i) vs 交叉逐轴 mean delta",
                      "combo_stratify": "[v3 升级⑥] 四元 (mode,UL,vb,db) 组合支持集"
                                        "（R1 二元 (mode,UL) 低估 OOD：IDLE 进入列"
                                        " 5,263 窗仅 465 窗四元有支持）；四轴全分层；"
                                        "IDLE 归因只引用 trained 格且覆盖过薄判不可识别",
                      "readout_calibration": "[v3 增补⑦→v3.1⑨] 真 token 留出识别门："
                                             "留出宏识别 <2*chance -> "
                                             "readout_unidentifiable；resub 与"
                                             "留出双口径落盘；v3.1 留出改按源原始"
                                             "段分组（v3 按过采样后段序号奇偶，副本"
                                             "跨两侧——v21 材料 99 原始段展开 215 段、"
                                             "39 段跨两侧，同源窗抬高识别），并附 "
                                             "stem_hash/stem_halves 敏感性",
                      "holdout_coverage_gate": "[v3.1⑩] 缺类判覆盖不足（缺类单列，"
                                               "verdict=readout_unidentifiable），"
                                               "不得静默缩小分类任务后对原任务"
                                               "随机线放行（v3 反例：8 类仅 2 类"
                                               "可留出且 token 全同以 0.5 误放行）",
                      "verdict_wording": "[v3 修正⑧] 撤回「架构问题证实」机制断言，"
                                         "verdict 只到该指标响应强弱；z_diag 收窄为"
                                         "「未观察到所测投影方差塌缩」",
                      "db_bin0": "bin0=段首方向±22.5°=前向（build_b4lite_vae_inputs.py）；"
                                 "v1 台账「db5=前向邻域」表述有误",
                  },
                  "criteria": "预注册（v3.1 修订）：任一目标类无双侧留出窗->"
                              "readout_unidentifiable(覆盖不足)；elif 真token源段"
                              "留出宏识别<2*chance->readout_unidentifiable（不可"
                              "判读）；elif cond_follow>=2*chance->"
                              "cond_available_on_readout；elif z_follow>=3*cond_"
                              "follow->z_dominant_on_readout；else "
                              "partial_on_readout；措辞只到「该指标下响应强弱」",
                  "mode_axis_note": "SLOW_WALK(1)/INJURED_WALK(6) 为 T-fam 零训练窗，"
                                    "mode 轴按 >=150 窗实况取值（8 度）"},
        "mode_ul_support": {
            "note": "train 窗级 (mode,UL) 组合计数；0=训练未见组合（数据侧修复候选"
                    "留给 owner：如 IDLE+none 语料有但池未选）",
            "col_names": [UL_NAMES[u] for u in range(3)],
            "row_names": sup_row_names,
            "row_embed_idx": sup_rows,
            "counts": [[int(sup_counts[m, u]) for u in range(3)] for m in sup_rows],
            "trained_pairs": {f"{mode_names.get(m, str(m))}|{UL_NAMES[u]}":
                              int(sup_counts[m, u])
                              for m in sup_rows for u in range(3)
                              if sup_counts[m, u] > 0},
            "absent_pairs": [f"{mode_names.get(m, str(m))}|{UL_NAMES[u]}"
                             for m in sup_rows for u in range(3)
                             if sup_counts[m, u] == 0],
        },
        "support_4way": {
            "note": "[v3⑥] (mode,UL,vb,db) 训练窗组合支持；主判分层与 IDLE 归因"
                    "一律用此口径（phase 连续不入键，作为覆盖局限登记）",
            "n_combos_supported": int(sup4_ok.sum()),
            "n_windows_in_supported_combos": int(win4_ok.sum()),
            "idle_entry_support_check": idle_entry_support,
        },
        "axes": results,
        "j9_hub": j9,
        "z_diag_idle_vs_loco": z_diag,
        "verdict_overall": overall,
    }
    jp = os.path.join(out_dir, "metrics_d046b_v3_1.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    sp = os.path.join(out_dir, "summary_d046b_v3_1.txt")
    with open(sp, "w", encoding="utf-8") as f:
        f.write(f"D046b 交叉 z 探针 v3.1 summary（owner 评审三轮修正版）\nsnapshot={sd_path}\nN={N}\n")
        f.write("\n(mode,UL) train-window 支持矩阵（0=缺席）：\n")
        f.write("mode\\UL    ".ljust(11) + "".join(UL_NAMES[u].rjust(9) for u in range(3)) + "\n")
        for m, rn_ in zip(sup_rows, sup_row_names):
            f.write(rn_[:10].ljust(11) + "".join(str(int(sup_counts[m, u])).rjust(9)
                                                 for u in range(3)) + "\n")
        f.write(f"support4: 四元组合 {int(sup4_ok.sum())} / 覆盖窗 "
                f"{int(win4_ok.sum())}/{N}；IDLE 进入列二元过检 "
                f"{idle_entry_support['n_binary_pass']} 窗中四元支持 "
                f"{idle_entry_support['n_4way_pass']}"
                f"（{idle_entry_support['share_4way_over_binary']:.1%}）\n")
        for a, r in results.items():
            if "cond_follow" not in r:
                f.write(f"\n[axis={a}] skipped: {r['verdict']}\n")
                continue
            f.write(f"\n[axis={a}] k={len(r['values'])} chance={r['chance']} "
                    f"cond_follow={r['cond_follow']} z_follow={r['z_follow']} "
                    f"baseline={r['baseline_follow_offdiag']} "
                    f"mean_delta={r['mean_delta_offdiag_cross_minus_baseline']} "
                    f"diag_ident={r['diag_identification']} "
                    f"z/c={r['z_over_cond_ratio']} cond/chance={r['cond_over_chance']}"
                    f"\nverdict: {r['verdict']}\n")
            ri = r["real_token_identification"]
            _reason = ri["not_identifiable_reason"]
            f.write(f"真token校准(源段留出): resub={ri['resub_macro_recall']} "
                    f"holdout={ri['holdout_macro_recall']}"
                    f"（2*chance={round(2 * r['chance'], 3)}，"
                    f"coverage={ri['coverage']}）identifiable="
                    f"{ri['readout_identifiable']}"
                    + (f"（{_reason}）" if _reason else "") + "\n")
            f.write(f"  敏感性固定分组: {ri['sensitivity_fixed_splits']}"
                    f"（身份={ri['source_provenance']}）\n")
            f.write(f"  holdout_per_class: {ri['holdout_per_class']}\n")
            f.write("values 核对: " + str(list(zip(r["values"], r["row_names"]))) + "\n")
            f.write("M_hit_j (P(pred==j | z_i, cond_j)):\n")
            f.write(fmt_matrix(np.asarray(r["matrix_hit_j"]),
                               r["row_names"], r["row_names"]) + "\n")
            f.write("baseline_hit_j (P(pred==j | z_i, cond_i)):\n")
            f.write(fmt_matrix(np.asarray(r["baseline_hit_j_samecond"]),
                               r["row_names"], r["row_names"]) + "\n")
            f.write("delta (cross - baseline):\n")
            f.write(fmt_matrix(np.asarray(r["matrix_delta_hit_j_minus_baseline"]),
                               r["row_names"], r["row_names"]) + "\n")
            f.write("matrix_hit_i (P(pred==i | z_i, cond_j), z_follow 源):\n")
            f.write(fmt_matrix(np.asarray(r["matrix_hit_i_zsource"]),
                               r["row_names"], r["row_names"]) + "\n")
            f.write("argmax target (per cell):\n")
            for rn_, trow, prow in zip(r["row_names"],
                                       r["matrix_argmax_target"],
                                       r["matrix_argmax_prob"]):
                cells = ", ".join(f"j={cn_}->{t_}({p_:.2f})"
                                  for cn_, t_, p_ in zip(r["row_names"], trow, prow))
                f.write(f"  z={rn_:<14s} {cells}\n")
            if r.get("combo_stratified"):
                for s in ("trained", "unseen"):
                    st = r["combo_stratified"][s]
                    f.write(f"stratified[{s}] M_hit_j "
                            f"(cond_follow_nonempty={st['cond_follow_offdiag_nonempty']}, "
                            f"z_follow_nonempty={st['z_follow_offdiag_nonempty']}, "
                            f"n_nonempty_offdiag={st['n_nonempty_offdiag_cells']}):\n")
                    f.write(fmt_matrix(np.array(
                        [[np.nan if v is None else v for v in row] for row in st["matrix_hit_j"]],
                        dtype=np.float64), r["row_names"], r["row_names"]) + "\n")
                    f.write(f"stratified[{s}] argmax target:\n")
                    for rn_, trow in zip(r["row_names"], st["argmax_target"]):
                        cells = ", ".join(f"j={cn_}->{t_}" for cn_, t_
                                          in zip(r["row_names"], trow) if t_ is not None)
                        f.write(f"  z={rn_:<14s} {cells}\n")
        if j9:
            f.write(f"\nJ9 hub: 离开={j9['idle_row_offdiag_mean']} "
                    f"进入={j9['idle_col_offdiag_mean']} "
                    f"(v2.1 dev-z 参考 0.004, chance={j9['chance']})\n")
            for s in ("trained", "unseen"):
                f.write(f"J9 stratified[{s}]: 行离开={j9.get(f'idle_row_offdiag_{s}')}"
                        f"(cells={j9.get(f'idle_row_nonempty_offdiag_cells_{s}')}) "
                        f"列进入={j9.get(f'idle_col_offdiag_{s}')}"
                        f"(cells={j9.get(f'idle_col_nonempty_offdiag_cells_{s}')})\n")
        if z_diag:
            f.write(f"zdiag: mean|shift|={z_diag['mean_abs_shift_top8']} "
                    f"std_ratio={z_diag['std_ratio_idle_over_loco_mean']}\n")
    print(f"\n[write] {jp}\n[write] {sp}", flush=True)
    print(json.dumps(overall, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
