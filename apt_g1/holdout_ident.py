"""D046b-R3 留出识别与过采样身份工具（numpy-only，probe_vae_cross.py 与回归测试共用）。

背景（owner 评审三轮 2026-09-07 P1/P2）：v3 探针的段级留出按过采样后段序号奇偶
分半，而 build_b4lite_vae_inputs_v2.py 的过采样按 mode 预算整段/截段复制原始
片段——副本段与源段落入不同半区（v21 材料服务器实证：99 原始段展开 215 段、
39 个原始段跨两侧，复制 token 逐位一致），留出识别率被同源窗相关性抬高。

三件套：
  1) derive_segment_sources：从 build_meta.json 的 oversample_plan 确定性重放
     过采样计划，给每个过采样后段标注源原始段下标（现役 v21 材料无
     segment_source.npy 时的兼容路径；copies 计数 + 逐段帧数 + 总帧数三重
     checksum，不符即 raise，不得静默错标）。
  2) holdout_splits：窗级半区掩码。主分组 stem_parity = 源原始段下标奇偶
     （build 顺序即 stem 升序，副本继承源段组别——同源窗必不跨半）；另附两个
     固定对照分组 stem_hash（stem 名 md5 奇偶）/ stem_halves（stem 升序对半）
     作敏感性报告；预注册门只认主分组。
  3) nearest_centroid_holdout：真 token 最近质心留出识别（resub + holdout），
     含覆盖门——任一目标类在任一侧无窗（覆盖不足）时 readout_identifiable
     强制 False 并列出缺类，不得静默缩小分类任务后用缩小任务的宏识别高过
     原任务 2*chance 线放行（owner 反例：8 类仅 2 类可留出、token 全同 →
     宏识别 0.5 > 2/8 被 v3 误放行）。
"""
from __future__ import annotations

import hashlib

import numpy as np

INTERMEDIATE_NAME = "intermediate"


def derive_segment_sources(bounds: np.ndarray, build_meta: dict) -> tuple[np.ndarray, list[str]]:
    """重放 build_b4lite_vae_inputs_v2.oversample_by_mode，返回 (源段下标, 原始 stem 表)。

    bounds = 过采样后 segment_bounds（save_arrays 落盘版）；build_meta 需含
    mode_table / oversample_plan / segment_detail.train（stem/target/n_frames，
    顺序不限，函数内部按 stem 排序恢复 build 顺序）。源段下标 = 原始段在
    build 顺序（stem 升序）中的下标；原始段映射到自身，副本映射到其源段。
    任何 checksum 不符（材料与当前构建器代码不同源）→ RuntimeError。
    """
    detail = sorted(build_meta["segment_detail"]["train"], key=lambda d: d["stem"])
    idx_by_name = {t["mode_name"]: int(t["embed_idx"]) for t in build_meta["mode_table"]}
    n_orig = len(detail)
    stems = [d["stem"] for d in detail]
    nframes = [int(d["n_frames"]) for d in detail]
    modes = [idx_by_name[d["target"]] for d in detail]
    inter_idx = idx_by_name[INTERMEDIATE_NAME]
    bounds = np.asarray(bounds, dtype=np.int64)

    if sorted(stems) != stems or len(set(stems)) != n_orig:
        raise RuntimeError("stem 表非严格升序唯一，与 build_split 排序口径不符")
    src: list[int] = []
    off = 0
    for i in range(n_orig):
        if int(bounds[i, 0]) != off or int(bounds[i, 1]) != off + nframes[i]:
            raise RuntimeError(f"原始段 {i}（{stems[i]}）bounds 与 n_frames 不符")
        src.append(i)
        off += nframes[i]

    plan = build_meta["oversample_plan"]
    orig_frames = {int(m): int(p["orig_frames"]) for m, p in plan.items()}
    budgets = {int(p["budget"]) for p in plan.values()}
    if len(budgets) != 1 or budgets.pop() != max(orig_frames.values()):
        raise RuntimeError("oversample_plan 预算口径与构建器（max 族帧预算）不符")
    for m, p in plan.items():
        got = sum(nframes[i] for i in range(n_orig)
                  if modes[i] != inter_idx and modes[i] == int(m))
        if got != int(p["orig_frames"]):
            raise RuntimeError(f"mode {m} orig_frames {p['orig_frames']} != 实数 {got}")

    for m in sorted(plan, key=int):
        need = int(plan[m]["budget"]) - int(plan[m]["orig_frames"])
        idxs = [i for i in range(n_orig) if modes[i] != inter_idx and modes[i] == int(m)]
        copies: dict[int, int] = {}
        added = 0
        while added < need:
            for i in idxs:                      # 与构建器同款 stem 升序轮转截段复制
                if added >= need:
                    break
                take = min(nframes[i], need - added)
                j = len(src)
                if j >= len(bounds):
                    raise RuntimeError("重放段数超出 bounds——材料非当前构建器产物")
                if (int(bounds[j, 0]), int(bounds[j, 1])) != (off, off + take):
                    raise RuntimeError(f"副本段 {j} bounds 与重放不符（期望 {take} 帧）")
                src.append(i)
                off += take
                added += take
                copies[i] = copies.get(i, 0) + 1
        plan_copies = {stems[i]: c for i, c in copies.items()}
        if plan_copies != {k: int(v) for k, v in plan[m]["copies"].items()}:
            raise RuntimeError(f"mode {m} copies 计数与 oversample_plan 不符："
                               f"重放 {plan_copies} vs plan {plan[m]['copies']}")
    if len(src) != len(bounds) or off != int(bounds[-1, 1]):
        raise RuntimeError(f"重放总段数/总帧数与 bounds 不符（{len(src)} vs {len(bounds)}，"
                           f"{off} vs {bounds[-1, 1]}）")
    return np.asarray(src, dtype=np.int64), stems


def holdout_splits(src_idx: np.ndarray, orig_stems: list[str]) -> dict[str, np.ndarray]:
    """窗级（或段级）半区掩码表，True = A 半（质心侧）。

    主分组 stem_parity：源段下标奇偶（build 顺序 = stem 升序，故等价于 stem
    字典序奇偶；副本继承源段组别）。对照分组仅作敏感性报告：stem_hash（stem
    名 md5 奇偶，与 build 顺序独立）、stem_halves（stem 升序前/后半）。
    """
    src = np.asarray(src_idx, dtype=np.int64)
    if src.min() < 0 or src.max() >= len(orig_stems):
        raise ValueError("src_idx 超出原始段表范围")
    rank = {s: r for r, s in enumerate(sorted(orig_stems))}
    hash_a = {s: int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % 2 == 0
              for s in orig_stems}
    half_rank = (len(orig_stems) + 1) // 2
    return {
        "stem_parity": (src % 2 == 0),
        "stem_hash": np.asarray([hash_a[orig_stems[i]] for i in src], dtype=bool),
        "stem_halves": np.asarray([rank[orig_stems[i]] < half_rank for i in src],
                                  dtype=bool),
    }


def nearest_centroid_holdout(points: np.ndarray, labels: np.ndarray,
                             values: list[int], half: np.ndarray,
                             names: list[str] | None = None) -> dict:
    """最近质心识别校准（resub + 段组留出），带覆盖门。

    points=(N,D) 真值特征（如 train 窗末 token），labels=(N,) 真值类，values=K 个
    目标类（完整任务，chance=1/K 按它算），half=(N,) True=留入半（质心侧）。
    覆盖门：某类在任一侧无窗 → coverage="partial"、readout_identifiable=False
    （留出任务不可估计，缺类单列；不以缩小任务后的宏识别对原任务随机线放行）。
    返回键与 probe_vae_cross v3 real_ident 兼容（readout_identifiable 等沿用）。
    """
    K = len(values)
    names = names or [str(v) for v in values]
    half = np.asarray(half, dtype=bool)

    def _nearest(c: np.ndarray, p: np.ndarray) -> np.ndarray:
        # argmin ||p-c||^2 = argmax(p·c - 0.5||c||^2)
        return np.argmax(p @ c.T - 0.5 * (c ** 2).sum(1)[None, :], axis=1)

    cents = np.stack([points[labels == v].mean(0) for v in values])
    resub_per_class = {}
    for t, v in enumerate(values):
        mv = labels == v
        resub_per_class[names[t]] = round(float((_nearest(cents, points[mv]) == t).mean()), 4)
    resub_macro = round(float(np.mean(list(resub_per_class.values()))), 4)

    feas_t = [t for t in range(K)
              if ((labels == values[t]) & half).any()
              and ((labels == values[t]) & ~half).any()]
    missing = [names[t] for t in range(K) if t not in feas_t]
    holdout_per_class = {names[t]: None for t in range(K)}
    conf = np.full((K, K), np.nan)
    cents_a = (np.stack([points[(labels == values[t2]) & half].mean(0)
                         for t2 in feas_t]) if feas_t else None)
    for pos, t in enumerate(feas_t):
        mh = (labels == values[t]) & ~half
        ph = _nearest(cents_a, points[mh])
        holdout_per_class[names[t]] = round(float((ph == pos).mean()), 4)
        for p2, t2 in enumerate(feas_t):
            conf[t, t2] = (ph == p2).mean()
    feas_vals = [holdout_per_class[names[t]] for t in feas_t]
    holdout_macro = round(float(np.mean(feas_vals)), 4) if feas_vals else None

    coverage = "full" if not missing else "partial"
    chance = 1.0 / K
    if coverage != "full":
        identifiable, reason = False, (
            f"holdout_coverage_insufficient: {','.join(missing)} 类无双侧窗，"
            f"留出任务不可估计（feasible={len(feas_t)}/{K}，缩小任务宏识别"
            f"{holdout_macro} 不得对完整任务 2*chance={2 * chance:.3f} 放行）")
    else:
        identifiable = holdout_macro is not None and holdout_macro >= 2 * chance
        reason = (None if identifiable else
                  f"holdout_macro_recall {holdout_macro} < 2*chance {2 * chance:.3f}")
    return {
        "chance": round(chance, 4),
        "reduced_chance": (round(1.0 / len(feas_t), 4) if feas_t else None),
        "coverage": coverage,
        "classes_missing_holdout": missing,
        "resub_macro_recall": resub_macro,
        "resub_per_class": resub_per_class,
        "holdout_macro_recall": holdout_macro,
        "holdout_per_class": holdout_per_class,
        "holdout_confusion_rowtrue": [[None if not np.isfinite(x) else round(float(x), 4)
                                       for x in row] for row in conf],
        "readout_identifiable": identifiable,
        "reason": reason,
    }
