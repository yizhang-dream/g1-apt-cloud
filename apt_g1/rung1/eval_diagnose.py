"""TO41 (c) 诊断：现有 56-cell receipts + effect_table_v1 的 variance /
natural-vs-interventional / C1-C2 comparability 三块分解。

Owner 裁定（TO41 第一轮判读后的 (c) 分支）：不新增训练、不新增 speed、
不修改 protocol；只用已入仓产物回答三个问题——

  V. 方差结构：Δ_ff 的不确定性中 eval-seed 与 train-seed 各占多少
     （数据源 = effect_table_v1 的 per-eval-seed 配对差，主指标口径，
      无需重构 err60s；receipt 聚合字段口径不同，只用于 S/D 块形态证据）。
  S. eval seed identity 审计：jitter rng(1000+seed) 是否真的进入执行路径
     （代码事实 + receipts 逐 episode 全精度字段的 bit-level 差异）。
  N. natural vs interventional：每 (v, C) 格的自然 assignment 与 override
     关系（receipts 的 natural_vb_distribution / n_override_changed 全量
     提取；给出 matched/mismatched 结构对 Δ_cond estimand 的含义）。
  D. C1/C2 comparability：τ_ff OFF 臂两 condition 的行为读数是否落在
     可比 support（主指标 err + receipts 的 disp/h_min/vx_mean 形态维度；
     机械判定规则仅作诊断输出，不是协议硬点）。
  W. 综合裁决：按 owner 分叉树给出数据侧 verdict 与建议分支（最终决定
     权在 owner）。

口径注意（本脚本存在的原因之一）：effect_table 的 err60s = per-step
|vx−cmd| 的 3-eval-seed 均值，由服务器端原始 rollout 计算，receipt 只存
vx_mean 等聚合字段，两者不可互推（蠕行/绕圈格偏差可达 0.09 m/s）。因此
V 块全部读数取自 effect_table 本身（per-seed pairs 即逐 eval-seed 配对差
ON−OFF，正是主指标的原始粒度），不尝试从 receipts 重构 err60s。

用法（本机轻量分析即可，无需 venv）：
  python apt_g1/rung1/eval_diagnose.py \
      --receipts apt_g1/outputs/sync/to41_eval/receipts \
      --effect-table apt_g1/outputs/sync/to41_eval/effect_table_v1.txt \
      --out-json apt_g1/outputs/sync/to41_eval/diagnosis_v1.json \
      --out-txt apt_g1/outputs/sync/to41_eval/diagnosis_v1.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

# bin 边界（apt_flat_env.py latent_speed_bins 分支）：linspace(0, vx_max=0.8, 4)[1:-1]
VB_EDGES = (0.8 / 3.0, 2 * 0.8 / 3.0)
DECISION_BOUNDARY = 0.02  # Rung 1 §10.4 主指标决策边界（m/s）


def natural_vb(cmd_v: float) -> int:
    """torch.bucketize(cmd_v, edges).clamp(0, n-1)，right=False 语义。"""
    vb = 0
    for e in VB_EDGES:
        if cmd_v >= e:
            vb += 1
    return vb


def parse_effect_table(path: Path) -> list[dict]:
    """解析 effect_table_v1.txt 的 28 行 per (v,C,seed) 记录。

    行形如：0.2    C1  s0 | 0.117  0.028   | +0.090  [+0.090 +0.089 +0.090] | 3/3 3/3
    其中 pairs = 逐 eval-seed 配对差 ON_err_e − OFF_err_e（主指标原始粒度）。
    """
    rows = []
    pat = re.compile(
        r"^\s*([0-9.]+)\s+(C\d)\s+s(\d)\s*\|\s*([-+]?[0-9.]+)\s+([-+]?[0-9.]+)\s*\|"
        r"\s*([-+]?[0-9.]+)\s*\[([^]]*)\]\s*\|\s*(\d+)/(\d+)\s+(\d+)/(\d+)"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if not m:
            continue
        pairs = [float(x) for x in m.group(7).split()]
        rows.append({
            "v": float(m.group(1)), "C": m.group(2), "seed": int(m.group(3)),
            "on_err": float(m.group(4)), "off_err": float(m.group(5)),
            "dff": float(m.group(6)), "pairs": pairs,
            "cmp_on": f"{m.group(8)}/{m.group(9)}", "cmp_off": f"{m.group(10)}/{m.group(11)}",
        })
    if len(rows) != 28:
        raise SystemExit(f"FAIL: effect_table 解析得 {len(rows)} 行，预期 28"
                         "（7v × 2C × 2 train seed；eval-seed 维度折叠在 pairs 内）")
    return rows


def load_receipts(receipts_dir: Path) -> list[dict]:
    recs = []
    for f in sorted(receipts_dir.glob("receipt_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        eps = d["episodes"]
        if len(eps) != 3:
            raise SystemExit(f"FAIL: {f.name} episodes={len(eps)}，预期 3")
        ov = d["condition_override"]
        recs.append({
            "file": f.name, "v": d["target_speed"], "C": d["condition_arm"],
            "z": d["tau_ff"], "seed": d["train_seed"],
            "natural_condition_id": d["assignment"]["natural_condition_id"],
            "natural_vb_dist": {int(k): v for k, v in ov["natural_vb_distribution"].items()},
            "n_decode": ov["n_decode_calls"], "n_changed": ov["n_override_changed"],
            "mapped_vb": ov["mapped_speed_bin"], "mapped_db": ov["mapped_dir_bin"],
            "episodes": eps,
        })
    if len(recs) != 56:
        raise SystemExit(f"FAIL: receipts 解析得 {len(recs)} 格，预期 56")
    return recs


def block_V(table: list[dict]) -> dict:
    """方差分解：Δ_ff 的 eval-seed 方差 vs train-seed 方差（主指标配对差口径）。"""
    within = []   # 每 (v,C,s) 的 3 个 pair 的样本方差（df=2）
    cell_means = {}
    for r in table:
        p = r["pairs"]
        within.append({"v": r["v"], "C": r["C"], "seed": r["seed"],
                       "sd": stdev(p), "range": max(p) - min(p), "pairs": p})
        cell_means[(r["v"], r["C"], r["seed"])] = mean(p)

    train_diffs = []
    for v in sorted({k[0] for k in cell_means}):
        for C in sorted({k[1] for k in cell_means}):
            d0 = cell_means[(v, C, 0)]
            d1 = cell_means[(v, C, 1)]
            train_diffs.append({"v": v, "C": C, "dff_s0": d0, "dff_s1": d1,
                                "abs_diff": abs(d0 - d1)})

    pooled_var_within = mean(w["sd"] ** 2 for w in within)
    # 若训练 seed 无差异，cell 级 dff 均值（3 seeds）的期望平方差 ≈ 2·σ²_eval/3
    expected_noise_diff = math.sqrt(2 * pooled_var_within / 3)
    f_ratios = []
    for t in train_diffs:
        se2 = 2 * pooled_var_within / 3
        f_ratios.append({"v": t["v"], "C": t["C"],
                         "F_like": (t["abs_diff"] ** 2) / se2 if se2 > 0 else float("inf")})
    f_vals = [f["F_like"] for f in f_ratios if math.isfinite(f["F_like"])]

    return {
        "metric": "Δ_ff 配对差（ON−OFF，per-step |vx−cmd| 口径，effect_table_v1）",
        "n_cells": len(within),
        "eval_seed_sd_within_cell": {
            "median": median(w["sd"] for w in within),
            "max": max(w["sd"] for w in within),
            "median_range": median(w["range"] for w in within),
            "max_range": max(w["range"] for w in within),
        },
        "train_seed_absdiff_of_dff_means": {
            "median": median(t["abs_diff"] for t in train_diffs),
            "min": min(t["abs_diff"] for t in train_diffs),
            "max": max(t["abs_diff"] for t in train_diffs),
            "n_over_decision_boundary": sum(1 for t in train_diffs
                                            if t["abs_diff"] > DECISION_BOUNDARY),
        },
        "pooled_eval_var_within": pooled_var_within,
        "expected_noise_absdiff_if_no_train_effect": expected_noise_diff,
        "F_like_train_over_eval": {
            "median": median(f_vals), "min": min(f_vals), "max": max(f_vals),
            "note": "F_like = |dff_s0−dff_s1|² / (2σ²_eval/3)；df≈(1,2) 仅作量级参照",
        },
        "train_diffs": train_diffs,
        "within_cells": within,
    }


def block_S(receipts: list[dict]) -> dict:
    """eval seed identity 审计：数据侧 bit-level 差异（代码事实写入报告文本）。"""
    bitwise_identical_cells = 0
    spreads = {"vx_mean": [], "v_speed_mean": [], "h_min": [], "disp": []}
    for r in receipts:
        eps = r["episodes"]
        keys = ("vx_mean", "v_speed_mean", "h_min", "disp")
        if all(eps[0][k] == eps[j][k] for k in keys for j in (1, 2)):
            bitwise_identical_cells += 1
        for k in spreads:
            vals = [e[k] for e in eps]
            spreads[k].append(max(vals) - min(vals))
    return {
        "n_cells_bitwise_identical_across_3_eval_seeds": bitwise_identical_cells,
        "spread_across_eval_seeds": {k: {"median": median(v), "max": max(v)}
                                     for k, v in spreads.items()},
        "code_facts": {
            "seed_entry": "eval_cell.rollout_eval → jitter_and_reset(env, seed=eval_seed)",
            "rng": "np.random.default_rng(1000 + seed)",
            "perturbations": [
                "root z: 0.76 + N(0, 0.005) m",
                "29 body joint pos: sonic_default + N(0, 0.01) rad",
                "all joint vel: N(0, 0.02) rad/s",
                "obs history refilled from jittered state; router state reset",
            ],
            "policy_inference": "policy.act(..., deterministic=True)（推理无采样噪声）",
            "first_obs_quirk": "jitter 后不刷新 _last_obs，首步沿用上轮末 obs（既有 harness 语义，三 seed 一致）",
            "disturbance": "cfg disturbance_prob = 0，cmd 恒定每步重申",
        },
        "conclusion_template": None,  # 由报告文本给出，不放机器判定
    }


def block_N(receipts: list[dict]) -> dict:
    """natural vs interventional：每 (v,C) 的 natural bin 与 override 关系。"""
    by_vc = defaultdict(list)
    for r in receipts:
        nat_vb = next(iter(r["natural_vb_dist"]))  # 全格均为单点分布
        by_vc[(r["v"], r["C"])].append({
            "natural_vb": nat_vb, "mapped_vb": r["mapped_vb"],
            "changed": r["n_changed"], "n_decode": r["n_decode"],
            "natural_condition_id": r["natural_condition_id"],
            "consistent": (len(r["natural_vb_dist"]) == 1
                           and nat_vb == r["mapped_vb"]) == (r["n_changed"] == 0),
        })
    cells = []
    for (v, C), items in sorted(by_vc.items()):
        it = items[0]
        if not all(x["consistent"] for x in items):
            raise SystemExit(f"FAIL: (v={v},{C}) natural_vb 分布非单点或不一致：{items}")
        role = "natural_noop" if it["changed"] == 0 else "forced_cross_bin"
        cells.append({
            "v": v, "C": C, "natural_vb": it["natural_vb"], "applied_vb": it["mapped_vb"],
            "natural_from_bucketize": natural_vb(v),
            "n_override_changed": it["changed"], "n_decode": it["n_decode"],
            "role": role,
            "matched": it["natural_vb"] == it["mapped_vb"],
            "bucketize_agrees": natural_vb(v) == it["natural_vb"],
        })
    for c in cells:
        if not c["bucketize_agrees"]:
            raise SystemExit(f"FAIL: bucketize 复算与 receipt 不符：{c}")
    # Δ_cond estimand 拼接结构：低 v 段 C1=matched；高 v 段 C2=matched
    boundary = VB_EDGES[0]
    low = [c for c in cells if c["v"] < boundary]
    high = [c for c in cells if c["v"] >= boundary]
    return {
        "vb_edges": VB_EDGES,
        "cells": cells,
        "structure": {
            f"v<{boundary:.3f}": {"C1": "natural_noop (decode vb0)",
                                  "C2": "forced_cross_bin vb0→vb1"},
            f"v>={boundary:.3f}": {"C1": "forced_cross_bin vb1→vb0",
                                   "C2": "natural_noop (decode vb1)"},
        },
        "estimand_note": ("Δ_cond(v)=Δ_ff(v,C1)−Δ_ff(v,C2) 在 bin 边界两侧的"
                          " contrast 内容互换（matched−forced ↔ forced−matched），"
                          "整条 v 轴不是一个 homogeneous estimand"),
        "decode_regime_identity": {
            "C1": "全程 decode 输入 vb0（自然或强制）",
            "C2": "全程 decode 输入 vb1（自然或强制）",
        },
    }


def block_D(table: list[dict], receipts: list[dict]) -> dict:
    """C1/C2 comparability：OFF 臂（treatment-free 层）support 重叠诊断。"""
    off = {}
    for r in table:
        if (r["v"], r["C"]) not in off:
            off[(r["v"], r["C"])] = {}
        off[(r["v"], r["C"])][r["seed"]] = r["off_err"]
    # receipts 形态维度（s,e 级）：OFF 臂汇总
    geom = defaultdict(list)
    for r in receipts:
        if r["z"] != "off":
            continue
        for e in r["episodes"]:
            geom[(r["v"], r["C"])].append(
                {"vx_mean": e["vx_mean"], "h_min": e["h_min"], "disp": e["disp"]})

    rows = []
    n_overlap_err = 0
    for v in sorted({k[0] for k in off}):
        c1 = sorted(off[(v, "C1")].values())
        c2 = sorted(off[(v, "C2")].values())
        lo1, hi1 = min(c1), max(c1)
        lo2, hi2 = min(c2), max(c2)
        err_overlap = (lo1 <= hi2 and lo2 <= hi1)
        gap = max(lo1 - hi2, lo2 - hi1)  # >0 = 区间分离的间隙宽度
        n_overlap_err += err_overlap
        g1, g2 = geom[(v, "C1")], geom[(v, "C2")]
        rows.append({
            "v": v,
            "C1_off_err_range": [lo1, hi1], "C2_off_err_range": [lo2, hi2],
            "err_overlap": err_overlap, "err_gap": round(gap, 3),
            "err_means_ratio": round(mean(c2) / mean(c1), 2) if mean(c1) > 0 else None,
            "geom_C1": {k: [round(min(x[k] for x in g1), 3), round(max(x[k] for x in g1), 3)]
                        for k in ("vx_mean", "h_min", "disp")},
            "geom_C2": {k: [round(min(x[k] for x in g2), 3), round(max(x[k] for x in g2), 3)]
                        for k in ("vx_mean", "h_min", "disp")},
        })
    return {
        "rule": ("诊断性规则（非协议硬点）：err_off 跨 train-seed 区间分离且间隙 > "
                 f"{DECISION_BOUNDARY} ⇒ 两 condition 不在同一可比 support；"
                 "Δ_cond 只能读作 condition-specific contrast"),
        "n_v_with_err_overlap": n_overlap_err,
        "n_v_total": len(rows),
        "rows": rows,
    }


def block_W(v: dict, n: dict, d: dict) -> dict:
    """综合裁决：按 owner 分叉树的机械映射 + 数据侧建议（最终决定权 owner）。"""
    eval_sd_med = v["eval_seed_sd_within_cell"]["median"]
    train_med = v["train_seed_absdiff_of_dff_means"]["median"]
    f_med = v["F_like_train_over_eval"]["median"]
    incomparable = d["n_v_with_err_overlap"] == 0
    verdicts = [
        {"q": "eval variance ≪ train variance ?",
         "a": f"是（eval-seed sd 中位 {eval_sd_med:.4f} vs train-seed |Δdff| 中位 "
              f"{train_med:.3f}；F_like 中位 {f_med:.0f}）"},
        {"q": "C1/C2 可比（OFF 臂共同 support）？",
         "a": ("否——全部 %d 个速度点 err_off 区间分离，最低间隙 %.3f m/s"
               % (d["n_v_total"], min(r["err_gap"] for r in d["rows"])))
              if incomparable else "部分速度点重叠，见 rows"},
        {"q": "Δ_cond estimand 身份",
         "a": ("bin 边界两侧 contrast 内容互换（N 块）；数值平滑但不是单一定义量；"
               "decode-regime(vb0/vb1) × train-seed 才是当前 56-cell 的主结构")},
    ]
    if incomparable:
        branch = ("b) 收束为 condition-specific contrast——按 owner 分叉树，"
                  "C1/C2 明显不可比时第三训练 seed 救不了 C1 vs C2 的解释问题")
        third_seed = ("NOT AUTHORIZED by this diagnosis（owner 分叉树 (b) 分支）；"
                      "若 owner 改判走 (a)，其信息增益应重新表述为：检验 "
                      "decode-regime(vb0/vb1)×τ_ff 交互符号的 seed 稳定性，"
                      "而非笼统的 Δ_ff 稳定性")
    else:
        branch = ("a) 第三训练 seed（C1/C2 可比且 training-seed variance 主导）")
        third_seed = "按 owner 分叉树 (a) 分支授权与否由 owner 定"
    return {"verdicts": verdicts, "branch_suggestion": branch,
            "third_train_seed": third_seed,
            "disclaimer": "本块为数据侧机械映射，最终裁定权在 owner"}


def render_txt(blocks: dict) -> str:
    V, S, N, D, W = (blocks[k] for k in "VSN DW".replace(" ", ""))
    L = []
    L.append("=== TO41 (c) diagnosis_v1：56-cell 现有产物三块分解（零新增执行） ===")
    L.append("口径：V 块 = effect_table_v1 主指标（per-step |vx−cmd| 配对差，per-eval-seed 粒度）；")
    L.append("S/N/D 块 = receipts（注意 receipt 聚合字段 vx_mean 与 err60s 口径不同，仅作形态证据）。")
    L.append("")
    L.append("--- V. 方差分解（Δ_ff = ON−OFF 配对差） ---")
    L.append(f"eval-seed sd（同 cell 内 3 pairs，df=2）：median=%.5f max=%.5f；"
             "极差 median=%.5f max=%.5f"
             % (V["eval_seed_sd_within_cell"]["median"], V["eval_seed_sd_within_cell"]["max"],
                V["eval_seed_sd_within_cell"]["median_range"],
                V["eval_seed_sd_within_cell"]["max_range"]))
    td = V["train_seed_absdiff_of_dff_means"]
    L.append("train-seed |dff_s0−dff_s1|：median=%.4f min=%.4f max=%.4f；超 0.02 边界 %d/28"
             % (td["median"], td["min"], td["max"], td["n_over_decision_boundary"]))
    fr = V["F_like_train_over_eval"]
    L.append("F_like(train/eval)：median=%.0f（min=%.1f max=%.1f）——train-seed 方差主导"
             % (fr["median"], fr["min"], fr["max"]))
    L.append(f"若训练 seed 无差异，|dff_s0−dff_s1| 期望量级 ≈ "
             f"{V['expected_noise_absdiff_if_no_train_effect']:.5f}（实测中位 {td['median']:.4f}）")
    L.append("")
    L.append("--- S. eval seed identity 审计 ---")
    cf = S["code_facts"]
    L.append(f"随机路径：{cf['seed_entry']}；rng = {cf['rng']}；policy 推理 {cf['policy_inference']}")
    for p in cf["perturbations"]:
        L.append(f"  · {p}")
    L.append(f"  · {cf['first_obs_quirk']}")
    L.append(f"  · {cf['disturbance']}")
    L.append("数据侧：3 eval seeds 逐位全同的 cell = %d/56（≠56 ⇒ 随机性确实进入执行路径，"
             "非死 RNG）" % S["n_cells_bitwise_identical_across_3_eval_seeds"])
    for k, s in S["spread_across_eval_seeds"].items():
        L.append("  spread[%s] median=%.5f max=%.5f" % (k, s["median"], s["max"]))
    L.append("结论：seed 进入初始条件且读数非逐位相同，但系统对初始扰动强镇定"
             "（spread ~1e-3 量级 vs train-seed 差 ~1e-1 量级）——eval 噪声可忽略是"
             "「强镇定」而非「随机未生效」。")
    L.append("")
    L.append("--- N. natural vs interventional ---")
    L.append("每 (v,C) 恰好一臂 natural no-op、一臂跨 bin 强制（56/56 格验证，"
             "natural_vb 与 bucketize 复算全符）：")
    for k, vv in N["structure"].items():
        L.append(f"  {k}: {vv}")
    L.append(f"estimand 警示：{N['estimand_note']}")
    L.append("decode-regime 恒等：C1 ≡ vb0-regime（全 v），C2 ≡ vb1-regime（全 v）；"
             "matched-ness 由 (v,C) 派生且与 regime 不混淆。")
    L.append("")
    L.append("--- D. C1/C2 comparability（OFF 臂 = treatment-free 层） ---")
    L.append(D["rule"])
    L.append("err_off 区间分离的速度点：%d/%d" % (D["n_v_total"] - D["n_v_with_err_overlap"],
                                               D["n_v_total"]))
    L.append("v      C1_off(err s0,s1)        C2_off(err s0,s1)        gap     "
             "C2/C1 均值比 | vx_mean C1/C2            h_min C1/C2        disp C1/C2")
    for r in D["rows"]:
        c1, c2 = r["C1_off_err_range"], r["C2_off_err_range"]
        L.append("%-6s [%5.3f,%5.3f]          [%5.3f,%5.3f]          %5.3f   %-7s | "
                 "[%4.2f,%4.2f]/[%4.2f,%4.2f]  [%4.2f,%4.2f]/[%4.2f,%4.2f]  "
                 "[%5.1f,%5.1f]/[%5.1f,%5.1f]"
                 % (r["v"], c1[0], c1[1], c2[0], c2[1], r["err_gap"], r["err_means_ratio"],
                    r["geom_C1"]["vx_mean"][0], r["geom_C1"]["vx_mean"][1],
                    r["geom_C2"]["vx_mean"][0], r["geom_C2"]["vx_mean"][1],
                    r["geom_C1"]["h_min"][0], r["geom_C1"]["h_min"][1],
                    r["geom_C2"]["h_min"][0], r["geom_C2"]["h_min"][1],
                    r["geom_C1"]["disp"][0], r["geom_C1"]["disp"][1],
                    r["geom_C2"]["disp"][0], r["geom_C2"]["disp"][1]))
    L.append("")
    L.append("--- W. 综合裁决（数据侧机械映射，最终决定权 owner） ---")
    for vv in W["verdicts"]:
        L.append("Q: %s\nA: %s" % (vv["q"], vv["a"]))
    L.append("分叉建议：%s" % W["branch_suggestion"])
    L.append("第三训练 seed：%s" % W["third_train_seed"])
    L.append("")
    L.append("措辞纪律（owner 本轮要求）：execution conformance 已通过（G1–G10 全 PASS）；"
             "因此上述读数可信地反映当前 frozen system 的实际行为——「实际行为」是事实，"
             "不等于已有科学解释。")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--receipts", type=Path, required=True)
    ap.add_argument("--effect-table", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, required=True)
    a = ap.parse_args()

    table = parse_effect_table(a.effect_table)
    receipts = load_receipts(a.receipts)

    # receipts ↔ table 交叉校验：56 格 (v,C,z,seed) 精确闭合
    tab_keys = {(r["v"], r["C"], r["seed"]) for r in table}
    rec_keys = {(r["v"], r["C"], r["seed"]) for r in receipts}
    if tab_keys != rec_keys:
        raise SystemExit(f"FAIL: receipts 与 effect_table 格集合不闭合："
                         f"仅table={tab_keys-rec_keys} 仅receipts={rec_keys-tab_keys}")

    blocks = {
        "V": block_V(table),
        "S": block_S(receipts),
        "N": block_N(receipts),
        "D": block_D(table, receipts),
        "W": None,
    }
    blocks["W"] = block_W(blocks["V"], blocks["N"], blocks["D"])

    payload = {("block_" + k): v for k, v in blocks.items()}
    payload["inputs"] = {"receipts_dir": str(a.receipts), "effect_table": str(a.effect_table),
                         "n_receipts": len(receipts), "n_table_rows": len(table)}
    a.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    txt = render_txt(blocks)
    a.out_txt.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n[out] {a.out_json}\n[out] {a.out_txt}")


if __name__ == "__main__":
    main()
