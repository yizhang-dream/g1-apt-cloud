"""TO42 本地分析（checker PASS 之后才允许运行——先审计后分析纪律）。

角色（SCRIPT_MAP 登记）：**read-only analysis**（本机消费 to42_artifacts.pt
或原始 receipts 目录；零执行）。产出 = 判读输入，不是结论本身：停止规则归支
与 TO41 前沿对照的机械计算，科学表述权在 tracker/HANDOFF 收束文档。

计算四块（TO42_PLAN §2 预注册口径）：
  A. err60s 效应表：per (arm, v, train_seed) = 3 eval seeds 均值（主指标口径
     = per-step |vx−cmd| 的 eval-seed 均值，receipt 原生）。
  B. selection-interface contrast：err60s(lsel) − err60s(fbkt)，配对（同 v 同
     train seed）；主对照段 = mid-band {0.275, 0.277, 0.300, 0.325}。
  C. TO41 绝对前沿对照（收束文 §2 PRIMARY OFF 臂逐字硬编码，标注来源）：
     per (v, seed) best-fixed-regime err = min(C1_off, C2_off)；H1 判定 =
     mid-band err60s(lsel) 相对前沿降低 ≥ 0.02。
  D. 机制预注册检验（H1 唯一通路）：realized speed 的 cmd 响应性——lsel 臂
     vx_mean(v) 斜率对照 TO41 ≈ 0（收束文 §3d：C1 恒 ~0.13 / C2 恒 ~0.61）；
     若 err 改善而速度斜率 ≈ 0 → 标记 metric artifact 线索，不升格结论。
  E. 停止规则归支：(a) 选择器塌缩（duty ≥90% 单 regime 且不随 v 变化）/
     (b) mid-band 无 ≥0.02 改善 / (c) 改善但随 seed 换向。

用法：python -m apt_g1.rung1.to42_analysis --artifacts <to42_artifacts.pt>
     （或 --eval-dir <receipts 目录>）
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

GRID7 = (0.200, 0.225, 0.250, 0.275, 0.277, 0.300, 0.325)
MID_BAND = (0.275, 0.277, 0.300, 0.325)
SEEDS = (0, 1)
ARMS = ("lsel", "fbkt")
# TO41 收束文 §2 PRIMARY 表 OFF 臂（err60s，逐字硬编码；来源 = effect_table_v1）
TO41_OFF = {
    0.200: {"C1": {0: 0.028, 1: 0.043}, "C2": {0: 0.382, 1: 0.386}},
    0.225: {"C1": {0: 0.053, 1: 0.065}, "C2": {0: 0.359, 1: 0.362}},
    0.250: {"C1": {0: 0.080, 1: 0.090}, "C2": {0: 0.333, 1: 0.339}},
    0.275: {"C1": {0: 0.104, 1: 0.115}, "C2": {0: 0.308, 1: 0.313}},
    0.277: {"C1": {0: 0.109, 1: 0.119}, "C2": {0: 0.306, 1: 0.309}},
    0.300: {"C1": {0: 0.138, 1: 0.150}, "C2": {0: 0.275, 1: 0.283}},
    0.325: {"C1": {0: 0.198, 1: 0.223}, "C2": {0: 0.263, 1: 0.270}},
}
# TO41 收束文 §3d realized speed（OFF 臂，近似中心值，供斜率对照）
TO41_VX = {"C1": 0.13, "C2": 0.61}


def load_receipts(args) -> dict:
    """返回 {(arm, seed, v): receipt}。优先 artifacts bundle，退回 receipts 目录。"""
    if args.artifacts:
        import torch

        payload = torch.load(args.artifacts, map_location="cpu", weights_only=False)
        raw = payload["receipts"]
    else:
        raw = {p.name: json.loads(p.read_text("utf-8"))
               for p in sorted(Path(args.eval_dir).glob("receipt_*.json"))}
    out = {}
    for _, rc in raw.items():
        out[(rc["arm"], int(rc["train_seed"]), round(float(rc["target_speed"]), 3))] = rc
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="TO42 本地分析（checker PASS 后）")
    ap.add_argument("--artifacts", type=Path, default=None)
    ap.add_argument("--eval-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not args.artifacts and not args.eval_dir:
        ap.error("--artifacts 或 --eval-dir 二选一")

    rcs = load_receipts(args)
    missing = [(a, s, v) for a in ARMS for s in SEEDS for v in GRID7
               if (a, s, v) not in rcs]
    if missing:
        raise SystemExit(f"FAIL: 缺 {len(missing)} receipts，如 {missing[:3]}")

    audit_verdict = None
    if args.artifacts:
        import torch

        payload = torch.load(args.artifacts, map_location="cpu", weights_only=False)
        audit_verdict = payload.get("audit", {}).get("verdict")
        print(f"[audit] checker verdict = {audit_verdict}"
              + ("（先审计后分析：以下读数仅在其为 PASS 时有效）"
                 if audit_verdict != "PASS" else ""))

    # A. 效应表
    table = {}
    for a in ARMS:
        for s in SEEDS:
            for v in GRID7:
                eps = rcs[(a, s, v)]["episodes"]
                table[(a, s, v)] = {
                    "err60s": round(float(np.mean([e["err60s"] for e in eps])), 4),
                    "vx_mean": round(float(np.mean([e["vx_mean"] for e in eps])), 4),
                    "disp": round(float(np.mean([e["disp"] for e in eps])), 2),
                    "h_min": round(min(e["h_min"] for e in eps), 3),
                    "n_switches": round(float(np.mean([e["n_switches"] for e in eps])), 1),
                    "p1": round(float(np.mean([e["sel_head_p1_mean"] for e in eps])), 4),
                }
    print("\n=== A. err60s 效应表（3 eval seeds 均值）===")
    print(f"{'v':>6} {'s':>2} | {'lsel':>7} {'fbkt':>7} {'vx(lsel)':>8} "
          f"{'switch':>6} {'p(vb1)':>7} | {'TO41前沿':>8}")
    for v in GRID7:
        for s in SEEDS:
            frontier = min(TO41_OFF[v]["C1"][s], TO41_OFF[v]["C2"][s])
            print(f"{v:>6.3f} {s:>2} | {table[('lsel', s, v)]['err60s']:>7.4f} "
                  f"{table[('fbkt', s, v)]['err60s']:>7.4f} "
                  f"{table[('lsel', s, v)]['vx_mean']:>8.3f} "
                  f"{table[('lsel', s, v)]['n_switches']:>6.1f} "
                  f"{table[('lsel', s, v)]['p1']:>7.3f} | {frontier:>8.3f}")

    # B. selection-interface contrast（配对）
    print("\n=== B. selection-interface contrast：err60s(lsel) − err60s(fbkt) ===")
    contrasts = {}
    for s in SEEDS:
        for v in GRID7:
            contrasts[(s, v)] = round(
                table[("lsel", s, v)]["err60s"] - table[("fbkt", s, v)]["err60s"], 4)
        mid = round(float(np.mean([contrasts[(s, v)] for v in MID_BAND])), 4)
        allv = round(float(np.mean([contrasts[(s, v)] for v in GRID7])), 4)
        print(f"seed {s}: mid-band 均值 = {mid:+.4f} | 全 v 均值 = {allv:+.4f} | "
              f"逐点 = {[f'{v:.3f}:{contrasts[(s, v)]:+.3f}' for v in GRID7]}")

    # C. TO41 前沿对照 + H1
    print("\n=== C. TO41 best-fixed 前沿对照（H1：mid-band 降低 ≥ 0.02）===")
    h1 = {}
    for s in SEEDS:
        gaps = {v: round(table[("lsel", s, v)]["err60s"]
                         - min(TO41_OFF[v]["C1"][s], TO41_OFF[v]["C2"][s]), 4)
                for v in GRID7}
        mid_gap = round(float(np.mean([gaps[v] for v in MID_BAND])), 4)
        h1[s] = {"mid_band_gap_vs_frontier": mid_gap, "per_v": gaps}
        print(f"seed {s}: mid-band err(lsel) − 前沿 = {mid_gap:+.4f} "
              f"(H1 阈 = −0.02) | 逐点 = "
              f"{[f'{v:.3f}:{gaps[v]:+.3f}' for v in GRID7]}")

    # D. realized speed cmd 响应性（机制预注册检验）
    print("\n=== D. realized speed vs cmd（lsel；TO41 对照 ≈ 0 斜率）===")
    slope = {}
    for s in SEEDS:
        vx = [table[("lsel", s, v)]["vx_mean"] for v in GRID7]
        k = (vx[-1] - vx[0]) / (GRID7[-1] - GRID7[0])
        slope[s] = round(k, 3)
        print(f"seed {s}: vx {vx[0]:.3f}→{vx[-1]:.3f} 斜率 ≈ {k:+.2f} "
              f"(TO41: C1/C2 各自 ≈0；cmd 扫描 ±63%)")

    # E. 停止规则归支（机械判据，科学表述归收束文档）
    print("\n=== E. 停止规则预检 ===")
    duties = {}
    for s in SEEDS:
        for v in GRID7:
            eps = rcs[("lsel", s, v)]["episodes"]
            fracs = []
            for e in eps:
                sel = np.frombuffer(base64.b64decode(e["sel_timeline_b64"]),
                                    dtype=np.uint8)
                fracs.append(float((sel == 1).mean()))
            duties[(s, v)] = round(float(np.mean(fracs)), 3)
    for s in SEEDS:
        d = [duties[(s, v)] for v in GRID7]
        collapse = (all(x >= 0.9 for x in d) or all(x <= 0.1 for x in d))
        var = round(float(np.var(d)), 4)
        print(f"seed {s}: duty(vb1) 逐 v = {d} "
              f"{'⚠ 塌缩嫌疑（停止规则 a）' if collapse and var < 0.001 else ''}")
    improved = {s: h1[s]["mid_band_gap_vs_frontier"] <= -0.02 for s in SEEDS}
    if all(improved.values()):
        branch = "H1 成立路径（mid-band 双 seed 改善 ≥0.02）——对照 D 定机制解读"
    elif any(improved.values()):
        branch = "停止规则 (c) 嫌疑：改善但 seed 依赖（须逐 seed 细读）"
    else:
        branch = "停止规则 (b) 方向：mid-band 无 ≥0.02 改善（混合算术预期；对照 D 定表述）"
    print(f"归支预检：{branch}")

    out = {
        "artifact": "to42-analysis/v1",
        "audit_verdict": audit_verdict,
        "table": {f"{a}|{s}|{v:.3f}": t for (a, s, v), t in table.items()},
        "contrasts_lsel_minus_fbkt": {f"{s}|{v:.3f}": c for (s, v), c in contrasts.items()},
        "h1_vs_frontier": h1,
        "vx_slope_lsel": slope,
        "duty_vb1_lsel": {f"{s}|{v:.3f}": d for (s, v), d in duties.items()},
        "stop_rule_preview": branch,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"\nOK analysis -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
