"""TO38 配对差分分析：读双臂评测 JSON，产出 floor 检查 + 主指标配对差分 +
三分支判定（决策表见 refine-logs/TO38_PLAN.md §4）。

两臂 best ckpt 各自独立（对称规则：train_log reward 50-iter 窗口最优段），
所以配对比iable按 (a_ckpt, b_ckpt) 组合跑；默认两组：
  it150,it300  — 各臂 best-window ckpt 对（主判定）
  final,final  — final ckpt 对（稳健性对照）

用法（lab-ts，evals 完成后）：
  python3 apt_g1/to38_analyze.py --out-dir apt_g1/outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DELTA = 0.03  # 等效边界（m/s）
CTAG = {0.2: "a020", 0.277: "a0277", 0.35: "a035"}  # 与 to38_eval.sh tag 一致


def load_arm(out: Path, arm: str, ck: str, ctag: str) -> dict | None:
    """匹配 {arm}_eval_{ck}[_pt]_{ctag}.json（eval 链把 .pt 转成 _pt）。"""
    hits = sorted(out.glob(f"{arm}_eval_{ck}*_{ctag}.json"))
    return json.loads(hits[0].read_text()) if hits else None


def a_sections(d: dict) -> dict:
    """A_walk60 -> {seed0: {...}}；latent 模式 aux/noaux 同结果，取 aux。"""
    a = d.get("A_walk60", {})
    if "aux" in a:
        a = a["aux"]
    return {k: v for k, v in a.items() if k.startswith("seed")}


def run_pair(out: Path, cka: str, ckb: str, cmds: list[float]) -> None:
    print(f"\n== 配对 a@{cka} vs b@{ckb} ==")
    floor_ok = {"to38a": True, "to38b": True}
    all_diff: list[float] = []
    for cmd in cmds:
        ctag = CTAG.get(cmd, f"a{round(cmd * 1000):03d}")
        da = load_arm(out, "to38a", cka, ctag)
        db = load_arm(out, "to38b", ckb, ctag)
        if da is None or db is None:
            print(f"cmd {cmd}: 缺 eval 文件（{cka}*_{ctag} / {ckb}*_{ctag}）——跳过")
            continue
        sa, sb = a_sections(da), a_sections(db)
        rows_a = [sa[s] for s in sorted(sa)]
        rows_b = [sb[s] for s in sorted(sb)]
        errs_a = [abs(r["vx"] - cmd) for r in rows_a]
        errs_b = [abs(r["vx"] - cmd) for r in rows_b]
        for arm, rows in (("to38a", rows_a), ("to38b", rows_b)):
            for r in rows:
                if not r["completed"] or r["h_min"] < 0.6 or r["disp"] <= 0.5:
                    floor_ok[arm] = False
        n = min(len(errs_a), len(errs_b))
        diff = [errs_a[i] - errs_b[i] for i in range(n)]  # <0 = 主臂更好
        all_diff += diff
        md = float(np.mean(diff)) if diff else float("nan")
        print(f"cmd {cmd}: a |vx-cmd| = {np.mean(errs_a):.3f}±{np.std(errs_a):.3f}"
              f"  b = {np.mean(errs_b):.3f}±{np.std(errs_b):.3f}"
              f"  配对差分 a-b = {md:+.3f} (n={n})")
        for arm, rows in (("a", rows_a), ("b", rows_b)):
            det = "; ".join(f"vx={r['vx']} h={r['h_min']} disp={r['disp']}"
                            f" done={r['completed']}" for r in rows)
            print(f"   {arm}: {det}")

    # full battery（B 推扰存活率，best ckpt 才有）
    for arm, ck in (("to38a", cka), ("to38b", ckb)):
        d = load_arm(out, arm, ck, "full")
        if d is None:
            print(f"{arm}: 缺 full battery eval（{ck}_full）")
            continue
        b = d.get("B_disturbance", {}).get("aux", {})
        if b:
            done = [v["completed"] for v in b.values()]
            a60 = d.get("A_walk60", {}).get("aux", {})
            std = [v["completed"] for v in a60.values()]
            print(f"{arm} full battery: A@0.8 {sum(std)}/{len(std)}、"
                  f"B 500/1250N 推扰 {sum(done)}/{len(done)} 完成")

    # 三分支
    if not (floor_ok["to38a"] and floor_ok["to38b"]):
        bad = [k for k, v in floor_ok.items() if not v]
        print(f"floor FAIL: {bad} —— 按 PLAN §4，只报失败/不可判定 + 归因方向")
    elif all_diff:
        md = float(np.mean(all_diff))
        if md < -DELTA:
            verdict = "分支一：a 显著优于 ctrl —— TO 参考 RL 可消化（低速带增益）"
        elif md > DELTA:
            verdict = "分支三：a 劣于 ctrl —— TO 参考与解码器先验冲突（负结果）"
        else:
            verdict = "分支二：等效 —— 注入被忽略（信噪比不足，E48 同族；需跟踪 RMSE 归因）"
        print(f"合并配对差分均值 = {md:+.4f}（δ={DELTA}，n={len(all_diff)}）→ {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="apt_g1/outputs")
    ap.add_argument("--cmds", default="0.2,0.277,0.35")
    ap.add_argument("--pairs", default="it150,it300;final,final",
                    help="分号分隔的 (a_ckpt,b_ckpt) 配对；每臂 best=各自窗口最优")
    args = ap.parse_args()
    out = Path(args.out_dir)
    cmds = [float(c) for c in args.cmds.split(",")]
    for pair in args.pairs.split(";"):
        parts = pair.split(",")
        run_pair(out, parts[0], parts[1] if len(parts) > 1 else parts[0], cmds)


if __name__ == "__main__":
    main()
