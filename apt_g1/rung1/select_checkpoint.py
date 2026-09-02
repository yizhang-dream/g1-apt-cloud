"""TO41 Rung 1 checkpoint 机械选择：train_log 50-iter 窗口最优 → selection manifest。

角色（SCRIPT_MAP 登记）：**read-only analysis**（只读 train_log 与 ckpt 文件、
写 selection manifest；不改任何训练/实验状态）。三十九轮 owner 裁定的
checkpoint selection 隔离纪律的机械化：

    training trajectory → ONE pre-registered checkpoint → 28 eval cells

规则（预注册，TO40C_PLAN §4 / TO38 同款，逐字机械化）：
各臂 train_log reward 的 50-iter 窗口最优段 → ckpt = 该窗口末的
policy_it_{N}.pt（训练脚本每 50 iter 落一个 ckpt）；并列取窗口序号最小者
（确定性 tie-break）。**对称、非手挑**：四臂（ctrl/t10 × seed 0/1）用同一
规则独立执行；禁止任何按 eval 表现 / 按 condition 的二次选择。另录
policy_final.pt 作稳健性对照（预注册语义，非 primary）。

产出的 manifest 是 eval_cell.py（driver 消费）与 eval_checker.py（审计比对）
的唯一 checkpoint 身份源：同一 (arm, seed) 的全部 14 eval cells 共用同一
selected checkpoint——C1→ckptA / C2→ckptB 或 ON/OFF 各选各的会被 checker
G2 判 FAIL。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from apt_g1.rung1.mode_a_runtime import REPO_ROOT, _utcnow, sha256_file

ARTIFACT_ID = "rung1-eval-ckpt-selection/v1"
WINDOW = 50
TRAIN_SEEDS = (0, 1)
POLICY_ARMS = ("ctrl", "t10")


def select_run(run_dir: Path) -> dict:
    """单 run 的窗口选择：argmax over 50-iter mean reward（tie → 最小窗口序）。"""
    log_path = run_dir / "train_log.json"
    if not log_path.exists():
        raise SystemExit(f"FAIL: {log_path} 不存在（训练未完成？）")
    log = json.loads(log_path.read_text(encoding="utf-8"))
    rewards = log["rewards"]
    if len(rewards) < WINDOW:
        raise SystemExit(f"FAIL: {log_path} rewards 长度 {len(rewards)} < {WINDOW}")

    ckpts = {}
    for p in run_dir.glob("policy_it_*.pt"):
        it = int(p.stem.split("_")[-1])
        ckpts[it] = p
    if not ckpts:
        raise SystemExit(f"FAIL: {run_dir} 无 policy_it_*.pt（训练未完成？）")

    best = None  # (mean_rew, window_end_it, path) — tie 取最小 window_end_it
    for it in sorted(ckpts):
        lo = it - WINDOW
        if lo < 0:
            continue
        mean_rew = sum(rewards[lo:it]) / WINDOW
        if best is None or mean_rew > best[0] + 1e-12:
            best = (mean_rew, it, ckpts[it])
    if best is None:
        raise SystemExit(f"FAIL: {run_dir} 无完整 50-iter 窗口")

    mean_rew, it, ckpt_path = best
    final_path = run_dir / "policy_final.pt"
    entry = {
        "run_dir": str(run_dir),
        "train_log_sha256": sha256_file(log_path),
        "n_train_iters": len(rewards),
        "window_iters": [it - WINDOW + 1, it],
        "window_mean_rew": round(mean_rew, 6),
        "ckpt_file": str(ckpt_path),
        "ckpt_sha256": sha256_file(ckpt_path),
        "final_ckpt_file": str(final_path) if final_path.exists() else None,
        "final_ckpt_sha256": sha256_file(final_path) if final_path.exists() else None,
        "rule": ("argmax mean(train_log.rewards) over 50-iter windows aligned to "
                 "policy_it_{N}.pt saves; tie-break = lowest window end iter; "
                 "pre-registered (TO40C_PLAN §4), applied symmetrically to all arms"),
    }
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rung 1 checkpoint 机械选择（50-iter 窗口最优 → manifest）")
    ap.add_argument("--runs-root", type=Path, default=REPO_ROOT / "apt_g1/outputs")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "apt_g1/outputs/sync/to41_eval/ckpt_selection.json")
    ap.add_argument("--runs", default=None,
                    help="逗号分隔子集（如 ctrl-s0），仅限工具自测；正式 manifest 必须四臂齐全")
    args = ap.parse_args()

    run_names = (args.runs.split(",") if args.runs
                 else [f"{arm}-s{seed}" for arm in POLICY_ARMS for seed in TRAIN_SEEDS])
    runs: dict = {"ctrl": {}, "t10": {}}
    for name in run_names:
        arm, seed = name.rsplit("-s", 1)
        if arm not in runs or seed not in {"0", "1"}:
            raise SystemExit(f"FAIL: 未知 run 名 {name}（期望 {arm}-s{{0,1}}）")
        runs[arm][seed] = select_run(args.runs_root / f"to41r1-{name}")

    if args.runs is None:
        for arm in POLICY_ARMS:
            for seed in map(str, TRAIN_SEEDS):
                if seed not in runs[arm]:
                    raise SystemExit(f"FAIL: runs.{arm}.{seed} 缺失（正式 manifest 必须四臂齐全）")

    manifest = {
        "artifact": ARTIFACT_ID,
        "rule": ("checkpoint selection mechanically derived from training "
                 "trajectories only; ONE pre-registered ckpt per (arm, seed) "
                 "shared by all 14 eval cells of that (arm, seed); eval "
                 "conditions (C1/C2, τ ON/OFF) play no role in selection"),
        "generated_utc": _utcnow(),
        "runs_root": str(args.runs_root),
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    for arm in runs:
        for seed, e in runs[arm].items():
            print(f"[select] {arm}-s{seed}: it{e['window_iters'][1]} "
                  f"rew={e['window_mean_rew']} -> {Path(e['ckpt_file']).name}")
    print(f"OK selection manifest -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
