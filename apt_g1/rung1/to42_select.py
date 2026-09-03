"""TO42 checkpoint 机械选择：train_log 50-iter 窗口最优 → selection manifest。

角色（SCRIPT_MAP 登记）：**read-only analysis**。规则逐字继承
select_checkpoint.py（TO40C_PLAN §4 预注册：argmax mean(rewards) over
50-iter 窗口，tie 取最小窗口序；对称、非手挑、禁按 eval 表现二次选择），
臂集合换为 TO42 的 {lsel, fbkt} × seeds {0, 1}。产物 manifest 是 eval driver
与 to42_checker 的唯一 checkpoint 身份源：同一 (arm, seed) 的全部 7 个 v
cells 共用同一 selected ckpt。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from apt_g1.rung1.mode_a_runtime import REPO_ROOT, _utcnow
from apt_g1.rung1.select_checkpoint import select_run

ARTIFACT_ID = "to42-ckpt-selection/v1"
TRAIN_SEEDS = (0, 1)
POLICY_ARMS = ("lsel", "fbkt")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="TO42 checkpoint 机械选择（50-iter 窗口最优 → manifest）")
    ap.add_argument("--runs-root", type=Path,
                    default=REPO_ROOT / "output/to42")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "output/to42/ckpt_selection.json")
    args = ap.parse_args()

    runs: dict = {arm: {} for arm in POLICY_ARMS}
    for arm in POLICY_ARMS:
        for seed in TRAIN_SEEDS:
            run_dir = args.runs_root / f"to42r1-{arm}-s{seed}"
            runs[arm][str(seed)] = select_run(run_dir)

    manifest = {
        "artifact": ARTIFACT_ID,
        "rule": ("checkpoint selection mechanically derived from training "
                 "trajectories only; ONE pre-registered ckpt per (arm, seed) "
                 "shared by all 7 eval v-cells of that (arm, seed); identical "
                 "rule to TO41 rung1 select_checkpoint (50-iter window argmax, "
                 "tie-break = lowest window end iter)"),
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
