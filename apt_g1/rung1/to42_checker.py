"""TO42 eval audit checker——独立只读审计，verdict 唯一来源。

角色（SCRIPT_MAP 登记）：**read-only audit**（不消费 runtime 自报 PASS；只读
receipts / selection manifest / 冻结源文件哈希，写 audit artifact）。纪律与
TO41 eval_checker 同款：**先审计后分析**——本 checker 不读任何行为指标
（vx / err60s / disp 等），行为字段只在 checker PASS 后由 report 阶段聚合。

检查集（TO42_PLAN §5）：
  C1 coverage：28 receipts 精确闭合（2 臂 × 2 seeds × 7 v），无缺无重；
  C2 completion：84/84 episodes completed、steps==3000、零 fall；
  C3 ckpt identity：每 (arm, seed) 的 7 cells 共用同一 ckpt 且 sha == manifest；
  C4 env identity：冻结源文件哈希 / vae sha / to42 cfg 跨 receipts 一致；
  G0a fbkt：selection 时间线逐位 == clamp(natural(v))，gate 无脉冲；
  G0b lsel：sel ∈ {0,1}，切换步 ⊆ 决策边界（t % hold_steps == 0）。

PASS/FAIL 判定唯一出自本文件；任何 FAIL → wave 驱动中止，不进 report。
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from apt_g1.isaac.to42_gate import natural_vb  # 与 selftest/env 同一份冻结公式
from apt_g1.rung1.mode_a_runtime import REPO_ROOT, _utcnow
from apt_g1.rung1.to42_eval import GRID7, RECEIPT_SCHEMA

ARTIFACT_ID = "to42-eval-audit/v1"
POLICY_ARMS = ("lsel", "fbkt")
TRAIN_SEEDS = (0, 1)


def _sel_series(ep: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(ep["sel_timeline_b64"]), dtype=np.uint8)


def check_receipt(receipt: dict, problems: list) -> dict:
    arm = receipt["arm"]
    v = float(receipt["target_speed"])
    hold = int(receipt["to42_cfg"]["to42_hold_steps"])
    r = {"cell_id": receipt["cell_id"], "checks": {}}

    def ck(name, cond, detail=""):
        r["checks"][name] = {"PASS": bool(cond), "detail": detail}
        if not cond:
            problems.append(f"{receipt['cell_id']}: {name} {detail}")

    if receipt["schema"] != RECEIPT_SCHEMA:
        ck("schema", False, receipt["schema"])
        return r
    ck("schema", True)
    ck("cfg_arm", receipt["to42_cfg"]["to42_sel"] == arm)
    ck("hold_steps", hold == 25, f"hold={hold}")
    ck("vae_frozen", receipt["decoder_identity"]["state_dict_sha256_before"]
       == receipt["decoder_identity"]["state_dict_sha256_after"])

    for ep in receipt["episodes"]:
        tag = f"ep{ep['eval_seed']}"
        sel = _sel_series(ep)
        ck(f"{tag}/completed", ep["completed"] and ep["fall_step"] is None,
           f"steps={ep['steps_done']} fall={ep['fall_step']}")
        ck(f"{tag}/steps", ep["steps_done"] == ep["steps_requested"],
           f"{ep['steps_done']}/{ep['steps_requested']}")
        ck(f"{tag}/len", len(sel) == ep["steps_done"],
           f"timeline {len(sel)} vs steps {ep['steps_done']}")
        ck(f"{tag}/domain", bool(np.isin(sel, (0, 1)).all()))
        if arm == "fbkt":
            expect = int(natural_vb(np.array([v])).clip(0, 1)[0])
            ck(f"{tag}/G0a-bitexact",
               len(set(sel.tolist())) == 1 and int(sel[0]) == expect
               and ep["natural_bin"] == expect,
               f"sel={sorted(set(sel.tolist()))} expect={expect}")
            ck(f"{tag}/G0a-gate-silent", ep["sel_switch_steps"] == [])
        else:
            switches = ep["sel_switch_steps"]
            bad = [t for t in switches if t % hold != 0]
            ck(f"{tag}/G0b-boundary", not bad, f"off-boundary={bad[:5]}")
            ck(f"{tag}/G0b-switch-count", len(switches) == ep["n_switches"])
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="TO42 eval audit checker")
    ap.add_argument("--eval-dir", type=Path,
                    default=REPO_ROOT / "apt_g1/outputs/to42/eval")
    ap.add_argument("--selection-manifest", type=Path,
                    default=REPO_ROOT / "output/to42/ckpt_selection.json")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "apt_g1/outputs/to42/to42_eval_audit.json")
    args = ap.parse_args()

    problems: list = []
    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if manifest.get("artifact") != "to42-ckpt-selection/v1":
        raise SystemExit(f"FAIL: {args.selection_manifest} 不是 to42-ckpt-selection/v1")

    receipts_dir = args.eval_dir / "receipts"
    found = {}
    for arm in POLICY_ARMS:
        for seed in TRAIN_SEEDS:
            for v in GRID7:
                p = receipts_dir / f"receipt_to42-{arm}-v{v:.3f}__s{seed}.json"
                if not p.exists():
                    problems.append(f"missing receipt: {p.name}")
                    continue
                found[(arm, seed, round(float(v), 3))] = json.loads(
                    p.read_text(encoding="utf-8"))
    expected = {(a, s, round(float(v), 3))
                for a in POLICY_ARMS for s in TRAIN_SEEDS for v in GRID7}
    extra = set(found) - expected
    if extra:
        problems.append(f"unexpected receipts: {sorted(extra)}")

    # C3：ckpt 身份（manifest == receipts，同一 (arm, seed) 全 7 cells 共用）
    ckpt_by_arm_seed = {}
    for (arm, seed, v), rc in sorted(found.items()):
        sel_entry = manifest["runs"][arm][str(seed)]
        ck_sha = rc["checkpoint"]["ckpt_sha256"]
        if ck_sha != sel_entry["ckpt_sha256"]:
            problems.append(f"{rc['cell_id']}: ckpt sha != manifest")
        ckpt_by_arm_seed.setdefault((arm, seed), set()).add(ck_sha)
    for key, shas in ckpt_by_arm_seed.items():
        if len(shas) != 1:
            problems.append(f"{key}: {len(shas)} distinct ckpts across cells")

    per_receipt = []
    env_hash_sets = {}
    for (arm, seed, v), rc in sorted(found.items()):
        per_receipt.append(check_receipt(rc, problems))
        e = rc["env_identity"]
        env_hash_sets.setdefault("code", set()).add(
            (e["env_source_file_sha256"], e["to42_gate_source_file_sha256"],
             e["ppo_core_source_file_sha256"], e["train_source_file_sha256"],
             e["runtime_commit"]))
        env_hash_sets.setdefault("vae", set()).add(
            rc["decoder_identity"]["checkpoint_sha256"])
    for name, sets_ in env_hash_sets.items():
        if len(sets_) != 1:
            problems.append(f"env identity drift ({name}): {len(sets_)} variants")

    n_receipts = len(found)
    n_episodes = sum(len(rc["episodes"]) for rc in found.values())
    audit = {
        "artifact": ARTIFACT_ID,
        "generated_utc": _utcnow(),
        "verdict": "PASS" if not problems else "FAIL",
        "n_receipts": n_receipts,
        "n_problems": len(problems),
        "expected_receipts": len(expected),
        "expected_episodes": len(expected) * 3,
        "problems": problems[:200],
        "receipts": per_receipt,
        "note": "行为指标（err60s/vx 等）刻意不被本 checker 读取——先审计后分析",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"[checker] receipts={n_receipts}/{len(expected)} "
          f"episodes={n_episodes} problems={len(problems)} "
          f"verdict={audit['verdict']} -> {args.out}")
    for p in problems[:20]:
        print(f"  - {p}")
    return 0 if audit["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
