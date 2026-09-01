#!/usr/bin/env python3
"""TO41 Rung 1 τ(v) material campaign driver（全门集判定 + 两字段 accounting）。

门集（PASS = 全部通过，TO41_RUNG1_IMPL.md §5.2/§5.7）：
  G1 必填字段      G2 solver terminal success（solve log 末级 success=True）
  G3 gate_a∧gate_b G4 energy_drift < 2.0   G5 ke_drop ≥ −1e-6
  G6 float 字段无 NaN/Inf                  G7 |v_avg − requested_v| ≤ 0.02
  G8 knots/mode 配置身份与 manifest 一致
两字段 accounting（十九轮）：solver_terminal_status 与 material_status 分离，
不得把「solver 中途崩」记成「material 无效」、也不得反向。
旧 PASS 事件被改判时必须留 superseded 标记，不覆盖（driver_buggy_classification
→ manual review → corrected 的审计链）。
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REQUIRED_FIELDS = {
    "X_left", "X_right", "XM_left", "XM_right", "lam_lr", "lam_rl",
    "v_aux_lr", "v_aux_rl", "T", "T_left", "T_right", "step", "v_avg",
    "knots", "mode", "energy_drift_left", "energy_drift_right",
    "gate_a", "gate_b", "ke_drop_lr", "ke_drop_rl",
}
TOL = 0.02
DRIFT_MAX = 2.0
KE_DROP_MIN = -1e-6


def gate_load(d):
    missing = REQUIRED_FIELDS - set(d.keys())
    return (not missing), f"missing={sorted(missing)}" if missing else "all present"


def gate_nan(d):
    bad = []
    for k in d.keys():
        a = d[k]
        if hasattr(a, "dtype") and a.dtype.kind == "f":
            f = np.atleast_1d(a).astype(float)
            if np.isnan(f).any() or np.isinf(f).any():
                bad.append(k)
    return (not bad), f"nan/inf in {bad}" if bad else "clean"


def gate_v(d, requested_v):
    v = float(d["v_avg"])
    err = abs(v - requested_v)
    return err <= TOL, f"v_avg={v:.4f} requested={requested_v} abs_err={err:.4f} tol={TOL}"


def gate_audit(d):
    ga, gb = bool(d["gate_a"]), bool(d["gate_b"])
    el, er = float(d["energy_drift_left"]), float(d["energy_drift_right"])
    kl, kr = float(d["ke_drop_lr"]), float(d["ke_drop_rl"])
    ok = ga and gb and el < DRIFT_MAX and er < DRIFT_MAX \
        and kl >= KE_DROP_MIN and kr >= KE_DROP_MIN
    return ok, (f"gate_a={ga} gate_b={gb} drift=({el:.3f},{er:.3f}) "
                f"ke_drop=({kl:.4f},{kr:.4f})")


def gate_solver(log_path, n_stages=5):
    """solver terminal status：末级 stage 成功才算 terminal success。"""
    txt = Path(log_path).read_text(errors="replace") if log_path else ""
    stages = re.findall(r"\[solve\] stage (\d+)/(\d+):", txt)
    results = re.findall(r"\[solve\] stage (\d+): success=(True|False)", txt)
    if not stages or not results:
        return False, "no stage records in solve log"
    max_idx = max(int(n) for n, _ in stages)
    last = {int(n): s for n, s in results}
    terminal_ok = last.get(max_idx) == "True" and max_idx == n_stages
    detail = f"reached stage {max_idx}/{n_stages}, terminal success={last.get(max_idx)}"
    return terminal_ok, detail


def validate(npz_path, requested_v, solve_log, expected_knots=None):
    d = dict(np.load(npz_path, allow_pickle=True))
    g1, m1 = gate_load(d)
    g6, m6 = gate_nan(d)
    g7, m7 = gate_v(d, requested_v)
    g345, m345 = gate_audit(d)
    g2, m2 = gate_solver(solve_log)
    knots_ok, m8 = True, "not checked"
    if expected_knots is not None:
        knots_ok = int(d["knots"]) == int(expected_knots)
        m8 = f"knots={int(d['knots'])} expected={expected_knots}"
    gates = {"G1_fields": g1, "G2_solver_terminal": g2, "G345_audit": g345,
             "G6_nan_inf": g6, "G7_v_tolerance": g7, "G8_config_identity": knots_ok}
    details = {"G1_fields": m1, "G2_solver_terminal": m2, "G345_audit": m345,
               "G6_nan_inf": m6, "G7_v_tolerance": m7, "G8_config_identity": m8}
    solver_terminal_status = "solver_success" if g2 else "solver_failed"
    material_status = "mechanically_valid" if all(gates.values()) else "mechanically_invalid"
    return {"npz": str(npz_path), "requested_v": requested_v,
            "v_realized": float(d["v_avg"]), "abs_error": abs(float(d["v_avg"]) - requested_v),
            "gates": gates, "details": details,
            "solver_terminal_status": solver_terminal_status,
            "material_status": material_status,
            "PASS": all(gates.values())}


def cmd_validate(a):
    r = validate(a.npz, a.requested_v, a.solve_log, a.knots)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    sys.exit(0 if r["PASS"] else 1)


def cmd_run(a):
    raise SystemExit("run 子命令在 hot-start source 冻结后启用（§5.8）；"
                     "当前仅 validate 可用——防半成品驱动污染 campaign log")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate", help="对单个 npz 做全门集判定")
    p.add_argument("--npz", required=True)
    p.add_argument("--requested-v", type=float, required=True)
    p.add_argument("--solve-log", default=None, help="该 start 的 solve 日志")
    p.add_argument("--knots", type=int, default=None)
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("run", help="按 manifest 执行 campaign（hot-start source 冻结后启用）")
    p.add_argument("--manifest", required=True)
    p.add_argument("--npz-dir", required=True)
    p.add_argument("--speeds", required=True, help="逗号分隔，如 0.277,0.30")
    p.set_defaults(fn=cmd_run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
