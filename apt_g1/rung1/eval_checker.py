"""TO41 Rung 1 eval 独立审计（read-only audit）。

角色（SCRIPT_MAP 登记）：**read-only audit**。Rung 1 正式评测的独立审计层：
不 import 被测代码（eval_cell / launch_sanity / mode_a_runtime 一概不碰）；
mapping / LUT manifest / 数组哈希 / natural bucketize 均为审计侧独立实现
（与 l_checker 同款纪律，helper 复用仅限审计侧共享代码 d_checker / l_checker）。
verdict 只出自本 checker（协议 §9：runtime 侧 receipt 是 record，不是 PASS）。

审计对象 = eval_cell.py 产出的 rung1-eval-receipt/v1（28 cells × train seeds）。
判据（G1–G10，全部机械化；本 checker 只判 execution/conformance，不做任何
科学统计——contrast/效应曲线属下游分析，与审计分离，三十九轮 §13 双线纪律）：

    G1  coverage        28 cells × train seeds 精确覆盖，无缺无重，schema 正确
    G2  checkpoint 身份  同 (arm, seed) 全部 cells 共用同一 selected ckpt
                        （== selection manifest）；C1→A / C2→B、ON/OFF 各选
                        各的 = FAIL（三十九轮 §5 隔离纪律）
    G3  LUT 消费身份     消费文件 5 字段数组 canonical sha == 冻结 manifest
    G4  Mode A same-τ    每 (v, seed) 四 cell 的 to_ref_npz / τ buffer 身份
                        全同且 == 冻结 LUT（τ(v,C1)=τ(v,C2)=τ_frozen(v) 的
                        env 层机械证明）
    G5  override 持久性  每 decode call applied == mapped（含边界后全部
                        calls）；decode/τ call 数 == 步数语义（ON =
                        steps×decimation，OFF = 0）；buffer pre==post
    G6  cfg 身份         归一化 cfg（去 to_ref_npz/to_tau）全局唯一；同
                        (v,seed) on/off diff == {to_tau}
    G7  target 身份      target_speed ∈ 冻结 grid、cell_id 自洽；
                        v_realized/abs_err 仅作 diagnostic 照录
    G8  边界簿记         episode_start/auto_reset 边界记录自洽
    G9  eval seeds       == 预注册清单 [0,1,2]
    G10 outcome 完整性   每 episode 聚合字段齐全（record 完整，非 verdict）

任何 FAIL → exit 1 + failures 列表；修复路径 = 修 implementation 重跑对应
cell（保险丝 1），禁止改 frozen specification / 材料 / mapping。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from apt_g1.rung1.d_checker import sha256_file
from apt_g1.rung1.l_checker import (
    LUT_ARRAY_FIELDS,
    _array_sha256,
    _scan_blocklist,
    load_lut_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPING_YAML = REPO_ROOT / "apt_g1/configs/rung1_tau_dec_mapping.yaml"
RECEIPT_SCHEMA = "rung1-eval-receipt/v1"
CONDITION_ARMS = ("C1", "C2")
TAU_FF_ARMS = ("on", "off")
POLICY_ARMS = {"on": "t10", "off": "ctrl"}


def _utcnow() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_mapping_independent(path: Path = MAPPING_YAML) -> dict:
    """mapping v2 的审计侧独立解析（不 import mode_a_runtime）。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("artifact_id") != "rung1-tau-dec-mapping" or raw.get("mapping_rule_version") != 2:
        raise SystemExit(f"FAIL: {path} 不是 mapping v2 artifact")
    conds = {cid: (int(spec["speed_bin"]), int(spec["dir_bin"]))
             for cid, spec in raw["decoder_conditions"].items()}
    rows: dict = {}
    grid: set = set()
    for row in raw["mappings"]:
        v = round(float(row["target_speed"]), 3)
        rows[(v, row["condition_arm"])] = row["decoder_condition_id"]
        grid.add(v)
    if len(rows) != 14 or len(grid) != 7:
        raise SystemExit(f"FAIL: mapping rows={len(rows)} grid={len(grid)} != 14/7")
    return {"grid": sorted(grid), "rows": rows, "conds": conds}


def enumerate_cells_independent(mapping: dict) -> list[dict]:
    """28-cell 冻结枚举的审计侧独立重排（cell_id 格式与 runtime 侧契约一致）。"""
    cells = []
    for v in mapping["grid"]:
        for arm in CONDITION_ARMS:
            for ff in TAU_FF_ARMS:
                cells.append({
                    "cell_id": f"v{v * 1000:04.0f}_{arm}_{ff}",
                    "target_speed": v,
                    "condition_arm": arm,
                    "tau_ff": ff,
                })
    return cells


def load_receipts(receipts_dir: Path) -> tuple[dict[str, dict], list[str]]:
    receipts: dict[str, dict] = {}
    errors: list[str] = []
    for p in sorted(receipts_dir.glob("receipt_*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{p.name}: JSON 解析失败 {e}")
            continue
        key = f"{r.get('cell_id')}__s{r.get('train_seed')}"
        if key in receipts:
            errors.append(f"{p.name}: 重复 key {key}")
        receipts[key] = r
    return receipts, errors


class Audit:
    def __init__(self):
        self.failures: list[str] = []
        self.checks: dict[str, dict] = {}
        self.n_receipts = 0

    def check(self, gid: str, ok: bool, ok_msg: str, fail_msgs: list[str]):
        self.checks[gid] = {
            "verdict": "PASS" if ok else "FAIL",
            "note": ok_msg if ok else f"{len(fail_msgs)} failures",
        }
        self.failures.extend(f"[{gid}] {m}" for m in fail_msgs)

    @property
    def overall(self) -> str:
        return "PASS" if not self.failures else "FAIL"


def audit(receipts_dir: Path, materials_roots: list[Path], selection_path: Path,
          train_seeds: list[int], eval_seeds_expected: list[int]) -> Audit:
    a = Audit()
    mapping = load_mapping_independent()
    lut_manifest = load_lut_manifest()
    cells = enumerate_cells_independent(mapping)
    expected_keys = {f"{c['cell_id']}__s{seed}" for c in cells for seed in train_seeds}

    if not receipts_dir.exists():
        a.check("G1", False, "", [f"receipts 目录不存在: {receipts_dir}"])
        return a
    receipts, parse_errors = load_receipts(receipts_dir)
    a.n_receipts = len(receipts)

    # ── G1 coverage ──
    g1_fail = list(parse_errors)
    got_keys = set(receipts)
    for k in sorted(expected_keys - got_keys):
        g1_fail.append(f"缺 receipt: {k}")
    for k in sorted(got_keys - expected_keys):
        g1_fail.append(f"多余 receipt: {k}")
    for k, r in receipts.items():
        if r.get("schema") != RECEIPT_SCHEMA:
            g1_fail.append(f"{k}: schema={r.get('schema')} != {RECEIPT_SCHEMA}")
        if r.get("smoke"):
            g1_fail.append(f"{k}: smoke receipt 混入正式目录")
        block: list[str] = []
        _scan_blocklist(r, k, block)
        g1_fail.extend(block)
    a.check("G1", not g1_fail, f"{len(receipts)}/{len(expected_keys)} receipts",
            g1_fail)
    if g1_fail and not receipts:
        return a

    # selection manifest
    try:
        sel = json.loads(selection_path.read_text(encoding="utf-8"))
        if sel.get("artifact") != "rung1-eval-ckpt-selection/v1":
            raise ValueError("artifact id")
    except Exception as e:  # noqa: BLE001
        a.check("G2", False, "", [f"selection manifest 不可用: {e}"])
        sel = None

    # ── G2 checkpoint identity ──
    g2_fail: list[str] = []
    groups: dict[tuple, list[dict]] = {}
    for k, r in receipts.items():
        groups.setdefault((r.get("policy_arm"), r.get("train_seed")), []).append(r)
    for (arm, seed), rs in sorted(groups.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        shas = {r["checkpoint"]["ckpt_sha256"] for r in rs if "checkpoint" in r}
        if len(shas) != 1:
            g2_fail.append(f"{arm}-s{seed}: 组内出现 {len(shas)} 个不同 ckpt sha"
                           "（C1/C2 或 ON/OFF 各选各的 = selection 污染）")
        if sel is not None:
            expect = sel["runs"].get(arm, {}).get(str(seed), {}).get("ckpt_sha256")
            got = next(iter(shas)) if len(shas) == 1 else None
            if got != expect:
                g2_fail.append(f"{arm}-s{seed}: ckpt sha != selection manifest"
                               f"（got {str(got)[:16]} expect {str(expect)[:16]}）")
    a.check("G2", not g2_fail,
            f"{len(groups)} (arm,seed) 组全部单 ckpt 且 == manifest", g2_fail)

    # ── G3 LUT identity ──
    g3_fail: list[str] = []
    for k, r in receipts.items():
        v = round(float(r["target_speed"]), 3)
        entry = lut_manifest["entries"].get(v)
        if entry is None:
            g3_fail.append(f"{k}: manifest 无 v={v}")
            continue
        lut_path = Path(r["tau_material"]["derived_lut"]["file"])
        if not lut_path.exists():
            g3_fail.append(f"{k}: LUT 文件不存在 {lut_path}")
            continue
        with np.load(lut_path) as z:
            for field in LUT_ARRAY_FIELDS:
                got = _array_sha256(np.asarray(z[field]))
                expect = entry["lut_array_sha256"][field]
                if got != expect:
                    g3_fail.append(f"{k}: LUT {field} sha != 冻结 manifest")
        if r["tau_material"]["derived_lut"].get("array_sha256") != entry["lut_array_sha256"]:
            g3_fail.append(f"{k}: receipt 记录的 array_sha256 != manifest")
    a.check("G3", not g3_fail, "全部消费 LUT 数组身份 == 冻结 manifest", g3_fail)

    # ── G4 Mode A same-τ ──
    g4_fail: list[str] = []
    by_v_seed: dict[tuple, dict[str, dict]] = {}
    for k, r in receipts.items():
        by_v_seed.setdefault((round(float(r["target_speed"]), 3), r.get("train_seed")), {})[k] = r
    for (v, seed), rs in sorted(by_v_seed.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        if len(rs) != 4:
            g4_fail.append(f"v={v} s{seed}: cell 数 {len(rs)} != 4")
            continue
        npzs = {r["cfg_snapshot"]["to_ref_npz"] for r in rs.values()}
        bufs = {r["tau_material"]["buffer_sha256_pre"] for r in rs.values()}
        arts = {r["tau_material"]["frozen_material"]["artifact"] for r in rs.values()}
        if len(npzs) != 1 or len(bufs) != 1 or len(arts) != 1:
            g4_fail.append(f"v={v} s{seed}: 四 cell 消费身份不唯一"
                           f"（npz {len(npzs)} / buffer {len(bufs)} / artifact {len(arts)}）")
        entry = lut_manifest["entries"].get(v)
        if entry and len(bufs) == 1:
            tau_expect = entry["lut_array_sha256"]["tau_ref6"]
            buf = next(iter(bufs))
            if buf != tau_expect:
                g4_fail.append(f"v={v} s{seed}: τ buffer sha != 冻结 LUT tau_ref6"
                               f"（got {buf[:16]}… expect {tau_expect[:16]}…）")
    a.check("G4", not g4_fail, "每 (v,seed) 四 cell τ 消费身份唯一且 == 冻结 LUT",
            g4_fail)

    # ── G5 override persistence + call-count semantics ──
    g5_fail: list[str] = []
    for k, r in receipts.items():
        mapped_vb = r["assignment"]["speed_bin"]
        mapped_db = r["assignment"]["dir_bin"]
        n_decode = r["condition_override"]["n_decode_calls"]
        steps_total = r["execution"]["steps_done_total"]
        per_call = r["condition_override"].get("per_call", [])
        if len(per_call) != n_decode:
            g5_fail.append(f"{k}: per_call 记录数 {len(per_call)} != n_decode_calls "
                           f"{n_decode}——probe 记录不完整")
        for c in per_call:
            if c["applied_vb"] != mapped_vb or c["applied_db"] != mapped_db:
                g5_fail.append(
                    f"{k}: call#{c['i']} applied=({c['applied_vb']},{c['applied_db']}) "
                    f"!= mapped=({mapped_vb},{mapped_db})——override 被覆写")
        if n_decode != steps_total:
            g5_fail.append(f"{k}: n_decode_calls={n_decode} != steps_done_total={steps_total}")
        n_tau = r["tau_consumption"]["n_tau_calls"]
        if r["tau_ff"] == "on":
            expect_tau = r["execution"]["expected_tau_calls_on"]
            if n_tau != expect_tau:
                g5_fail.append(f"{k}: ON n_tau_calls={n_tau} != steps×decimation={expect_tau}")
            if r["tau_consumption"]["n_nonfinite_tau_calls"] != 0:
                g5_fail.append(f"{k}: ON 臂出现非有限 τ")
        else:
            if n_tau != 0:
                g5_fail.append(f"{k}: OFF 臂 n_tau_calls={n_tau} != 0（τ 泄漏）")
        if r["tau_material"]["buffer_sha256_pre"] != r["tau_material"]["buffer_sha256_post"]:
            g5_fail.append(f"{k}: τ buffer pre != post（episode 期间 buffer 被改动）")
    a.check("G5", not g5_fail, "override 全 call 生效 + call 数语义 + buffer 恒定",
            g5_fail)

    # ── G6 cfg identity ──
    g6_fail: list[str] = []
    norm_ref = None
    for k, r in receipts.items():
        cfg = dict(r["cfg_snapshot"])
        cfg.pop("to_ref_npz", None)
        cfg.pop("to_tau", None)
        if norm_ref is None:
            norm_ref = (k, cfg)
        elif cfg != norm_ref[1]:
            diff = {kk for kk in set(cfg) | set(norm_ref[1])
                    if cfg.get(kk) != norm_ref[1].get(kk)}
            g6_fail.append(f"{k}: 归一化 cfg != {norm_ref[0]}（diff={sorted(diff)}）"
                           "——未授权 config 变更")
    for (v, seed), rs in by_v_seed.items():
        on = [r for r in rs.values() if r["tau_ff"] == "on"]
        off = [r for r in rs.values() if r["tau_ff"] == "off"]
        if on and off:
            d1 = set(on[0]["cfg_snapshot"]) ^ set(off[0]["cfg_snapshot"])
            d2 = {kk for kk in set(on[0]["cfg_snapshot"]) & set(off[0]["cfg_snapshot"])
                  if on[0]["cfg_snapshot"][kk] != off[0]["cfg_snapshot"][kk]}
            if (d1 or d2) != {"to_tau"}:
                g6_fail.append(f"v={v} s{seed}: on/off cfg diff != {{to_tau}}"
                               f"（got {sorted(d1 | d2)}）")
    a.check("G6", not g6_fail, "归一化 cfg 全局唯一；on/off diff == {to_tau}", g6_fail)

    # ── G7 target identity ──
    g7_fail: list[str] = []
    for k, r in receipts.items():
        v = round(float(r["target_speed"]), 3)
        if v not in mapping["grid"]:
            g7_fail.append(f"{k}: target_speed {v} 不在冻结 grid")
        if not k.startswith(f"v{v * 1000:04.0f}_"):
            g7_fail.append(f"{k}: cell_id 前缀与 target_speed 不自洽")
        fm = r["tau_material"]["frozen_material"]
        if "v_realized" not in fm or "abs_err" not in fm:
            g7_fail.append(f"{k}: material diagnostic 字段缺失（v_realized/abs_err）")
        expected_cond = mapping["rows"].get((v, r["condition_arm"]))
        if expected_cond != r["assignment"]["decoder_condition_id"]:
            g7_fail.append(f"{k}: condition != mapping v2 lookup"
                           f"（got {r['assignment']['decoder_condition_id']} "
                           f"expect {expected_cond}）")
        exp_vb, exp_db = mapping["conds"][expected_cond] if expected_cond else (None, None)
        if (r["assignment"]["speed_bin"], r["assignment"]["dir_bin"]) != (exp_vb, exp_db):
            g7_fail.append(f"{k}: bins != mapping v2")
    a.check("G7", not g7_fail, "target/grid/cell_id/assignment 自洽；diagnostic 照录",
            g7_fail)

    # ── G8 boundary bookkeeping ──
    g8_fail: list[str] = []
    for k, r in receipts.items():
        total_done = 0
        for e in r["episodes"]:
            b = e["boundaries"]
            if not b or b[0]["type"] != "episode_start":
                g8_fail.append(f"{k} seed{e['eval_seed']}: 首 boundary 非 episode_start")
            if e["reset_count"] != 1 + e["n_auto_resets"]:
                g8_fail.append(f"{k} seed{e['eval_seed']}: reset_count 簿记不自洽")
            total_done += e["steps_done"]
        if total_done != r["execution"]["steps_done_total"]:
            g8_fail.append(f"{k}: Σ episodes steps_done != steps_done_total")
    a.check("G8", not g8_fail, "边界/reset 簿记自洽", g8_fail)

    # ── G9 eval seeds ──
    g9_fail: list[str] = []
    for k, r in receipts.items():
        if list(r.get("eval_seeds", [])) != eval_seeds_expected:
            g9_fail.append(f"{k}: eval_seeds {r.get('eval_seeds')} != {eval_seeds_expected}")
    a.check("G9", not g9_fail, f"eval seeds == {eval_seeds_expected}（预注册清单）", g9_fail)

    # ── G10 outcome completeness ──
    g10_fail: list[str] = []
    need = ("steps_done", "completed", "fall_step", "h_min", "vx_mean",
            "v_speed_mean", "disp", "reset_count")
    for k, r in receipts.items():
        if len(r["episodes"]) != len(eval_seeds_expected):
            g10_fail.append(f"{k}: episodes 数 {len(r['episodes'])} != {len(eval_seeds_expected)}")
        for e in r["episodes"]:
            for f in need:
                if f not in e:
                    g10_fail.append(f"{k} seed{e.get('eval_seed')}: 缺 outcome 字段 {f}")
    a.check("G10", not g10_fail, "outcome 聚合字段完整（record 层）", g10_fail)

    return a


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rung 1 eval receipts 独立审计（read-only；verdict 唯一出口）")
    ap.add_argument("--receipts-dir", type=Path, required=True)
    ap.add_argument("--materials-root", type=Path, nargs="+",
                    default=[REPO_ROOT / "apt_g1/outputs"])
    ap.add_argument("--selection-manifest", type=Path,
                    default=REPO_ROOT / "apt_g1/outputs/sync/to41_eval/ckpt_selection.json")
    ap.add_argument("--train-seeds", default="0,1")
    ap.add_argument("--eval-seeds", default="0,1,2")
    ap.add_argument("--env-tag", choices=["lab-ts", "local"], required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="审计 JSON 输出路径（默认 receipts_dir 同级 eval_audit.json）")
    args = ap.parse_args()

    train_seeds = [int(s) for s in args.train_seeds.split(",")]
    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]
    a = audit(args.receipts_dir, list(args.materials_root), args.selection_manifest,
              train_seeds, eval_seeds)

    report = {
        "artifact": "rung1-eval-audit/v1",
        "generated_utc": _utcnow(),
        "env_tag": args.env_tag,
        "receipts_dir": str(args.receipts_dir),
        "n_receipts": a.n_receipts,
        "checks": a.checks,
        "failures": a.failures,
        "overall": a.overall,
    }
    out_path = args.out or (args.receipts_dir.parent / "eval_audit.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    for gid, c in a.checks.items():
        print(f"{gid:4s} {c['verdict']:4s} {c['note']}")
    print(f"overall: {a.overall} -> {out_path}")
    return 0 if a.overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
