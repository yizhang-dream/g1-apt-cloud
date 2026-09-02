"""TO41 Rung 1 launch sanity independent checker（read-only audit）。

角色（SCRIPT_MAP 登记）：**read-only audit**。L1–L4 的 verdict 唯一来源：
不 import launch_sanity / env_wiring（被测 execution code），只读 28 份
receipt（record）+ 冻结工件（mapping v2 / availability map / source
registry / 材料 npz），自行重算：

- L1 τ material consumption：buffer 哈希（env._to_tau 快照）= checker 独立
  np.load(tau_ref6)→float32 重算；4 cell/v Mode A fingerprint；ON 臂消费
  计数、OFF 臂零注入；
- L2 override persistence：逐 decode call 的 natural/applied 双记录 vs
  checker 重算的自然 bucketize；boundary 前后调用计数；
- L3 ON/OFF isolation：两臂 cfg 快照 diff == {to_tau}（唯一预注册干预）；
- L4 28-cell receipt：冻结枚举全覆盖 + 冻结 env 源哈希 = mapping
  preprocessing_hash（wiring 零 env 文件改动的机械证明）+ decoder 身份
  （签名表/超参自洽/前后全等）。

与 d_checker 同款纪律（协议 §9）：runtime 侧 receipt 任何自报 verdict
字段 = schema FAIL；本 checker 恒 read-only，发现问题的唯一正当出口 =
report FAIL → 保险丝。判据唯一事实源 = refine-logs/TO41_LAUNCH_SANITY.md。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import yaml

from apt_g1.rung1.d_checker import (
    BLOCKLIST_VERDICT_KEYS,
    EXPECTED_SD_SIGNATURE,
    enumerate_cells,
    load_availability,
    load_mapping,
    load_registry,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_SCHEMA = "rung1-launch-sanity-receipt/v2"
LUT_MANIFEST_PATH = REPO_ROOT / "apt_g1/outputs/sync/to41_sanity/luts/lut_manifest.json"
LUT_ARRAY_FIELDS = ("q_ref6", "tau_ref6", "pitch", "z", "heel_rel")

RECEIPT_TOP_KEYS = {
    "schema", "cell_id", "cell_index", "target_speed", "condition_arm",
    "tau_ff", "assignment", "env_identity", "cfg_snapshot", "tau_material",
    "condition_override", "tau_consumption", "decoder_identity", "execution",
}


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _scan_blocklist(node, path: str, violations: list[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if str(k).lower() in BLOCKLIST_VERDICT_KEYS:
                violations.append(f"{path}.{k}: 自报 verdict 类字段（协议 §9 禁止）")
            _scan_blocklist(v, f"{path}.{k}", violations)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scan_blocklist(v, f"{path}[{i}]", violations)


def _array_sha256(a) -> str:
    """冻结数组身份的 checker 侧独立实现（与 env_wiring 同一规范化 spec：
    f"shape=...;dtype=...;data=" + float32 C-contiguous 小端字节）。"""
    arr = np.ascontiguousarray(a, dtype=np.float32)
    h = hashlib.sha256()
    h.update(f"shape={arr.shape};dtype={arr.dtype.str};data=".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _natural_bin(v: float, vx_max: float = 0.8, n: int = 3) -> int:
    """冻结 bucketize（right=False）的 checker 侧重算（numpy 等价）。"""
    edges = np.linspace(0.0, vx_max, n + 1)[1:-1]
    return int(np.searchsorted(edges, v, side="left"))


def find_material(artifact: str, roots: list[Path]) -> Path | None:
    for root in roots:
        p = root / artifact
        if p.exists():
            return p
    return None


def load_lut_manifest(path: Path = LUT_MANIFEST_PATH) -> dict:
    """L0 manifest 的 checker 侧独立解析（不 import launch_sanity）。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("artifact_id") != "rung1-material-lut-manifest" or raw.get("schema_version") != 1:
        raise SystemExit(f"FAIL: {path} 不是 rung1-material-lut-manifest/v1")
    entries = {round(float(e["target_speed"]), 3): e for e in raw["entries"]}
    return {"manifest_path": path, "manifest_sha256": sha256_file(path), "entries": entries}


def validate_receipt(r: dict) -> list[str]:
    bad: list[str] = []
    if r.get("schema") != RECEIPT_SCHEMA:
        bad.append(f"schema != {RECEIPT_SCHEMA}")
    keys = set(r.keys())
    if keys != RECEIPT_TOP_KEYS:
        bad.append(f"顶层字段集 ≠ 封闭 schema：多 {sorted(keys - RECEIPT_TOP_KEYS)} "
                   f"缺 {sorted(RECEIPT_TOP_KEYS - keys)}")
    bl: list[str] = []
    _scan_blocklist(r, "receipt", bl)
    bad.extend(bl)
    return bad


# ---------------------------------------------------------------------------
# L1 — τ material consumption（Mode A env-level fingerprint）
# ---------------------------------------------------------------------------

def check_l1(receipts: dict[str, dict], mapping: dict, materials_roots: list[Path],
             lut_manifest: dict):
    """τ material consumption（双层身份链 + Mode A env-level fingerprint）。

    身份链：冻结材料 npz（canonical，D 链锚定）--manifest source_sha256-->
    derived LUT（env cfg.to_ref_npz 消费形式）--> env._to_tau buffer。
    checker 独立重算：材料文件 sha / LUT 文件 sha / LUT 5 字段数组级规范
    哈希 / buffer（= LUT tau_ref6 float32 规范哈希）。
    """
    failures: list[str] = []
    per_v: list[dict] = []
    for v in mapping["grid"]:
        v3 = round(float(v), 3)
        cells = [c for c in enumerate_cells(mapping) if c["target_speed"] == v]
        bad: list[str] = []
        row = {"target_speed": v, "cells": {}, "verdict": "PASS"}
        hashes_frozen: set[str] = set()
        hashes_lut: set[str] = set()
        hashes_buf: set[str] = set()
        lineages: set[str] = set()
        lut_entry = lut_manifest["entries"].get(v3)
        if lut_entry is None:
            bad.append(f"LUT manifest 缺 v={v}")
        lut_path = (lut_manifest["manifest_path"].parent / lut_entry["lut_file"]) \
            if lut_entry else None
        for cell in cells:
            cid = cell["cell_id"]
            r = receipts.get(cid)
            if r is None:
                bad.append(f"{cid}: receipt 缺失")
                row["cells"][cid] = "FAIL"
                continue
            tm = r["tau_material"]
            tc = r["tau_consumption"]
            fm = tm["frozen_material"]
            dl = tm["derived_lut"]
            cell_bad: list[str] = []
            # (a) 冻结材料层：buffer episode 前后全等（缓冲不被换）
            if tm["buffer_sha256_pre"] != tm["buffer_sha256_post"]:
                cell_bad.append("τ buffer 哈希 episode 前后不一致")
            hashes_frozen.add(fm["file_sha256"])
            hashes_lut.add(dl["file_sha256"])
            hashes_buf.add(tm["buffer_sha256_pre"])
            lineages.add(fm["source_lineage"])
            # (b) 冻结材料文件独立重算
            mp = find_material(fm["artifact"], materials_roots)
            if mp is None:
                cell_bad.append(f"material 不可达: {fm['artifact']}")
            else:
                if sha256_file(mp) != fm["file_sha256"]:
                    cell_bad.append("冻结材料文件 sha256 != receipt 记录")
            # (c) LUT 层：文件 sha + 5 字段数组级哈希独立重算 + source 链闭合
            if lut_path is None or not lut_path.exists():
                cell_bad.append("LUT 不可达")
            else:
                if sha256_file(lut_path) != dl["file_sha256"]:
                    cell_bad.append("LUT 文件 sha256 != receipt 记录")
                with np.load(lut_path) as z:
                    recomputed = {k: _array_sha256(z[k]) for k in LUT_ARRAY_FIELDS
                                  if k in z.files}
                for k in LUT_ARRAY_FIELDS:
                    if recomputed.get(k) != dl["array_sha256"].get(k):
                        cell_bad.append(f"LUT 数组身份 {k} != checker 独立重算")
                with np.load(lut_path) as z:
                    buf_expect = _array_sha256(z["tau_ref6"])
                if tm["buffer_sha256_pre"] != buf_expect:
                    cell_bad.append("buffer 哈希 != checker 独立重算 LUT tau_ref6")
                # source 链闭合：LUT 必须声称源自该冻结材料，且 manifest 同
                if dl.get("source_sha256_manifest") != fm["file_sha256"]:
                    cell_bad.append("derived_lut.source_sha256 != 冻结材料 sha（链断裂）")
                if lut_entry and lut_entry.get("source_sha256") != fm["file_sha256"]:
                    cell_bad.append("manifest source_sha256 != 冻结材料 sha（链断裂）")
                if lut_entry and lut_entry.get("lut_file_sha256") != dl["file_sha256"]:
                    cell_bad.append("manifest lut_file_sha256 != receipt 记录")
            if lut_entry and tm.get("lut_manifest_sha256") != lut_manifest["manifest_sha256"]:
                cell_bad.append("receipt 的 lut_manifest_sha256 != checker 读到的 manifest")
            if not fm.get("source_lineage"):
                cell_bad.append("tau_source_lineage 缺失（协议加严：hash 相等也不足）")
            # (d) 消费语义：ON 臂有消费且无 NaN；OFF 臂零注入
            if cell["tau_ff"] == "on":
                if tc["n_tau_calls"] < 1:
                    cell_bad.append("τ ON 臂 n_tau_calls == 0（冻结消费点未执行）")
                if tc.get("n_nonfinite_tau_calls", 0) != 0:
                    cell_bad.append("消费序列出现非有限 τ")
            else:
                if tc["n_tau_calls"] != 0:
                    cell_bad.append("τ OFF 臂发生注入（n_tau_calls != 0）")
            row["cells"][cid] = "FAIL" if cell_bad else "PASS"
            for b in cell_bad:
                failures.append(f"{cid}: {b}")
        for hashes, label in [(hashes_frozen, "冻结材料"), (hashes_lut, "LUT"),
                              (hashes_buf, "buffer")]:
            if len(hashes) > 1:
                bad.append(f"Mode A fingerprint 破坏：4 cell {label}哈希不唯一 "
                           f"({len(hashes)} 种)")
        if len(lineages) > 1:
            bad.append("lineage 不一致（同 v 不同冻结 artifact 引用）")
        row["frozen_sha256_16"] = sorted(h[:16] for h in hashes_frozen)
        row["buffer_sha256_16"] = sorted(h[:16] for h in hashes_buf)
        row["verdict"] = "FAIL" if (bad or any(
            s == "FAIL" for s in row["cells"].values())) else "PASS"
        for b in bad:
            failures.append(f"v={v}: {b}")
        per_v.append(row)
    return ("FAIL" if failures else "PASS"), failures, per_v


# ---------------------------------------------------------------------------
# L2 — override persistence（per-call natural vs applied + boundary）
# ---------------------------------------------------------------------------

def check_l2(receipts: dict[str, dict], mapping: dict):
    failures: list[str] = []
    per_cell: list[dict] = []
    for cell in enumerate_cells(mapping):
        cid = cell["cell_id"]
        r = receipts.get(cid)
        if r is None:
            failures.append(f"{cid}: receipt 缺失（L2 覆盖洞）")
            per_cell.append({"cell": cid, "verdict": "FAIL",
                             "failures": ["receipt 缺失"]})
            continue
        bad: list[str] = []
        co = r["condition_override"]
        ex = r["execution"]
        asg = r["assignment"]
        cfgs = r["cfg_snapshot"]
        n = co["n_decode_calls"]
        if n != ex["steps_done"]:
            bad.append(f"decode 调用数 {n} != steps_done {ex['steps_done']}")
        if ex["steps_done"] != ex["steps_requested"]:
            bad.append("steps_done != steps_requested（执行不完整）")
        if n < 1:
            bad.append("decode 调用数为 0（冻结 decode 路径未被 exercise）")
        # per-call：applied == mapped；natural_db == dir_bin；natural 分布 == 重算
        exp_natural = _natural_bin(r["target_speed"], cfgs["vx_max"],
                                   cfgs["latent_vae_n_bins"])
        if exp_natural != asg["natural_speed_bin"]:
            bad.append("checker 重算自然 bin != receipt 记录 natural_speed_bin")
        dist: dict[int, int] = {}
        changed = 0
        for rec in co["per_call"]:
            if rec["applied_vb"] != asg["speed_bin"]:
                bad.append(f"call#{rec['i']} applied_vb {rec['applied_vb']} != mapped {asg['speed_bin']}")
            if rec["applied_db"] != asg["dir_bin"]:
                bad.append(f"call#{rec['i']} applied_db != mapped dir_bin")
            if rec["natural_db"] != asg["dir_bin"]:
                bad.append(f"call#{rec['i']} natural_db != {asg['dir_bin']}（cmd 应恒定 forward）")
            dist[rec["natural_vb"]] = dist.get(rec["natural_vb"], 0) + 1
            if rec["natural_vb"] != rec["applied_vb"]:
                changed += 1
        if dist != {exp_natural: n}:
            bad.append(f"natural_vb 分布 {dist} != 重算 {{{exp_natural}: {n}}}"
                       "（cmd_vx 恒定被破坏或自然 assignment 漂移）")
        if changed != co["n_override_changed"]:
            bad.append("n_override_changed 与 per-call 记录不符")
        exp_changed = n if exp_natural != asg["speed_bin"] else 0
        if changed != exp_changed:
            bad.append(f"override 生效计数 {changed} != 期望 {exp_changed}")
        # boundary persistence：强制边界后仍有 decode 调用
        forced = [b for b in ex["boundaries"] if b["type"] == "forced"]
        if not forced:
            bad.append("无强制 episode boundary（L2 设计要求 ≥1）")
        for b in forced:
            if b.get("cond_calls_after", 0) < 1:
                bad.append(f"step{b['step']} 强制边界后无 decode 调用（persistence 不可证）")
        for b in bad:
            failures.append(f"{cid}: {b}")
        per_cell.append({"cell": cid,
                         "natural_bin": exp_natural,
                         "mapped_bin": asg["speed_bin"],
                         "n_calls": n,
                         "n_override_changed": changed,
                         "n_boundaries": len(ex["boundaries"]),
                         "verdict": "FAIL" if bad else "PASS",
                         "failures": bad})
    return ("FAIL" if failures else "PASS"), failures, per_cell


# ---------------------------------------------------------------------------
# L3 — ON/OFF isolation（同 (v, arm) 两臂唯一 cfg 差 = to_tau）
# ---------------------------------------------------------------------------

def check_l3(receipts: dict[str, dict], mapping: dict):
    failures: list[str] = []
    per_pair: list[dict] = []
    for v in mapping["grid"]:
        for arm in ("C1", "C2"):
            off = receipts.get(f"v{v * 1000:04.0f}_{arm}_off")
            on = receipts.get(f"v{v * 1000:04.0f}_{arm}_on")
            cid = f"v{v * 1000:04.0f}_{arm}"
            if off is None or on is None:
                failures.append(f"{cid}: 两臂 receipt 不齐")
                per_pair.append({"pair": cid, "verdict": "FAIL"})
                continue
            bad: list[str] = []
            diff = {}
            for k in set(off["cfg_snapshot"]) | set(on["cfg_snapshot"]):
                if off["cfg_snapshot"].get(k) != on["cfg_snapshot"].get(k):
                    diff[k] = (off["cfg_snapshot"].get(k), on["cfg_snapshot"].get(k))
            if diff != {"to_tau": (False, True)}:
                bad.append(f"两臂 cfg diff {diff} != {{'to_tau': (False, True)}}")
            for a, b_, label in [
                (off["decoder_identity"], on["decoder_identity"], "decoder_identity"),
                (off["tau_material"]["frozen_material"]["file_sha256"],
                 on["tau_material"]["frozen_material"]["file_sha256"], "frozen material sha"),
                (off["tau_material"]["derived_lut"]["file_sha256"],
                 on["tau_material"]["derived_lut"]["file_sha256"], "derived LUT sha"),
                (off["tau_material"]["derived_lut"]["array_sha256"],
                 on["tau_material"]["derived_lut"]["array_sha256"], "derived LUT array sha"),
                (off["tau_material"]["buffer_sha256_pre"],
                 on["tau_material"]["buffer_sha256_pre"], "buffer sha"),
                (off["assignment"]["decoder_condition_id"],
                 on["assignment"]["decoder_condition_id"], "condition"),
                (off["execution"]["steps_requested"], on["execution"]["steps_requested"], "steps"),
                (off["execution"]["boundary_step"], on["execution"]["boundary_step"], "boundary_step"),
                (off["execution"]["python_version"], on["execution"]["python_version"], "python"),
                (off["execution"]["torch_version"], on["execution"]["torch_version"], "torch"),
                (off["condition_override"]["mechanism"], on["condition_override"]["mechanism"], "wiring mechanism"),
            ]:
                if a != b_:
                    bad.append(f"{label} 两臂不一致")
            for b in bad:
                failures.append(f"{cid}: {b}")
            per_pair.append({"pair": cid, "cfg_diff_keys": sorted(diff.keys()),
                             "verdict": "FAIL" if bad else "PASS", "failures": bad})
    return ("FAIL" if failures else "PASS"), failures, per_pair


# ---------------------------------------------------------------------------
# L4 — 28-cell receipt 覆盖 + env/decoder 身份
# ---------------------------------------------------------------------------

def check_l4(receipts: dict[str, dict], mapping: dict):
    failures: list[str] = []
    per_cell: list[dict] = []
    cells = enumerate_cells(mapping)
    if len(receipts) != 28:
        failures.append(f"receipt 数 {len(receipts)} != 28")
    if set(receipts) != {c["cell_id"] for c in cells}:
        failures.append("receipt cell_id 集合 != 冻结枚举")
    for i, cell in enumerate(cells):
        cid = cell["cell_id"]
        r = receipts.get(cid)
        if r is None:
            continue
        bad: list[str] = []
        if r["cell_index"] != i:
            bad.append(f"cell_index {r['cell_index']} != 冻结枚举位 {i}")
        if r["target_speed"] != cell["target_speed"] or r["tau_ff"] != cell["tau_ff"] \
                or r["condition_arm"] != cell["condition_arm"]:
            bad.append("cell 身份字段与冻结枚举不一致")
        if r["execution"]["status"] != "completed":
            bad.append("execution.status != completed")
        # 冻结 env 源未被接线改动（本 gate 的机械核心）
        ei = r["env_identity"]
        if ei["env_source_file_sha256"] != mapping["hashes"]["preprocessing"]:
            bad.append("env 源文件 hash != mapping preprocessing_hash（冻结 env 被改动）")
        if ei["arch_source_file_sha256"] != mapping["hashes"]["decoder_architecture"]:
            bad.append("arch 源文件 hash != mapping decoder_architecture_hash")
        # decoder 身份（D1 同款技术：文件哈希 + 签名表 + 前后全等）
        di = r["decoder_identity"]
        if di["checkpoint_sha256"] != mapping["hashes"]["decoder_checkpoint"]:
            bad.append("checkpoint sha256 != mapping 冻结值")
        if di["state_dict_sha256_before_episode"] != di["state_dict_sha256_after_episode"]:
            bad.append("state_dict episode 前后不一致（decoder 被改动）")
        shapes = di.get("state_dict_key_shapes", {})
        for k, exp in EXPECTED_SD_SIGNATURE.items():
            got = shapes.get(k)
            if got is None:
                bad.append(f"state_dict 缺 {k}")
            elif tuple(got) != exp:
                bad.append(f"state_dict {k} shape {got} != 期望 {list(exp)}")
        arch = di["architecture"]
        for k, v in {"class": "DirSpeedPhaseTokenVAE", "token_dim": 64, "window": 10,
                     "latent_dim": 16, "n_vbins": 3, "n_dbins": 8}.items():
            if arch.get(k) != v:
                bad.append(f"architecture.{k}={arch.get(k)} != {v}")
        if di["mode_layout"]["decode_call_form"] != "decode(z, phase_sc(sin,cos), v_bin, d_bin)":
            bad.append("decode 调用形 != 冻结 4 参形式")
        # owner L4 receipt 必备字段非空
        if not r["tau_material"]["frozen_material"]["source_lineage"]:
            bad.append("tau_source_lineage 空")
        if not r["tau_material"]["derived_lut"].get("source_sha256_manifest"):
            bad.append("derived_lut.source_sha256 空（LUT→冻结材料链缺失）")
        if not r["tau_consumption"].get("consumer_identity"):
            bad.append("consumer_identity 空")
        for b in bad:
            failures.append(f"{cid}: {b}")
        per_cell.append({"cell": cid, "verdict": "FAIL" if bad else "PASS",
                         "failures": bad})
    return ("FAIL" if failures else "PASS"), failures, per_cell


def material_baseline(mapping: dict, materials_roots: list[Path]) -> list[dict]:
    availability = load_availability()
    rows = []
    for v, mat in sorted(availability.items()):
        p = find_material(mat["artifact"], materials_roots)
        sha = sha256_file(p) if p else None
        rows.append({
            "target_speed": v,
            "artifact": mat["artifact"],
            "v_realized": mat["v_realized"],
            "abs_err": mat["abs_err"],
            "checker_recomputed_sha256_16": sha[:16] if sha else None,
            "note": ("accepted under the pre-registered ±0.02 m/s realization "
                     "tolerance（禁触发重解）") if v in (0.3, 0.325) else "",
        })
    return rows


def build_report(receipts_dir: Path, materials_roots: list[Path], env_tag: str,
                 lut_manifest: dict | None = None) -> dict:
    receipts: dict[str, dict] = {}
    schema_failures: list[str] = []
    found = sorted(receipts_dir.glob("receipt_*.json"))
    for p in found:
        r = json.loads(p.read_text(encoding="utf-8"))
        receipts[r.get("cell_id", p.stem)] = r
        schema_failures.extend(f"{p.name}: {b}" for b in validate_receipt(r))
    if len(found) != 28:
        schema_failures.append(f"receipt 文件数 {len(found)} != 28")

    mapping = load_mapping()
    report = {
        "artifact": "rung1-launch-sanity-report/v1",
        "environment_tag": env_tag,
        "generated_utc": _utcnow(),
        "inputs": {
            "receipts_dir": str(receipts_dir),
            "receipts_found": len(found),
            "materials_roots": [str(p) for p in materials_roots],
            "lut_manifest": str(lut_manifest["manifest_path"]),
            "lut_manifest_sha256": lut_manifest["manifest_sha256"],
        },
        "schema_check": {"verdict": "FAIL" if schema_failures else "PASS",
                         "failures": schema_failures},
        "discipline_notes": [
            "L gate 只回答接线（wiring）问题：τ 材料消费 / condition override "
            "persistence / 两臂隔离 / 28-cell receipt 来自真实 env 执行路径；"
            "不含任何 reward / walking quality / stability 字段（三十七轮裁定）。",
            "verdict 全部由本 checker 独立重算；runtime receipt 仅 record。",
            "L PASS 的解释上限 = treatment specification 在真实 env 执行路径上"
            "可忠实实现；不构成任何 Rung 1 科学结论。",
        ],
    }
    l1v, l1f, l1t = check_l1(receipts, mapping, materials_roots, lut_manifest)
    l2v, l2f, l2t = check_l2(receipts, mapping)
    l3v, l3f, l3t = check_l3(receipts, mapping)
    l4v, l4f, l4t = check_l4(receipts, mapping)
    report["L1"] = {"verdict": l1v, "failures": l1f, "per_v": l1t}
    report["L2"] = {"verdict": l2v, "failures": l2f, "per_cell": l2t}
    report["L3"] = {"verdict": l3v, "failures": l3f, "per_pair": l3t}
    report["L4"] = {"verdict": l4v, "failures": l4f, "per_cell": l4t}
    report["material_baseline"] = material_baseline(mapping, materials_roots)
    overall = "PASS" if all(
        report[k]["verdict"] == "PASS" for k in ("schema_check", "L1", "L2", "L3", "L4")
    ) else "FAIL"
    report["verdict"] = {
        "schema_check": report["schema_check"]["verdict"],
        "L1": l1v, "L2": l2v, "L3": l3v, "L4": l4v, "overall": overall,
    }
    return report


def report_markdown(rep: dict) -> str:
    lines = ["# TO41 Rung 1 launch sanity report（machine-generated）", ""]
    lines.append(f"- environment_tag: `{rep['environment_tag']}`  ")
    lines.append(f"- generated_utc: `{rep['generated_utc']}`  ")
    lines.append(f"- receipts: `{rep['inputs']['receipts_found']}/28`  ")
    lines.append("")
    lines.append("## verdict")
    lines.append("| check | verdict |")
    lines.append("|---|---|")
    for k in ("schema_check", "L1", "L2", "L3", "L4"):
        lines.append(f"| {k} | {rep[k]['verdict']} |")
    lines.append(f"| **overall** | **{rep['verdict']['overall']}** |")
    lines.append("")
    if rep["L2"]["per_cell"]:
        lines.append("## L2 override persistence（per cell）")
        lines.append("| cell | natural | mapped | calls | changed | boundaries | verdict |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in rep["L2"]["per_cell"]:
            lines.append(f"| {row['cell']} | vb{row['natural_bin']} | vb{row['mapped_bin']} "
                         f"| {row['n_calls']} | {row['n_override_changed']} "
                         f"| {row['n_boundaries']} | {row['verdict']} |")
        lines.append("")
    if rep["L1"]["per_v"]:
        lines.append("## L1 Mode A env-level fingerprint（per v）")
        lines.append("| target_speed | buffer sha(16) | verdict |")
        lines.append("|---:|---|---|")
        for row in rep["L1"]["per_v"]:
            lines.append(f"| {row['target_speed']} | {', '.join(row['buffer_sha256_16'])} "
                         f"| {row['verdict']} |")
        lines.append("")
    lines.append("## material baseline（checker 独立重算）")
    lines.append("| target | artifact | v_realized | abs_err | sha(16) | note |")
    lines.append("|---:|---|---:|---:|---|---|")
    for row in rep["material_baseline"]:
        lines.append(f"| {row['target_speed']} | {row['artifact']} | {row['v_realized']} "
                     f"| {row['abs_err']} | {row['checker_recomputed_sha256_16']} "
                     f"| {row['note']} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="L1–L4 independent checker（read-only）")
    ap.add_argument("--receipts-dir", type=Path, required=True)
    ap.add_argument("--materials-root", type=Path, nargs="+",
                    default=[REPO_ROOT / "apt_g1/outputs"])
    ap.add_argument("--env-tag", choices=["lab-ts", "local"], required=True)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "apt_g1/outputs/sync/to41_sanity")
    args = ap.parse_args()

    if args.env_tag == "lab-ts" and platform.platform().lower().startswith("windows"):
        raise SystemExit("FAIL: 本机禁止以 lab-ts 身份出报告")
    if args.env_tag == "lab-ts" and not args.materials_root:
        raise SystemExit("FAIL: lab-ts 报告必须 --materials-root（checker 独立重算材料）")

    lut_manifest = load_lut_manifest()
    rep = build_report(args.receipts_dir, list(args.materials_root), args.env_tag,
                       lut_manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.env_tag == "lab-ts" else "_local_not_L_artifact"
    jp = args.out / f"L_report{suffix}.json"
    mp = args.out / f"L_report{suffix}.md"
    jp.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mp.write_text(report_markdown(rep), encoding="utf-8")
    v = rep["verdict"]
    print(f"L_report: schema={v['schema_check']} L1={v['L1']} L2={v['L2']} "
          f"L3={v['L3']} L4={v['L4']} overall={v['overall']}")
    print(f"  -> {jp}")
    return 0 if v["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
