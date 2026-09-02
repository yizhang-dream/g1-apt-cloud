"""TO41 D 阶段 independent checker（read-only audit，禁止修改任何实验状态）。

角色（协议 §10.2 / SCRIPT_MAP 登记）：**read-only audit**。只读取与独立计算，
verdict 一律自行重算（协议 §9）：runtime 侧 receipt 只提供 *record*，本模块
从不消费任何 runtime 自报 PASS/ok flag（receipt schema 全域封禁此类字段，
出现即 FAIL）。

独立性纪律：本模块**不 import** apt_g1.rung1.mode_a_runtime；frozen artifacts
（mapping v2 YAML / source registry / G_DOWN_SPEC §9 availability map）由本
模块用自己的解析代码独立读取，与 runtime 侧解析互为交叉验证（selftest 覆盖）。

判据唯一事实源 = refine-logs/TO41_D_DRYRUN_PROTOCOL.md：
- D1  decoder invariance（七字段清单 + mode/layout/shapes runtime identity）
- D2  assignment + same-τ identity（Mode A fingerprint；hash 相等 + lineage 必在）
- D3A assignment conformance：T_runtime(v,C) = T_mapping(v,C)
- D3B material conformance：τ_runtime(v) = τ_frozen(v)
D 禁收任何 performance / locomotion 字段（协议 §4）；发现问题的唯一正当出口
= report FAIL → 协议 §7 保险丝。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import platform
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPING_YAML = REPO_ROOT / "apt_g1/configs/rung1_tau_dec_mapping.yaml"
REGISTRY_YAML = REPO_ROOT / "apt_g1/configs/rung1_source_registry.yaml"
GDOWN_SPEC_MD = REPO_ROOT / "refine-logs/TO41_G_DOWN_SPEC.md"

REPORT_SCHEMA = "rung1-d-audit/v1"
SELFTEST_SCHEMA = "rung1-d-selftest-audit/v1"
CONDITION_ARMS = ("C1", "C2")
TAU_FF_ARMS = ("on", "off")

# receipt 顶层 schema（封闭契约；协议 §4 字段）
RECEIPT_TOP_KEYS = {
    "schema", "cell_id", "target_speed", "condition_arm", "tau_ff",
    "runtime_assignment", "tau_material", "decoder_identity",
    "mode_layout_shapes", "execution",
}
RECEIPT_SCHEMA_ID = "rung1-d-receipt/v1"
# 自报 verdict 类字段：任何层级出现即 schema FAIL（协议 §9 禁自证）
BLOCKLIST_VERDICT_KEYS = {
    "pass", "passed", "ok", "okay", "verdict", "conformant", "assignment_ok",
    "condition_ok", "tau_ok", "material_ok", "decoder_ok", "d1", "d2", "d3",
    "d1_pass", "d2_pass", "d3_pass", "d3a_pass", "d3b_pass", "result_pass",
}
EXPECTED_SELECTION_SOURCE = "frozen_mapping_v2_lookup"
EXPECTED_DECODE_LAYOUT = ["z", "phase_sc(sin,cos)", "v_bin", "d_bin"]
EXPECTED_PROBE_SHAPES = {"z": [1, 16], "phase_sc": [1, 2], "v_bin": [1], "d_bin": [1]}

# 期望 state_dict 签名（trace：nvbins=3/ndbins=8 ← env latent_vae_n_bins/
# latent_vae_n_dbins 默认，env 源被 preprocessing_hash 冻结；token 64/window 10/
# latent 16/hidden 256/phase 2 ← train_token_vae_e39.py 默认，源被
# decoder_architecture_hash 冻结；decoder.0 in-features = latent+phase+8+8）
EXPECTED_SD_SIGNATURE = {
    "speed_embed.weight": (3, 8),
    "dir_embed.weight": (8, 8),
    "decoder.0.weight": (256, 34),
    "decoder.0.bias": (256,),
    "decoder.2.weight": (256, 256),
    "decoder.2.bias": (256,),
    "decoder.4.weight": (64, 256),
    "decoder.4.bias": (64,),
}
EXPECTED_LOAD_PREFIXES = ("encoder.", "mu.", "logvar.")  # strict=False 丢弃的 encoder 侧


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 独立 frozen artifact 解析（与 runtime 侧不共享代码）
# ---------------------------------------------------------------------------

def load_mapping(path: Path = MAPPING_YAML) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["artifact_id"] == "rung1-tau-dec-mapping"
    assert raw["schema_version"] == 2 and raw["mapping_rule_version"] == 2
    conds = {}
    for cid, spec in (raw["decoder_conditions"]).items():
        m = re.fullmatch(r"vb(\d+)_db(\d+)", cid)
        assert m, f"bad condition id {cid}"
        conds[cid] = {"speed_bin": int(m.group(1)), "dir_bin": int(m.group(2)),
                      "arm": spec["arm"]}
    rows, grid = {}, []
    for r in raw["mappings"]:
        v = round(float(r["target_speed"]), 3)
        arm, cid = r["condition_arm"], r["decoder_condition_id"]
        assert arm in CONDITION_ARMS and cid in conds
        assert conds[cid]["arm"] == arm
        rows[(v, arm)] = cid
        if v not in grid:
            grid.append(v)
    assert len(rows) == 14 and len(grid) == 7, "mapping v2 必须 14 rows / 7 grid"
    return {
        "grid": sorted(grid), "conditions": conds, "rows": rows,
        "hashes": {
            "decoder_checkpoint": raw["decoder_checkpoint_hash"],
            "decoder_architecture": raw["decoder_architecture_hash"],
            "preprocessing": raw["preprocessing_hash"],
            "normalization": raw["normalization_hash"],
        },
        "freeze_status": raw.get("freeze_status"),
    }


def load_registry(path: Path = REGISTRY_YAML) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["artifact_id"] == "rung1-source-registry"
    by_path = {}
    for s in raw["sources"]:
        by_path[s["path"]] = {"id": s["id"], "sha256_16": s.get("sha256_16"),
                              "mode": s.get("mode"), "knots": s.get("knots"),
                              "v_dump": s.get("v_dump"), "r_valid": s.get("r_valid")}
    return {"by_path": by_path}


def load_availability(path: Path = GDOWN_SPEC_MD) -> dict:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^## 9\. ", text, re.M)
    assert m, "G_DOWN_SPEC §9 未找到"
    body = text[m.end():]
    m10 = re.search(r"^## 10\. ", body, re.M)
    if m10:
        body = body[:m10.start()]
    table = {}
    for line in body.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        try:
            v = round(float(cells[0]), 3)
            v_real, abs_err = float(cells[2]), float(cells[3])
        except ValueError:
            continue
        table[v] = {"artifact": cells[1], "v_realized": v_real,
                    "abs_err": abs_err, "source": cells[4], "determinism": cells[5]}
    assert len(table) == 7, f"availability map 行数 {len(table)} != 7"
    return table


def enumerate_cells(mapping: dict) -> list[dict]:
    cells = []
    for v in mapping["grid"]:
        for arm in CONDITION_ARMS:
            for ff in TAU_FF_ARMS:
                cells.append({"cell_id": f"v{v * 1000:04.0f}_{arm}_{ff}",
                              "target_speed": v, "condition_arm": arm, "tau_ff": ff})
    return cells


# ---------------------------------------------------------------------------
# receipt 校验（record 契约；verdict 一律重算）
# ---------------------------------------------------------------------------

def _scan_blocklist(node, path: str, violations: list[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if str(k).lower() in BLOCKLIST_VERDICT_KEYS:
                violations.append(f"{path}.{k}: 自报 verdict 类字段（协议 §9 禁止）")
            _scan_blocklist(v, f"{path}.{k}", violations)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scan_blocklist(v, f"{path}[{i}]", violations)


def validate_receipt(r: dict) -> list[str]:
    errs: list[str] = []
    if r.get("schema") != RECEIPT_SCHEMA_ID:
        errs.append(f"schema != {RECEIPT_SCHEMA_ID}")
    missing = RECEIPT_TOP_KEYS - set(r)
    extra = set(r) - RECEIPT_TOP_KEYS
    if missing:
        errs.append(f"顶层缺字段: {sorted(missing)}")
    if extra:
        errs.append(f"顶层多字段（封闭契约）: {sorted(extra)}")
    _scan_blocklist(r, "receipt", errs)
    if r.get("execution", {}).get("status") != "completed":
        errs.append("execution.status != completed")
    for block, keys in (
        ("runtime_assignment", {"decoder_condition_id", "speed_bin", "dir_bin", "selection_source"}),
        ("tau_material", {"artifact", "sha256", "sha256_16", "source_lineage", "v_realized",
                          "abs_err", "registry_id", "registry_sha256_16", "mode", "knots",
                          "npz_keys_shapes", "applied_to_env"}),
        ("decoder_identity", {"checkpoint_path", "checkpoint_sha256", "state_dict_sha256_before",
                              "state_dict_sha256_after", "architecture", "state_dict_key_shapes",
                              "load_missing_keys", "load_unexpected_keys",
                              "arch_source_file_sha256", "env_source_file_sha256",
                              "token_window_vae_source_sha256"}),
        ("mode_layout_shapes", {"mode", "decode_arg_layout", "input_shapes", "input_dtypes",
                                "output_shape", "output_dtype", "output_min", "output_max",
                                "output_sha256"}),
        ("execution", set()),
    ):
        got = set(r.get(block, {}))
        lack = keys - got
        if lack:
            errs.append(f"{block} 缺字段: {sorted(lack)}")
    return errs


def _close(a: float, b: float, rel: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=0.0)


# ---------------------------------------------------------------------------
# D1 / D2 / D3A / D3B
# ---------------------------------------------------------------------------

def check_d1(receipts: dict[str, dict], mapping: dict) -> tuple[str, list[str], list[dict]]:
    """decoder invariance：七字段清单 + mode/layout/shapes（逐 cell，全过才 PASS）。"""
    failures: list[str] = []
    per_cell: list[dict] = []
    sd_global: str | None = None
    out_pair: dict[tuple, str] = {}
    for cell in enumerate_cells(mapping):
        cid = cell["cell_id"]
        r = receipts.get(cid)
        if r is None:
            failures.append(f"{cid}: receipt 缺失")
            per_cell.append({"cell": cid, "verdict": "FAIL"})
            continue
        bad: list[str] = []
        di = r["decoder_identity"]
        mls = r["mode_layout_shapes"]
        # (1) checkpoint hash
        if di["checkpoint_sha256"] != mapping["hashes"]["decoder_checkpoint"]:
            bad.append("checkpoint_sha256 != mapping 冻结值")
        # (2) state_dict before==after + 全 cell 一致
        if di["state_dict_sha256_before"] != di["state_dict_sha256_after"]:
            bad.append("state_dict hash decode 前后不一致")
        if sd_global is None:
            sd_global = di["state_dict_sha256_before"]
        elif di["state_dict_sha256_before"] != sd_global:
            bad.append("state_dict hash 与其他 cell 不一致")
        # (3) architecture identity（源 hash + state_dict 签名 + 声明超参自洽）
        if di["arch_source_file_sha256"] != mapping["hashes"]["decoder_architecture"]:
            bad.append("arch 源文件 hash != mapping decoder_architecture_hash")
        shapes = di["state_dict_key_shapes"]
        for k, exp in EXPECTED_SD_SIGNATURE.items():
            got = shapes.get(k)
            if got is None:
                bad.append(f"state_dict 缺 {k}")
            elif tuple(got) != exp:
                bad.append(f"state_dict {k} shape {got} != 期望 {list(exp)}")
        arch = di["architecture"]
        exp_hp = {"class": "DirSpeedPhaseTokenVAE", "token_dim": 64, "window": 10,
                  "latent_dim": 16, "hidden_dim": 256, "phase_dim": 2,
                  "n_vbins": 3, "n_dbins": 8}
        for k, v in exp_hp.items():
            if arch.get(k) != v:
                bad.append(f"architecture.{k}={arch.get(k)} != {v}")
        # (4) latent dimensionality（34 = latent 16 + phase 2 + embed 16）
        if tuple(shapes.get("decoder.0.weight", (0, 0)))[1] != 34:
            bad.append("latent dimensionality 校验失败（decoder.0 in-features != 34）")
        # (5)(6) input transform / normalization identity（同源 env 文件，双 hash 对照）
        if di["env_source_file_sha256"] != mapping["hashes"]["preprocessing"]:
            bad.append("env 源文件 hash != mapping preprocessing_hash")
        if di["env_source_file_sha256"] != mapping["hashes"]["normalization"]:
            bad.append("env 源文件 hash != mapping normalization_hash")
        if mls["decode_arg_layout"] != EXPECTED_DECODE_LAYOUT:
            bad.append(f"decode_arg_layout != 冻结调用序 {EXPECTED_DECODE_LAYOUT}")
        # (7) output contract：shape/dtype/range（Tanh）
        if mls["output_shape"] != [1, 64]:
            bad.append(f"output_shape {mls['output_shape']} != [1, 64]")
        if mls["output_dtype"] != "torch.float32":
            bad.append(f"output_dtype {mls['output_dtype']} != torch.float32")
        if not (-1.0 <= mls["output_min"] and mls["output_max"] <= 1.0):
            bad.append("output 超出 Tanh 值域 [-1, 1]")
        # (8) runtime identity：mode/layout/shapes + 加载身份
        if not isinstance(mls.get("mode"), str) or not mls["mode"]:
            bad.append("mode_layout_shapes.mode 缺失")
        if mls["input_shapes"] != EXPECTED_PROBE_SHAPES:
            bad.append(f"probe input_shapes {mls['input_shapes']} != {EXPECTED_PROBE_SHAPES}")
        if di["load_missing_keys"]:
            bad.append(f"load missing keys 非空: {di['load_missing_keys']}")
        if not all(k.startswith(EXPECTED_LOAD_PREFIXES) for k in di["load_unexpected_keys"]):
            bad.append("load unexpected keys 超出 encoder/mu/logvar 前缀")
        # D1 伴生检查：τ_ff OFF/ON 不得改变 decode（decoder 计算语义不变）
        key = (cell["target_speed"], cell["condition_arm"])
        if cell["tau_ff"] == "on":
            out_pair[key] = mls["output_sha256"]
        else:
            if out_pair.get(key) not in (None, mls["output_sha256"]):
                bad.append("τ OFF 与 ON 的 decode 输出 hash 不一致（decoder 不应变）")
        for b in bad:
            failures.append(f"{cid}: {b}")
        per_cell.append({"cell": cid, "verdict": "FAIL" if bad else "PASS",
                         "failures": bad})
    return ("FAIL" if failures else "PASS"), failures, per_cell


def check_d2(receipts: dict[str, dict], mapping: dict) -> tuple[str, list[str], list[dict]]:
    """Mode A fingerprint：∀v，4 cell 同 τ hash + lineage 在场；并含 assignment 组成检查。"""
    failures: list[str] = []
    table: list[dict] = []
    for v in mapping["grid"]:
        rec = {"target_speed": v}
        hashes, lineages = {}, {}
        for arm in CONDITION_ARMS:
            for ff in TAU_FF_ARMS:
                key = f"v{v * 1000:04.0f}_{arm}_{ff}"
                r = receipts.get(key)
                if r is None:
                    failures.append(f"{key}: receipt 缺失")
                    continue
                hashes[(arm, ff)] = r["tau_material"]["sha256"]
                lin = r["tau_material"].get("source_lineage")
                if not (isinstance(lin, str) and lin.strip()):
                    failures.append(f"{key}: tau_source_lineage 缺失（hash 相等也不足，协议 §2）")
                lineages[(arm, ff)] = lin
            # assignment 组成检查（D2 两组成之一）
            expected = mapping["rows"][(v, arm)]
            r_on = receipts.get(f"v{v * 1000:04.0f}_{arm}_on")
            if r_on and r_on["runtime_assignment"]["decoder_condition_id"] != expected:
                failures.append(
                    f"v{v * 1000:04.0f}_{arm}: condition {r_on['runtime_assignment']['decoder_condition_id']} != mapping 期望 {expected}")
        uniq = set(hashes.values())
        same = len(uniq) == 1 and len(hashes) == 4
        lin_uniq = set(lineages.values())
        rec.update({
            "tau_hash_C1": hashes.get(("C1", "on")), "tau_hash_C2": hashes.get(("C2", "on")),
            "same_tau_identity": same, "lineage_uniform": len(lin_uniq) == 1,
            "verdict": "PASS" if (same and len(lin_uniq) == 1 and len(hashes) == 4) else "FAIL",
        })
        if len(hashes) == 4 and not same:
            failures.append(
                f"v={v}: C1/C2 τ hash 不一致（各自合法也判 FAIL = Mode A 被实现成 Mode B，协议 §2）")
        table.append(rec)
    return ("FAIL" if failures else "PASS"), failures, table


def check_d3a(receipts: dict[str, dict], mapping: dict) -> tuple[str, list[str], list[dict]]:
    """assignment conformance：T_runtime(v,C) = T_mapping(v,C)，逐 cell 对照。"""
    failures: list[str] = []
    table: list[dict] = []
    out_by_v: dict[float, dict[str, str]] = {}
    for cell in enumerate_cells(mapping):
        cid, v, arm = cell["cell_id"], cell["target_speed"], cell["condition_arm"]
        r = receipts.get(cid)
        if r is None:
            failures.append(f"{cid}: receipt 缺失")
            table.append({"cell": cid, "yaml": mapping["rows"][(v, arm)],
                          "runtime": None, "verdict": "FAIL"})
            continue
        expected = mapping["rows"][(v, arm)]
        ra = r["runtime_assignment"]
        bad = []
        if ra["decoder_condition_id"] != expected:
            bad.append(f"condition {ra['decoder_condition_id']} != yaml {expected}")
        cond = mapping["conditions"][expected]
        if ra["speed_bin"] != cond["speed_bin"] or ra["dir_bin"] != cond["dir_bin"]:
            bad.append("speed_bin/dir_bin 与 condition 注册表不一致")
        if ra["selection_source"] != EXPECTED_SELECTION_SOURCE:
            bad.append(f"selection_source {ra['selection_source']} 非 frozen lookup 路径")
        # condition selection 真正到达 decode：同 v 的 C1/C2 输出必须不同
        out_by_v.setdefault(v, {})[arm] = r["mode_layout_shapes"]["output_sha256"]
        for b in bad:
            failures.append(f"{cid}: {b}")
        table.append({"cell": cid, "yaml": expected,
                      "runtime": ra["decoder_condition_id"],
                      "verdict": "FAIL" if bad else "PASS", "failures": bad})
    for v, arms in out_by_v.items():
        if len(arms) == 2 and arms["C1"] == arms["C2"]:
            failures.append(f"v={v}: C1/C2 decode 输出 hash 相同——condition 未到达 decode")
            for row in table:
                if row["cell"].startswith(f"v{v * 1000:04.0f}_"):
                    row["verdict"] = "FAIL"
    return ("FAIL" if failures else "PASS"), failures, table


def check_d3b(receipts: dict[str, dict], mapping: dict, availability: dict,
              registry: dict, materials_root: Path | None) -> tuple[str, list[str], list[dict], dict]:
    """material conformance：τ_runtime(v) = τ_frozen(v)；含独立 hash 重算。"""
    failures: list[str] = []
    table: list[dict] = []
    recompute: dict[str, str] = {}
    for cell in enumerate_cells(mapping):
        cid, v = cell["cell_id"], cell["target_speed"]
        r = receipts.get(cid)
        if r is None:
            failures.append(f"{cid}: receipt 缺失")
            continue
        tm = r.get("tau_material", {})
        spec = availability[v]
        bad = []
        if tm.get("artifact") != spec["artifact"]:
            bad.append(f"material {tm.get('artifact')} != availability map {spec['artifact']}")
        if tm.get("v_realized") is None or abs(tm["v_realized"] - spec["v_realized"]) > 1e-9:
            bad.append(f"v_realized {tm.get('v_realized')} != 冻结表 {spec['v_realized']}")
        if tm.get("abs_err") is None or not _close(tm["abs_err"], spec["abs_err"]):
            bad.append(f"abs_err {tm.get('abs_err')} != 冻结表 {spec['abs_err']}")
        if not (isinstance(tm.get("source_lineage"), str) and tm["source_lineage"].strip()):
            bad.append("source_lineage 缺失")
        reg = registry["by_path"].get(spec["artifact"])
        if reg is not None:  # registry 既有三件：16-hex 冻结锚
            if tm.get("registry_sha256_16") != reg["sha256_16"]:
                bad.append("registry_sha256_16 与冻结 registry 不一致")
            if str(tm.get("sha256", ""))[:16] != reg["sha256_16"]:
                bad.append(f"material sha256 前缀 != registry 冻结 {reg['sha256_16']}")
            if tm.get("mode") != reg["mode"]:
                bad.append(f"mode {tm.get('mode')} != registry {reg['mode']}")
        if materials_root is not None:
            mp = materials_root / spec["artifact"]
            if not mp.exists():
                bad.append(f"独立重算失败：{mp} 不存在")
                recompute[spec["artifact"]] = "missing"
            else:
                live = sha256_file(mp)
                recompute[spec["artifact"]] = live
                if live != tm["sha256"]:
                    bad.append(f"独立重算 sha256 != receipt 记录（{spec['artifact']}）")
        for b in bad:
            failures.append(f"{cid}: {b}")
        table.append({"cell": cid, "target": v, "artifact": tm["artifact"],
                      "tau_hash16": tm.get("sha256_16"), "lineage": tm.get("source_lineage"),
                      "verdict": "FAIL" if bad else "PASS", "failures": bad})
    return ("FAIL" if failures else "PASS"), failures, table, recompute


# ---------------------------------------------------------------------------
# 静态覆盖 + 材料基线段（协议 §4 固定附段）
# ---------------------------------------------------------------------------

def static_coverage(mapping: dict, availability: dict) -> list[dict]:
    rows = []
    for cell in enumerate_cells(mapping):
        v, arm = cell["target_speed"], cell["condition_arm"]
        exp_c = mapping["rows"][(v, arm)]
        exp_m = availability[v]["artifact"]
        rows.append({"cell": cell["cell_id"], "target_speed": v, "condition_arm": arm,
                     "tau_ff": cell["tau_ff"], "expected_condition": exp_c,
                     "expected_material": exp_m})
    return rows


def material_baseline(availability: dict, registry: dict,
                      receipts: dict[str, dict] | None) -> list[dict]:
    rows = []
    for v in sorted(availability):
        spec = availability[v]
        reg = registry["by_path"].get(spec["artifact"])
        tau_hash = None
        if receipts:
            r = receipts.get(f"v{v * 1000:04.0f}_C1_on")
            if r:
                tau_hash = r["tau_material"]["sha256"]
        mark = ""
        if v in (0.300, 0.325):
            mark = ("<< accepted under the pre-registered ±0.02 m/s realization "
                    "tolerance（禁触发重解，协议 §4）")
        rows.append({
            "target": v, "artifact": spec["artifact"], "v_realized": spec["v_realized"],
            "abs_err": spec["abs_err"], "source": spec["source"],
            "determinism": spec["determinism"],
            "registry_id": reg["id"] if reg else None,
            "registry_sha256_16": reg["sha256_16"] if reg else None,
            "tau_hash_runtime": tau_hash, "note": mark,
        })
    return rows


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _observations(mapping: dict, availability: dict, registry: dict) -> list[str]:
    obs = [
        ("mapping YAML freeze_status = "
         f"{mapping['freeze_status']!r}（generated-not-frozen 字样；owner 冻结动作记录于"
         " tracker/TO.md 三十三轮——implementation 不改冻结工件，仅如实记录）"),
        ("eval 侧 decoder 类源文件 apt_g1/isaac/token_window_vae.py 在 mapping artifact 中"
         "无冻结 hash；D1 架构身份以 state_dict 签名 + 调用点（apt_flat_env.py，"
         "preprocessing_hash 冻结）+ 训练源（decoder_architecture_hash 冻结）覆盖，"
         "类源 sha 记录于各 receipt 供追溯"),
        "decode-only dry-run：τ 注入 env 的 exercise 属 Rung 1 launch sanity（IMPL §6），不在 D 范围",
    ]
    return obs


def build_report(receipts_dir: Path | None, materials_root: Path | None,
                 env_tag: str, selftest: bool = False,
                 mapping_path: Path = MAPPING_YAML,
                 registry_path: Path = REGISTRY_YAML,
                 gdown_path: Path = GDOWN_SPEC_MD) -> dict:
    """selftest 模式允许 registry_path 指向 synthetic 副本（真锚只在 lab-ts 验证）；
    正式 D report 恒用仓库内冻结工件。"""
    mapping = load_mapping(mapping_path)
    availability = load_availability(gdown_path)
    registry = load_registry(registry_path)

    receipts: dict[str, dict] = {}
    receipt_hashes: dict[str, str] = {}
    schema_failures: list[str] = []
    if receipts_dir is not None and receipts_dir.exists():
        for rp in sorted(receipts_dir.glob("receipt_*.json")):
            r = json.loads(rp.read_text(encoding="utf-8"))
            cid = r.get("cell_id", rp.stem)
            receipts[cid] = r
            receipt_hashes[cid] = sha256_file(rp)
            for e in validate_receipt(r):
                schema_failures.append(f"{cid}: {e}")

    report = {
        "artifact": SELFTEST_SCHEMA if selftest else REPORT_SCHEMA,
        "selftest": selftest,
        "generated_utc": _utcnow(),
        "environment_tag": env_tag,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "inputs": {
            "mapping_yaml": str(mapping_path),
            "mapping_yaml_sha256": sha256_file(mapping_path),
            "registry_yaml": str(registry_path),
            "registry_yaml_sha256": sha256_file(registry_path),
            "gdown_spec": str(gdown_path),
            "gdown_spec_sha256": sha256_file(gdown_path),
            "receipts_dir": str(receipts_dir) if receipts_dir else None,
            "receipt_count": len(receipts),
            "receipt_sha256": receipt_hashes,
            "materials_root": str(materials_root) if materials_root else None,
        },
        "static_coverage": {
            "expected": 28, "resolved": len(static_coverage(mapping, availability)),
        },
        "material_baseline": material_baseline(availability, registry, receipts or None),
        "observations": _observations(mapping, availability, registry),
        "scope_notes": [
            "D 只回答 plumbing 是否忠实执行 specification（协议 §4）；"
            "PASS 解释上限 = implementation conforms to treatment specification",
            "本 report 不含任何 performance / locomotion 字段（协议 §4 禁收）",
        ],
    }

    expected_cells = {c["cell_id"] for c in enumerate_cells(mapping)}
    got_cells = set(receipts)
    coverage = {
        "expected_cells": sorted(expected_cells),
        "missing": sorted(expected_cells - got_cells),
        "extra": sorted(got_cells - expected_cells),
    }
    report["execution_coverage"] = coverage

    if receipts:
        d1v, d1f, d1c = check_d1(receipts, mapping)
        d2v, d2f, d2t = check_d2(receipts, mapping)
        d3av, d3af, d3at = check_d3a(receipts, mapping)
        d3bv, d3bf, d3bt, recompute = check_d3b(receipts, mapping, availability,
                                                registry, materials_root)
        schema_v = "FAIL" if schema_failures else "PASS"
        d3v = "PASS" if (d3av == "PASS" and d3bv == "PASS") else "FAIL"
        overall = "PASS" if all(
            x == "PASS" for x in (schema_v, d1v, d2v, d3av, d3bv)) else "FAIL"
        if coverage["missing"] or coverage["extra"]:
            overall = "FAIL"
        report.update({
            "schema_check": {"verdict": schema_v, "failures": schema_failures},
            "D1": {"verdict": d1v, "failures": d1f, "per_cell": d1c},
            "D2": {"verdict": d2v, "failures": d2f, "table": d2t},
            "D3A": {"verdict": d3av, "failures": d3af, "table": d3at},
            "D3B": {"verdict": d3bv, "failures": d3bf, "table": d3bt,
                    "independent_hash_recompute": recompute},
            "D3": {"verdict": d3v},
            "overall": overall,
        })
    else:
        report.update({
            "schema_check": {"verdict": "NOT_RUN", "failures": []},
            "D1": {"verdict": "NOT_RUN", "failures": [], "per_cell": []},
            "D2": {"verdict": "NOT_RUN", "failures": [], "table": []},
            "D3A": {"verdict": "NOT_RUN", "failures": [], "table": []},
            "D3B": {"verdict": "NOT_RUN", "failures": [], "table": [],
                    "independent_hash_recompute": {}},
            "D3": {"verdict": "NOT_RUN"},
            "overall": "NOT_RUN",
        })
    return report


def report_markdown(rep: dict) -> str:
    lines = ["# TO41 D dry-run audit report（machine-generated）", ""]
    lines.append(f"- artifact: `{rep['artifact']}`  selftest={rep['selftest']}")
    lines.append(f"- generated: {rep['generated_utc']}  env_tag: {rep['environment_tag']}")
    lines.append(f"- platform: {rep['platform']} / python {rep['python']}")
    lines.append(f"- receipts: {rep['inputs']['receipt_count']} @ {rep['inputs']['receipts_dir']}")
    lines.append(f"- static coverage: {rep['static_coverage']['resolved']}/28")
    lines.append("")
    lines.append("## verdict")
    lines.append("")
    lines.append("| check | verdict |")
    lines.append("|---|---|")
    for name in ("schema_check", "D1", "D2", "D3A", "D3B", "D3", "overall"):
        v = rep[name]["verdict"] if name != "overall" else rep["overall"]
        lines.append(f"| {name} | **{v}** |")
    lines.append("")
    if rep.get("D2", {}).get("table"):
        lines.append("## D2 Mode A fingerprint（same v → same τ identity）")
        lines.append("")
        lines.append("| v | C1 τ hash(16) | C2 τ hash(16) | equal | lineage uniform | verdict |")
        lines.append("|---:|---|---|---|---|---|")
        for row in rep["D2"]["table"]:
            h1 = (row["tau_hash_C1"] or "")[:16]
            h2 = (row["tau_hash_C2"] or "")[:16]
            lines.append(f"| {row['target_speed']:.3f} | {h1} | {h2} | "
                         f"{row['same_tau_identity']} | {row['lineage_uniform']} | "
                         f"{row['verdict']} |")
        lines.append("")
    if rep.get("D3A", {}).get("table"):
        lines.append("## D3A assignment conformance（逐 cell）")
        lines.append("")
        lines.append("| cell | yaml | runtime | verdict |")
        lines.append("|---|---|---|---|")
        for row in rep["D3A"]["table"]:
            lines.append(f"| {row['cell']} | {row['yaml']} | {row['runtime']} | {row['verdict']} |")
        lines.append("")
    if rep.get("D3B", {}).get("table"):
        lines.append("## D3B material conformance（逐 cell）")
        lines.append("")
        lines.append("| cell | artifact | τ hash(16) | lineage | verdict |")
        lines.append("|---|---|---|---|---|")
        for row in rep["D3B"]["table"]:
            lines.append(f"| {row['cell']} | {row['artifact']} | {row['tau_hash16']} | "
                         f"{row['lineage']} | {row['verdict']} |")
        lines.append("")
    lines.append("## material baseline（G_DOWN_SPEC §9 照录）")
    lines.append("")
    lines.append("| target | artifact | v_realized | abs_err | source | determinism | note |")
    lines.append("|---:|---|---|---|---|---|---|")
    for row in rep["material_baseline"]:
        lines.append(f"| {row['target']:.3f} | {row['artifact']} | {row['v_realized']} | "
                     f"{row['abs_err']:.1e} | {row['source']} | {row['determinism']} | {row['note']} |")
    lines.append("")
    fails = []
    for name in ("schema_check", "D1", "D2", "D3A", "D3B"):
        fails.extend(rep[name].get("failures", []))
    if fails:
        lines.append("## failures")
        lines.append("")
        for f in fails:
            lines.append(f"- {f}")
        lines.append("")
    lines.append("## scope notes / observations")
    lines.append("")
    for s in rep["scope_notes"] + rep["observations"]:
        lines.append(f"- {s}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="TO41 D independent checker（read-only audit）")
    ap.add_argument("--receipts-dir", type=Path, default=None)
    ap.add_argument("--materials-root", type=Path, default=None,
                    help="提供后对每个 material 独立重算 sha256（lab-ts 执行必带）")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "apt_g1/outputs/sync/to41_d")
    ap.add_argument("--env-tag", choices=["lab-ts", "local"], required=True)
    ap.add_argument("--selftest", action="store_true",
                    help="自测模式：report 打 selftest 标记，永不作为 D artifact")
    ap.add_argument("--mapping", type=Path, default=MAPPING_YAML)
    ap.add_argument("--report-stem", type=str, default="D_report")
    args = ap.parse_args()

    if args.env_tag == "lab-ts" and platform.system().lower().startswith("windows"):
        raise SystemExit("FAIL: 本机禁止以 lab-ts 身份出 D 报告（协议 §10.1）")
    if args.env_tag == "lab-ts" and args.materials_root is None:
        raise SystemExit("FAIL: lab-ts D 报告必须提供 --materials-root（独立 hash 重算，协议 §9）")

    rep = build_report(args.receipts_dir, args.materials_root, args.env_tag,
                       selftest=args.selftest, mapping_path=args.mapping)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.report_stem}.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / f"{args.report_stem}.md").write_text(
        report_markdown(rep), encoding="utf-8")
    print(f"overall: {rep['overall']}  (schema={rep['schema_check']['verdict']} "
          f"D1={rep['D1']['verdict']} D2={rep['D2']['verdict']} "
          f"D3A={rep['D3A']['verdict']} D3B={rep['D3B']['verdict']})")
    print(f"report -> {args.out / (args.report_stem + '.json')}")
    return 0 if rep["overall"] in ("PASS", "NOT_RUN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
