"""D066 C2/C3 门执行面（计划 = refine-logs/ds/DS_CONTINUOUS_EXECUTION_PLAN.md §5w）。

A0 前置门（§5w）三条判据里，本脚本负责 **C2 特权可分** 与 **C3 不变量守恒（decoder
ONNX md5）** 的可执行部分；C1（命令保真）不在本文件范围。

C2（§5w 原文口径）：四族地形（plane / rough@0.04 / rough@0.08 / climb_box@0.5）
各抽验 elevation patch——**复杂族非零占比 > 0，且族间可分（plane 与复杂族均值差
≥ 复杂族内标准差）**。本栈特权观测 = 官方 187 维 height_scan（RayCaster 打
/World/ground，critic 组无噪原值，g1_velocity_decoder_env.py O1），与地板地形族
一一对应；故「elevation patch 非零性/可分性」在 C2 的实现 = **critic 组 height_scan
逐点样本的统计量**（每族一个独立子进程起 env 采集，父进程跨族判据）。

口径与既有实现对齐（不复制判定逻辑）：
  - 取 critic 组 height_scan 的切片法同 train_g1_decoder.py:424-456
    `_height_scan_obs`（2.1.0 ObservationManager：按 term 名定位 index + 前缀维度
    累加切片）；该函数硬编码 "policy" 组，故此处参数化 group 名（critic 为
    §5w 要求的「无噪」侧，O1 已把该组全部去噪）。
  - 复杂族「非零」阈值沿用 train_g1_decoder.py:477 的既有判据常量
    （max>0.02 且 std>1e-3 的绝对值版）→ 本文件 NONZERO_ABS_TOL。
  - plane 与复杂族可分：见 `c2_family_separable`。**假设声明**：rough_paper 是
    零均值对称噪声（HfRandomUniformTerrainCfg noise_range=(-noise,+noise)），
    带符号均值恒 ≈0，若直接用带符号均值则会恒判 FAIL——故判据落在 elevation
    **幅值** |height_scan| 上（与既有 `_terrain_verdict` 用 mean_abs 同口径），
    JSON 同时保留带符号 mean/std 供核对。plane 侧：本栈 plane 变体走官方 flat
    语义（height_scan 整项置 None，flat_env_cfg.py:22-23），按 D064 既有约定
    「term-absent → ≈0」取幅值 0（即 plane_abs_mean=0），非实采 187 点。

C3（§5w 原文口径）：修复不触碰冻结 decoder——本脚本以 **decoder ONNX 文件 md5
与在案基线一致** 落地（`--onnx-path` / `--onnx-md5-baseline` / `--baseline-json`
均可参）。默认基线 = D064 G0 在案 md5（refine-logs/tracker/D.md D064 行）。

用法（服务器，仓根 cwd；本机无 isaaclab 时只跑 --selftest）：
    # 全门：四族各起子进程采 critic height_scan → C2 判据 → C3 md5 → JSON+TERRAIN_GATE 行
    python apt_g1/isaac/d066_privileged_gate.py \
        --onnx-path $G/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
        --out-dir apt_g1/outputs/sync/d066
    # 本机 CPU 自检（纯数学/装配逻辑，不 import isaaclab）
    python apt_g1/isaac/d066_privileged_gate.py --selftest

机器可读行（供盯守代理 grep，均 flush）：
    子进程： "TERRAIN_GATE_CHILD <family> PASS|FAIL <stats-json>"
    父进程： "TERRAIN_GATE PASS|FAIL C2=... C3=... json=<path>"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# 仓根（本文件位于 <repo>/apt_g1/isaac/）
REPO_ROOT = Path(__file__).resolve().parents[2]

# 默认输出目录 = §5w 产物落点（outputs/sync/ 是跨端同步白名单，.gitignore:73-74）
DEFAULT_OUT_DIR = "apt_g1/outputs/sync/d066"
DEFAULT_OUT_NAME = "d066_gate.json"
# decoder ONNX（相对服务器执行根；g1_velocity_decoder_env.py:173 同一默认值）
DEFAULT_ONNX_PATH = "gear_sonic_deploy/policy/release/model_decoder.onnx"
# D064 G0 在案 decoder md5（refine-logs/tracker/D.md D064 行、train_g1_decoder 冒烟）
DEFAULT_ONNX_MD5_BASELINE = "1d4391ad404d0ff36281abec349b23fd"

# 复杂族「非零」幅值阈值（同 train_g1_decoder.py:477 的 mx>0.02 口径；std 阈值同源）
NONZERO_ABS_TOL = 0.02
NONZERO_STD_TOL = 1.0e-3
# 单点子「非零」判定（用于 nonzero_frac 计数）：远小于地形高度量级
POINT_NONZERO_TOL = 1.0e-3

# 单个族子进程上限（秒）：起 Isaac app + env + reset + N 步，同 train 的 480s 放宽
DEFAULT_CHILD_TIMEOUT_S = 900

# 族清单（§5w 原文四族；顺序固定，JSON 与判据按此序）
FAMILY_ORDER = ("plane", "rough_paper_0.04", "rough_paper_0.08", "climb_box_0.5")
FAMILY_SPECS: dict[str, dict] = {
    "plane": {
        "terrain": "plane",
        "kind": "plane",
        "complex": False,
        "noise": None,
        "box_height": None,
        "desc": "官方 G1FlatEnvCfg flat 语义（_apply_flat_semantics；height_scan 整项置 None）→ 幅值约定 ≈0",
    },
    "rough_paper_0.04": {
        "terrain": "rough_paper",
        "kind": "rough_paper",
        "complex": True,
        "noise": 0.04,
        "box_height": None,
        "desc": "terrain_cfg.rough_paper noise=0.04（G0 论文形对称噪声，E 系口径）",
    },
    "rough_paper_0.08": {
        "terrain": "rough_paper",
        "kind": "rough_paper",
        "complex": True,
        "noise": 0.08,
        "box_height": None,
        "desc": "terrain_cfg.rough_paper noise=0.08（E 系 0.08 悬崖档）",
    },
    "climb_box_0.5": {
        "terrain": "climb_box",
        "kind": "climb_box",
        "complex": True,
        "noise": None,
        "box_height": 0.5,
        "desc": "terrain_cfg.climb_box（MeshRandomGridTerrainCfg boxes，高度档 0.5m 量级；语料 climbing_box{height:0.5}）",
    },
}
COMPLEX_FAMILIES = tuple(f for f in FAMILY_ORDER if FAMILY_SPECS[f]["complex"])

# §5w 原文判据（进 JSON 供判读逐字核对，不在代码里改写）
C2_CRITERION_ZH = "四族地形 elevation patch 复杂族非零占比>0，且族间可分（plane 与复杂族均值差 ≥ 复杂族内标准差）"
C3_CRITERION_ZH = "修复不触碰冻结 decoder（权重 md5 修复前后一致）；不改变 author token 接口"


# ------------------------------------------------------------------ 纯函数（本机可测）
def family_specs() -> dict[str, dict]:
    """族配置表副本（调用方可改动返回体而不污染模块常量）。"""
    return {k: dict(v) for k, v in FAMILY_SPECS.items()}


def elevation_stats(values) -> dict:
    """elevation 样本统计。values = height_scan 展平样本（带符号，米）。

    返回键：n / n_finite / all_finite / mean / std / abs_mean / abs_std / abs_max /
    nonzero_frac。abs_* 为幅值口径（C2 判据用），mean/std 带符号（供核对）。
    std 用总体标准差（ddof=0）——样本即全量采集点，非抽样估计。
    """
    import numpy as np

    a = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(a.size)
    finite = np.isfinite(a)
    n_finite = int(finite.sum())
    if n == 0:
        return {
            "n": 0, "n_finite": 0, "all_finite": True,
            "mean": 0.0, "std": 0.0, "abs_mean": 0.0, "abs_std": 0.0,
            "abs_max": 0.0, "nonzero_frac": 0.0,
        }
    af = a[finite]
    absf = np.abs(af)
    return {
        "n": n,
        "n_finite": n_finite,
        "all_finite": bool(n_finite == n),
        "mean": float(af.mean()) if n_finite else 0.0,
        "std": float(af.std()) if n_finite else 0.0,
        "abs_mean": float(absf.mean()) if n_finite else 0.0,
        "abs_std": float(absf.std()) if n_finite else 0.0,
        "abs_max": float(absf.max()) if n_finite else 0.0,
        "nonzero_frac": float((absf > POINT_NONZERO_TOL).mean()) if n_finite else 0.0,
    }


def family_nonzero(stats: dict) -> bool:
    """复杂族「非零有方差」判据（同 train_g1_decoder.py:477 的幅值版口径）。"""
    return bool(stats["all_finite"] and stats["abs_max"] > NONZERO_ABS_TOL
                and stats["abs_std"] > NONZERO_STD_TOL)


def c2_family_separable(plane_stats: dict, complex_stats: dict) -> tuple[bool, dict]:
    """单族 C2 可分判据：|plane_abs_mean - complex_abs_mean| ≥ complex_abs_std。

    幅值口径理由见模块 docstring「假设声明」。附加条件照 §5w 原文前半句 =
    复杂族非零占比 > 0（nonzero_frac > 0）；带方差版诊断（family_nonzero，
    同 train_g1_decoder.py:477 幅值口径）只作报告字段，不进门的合取——§5w
    原文只要求「非零占比>0」。返回 (ok, detail)。
    """
    diff = abs(float(complex_stats["abs_mean"]) - float(plane_stats["abs_mean"]))
    nonzero_ok = bool(complex_stats["nonzero_frac"] > 0.0)
    ok = bool(nonzero_ok and diff >= float(complex_stats["abs_std"]))
    return ok, {
        "plane_abs_mean": float(plane_stats["abs_mean"]),
        "complex_abs_mean": float(complex_stats["abs_mean"]),
        "complex_abs_std": float(complex_stats["abs_std"]),
        "complex_nonzero_frac": float(complex_stats["nonzero_frac"]),
        "mean_diff": diff,
        "nonzero_ok": nonzero_ok,
        "nonzero_with_variance": family_nonzero(complex_stats),
        "criterion": "|plane_abs_mean - complex_abs_mean| >= complex_abs_std 且 complex_nonzero_frac > 0",
        "pass": ok,
    }


def c2_verdict(family_stats: dict[str, dict], family_errors: dict[str, str] | None = None) -> dict:
    """跨族 C2 判据：plane 作锚，逐复杂族判可分；全过 = PASS。

    family_errors 非空（某族子进程失败/超时）时**整体判 FAIL**——否则失败族会用
    占位零统计参与判据，plane 失败时其锚被充成 0 反可能放过其余族（假 PASS）。
    """
    plane = family_stats.get("plane")
    family_errors = dict(family_errors or {})
    per_family: dict[str, dict] = {}
    if plane is None:
        return {"pass": False, "error": "plane 族统计缺失", "per_family": per_family}
    for fam in COMPLEX_FAMILIES:
        st = family_stats.get(fam)
        if st is None:
            per_family[fam] = {"pass": False, "error": "族统计缺失"}
            continue
        ok, detail = c2_family_separable(plane, st)
        per_family[fam] = detail
    passed = bool(per_family) and all(v.get("pass") for v in per_family.values())
    return {
        "pass": bool(passed and not family_errors),
        "plane_abs_mean": float(plane["abs_mean"]),
        "per_family": per_family,
        "family_errors": family_errors,
        "nonzero_tol_abs": NONZERO_ABS_TOL,
        "nonzero_tol_std": NONZERO_STD_TOL,
        "point_nonzero_tol": POINT_NONZERO_TOL,
        "criterion_zh": C2_CRITERION_ZH,
    }


def file_md5(path: str) -> str | None:
    """文件 md5 十六进制串；不存在/不可读返回 None（同 eval_g1_decoder.py:49-57）。"""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.md5()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def extract_baseline_md5(payload) -> str | None:
    """从在案身份信封 JSON 里取 decoder/onnx md5（多候选键，先命中先得）。

    覆盖本仓既有信封格式：d064 run.json（decoder_onnx.md5，train_g1_decoder.py:489-492）
    与 d064 eval JSON（identity.decoder_onnx_md5，eval_g1_decoder.py:453）。
    """
    if not isinstance(payload, dict):
        return None
    direct = payload.get("onnx_md5") or payload.get("decoder_md5")
    if isinstance(direct, str) and direct:
        return direct
    onnx = payload.get("decoder_onnx")
    if isinstance(onnx, dict):
        v = onnx.get("md5")
        if isinstance(v, str) and v and v != "absent":
            return v
    ident = payload.get("identity")
    if isinstance(ident, dict):
        v = ident.get("decoder_onnx_md5")
        if isinstance(v, str) and v:
            return v
    return None


def c3_onnx_check(onnx_path: str, baseline_md5: str, baseline_json: str = "") -> dict:
    """C3 落地面：decoder ONNX 文件 md5 与在案基线一致性。

    baseline_json 命中时优先取该文件内记录的 md5（读取路径参数化）；否则用
    baseline_md5 字面量。文件缺失/基线缺失 = FAIL（如实，不静默 pass）。
    """
    actual = file_md5(onnx_path)
    source = "cli:--onnx-md5-baseline"
    expected = baseline_md5 or ""
    if baseline_json:
        try:
            payload = json.loads(Path(baseline_json).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return {
                "pass": False,
                "onnx_path": str(onnx_path),
                "onnx_md5": actual,
                "baseline_md5": None,
                "baseline_source": f"cli:--baseline-json ({type(exc).__name__}: {exc})",
                "error": "baseline-json 不可读/不可解析",
            }
        source = f"json:{baseline_json}"
        expected = extract_baseline_md5(payload) or ""
        if not expected:
            return {
                "pass": False,
                "onnx_path": str(onnx_path),
                "onnx_md5": actual,
                "baseline_md5": None,
                "baseline_source": source,
                "error": "baseline-json 内未找到 decoder/onnx md5 键",
            }
    ok = bool(actual is not None and expected and actual == expected)
    return {
        "pass": ok,
        "onnx_path": str(onnx_path),
        "onnx_md5": actual,
        "baseline_md5": expected or None,
        "baseline_source": source,
        "criterion_zh": C3_CRITERION_ZH,
        "error": None if ok else (
            "onnx 文件缺失/不可读" if actual is None else
            "基线 md5 缺失" if not expected else "md5 不一致"),
    }


def terrain_gate_line(verdict: str, c2_pass: bool, c3_pass: bool, json_path: str) -> str:
    """父进程汇总行（单行机器可读）。"""
    return (
        f"TERRAIN_GATE {verdict} "
        f"C2={'PASS' if c2_pass else 'FAIL'} C3={'PASS' if c3_pass else 'FAIL'} json={json_path}"
    )


def child_line(family: str, ok: bool, payload: dict) -> str:
    """子进程单行机器可读输出。"""
    return f"TERRAIN_GATE_CHILD {family} {'PASS' if ok else 'FAIL'} " + json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    )


def build_payload(cli, family_stats: dict, c2: dict, c3: dict, git_head: str) -> dict:
    """门 JSON 载荷（结构固定；selftest 断言键集合）。"""
    return {
        "entry": "apt_g1/isaac/d066_privileged_gate.py",
        "format": 1,
        "experiment": "D066",
        "gate": "A0-privileged (C2+C3)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": git_head,
        "cli": {
            "arm": cli.arm,
            "num_envs": cli.num_envs,
            "steps": cli.steps,
            "seed": cli.seed,
            "onnx_path": cli.onnx_path or DEFAULT_ONNX_PATH,
            "onnx_md5_baseline": cli.onnx_md5_baseline,
            "baseline_json": cli.baseline_json,
        },
        "family_order": list(FAMILY_ORDER),
        "complex_families": list(COMPLEX_FAMILIES),
        "families": family_stats,
        "c2": c2,
        "c3": c3,
        "verdict": "PASS" if (c2.get("pass") and c3.get("pass")) else "FAIL",
    }


# ------------------------------------------------------------------ CLI
class _D066ArgumentParser(argparse.ArgumentParser):
    """跨字段守卫：--arm decoder 缺 --token-stats 直接 argparse error（同 train 纪律）。"""

    def parse_args(self, args=None, namespace=None):
        ns = super().parse_args(args, namespace)
        if ns.arm == "decoder" and not ns.token_stats:
            self.error(
                "--arm decoder 需要 --token-stats <npz>（SonicDecoderActionTermCfg.token_stats"
                " 为空串时 term init 直接抛 ValueError）；C2 门只需 obs 时建议 --arm direct"
            )
        return ns


def build_args() -> argparse.ArgumentParser:
    ap = _D066ArgumentParser(
        description="D066 C2/C3 门：四族特权 height_scan 非零性与族间可分 + decoder ONNX md5 守恒"
    )
    ap.add_argument(
        "--selftest", action="store_true",
        help="本机 CPU 自检（纯数学/装配逻辑，不 import isaaclab）；全过 exit 0",
    )
    ap.add_argument(
        "--family", choices=FAMILY_ORDER, default=None, metavar="NAME",
        help=argparse.SUPPRESS,  # 隐藏子命令：父进程按族 spawn 的子进程体
    )
    ap.add_argument(
        "--arm", choices=("direct", "decoder"), default="direct",
        help="动作通路（C2 只读 obs，与 arm 无关；默认 direct 免 token 资产）",
    )
    ap.add_argument("--num-envs", type=int, default=8, help="每族 env 数（默认 8，冒烟规模）")
    ap.add_argument("--steps", type=int, default=20, help="reset 后采样步数（默认 20）")
    ap.add_argument("--seed", type=int, default=0, help="地形/环境 seed（默认 0）")
    ap.add_argument("--out-dir", default="", help=f"JSON 输出目录（默认 {DEFAULT_OUT_DIR}，锚定仓根）")
    ap.add_argument("--onnx-path", default="", help=f"decoder ONNX 路径（默认 {DEFAULT_ONNX_PATH}）")
    ap.add_argument(
        "--onnx-md5-baseline", default=DEFAULT_ONNX_MD5_BASELINE,
        help="在案基线 md5（默认 D064 G0 在案值）；--baseline-json 命中时被覆盖",
    )
    ap.add_argument("--baseline-json", default="", help="在案身份信封 JSON（内含 decoder md5；读取路径参数）")
    ap.add_argument("--token-stats", default="", help="--arm decoder 必填（token 仿射统计 npz）")
    ap.add_argument("--token-alpha", type=float, default=None)
    ap.add_argument("--token-bound", choices=("none", "tanh"), default=None)
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--child-timeout", type=float, default=DEFAULT_CHILD_TIMEOUT_S,
        help=f"单族子进程上限秒（默认 {DEFAULT_CHILD_TIMEOUT_S}）",
    )
    return ap


def _resolve_out_dir(cli) -> Path:
    raw = cli.out_dir or DEFAULT_OUT_DIR
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _git_head() -> str:
    try:
        g = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=5,
        )
        return g.stdout.strip() if g.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ 父进程
def _child_argv(cli, family: str) -> list[str]:
    argv = [sys.executable, os.path.abspath(__file__), "--family", family]
    argv += ["--arm", cli.arm, "--num-envs", str(cli.num_envs), "--steps", str(cli.steps)]
    argv += ["--seed", str(cli.seed)]
    argv += ["--token-stats", cli.token_stats or ""]
    if cli.onnx_path:
        argv += ["--onnx-path", cli.onnx_path]
    if cli.token_alpha is not None:
        argv += ["--token-alpha", str(cli.token_alpha)]
    if cli.token_bound is not None:
        argv += ["--token-bound", cli.token_bound]
    # 不传 --out-dir/--device：子进程不写盘，device 自 AppLauncher 取（train r4 教训）
    if not cli.headless:
        argv.append("--no-headless")
    return argv


def _run_family(cli, family: str) -> dict:
    """起一族的子进程并解析 TERRAIN_GATE_CHILD 行；任何失败都落成带 error 的条目。"""
    argv = _child_argv(cli, family)
    print(f"[d066] family {family} -> spawn", flush=True)
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"spawn-failed: {type(exc).__name__}: {exc}"}
    out = err = ""
    timed_out = False
    try:
        out, err = proc.communicate(timeout=cli.child_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        timed_out = True
    if err:
        sys.stderr.write(err if err.endswith("\n") else err + "\n")
        sys.stderr.flush()
    if out:
        sys.stdout.write(out if out.endswith("\n") else out + "\n")
        sys.stdout.flush()
    line = next((ln for ln in (out or "").splitlines() if ln.startswith(f"TERRAIN_GATE_CHILD {family} ")), None)
    if line is None:
        why = f"timeout>{cli.child_timeout}s" if timed_out else f"no-line (exit {proc.returncode})"
        return {"error": why}
    parts = line.split(" ", 3)
    if len(parts) < 4:
        return {"error": f"malformed child line: {line}"}
    try:
        return json.loads(parts[3])
    except json.JSONDecodeError as exc:
        return {"error": f"child json decode: {exc}"}


def parent_main(cli) -> int:
    out_dir = _resolve_out_dir(cli)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / DEFAULT_OUT_NAME

    family_stats: dict[str, dict] = {}
    for fam in FAMILY_ORDER:
        spec = dict(FAMILY_SPECS[fam])
        res = _run_family(cli, fam)
        entry = {"spec": spec, **res}
        family_stats[fam] = entry
        print(f"[d066] family {fam}: {'ok' if 'error' not in res else res['error']}", flush=True)

    # C2：每族 critic 统计（plane term-absent → 幅值 0 约定，见模块 docstring）
    crit_stats: dict[str, dict] = {}
    family_errors: dict[str, str] = {}
    for fam in FAMILY_ORDER:
        e = family_stats[fam]
        if e.get("error"):
            # 子进程失败：不得用占位零统计冒充实采（plane 失败会伪造零锚 → 假 PASS）
            family_errors[fam] = str(e["error"])
            continue
        st = e.get("critic")
        if isinstance(st, dict):
            crit_stats[fam] = st
        else:
            family_errors[fam] = "critic 统计缺失"
    c2 = c2_verdict(crit_stats, family_errors)

    # C3：decoder ONNX md5
    onnx_path = cli.onnx_path or DEFAULT_ONNX_PATH
    c3 = c3_onnx_check(onnx_path, cli.onnx_md5_baseline, cli.baseline_json)

    payload = build_payload(cli, family_stats, c2, c3, _git_head())
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(terrain_gate_line(payload["verdict"], bool(c2.get("pass")), bool(c3.get("pass")), str(out_json)),
          flush=True)
    return 0 if payload["verdict"] == "PASS" else 1


# ------------------------------------------------------------------ 子进程（含 Isaac）
def _import_train():
    """双兼容 import train_g1_decoder（同 eval_g1_decoder.py:366-372）。"""
    try:
        from isaac import train_g1_decoder as td
    except ImportError:  # pragma: no cover（本机走此支）
        from apt_g1.isaac import train_g1_decoder as td
    return td


def _import_terrain_cfg():
    """双兼容 import terrain_cfg（取 make_terrain_importer_cfg / CLIMB_BOX_HEIGHT_RANGE）。"""
    try:
        from isaac.terrain_cfg import CLIMB_BOX_HEIGHT_RANGE, make_terrain_importer_cfg
    except ImportError:  # pragma: no cover
        from apt_g1.isaac.terrain_cfg import CLIMB_BOX_HEIGHT_RANGE, make_terrain_importer_cfg
    return make_terrain_importer_cfg, CLIMB_BOX_HEIGHT_RANGE


def _height_scan_slice(om, group: str):
    """按组取 height_scan 的 (start, width, how)。

    逻辑逐字对照 train_g1_decoder.py:424-456 `_height_scan_obs`（按 term 名定位
    index + 前缀 term 维度累加切片；异常必带类型名），差异仅在其硬编码 policy
    组处参数化 group（C2 要 critic 无噪组）。取不到返回 (None, None, how)。
    """
    import torch

    try:
        names = om.active_terms[group]
        dims = om.group_obs_term_dim[group]
        if "height_scan" not in names:
            return None, None, "term-absent"
        idx = names.index("height_scan")
        if idx >= len(dims):
            return None, None, f"IndexError: names/dims 错位 names={len(names)} dims={len(dims)}"
        start = 0
        for term_dims in dims[:idx]:
            start += int(torch.tensor(list(term_dims)).prod())
        width = int(torch.tensor(list(dims[idx])).prod())
        return start, width, f"slice@{idx}[{start}:{start + width}]"
    except Exception as exc:  # noqa: BLE001
        return None, None, f"{type(exc).__name__}: {exc}"


def _build_family_cfg(hv, spec: dict, cli, device: str):
    """按族构建 env cfg（复用工厂；rough_paper 覆写同 eval_g1_decoder.py:301-310）。"""
    make_terrain_importer_cfg, climb_range = _import_terrain_cfg()
    tok = {}
    if cli.arm == "decoder":
        tok["token_stats"] = cli.token_stats
        if cli.onnx_path:
            tok["onnx_path"] = cli.onnx_path
        if cli.token_alpha is not None:
            tok["token_alpha"] = cli.token_alpha
        if cli.token_bound is not None:
            tok["token_bound"] = cli.token_bound

    kind = spec["kind"]
    if kind == "plane":
        # plane = 官方 flat 语义（height_scan 整项 None；C2 plane 幅值 0 约定）
        cfg = hv.make_env_cfg("plane", action=cli.arm, **tok)
    elif kind == "rough_paper":
        # rough 底 cfg（保 obs=321 含 height_scan）+ 覆写 scene.terrain 为 rough_paper
        cfg = hv.make_env_cfg("rough", action=cli.arm, **tok)
        cfg.scene.terrain = make_terrain_importer_cfg("rough_paper", noise=spec["noise"], seed=cli.seed)
        cfg.curriculum.terrain_levels = None
        gen = getattr(cfg.scene.terrain, "terrain_generator", None)
        if gen is not None and hasattr(gen, "curriculum"):
            gen.curriculum = False
    elif kind == "climb_box":
        if spec["box_height"] is not None and abs(max(climb_range) - float(spec["box_height"])) > 1e-9:
            raise RuntimeError(
                f"climb_box 高度档漂移：terrain_cfg.CLIMB_BOX_HEIGHT_RANGE={climb_range} "
                f"与族表 box_height={spec['box_height']} 不一致（单变量纪律，拒绝继续）"
            )
        cfg = hv.make_env_cfg("climb_box", action=cli.arm, **tok)
    else:
        raise ValueError(f"unknown family kind {kind!r}")
    cfg.scene.num_envs = cli.num_envs
    if hasattr(cfg.sim, "device"):
        cfg.sim.device = device
    return cfg, climb_range


def child_main(cli, device: str) -> int:
    """子进程体：起一族 env → reset + N 步 → 采 critic/policy height_scan → 统计行。

    子进程输出（JSON 载荷）：
      family / spec / groups{g:how} / height_scan{group: {term_absent, how, stats}} /
      critic（= height_scan["critic"].stats，父进程直接消费，C2 用）/
      policy（= height_scan["policy"].stats，仅核对用）/ error（失败时）。
    """
    import numpy as np

    family = cli.family
    spec = dict(FAMILY_SPECS[family])
    payload: dict = {"family": family, "spec": spec, "height_scan": {}, "critic": None, "policy": None}
    print(f"[d066] child family={family} arm={cli.arm} num_envs={cli.num_envs} "
          f"steps={cli.steps} pid={os.getpid()}", flush=True)

    env = None
    ok = False
    try:
        td = _import_train()
        hv = td._import_heavy()
        torch = hv.torch
        _, climb_range = _import_terrain_cfg()
        payload["climb_box_height_range"] = list(climb_range)

        cfg, _ = _build_family_cfg(hv, spec, cli, device)
        env = hv.ManagerBasedRLEnv(cfg=cfg)
        obs, _ = env.reset()
        adim = int(env.action_manager.total_action_dim)

        om = env.observation_manager
        groups = [g for g in ("critic", "policy") if g in getattr(om, "active_terms", {})]
        slices = {g: _height_scan_slice(om, g) for g in groups}
        payload["groups"] = {g: slices[g][2] for g in groups}

        acc: dict[str, list] = {g: [] for g in groups}
        for _ in range(int(cli.steps)):
            obs, *_ = env.step(torch.zeros(int(cli.num_envs), adim, device=env.device))
            for g in groups:
                acc[g].append(obs[g].detach().cpu().numpy())
        for g in groups:
            start, width, how = slices[g]
            if start is None:
                # term-absent（plane 官方 flat 语义整项置 None）→ 幅值 0 约定（非实采）
                stats = elevation_stats(np.zeros((int(cli.num_envs), 1), dtype=float))
                payload["height_scan"][g] = {"term_absent": True, "how": how, "stats": stats}
            else:
                block = np.concatenate([a[:, start:start + width] for a in acc[g]], axis=0)
                payload["height_scan"][g] = {"term_absent": False, "how": how,
                                             "stats": elevation_stats(block)}
        if "critic" in payload["height_scan"]:
            payload["critic"] = payload["height_scan"]["critic"]["stats"]
        if "policy" in payload["height_scan"]:
            payload["policy"] = payload["height_scan"]["policy"]["stats"]
        ok = payload["critic"] is not None
        if not ok:
            payload["error"] = f"critic 组不可用（active_terms={list(getattr(om, 'active_terms', {}))}）"
    except Exception as exc:  # noqa: BLE001 —— 任何失败都落到 FAIL 行
        payload["error"] = f"{type(exc).__name__}: {exc}"
        ok = False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    print(child_line(family, ok, payload), flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------ selftest（纯本地）
def run_selftest() -> int:
    import numpy as np

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    print("[d066-selftest] A. 族配置表")
    specs = family_specs()
    check("族数 = 4（§5w 四族）", len(specs) == 4, str(sorted(specs)))
    check("族名顺序固定", tuple(specs) == FAMILY_ORDER, str(FAMILY_ORDER))
    check("复杂族 = 3 且不含 plane",
          COMPLEX_FAMILIES == ("rough_paper_0.04", "rough_paper_0.08", "climb_box_0.5"))
    check("plane 非复杂 + terrain=plane", specs["plane"]["terrain"] == "plane" and specs["plane"]["complex"] is False)
    check("rough@0.04/0.08 noise 正确",
          specs["rough_paper_0.04"]["noise"] == 0.04 and specs["rough_paper_0.08"]["noise"] == 0.08)
    check("climb_box@0.5 terrain/height",
          specs["climb_box_0.5"]["terrain"] == "climb_box" and specs["climb_box_0.5"]["box_height"] == 0.5)
    specs["plane"]["terrain"] = "zzz"
    check("族表副本不污染模块常量", FAMILY_SPECS["plane"]["terrain"] == "plane")

    print("[d066-selftest] B. elevation_stats 数学")
    z = elevation_stats([0.0, 0.0, 0.0, 0.0])
    check("全零：abs_mean/std/frac 均 0",
          z["abs_mean"] == 0.0 and z["abs_std"] == 0.0 and z["nonzero_frac"] == 0.0)
    c = elevation_stats([0.1, -0.1, 0.1, -0.1])
    check("±0.1：带符号 mean=0 / std=0.1", abs(c["mean"]) < 1e-12 and abs(c["std"] - 0.1) < 1e-12)
    check("±0.1：abs_mean=0.1 / abs_std=0 / frac=1", abs(c["abs_mean"] - 0.1) < 1e-12
          and abs(c["abs_std"]) < 1e-12 and c["nonzero_frac"] == 1.0)
    s = elevation_stats([0.0, 0.2])
    check("[0,0.2]：abs_mean=0.1 / abs_std=0.1 / frac=0.5",
          abs(s["abs_mean"] - 0.1) < 1e-12 and abs(s["abs_std"] - 0.1) < 1e-12 and s["nonzero_frac"] == 0.5)
    check("空样本安全", elevation_stats([])["n"] == 0 and elevation_stats([])["abs_mean"] == 0.0)
    check("非有限计入 all_finite=False",
          elevation_stats([0.1, float("nan"), float("inf")])["all_finite"] is False)
    check("2D 展平", elevation_stats(np.zeros((3, 4)))["n"] == 12)

    print("[d066-selftest] C. 非零有方差判据")
    check("常量 0.5：abs_max 过但 std=0 → 非零判 FAIL", family_nonzero(elevation_stats([0.5] * 10)) is False)
    check("0.1±抖动：非零判 PASS", family_nonzero(elevation_stats([0.1, 0.11, 0.09, 0.1])) is True)
    check("全零：非零判 FAIL", family_nonzero(elevation_stats([0.0] * 8)) is False)

    print("[d066-selftest] D. C2 可分判据")
    plane0 = elevation_stats(np.zeros(200))
    ok1, d1 = c2_family_separable(plane0, elevation_stats(np.full(200, 0.1)))
    check("常量 0.1 族：diff=0.1 ≥ std=0 → PASS（分母为 0 边界）", ok1 is True and d1["pass"] is True)
    ok2, d2 = c2_family_separable(plane0, elevation_stats(np.concatenate([np.full(100, 0.4), np.full(100, 0.1)])))
    check("0.25±0.15：abs_mean=0.25 ≥ abs_std=0.15 → PASS（清晰非边界）",
          ok2 is True and abs(d2["mean_diff"] - 0.25) < 1e-9 and abs(d2["complex_abs_std"] - 0.15) < 1e-9)
    ok3, d3 = c2_family_separable(plane0, elevation_stats(np.concatenate([np.full(100, 0.2), np.full(100, -0.2)])))
    check("±0.2 对半：abs_mean=0.2 / abs_std=0 → PASS", ok3 is True)
    ok4, d4 = c2_family_separable(plane0, elevation_stats(np.zeros(200)))
    check("全零复杂族：frac=0 → FAIL", ok4 is False and d4["complex_nonzero_frac"] == 0.0)
    # 低幅值高离散反例：[0.5]×20+[0]×180 → abs_mean=0.05 < abs_std=0.1 → 不可分
    wide = np.concatenate([np.full(20, 0.5), np.zeros(180)])
    ok5, d5 = c2_family_separable(plane0, elevation_stats(wide))
    check("abs_mean(0.05) < abs_std(0.1) → FAIL（判据有区分力）",
          ok5 is False and d5["mean_diff"] < d5["complex_abs_std"])
    plane_off = elevation_stats(np.full(100, 0.5))
    ok6, _ = c2_family_separable(plane_off, elevation_stats(np.full(100, 0.52)))
    check("plane 非零锚时 diff=0.02 仍可比（幅值差口径）", ok6 is True)

    print("[d066-selftest] E. C2 跨族裁决")
    fam_stats = {
        "plane": elevation_stats(np.zeros(100)),
        "rough_paper_0.04": elevation_stats(np.concatenate([np.full(50, 0.04), np.zeros(50)])),
        "rough_paper_0.08": elevation_stats(np.concatenate([np.full(50, 0.08), np.zeros(50)])),
        # climb_box@0.5：0.4/0.5 两档网格（abs_mean=0.225, abs_std≈0.221 → 略过门）
        "climb_box_0.5": elevation_stats(np.concatenate([np.full(50, 0.5), np.full(50, 0.4)])),
    }
    v = c2_verdict(fam_stats)
    check("三复杂族全可分 → C2 PASS", v["pass"] is True)
    bad = dict(fam_stats); bad["climb_box_0.5"] = elevation_stats(np.zeros(100))
    check("缺非零族 → C2 FAIL", c2_verdict(bad)["pass"] is False)
    check("plane 缺失 → C2 FAIL", c2_verdict({"rough_paper_0.04": elevation_stats(np.ones(4))})["pass"] is False)
    check("族子进程失败 → C2 FAIL（即使其余族全可分，防假 PASS）",
          c2_verdict(fam_stats, {"plane": "timeout>900s"})["pass"] is False
          and c2_verdict(fam_stats, {"rough_paper_0.08": "exit 1"})["pass"] is False)
    check("family_errors 入 JSON 供判读", c2_verdict(fam_stats, {"plane": "x"})["family_errors"] == {"plane": "x"})

    print("[d066-selftest] F. C3 md5 检查")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="d066_selftest_") as td_:
        f = Path(td_) / "model_decoder.onnx"
        f.write_bytes(b"onnx-bytes")
        md5 = file_md5(str(f))
        check("file_md5 可算", isinstance(md5, str) and len(md5) == 32)
        check("md5 匹配 → C3 PASS", c3_onnx_check(str(f), md5)["pass"] is True)
        check("md5 不匹配 → C3 FAIL", c3_onnx_check(str(f), "0" * 32)["pass"] is False)
        check("文件缺失 → C3 FAIL", c3_onnx_check(str(Path(td_) / "nope.onnx"), md5)["pass"] is False)
        bj = Path(td_) / "run.json"
        bj.write_text(json.dumps({"decoder_onnx": {"path": "x", "md5": md5}}), encoding="utf-8")
        check("baseline-json(decoder_onnx.md5) 命中 → PASS",
              c3_onnx_check(str(f), "", str(bj))["pass"] is True)
        bj2 = Path(td_) / "eval.json"
        bj2.write_text(json.dumps({"identity": {"decoder_onnx_md5": md5}}), encoding="utf-8")
        check("baseline-json(identity.decoder_onnx_md5) 命中 → PASS",
              c3_onnx_check(str(f), "", str(bj2))["pass"] is True)
        bj3 = Path(td_) / "empty.json"
        bj3.write_text(json.dumps({"foo": 1}), encoding="utf-8")
        check("baseline-json 无 md5 键 → FAIL",
              c3_onnx_check(str(f), "", str(bj3))["pass"] is False)
        check("baseline-json 不可解析 → FAIL",
              c3_onnx_check(str(f), "", str(Path(td_) / "gone.json"))["pass"] is False)
        check("extract_baseline_md5 顶层键",
              extract_baseline_md5({"onnx_md5": md5}) == md5 and extract_baseline_md5({"x": 1}) is None)

    print("[d066-selftest] G. 参数解析")
    ap = build_args()
    d = ap.parse_args(["--selftest"])
    check("默认 arm=direct", d.arm == "direct")
    check("默认 num-envs=8 / steps=20 / seed=0",
          d.num_envs == 8 and d.steps == 20 and d.seed == 0)
    check("默认 out-dir 空（运行时锚定 DEFAULT_OUT_DIR）", d.out_dir == "")
    check("默认 basline md5 = D064 在案", d.onnx_md5_baseline == DEFAULT_ONNX_MD5_BASELINE)
    check("--family 接受四族之一", ap.parse_args(["--family", "climb_box_0.5"]).family == "climb_box_0.5")
    dec = ap.parse_args(["--arm", "decoder", "--token-stats", "t.npz"])
    check("decoder + token-stats 可解析", dec.arm == "decoder" and dec.token_stats == "t.npz")
    try:
        ap.parse_args(["--arm", "decoder"])
        guard = False
    except SystemExit:
        guard = True
    check("decoder 缺 token-stats → SystemExit 守卫", guard)
    try:
        ap.parse_args(["--family", "moon"])
        bad_fam = False
    except SystemExit:
        bad_fam = True
    check("非法族名 → SystemExit", bad_fam)
    check("--no-headless 生效", ap.parse_args(["--no-headless"]).headless is False)

    print("[d066-selftest] H. JSON 载荷与输出行结构")
    cli = ap.parse_args(["--selftest"])
    c2 = c2_verdict(fam_stats)
    c3 = {"pass": True, "onnx_md5": "a", "baseline_md5": "a"}
    pay = build_payload(cli, fam_stats, c2, c3, "deadbeef")
    rt = json.loads(json.dumps(pay, ensure_ascii=False))
    req = {"entry", "format", "experiment", "gate", "created_at", "git_head", "cli",
           "family_order", "complex_families", "families", "c2", "c3", "verdict"}
    check("载荷键集合齐备", req.issubset(rt.keys()), str(sorted(req - set(rt.keys()))))
    check("verdict 由 C2∧C3 决定", rt["verdict"] == ("PASS" if (c2["pass"] and c3["pass"]) else "FAIL"))
    c3f = {"pass": False}
    check("C3 FAIL → verdict FAIL", build_payload(cli, fam_stats, c2, c3f, "x")["verdict"] == "FAIL")
    check("c2 判据原文入 JSON", rt["c2"]["criterion_zh"] == C2_CRITERION_ZH)
    line = terrain_gate_line("PASS", True, True, "p.json")
    check("TERRAIN_GATE 行格式", line == "TERRAIN_GATE PASS C2=PASS C3=PASS json=p.json")
    cl_ = child_line("plane", True, {"a": 1})
    check("TERRAIN_GATE_CHILD 行可解析",
          cl_.startswith("TERRAIN_GATE_CHILD plane PASS ") and json.loads(cl_.split(" ", 3)[3]) == {"a": 1})

    print("[d066-selftest] I. 延迟 import 纪律（本机无 isaaclab 也必须可跑）")
    check("isaaclab 未 import", "isaaclab" not in sys.modules)
    check("torch 未 import（子进程路径才 import）", "torch" not in sys.modules)

    n_fail = sum(1 for _, c_, _ in checks if not c_)
    print(f"[d066-selftest] {len(checks) - n_fail}/{len(checks)} 项通过")
    if n_fail:
        print(f"SELFTEST FAIL: {n_fail} 项未过", flush=True)
        return 1
    print("SELFTEST PASS", flush=True)
    return 0


# ------------------------------------------------------------------ main
def main() -> int:
    cli = build_args().parse_args()
    if cli.selftest:
        return run_selftest()
    if cli.family:
        # 子进程体：AppLauncher 链（同 train_g1_decoder.py:1012-1021）
        from isaaclab.app import AppLauncher

        lp = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(lp)
        launcher_args, _ = lp.parse_known_args()
        launcher_args.num_envs = cli.num_envs
        launcher_args.headless = cli.headless
        app_launcher = AppLauncher(launcher_args)
        simulation_app = app_launcher.app
        code = 1
        try:
            code = child_main(cli, getattr(launcher_args, "device", "cuda:0"))
        finally:
            simulation_app.close()
        return code
    return parent_main(cli)


if __name__ == "__main__":
    sys.exit(main())
