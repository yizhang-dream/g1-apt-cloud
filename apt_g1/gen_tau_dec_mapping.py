#!/usr/bin/env python3
"""生成 Rung 1 treatment mapping artifact：rung1_tau_dec_mapping.yaml。

身份：pre-registered treatment mapping artifact（TO40C_PLAN §10.3 补充纪律）——
treatment specification，不是结果文件；pre-run-only，禁一切 observed/expected 字段。

确定性：cfg 默认值从冻结源文件 apt_flat_env.py 正则提取（不硬编码）；
bin 算子与冻结代码逐字同源（torch.linspace + torch.bucketize right=False +
dir-bin 公式镜像 apt_flat_env.py dir 路径），不重新实现判定逻辑。
同输入重跑必须 byte-identical——因此 YAML 不含任何时间戳/路径类易变字段。

用法：python apt_g1/gen_tau_dec_mapping.py   （在仓库根的任意位置均可，路径按本文件定位）
"""
import hashlib
import math
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
ENV_PY = ROOT / "apt_g1/isaac/apt_flat_env.py"
VAE_PT = ROOT / "apt_g1/outputs/token_vae_e39/vae.pt"
VAE_PY = ROOT / "apt_g1/train_token_vae_e39.py"
OUT = ROOT / "apt_g1/configs/rung1_tau_dec_mapping.yaml"

# 已批准网格（TO40C_PLAN §10.3，四轮批准）：6 均匀点 + 0.277 TO 锚点 = 7 records
GRID = [0.200, 0.225, 0.250, 0.275, 0.277, 0.300, 0.325]
FORBIDDEN_KEYS = ("observed", "expected_mismatch", "posthoc", "result")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(path: Path) -> str:
    """输入材料所在 commit（source_commit 语义：材料版本，非生成时 HEAD）。"""
    out = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", path.as_posix()],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not out:
        raise SystemExit(f"FAIL: {path} 未被 git 跟踪，无法确立 source commit")
    return out


def extract_frozen_cfg(pattern: str, name: str) -> str:
    """从冻结源文件正则提取 cfg 默认值；找不到即硬失败（禁止静默回退硬编码）。"""
    m = re.search(pattern, ENV_PY.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"FAIL: 冻结 cfg {name} 在 {ENV_PY} 中未匹配到: {pattern}")
    return m.group(1)


def main() -> None:
    n_bins = int(extract_frozen_cfg(r"latent_vae_n_bins:\s*int\s*=\s*(\d+)", "latent_vae_n_bins"))
    n_dbins = int(extract_frozen_cfg(r"latent_vae_n_dbins:\s*int\s*=\s*(\d+)", "latent_vae_n_dbins"))
    vx_max = float(extract_frozen_cfg(r"vx_max:\s*float\s*=\s*([\d.]+)", "vx_max"))

    # ── 与 apt_flat_env.py dir-bin 路径逐字同源的冻结函数 ──
    edges = torch.linspace(0.0, vx_max, n_bins + 1)[1:-1]          # 同冻结 edges 行
    cmd_v = torch.tensor(GRID, dtype=torch.float32)
    vb = torch.bucketize(cmd_v, edges).tolist()                    # right=False，同冻结
    # dir bin：forward cmd（y=0）→ ang=0；公式镜像冻结行（*8/%8 为冻结实现硬编码）
    db = int(math.floor((0.0 + math.pi) / (2.0 * math.pi) * n_dbins)) % n_dbins

    records = [dict(target_speed=v, decoder_condition_id=f"vb{b}_db{db}")
               for v, b in zip(GRID, vb)]
    conditions = {}
    for r in records:
        conditions.setdefault(r["decoder_condition_id"], []).append(r["target_speed"])

    # ── gate A 自检：schema 完整性 + pre-run-only ──
    assert len(records) == 7, f"records={len(records)} != 7"
    assert len(conditions) == 2, f"unique conditions={len(conditions)} != 2"
    assert sum(len(v) for v in conditions.values()) == 7, "condition 注册表未覆盖全部 records"
    hashes = {
        "decoder_checkpoint_hash": sha256(VAE_PT),
        "decoder_architecture_hash": sha256(VAE_PY),
        "preprocessing_hash": sha256(ENV_PY),
        "normalization_hash": sha256(ENV_PY),  # 本栈 preprocess/normalize 同源 apt_flat_env.py，见 notes
    }
    env_commit, vae_commit, vae_py_commit = git_commit(ENV_PY), git_commit(VAE_PT), git_commit(VAE_PY)

    lines = [
        "# Rung 1 pre-registered treatment mapping artifact",
        "# 身份 = treatment specification（pre-run-only，无任何结果字段）；冻结后改动 = Rung 1 身份失效",
        "artifact_id: rung1-tau-dec-mapping",
        "schema_version: 1",
        "mapping_rule_version: 1",
        "created_from:",
        f"  binning_source: apt_g1/isaac/apt_flat_env.py@{env_commit}",
        f"  decoder_checkpoint: apt_g1/outputs/token_vae_e39/vae.pt@{vae_commit}",
        f"  decoder_architecture: apt_g1/train_token_vae_e39.py@{vae_py_commit}",
        "freeze_status: generated-not-frozen",
        "",
        "mapping_provenance:",
        f"  source_commit: {env_commit}",
        "  generation_procedure: >-",
        "    gen_tau_dec_mapping.py v1：正则提取 apt_flat_env.py 冻结 cfg 默认值"
        " (latent_vae_n_bins, latent_vae_n_dbins, vx_max) → edges = torch.linspace(0, vx_max, n+1)[1:-1]"
        " → vb = torch.bucketize(cmd_vx, edges) (right=False) → dir bin 镜像冻结公式；",
        "    算子与冻结代码逐字同源，不重新实现判定逻辑；同输入重跑 byte-identical。",
        "",
        "command_regime: target_speed = episode 恒定 cmd_vx（eval 实物口径，单值贯穿 60s）；forward cmd (y=0)",
        "",
        "decoder_conditions:",
    ]
    band = {0: "slow", 1: "mid"}
    for cid in sorted(conditions):
        b = int(cid[2:cid.index("_")])
        lines.append(f"  {cid}: {{speed_bin: {b}, dir_bin: {db}, band: {band.get(b, f'vb{b}')}}}")
    lines.append("mappings:")
    for r in records:
        lines.append(f"  - {{target_speed: {r['target_speed']:.3f}, decoder_condition_id: {r['decoder_condition_id']}}}")
    lines += [
        "",
        "decoder_checkpoint_hash: " + hashes["decoder_checkpoint_hash"],
        "decoder_architecture_hash: " + hashes["decoder_architecture_hash"],
        "preprocessing_hash: " + hashes["preprocessing_hash"],
        "normalization_hash: " + hashes["normalization_hash"],
        "",
        "generation_notes:",
        "  boundary_semantics: >-",
        f"    bucketize right=False：恰在边界的值归上侧 bin；edges = [{edges[0].item():.7f},",
        f"    {edges[1].item():.7f}]，本网格无点触界（0.2667 是机制边界，非网格点，不新增测试点）。",
        "  structure: 7 target-speed records × 2 unique decoder conditions（非逐速条件化）。",
        "  diagnostic_alignment: >-",
        "    slow/mid 边界 0.2667 与 TO40C 观察的 0.25→0.277 行为分裂带对齐——",
        "    post hoc consistency observation（诊断事实），不是网格设计依据，",
        "    不得反向用作 grid validity 论证。",
        "  assignment_integrity: >-",
        "    bin 选择仅依赖 cmd_vx，τ_ff ON/OFF 不进入计算——treatment assignment",
        "    构造上正交；D gate dry-run 复核两臂同 target_speed 的 bin assignment 一致。",
        "  deterministic_not_sufficient: >-",
        "    无人工挑选自由度 ≠ 两态 conditioning 足以控制 decoder mismatch；",
        "    后者正是 Rung 1 的识别问题。",
        f"  environment: torch/{torch.__version__} python/{sys.version.split()[0]}（同环境重跑 byte-identical）",
    ]
    text = "\n".join(lines) + "\n"

    low = text.lower()
    for bad in FORBIDDEN_KEYS:
        assert bad not in low, f"pre-run-only 违例：出现结果类字段 {bad}"

    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"OK: {OUT}（{len(records)} records / {len(conditions)} conditions）")
    print("gate A (schema) PASS；gate B 请重跑本脚本比对 byte-identical。")


if __name__ == "__main__":
    main()
