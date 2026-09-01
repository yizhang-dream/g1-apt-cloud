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

    natural = {v: f"vb{b}_db{db}" for v, b in zip(GRID, vb)}
    # 十二轮裁定 Decision 1 = B：condition 为可干预 treatment axis，C 与 v 全交叉。
    # C1/C2 是 assignment 槽位（identity 取自冻结 binning 的两个 condition 注册项），
    # 不声称 gait 语义；non-natural 配对（如 0.20→vb1）即干预所在。
    CONDITION_ARMS = {"C1": "vb0_db4", "C2": "vb1_db4"}
    records = [dict(target_speed=v, condition_arm=arm, decoder_condition_id=cid,
                    natural_condition=natural[v])
               for v in GRID for arm, cid in CONDITION_ARMS.items()]
    conditions = dict(CONDITION_ARMS)

    # ── gate A 自检：schema 完整性 + pre-run-only ──
    assert len(records) == 14, f"records={len(records)} != 14 (7 speeds × 2 arms)"
    assert set(conditions.values()) == {"vb0_db4", "vb1_db4"}, "condition 注册表不符"
    assert len({(r["target_speed"], r["condition_arm"]) for r in records}) == 14, "(v, arm) 不唯一"
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
        "schema_version: 2",
        "mapping_rule_version: 2",
        "supersedes: schema v1（自然 bucketize 恒等物化，十二轮 B reopen 后退役为自然 assignment 参照，见 git 历史 468a1e7）",
        "created_from:",
        f"  binning_source: apt_g1/isaac/apt_flat_env.py@{env_commit}",
        f"  decoder_checkpoint: apt_g1/outputs/token_vae_e39/vae.pt@{vae_commit}",
        f"  decoder_architecture: apt_g1/train_token_vae_e39.py@{vae_py_commit}",
        "freeze_status: generated-not-frozen",
        "",
        "mapping_provenance:",
        f"  source_commit: {env_commit}",
        "  generation_procedure: >-",
        "    gen_tau_dec_mapping.py v2（十二轮 Decision 1=B reopen）：全交叉规则",
        "    T(v, c) = cond_c（C1=vb0_db4、C2=vb1_db4，槽位 identity 取自冻结 binning",
        "    注册表）；每 speed × 每 arm 一行，共 14 行；零人工挑选自由度；",
        "    同输入重跑 byte-identical。",
        "",
        "command_regime: target_speed = episode 恒定 cmd_vx（eval 实物口径，单值贯穿 60s）；forward cmd (y=0)",
        "",
        "decoder_conditions:",
    ]
    band = {0: "slow", 1: "mid"}
    for arm, cid in sorted(conditions.items()):
        b = int(cid[2:cid.index("_")])
        lines.append(f"  {cid}: {{arm: {arm}, speed_bin: {b}, dir_bin: {db}, bin_label: {band.get(b, f'vb{b}')}}}")
    lines.append("mappings:")
    for r in records:
        lines.append(
            f"  - {{target_speed: {r['target_speed']:.3f}, condition_arm: {r['condition_arm']}, "
            f"decoder_condition_id: {r['decoder_condition_id']}, natural_condition: {r['natural_condition']}}}"
        )
    lines += [
        "",
        "decoder_checkpoint_hash: " + hashes["decoder_checkpoint_hash"],
        "decoder_architecture_hash: " + hashes["decoder_architecture_hash"],
        "preprocessing_hash: " + hashes["preprocessing_hash"],
        "normalization_hash: " + hashes["normalization_hash"],
        "",
        "generation_notes:",
        "  ruling_provenance: >-",
        "    十二轮 owner 裁定 Decision 1 = B（condition 为可干预 treatment axis）：",
        "    同一 target speed 下主动改变 decoder condition 以识别",
        "    condition-dependent τ_ff effect；随附推翻条款——若同速双 condition",
        "    最终无法科学构造，推翻 B 而非硬造 condition。",
        "  structure: >-",
        "    14 mapping rows（7 target speeds × 2 condition arms）→ 28 eval cells",
        "    （× τ_ff ON/OFF）；natural_condition 列标出非自然配对（干预所在：",
        "    如 0.200→C2/vb1、0.325→C1/vb0）。全交叉使 positivity/cell coverage",
        "    by construction 成立；任何 cell 运行失败 = invalid（不插值/不跳过/",
        "    不合并）。",
        "  condition_semantics: >-",
        "    C1/C2 为 assignment 槽位，本 artifact 只支持 controlled",
        "    decoder-condition contrast；不得称 gait-condition effect——bin=gait",
        "    属后续解释，非本 artifact 身份。Δ_cond(v2 恒等式) = Δ_ff(v,C1) −",
        "    Δ_ff(v,C2)，其中 Δ_ff(v,C) = Y(τ ON,C) − Y(τ OFF,C)。",
        "  realization: >-",
        "    条件覆写为 eval 时作用于冻结 decode 路径的干预；τ_ff 为另一",
        "    treatment 轴，不在本 artifact（mapping 只管 condition assignment）。",
        "  boundary_semantics: >-",
        f"    bucketize right=False：恰在边界的值归上侧 bin；edges = [{edges[0].item():.7f},",
        f"    {edges[1].item():.7f}]，本网格无点触界（0.2667 是机制边界，非网格点，不新增测试点）。",
        "  diagnostic_alignment: >-",
        "    自然 bin 边界 0.2667 与 TO40C 观察的 0.25→0.277 行为分裂带对齐——",
        "    post hoc consistency observation（诊断事实），不是网格设计依据。",
        "  deterministic_not_sufficient: >-",
        "    D3/conformance PASS 只说 implementation conforms to treatment",
        "    specification；不证明 conditioning valid 或 C1/C2 是两个真实 gait",
        "    regimes；无人工挑选自由度 ≠ mismatch 已被控制。",
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
