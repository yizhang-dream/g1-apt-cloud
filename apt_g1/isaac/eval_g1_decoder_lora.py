"""D065 C 臂 v2 评测电池（FORK，同源 `eval_g1_decoder.py` 分叉不回并）。

三节电池与判据口径**逐条继承父本**（每条件独立子进程 / ep_steps=3000=60s@50Hz /
survived=整段未 terminated / flat 的 vx_rmse 仅算未 done 步 / det 推理）：
  rough_levels : 官方 rough 课程地形按固定行出生（terrain_levels 直写），指标=存活/摔步/净位移。
  rough_paper  : E 系连续性口径——terrain_cfg.rough_paper{noise} 固定噪声档（0.06/0.08）。
  flat         : 平地固定 vx 命令 {0.4,0.6}，指标=存活/vx RMSE。

【v2（in-graph）相对父本 eval_g1_decoder.py 的增量】
  E1 `--lora-scale`（默认 1.0）：0 = **遗忘检查**——μ 路径置零包络后与「未适配 base decoder 的 μ」
     逐位一致**硬断言**（同一 head 输出 + 同一真实 env obs，maxdiff 必须为 0，否则 EVAL_COND_FAIL）；
     同时报 scale=1 时 adapter 的真实效应（==0 ⇒ adapter 未生效，打 LORA_EFFECT_WARN）。
  E2 策略构造（v2 核心差异）：评测的 policy = `SonicLoRAPolicy`（decoder 在 policy 内）；
     env 侧 `action="lora_policy"` = **A 臂动作通路逐字**（29 维关节目标直通）+ policy 组末位
     930 维本体观测；权重从 ckpt 的 `model_state_dict` 恢复（adapter/base/head/critic/σ 全在内）；
     rank/alpha 取 ckpt 内 `lora` 信封（`--lora-ckpt` 或侧车可覆盖），缺则 CLI 默认。
  E3 **命令通道核验（D064 教训 + SF-5）**：flat 条件**每个控制步重戳**命令（父本只在 reset 后
     戳一次，被 resampling 覆写即失效），但**判据不采信「刚写进去的值」**——核验改为从
     **策略实际消费的 obs** 里切 `velocity_commands` 槽（`_policy_obs_cmd_slot` 运行期按
     observation_manager 的 term 序定位，见该函数依据）采样，落 `diag.obs_cmd_vx_*`；
     `cmd_vx_mean_last_batch`（批次末重读命令项）也进判据。父进程跨条件汇总
     `cmd_channel_check`（obs 槽采样必须等于标称 + 批次末命令等于标称 + 命令随条件变化，
     否则判通道失效，flat 数字不采信）。
  E4 输出文件名/身份串改 d065（`d065_eval_c_<ckpt>.json`），identity 块增 lora 段。
  E5 `--token-fidelity-probe <npz>`（遗忘判据②的仪表，SF-6）：token 窗口数组（(N,64) 或
     (N,T,64)，键名自动探测）过一遍 μ 路径，报 adapter 引起的输出 delta（μ_lora−μ_base 的
     max/mean）与 proprio 来源；**未提供时 JSON 显式记 `"token_fidelity": "not_probed"`**
     （判据降级可见化，不静默当作已过）。

运行环境：IsaacLab 2.1.0 / rsl-rl-lib 2.3.3 / torch 2.5.1（本机无 isaaclab，仅 py_compile 验证）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_LEVELS = "1,3,5,7,9"
DEFAULT_NOISE = "0.06,0.08"
DEFAULT_VX = "0.4,0.6"
EVAL_NUM_ENVS = 8
COND_TIMEOUT_S = 1500  # 单条件子进程上限：env 启动 ~1min + 批次 rollout，宽裕
LORA_ADAPTER_FILENAME = "lora_adapter.pt"  # 与 train_g1_decoder_lora 同名（侧车约定）
# 命令通道判据（E3）：命令与标称的允许偏差、近静止阈值（D064 教训：det 推理 actual_vx≈0）
CMD_NOMINAL_TOL = 1e-6
NEAR_STATIC_VX = 0.05


def _parse_csv_float(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_csv_int(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _file_md5(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


class _EvalArgumentParser(argparse.ArgumentParser):
    """跨字段守卫放 parse_args 本体：缺 token 资产 / ckpt / 非法 scale 即 SystemExit(2)。"""

    def parse_args(self, args=None, namespace=None):
        cli = super().parse_args(args, namespace)
        if not cli.token_stats:
            self.error("--token-stats <npz> 必填（token 仿射在 policy 内，需要 mean/std 统计）")
        if not cli.ckpt:
            self.error("--ckpt <path> 必填（待评 ckpt）")
        if cli.lora_scale < 0.0:
            self.error(f"--lora-scale 须 >= 0（0=遗忘检查置零），收到 {cli.lora_scale}")
        if int(cli.rank) < 1:
            self.error(f"--rank 须 >= 1，收到 {cli.rank}")
        return cli


def build_args() -> argparse.ArgumentParser:
    ap = _EvalArgumentParser(description="D065 C 臂 v2 评测电池（判据=§5u 继承 + §4 遗忘检查/命令通道核验）")
    ap.add_argument("--ckpt", default="", help="待评 ckpt（rsl_rl model_*.pt，含 model_state_dict + lora 信封）")
    ap.add_argument(
        "--arm",
        choices=("lora_policy",),
        default="lora_policy",
        help="C 臂 v2 只有一臂：env 侧 = A 臂动作通路逐字 + 930 维本体观测（decoder 在 policy 内）",
    )
    ap.add_argument(
        "--battery",
        choices=("rough_levels", "rough_paper", "flat", "all"),
        default="all",
        help="跑哪节电池（all=三节全跑）",
    )
    ap.add_argument("--levels", default=DEFAULT_LEVELS, help="rough_levels 出生行 csv（默认 1,3,5,7,9）")
    ap.add_argument("--noise", default=DEFAULT_NOISE, help="rough_paper 噪声档 csv（默认 0.06,0.08）")
    ap.add_argument("--vx-cmds", default=DEFAULT_VX, help="flat 固定 vx 命令 csv（默认 0.4,0.6）")
    ap.add_argument("--episodes", type=int, default=3, help="每条件批次数（每批 num_envs 局）")
    ap.add_argument("--ep-steps", type=int, default=3000, help="每局步数=60s@50Hz（判据冻结）")
    ap.add_argument("--num-envs", type=int, default=EVAL_NUM_ENVS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="", help="JSON 输出目录（默认 ckpt 同目录）")
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--token-stats", default="", help="必填（token 仿射在 policy 内；守卫见 _EvalArgumentParser）")
    ap.add_argument("--onnx-path", default="", help="SONIC ONNX decoder（policy 侧 LoRA decoder 的 base；缺省用默认）")
    ap.add_argument("--token-alpha", type=float, default=None)
    ap.add_argument("--token-bound", choices=("none", "tanh"), default=None)
    # ---- D065 v2 新增（E1/E2）----
    ap.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA 包络：1.0=正常推理；0.0=遗忘检查（置零后 μ 与未适配 base 逐位一致断言）",
    )
    ap.add_argument(
        "--lora-ckpt",
        default="",
        help="LoRA 信封来源（缺省：ckpt 内 lora 键 -> ckpt 同目录 lora_adapter.pt 侧车）",
    )
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank（信封缺 rank 时的回落值；须与训练一致）")
    ap.add_argument("--alpha", type=float, default=1.0, help="LoRA alpha（信封缺 alpha 时的回落值）")
    ap.add_argument(
        "--token-fidelity-probe",
        default="",
        help="token 保真抽测 npz（遗忘判据②的仪表，E5）：token 窗口数组（(N,64) 或 (N,T,64)，"
        "键名自动探测 token/window）过 μ 路径，报 adapter 输出 delta（μ_lora−μ_base 的 max/mean）；"
        "缺省空串 = 不探测（JSON 记 token_fidelity=not_probed）",
    )
    ap.add_argument(
        "--eval-one",
        default=None,
        metavar="COND_JSON",
        help=argparse.SUPPRESS,  # 内部子命令：父进程按条件逐个 spawn，输出 EVAL_COND 行
    )
    return ap


def expand_conditions(cli) -> list[dict]:
    """按 battery 展开条件列表（顺序稳定，父进程逐个派子进程）。"""
    conds: list[dict] = []
    if cli.battery in ("rough_levels", "all"):
        for lv in _parse_csv_int(cli.levels):
            conds.append({"kind": "rough_levels", "level": lv})
    if cli.battery in ("rough_paper", "all"):
        for nz in _parse_csv_float(cli.noise):
            conds.append({"kind": "rough_paper", "noise": nz})
    if cli.battery in ("flat", "all"):
        for vx in _parse_csv_float(cli.vx_cmds):
            conds.append({"kind": "flat", "vx": vx})
    return conds


def _child_argv(cli, cond: dict) -> list[str]:
    """从父 cli 重建子进程 argv（num-envs/episodes/ep-steps 透传；--eval-one 注入条件）。"""
    argv = [sys.executable, os.path.abspath(__file__)]
    argv += ["--arm", cli.arm, "--ckpt", cli.ckpt, "--battery", cli.battery]
    argv += ["--levels", cli.levels, "--noise", cli.noise, "--vx-cmds", cli.vx_cmds]
    argv += ["--episodes", str(cli.episodes), "--ep-steps", str(cli.ep_steps)]
    argv += ["--num-envs", str(cli.num_envs), "--seed", str(cli.seed)]
    argv += ["--token-stats", cli.token_stats or ""]
    if cli.onnx_path:
        argv += ["--onnx-path", cli.onnx_path]
    if cli.token_alpha is not None:
        argv += ["--token-alpha", str(cli.token_alpha)]
    if cli.token_bound is not None:
        argv += ["--token-bound", cli.token_bound]
    argv += ["--lora-scale", str(cli.lora_scale), "--rank", str(cli.rank), "--alpha", str(cli.alpha)]
    if cli.lora_ckpt:
        argv += ["--lora-ckpt", cli.lora_ckpt]
    if cli.token_fidelity_probe:
        argv += ["--token-fidelity-probe", cli.token_fidelity_probe]
    if not cli.headless:
        argv.append("--no-headless")
    argv += ["--eval-one", json.dumps(cond, ensure_ascii=False)]
    return argv


def _cmd_channel_check(results: list[dict]) -> dict:
    """E3/SF-5：跨条件命令通道核验（D064 教训——通道失效则 flat vx RMSE 不采信）。

    判据（全部基于**策略实际消费的 obs 槽**或**批次末重读命令项**，不是「刚写进去的值」）：
    ① 每个 flat 条件的 obs 命令槽采样都与标称相等（`cmd_matches_nominal`，SF-5 改为
       `diag.obs_cmd_vx_*` 口径）；
    ② 每个 flat 条件的 `cmd_vx_mean_last_batch`（批次末重读 `vel_command_b`）与标称相等；
    ③ 不同条件的命令互不相同（否则「命令根本没在变」，通道可能被钉死/覆写）；
    ④ 至少两个 flat 条件可比。全过 = PASS；任一不过 = FAIL；无条件 = INSUFFICIENT。
    """
    flat = [r for r in results if isinstance(r, dict) and (r.get("cond") or {}).get("kind") == "flat"]
    rows = []
    for r in flat:
        d = r.get("diag") or {}
        rows.append({
            "vx_nominal": (r.get("cond") or {}).get("vx"),
            "cmd_matches_nominal": d.get("cmd_matches_nominal"),
            "obs_cmd_vx_mean": d.get("obs_cmd_vx_mean"),
            "obs_cmd_vx_min": d.get("obs_cmd_vx_min"),
            "obs_cmd_vx_max": d.get("obs_cmd_vx_max"),
            "obs_cmd_vx_n_samples": d.get("obs_cmd_vx_n_samples"),
            "cmd_vx_mean_last_batch": d.get("cmd_vx_mean_last_batch"),
            "cmd_vx_mean_over_steps": d.get("cmd_vx_mean_over_steps"),
            "cmd_vx_min_over_steps": d.get("cmd_vx_min_over_steps"),
            "cmd_vx_max_over_steps": d.get("cmd_vx_max_over_steps"),
            "actual_vx_mean_all_envs": d.get("actual_vx_mean_all_envs"),
            "actual_vx_near_static": d.get("actual_vx_near_static"),
            "error": r.get("error"),
        })
    nominals = [row["vx_nominal"] for row in rows if row["vx_nominal"] is not None]
    means = [row["obs_cmd_vx_mean"] for row in rows if row["obs_cmd_vx_mean"] is not None]

    def _eq_nominal(v, nominal):
        return v is not None and nominal is not None and abs(float(v) - float(nominal)) <= CMD_NOMINAL_TOL

    all_match = bool(rows) and all(row["cmd_matches_nominal"] is True for row in rows)
    all_last_batch_match = bool(rows) and all(
        _eq_nominal(row["cmd_vx_mean_last_batch"], row["vx_nominal"]) for row in rows
    )
    distinct = len(set(nominals)) == len(nominals) and len(set(means)) == len(means)
    enough = len(rows) >= 2
    verdict = (
        "INSUFFICIENT" if (not rows or not enough)
        else ("PASS" if (all_match and all_last_batch_match and distinct) else "FAIL")
    )
    return {
        "verdict": verdict,
        "criteria": {
            "all_conditions_cmd_matches_nominal": all_match,
            "all_conditions_last_batch_matches_nominal": all_last_batch_match,
            "commands_distinct_across_conditions": distinct,
            "n_flat_conditions_ge_2": enough,
        },
        "flat_conditions": rows,
        "note": "FAIL ⇒ 命令通道未生效（D064 教训：resampling 覆写 vel_command_b），flat vx RMSE 不采信；"
        "cmd_matches_nominal 采自策略实际消费的 obs 命令槽（obs_cmd_vx_*），"
        "actual_vx_near_static=True 是 D064 已记录的 det 近静止现象，不作通道失效证据",
    }


def _ckpt_c_arm_key_guard(sd, missing) -> int:
    """SF-2：C 臂 ckpt 的键守卫（非 C 臂产物**拒绝出数**，不静默 WARN 放行）。

    - `missing`（`policy.load_state_dict(strict=False)` 的缺键）里凡含 `".decoder."` 者
      即 ckpt 内无 decoder 权重。**真实前缀是 `mu_path.decoder.*`**（decoder 是
      `SonicLoRAMuPath` 的子模块，见 sonic_lora_policy.SonicLoRAPolicy.__init__）——
      旧判据 `k.startswith("decoder.")` 永不匹配（死代码），A 臂 ckpt 只会打 WARN。
    - ckpt 内 `adapters.*` 键数为 0 ⇒ 该 ckpt 未挂 LoRA（不是 C 臂），同样拒绝。
    返回 ckpt 内的 adapter 键数（进报告，便于判读载荷完整性）。
    """
    bad_missing = [k for k in missing if isinstance(k, str) and ".decoder." in k]
    if bad_missing:
        raise RuntimeError(
            f"ckpt 缺 decoder 权重键（{bad_missing[:5]}）——该 ckpt 不是 C 臂 v2 产物"
            "（adapter/base 不在 model_state_dict 内），拒绝出数"
        )
    n_adapter_keys = sum(1 for k in sd if isinstance(k, str) and ".adapters." in k)
    if n_adapter_keys == 0:
        raise RuntimeError(
            "ckpt 内 adapters.* 键数为 0——adapter 权重不在 model_state_dict 内，"
            "该 ckpt 不是挂 LoRA 的 C 臂产物，拒绝出数"
        )
    return int(n_adapter_keys)


def _load_token_fidelity_windows(path: str):
    """读 token 保真抽测 npz（E5/SF-6）：键名自动探测 + 形状规整。

    - **token 键**：优先含 "token" 的键，其次含 "window" 的键；数组须 (N,64)（视为 T=1）
      或 (N,T,64) ⇒ 展平为 (N·T, 64)（decoder 的 token 输入契约）。
    - **proprio 键**（可选）：含 "proprio" 的键；(N,930) 按窗复制到 T 帧，(N,T,930) 原样展平；
      缺键 ⇒ 返回 None（调用方用零 proprio 并在报告注明）。
    返回 (tokens (M,64) float32, proprio (M,930) float32 | None, meta dict)。
    """
    import numpy as np  # noqa: WPS433 —— 仅探测路径需要

    slp = _policy_module()
    z = np.load(str(path))
    keys = [str(k) for k in z.files]
    tok_key = next((k for k in keys if "token" in k.lower()), None)
    if tok_key is None:
        tok_key = next((k for k in keys if "window" in k.lower()), None)
    if tok_key is None:
        raise ValueError(f"token 保真 npz 里找不到 token 窗口键（现有键 {keys}）——键名须含 token/window 字样")
    arr = np.asarray(z[tok_key], dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    if arr.ndim != 3 or int(arr.shape[-1]) != int(slp.TOKEN_DIM):
        raise ValueError(
            f"token 数组须 (N,{slp.TOKEN_DIM}) 或 (N,T,{slp.TOKEN_DIM})，收到 {tuple(arr.shape)}"
        )
    n_win, t_len = int(arr.shape[0]), int(arr.shape[1])
    if n_win < 1 or t_len < 1:
        raise ValueError(f"token 数组窗数/帧数非法：{tuple(arr.shape)}")
    tokens = arr.reshape(n_win * t_len, int(slp.TOKEN_DIM))
    prop_key = next((k for k in keys if "proprio" in k.lower()), None)
    prop = None
    if prop_key is not None:
        p = np.asarray(z[prop_key], dtype=np.float32)
        if p.ndim == 2 and tuple(p.shape) == (n_win, int(slp.PROPRIO_DIM)):
            p = np.repeat(p[:, None, :], t_len, axis=1)
        elif p.ndim == 3 and tuple(p.shape) == (n_win, t_len, int(slp.PROPRIO_DIM)):
            pass
        else:
            raise ValueError(
                f"proprio 数组须 ({n_win},{slp.PROPRIO_DIM}) 或 ({n_win},{t_len},{slp.PROPRIO_DIM})，"
                f"收到 {tuple(p.shape)}"
            )
        prop = p.reshape(n_win * t_len, int(slp.PROPRIO_DIM))
    meta = {"token_key": tok_key, "proprio_key": prop_key, "keys": keys,
            "n_windows": n_win, "window_len": t_len}
    return tokens, prop, meta


def _token_fidelity_report(probe_npz, mu_path, base_decoder, torch, device=None) -> dict:
    """E5/SF-6：token 窗口过 μ 路径，报 **adapter 引起的输出 delta**（μ_lora − μ_base）。

    - 组装 (M,994) = cat([token(64), proprio(930)])；proprio 取自 npz（含 proprio 键）
      或**零 proprio**（npz 无该键 = 合成站立 proprio 未提供，已在 `proprio_source` 注明；
      零 proprio 的 gravity 通道为 0 而非站立帧的 [0,0,-1]，故本探针只用于「adapter 输出
      扰动幅度」而非 decoder 精度判据）；
    - 两条路径吃同一 994：μ_lora = `mu_path.decoder.decode_obs`（含当前包络的 adapter）、
      μ_base = `base_decoder.decode_obs`（同结构、零 adapter = 未适配 base）；
    - `delta_absmax/absmean` = |μ_lora − μ_base| 的 max/mean（判据②的抽测仪表：adapter
      非零而 delta==0 ⇒ adapter 未生效）。
    """
    slp = _policy_module()
    tokens, proprio, meta = _load_token_fidelity_windows(probe_npz)
    if proprio is None:
        proprio_source = "zeros（npz 无 proprio 键：合成站立 proprio 未提供，零 proprio 仅作扰动探针）"
        prop_t = torch.zeros(int(tokens.shape[0]), int(slp.PROPRIO_DIM), dtype=torch.float32)
    else:
        proprio_source = f"npz:{meta['proprio_key']}"
        prop_t = torch.as_tensor(proprio, dtype=torch.float32)
    tok_t = torch.as_tensor(tokens, dtype=torch.float32)
    if device is not None:
        tok_t = tok_t.to(device)
        prop_t = prop_t.to(device)
    obs = torch.cat([tok_t, prop_t], dim=1)
    with torch.no_grad():
        mu_lora = mu_path.decoder.decode_obs(obs)
        mu_base = base_decoder.decode_obs(obs)
    delta = (mu_lora - mu_base).abs()
    return {
        "status": "probed",
        "probe_npz": str(probe_npz),
        "token_key": meta["token_key"],
        "proprio_key": meta["proprio_key"],
        "proprio_source": proprio_source,
        "npz_keys": meta["keys"],
        "n_windows": meta["n_windows"],
        "window_len": meta["window_len"],
        "n_frames": int(tok_t.shape[0]),
        "envelope": float(getattr(mu_path.decoder, "lora_scale", float("nan"))),
        "delta_absmax": float(delta.max().item()),
        "delta_absmean": float(delta.mean().item()),
        "mu_lora_absmax": float(mu_lora.abs().max().item()),
        "mu_base_absmax": float(mu_base.abs().max().item()),
        "note": "|μ_lora − μ_base|（同一 (M,994) 输入；adapter 对解码输出的扰动幅度，"
                "遗忘判据②的抽测仪表）",
    }


def _first_token_fidelity(results: list[dict]):
    """跨条件取首个非 not_probed 的 token 保真报告（E5）；全缺 ⇒ "not_probed"。"""
    for r in results:
        if isinstance(r, dict) and isinstance(r.get("lora"), dict):
            tf = r["lora"].get("token_fidelity")
            if tf is not None and tf != "not_probed":
                return tf
    return "not_probed"


def _build_policy_for_eval(cli, env, obs_probe, device, hv, tl):
    """v2：构造 `SonicLoRAPolicy`（decoder 在 policy 内）+ 恢复 ckpt 权重 + μ 路径遗忘检查。

    - 结构：token head(64) → token 仿射 → 994（token + env 侧 930 本体观测）→ LoRA decoder → μ(29)；
      rank/alpha 取 ckpt 内 `lora` 信封（`_resolve_lora_envelope`），缺则 CLI 默认；
    - 权重：`load_state_dict(model_state_dict, strict=False)`（adapter/base/head/critic/σ 全在内），
      decoder.* 缺键即报错（评的不是 C 臂）；
    - **遗忘检查（E1，--lora-scale 0）**：置零包络后的 μ 必须与「未适配 base decoder 的 μ」
      逐位一致（同 head 输出 + 同 obs）——不一致即抛异常 → EVAL_COND_FAIL，拒绝出数；
    - 同时报 scale=1 的 adapter 真实效应（==0 ⇒ adapter 未生效，附告警行）。
    """
    import torch  # noqa: WPS433 —— 子进程内 torch（App 起来后）

    slp = _policy_module()
    layout = tl._obs_layout(env, torch)
    onnx_path = cli.onnx_path or tl.SONIC_DECODER_ONNX_DEFAULT
    envelope, envelope_src = _resolve_lora_envelope(cli, torch, tl)
    rank = int(envelope.get("rank", cli.rank))
    alpha = float(envelope.get("alpha", cli.alpha))
    token_mean, token_std = slp.load_token_stats(cli.token_stats, device=device)
    token_alpha = float(cli.token_alpha) if cli.token_alpha is not None else 1.0
    token_bound = cli.token_bound or "tanh"

    obs_dim = int(layout["obs_dim"])
    critic_dim = int(getattr(env, "num_privileged_obs", 0) or layout["official_obs_dim"])
    decoder = slp.build_decoder_from_onnx(onnx_path, rank=rank, alpha=alpha, device=device)
    policy = slp.SonicLoRAPolicy(
        num_actor_obs=obs_dim,
        num_critic_obs=critic_dim,
        num_actions=29,
        actor_hidden_dims=[512, 256, 128],   # = train 侧 RslRlPpoActorCriticCfg（g1_agents_ppo.py:18-23）
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        official_obs_dim=layout["official_obs_dim"],
        proprio_dim=slp.PROPRIO_DIM,
        decoder=decoder,
        token_mean=token_mean,
        token_std=token_std,
        token_alpha=token_alpha,
        token_bound=token_bound,
        trainable="policy+adapter",  # 评测不训练；只影响 requires_grad，不影响 det 推理
    ).to(device)
    payload = torch.load(cli.ckpt, map_location=device, weights_only=False)
    sd = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    policy.eval()
    n_adapter_keys = _ckpt_c_arm_key_guard(sd, missing)  # SF-2：非 C 臂 ckpt 拒绝出数
    if missing or unexpected:
        print(
            f"[WARN] policy.load_state_dict 非严格匹配：missing={list(missing)[:6]} "
            f"unexpected={list(unexpected)[:6]}",
            flush=True,
        )

    # ---- 遗忘检查（μ 路径；探针 = 调用方给的真实 env obs）----
    with torch.no_grad():
        head_out = policy.actor(obs_probe[:, : policy.official_obs_dim])
        base_decoder = slp.build_decoder_from_onnx(onnx_path, rank=rank, alpha=alpha, device=device)
        base_mu_path = slp.SonicLoRAMuPath(
            official_obs_dim=policy.official_obs_dim,
            decoder=base_decoder,
            token_mean=token_mean,
            token_std=token_std,
            token_alpha=token_alpha,
            token_bound=token_bound,
        )
        mu_base = base_mu_path(head_out, obs_probe)
        policy.mu_path.set_lora_scale(0.0)
        mu_zero = policy.mu(obs_probe)
        policy.mu_path.set_lora_scale(1.0)
        mu_one = policy.mu(obs_probe)
        policy.mu_path.set_lora_scale(float(cli.lora_scale))
        mu_req = policy.mu(obs_probe)
    maxdiff_zero = float((mu_zero - mu_base).abs().max().item())
    bitwise_zero = bool(torch.equal(mu_zero, mu_base))
    maxdiff_one = float((mu_one - mu_base).abs().max().item())
    report = {
        "arch": "in_graph_v2",
        "policy_class": type(policy).__name__,
        "payload_source": envelope_src,
        "payload_found": bool(envelope),
        "rank": rank,
        "alpha": alpha,
        "lora_scale_requested": float(cli.lora_scale),
        "base_weight_md5": policy.mu_path.base_weight_md5(),
        "official_obs_dim": int(policy.official_obs_dim),
        "proprio_dim": int(policy.proprio_dim),
        "state_dict": {"n_missing": len(missing), "n_unexpected": len(unexpected),
                       "missing_head": list(missing)[:6], "unexpected_head": list(unexpected)[:6],
                       "ckpt_adapter_keys": int(n_adapter_keys)},
        "envelope_from_payload": {
            "rank": envelope.get("rank"),
            "alpha": envelope.get("alpha"),
            "sigma": envelope.get("sigma"),
            "trainable_mode": envelope.get("trainable_mode"),
            "git_commit": envelope.get("git_commit"),
            "seed": envelope.get("seed"),
            "onnx_md5": envelope.get("onnx_md5"),
            "construction_identity": envelope.get("construction_identity"),
        },
        "forgetting_check": {
            "probe": "env obs（policy 组）上的 μ；base = 未适配 base decoder 的同 head 输出",
            "maxdiff_scale0_vs_base": maxdiff_zero,
            "bitwise_scale0_vs_base": bitwise_zero,
            "maxdiff_requested_vs_base": float((mu_req - mu_base).abs().max().item()),
            "assertion": "maxdiff==0（预注册 §4 遗忘检查①）",
        },
        "adapter_effect": {
            "maxdiff_scale1_vs_base": maxdiff_one,
            "effective": bool(maxdiff_one > 0.0),
            "note": "0 ⇒ adapter 未生效/未训练（v2 下应非 0；若为 0 说明 adapter 权重恒 0）",
        },
    }
    # ---- E5/SF-6：token 保真抽测（遗忘判据②的仪表；未提供时显式降级）----
    if cli.token_fidelity_probe:
        report["token_fidelity"] = _token_fidelity_report(
            cli.token_fidelity_probe, policy.mu_path, base_decoder, torch, device=device
        )
        tf = report["token_fidelity"]
        print(
            f"[d065-eval] token 保真抽测: n_windows={tf['n_windows']} T={tf['window_len']} "
            f"proprio={tf['proprio_source']} delta(max/mean)={tf['delta_absmax']:.3e}/{tf['delta_absmean']:.3e}",
            flush=True,
        )
    else:
        report["token_fidelity"] = "not_probed"
        print(
            "[d065-eval] token 保真抽测未提供 --token-fidelity-probe ⇒ token_fidelity=not_probed"
            "（预注册 §4 遗忘判据②降级显式化）",
            flush=True,
        )
    print(
        f"[d065-eval] LoRA policy: class={type(policy).__name__} rank={rank} alpha={alpha} "
        f"scale={cli.lora_scale} official_obs_dim={policy.official_obs_dim} "
        f"envelope={envelope_src} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    print(
        f"[d065-eval] LoRA 遗忘检查(μ 路径): maxdiff(scale0 vs base)={maxdiff_zero:.3e} bitwise={bitwise_zero} "
        f"| adapter 效应 maxdiff(scale1 vs base)={maxdiff_one:.3e}",
        flush=True,
    )
    if not bitwise_zero:
        raise RuntimeError(
            f"遗忘检查失败：lora_scale=0 后的 μ 与未适配 base 的 μ 不逐位一致（maxdiff={maxdiff_zero:.3e}）"
            "——预注册 §4 判据①不成立，拒绝出数"
        )
    if maxdiff_one == 0.0:
        print(
            "LORA_EFFECT_WARN adapter 对 μ 无任何影响（maxdiff(scale1 vs base)==0）："
            "adapter 权重恒 0（未收到梯度或载荷全零）⇒ 本 ckpt 的 C 臂等价于「无适配器的 A+decoder」",
            flush=True,
        )
    return policy, report


def _stamp_command(env, vx: float):
    """把命令钉死为 (vx, 0, 0)，返回本步命令的 vx 列（供逐步采样）。

    D064 教训（预注册 §7.2）：父本只在 reset 后戳一次，实测 cmd_vx_mean ≠ 标称且随条件
    漂移 = resampling 覆写 `vel_command_b`。本 fork 改为**每个控制步重戳**（通道正常时
    是幂等的 no-op，被覆写时立刻纠正），并把每步命令采样落盘做通道核验（E3）。
    """
    import torch  # noqa: WPS433

    term = env.command_manager.get_term("base_velocity")
    n = term.vel_command_b.shape[0]  # 冒烟核对：UniformVelocityCommand.vel_command_b（2.1.0）
    term.vel_command_b[:] = torch.tensor([vx, 0.0, 0.0], device=term.vel_command_b.device).repeat(n, 1)
    return term.vel_command_b[:, 0].clone()


def _policy_obs_cmd_slot(env, torch) -> dict:
    """定位 policy obs 组里 `velocity_commands` 的切片（SF-5 的采样槽）。

    依据（静态 + 运行期双保险）：
    - **静态**：policy 组 = `velocity_env_cfg.ObservationsCfg.PolicyCfg` 的 term 序
      （tmp/isaac_ref/velocity_env_cfg.py:124-139：base_lin_vel, base_ang_vel,
      projected_gravity, **velocity_commands**, joint_pos, joint_vel, actions, height_scan）
      + C 臂在**末位**追加 `sonic_proprio`（g1_velocity_decoder_env.py:358-366 的
      `LoRAPolicyObservationsCfg.PolicyCfg`）⇒ velocity_commands 预期 index=3、start=9、
      width=3（3+3+3 之后）。该 term 即 `mdp.generated_commands(command_name="base_velocity")`
      （g1_velocity_decoder_env.py:268 的 critic 同项 + 官方 policy 同项）。
    - **运行期**：term 序 = `observation_manager.active_terms[group]` 顺序（拼接序），
      逐 term 维度乘积累加得起点——与 train 侧 `_obs_layout`（train_g1_decoder_lora.py:414-421）
      同一读法，不硬编码索引；term 序若漂移，此处随之漂移并在下方显式报错而非静默错位。

    采样链的独立性说明（防「断言自己刚写的值」）：该槽取值 = obs 项 `mdp.generated_commands`
    读命令项的 `command` 属性（2.1.0 的 `UniformVelocityCommand.command` = `vel_command_b`），
    而 `_stamp_command` 写的是同一个 `vel_command_b` ⇒ 本采样验证的是「写入 → 命令项 → obs
    计算 → 策略输入」这段**管线**（含 resampling 覆写、obs 槽位错位、噪声项），比「读回自己
    刚写的张量」强得多；但两条通道同源，**真正独立于命令通道的证据是 `actual_vx_*`**
    （实测机体速度，D064 教训的原始观察面），两者需并读。
    """
    om = env.observation_manager
    names = [str(n) for n in om.active_terms["policy"]]
    dims = list(om.group_obs_term_dim["policy"])
    if len(names) != len(dims):
        raise RuntimeError(f"policy 组 names/dims 错位：{len(names)} vs {len(dims)}")
    if "velocity_commands" not in names:
        raise RuntimeError(
            f"policy 组无 velocity_commands 项（现有 {names}）——命令通道核验无处采样，拒绝出数"
        )
    idx = names.index("velocity_commands")
    start = 0
    for term_dims in dims[:idx]:
        start += int(torch.tensor(list(term_dims)).prod())
    width = int(torch.tensor(list(dims[idx])).prod())
    if width < 1:
        raise RuntimeError(f"velocity_commands 槽宽 {width} 非法（dims={dims[idx]}）")
    return {"term": "velocity_commands", "index": int(idx), "start": int(start),
            "width": int(width), "policy_terms": names}


def _obs_cmd_vx(obs, slot):
    """从 policy obs 里切出命令槽的 vx 列（(N,)）——SF-5 的采样口径（不是刚写进去的值）。"""
    return obs[:, slot["start"]: slot["start"] + slot["width"]][:, 0]


# ---------------------------------------------------------------------------
# D065 LoRA 段（E1 遗忘检查 / E2 载荷装载）
# ---------------------------------------------------------------------------
def _sibling_module():
    """train_g1_decoder_lora 模块本体（复用其 LoRA 接线与载荷读写，避免双份漂移）。"""
    try:
        from isaac import train_g1_decoder_lora as tl
    except ImportError:  # 仓根包 import
        from apt_g1.isaac import train_g1_decoder_lora as tl
    return tl


def _policy_module():
    """`sonic_lora_policy` 模块本体（v2 策略类与 μ 路径；双路径 import）。"""
    try:
        from isaac import sonic_lora_policy as slp
    except ImportError:  # 仓根包 import
        from apt_g1.isaac import sonic_lora_policy as slp
    return slp


def _resolve_lora_envelope(cli, torch, tl) -> tuple[dict, str]:
    """LoRA 信封来源：--lora-ckpt > ckpt 内 `lora` 键 > ckpt 同目录侧车。

    v2 的 adapter 权重本身在 ckpt 的 `model_state_dict` 内（decoder 属 policy 子模块），
    这里只为取 rank/alpha/σ/trainable 等结构超参（缺则回落 CLI 默认 16/1.0）。
    """
    if cli.lora_ckpt:
        payload = tl._load_lora_payload(None, Path(cli.lora_ckpt), torch)
        src = f"--lora-ckpt({cli.lora_ckpt})"
    else:
        sidecar = Path(cli.ckpt).resolve().parent / LORA_ADAPTER_FILENAME
        payload = tl._load_lora_payload(cli.ckpt, sidecar, torch)
        src = f"ckpt.lora 键 / 侧车({sidecar.name})"
    if not isinstance(payload, dict):
        return {}, src + "（未命中，回落 CLI --rank/--alpha）"
    envelope = payload.get("envelope", payload)
    return (envelope if isinstance(envelope, dict) else {}), src


def _run_batches(env, policy, cli, cond, hv) -> dict:
    """批次 rollout：每 env 记 首次 terminated 步/净位移/vx RMSE（首次 done 后不计）。

    D065 E3/SF-5：flat 条件每控制步重戳命令，**核验采样走策略实际消费的 obs 命令槽**
    （`_policy_obs_cmd_slot` + `_obs_cmd_vx`；不是刚写进去的值）；diag 增 `obs_cmd_vx_*`
    与 `actual_vx_*`（含 near_static 标志）。
    """
    torch = hv.torch
    device = env.device
    n = cli.num_envs
    episodes: list[dict] = []
    cmd_hist: list[float] = []  # 每控制步「戳写后」的命令均值（写通道自检，保留对照）
    obs_cmd_hist: list[float] = []  # SF-5：策略实际消费的 obs 命令槽采样（判据用）
    obs_cmd_step0: list[float] = []  # 第 0 步 obs 来自 env.reset()（命令重采样在戳写之前）
    slot = _policy_obs_cmd_slot(env, torch) if cond["kind"] == "flat" else None
    for _batch in range(cli.episodes):
        obs, _ = env.reset()
        if cond["kind"] == "flat":
            _stamp_command(env, cond["vx"])
        robot = env.scene["robot"].data
        root0 = robot.root_pos_w[:, :2].clone()
        alive = torch.ones(n, dtype=torch.bool, device=device)
        fall_step = torch.full((n,), -1, dtype=torch.long, device=device)
        # IsaacLab 的 step() 返回前就对 done env 完成自动复位（apt_flat_env.py:1640-1662 /
        # eval_apt_isaac.py:541-547 的 D048e 教训）——终末量绝不能读 step 之后的 robot.data。
        # 策略=「步前缓存」：done 者定格步前位置（滞后一个控制步 20ms，可接受）；未 done 者滚动更新。
        root_end = root0.clone()
        vx_sq_sum = torch.zeros(n, device=device)
        vx_cnt = torch.zeros(n, device=device)
        vx_sum = torch.zeros(n, device=device)  # 诊断：命令通道核验（D048h 纪律——指标身份先于机制归因）
        obs_p = obs["policy"]
        for step in range(cli.ep_steps):
            alive_pre = alive.clone()
            pre_pos = robot.root_pos_w[:, :2].clone()
            pre_vel = robot.root_lin_vel_b[:, 0].clone()
            if cond["kind"] == "flat":
                # SF-5：核验采样 = 策略本步实际消费的 obs（obs_p）里的命令槽。
                # 第 0 步的 obs 来自 env.reset()（reset 内命令重采样发生在戳写之前、obs 未重算）
                # ⇒ 单独记录、不参与判据；step>=1 的 obs 含上一步戳写值（resampling_time_range
                # 已拉到 1e9，命令项不再重采样）。done/reset 行在 reset 时被重采样（D064 现象），
                # 只对仍存活的行断言（alive_pre）。
                obs_cmd = _obs_cmd_vx(obs_p, slot)
                if step == 0:
                    obs_cmd_step0.append(float(obs_cmd.mean().item()))
                elif bool(alive_pre.any()):
                    obs_cmd_hist.extend(float(v) for v in obs_cmd[alive_pre].tolist())
                # E3：每控制步重戳（D064 教训：只在 reset 后戳会被 resampling 覆写）
                cmd_hist.append(float(_stamp_command(env, cond["vx"]).mean().item()))
            with torch.no_grad():
                action = policy.act_inference(obs_p)  # det：μ（29 维关节目标；C 臂 v2 = A 臂动作空间）
            obs, _, terminated, truncated, _ = env.step(action)
            obs_p = obs["policy"]
            done = terminated | truncated
            newly_fallen = alive_pre & terminated   # truncated=到时不算摔
            fall_step[newly_fallen] = step
            root_end[newly_fallen] = pre_pos[newly_fallen]             # 首摔定格（步前位置）
            root_end[alive_pre & done & ~terminated] = pre_pos[alive_pre & done & ~terminated]  # 到时定格
            keep = alive_pre & ~done
            root_end[keep] = robot.root_pos_w[:, :2][keep]             # 未复位者滚动更新
            if cond["kind"] == "flat":
                m = keep  # 摔步/到时步不计入 RMSE（摔步速度若计入=复位后≈0 的伪样本）
                vx_sq_sum[m] += (pre_vel[m] - cond["vx"]) ** 2
                vx_sum[m] += pre_vel[m]
                vx_cnt[m] += 1
            alive = keep
            if not bool(alive.any()):
                break
        disp = (root_end - root0).norm(dim=1)
        disp_x = root_end[:, 0] - root0[:, 0]
        for i in range(n):
            episodes.append(
                {
                    "survived": bool(fall_step[i].item() < 0),
                    "fall_step": int(fall_step[i].item()) if fall_step[i].item() >= 0 else None,
                    "disp": round(float(disp[i].item()), 3),
                    "disp_x": round(float(disp_x[i].item()), 3),
                    "vx_rmse": round(float((vx_sq_sum[i] / vx_cnt[i]).sqrt().item()), 4)
                    if cond["kind"] == "flat" and vx_cnt[i].item() > 0
                    else None,
                }
            )
    surv = sum(e["survived"] for e in episodes)
    agg = {
        "kind": cond["kind"],
        "cond": cond,
        "n_episodes": len(episodes),
        "survival_rate": surv / len(episodes),
        "mean_disp_x": sum(e["disp_x"] for e in episodes) / len(episodes),
        "mean_vx_rmse": (
            sum(e["vx_rmse"] for e in episodes if e["vx_rmse"] is not None) / max(1, len(episodes))
            if cond["kind"] == "flat"
            else None
        ),
        "episodes": episodes,
    }
    if cond["kind"] == "flat":
        # 诊断（命令通道核验，E3）：命令是否真的钉在 vx、实际速度是否≈0——区分
        # 「策略没学会走」与「评测命令通道失效」两种解释，判定前必读。
        cmd_term = env.command_manager.get_term("base_velocity")
        nominal = float(cond["vx"])
        cmd_mean = sum(cmd_hist) / len(cmd_hist) if cmd_hist else None
        cmd_var = (
            sum((v - cmd_mean) ** 2 for v in cmd_hist) / len(cmd_hist) if cmd_hist else None
        )
        obs_mean = sum(obs_cmd_hist) / len(obs_cmd_hist) if obs_cmd_hist else None
        obs_var = (
            sum((v - obs_mean) ** 2 for v in obs_cmd_hist) / len(obs_cmd_hist) if obs_cmd_hist else None
        )
        actual_per_env = [
            round(float((vx_sum[i] / vx_cnt[i]).item()), 4) if vx_cnt[i].item() > 0 else None
            for i in range(n)
        ]
        actual_valid = [x for x in actual_per_env if x is not None]
        actual_mean = sum(actual_valid) / len(actual_valid) if actual_valid else None
        agg["diag"] = {
            "cmd_vx_nominal": nominal,
            "cmd_vx_mean_last_batch": float(cmd_term.vel_command_b[:, 0].mean().item()),
            "cmd_vx_mean_over_steps": cmd_mean,
            "cmd_vx_min_over_steps": min(cmd_hist) if cmd_hist else None,
            "cmd_vx_max_over_steps": max(cmd_hist) if cmd_hist else None,
            "cmd_vx_std_over_steps": (cmd_var ** 0.5) if cmd_var is not None else None,
            "cmd_steps_sampled": len(cmd_hist),
            # ---- SF-5：判据采样 = 策略实际消费的 obs 命令槽 ----
            "obs_cmd_slot": {"term": slot["term"], "index": slot["index"], "start": slot["start"],
                             "width": slot["width"]},
            "obs_cmd_vx_n_samples": len(obs_cmd_hist),
            "obs_cmd_vx_mean": obs_mean,
            "obs_cmd_vx_min": min(obs_cmd_hist) if obs_cmd_hist else None,
            "obs_cmd_vx_max": max(obs_cmd_hist) if obs_cmd_hist else None,
            "obs_cmd_vx_std": (obs_var ** 0.5) if obs_var is not None else None,
            "obs_cmd_vx_step0_mean": (sum(obs_cmd_step0) / len(obs_cmd_step0)) if obs_cmd_step0 else None,
            "cmd_matches_nominal": bool(
                obs_cmd_hist and all(abs(v - nominal) <= CMD_NOMINAL_TOL for v in obs_cmd_hist)
            ),
            "actual_vx_mean_per_env": actual_per_env,
            "actual_vx_mean_all_envs": actual_mean,
            "actual_vx_near_static": (abs(actual_mean) < NEAR_STATIC_VX) if actual_mean is not None else None,
            "note": "obs_cmd_* = 策略消费的 obs 命令槽采样（SF-5 判据口径；第 0 步 obs 出自 "
            "env.reset() 不参与、done/reset 行排除）；cmd_* = 戳写路径采样（写通道自检）；"
            "actual_* = 未 done 步的 root_lin_vel_b[:,0]",
        }
        print(
            f"[d065-eval][diag] flat vx={nominal} obs_cmd_mean={obs_mean} "
            f"obs_cmd[min,max]=[{agg['diag']['obs_cmd_vx_min']},{agg['diag']['obs_cmd_vx_max']}] "
            f"obs_cmd_n={agg['diag']['obs_cmd_vx_n_samples']} "
            f"matches_nominal={agg['diag']['cmd_matches_nominal']} "
            f"cmd_last_batch={agg['diag']['cmd_vx_mean_last_batch']} "
            f"cmd_stamp_mean={cmd_mean} "
            f"actual_vx_mean={actual_mean} near_static={agg['diag']['actual_vx_near_static']} "
            f"actual_vx_mean(env0-2)={actual_per_env[:3]}",
            flush=True,
        )
    return agg


def _eval_one_main(cli, launcher_args) -> int:
    """--eval-one 子命令体：起 env → 批次 rollout → 打一行 EVAL_COND <json>。exit 0/1。"""
    cond = json.loads(cli.eval_one)
    hv = _import_heavy()  # 同款重型 import 入口（train_g1_decoder.py:496-560 的镜像，见下）
    torch = hv.torch
    device = getattr(launcher_args, "device", "cuda:0")
    print(f"[d065-eval] child: cond={cond} arm={cli.arm} pid={os.getpid()} lora_scale={cli.lora_scale}", flush=True)
    env = None
    try:
        kw = dict(
            # flat 也建 rough cfg 再换平面几何：ckpt 训练于 rough（policy obs 含 height_scan），
            # plane cfg 的 obs 维会让 head 首层 size mismatch（12269 实证）；换几何后
            # height_scan 在平地上读 ≈0 = flat 语义，奖励项在评测中不被消费。
            terrain="rough",
            num_envs=cli.num_envs,
            action=cli.arm,  # "lora_policy"：A 臂动作通路逐字 + policy 组末位 930 维本体观测
            seed=cli.seed,
        )
        # token/onnx 参数不进 env（v2 里它们在 policy 内，见 _build_policy_for_eval）
        cfg = hv.make_env_cfg(**kw)
        if cond["kind"] == "rough_levels":
            # 出生行固定：关课程（多行假设不成立时直写 terrain_levels 语义更清晰）+ reset 前写行号
            cfg.curriculum.terrain_levels = None  # 冒烟核对：configclass 置 None 是否等效删 term
        if cond["kind"] == "rough_paper":
            from isaac.terrain_cfg import make_terrain_importer_cfg  # 双路径见 _import_heavy 同款处理

            cfg.scene.terrain = make_terrain_importer_cfg(
                "rough_paper", noise=cond["noise"], seed=cli.seed
            )
            cfg.curriculum.terrain_levels = None
            _gen = getattr(cfg.scene.terrain, "terrain_generator", None)
            if _gen is not None and hasattr(_gen, "curriculum"):
                _gen.curriculum = False  # 与工厂 stairs/stones/discrete 分支对齐（同构不漂移）
        if cond["kind"] == "flat":
            from isaac.terrain_cfg import make_terrain_importer_cfg  # 双路径见 _import_heavy 同款处理

            # flat=rough cfg + 平面几何（保 obs 321，见上 kw 注释）
            cfg.scene.terrain = make_terrain_importer_cfg("plane", seed=cli.seed)
            cfg.curriculum.terrain_levels = None
            _gen_f = getattr(cfg.scene.terrain, "terrain_generator", None)
            if _gen_f is not None and hasattr(_gen_f, "curriculum"):
                _gen_f.curriculum = False
            # 命令钉死第一步：重采样区间拉至无穷大（cfg 侧），批次内再戳 vel_command_b
            cmd_cfg = cfg.commands.base_velocity
            cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)  # 冒烟核对：term cfg 字段名
        # B2：落地 60s 口径——env 继承官方 episode_length_s=20s，不覆写则 ~1000 步即全体 truncated
        cfg.episode_length_s = cli.ep_steps / 50.0
        env = hv.ManagerBasedRLEnv(cfg=cfg)
        mel = getattr(env, "max_episode_length", None)
        ctrl_dt = (getattr(cfg, "decimation", None), getattr(cfg.sim, "dt", None))
        print(
            f"[d065-eval] episode_length_s={cfg.episode_length_s} max_episode_length={mel} "
            f"(decimation,sim_dt)={ctrl_dt}（60s 口径=ep_steps/50Hz）",
            flush=True,
        )
        if mel is not None:
            assert int(mel) == cli.ep_steps, (
                f"max_episode_length={mel} != ep_steps={cli.ep_steps}（60s 口径未落地，判据失效，拒绝继续）"
            )
        if cond["kind"] == "rough_levels":
            lv = cond["level"]
            # 官方写法=env.scene.terrain（gear_sonic/envs/manager_env/mdp/curriculum.py:53-66）；
            # update_env_origins 是课程 ±1 行 API 不可借用，改用无参重算 origins（冒烟核对：2.1.0 名/元数）
            terrain_imp = env.scene.terrain
            terrain_imp.terrain_levels[:] = lv  # 冒烟核对：TerrainImporter.terrain_levels（2.1.0）
            terrain_imp.configure_env_origins()  # 冒烟核对：无参重算 env_origins（2.1.0 API 名/元数）
            # SF1 读回保护：若该 API 会按 max_init_terrain_level 重采样行号，这里立刻响（防静默错行失真主指标①）
            lv_back = terrain_imp.terrain_levels
            assert bool((lv_back == lv).all()), f"出生行未固定：期望 {lv}，实得 {lv_back.tolist()}"
            print(f"[d065-eval] rough_levels fixed row={lv} confirmed", flush=True)
        obs, _ = env.reset()
        action_dim = int(env.action_manager.total_action_dim)
        # ---- D065 v2 E1/E2：policy 构造（decoder 在 policy 内）+ 遗忘检查（rollout 之前；
        #      失败即 EVAL_COND_FAIL，不静默降级）----
        tl = _sibling_module()
        policy, lora_report = _build_policy_for_eval(cli, env, obs["policy"], device, hv, tl)
        agg = _run_batches(env, policy, cli, cond, hv)
        agg["lora"] = lora_report
        print("EVAL_COND " + json.dumps(agg, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 —— 判读要可见：类型名必进 FAIL 行
        print(f"EVAL_COND_FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _import_heavy():
    """App 起来后的重型 import（复用 train_g1_decoder_lora._import_heavy 的组装，避免双份漂移）。

    D065 v2 fork：入口改指 LoRA 版训练模块（namespace 供 env 工厂/LoRA 信封读写用；
    策略类与 μ 路径直接走 `sonic_lora_policy` 模块，见 `_policy_module`）。
    """
    return _sibling_module()._import_heavy()


def main() -> None:
    cli = build_args().parse_args()
    if cli.eval_one is not None:
        # 子进程体：AppLauncher 链与 train 同款
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
            code = _eval_one_main(cli, launcher_args)
        finally:
            simulation_app.close()
        sys.exit(code)

    # ---- 父进程：展开条件 → 逐条件子进程 → 聚合 JSON ----
    conds = expand_conditions(cli)
    if not conds:
        build_args().error("battery 展开为空（检查 --levels/--noise/--vx-cmds）")
    ckpt_md5 = _file_md5(cli.ckpt)
    decoder_md5 = _file_md5(cli.onnx_path) if cli.arm == "decoder" and cli.onnx_path else None
    out_dir = Path(cli.output_dir) if cli.output_dir else Path(cli.ckpt).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(cli.ckpt).stem
    out_json = out_dir / f"d065_eval_{cli.arm}_{ckpt_name}.json"
    # LoRA 载荷身份（E2）：显式 --lora-ckpt 优先，否则 ckpt 同目录侧车（可能不存在=载荷在 ckpt 内）
    lora_payload_path = cli.lora_ckpt or str(Path(cli.ckpt).resolve().parent / LORA_ADAPTER_FILENAME)
    lora_payload_md5 = _file_md5(lora_payload_path) if Path(lora_payload_path).is_file() else None

    results: list[dict] = []
    for cond in conds:
        argv = _child_argv(cli, cond)
        print(f"[d065-eval] condition {cond} -> spawn", flush=True)
        try:
            # encoding 必须显式：C locale 下 text=True 默认 ASCII 解码遇 UTF-8 崩（train r5 实证）
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"kind": cond["kind"], "cond": cond, "error": f"spawn-failed: {type(exc).__name__}: {exc}"})
            continue
        out = err = ""
        timed_out = False
        try:
            out, err = proc.communicate(timeout=COND_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            timed_out = True
        if out:
            print(out, flush=True)  # capture-and-forward：tee 可见完整 Isaac 日志
        cond_res = None
        for line in out.splitlines():
            if line.startswith("EVAL_COND "):
                try:
                    cond_res = json.loads(line[len("EVAL_COND "):])
                except json.JSONDecodeError:
                    cond_res = None
        if cond_res is not None:
            results.append(cond_res)
        else:
            why = f"timeout>{COND_TIMEOUT_S}s" if timed_out else f"no-line (exit {proc.returncode})"
            for line in out.splitlines():
                if line.startswith("EVAL_COND_FAIL "):
                    why = line[len("EVAL_COND_FAIL "):]
                    break
            else:
                if err:  # 段错误类原生崩只有 exit code：stderr 尾部是唯一线索，转发助诊
                    print("[d065-eval] child stderr tail:\n" + "\n".join(err.splitlines()[-5:]), flush=True)
            results.append({"kind": cond["kind"], "cond": cond, "error": why})

    payload = {
        "identity": {
            "ckpt": str(Path(cli.ckpt).resolve()),
            "ckpt_md5": ckpt_md5,
            "decoder_onnx_md5": decoder_md5,
            "arm": cli.arm,
            "battery": cli.battery,
            "episodes": cli.episodes,
            "ep_steps": cli.ep_steps,
            "num_envs": cli.num_envs,
            "seed": cli.seed,
            "levels": cli.levels,
            "noise": cli.noise,
            "vx_cmds": cli.vx_cmds,
            "git_head": _git_head(),
            # ---- D065：C 臂身份（LoRA 载荷 + 包络 + 遗忘检查结论）----
            "lora_scale": cli.lora_scale,
            "lora_payload": lora_payload_path,
            "lora_payload_md5": lora_payload_md5,
            "lora": next((r["lora"] for r in results if isinstance(r, dict) and r.get("lora")), "absent"),
            # E5/SF-6：token 保真抽测结论（未提供 --token-fidelity-probe ⇒ 显式 not_probed）
            "token_fidelity_probe": cli.token_fidelity_probe,
            "token_fidelity": _first_token_fidelity(results),
        },
        "cmd_channel_check": _cmd_channel_check(results),
        "results": results,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    chk = payload["cmd_channel_check"]
    print(f"EVAL CMDCHANNEL {chk['verdict']} {json.dumps(chk, ensure_ascii=False)}", flush=True)
    bad = [r for r in results if "error" in r]
    if bad:
        print(f"EVAL FAIL: {len(bad)}/{len(results)} 条件失败 -> {out_json}", flush=True)
        sys.exit(1)
    print(f"EVAL DONE: {out_json}", flush=True)


if __name__ == "__main__":
    main()
