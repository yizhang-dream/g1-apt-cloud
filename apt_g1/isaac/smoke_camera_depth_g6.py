"""G6 深度接入第一步冒烟件：Isaac 相机能否渲染 / 单步渲染耗时 / 显存增量。

本脚本只回答三个问题（一次运行给三个数字，全部落 JSON + 单行机读行）：

  ① 渲染可行性：``enable_cameras=True`` 能否过 AppLauncher / env 构造 / 首 step，
     以及运行时挂载 TiledCamera(rgb+depth) 后能否取到 depth 帧。
     段错误复现即 FAIL，**不重试**（如实落账；这是本仓的历史事实，见下）。
  ② 单步渲染耗时：同进程内先跑 baseline（renderer 开、不挂相机）再挂相机跑第二
     相位，得到 ``render_overhead_ms``；另有 ``render_overhead_with_update_ms``
     （含显式 ``cam.update()`` 成本，见「设计决定 3」）。``--baseline-no-camera``
     只跑 baseline 相位（对照开关）。
  ③ 显存增量：两相位各自 ``reset_peak_memory_stats()`` 后的
     ``torch.cuda.max_memory_allocated()`` 差值；无 cuda 记 ``n/a``。

副产：若干帧 depth 统计（finite 占比 / 非有限值按 0 计入后的 min/mean/max / 非零占比）。

背景（为什么值得单独冒烟）
--------------------------
- 本仓有 ``enable_cameras=True`` 的 hydra 段错误史：``refine-logs/logs/ISAAC_APT_LOG.md``
  记 E29 阶段 ``render_walk.py`` 在本服务器开相机时于 stage 创建阶段段错误
  （viewport hydra ``__enable_hydra_engine``，崩在 AppLauncher 初始化、业务代码之前；
  重试复现；服务器历史上从无 Isaac 渲染产物）。此后 ``rollout_log_joints.py`` /
  ``render_walk.py`` 一律走相机无关路径或 MuJoCo 离屏渲染。
- 渲染开销在 4096 envs 档**零实测**（本仓所有 Isaac 训练都是 enable_cameras=False）。
  故本脚本给出可扫的 ``--num-envs``（1/8/64），默认 8。

用法（服务器 cvgl，cwd=仓根，EULA 由包装脚本 / 本脚本 export）
-------------------------------------------------------------
    bash /tmp/run_apt_isaac.sh \
        /home/cvgluser/ros2_data/apt_g1/isaac/smoke_camera_depth_g6.py --num-envs 8
    # 对照：不挂相机（renderer 仍开）
    bash /tmp/run_apt_isaac.sh .../smoke_camera_depth_g6.py --num-envs 8 --baseline-no-camera
    # 本机（无 isaaclab）：参数/统计/JSON/输出行自检，不 import isaaclab
    python apt_g1/isaac/smoke_camera_depth_g6.py --selftest
    # 输出：<out>/smoke_camera_depth_g6.json（--out 默认 outputs/g6/，相对路径锚定仓根）
    #       单行 "SMOKE_CAMERA PASS|FAIL ..."（供盯守代理 grep）；退出码 0/1 如实。

设计决定（每条附依据）
----------------------
1. 单进程两相位（baseline -> 运行时挂载相机），**不建第二个 env**：
   父进程内重建第二个 ``ManagerBasedRLEnv`` 在 Isaac Lab 是已知雷区
   （``train_g1_decoder.py:850-853``：主 env.close() 后场景/terrain 重建静默挂起，
   try/except 兜不住 hang）。运行时挂载是 ``render_walk.py:117-121`` 的既有先例
   （``env.scene._sensors[name] = TiledCamera(cfg)`` -> ``_initialize_impl()`` ->
   ``reset()``）。这样同一进程内就能得到 baseline/相机两相位的耗时与显存差值。
2. ``enable_cameras`` 恒 True（含 --baseline-no-camera）：① 问的就是
   enable_cameras=True 的可行性，且 baseline 相位要保持 renderer 开才能隔离出
   「相机 sensor」本身的开销，而不是「renderer 开没开」。
3. 相机 ``update()`` 显式调用（照 ``render_walk.py:159`` 先例），且把
   ``env.step`` 与 ``cam.update`` 分开计时：若 scene 亦自动更新则属重复渲染，
   故 ``render_overhead_ms``（仅 env.step）是下界、``..._with_update_ms`` 是
   含显式渲染的上界，两个都给，不藏。
4. 段错误不重试、不留假 PASS：AppLauncher 之前先落一份 preflight stub JSON
   （``verdict="PENDING"``），若进程在 stage 创建段错误（历史故障点），stub 即
   最终产物，配合 ``faulthandler`` 的 stderr traceback 与退出码定位。
5. 深度统计纯 Python（不引 numpy），帧内采样上限 ``MAX_DEPTH_VALUES``
   （64 envs × 192x108 = 1.33M 值/帧，全量统计太慢；按 stride 抽样并把
   ``n_total``/``stride`` 一并落账）。同一套纯函数由 ``--selftest`` 在本机覆盖。
6. 重型依赖全延迟 import：``--help``/``--selftest`` 在任何 isaaclab import 之前
   完成（同 ``train_g1_decoder.py:1000-1001`` 的编排）。

与 gear_sonic 相机配方的对照（``gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:849-889``）
------------------------------------------------------------------------------------------------
抄（结构/常量逐字段）:
  - ``TiledCameraCfg`` + ``spawn=sim_utils.PinholeCameraCfg(focal_length=1.88,
    focus_distance=0.5, horizontal_aperture=2.6035, vertical_aperture=1.4621,
    clipping_range=(0.1, 20.0))``（:849-855）；
  - ``OffsetCfg(pos=(0,0,0), rot=(1,0,0,0) wxyz, convention="world")``（:866-869）；
  - ``update_period=0.0`` + ``update_latest_camera_pose=True``（:887-888）；
  - prim 路径拼法 ``f"{{ENV_REGEX_NS}}/Robot/{attached_link}/ego_camera"``（:859-863）。
改（本仓事实）:
  - 挂载 link：``d435_link`` -> ``head_link``（``gear_sonic/data/assets/robot_description/
    urdf/g1/main.urdf:569`` 有 ``head_link``，无 ``d435_link``；模块常量
    ``CAMERA_PRIM_PATH``）；
  - ``data_types``：``["rgb"]`` -> ``["rgb", "depth"]``（本冒烟的对象就是 depth）；
  - ``debug_vis``：True -> False（headless 无 viewport）；
  - 分辨率由 ``--res`` 给（默认 192x108，即 WxH -> height=108, width=192，
    与 gear_sonic ``camera_resolution=[H, W]`` 的 108/192 同形，见 ``_parse_res``）；
  - 挂载方式：cfg 声明式（gear_sonic 在 cfg ``__init__`` 里 ``self.ego_camera = ...``）
    -> 运行时 ``env.scene._sensors`` 注入（设计决定 1，为拿同进程 baseline）。
    ``_camera_cfg`` 是独立纯函数，声明式接线的生产路径可直接复用它。

不改任何既有文件；SCRIPT_MAP 登记由主会话另办。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from importlib.metadata import version as _pkg_version
from pathlib import Path
from types import SimpleNamespace

# 仓根（本文件位于 <repo>/apt_g1/isaac/）
REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- CLI 默认值 -----------------------------------------------------------
DEFAULT_NUM_ENVS = 8
DEFAULT_DURATION_S = 30.0
DEFAULT_RES = "192x108"
DEFAULT_OUT = "outputs/g6"
# 可扫的 num_envs 档（仅文档用；不设 choices，允许任意值）
NUM_ENVS_SWEEP = (1, 8, 64)
TERRAINS = ("plane", "rough")

# ---- 相位参数 -------------------------------------------------------------
WARMUP_STEPS = 5
MAX_STEPS_PER_PHASE = 2000
DEPTH_EVERY = 20          # 每 N 步取一帧 depth 统计
MAX_DEPTH_FRAMES = 5
MAX_DEPTH_VALUES = 500_000  # 单帧统计的采样上限（超出按 stride 抽样）
ENV_SPACING = 4.0         # 带相机 env 的仓内惯例（render_walk.py:55 / smoke_isaac.py:35）

# ---- 相机（结构照 gear_sonic modular_tracking_env_cfg.py:849-889）---------
CAMERA_LINK = "head_link"
CAMERA_NAME = "ego_camera"
CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/head_link/ego_camera"
CAMERA_DATA_TYPES = ("rgb", "depth")
CAMERA_FOCAL_LENGTH = 1.88
CAMERA_FOCUS_DISTANCE = 0.5
CAMERA_HORIZONTAL_APERTURE = 2.6035
CAMERA_VERTICAL_APERTURE = 1.4621
CAMERA_CLIPPING_RANGE = (0.1, 20.0)
CAMERA_OFFSET_POS = (0.0, 0.0, 0.0)
CAMERA_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0)  # wxyz

# ---- EULA（栈惯例，HANDOFF/04_SERVER_GUIDE.md:43）-------------------------
EULA_ENV = {
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "ACCEPT_EULA": "Y",
    "PRIVACY_CONSENT": "Y",
}

JSON_NAME = "smoke_camera_depth_g6.json"


##
# 纯函数区（--selftest 全覆盖；不 import isaaclab/numpy/torch）
##


def _parse_res(res: str) -> tuple[int, int]:
    """``--res`` 字符串 -> (height, width)。

    约定：``WxH``（宽 x 高，相机分辨率的通行写法）。故默认 ``"192x108"``
    -> height=108, width=192，与 gear_sonic ``camera_resolution=[108, 192]``
    （``height=resolution[0], width=resolution[1]``，:875-885）同形。
    非法输入直接 ValueError（不静默取默认值）。
    """
    raw = str(res).strip().lower().replace(" ", "")
    parts = raw.split("x")
    if len(parts) != 2:
        raise ValueError(f"--res 应为 'WxH' 形式（如 192x108），实际 {res!r}")
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"--res 分量应为整数（'WxH'），实际 {res!r}") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"--res 分量应 > 0，实际 {res!r}")
    return h, w


def _res_tag(height: int, width: int) -> str:
    """(height, width) -> 回显用标签 'WxH'（与 --res 输入同向）。"""
    return f"{int(width)}x{int(height)}"


def depth_stats(values) -> dict:
    """depth 值的统计（纯 Python）。

    - ``finite_ratio``：``math.isfinite`` 占比（depth 里天空/远平面常为 inf，
      nan 亦计入非有限）。
    - 非有限值按 **0** 计入后再算 ``min`` / ``mean`` / ``max``（即「inf→0 后」）。
    - ``nonzero_ratio``：上述清洗后 != 0 的占比（反映有效深度覆盖）。
    - 空输入 -> 各统计项 None（``n=0``），不抛异常。
    浮点结果 round 到 6 位，便于 JSON 阅读。
    """
    vals = list(values)
    n = len(vals)
    if n == 0:
        return {
            "n": 0,
            "n_nonfinite": 0,
            "finite_ratio": None,
            "min": None,
            "mean": None,
            "max": None,
            "nonzero_ratio": None,
        }
    cleaned = []
    n_nonfinite = 0
    for v in vals:
        f = float(v)
        if math.isfinite(f):
            cleaned.append(f)
        else:
            n_nonfinite += 1
            cleaned.append(0.0)
    n_nonzero = sum(1 for v in cleaned if v != 0.0)
    return {
        "n": n,
        "n_nonfinite": n_nonfinite,
        "finite_ratio": round((n - n_nonfinite) / n, 6),
        "min": round(min(cleaned), 6),
        "mean": round(sum(cleaned) / n, 6),
        "max": round(max(cleaned), 6),
        "nonzero_ratio": round(n_nonzero / n, 6),
    }


def _frame_depth_stats(values, cap: int = MAX_DEPTH_VALUES) -> dict:
    """单帧 depth 统计：超过 ``cap`` 个值时按 stride 抽样，并落账 n_total/stride。"""
    total = len(values)
    if total == 0:
        out = {"n_total": 0, "stride": 1}
        out.update(depth_stats([]))
        return out
    stride = max(1, math.ceil(total / max(1, int(cap))))
    out = {"n_total": total, "stride": stride}
    out.update(depth_stats(values[::stride]))
    return out


def _depth_summary(frames: list[dict]) -> dict:
    """多帧 depth 统计的汇总（每帧一项 -> 跨帧极值/均值）。"""
    if not frames:
        return {
            "n_frames": 0,
            "mean_of_means": None,
            "min_of_mins": None,
            "max_of_maxes": None,
            "min_finite_ratio": None,
        }
    means = [f["mean"] for f in frames if f.get("mean") is not None]
    mins = [f["min"] for f in frames if f.get("min") is not None]
    maxs = [f["max"] for f in frames if f.get("max") is not None]
    fr = [f["finite_ratio"] for f in frames if f.get("finite_ratio") is not None]
    return {
        "n_frames": len(frames),
        "mean_of_means": round(sum(means) / len(means), 6) if means else None,
        "min_of_mins": round(min(mins), 6) if mins else None,
        "max_of_maxes": round(max(maxs), 6) if maxs else None,
        "min_finite_ratio": round(min(fr), 6) if fr else None,
    }


def _fmt_num(x, nd: int = 2) -> str:
    """机读行里的数字格式：None -> 'n/a'。"""
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _resolve_out_dir(out: str) -> Path:
    """``--out`` -> 绝对目录；相对路径锚定仓根（REPO_ROOT）。"""
    p = Path(str(out)).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def _assemble_payload(
    *,
    cli,
    mode: str,
    camera: dict,
    feasibility: dict,
    timing: dict,
    vram: dict,
    depth: dict,
    versions: dict,
    git_commit: str,
    timestamp: str,
    out_dir,
) -> dict:
    """JSON 信封（纯函数：便于 --selftest 校验结构与可序列化性）。"""
    return {
        "entry": "apt_g1/isaac/smoke_camera_depth_g6.py",
        "timestamp": timestamp,
        "format": 1,
        "git_commit": git_commit,
        "mode": mode,
        "out_dir": str(out_dir),
        "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cli).items()},
        "camera": camera,
        "feasibility": feasibility,
        "timing": timing,
        "vram": vram,
        "depth": depth,
        "answers": {
            "render_feasible": bool(feasibility.get("verdict") == "PASS"),
            "step_overhead_ms": timing.get("render_overhead_ms"),
            "vram_delta_mb": vram.get("delta_mb"),
        },
        "versions": versions,
    }


def _summary_line(payload: dict) -> str:
    """单行机读汇总（盯守代理 grep 'SMOKE_CAMERA '）。

    PASS 行含三个数字（step/base/overhead、vram_delta、depth 统计）；FAIL 行含
    reason（自由文本，故该行可能含空格，grep 用前缀即可）。
    """
    f = payload["feasibility"]
    t = payload["timing"]
    v = payload["vram"]
    d = payload["depth"].get("summary") or {}
    cam = payload["camera"]
    verdict = f.get("verdict", "FAIL")
    parts = [
        f"envs={t.get('num_envs')}",
        f"res={cam.get('res_tag')}",
        f"mode={payload['mode']}",
    ]
    if verdict == "PASS":
        parts += [
            f"step_ms={_fmt_num(t.get('camera_step_ms'))}",
            f"base_ms={_fmt_num(t.get('baseline_step_ms'))}",
            f"overhead_ms={_fmt_num(t.get('render_overhead_ms'))}",
            f"cam_update_ms={_fmt_num(t.get('camera_update_ms'))}",
            f"vram_delta_mb={_fmt_num(v.get('delta_mb'))}",
            f"depth_frames={d.get('n_frames', 0)}",
            f"depth_finite={_fmt_num(d.get('min_finite_ratio'), 3)}",
            f"depth_mean={_fmt_num(d.get('mean_of_means'), 4)}",
        ]
    else:
        parts.append(f"reason={f.get('error') or 'unknown'}")
    parts.append(f"out={payload['out_dir']}")
    return f"SMOKE_CAMERA {verdict} " + " ".join(parts)


def build_args() -> argparse.ArgumentParser:
    """CLI（风格照 apt_g1/isaac/train_g1_decoder.py:96-182 的 build_args 段）。"""
    ap = argparse.ArgumentParser(
        description=(
            "G6 深度接入冒烟：Isaac 相机能否渲染 / 单步渲染耗时 / 显存增量"
            "（--selftest 本机可跑，不 import isaaclab）"
        )
    )
    ap.add_argument(
        "--num-envs",
        type=int,
        default=DEFAULT_NUM_ENVS,
        help=f"并行环境数（默认 {DEFAULT_NUM_ENVS}；可扫 {'/'.join(map(str, NUM_ENVS_SWEEP))}）",
    )
    ap.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"每个相位（baseline / 相机）的计时时长上限秒数（默认 {DEFAULT_DURATION_S}；"
        f"另有 {MAX_STEPS_PER_PHASE} 步硬上限）",
    )
    ap.add_argument(
        "--res",
        default=DEFAULT_RES,
        help=f"相机分辨率 'WxH'（默认 {DEFAULT_RES} -> height=108, width=192）",
    )
    ap.add_argument(
        "--terrain",
        choices=TERRAINS,
        default="plane",
        help="地形（透传 sibling make_env_cfg 工厂；plane 最省，默认）",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子（cfg.seed，默认 0）",
    )
    ap.add_argument(
        "--baseline-no-camera",
        action="store_true",
        help="对照开关：只跑 baseline 相位（renderer 仍开，但不挂相机、不取 depth）",
    )
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"输出目录（默认 {DEFAULT_OUT}/，相对路径锚定仓根）",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="本机自检：参数/分辨率解析、统计数学、JSON 结构、输出行格式（不 import isaaclab）",
    )
    ap.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="无头启动（默认开；--no-headless 开视口，本机/服务器渲染冒烟不建议）",
    )
    return ap


##
# 本机自检
##


def _selftest() -> int:
    """本机可测部分全检（不 import isaaclab/numpy/torch）。返回 0/1。"""
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            results.append((name, True, ""))
        except Exception as exc:  # noqa: BLE001 —— 自检要报出所有失败项
            results.append((name, False, f"{type(exc).__name__}: {exc}"))

    def _raises(exc_type, fn):
        try:
            fn()
        except exc_type:
            return
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"期望 {exc_type.__name__}，实际 {type(exc).__name__}: {exc}") from exc
        raise AssertionError(f"期望 {exc_type.__name__}，但未抛异常")

    # ---- 分辨率解析 ----
    check("res_parse_default", lambda: _assert_eq(_parse_res("192x108"), (108, 192)))
    check("res_parse_caps_space", lambda: _assert_eq(_parse_res(" 64X48 "), (48, 64)))
    check("res_parse_320x240", lambda: _assert_eq(_parse_res("320x240"), (240, 320)))
    check("res_parse_reject", lambda: [
        _raises(ValueError, lambda: _parse_res(s)) for s in ("abc", "192", "0x10", "192x-1", "192xx108", "")
    ])
    check("res_tag_roundtrip", lambda: _assert_eq(_res_tag(*_parse_res("192x108")), "192x108"))

    # ---- depth 统计数学 ----
    check("depth_stats_basic", lambda: _assert_eq(
        _pick(depth_stats([0.0, 1.0, 2.0, 3.0]), "n", "min", "mean", "max", "nonzero_ratio", "finite_ratio", "n_nonfinite"),
        {"n": 4, "min": 0.0, "mean": 1.5, "max": 3.0, "nonzero_ratio": 0.75, "finite_ratio": 1.0, "n_nonfinite": 0},
    ))
    check("depth_stats_nonfinite_to_zero", lambda: _assert_eq(
        _pick(depth_stats([float("inf"), float("nan"), float("-inf"), 2.0]),
              "n", "n_nonfinite", "finite_ratio", "min", "mean", "max", "nonzero_ratio"),
        {"n": 4, "n_nonfinite": 3, "finite_ratio": 0.25, "min": 0.0, "mean": 0.5, "max": 2.0, "nonzero_ratio": 0.25},
    ))
    check("depth_stats_empty", lambda: _assert_eq(
        depth_stats([]),
        {"n": 0, "n_nonfinite": 0, "finite_ratio": None, "min": None, "mean": None, "max": None, "nonzero_ratio": None},
    ))
    check("depth_stats_constant_one", lambda: _assert_eq(
        _pick(depth_stats([1.0] * 8), "n", "min", "mean", "max", "nonzero_ratio", "finite_ratio"),
        {"n": 8, "min": 1.0, "mean": 1.0, "max": 1.0, "nonzero_ratio": 1.0, "finite_ratio": 1.0},
    ))
    check("frame_depth_stats_subsample", lambda: _assert_eq(
        _pick(_frame_depth_stats([float(i) for i in range(10)], cap=3), "n_total", "stride", "n"),
        {"n_total": 10, "stride": 4, "n": 3},
    ))
    check("frame_depth_stats_empty", lambda: _assert_eq(
        _pick(_frame_depth_stats([]), "n_total", "stride", "n", "mean"),
        {"n_total": 0, "stride": 1, "n": 0, "mean": None},
    ))
    check("depth_summary", lambda: _assert_eq(
        _depth_summary([
            {"mean": 1.0, "min": 0.5, "max": 2.0, "finite_ratio": 0.9},
            {"mean": 3.0, "min": 0.25, "max": 4.0, "finite_ratio": 0.8},
        ]),
        {"n_frames": 2, "mean_of_means": 2.0, "min_of_mins": 0.25, "max_of_maxes": 4.0, "min_finite_ratio": 0.8},
    ))
    check("depth_summary_empty", lambda: _assert_eq(
        _pick(_depth_summary([]), "n_frames", "mean_of_means", "min_finite_ratio"),
        {"n_frames": 0, "mean_of_means": None, "min_finite_ratio": None},
    ))

    # ---- 输出行格式 ----
    pass_payload = _dummy_payload(verdict="PASS")
    fail_payload = _dummy_payload(verdict="FAIL", error="RuntimeError: boom")
    check("summary_line_pass", lambda: _assert_line(
        _summary_line(pass_payload),
        must=["SMOKE_CAMERA PASS ", "envs=8", "res=192x108", "mode=camera",
              "step_ms=12.34", "base_ms=11.00", "overhead_ms=1.34",
              "vram_delta_mb=42.10", "depth_frames=5", "depth_finite=1.000", "depth_mean=0.812"],
    ))
    check("summary_line_pass_na", lambda: _assert_line(
        _summary_line(_dummy_payload(verdict="PASS", mode="baseline_no_camera", camera_step_ms=None,
                                     overhead=None, delta=None, depth_frames=0)),
        must=["SMOKE_CAMERA PASS ", "mode=baseline_no_camera", "step_ms=n/a",
              "overhead_ms=n/a", "vram_delta_mb=n/a", "depth_frames=0"],
    ))
    check("summary_line_fail", lambda: _assert_line(
        _summary_line(fail_payload), must=["SMOKE_CAMERA FAIL ", "reason=RuntimeError: boom"],
    ))

    # ---- JSON 结构与可序列化 ----
    def _json_ok() -> None:
        payload = _dummy_payload(verdict="PASS")
        for key in ("entry", "timestamp", "format", "git_commit", "mode", "out_dir",
                    "cli", "camera", "feasibility", "timing", "vram", "depth",
                    "answers", "versions"):
            if key not in payload:
                raise AssertionError(f"payload 缺顶层键 {key!r}")
        if set(payload["answers"]) != {"render_feasible", "step_overhead_ms", "vram_delta_mb"}:
            raise AssertionError(f"answers 键集合异常: {sorted(payload['answers'])}")
        if not payload["answers"]["render_feasible"]:
            raise AssertionError("PASS payload 的 render_feasible 应为 True")
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        back = json.loads(text)
        if back["entry"] != payload["entry"]:
            raise AssertionError("JSON round-trip 不一致")

    check("payload_structure_json", _json_ok)

    def _fail_payload_ok() -> None:
        p = _dummy_payload(verdict="FAIL", error="X")
        if p["answers"]["render_feasible"]:
            raise AssertionError("FAIL payload 的 render_feasible 应为 False")
        json.dumps(p, ensure_ascii=False)

    check("payload_fail_json", _fail_payload_ok)

    # ---- CLI 解析 ----
    check("cli_defaults", lambda: _assert_eq(
        _pick(vars(build_args().parse_args([])),
              "num_envs", "duration_s", "res", "out", "baseline_no_camera", "selftest", "headless", "terrain", "seed"),
        {"num_envs": 8, "duration_s": 30.0, "res": "192x108", "out": "outputs/g6",
         "baseline_no_camera": False, "selftest": False, "headless": True, "terrain": "plane", "seed": 0},
    ))
    check("cli_overrides", lambda: _assert_eq(
        _pick(vars(build_args().parse_args(
            ["--num-envs", "64", "--duration-s", "5", "--res", "64x48", "--baseline-no-camera",
             "--out", "/tmp/x", "--no-headless", "--terrain", "rough", "--seed", "7"])),
            "num_envs", "duration_s", "res", "out", "baseline_no_camera", "headless", "terrain", "seed"),
        {"num_envs": 64, "duration_s": 5.0, "res": "64x48", "out": "/tmp/x",
         "baseline_no_camera": True, "headless": False, "terrain": "rough", "seed": 7},
    ))
    check("cli_bad_terrain", lambda: _raises(SystemExit, lambda: build_args().parse_args(["--terrain", "mars"])))

    # ---- 常量/路径纪律 ----
    def _camera_constants() -> None:
        if CAMERA_PRIM_PATH != "{ENV_REGEX_NS}/Robot/head_link/ego_camera":
            raise AssertionError(f"CAMERA_PRIM_PATH 异常: {CAMERA_PRIM_PATH}")
        if "d435" in CAMERA_PRIM_PATH:
            raise AssertionError("prim 路径不应含 d435（main.urdf 无 d435_link）")
        if CAMERA_LINK not in CAMERA_PRIM_PATH or "{ENV_REGEX_NS}" not in CAMERA_PRIM_PATH:
            raise AssertionError("prim 路径应含 {ENV_REGEX_NS} 与 head_link")
        if "depth" not in CAMERA_DATA_TYPES:
            raise AssertionError("data_types 必须含 depth")
        for key in EULA_ENV:
            if not key:
                raise AssertionError("EULA_ENV 键为空")

    check("camera_eula_constants", _camera_constants)

    check("out_dir_relative_anchored", lambda: _assert_eq(
        _resolve_out_dir("outputs/g6/"), REPO_ROOT / "outputs" / "g6",
    ))
    # 绝对路径透传用 OS 原生绝对路径测（POSIX 的 "/tmp/g6" 在 Windows 上
    # 非绝对，会被锚到仓根——本机 selftest 需跨平台）
    check("out_dir_absolute_passthrough", lambda: _assert_eq(
        _resolve_out_dir(str(Path.cwd())), Path.cwd(),
    ))

    # ---- 汇总 ----
    n_total = len(results)
    n_pass = sum(1 for _n, ok, _m in results if ok)
    for name, ok, msg in results:
        print(f"[selftest] {name}: {'PASS' if ok else 'FAIL'}" + (f" — {msg}" if msg else ""), flush=True)
    if n_pass == n_total:
        print(f"SELFTEST PASS: {n_pass}/{n_total} checks (no isaaclab import)", flush=True)
        return 0
    print(f"SELFTEST FAIL: {n_pass}/{n_total} checks passed", flush=True)
    return 1


def _assert_eq(got, want) -> None:
    if got != want:
        raise AssertionError(f"{got!r} != {want!r}")


def _pick(d: dict, *keys) -> dict:
    missing = [k for k in keys if k not in d]
    if missing:
        raise AssertionError(f"缺键 {missing}（实际 {sorted(d)}）")
    return {k: d[k] for k in keys}


def _assert_line(line: str, must: list[str]) -> None:
    for token in must:
        if token not in line:
            raise AssertionError(f"行内缺 {token!r}: {line}")


def _dummy_payload(*, verdict: str, mode: str = "camera", error: str | None = None,
                   camera_step_ms=12.34, overhead=1.34, delta=42.1, depth_frames: int = 5) -> dict:
    """--selftest 用的假 payload（走 _assemble_payload 真实装配路径）。"""
    cli = SimpleNamespace(num_envs=8, duration_s=30.0, res="192x108", out="outputs/g6",
                          baseline_no_camera=(mode != "camera"), selftest=False, headless=True,
                          terrain="plane", seed=0)
    frames = [{"n_total": 165888, "stride": 1, "n": 165888, "n_nonfinite": 0,
               "finite_ratio": 1.0, "min": 0.1, "mean": 0.812, "max": 3.4,
               "nonzero_ratio": 1.0} for _ in range(depth_frames)]
    depth_summary = _depth_summary(frames) if frames else _depth_summary([])
    if not frames:
        depth_summary["min_finite_ratio"] = None
        depth_summary["mean_of_means"] = None
    return _assemble_payload(
        cli=cli,
        mode=mode,
        camera={"link": CAMERA_LINK, "name": CAMERA_NAME, "prim_path": CAMERA_PRIM_PATH,
                "height": 108, "width": 192, "res_tag": "192x108",
                "data_types": list(CAMERA_DATA_TYPES), "attach_via": "runtime_scene_sensors"},
        feasibility={"enable_cameras": True, "app_launcher_ok": True, "env_built_ok": True,
                     "first_step_ok": True, "camera_attached": verdict == "PASS",
                     "depth_ok": verdict == "PASS", "verdict": verdict, "error": error},
        timing={"num_envs": 8, "duration_s": 30.0, "max_steps": MAX_STEPS_PER_PHASE,
                "warmup_steps": WARMUP_STEPS, "baseline_steps": 100, "baseline_step_ms": 11.0,
                "baseline_elapsed_s": 1.1, "camera_steps": 100 if camera_step_ms else None,
                "camera_step_ms": camera_step_ms, "camera_update_ms": 0.5,
                "camera_elapsed_s": 1.2, "render_overhead_ms": overhead,
                "render_overhead_with_update_ms": None if overhead is None else round(overhead + 0.5, 4)},
        vram={"device": "cuda:0", "has_cuda": True, "baseline_peak_mb": 100.0,
              "camera_peak_mb": None if delta is None else 100.0 + delta,
              "delta_mb": delta, "note": "" if delta is not None else "n/a (baseline_no_camera)"},
        depth={"frames": frames, "summary": depth_summary, "error": None},
        versions={"isaaclab": "test", "isaacsim": "test", "torch": "test"},
        git_commit="deadbeef",
        timestamp="2026-09-22T00:00:00+0800",
        out_dir=REPO_ROOT / "outputs" / "g6",
    )


##
# 重型依赖（App 起来之后才允许 import）
##


def _import_heavy() -> SimpleNamespace:
    """App 启动后的 isaaclab / sibling imports（app 起来之前禁止）。

    双兼容 import 与 ``train_g1_decoder.py:535-558`` 完全同构（主支 ``isaac.*``，
    回退 ``apt_g1.isaac.*``：服务器 PYTHONPATH 同时含仓根与仓根/apt_g1）。
    """
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCamera, TiledCameraCfg

    if str(REPO_ROOT) not in sys.path:
        # 以 `python apt_g1/isaac/smoke_camera_depth_g6.py` 直跑时 sys.path[0] 是
        # 脚本目录而非仓根，补上保证 apt_g1.* 可导入（服务器 PYTHONPATH 亦含仓根）
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from isaac.g1_velocity_decoder_env import make_env_cfg
    except ImportError:
        try:
            from apt_g1.isaac.g1_velocity_decoder_env import make_env_cfg
        except ImportError as exc:
            raise RuntimeError(
                "sibling g1_velocity_decoder_env.make_env_cfg 不可导入"
                "（isaac.* 与 apt_g1.isaac.* 均失败）"
            ) from exc

    return SimpleNamespace(
        torch=torch,
        ManagerBasedRLEnv=ManagerBasedRLEnv,
        sim_utils=sim_utils,
        TiledCamera=TiledCamera,
        TiledCameraCfg=TiledCameraCfg,
        make_env_cfg=make_env_cfg,
    )


def _camera_cfg(sim_utils, TiledCameraCfg, height: int, width: int):
    """相机 cfg（结构照 gear_sonic modular_tracking_env_cfg.py:849-889，改 4 处见模块 docstring）。

    独立纯函数：运行时注入（本冒烟）与声明式 ``cfg.scene.<name> = ...``（生产路径）
    共用同一份配方。
    """
    return TiledCameraCfg(
        prim_path=CAMERA_PRIM_PATH,
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_OFFSET_POS, rot=CAMERA_OFFSET_ROT, convention="world"
        ),
        data_types=list(CAMERA_DATA_TYPES),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            focus_distance=CAMERA_FOCUS_DISTANCE,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
            vertical_aperture=CAMERA_VERTICAL_APERTURE,
            clipping_range=CAMERA_CLIPPING_RANGE,
        ),
        height=int(height),
        width=int(width),
        debug_vis=False,           # 改：headless 无 viewport（gear_sonic 为 True）
        update_period=0.0,
        update_latest_camera_pose=True,
    )


def _attach_camera(env, hv, height: int, width: int):
    """运行时把 TiledCamera 注入 ``env.scene._sensors``（先例 render_walk.py:117-121）。

    单 env、不重建场景（设计决定 1）。初始化仅在构造未初始化时补调
    （``_initialize_impl`` 在部分版本由 ``SensorBase.__init__`` 已调用）。
    """
    sensors = getattr(env.scene, "_sensors", None)
    if not isinstance(sensors, dict):
        raise RuntimeError(
            "env.scene 无 dict 型 _sensors 属性（IsaacLab 版本 API 变动？）"
            "—— 运行时挂载路径不可用"
        )
    cam = hv.TiledCamera(_camera_cfg(hv.sim_utils, hv.TiledCameraCfg, height, width))
    sensors[CAMERA_NAME] = cam
    if not getattr(cam, "_is_initialized", False):
        cam._initialize_impl()
    cam.reset()
    return cam


def _read_depth_flat(cam) -> list[float]:
    """取相机 depth 输出并展平为 float 列表（失败即 RuntimeError，带明确原因）。"""
    out = getattr(getattr(cam, "data", None), "output", None)
    if not isinstance(out, dict) or "depth" not in out or out["depth"] is None:
        keys = sorted(out) if isinstance(out, dict) else type(out).__name__
        raise RuntimeError(
            f"camera.data.output 无可用 'depth'（实际键 {keys}）——"
            "TiledCamera data_types 未含 depth 或版本 API 变动"
        )
    return out["depth"].detach().cpu().reshape(-1).tolist()


def _warmup(env, action, n_steps: int, cam=None, dt: float = 0.0) -> None:
    for _ in range(int(n_steps)):
        env.step(action)
        if cam is not None:
            cam.update(dt)


def _timed_steps(env, action, cam, duration_s: float, dt: float,
                 collect_depth: bool = False) -> dict:
    """计时循环：``env.step`` 与 ``cam.update`` 分开计时（设计决定 3）。

    跑满 ``duration_s`` 秒或 ``MAX_STEPS_PER_PHASE`` 步（先到者为准）；
    ``collect_depth`` 时每 ``DEPTH_EVERY`` 步取一帧 depth 统计（上限 MAX_DEPTH_FRAMES）。
    """
    deadline = time.perf_counter() + float(duration_s)
    step_times: list[float] = []
    upd_times: list[float] = []
    frames: list[dict] = []
    n = 0
    while n < MAX_STEPS_PER_PHASE and time.perf_counter() < deadline:
        t0 = time.perf_counter()
        env.step(action)
        step_times.append(time.perf_counter() - t0)
        if cam is not None:
            t1 = time.perf_counter()
            cam.update(dt)
            upd_times.append(time.perf_counter() - t1)
            if collect_depth and n % DEPTH_EVERY == 0 and len(frames) < MAX_DEPTH_FRAMES:
                frames.append(_frame_depth_stats(_read_depth_flat(cam)))
        n += 1
    return {
        "n_steps": n,
        "step_ms": round(1000.0 * sum(step_times) / len(step_times), 4) if step_times else None,
        "update_ms": round(1000.0 * sum(upd_times) / len(upd_times), 4) if upd_times else None,
        "elapsed_s": round(sum(step_times) + sum(upd_times), 4),
        "frames": frames,
    }


##
# 运行
##


def _git_head() -> str:
    """git rev-parse HEAD；失败返回 ""（同 train_g1_decoder.py:189-201）。"""
    try:
        g = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=5,
        )
        return g.stdout.strip() if g.returncode == 0 else ""
    except Exception:
        return ""


def _pkg_ver(name: str) -> str:
    try:
        return _pkg_version(name)
    except Exception:
        return "unknown"


def _versions() -> dict:
    return {
        "isaaclab": _pkg_ver("isaaclab"),
        "isaacsim": _pkg_ver("isaacsim"),
        "torch": _pkg_ver("torch"),
    }


def _write_json(out_dir: Path, payload: dict) -> Path:
    path = out_dir / JSON_NAME
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _camera_block(height: int, width: int, attach_via: str) -> dict:
    return {
        "link": CAMERA_LINK,
        "name": CAMERA_NAME,
        "prim_path": CAMERA_PRIM_PATH,
        "height": int(height),
        "width": int(width),
        "res_tag": _res_tag(height, width),
        "data_types": list(CAMERA_DATA_TYPES),
        "spawn": {
            "focal_length": CAMERA_FOCAL_LENGTH,
            "focus_distance": CAMERA_FOCUS_DISTANCE,
            "horizontal_aperture": CAMERA_HORIZONTAL_APERTURE,
            "vertical_aperture": CAMERA_VERTICAL_APERTURE,
            "clipping_range": list(CAMERA_CLIPPING_RANGE),
        },
        "offset": {"pos": list(CAMERA_OFFSET_POS), "rot_wxyz": list(CAMERA_OFFSET_ROT),
                   "convention": "world"},
        "attach_via": attach_via,
    }


def _preflight_stub(cli, out_dir: Path, mode: str, height: int, width: int) -> None:
    """AppLauncher 之前的 stub：若进程在 stage 创建段错误，本 stub 即最终产物。"""
    payload = _assemble_payload(
        cli=cli,
        mode=mode,
        camera=_camera_block(height, width, "pending"),
        feasibility={"enable_cameras": True, "app_launcher_ok": False, "env_built_ok": False,
                     "first_step_ok": False, "camera_attached": False, "depth_ok": False,
                     "verdict": "PENDING",
                     "error": "preflight stub：进程在写完整 JSON 前退出"
                              "（AppLauncher/stage 创建段错误则本 stub 即最终产物）"},
        timing={"num_envs": cli.num_envs, "duration_s": cli.duration_s,
                "max_steps": MAX_STEPS_PER_PHASE, "warmup_steps": WARMUP_STEPS},
        vram={"device": None, "has_cuda": None, "baseline_peak_mb": None,
              "camera_peak_mb": None, "delta_mb": None, "note": "preflight"},
        depth={"frames": [], "summary": _depth_summary([]), "error": "preflight"},
        versions=_versions(),
        git_commit=_git_head(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        out_dir=out_dir,
    )
    _write_json(out_dir, payload)


def _run(cli, out_dir: Path, launcher_args) -> dict:
    """App 起来之后的全部逻辑；返回 payload（相机相位失败不抛，落 FAIL 后如实返回）。"""
    hv = _import_heavy()
    torch = hv.torch
    device = getattr(launcher_args, "device", "cuda:0")
    has_cuda = bool(torch.cuda.is_available())
    height, width = _parse_res(cli.res)
    mode = "baseline_no_camera" if cli.baseline_no_camera else "camera"
    dt = 0.0

    feasibility = {
        "enable_cameras": True,
        "app_launcher_ok": True,   # 能进 _run 即已过 AppLauncher
        "env_built_ok": False,
        "first_step_ok": False,
        "camera_attached": False,
        "depth_ok": None,
        "verdict": "FAIL",
        "error": None,
    }
    depth_block = {"frames": [], "summary": _depth_summary([]), "error": None}
    timing = {
        "num_envs": cli.num_envs, "duration_s": cli.duration_s, "mode": mode,
        "max_steps": MAX_STEPS_PER_PHASE, "warmup_steps": WARMUP_STEPS,
        "baseline_steps": None, "baseline_step_ms": None, "baseline_elapsed_s": None,
        "camera_steps": None, "camera_step_ms": None, "camera_update_ms": None,
        "camera_elapsed_s": None, "render_overhead_ms": None,
        "render_overhead_with_update_ms": None,
    }
    vram = {"device": device, "has_cuda": has_cuda, "baseline_peak_mb": None,
            "camera_peak_mb": None, "delta_mb": None, "note": ""}
    attach_via = "none (baseline_no_camera)" if mode == "baseline_no_camera" else "runtime_scene_sensors"

    print(
        f"[g6] CFG out={out_dir} mode={mode} num_envs={cli.num_envs} res={width}x{height} "
        f"terrain={cli.terrain} seed={cli.seed} duration_s={cli.duration_s} "
        f"device={device} cuda={has_cuda} headless={cli.headless}",
        flush=True,
    )

    cfg = hv.make_env_cfg(terrain=cli.terrain, num_envs=cli.num_envs, action="direct", seed=cli.seed)
    cfg.scene.env_spacing = ENV_SPACING
    if hasattr(cfg, "seed"):
        cfg.seed = cli.seed
    if hasattr(cfg.sim, "device"):
        cfg.sim.device = device
    print(f"[g6] env cfg via make_env_cfg(terrain={cli.terrain}, action=direct, "
          f"num_envs={cli.num_envs})", flush=True)

    env = None
    try:
        env = hv.ManagerBasedRLEnv(cfg=cfg)
        feasibility["env_built_ok"] = True
        obs, _ = env.reset()
        if not (isinstance(obs, dict) and "policy" in obs):
            raise RuntimeError(
                f"env obs 应为含 'policy' 组的 dict，实际 {type(obs)} "
                f"keys={list(obs) if isinstance(obs, dict) else '-'}"
            )
        action_dim = int(env.action_manager.total_action_dim)
        action = torch.zeros(cli.num_envs, action_dim, device=env.device)
        dt = float(getattr(env.cfg.sim, "dt", 0.0))
        print(f"[g6] env built: num_envs={env.num_envs} action_dim={action_dim} "
              f"sim_dt={dt} scene_sensors={sorted(getattr(env.scene, '_sensors', {}) or {})}",
              flush=True)

        # ---- 相位 A：baseline（renderer 开、不挂相机）----
        _warmup(env, action, WARMUP_STEPS)
        feasibility["first_step_ok"] = True
        if has_cuda:
            torch.cuda.reset_peak_memory_stats()
        base = _timed_steps(env, action, None, cli.duration_s, dt)
        timing["baseline_steps"] = base["n_steps"]
        timing["baseline_step_ms"] = base["step_ms"]
        timing["baseline_elapsed_s"] = base["elapsed_s"]
        if has_cuda:
            vram["baseline_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024.0 ** 2), 3)
        print(f"[g6] phase A (baseline, no camera): steps={base['n_steps']} "
              f"step_ms={base['step_ms']} peak_mb={vram['baseline_peak_mb']}", flush=True)

        if mode == "baseline_no_camera":
            vram["note"] = "n/a (baseline_no_camera 对照运行，未挂相机)"
            depth_block["error"] = "n/a (baseline_no_camera)"
            feasibility["depth_ok"] = None
            feasibility["verdict"] = "PASS"   # ① enable_cameras=True 过 AppLauncher/env/首 step
            feasibility["error"] = None
        else:
            # ---- 相位 B：运行时挂载相机 + depth ----
            cam = _attach_camera(env, hv, height, width)
            feasibility["camera_attached"] = True
            print(f"[g6] camera attached: prim_path={CAMERA_PRIM_PATH} "
                  f"{width}x{height} data_types={list(CAMERA_DATA_TYPES)}", flush=True)
            _warmup(env, action, WARMUP_STEPS, cam=cam, dt=dt)
            if has_cuda:
                torch.cuda.reset_peak_memory_stats()
            camphase = _timed_steps(env, action, cam, cli.duration_s, dt, collect_depth=True)
            timing["camera_steps"] = camphase["n_steps"]
            timing["camera_step_ms"] = camphase["step_ms"]
            timing["camera_update_ms"] = camphase["update_ms"]
            timing["camera_elapsed_s"] = camphase["elapsed_s"]
            if has_cuda:
                vram["camera_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024.0 ** 2), 3)
            if timing["baseline_step_ms"] is not None and camphase["step_ms"] is not None:
                timing["render_overhead_ms"] = round(camphase["step_ms"] - timing["baseline_step_ms"], 4)
                if camphase["update_ms"] is not None:
                    timing["render_overhead_with_update_ms"] = round(
                        timing["render_overhead_ms"] + camphase["update_ms"], 4
                    )
            depth_block["frames"] = camphase["frames"]
            depth_block["summary"] = _depth_summary(camphase["frames"])
            if not camphase["frames"]:
                raise RuntimeError("相机相位未取到任何 depth 帧（DEPTH_EVERY 与步数不匹配？）")
            feasibility["depth_ok"] = True
            feasibility["verdict"] = "PASS"
            print(f"[g6] phase B (camera): steps={camphase['n_steps']} "
                  f"step_ms={camphase['step_ms']} cam_update_ms={camphase['update_ms']} "
                  f"overhead_ms={timing['render_overhead_ms']} "
                  f"peak_mb={vram['camera_peak_mb']} depth_frames={len(camphase['frames'])}",
                  flush=True)
        if has_cuda and vram["baseline_peak_mb"] is not None and vram["camera_peak_mb"] is not None:
            vram["delta_mb"] = round(vram["camera_peak_mb"] - vram["baseline_peak_mb"], 3)
            vram["note"] = "camera_peak - baseline_peak（两相位各自 reset_peak_memory_stats）"
    except Exception as exc:  # noqa: BLE001 —— 相机相位任何失败都落 FAIL（不重试）
        feasibility["verdict"] = "FAIL"
        feasibility["error"] = f"{type(exc).__name__}: {exc}"
        if feasibility["camera_attached"] and not depth_block["frames"]:
            feasibility["depth_ok"] = False
            depth_block["error"] = feasibility["error"]
        elif not feasibility["camera_attached"]:
            # 相机相位未走到（baseline 相位或 env 构造期失败）：depth 未采集
            depth_block["error"] = depth_block["error"] or feasibility["error"]
        print(f"[g6] FAIL: {feasibility['error']}", flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    if not has_cuda:
        vram["note"] = (vram["note"] + " " if vram["note"] else "") + "n/a (no cuda)"

    return _assemble_payload(
        cli=cli,
        mode=mode,
        camera=_camera_block(height, width, attach_via),
        feasibility=feasibility,
        timing=timing,
        vram=vram,
        depth=depth_block,
        versions=_versions(),
        git_commit=_git_head(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        out_dir=out_dir,
    )


def main() -> None:
    # CLI 解析放最前：-h/--help 与 --selftest 在任何 isaaclab import 之前完成
    cli = build_args().parse_args()

    if cli.selftest:
        sys.exit(_selftest())

    out_dir = _resolve_out_dir(cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "baseline_no_camera" if cli.baseline_no_camera else "camera"
    height, width = _parse_res(cli.res)

    # EULA（栈惯例；已设则不动）
    for key, val in EULA_ENV.items():
        os.environ.setdefault(key, val)

    # segfault 时 stderr 给 traceback（历史故障点在 AppLauncher/stage 创建）
    try:
        import faulthandler

        faulthandler.enable()
    except Exception:
        pass

    # AppLauncher 之前先落 stub：若在 stage 创建段错误，stub 即最终产物（设计决定 4）
    _preflight_stub(cli, out_dir, mode, height, width)
    print(f"[g6] preflight stub -> {out_dir / JSON_NAME}", flush=True)
    print(f"[g6] launching AppLauncher (enable_cameras=True, headless={cli.headless}, "
          f"num_envs={cli.num_envs}) ...", flush=True)

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = cli.num_envs
    launcher_args.headless = cli.headless
    launcher_args.enable_cameras = True
    launcher_args.env_spacing = ENV_SPACING
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    exit_code = 1
    try:
        payload = _run(cli, out_dir, launcher_args)
        json_path = _write_json(out_dir, payload)
        print(_summary_line(payload), flush=True)
        print(f"[g6] json -> {json_path}", flush=True)
        exit_code = 0 if payload["feasibility"]["verdict"] == "PASS" else 1
    except Exception as exc:  # noqa: BLE001 —— env 构造之前的失败（如 make_env_cfg）
        payload = _assemble_payload(
            cli=cli,
            mode=mode,
            camera=_camera_block(height, width, "failed_before_attach"),
            feasibility={"enable_cameras": True, "app_launcher_ok": True, "env_built_ok": False,
                         "first_step_ok": False, "camera_attached": False, "depth_ok": None,
                         "verdict": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
            timing={"num_envs": cli.num_envs, "duration_s": cli.duration_s, "mode": mode,
                    "max_steps": MAX_STEPS_PER_PHASE, "warmup_steps": WARMUP_STEPS},
            vram={"device": None, "has_cuda": None, "baseline_peak_mb": None,
                  "camera_peak_mb": None, "delta_mb": None, "note": "n/a (env 未建成)"},
            depth={"frames": [], "summary": _depth_summary([]), "error": "n/a (env 未建成)"},
            versions=_versions(),
            git_commit=_git_head(),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            out_dir=out_dir,
        )
        _write_json(out_dir, payload)
        print(_summary_line(payload), flush=True)
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
    sys.exit(exit_code)


if __name__ == "__main__":
    main()