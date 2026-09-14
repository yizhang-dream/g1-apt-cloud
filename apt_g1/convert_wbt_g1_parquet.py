"""D057 G1/G3：UnifoLM-WBT LeRobot parquet -> 1762-d g1-mode encoder obs ->
frozen release encoder tokens（+ 可选 decoder 回环），产物与 D044
BONES-SEED 转换完全同构（npz 键 / manifest 语义），走同一编码链与门径。

数据契约（服务器 lab-ts，~/ros2_data/apt_g1/data/ds_wbt/raw/<RepoName>/
{meta/*.json, data/**/*.parquet}）：UnifoLM-WBT 为 LeRobot 格式 30 fps，
双 schema 并存，逐文件自适应（仿 tmp/wbt_probe.py read_file）——
  老款 q36   : observation.state.robot_q_current[36] = 根位姿 7
               （xyz 米 + 四元数 wxyz）+ 29 关节（rad）
  新款 grouped: observation.state.state_base_pose[7]（同根 7）+ 关节分组
               lower_body[15] + left_arm[7] + right_arm[7] = 29（拼接顺序即此）
29 关节顺序假设 = MuJoCo 序（腿 L6/R6 -> 腰 3 -> 臂 L7/R7，与
convert_bones_g1_csv.py MUJOCO_JOINT_NAMES 同构）。本脚本负责用纯 numpy
"关节签名断言"逐 run 验证该假设，不猜测。

编码链逐字镜像 convert_bones_g1_csv.py（D044）encode_segment/main，零漂移
import encode_bones_smoke（D036/D038 验证过的 v2 ref-rel 路径），仅五处不同：
  1. 根四元数直接来自数据（wxyz，先 hemisphere_align 半球连续化再
     resample_quat——数据 quat 可能逐帧符号翻转，Euler 导出的无此问题），
     无 euler_to_quat / --calibrate；
  2. FPS_SRC=30（模块级常量）。注意 30->50 Hz 是上采样：D036 及后续所有
     resample 先例（120->50、90->50…）均为降采样，上采样无先例=本路线
     已知风险点；帧数口径沿用 encode_bones_smoke.resample 源码语义
     int(round(n*fps_out/fps_in))；
  3. odom 跳变切段：帧间 XY 位移模长 > 0.5 m 判 teleport/复位，episode
     切成 run；< 2s（60 帧@30Hz）的 run 丢弃；n_jumps/dropped_runs 计数
     入 manifest wbt 块；
  4. 关节签名断言（逐 run，纯 numpy）：膝 idx3/idx9 median>0.05 且
     max<2.2；L 肩滚 idx16 median>=-0.2；R 肩滚 idx23 median<=0.2；四元数
     模长 ∈[0.98,1.02]（越界先归一化并记 warning；非有限/近零直接报错）。
     任一断言失败 -> 该 run 记 manifest error 条目跳过，不猜测。
     --roundtrip 回环 MAE 是关节序的兜底校验（序错则 MAE 爆表）；
  5. stem/manifest：stem = wbt_{repo 缩写小写去 G1_WBT_ 前缀}_ep{episode:04d}
     _run{run_idx}（run_idx 按切段后的原始段序编号，含被丢弃段，可回溯）；
     manifest 条目含嵌套 wbt 块（repo/episode/run_idx/schema/fps_src/
     v_med_src/v_p90_src/n_jumps/dropped_runs，速度为 XY 后向差分×fps 口径）。

可选 --limits-check（默认关，服务器专用）：import mujoco 载入 deploy MJCF，
按关节名取 29 关节限位，断言观测值域 ⊂ 限位 ± 0.15 rad。

【2026-09-14 性能旗标 --batch-encode（D039 性能旗标先例：opt-in + 等价验证；
D057 G3 全量转换提速用）】默认关 = 逐帧原始行为逐位不变（与 D044 镜像一致）。
服务器实测逐帧 ONNX 推理 ~30-40 帧/s，5.6M 帧全量需数十小时，瓶颈是 session
逐帧调用开销而非 obs 构建（build_obs/build_decoder_obs 仍逐帧调用、调用方式
一字不改）；旗标开时逐帧收集 obs 后并成一次批 session.run（encoder
(n,1762)->(n,64)，decoder (n,994)->(n,29)），roundtrip err/err0 向量化同式
计算（q_des = sonic_default_isaac + acts*sonic_scale_isaac，与逐帧公式同式）。
批/逐帧数值等价由首个成功 run 的运行时护栏保证：max|Δtokens| <= 1e-4 且
lattice_rate np.isclose，不符 raise RuntimeError（不静默回退），此后各 run
只走批路径。

输出（服务器）：<out-dir>/npz/<stem>.npz（np.savez：tokens (n,64) f32 @50Hz
| jp_isaac/jp_mj (n,29) | quat_wxyz (n,4) | trans_m (n,3) | jv_isaac (n,29)
| meta json 字符串，与 D044 完全同键）+ <out-dir>/manifest.json（按 stem
合并旧 manifest，语义照抄 D044）。

用法（服务器，mjlab venv，cwd=GR00T-WholeBodyControl，经
/tmp/run_apt_isaac.sh 包装）：
  # 全量转换 + 回环：
  python apt_g1/convert_wbt_g1_parquet.py --roundtrip
  # 全量转换提速（批 ONNX 推理，默认关 = 逐帧原始行为）：
  python apt_g1/convert_wbt_g1_parquet.py --roundtrip --batch-encode
  # 冒烟：按 episode v_med 分快/中/慢各取 N/3 个：
  python apt_g1/convert_wbt_g1_parquet.py --sample-episodes 12 --roundtrip
  # 本机自测（numpy-only，不碰 mujoco/onnxruntime/pyarrow）：
  python apt_g1/convert_wbt_g1_parquet.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
HOME = os.path.expanduser("~")
DEFAULT_RAW_ROOT = f"{HOME}/ros2_data/apt_g1/data/ds_wbt/raw"
FPS_SRC = 30.0  # UnifoLM-WBT LeRobot native rate（数据契约）
FPS_ENC = 50.0  # encoder stride-5 assumption (planner_sonic.py)

JUMP_THRESH_M = 0.5     # 帧间 XY 位移模长超过此值判 odom 跳变（teleport/复位）
MIN_RUN_FRAMES = 60     # < 2s @30Hz 的 run 丢弃
LIMIT_MARGIN_RAD = 0.15  # --limits-check：观测值域允许越出限位的余量

# 新款 grouped schema 的关节分组列与拼接顺序（= 契约：腿+腰 15 -> L 臂 7 -> R 臂 7）
GROUPED_JOINT_COLS = (
    "observation.state.lower_body",
    "observation.state.left_arm",
    "observation.state.right_arm",
)

# Canonical MuJoCo G1 29-dof joint order（gear_sonic deploy 语义；与
# convert_bones_g1_csv.py MUJOCO_JOINT_NAMES 同构，--limits-check 按名取限位）。
MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, REPO)

# D036-verified pieces, zero-drift import（encode_bones_smoke 模块级仅依赖
# numpy + stdlib，本机 --selftest 可直接复用其 resample；重依赖另见 main 内
# 的 onnxruntime / mujoco / pyarrow）。
from encode_bones_smoke import (  # noqa: E402
    FPS_ENC as _FPS_ENC_SMOKE,
    LATTICE_TOL,
    _heading,
    _heading_inv,
    _qconj,
    _qn,
    _qmul,
    _quat_rotate_inverse,
    _rotmat,
    build_obs,
    resample,
    resample_quat,
)

assert _FPS_ENC_SMOKE == FPS_ENC, "encoder rate assumption diverged from encode_bones_smoke"


# ------------------------------------------- numpy-only helpers（selftest 可测）
def hemisphere_align(quats):
    """四元数半球连续化（wxyz）：逐帧与前一帧 dot<0 则取反。幂等、不改旋转；
    数据 quat 可能逐帧符号翻转（Euler 导出的天然连续），resample_quat 内部
    也做一次对齐，这里显式前置以保证语义与可测性。"""
    q = np.asarray(quats, dtype=np.float64).copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def split_runs(trans_xy, jump_thresh_m=JUMP_THRESH_M, min_frames=MIN_RUN_FRAMES):
    """odom 跳变切段：帧间 XY 位移模长 > jump_thresh_m 判跳变（teleport/
    复位），episode 切成 run；长度 < min_frames 的段丢弃。

    Returns (segments, kept_idx, n_jumps)：
      segments = [(start, stop), ...]（stop exclusive，切段后全部原始段）
      kept_idx = 保留下来的段在 segments 中的下标（dropped = 全段数 - len）
      n_jumps  = 跳变次数（= 段数 - 1）
    run_idx 编号用 segments 的原始下标（含被丢弃段），保证 manifest 可回溯。
    """
    xy = np.asarray(trans_xy, dtype=np.float64)[:, :2]
    n = len(xy)
    if n == 0:
        return [], [], 0
    if n == 1:
        segs = [(0, 1)]
        return segs, ([] if 1 < min_frames else [0]), 0
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cuts = [0] + [int(i) + 1 for i in np.where(d > jump_thresh_m)[0]] + [n]
    segs = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    kept = [i for i, (a, b) in enumerate(segs) if b - a >= min_frames]
    return segs, kept, len(segs) - 1


def assert_joint_signature(jp29, quat):
    """关节签名断言（逐 run，纯 numpy）——验证 29 关节 = MuJoCo 序假设。

    per-run 只拦粗错（不可能值）：膝 idx3/idx9 max<2.2（限位 ~2.05）且
    median>-0.25（膝大幅持续负弯=序错/镜像特征）；L 肩滚 idx16
    median>=-0.2；R 肩滚 idx23 median<=0.2。膝 median<=0.05 不再报错
    （站立 episode 膝位自然在 0 附近——冒烟实测 6/6 被旧口径误杀，
    09-14 修正），降级为 warning 供全集聚合审阅（正偏证据看
    signature_summary.json 的全量 min/max 表，roundtrip MAE 仍为关节序
    终极兜底）。四元数模长 ∈[0.98,1.02]：越界 -> 归一化由调用方执行，
    这里记 warning；非有限或近零 -> 直接失败。任一硬断言失败 raise
    ValueError（调用方记 manifest error 条目跳过该 run，不猜测）。
    返回 warnings 列表。
    """
    jp = np.asarray(jp29, dtype=np.float64)
    q = np.asarray(quat, dtype=np.float64)
    warn = []
    if jp.shape[1] != 29:
        raise ValueError(f"jp29 has {jp.shape[1]} cols, expected 29")
    nrm = np.linalg.norm(q, axis=1)
    if not np.all(np.isfinite(nrm)) or float(np.min(nrm)) < 0.5:
        raise ValueError(f"quat norm invalid: min={np.min(nrm):.3f} max={np.max(nrm):.3f}")
    bad = (nrm < 0.98) | (nrm > 1.02)
    if bad.any():
        warn.append(f"quat norm outside [0.98,1.02] on {int(bad.sum())}/{len(nrm)} "
                    f"frames (min {nrm.min():.4f} max {nrm.max():.4f}); normalized")
    for idx in (3, 9):  # 左/右膝
        med, mx = float(np.median(jp[:, idx])), float(jp[:, idx].max())
        if not (med > -0.25 and mx < 2.2):
            raise ValueError(f"knee joint idx{idx} signature failed: median={med:.3f} "
                             f"max={mx:.3f} (expect median>-0.25, max<2.2)")
        if med <= 0.05:
            warn.append(f"knee idx{idx} median {med:.3f} <= 0.05 "
                        f"(standing-like run; corpus-level check in signature_summary)")
    med16 = float(np.median(jp[:, 16]))
    if med16 < -0.2:
        raise ValueError(f"L shoulder roll idx16 median {med16:.3f} < -0.2")
    med23 = float(np.median(jp[:, 23]))
    if med23 > 0.2:
        raise ValueError(f"R shoulder roll idx23 median {med23:.3f} > 0.2")
    return warn


def xy_speed_stats(trans_xy):
    """(v_med, v_p90) m/s：XY 后向差分 × FPS_SRC 口径（源帧率，wbt 块专用，
    区别于 D044 b4lite 的 @50Hz kinematics_stats）。"""
    xy = np.asarray(trans_xy, dtype=np.float64)[:, :2]
    if len(xy) < 2:
        return 0.0, 0.0
    v = np.linalg.norm(np.diff(xy, axis=0), axis=1) * FPS_SRC
    return float(np.median(v)), float(np.percentile(v, 90))


def repo_short_name(repo):
    """repo 目录名缩写：去 G1_WBT_ 前缀、转小写。"""
    return re.sub(r"^G1_WBT_", "", repo).lower()


def make_stem(repo, episode, run_idx):
    return f"wbt_{repo_short_name(repo)}_ep{int(episode):04d}_run{int(run_idx)}"


# --------------------------------------------------------------------- selftest
def run_selftest():
    """本机 numpy-only 自测：不 import mujoco/onnxruntime/pyarrow（重依赖均
    在 main()/read_parquet_file 内延迟装载，模块导入不触发）。"""
    failures = []

    def check(name, cond, detail=""):
        print(f"[selftest] {'PASS' if cond else 'FAIL'} {name}"
              + ("" if cond or not detail else f"  ({detail})"))
        if not cond:
            failures.append(name)

    # ① odom 跳变切段（含边界：首帧跳变 / 尾段过短丢弃）
    xy = np.zeros((200, 2))
    xy[:, 0] = np.arange(200) * 0.03  # ~0.9 m/s 步进，无跳变
    xy[100:] += 10.0                  # 帧间跳变 10 m
    segs, kept, n_jumps = split_runs(xy)
    check("split: 单跳变 -> 两段全保留", segs == [(0, 100), (100, 200)] and kept == [0, 1] and n_jumps == 1)
    xy2 = np.zeros((90, 2))
    xy2[:, 0] = np.arange(90) * 0.03
    xy2[70:] += 10.0
    segs2, kept2, n_jumps2 = split_runs(xy2)
    check("split: 尾段 20 帧 <60 丢弃", segs2 == [(0, 70), (70, 90)] and kept2 == [0] and n_jumps2 == 1)
    xy3 = np.zeros((71, 2))
    xy3[1:] += 10.0  # 第 0->1 帧即跳变
    segs3, kept3, n_jumps3 = split_runs(xy3)
    check("split: 首帧跳变首段(1 帧)丢弃", segs3 == [(0, 1), (1, 71)] and kept3 == [1] and n_jumps3 == 1)
    check("split: 无跳变整段保留", split_runs(xy[:100]) == ([(0, 100)], [0], 0))

    # ② 关节签名断言 pass/fail
    base = np.zeros((200, 29))
    base[:, 3] = 0.6   # 左膝
    base[:, 9] = 0.66  # 右膝
    base[:, 16] = 0.1  # L 肩滚
    base[:, 23] = -0.1  # R 肩滚
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (200, 1))
    check("signature: 正常步态通过无警告", assert_joint_signature(base, quat) == [])
    flat = base.copy()
    flat[:, 3] = 0.01  # 站立位膝伸直 -> 09-14 起降级为 warning 不再报错
    wflat = assert_joint_signature(flat, quat)
    check("signature: 膝 idx3 median<=0.05 只警告不报错",
          len(wflat) == 1 and "knee idx3" in wflat[0] and "standing-like" in wflat[0])
    negk = base.copy()
    negk[:, 9] = -0.4  # 膝大幅持续负弯 -> 序错/镜像硬特征
    try:
        assert_joint_signature(negk, quat)
        knee_fail = False
    except ValueError:
        knee_fail = True
    check("signature: 膝 idx9 median<-0.25 报错跳过", knee_fail)
    lroll = base.copy()
    lroll[:, 16] = -0.5  # L 肩滚越界（若 L/R 腿序错会镜像出此类特征）
    try:
        assert_joint_signature(lroll, quat)
        roll_fail = False
    except ValueError:
        roll_fail = True
    check("signature: L 肩滚 idx16 median<-0.2 报错跳过", roll_fail)
    wq = assert_joint_signature(base, quat * 1.05)  # 模长 1.05 越界
    check("signature: 四元数模长越界只警告不报错", len(wq) == 1 and "quat norm" in wq[0])

    # ③ 半球连续化（180° 翻转序列）
    ang = np.linspace(0.0, np.pi / 2.0, 50)
    qz = np.stack([np.cos(ang / 2), np.zeros(50), np.zeros(50), np.sin(ang / 2)], axis=1)
    qflip = qz.copy()
    qflip[1::2] *= -1.0  # 隔帧符号翻转（同一旋转）
    qa = hemisphere_align(qflip)
    dots = np.einsum("ij,ij->i", qa[1:], qa[:-1])
    same = np.minimum(np.abs(qa - qz).max(axis=1), np.abs(qa + qz).max(axis=1))
    check("hemisphere: 连续 dot>=0 且旋转不变(±q)", bool(dots.min() >= 0) and float(same.max()) < 1e-12)

    # ④ 30->50 重采样帧数（encode_bones_smoke.resample 源码口径：
    #    n_out = int(round(n_in * fps_out / fps_in))，round 而非 ceil，如 92->153）
    anchors = {90: 150, 91: 152, 92: 153, 150: 250, 900: 1500, 1234: 2057}
    ok_all, detail = True, ""
    for n, want in anchors.items():
        got = len(resample(np.zeros((n, 2)), FPS_SRC, FPS_ENC))
        if got != want:
            ok_all, detail = False, f"n={n}: got {got}, want {want} (int(round(n*5/3)))"
    check("resample 30->50 帧数 = int(round(n*5/3))", ok_all, detail)
    rs = resample(np.full((90, 3), 2.5), FPS_SRC, FPS_ENC)
    check("resample: 常数列线性插值不变", len(rs) == 150 and bool(np.allclose(rs, 2.5)))

    # ⑤ stem 命名
    s1 = make_stem("G1_WBT_Brainco_Walk_To_Table_Put_Cups_And_Rack_Plates_In_Dishwasher", 3, 0)
    check("stem: 去 G1_WBT_ 前缀小写 + ep 四位补零",
          s1 == "wbt_brainco_walk_to_table_put_cups_and_rack_plates_in_dishwasher_ep0003_run0")
    s2 = make_stem("G1_WBT_Inspire_Put_Drinks_Into_Fridge", 42, 2)
    check("stem: 常规命名", s2 == "wbt_inspire_put_drinks_into_fridge_ep0042_run2")

    # ⑥ --batch-encode 批推理路径（mock session 确定性恒等投影，numpy-only，
    #    不 import onnxruntime——批路径已抽成「obs 数组 + session 对象」纯函数）
    class _MockEnc:
        def __init__(self, scale=1.0):
            self.scale = scale

        def run(self, _, feed):
            obs = feed[list(feed)[0]]
            return [obs @ np.eye(obs.shape[-1], dtype=obs.dtype)[:, :64] * self.scale]

    class _MockDec:
        def run(self, _, feed):
            obs = feed[list(feed)[0]]
            return [obs @ np.eye(obs.shape[-1], dtype=obs.dtype)[:, :29]]

    rng = np.random.default_rng(20260914)
    obs_small = rng.standard_normal((7, 1762)).astype(np.float32)
    t_batch = encode_tokens_batch(obs_small, _MockEnc(), "obs")
    t_loop = encode_tokens_loop(obs_small, _MockEnc(), "obs")
    dec_small = rng.standard_normal((5, 994)).astype(np.float32)
    a_batch = decode_actions_batch(dec_small, _MockDec(), "i", "o")
    md = _MockDec()
    a_loop = np.stack([md.run(None, {"i": dec_small[t][None]})[0][0]
                       for t in range(len(dec_small))])
    check("batch-encode: 批/逐帧收集+推理路径逻辑等（encoder (7,1762)->(7,64) "
          "+ decoder (5,994)->(5,29)，mock session 恒等投影）",
          t_batch.shape == (7, 64) and t_batch.dtype == np.float32
          and bool(np.allclose(t_batch, t_loop, atol=1e-6))
          and a_batch.shape == (5, 29) and bool(np.allclose(a_batch, a_loop, atol=1e-6)))

    # 护栏：等值放行；mock 第二个 session（scale 不同 -> 返回不同值）-> RuntimeError
    try:
        assert_batch_encode_equivalence(t_batch, t_loop,
                                        lattice_rate_of(t_batch), lattice_rate_of(t_loop))
        guard_pass = True
    except RuntimeError:
        guard_pass = False
    t_bad = encode_tokens_batch(obs_small, _MockEnc(scale=1.01), "obs")
    guard_raised, guard_msg = False, ""
    try:
        assert_batch_encode_equivalence(t_bad, t_loop,
                                        lattice_rate_of(t_bad), lattice_rate_of(t_loop))
    except RuntimeError as exc:
        guard_raised, guard_msg = True, str(exc)
    check("batch-encode: 等价护栏等值通过 / 不等值抛 RuntimeError 且消息含两侧数值",
          guard_pass and guard_raised and "lattice_rate" in guard_msg)

    # 默认关 = 调用链不变：CLI default False（store_true）+ encode_segment 默认 False
    import inspect
    sig = inspect.signature(encode_segment)
    check("batch-encode: 默认关（CLI parse_args([]).batch_encode is False + "
          "encode_segment 签名默认 False）",
          build_arg_parser().parse_args([]).batch_encode is False
          and sig.parameters["batch_encode"].default is False)

    if failures:
        print(f"[selftest] FAILED: {len(failures)} checks: {failures}")
        return 1
    print("[selftest] ALL PASS")
    return 0


# ------------------------------------ --batch-encode 批推理纯函数（selftest 可测）
# 模块级护栏状态：首个成功编码 run 完成批/逐帧等价验证后置 True，之后的 run
# 只走批路径（验证只做一次，避免此后每 run 双跑浪费；验证失败则保持 False，
# 各 run 继续触发护栏即继续响亮失败，绝不静默回退逐帧路径）。
_BATCH_ENCODE_VERIFIED = False


def collect_obs_batch(n_rs, jp_isaac, jv_isaac, bq, apply_delta, anchor="ref-rel"):
    """--batch-encode 用：逐帧 build_obs（调用方式与默认逐帧循环一字不差）
    收集为 (n_rs,1762) f32。obs 构建本非瓶颈（瓶颈是 ONNX session 逐帧调用
    开销），故只批推理、不批构建。"""
    obs_batch = np.zeros((n_rs, 1762), dtype=np.float32)
    for t in range(n_rs):
        obs_batch[t] = build_obs(t, jp_isaac, jv_isaac, bq, apply_delta, anchor=anchor)
    return obs_batch


def encode_tokens_loop(obs_batch, session, iname):
    """逐帧编码路径（原行为的函数化镜像：逐行送 session、[0][0] 取帧），供
    --batch-encode 首段护栏与 selftest 对照；encode_segment 默认分支仍走其
    内联原循环。"""
    tokens = np.zeros((len(obs_batch), 64), dtype=np.float32)
    for t in range(len(obs_batch)):
        tokens[t] = session.run(None, {iname: obs_batch[t][None]})[0][0].astype(np.float32)
    return tokens


def encode_tokens_batch(obs_batch, session, iname):
    """批编码路径：obs (n,1762) 一次 session.run 取回整块 (n,64) tokens
    （enc.run 返回结构为 [输出数组]，批模式取 [0]，区别于逐帧的 [0][0]）。"""
    out = session.run(None, {iname: np.asarray(obs_batch)})
    return np.asarray(out[0], dtype=np.float32)


def decode_actions_batch(dec_obs_batch, session, iname, oname):
    """roundtrip 批解码：build_decoder_obs 逐帧收集的 (n,994) 一次
    session.run 取回 (n,29) actions（原逐帧 [0][0] 的批对应物是 [0]）。"""
    out = session.run([oname], {iname: np.asarray(dec_obs_batch)})
    return out[0]


def lattice_rate_of(tokens):
    """lattice 命中率（与 encode_segment 主路径内联公式同式，护栏对照用）。"""
    lat = np.asarray(tokens, dtype=np.float64) * 16.0
    return float((np.abs(lat - np.round(lat)) > LATTICE_TOL).mean())


def assert_batch_encode_equivalence(tokens_batch, tokens_loop, lr_batch, lr_loop):
    """--batch-encode 首段等价护栏：max|tokens_batch - tokens_loop| <= 1e-4 且
    lattice_rate np.isclose；不符 raise RuntimeError（消息含两侧数值），
    不许静默回退。"""
    diff = float(np.max(np.abs(np.asarray(tokens_batch, dtype=np.float64)
                               - np.asarray(tokens_loop, dtype=np.float64))))
    if diff > 1e-4 or not bool(np.isclose(lr_batch, lr_loop)):
        raise RuntimeError(
            f"--batch-encode equivalence guard FAILED (no silent fallback): "
            f"max|tokens_batch-tokens_loop|={diff:.3e} (tol 1e-4); "
            f"lattice_rate batch={lr_batch:.6e} loop={lr_loop:.6e}")


# ------------------------------------------------------------- encoder chain
def encode_segment(dof_mj, quat, trans_m, enc, iname, m2i, default_mj,
                   anchor="ref-rel", dec_bundle=None, batch_encode=False):
    """WBT run arrays（rad、wxyz、m；native 30 fps，MuJoCo order）-> tokens +
    reference arrays + optional decoder roundtrip。逐字镜像
    convert_bones_g1_csv.py encode_segment（D044 / encode_bones_smoke 主流程），
    仅根四元数来源不同：数据自带 wxyz，先 hemisphere_align 半球连续化再
    resample_quat（无 euler_to_quat/calibrate）；FPS_SRC 为模块级 30。
    batch_encode=False（默认）走原逐帧路径；True 走批推理路径（首个成功 run
    经 assert_batch_encode_equivalence 护栏验证等价，见模块 docstring）。"""
    global _BATCH_ENCODE_VERIFIED
    dof_mj = np.asarray(dof_mj, dtype=np.float64)
    quat = hemisphere_align(np.asarray(quat, dtype=np.float64))
    dof_mj_rs = resample(dof_mj, FPS_SRC, FPS_ENC)
    quat_rs = resample_quat(quat, FPS_SRC, FPS_ENC)
    trans_rs = resample(trans_m, FPS_SRC, FPS_ENC)
    n_rs = len(dof_mj_rs)

    jp_mj = dof_mj_rs
    jp_isaac = jp_mj[:, m2i]
    # planner_sonic.py L109: finite diff gives rad/step; encoder wants rad/s
    jv_mj = np.vstack([np.zeros((1, 29)), np.diff(jp_mj, axis=0) * FPS_ENC])
    jv_isaac = jv_mj[:, m2i]
    bq = np.asarray(quat_rs, dtype=np.float64).reshape(-1, 1, 4)
    apply_delta = _qn(_qmul(_heading(np.array([1.0, 0, 0, 0])), _heading_inv(bq[0, 0])))

    tokens = np.zeros((n_rs, 64), dtype=np.float32)
    if not batch_encode:
        for t in range(n_rs):
            obs = build_obs(t, jp_isaac, jv_isaac, bq, apply_delta, anchor=anchor)
            tokens[t] = enc.run(None, {iname: obs[None]})[0][0].astype(np.float32)
    else:
        obs_batch = collect_obs_batch(n_rs, jp_isaac, jv_isaac, bq, apply_delta,
                                      anchor=anchor)
        tokens = encode_tokens_batch(obs_batch, enc, iname)
        if not _BATCH_ENCODE_VERIFIED:
            # 首个成功编码 run 的运行时护栏：同时用原逐帧路径算一份 tokens_loop，
            # 数值等价（tol 1e-4 + lattice_rate 一致）才置已验证并放行后续批路径。
            # 失败 = SystemExit 直接中止进程（reviewer note：逐 run 记 error 继续
            # 跑会空烧双份推理且 converted 0/N，宁可响亮退出不回退）。
            tokens_loop = encode_tokens_loop(obs_batch, enc, iname)
            try:
                assert_batch_encode_equivalence(tokens, tokens_loop,
                                                lattice_rate_of(tokens),
                                                lattice_rate_of(tokens_loop))
            except RuntimeError as exc:
                raise SystemExit(f"[batch-encode] guard failed, aborting: {exc}")
            _BATCH_ENCODE_VERIFIED = True

    # ref-rel sanity at t=0: f=0 anchor is exactly identity
    obs0 = build_obs(0, jp_isaac, jv_isaac, bq, apply_delta, anchor=anchor)
    sanity = float(np.abs(obs0[601:607] - np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)).max())
    if sanity > 1e-5:
        raise RuntimeError(f"f=0 anchor sanity broken: {sanity}")

    lat = tokens.astype(np.float64) * 16.0
    lattice_rate = float((np.abs(lat - np.round(lat)) > LATTICE_TOL).mean())

    out = {
        "tokens": tokens, "jp_mj": jp_mj, "jp_isaac": jp_isaac,
        "jv_isaac": jv_isaac, "quat_wxyz": quat_rs, "trans_m": trans_rs,
        "lattice_rate": lattice_rate, "anchor_sanity": sanity, "n_rows": n_rs,
        "roundtrip_mae": None, "roundtrip_mae_default_baseline": None,
    }

    if dec_bundle is not None:
        env = dec_bundle["env"]
        dec = env.sonic_decoder
        m2i_b, default_mj_b = dec_bundle["m2i"], dec_bundle["default_mj"]
        # body-frame angular velocity from root quat finite diff (smoke verbatim)
        omega_body = np.zeros((n_rs, 3))
        for t in range(n_rs):
            a, b = quat_rs[min(t + 1, n_rs - 1)], quat_rs[max(t - 1, 0)]
            step = (min(t + 1, n_rs - 1) - max(t - 1, 0)) / FPS_ENC
            dq = _qmul(a, _qconj(b))
            if dq[0] < 0:
                dq = -dq
            w_world = 2.0 * dq[1:] / max(dq[0], 1e-6) / max(step, 1e-6)
            omega_body[t] = _quat_rotate_inverse(quat_rs[t], w_world)
        grav = np.array([_quat_rotate_inverse(qq, np.array([0.0, 0.0, -1.0])) for qq in quat_rs])

        if batch_encode:
            dec_obs_batch = np.zeros((n_rs, int(dec.input_dim)), dtype=np.float32)
            for t in range(n_rs):
                idx = np.clip(np.arange(t - 9, t + 1), 0, n_rs - 1)
                hist = {
                    "base_angular_velocity": omega_body[idx].astype(np.float32),
                    "body_joint_positions": ((jp_mj[idx] - default_mj_b)[:, m2i_b]).astype(np.float32),
                    "body_joint_velocities": jv_isaac[idx].astype(np.float32),
                    "last_actions": (((jp_mj[idx] - default_mj_b) / env.sonic_scale_mujoco)[:, m2i_b]).astype(np.float32),
                    "gravity_dir": grav[idx].astype(np.float32),
                }
                dec_obs_batch[t] = dec.build_decoder_obs(tokens[t], hist)[0]
            acts = decode_actions_batch(dec_obs_batch, dec.session,
                                        dec.input_name, dec.output_name)
            # 与逐帧公式完全同式（向量化）：q_des = sonic_default_isaac +
            # acts*sonic_scale_isaac，与 jp_isaac 逐帧绝对差；err0（default
            # baseline）不受批影响照旧。
            q_des_isaac = env.sonic_default_isaac + acts.astype(np.float64) * env.sonic_scale_isaac
            err = np.abs(q_des_isaac - jp_isaac)
            err0 = np.abs(default_mj_b[m2i_b][None, :] - jp_isaac)
        else:
            err, err0 = [], []
            for t in range(n_rs):
                idx = np.clip(np.arange(t - 9, t + 1), 0, n_rs - 1)
                hist = {
                    "base_angular_velocity": omega_body[idx].astype(np.float32),
                    "body_joint_positions": ((jp_mj[idx] - default_mj_b)[:, m2i_b]).astype(np.float32),
                    "body_joint_velocities": jv_isaac[idx].astype(np.float32),
                    "last_actions": (((jp_mj[idx] - default_mj_b) / env.sonic_scale_mujoco)[:, m2i_b]).astype(np.float32),
                    "gravity_dir": grav[idx].astype(np.float32),
                }
                obs = dec.build_decoder_obs(tokens[t], hist)
                act_isaac = dec.session.run([dec.output_name], {dec.input_name: obs})[0][0]
                q_des_isaac = env.sonic_default_isaac + act_isaac.astype(np.float64) * env.sonic_scale_isaac
                err.append(np.abs(q_des_isaac - jp_isaac[t]))
                err0.append(np.abs(default_mj_b[m2i_b] - jp_isaac[t]))
        out["roundtrip_mae"] = float(np.mean(err))
        out["roundtrip_mae_default_baseline"] = float(np.mean(err0))
    return out


# --------------------------------------------------------------- parquet load
def read_parquet_file(path):
    """单 parquet 文件 schema 自适应读取（仿 tmp/wbt_probe.py read_file）。
    返回 {schema, ep, frame, root7, j29}；frame 为帧序（frame_index，退而
    index，再退 None=按行序）。pyarrow 延迟装载，模块导入/selftest 不触发。"""
    import pyarrow.parquet as pq  # noqa: PLC0415  服务器专用

    t = pq.read_table(path)
    names = set(t.column_names)

    def col(name):
        return np.asarray(t.column(name).to_pylist(), dtype=np.float64)

    ep = np.asarray(t.column("episode_index").to_pylist(), dtype=np.int64)
    if "frame_index" in names:
        frame = np.asarray(t.column("frame_index").to_pylist(), dtype=np.int64)
    elif "index" in names:
        frame = np.asarray(t.column("index").to_pylist(), dtype=np.int64)
    else:
        frame = None
    if "observation.state.robot_q_current" in names:
        schema = "q36"
        q36 = col("observation.state.robot_q_current")
        if q36.ndim != 2 or q36.shape[1] != 36:
            raise ValueError(f"{path}: robot_q_current shape {q36.shape} != (n,36)")
        root7, j29 = q36[:, :7], q36[:, 7:36]
    else:
        schema = "grouped"
        missing = [g for g in GROUPED_JOINT_COLS if g not in names]
        if missing or "observation.state.state_base_pose" not in names:
            raise ValueError(f"{path}: neither q36 nor complete grouped schema "
                             f"(missing {missing})")
        root7 = col("observation.state.state_base_pose")
        # 拼接顺序即契约：lower_body[15] + left_arm[7] + right_arm[7] = 29
        j29 = np.concatenate([col(g) for g in GROUPED_JOINT_COLS], axis=1)
    if root7.ndim != 2 or root7.shape[1] != 7 or j29.shape[1] != 29:
        raise ValueError(f"{path}: root7 {root7.shape} / j29 {j29.shape} not (n,7)/(n,29)")
    return {"schema": schema, "ep": ep, "frame": frame, "root7": root7, "j29": j29}


def load_repo_episodes(raw_root, repo):
    """读一个 repo 的全部 data parquet，按 episode_index 分 episode（帧序
    排序 + 帧步检查），返回 episode 记录列表（含 schema 与 repo 级 warnings）。"""
    repo_dir = os.path.join(raw_root, repo)
    files = sorted(glob.glob(os.path.join(repo_dir, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise ValueError(f"no data parquet under {repo_dir}/data")
    recs = [read_parquet_file(f) for f in files]

    repo_warn = []
    info_p = os.path.join(repo_dir, "meta", "info.json")
    if os.path.isfile(info_p):
        with open(info_p, encoding="utf-8") as f:
            fps_info = json.load(f).get("fps")
        if fps_info is not None and float(fps_info) != FPS_SRC:
            repo_warn.append(f"meta/info.json fps={fps_info} != contract FPS_SRC={FPS_SRC}")

    ep_all = np.concatenate([r["ep"] for r in recs])
    root_all = np.concatenate([r["root7"] for r in recs])
    j_all = np.concatenate([r["j29"] for r in recs])
    schema_all = np.concatenate([np.full(len(r["ep"]), r["schema"], dtype=object) for r in recs])
    if all(r["frame"] is not None for r in recs):
        frame_all = np.concatenate([r["frame"] for r in recs])
    else:
        frame_all = None
        if any(r["frame"] is not None for r in recs):
            repo_warn.append("frame-index availability mixed across files; row order assumed")

    episodes = []
    for e in np.unique(ep_all):
        m = np.where(ep_all == e)[0]
        ep_warn = list(repo_warn)
        if frame_all is not None:
            m = m[np.argsort(frame_all[m], kind="stable")]
            fd = np.diff(frame_all[m])
            if not np.all(fd == 1):
                ep_warn.append(f"frame step not constant: {np.unique(fd)[:5]}")
        schemas = sorted(set(schema_all[m]))
        if len(schemas) != 1:
            raise ValueError(f"ep{e}: mixed schemas within episode: {schemas}")
        root7 = root_all[m]
        v_med, _ = xy_speed_stats(root7[:, :2])
        episodes.append({"repo": repo, "ep": int(e), "schema": schemas[0],
                         "root7": root7, "j29": j_all[m], "v_med": v_med,
                         "warnings": ep_warn})
    return episodes


def sample_episodes(episodes, n_sample):
    """冒烟模式：按 episode 整段 v_med 排序，快/中/慢三档各取 max(1, N//3)
    个（池 = 全部选中 repo 的 episode；确定性，无随机）。"""
    eps = sorted(episodes, key=lambda r: (r["v_med"], r["repo"], r["ep"]))
    k = max(1, n_sample // 3)
    n = len(eps)
    mid0 = max(0, n // 2 - k // 2)
    picks = eps[:k] + eps[mid0:mid0 + k] + eps[max(0, n - k):]
    seen, out = set(), []
    for r in picks:
        key = (r["repo"], r["ep"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ------------------------------------------------------------- limits-check
def load_deploy_joint_limits():
    """--limits-check 专用：import mujoco 载入 deploy MJCF（env 默认
    scene_43dof.xml），按关节名取 29 关节 (lo,hi)（MuJoCo 序）。仅服务器。"""
    import mujoco  # noqa: PLC0415  服务器专用

    xml = f"{REPO}/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
    model = mujoco.MjModel.from_xml_path(xml)
    lims = np.zeros((29, 2))
    for i, nm in enumerate(MUJOCO_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        if jid < 0:
            raise ValueError(f"joint {nm} not found in deploy MJCF")
        lims[i] = model.jnt_range[jid]
    return lims


def check_joint_limits(jp29, lims, margin=LIMIT_MARGIN_RAD):
    """断言该 run 的 29 关节观测值域 ⊂ 限位 ± margin rad；违例 raise。"""
    lo, hi = lims[:, 0] - margin, lims[:, 1] + margin
    jmin, jmax = jp29.min(axis=0), jp29.max(axis=0)
    bad = np.where((jmin < lo) | (jmax > hi))[0]
    if len(bad):
        det = ", ".join(f"idx{int(i)} obs[{jmin[i]:.3f},{jmax[i]:.3f}] "
                        f"lim[{lo[i]:.3f},{hi[i]:.3f}]" for i in bad[:5])
        raise ValueError(f"joint values outside deploy limits +/-{margin} rad: {det}")


# -------------------------------------------------------------------- main
def build_arg_parser():
    """CLI 构造独立成函数：selftest ⑥ 用 parse_args([]) 验证 --batch-encode
    默认 False（默认关 = 逐帧原始行为）。"""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT,
                    help="UnifoLM-WBT 原始数据根（<RepoName>/{meta,data}）")
    ap.add_argument("--repos", default=None,
                    help="逗号分隔 raw-root 下的 repo 目录名（缺省 = 全部含 data/ 的）")
    ap.add_argument("--sample-episodes", type=int, default=0,
                    help="冒烟模式：按 episode v_med 分快/中/慢三档各取 N/3 个")
    ap.add_argument("--out-dir", default=None,
                    help="缺省 = raw-root 同级 g1wbt_conv")
    ap.add_argument("--roundtrip", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0,
                    help="调试上限：本次写入 manifest 的 run 条目数（成功+错误）")
    ap.add_argument("--limits-check", action="store_true",
                    help="（服务器）mujoco 载入 deploy MJCF，断言观测值域 ⊂ 限位±0.15 rad")
    ap.add_argument("--fresh-manifest", action="store_true",
                    help="不合并旧 manifest，整体重写（默认按 stem 合并，防分批转换"
                         "覆盖丢失先前段，语义照抄 D044）")
    ap.add_argument("--batch-encode", action="store_true",
                    help="性能旗标（默认关 = 逐帧原始行为，逐位不变）：obs 逐帧收集后"
                         "并成一次批 ONNX 推理（encoder/decoder 各一次/run；服务器实测"
                         "逐帧 ~30-40 帧/s、全量需数十小时，瓶颈是 session 逐帧调用开销）。"
                         "首个成功 run 由护栏验证批/逐帧数值等价（max|Δtokens|<=1e-4 且 "
                         "lattice_rate 一致），不符报错退出、不静默回退（D039 旗标先例）")
    ap.add_argument("--selftest", action="store_true",
                    help="本机 numpy-only 自测，不碰重依赖与数据")
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    if not os.path.isdir(args.raw_root):
        ap.error(f"--raw-root not a dir: {args.raw_root}")
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.raw_root)), "g1wbt_conv")
    if args.repos:
        repos = [r.strip() for r in args.repos.split(",") if r.strip()]
        for r in repos:
            if not os.path.isdir(os.path.join(args.raw_root, r)):
                ap.error(f"repo dir not found under raw-root: {r}")
    else:
        repos = sorted(d for d in os.listdir(args.raw_root)
                       if os.path.isdir(os.path.join(args.raw_root, d, "data")))
        if not repos:
            ap.error(f"no <repo>/data dirs under {args.raw_root}")
    print(f"[wbt] raw_root={args.raw_root} repos={repos} -> out_dir={out_dir}")

    # ---- 重依赖装载（镜像 convert_bones_g1_csv.py main；py_compile/selftest 不触发）
    import onnxruntime as ort
    from apt_g1.envs.mujoco_g1_flat_env import (
        G1_MUJOCO_TO_ISAACLAB_DOF,
        SONIC_DEFAULT_ANGLES_MUJOCO,
        MujocoG1FlatEnv,
    )
    from eval_torque_srb import NoQuantDecoder

    m2i = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)
    default_mj = SONIC_DEFAULT_ANGLES_MUJOCO.astype(np.float64)
    enc = ort.InferenceSession(
        f"{REPO}/gear_sonic_deploy/policy/release/model_encoder.onnx",
        providers=["CPUExecutionProvider"])
    ins = enc.get_inputs()
    assert len(ins) == 1 and ins[0].shape[-1] == 1762, f"encoder input {ins[0].shape}"

    lims = load_deploy_joint_limits() if args.limits_check else None

    npz_dir = os.path.join(out_dir, "npz")
    os.makedirs(npz_dir, exist_ok=True)
    dec_bundle = None
    if args.roundtrip:
        env = MujocoG1FlatEnv(NoQuantDecoder(
            f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"), REPO,
            use_elastic_band=False, stand_only=True)
        dec_bundle = {"env": env, "m2i": m2i, "default_mj": default_mj}

    # ---- 读数据（每文件 schema 自适应），按 episode 分组
    episodes = []
    for repo in repos:
        try:
            episodes.extend(load_repo_episodes(args.raw_root, repo))
        except Exception as e:  # noqa: BLE001  记录不阻塞其它 repo
            print(f"[load] FAIL repo {repo}: {type(e).__name__}: {e}", flush=True)
    if not episodes:
        ap.error("no episodes loaded (check --raw-root/--repos)")
    if args.sample_episodes:
        episodes = sample_episodes(episodes, args.sample_episodes)
        print(f"[wbt] sample-episodes={args.sample_episodes}: kept {len(episodes)} "
              f"episodes (v_med {episodes[0]['v_med']:.2f}..{episodes[-1]['v_med']:.2f} m/s)")

    # ---- 逐 episode：跳变切段 -> 逐 run：签名断言 -> 编码 -> npz + manifest
    manifest, skipped, n_entries, stop = [], 0, 0, False
    jp_lo = np.full(29, np.inf)   # 全集聚合关节值域（MuJoCo 序 name-assert 证据）
    jp_hi = np.full(29, -np.inf)
    knee_med_agg = []
    for ep_rec in episodes:
        if stop:
            break
        repo, ep = ep_rec["repo"], ep_rec["ep"]
        root7, j29 = ep_rec["root7"], ep_rec["j29"]
        segs, kept_idx, n_jumps = split_runs(root7[:, :2])
        n_dropped = len(segs) - len(kept_idx)
        kept_set = set(kept_idx)
        if not kept_idx:
            manifest.append({
                "stem": make_stem(repo, ep, 0), "class": "walk", "actor": "wbt",
                "error": f"no run >= {MIN_RUN_FRAMES} frames ({FPS_SRC:.0f}Hz) after jump split",
                "wbt": {"repo": repo, "episode": ep, "run_idx": 0,
                        "schema": ep_rec["schema"], "fps_src": FPS_SRC,
                        "n_jumps": int(n_jumps), "dropped_runs": int(n_dropped)},
            })
            n_entries += 1
            continue
        for run_idx, (a, b) in enumerate(segs):
            if run_idx not in kept_set:
                continue
            if args.max_runs and n_entries >= args.max_runs:
                stop = True
                break
            stem = make_stem(repo, ep, run_idx)
            npz_path = os.path.join(npz_dir, stem + ".npz")
            if os.path.isfile(npz_path) and not args.force:
                skipped += 1
                continue
            jp29, quat, trans = j29[a:b], root7[a:b, 3:7], root7[a:b, :3]
            v_med, v_p90 = xy_speed_stats(trans[:, :2])
            wbt_block = {
                "repo": repo, "episode": ep, "run_idx": int(run_idx),
                "schema": ep_rec["schema"], "fps_src": FPS_SRC,
                "v_med_src": v_med, "v_p90_src": v_p90,
                "n_jumps": int(n_jumps), "dropped_runs": int(n_dropped),
            }
            try:
                warn = list(ep_rec["warnings"]) + assert_joint_signature(jp29, quat)
                if lims is not None:
                    check_joint_limits(jp29, lims)
                # 四元数统一归一化（模长 warning 场景的修复动作）
                quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
                seg = encode_segment(jp29, quat, trans, enc, ins[0].name, m2i,
                                     default_mj, dec_bundle=dec_bundle,
                                     batch_encode=args.batch_encode)
            except Exception as e:  # noqa: BLE001  单 run 失败记条目跳过，不猜测
                print(f"[convert] FAIL {stem}: {type(e).__name__}: {e}", flush=True)
                manifest.append({"stem": stem, "class": "walk", "actor": "wbt",
                                 "error": f"{type(e).__name__}: {e}", "wbt": wbt_block})
                n_entries += 1
                continue
            meta = {
                "stem": stem, "class": "walk", "actor": "wbt",
                "n_rows_src": int(b - a), "n_rows_enc": seg["n_rows"],
                "lattice_rate": seg["lattice_rate"],
                "roundtrip_mae": seg["roundtrip_mae"],
                "roundtrip_mae_default_baseline": seg["roundtrip_mae_default_baseline"],
                "path_len_m": float(np.linalg.norm(
                    np.diff(seg["trans_m"][:, :2], axis=0), axis=1).sum()),
                "npz_path": os.path.abspath(npz_path),
                "warnings": warn,
                # D057 wbt 块：repo/episode/run 溯源 + 切段计数 + 源速度口径
                "wbt": wbt_block,
            }
            np.savez(npz_path,
                     tokens=seg["tokens"], jp_isaac=seg["jp_isaac"], jp_mj=seg["jp_mj"],
                     jv_isaac=seg["jv_isaac"], quat_wxyz=seg["quat_wxyz"],
                     trans_m=seg["trans_m"], meta=np.array(json.dumps(meta)))
            manifest.append(meta)
            n_entries += 1
            jp_lo = np.minimum(jp_lo, seg["jp_mj"].min(axis=0))
            jp_hi = np.maximum(jp_hi, seg["jp_mj"].max(axis=0))
            knee_med_agg.append(float(np.median(seg["jp_mj"][:, 3])))
            rt = "" if seg["roundtrip_mae"] is None else f" rt_mae={seg['roundtrip_mae']:.4f}"
            print(f"[convert] {repo} ep{ep} run{run_idx}: {b - a}@30 -> "
                  f"{seg['n_rows']}@50 v_med={v_med:.2f} lat={seg['lattice_rate']:.1e}{rt}",
                  flush=True)

    mpath = os.path.join(out_dir, "manifest.json")
    # 默认与旧 manifest 按 stem 合并（本 run 条目覆盖同名，其余保留）；
    # skipped-existing 的分批转换不再整体覆盖丢失先前段（D044 坑① 语义照抄）
    if os.path.exists(mpath) and not args.fresh_manifest:
        try:
            with open(mpath) as f:
                old = json.load(f)
            new_stems = {m.get("stem") for m in manifest}
            old_keep = [e for e in old
                        if isinstance(e, dict) and e.get("stem") not in new_stems]
            manifest = old_keep + manifest
            print(f"[manifest] merged with previous run: kept {len(old_keep)}, "
                  f"total {len(manifest)}")
        except (OSError, ValueError) as exc:
            print(f"[manifest] WARN: merge failed ({exc}); overwriting")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[manifest] {len(manifest)} entries -> {mpath}")
    ok = [m for m in manifest if "error" not in m]
    rts = [m["roundtrip_mae"] for m in ok if m.get("roundtrip_mae") is not None]
    print(f"SUMMARY: converted {len(ok)}/{len(manifest)} (skipped existing {skipped})"
          + (f" | roundtrip MAE mean {np.mean(rts):.4f} rad, max {np.max(rts):.4f}" if rts else ""))
    # 全集聚合签名证据（09-14 起 per-run 膝 median 降级 warning 后的序证据载体）：
    # 只打印到 stdout（证据进运行日志；随时可从各 npz 的 jp_mj 重算，不另落文件）
    if np.isfinite(jp_lo).all():
        knees = [(i, MUJOCO_JOINT_NAMES[i], round(float(jp_lo[i]), 2), round(float(jp_hi[i]), 2))
                 for i in (3, 9)]
        rolls = [(i, MUJOCO_JOINT_NAMES[i], round(float(jp_lo[i]), 2), round(float(jp_hi[i]), 2))
                 for i in (16, 23)]
        print(f"[signature] n_runs={len(knee_med_agg)} knee_med_over_runs "
              f"min/mean/max = {min(knee_med_agg):.3f}/{np.mean(knee_med_agg):.3f}/"
              f"{max(knee_med_agg):.3f}")
        print(f"[signature] knee ranges {knees}")
        print(f"[signature] shoulder-roll ranges {rolls}")


if __name__ == "__main__":
    main()
