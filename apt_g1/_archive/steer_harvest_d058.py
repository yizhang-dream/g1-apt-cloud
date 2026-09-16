"""D058：冻结 decoder 闭环 z 导引采直线行走语料（steer-to-harvest）。

DS_CONTINUOUS_EXECUTION_PLAN §5r 预注册。骨架照抄 z_sweep_cem_d048r.py
（CemEnv 子类旁路：z→VAE.decode→冻结 decoder；PRE-step 纪律 = done 步
auto-reset 污染数据一律丢弃；AppLauncher headless 默认 + daemon close +
os._exit(0) 防挂死）。把 D048r R3 的"搜索产物"升级成"语料"：三速度带
{0.35, 0.70, 1.00} m/s 目标下，warmup CEM 搜 z 池 → fixed（开环常 z）vs
controller（闭环边走边调：0.5s 控制拍、3°/0.10 双阈值连续两拍触发、池内
argmin 换 z、12 步线性交叉淡入）三臂对照 → 最优臂滚收割 → segments 打包
D044 同构 npz 喂 build_wbt_vae_inputs（D057 G4 管线）重训 VAE。

闭环控制器（预注册口径，实现逐字对齐 §5r）：
  - 控制周期 --ctrl-every 25 步（0.5s）：每拍用最近 25 步（PRE-step、剔除
    done 污染步）算 trailing 净前向速度（初始航向系投影，z_sweep 口径）与
    yaw 误差（相对本局初始 yaw，wrap ±π）；
  - 触发条件 = |yaw_err|>3.0° 或 |v_trail−带目标|>0.10，连续两拍才切；
  - 切换目标 = 池中 argmin(|v_prior−目标| + 0.3·|yaw_rate_prior|·
    1[yaw_err 与 prior yaw_rate 同号])——速度贴带优先、偏向漂移反侧候选；
  - 切换后 --fade-steps 12 步线性交叉淡入（α=j/12，端点精确），淡入期不
    响应新触发；淡入混合在驱动侧算好写 env._steer_z，env 侧不懂淡入。
收割口径：episodic 准入 = 不摔 ∧ h_min≥0.40 ∧ 直度(net_disp/path_len)≥0.90
∧ |净前向均值−带目标|≤0.15；20s 局切 4s 窗 stride 2s（起点 0,2,...,16 共
9 窗），窗级二级滤 |v_med−目标|>0.25 剔除计数；v_med 用
build_d048n.frame_speed_bwd（XY 后向差分×50，v[0]=v[1]，零漂移 import）。
G2 改宗规则：warmup 后某带无候选 |v_prior−目标|≤0.20 → 带目标就近改宗
池中最近 v_prior 并落账（不静默改）。

六模式 --arm：selftest（纯 numpy 单测，不 import torch/isaac）/ warmup
（CEM 池）/ fixed（池中最贴带目标者常 z 开环）/ controller（闭环）/
harvest（滚收割到窗数配额）/ segments（离线打包已准入局，不启 Isaac）。

torch/isaac 全部延迟进 main()（SteerEnv 类定义也在 main() 内），本机无
torch 可跑 --selftest。

Run on lab-ts/CVGL (Isaac wrapper, cwd=GR00T-WholeBodyControl):
  nohup bash /tmp/run_apt_isaac.sh \
    /home/cvgluser/ros2_data/apt_g1/isaac/steer_harvest_d058.py \
    --latent-vae-path <vae.pt> --arm warmup --tag <vaeTag> \
    > <out>/d058/warmup.log 2>&1 < /dev/null & disown
  # fixed/controller G2：--arm controller --pool-json <warmup>/pool.json \
  #   --episodes-per-band 20 --seeds 0
  # harvest：--arm harvest --harvest-arm controller --min-windows-per-band 300
  # segments（离线打包）：--arm segments --segments-from controller
产物 <out>/d058_<tag>/<arm>/episodes/ep_b{bi}_s{seed}_{idx}.npz（整局：
tokens/trans_m/quat_wxyz/jp_isaac/z_t/meta）+ <arm>_summary.json；
warmup 另产 <out>/d058_<tag>/warmup/pool.json（每带 top --pool-per-band
精英）；segments 产 <out>/d058_<tag>/walk_segments.json（D057 同构）。
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# 服务器执行根平铺 import / 仓库根包 import 双兼容（apt_g1/__main__.py 同款
# sys.path 注入；本机 `python apt_g1/isaac/本文件 --selftest` 也能解析）
_HERE = Path(__file__).resolve().parent            # .../apt_g1/isaac
_APT = _HERE.parent                                # .../apt_g1
for _p in (str(_HERE), str(_APT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:                      # 服务器执行根平铺 import / 仓库根包 import 双兼容
    from z_sweep_cem_d048r import cem_next_distribution, cem_sample
except ImportError:
    from apt_g1.isaac.z_sweep_cem_d048r import cem_next_distribution, cem_sample
try:                      # frame_speed_bwd 零漂移 import（D057 同款惯例）
    from build_d048n_vb_speed_labels import frame_speed_bwd
except ImportError:
    from build_d048n_vb_speed_labels import frame_speed_bwd

HOME = os.path.expanduser("~")
DEFAULT_OUT_BASE = f"{HOME}/ros2_data/apt_g1/outputs/d058"
DEFAULT_ROUTER_DIR = f"{HOME}/ros2_data/apt_g1/outputs/distill_final"

FPS = 50.0                       # 上游契约：50Hz 控制步频（口径勿改）
Z_DIM = 16                       # latent z 维度（D044 同构）

# ---- §5r 预注册常量（口径冻结，不得随运行调整） --------------------------
BAND_TARGETS_DEF = "0.35,0.70,1.00"          # 三速度带目标 m/s
BAND_VB_MAP_DEF = "0.35:1,0.70:1,1.00:2"     # 带目标 → vb 解码档（冒烟默认）
YAW_TRIG_RAD = math.radians(3.0)             # yaw 触发阈 3.0°
V_TRIG = 0.10                                # 速度触发阈 0.10 m/s
TRIGGER_CONSEC = 2                           # 连续两拍才切
YAW_BIAS_W = 0.3                             # 候选评分 yaw 同号惩罚权重
RETARGET_TOL = 0.20                          # G2 改宗判据 |v−目标|≤0.20
WIN_S, STRIDE_S = 4.0, 2.0                   # 收割窗 4s、步长 2s（20s 局 9 窗）
WIN_V_TOL = 0.25                             # 窗级二级滤 |v_med−目标|
EP_H_MIN = 0.40                              # episodic 准入 h_min
STRAIGHT_MIN = 0.90                          # episodic 准入直度
EP_V_TOL = 0.15                              # episodic 准入 |净前向−目标|
UPRIGHT_MIN = 0.9                            # 终末 upright（D048r 同式判负）


# ------------------------------------------------------------- 纯 numpy 控制器
# （torch/isaac 零依赖；--selftest 覆盖本节全部函数）


def wrap_pi(a):
    """角度 wrap 到 (−π, π]（标量或数组）。"""
    a = np.asarray(a, dtype=np.float64)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def yaw_of_np(quat):
    """(m,4) w-first quat → (m,) yaw（z_sweep `_yaw_vec` 的 numpy 同式）。"""
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def upright_np(quat):
    """(m,4) quat → (m,) upright=exp(−g_xy²/0.1)（eval 同式：直立 1）。"""
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    g_xy = 2.0 * np.sqrt((x * z - w * y) ** 2 + (y * z + w * x) ** 2)
    return np.exp(-(g_xy ** 2) / 0.1)


def trailing_v_fwd(xy_a, xy_b, f0, n_steps):
    """a→b trailing 净前向速度（初始航向系投影；z_sweep 净前向口径的时间窗
    版）。n_steps=两锚点间实际经过的控制步数（done 污染步不作数据点，但
    墙钟照走——速度分母按真实步数计）。"""
    dt = max(int(n_steps), 1) / FPS
    d = np.asarray(xy_b, dtype=np.float64) - np.asarray(xy_a, dtype=np.float64)
    return float(np.dot(d, np.asarray(f0, dtype=np.float64))) / dt


def trigger_update(consec, cond, fading):
    """一拍触发判定（§5r 预注册）：淡入期不评估（计数清零、不切）；条件成立
    计数 +1、连续 TRIGGER_CONSEC 拍才切（切后计数清零归零重计）；条件不成立
    计数清零。返回 (consec', fire)。"""
    if fading:
        return 0, False
    if cond:
        consec = int(consec) + 1
        if consec >= TRIGGER_CONSEC:
            return 0, True
        return consec, False
    return 0, False


def score_candidates(pool_v, pool_yaw_rate, target, yaw_err):
    """切换候选评分（§5r 预注册式）：argmin(|v_prior−target| +
    YAW_BIAS_W·|yaw_rate_prior|·1[yaw_err 与 prior yaw_rate 同号])——速度贴带
    优先；yaw_rate 会加剧当前漂移（同号）的候选吃惩罚 = 偏向漂移反侧。
    yaw_err≈0 时同号项全 0（无偏置）。"""
    pv = np.asarray(pool_v, dtype=np.float64)
    pyr = np.asarray(pool_yaw_rate, dtype=np.float64)
    same = (np.sign(pyr) == np.sign(yaw_err)) & (pyr != 0.0)
    return np.abs(pv - target) + YAW_BIAS_W * np.abs(pyr) * same


def pick_fixed_idx(pool_v, target):
    """fixed 臂（及 controller 起步）选 z：池中 prior |v−目标| 最小者（并列
    取先，argmin 确定性）。"""
    return int(np.argmin(np.abs(np.asarray(pool_v, dtype=np.float64) - target)))


def fade_blend(z_from, z_to, j, fade_steps):
    """线性交叉淡入第 j 步（j=1..fade_steps）混合 z：α=j/fade_steps；
    j≤0 → 纯 z_from，j≥fade_steps → 纯 z_to（端点精确，selftest 断言）。"""
    a = min(max(float(j) / float(fade_steps), 0.0), 1.0)
    zf = np.asarray(z_from, dtype=np.float64)
    zt = np.asarray(z_to, dtype=np.float64)
    return (1.0 - a) * zf + a * zt


def retarget_band(target, pool_v, tol=RETARGET_TOL):
    """G2 改宗规则：池内存在 |v_prior−target|≤tol → 维持原目标；否则改宗池中
    最近 v_prior（就近改宗实际可达值，落账不静默）。返回 (target_eff, retargeted)。"""
    pv = np.asarray(pool_v, dtype=np.float64)
    if len(pv) == 0:
        return float(target), True
    d = np.abs(pv - float(target))
    if float(d.min()) <= tol:
        return float(target), False
    return float(pv[int(np.argmin(d))]), True


def window_bounds(n_steps, win_s=WIN_S, stride_s=STRIDE_S, fps=FPS):
    """4s 窗 stride 2s 滑窗（20s=1000 帧 → 起点 0,2,...,16s 共 9 窗）；尾部
    不满一整窗丢弃。返回 [(i0, i1), ...]（i1−i0=窗长帧数）。"""
    w = int(round(win_s * fps))
    s = int(round(stride_s * fps))
    return [(i0, i0 + w) for i0 in range(0, int(n_steps) - w + 1, s)]


def window_v_stats(trans_w):
    """窗内 (v_med, v_p90)：frame_speed_bwd XY 后向差分×FPS（v[0]=v[1]）。"""
    v = frame_speed_bwd(np.asarray(trans_w, dtype=np.float64), FPS, "xy")
    return float(np.median(v)), float(np.quantile(v, 0.9))


def straightness_of(xy):
    """直度 = 净位移/路径长（XY 序列；退化路径返 0.0）。"""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 2:
        return 0.0
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if path <= 0.0:
        return 0.0
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    return net / path


def band_key(target):
    """带目标 → 规范化 JSON 键（2 位小数字符串，如 \"0.70\"）。"""
    return f"{round(float(target), 2):.2f}"


def parse_band_vb_map(s):
    """\"0.35:1,0.70:1,1.00:2\" → {0.35: 1, 0.7: 1, 1.0: 2}（键 round 2dp）。"""
    out = {}
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":")
        out[round(float(k), 2)] = int(v)
    return out


def load_z_init(path):
    """D048r z_sweep_summary.json → {vb(int): z(16,) np}（per_vb[str].best.z）。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for k, d in (data.get("per_vb") or {}).items():
        z = (d.get("best") or {}).get("z")
        if z is not None:
            out[int(k)] = np.asarray(z, dtype=np.float64)
    return out


def mix_z_init(rng, z_star, pop):
    """赢家 z 混入 warmup 首轮分布：前半 = z_star+N(0,0.1²)（局部探索）、
    后半 = N(0,I)（保全局覆盖）。"""
    z_star = np.asarray(z_star, dtype=np.float64)
    z = rng.normal(0.0, 1.0, (int(pop), len(z_star)))
    k = int(pop) // 2
    if k > 0:
        z[:k] = z_star[None, :] + rng.normal(0.0, 0.1, (k, len(z_star)))
    return z


def save_json(path, obj):
    """JSON 落盘（z_sweep 同款参数：ensure_ascii=False + indent=1）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_pool_file(path):
    """pool JSON → (pool dict, targets_effective dict)。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pool = data.get("pool")
    if not isinstance(pool, dict):
        raise RuntimeError(f"pool JSON 缺 pool 键: {path}")
    return pool, data.get("targets_effective", {})


def pool_band_arrays(pool, band):
    """带 → (entries, zs(K,16), vs(K,), yrs(K,))；键按 2dp 匹配（容 1e-9）。"""
    key = band_key(band)
    if key not in pool:
        for k in pool:
            try:
                if abs(float(k) - float(band)) < 1e-9:
                    key = k
                    break
            except (TypeError, ValueError):
                continue
    entries = pool.get(key)
    if not entries:
        raise RuntimeError(f"pool 无带 {band_key(band)} 条目（--arm warmup 先跑）")
    zs = np.asarray([e["z"] for e in entries], dtype=np.float64)
    vs = np.asarray([float(e["v_fwd_med"]) for e in entries], dtype=np.float64)
    yrs = np.asarray([float(e.get("yaw_rate_med", 0.0)) for e in entries],
                     dtype=np.float64)
    if zs.ndim != 2 or zs.shape[1] != Z_DIM or \
            not (len(zs) == len(vs) == len(yrs)):
        raise RuntimeError(f"pool 带 {band_key(band)} 条目 z/字段异常")
    return entries, zs, vs, yrs


def save_episode_npz(path, *, tokens, trans_m, quat_wxyz, jp_isaac, z_t, meta):
    """整局 D044 同构 npz：tokens (n,64) f32 / trans_m (n,3) / quat_wxyz (n,4)
    （Isaac root_quat_w wxyz 直接存，不转 xyzw）/ jp_isaac (n,29) / z_t (n,16) /
    meta（JSON 字符串 0-d 数组，D057 转换器同款、allow_pickle=False 可读）。"""
    np.savez_compressed(
        path,
        tokens=np.asarray(tokens, dtype=np.float32),
        trans_m=np.asarray(trans_m, dtype=np.float64),
        quat_wxyz=np.asarray(quat_wxyz, dtype=np.float64),
        jp_isaac=np.asarray(jp_isaac, dtype=np.float64),
        z_t=np.asarray(z_t, dtype=np.float32),
        meta=np.array(json.dumps(meta, ensure_ascii=False)))


def build_segments_entries(ep_dir):
    """扫描 episodes 目录中 meta.admitted=True 的整局 npz，重算 4s/2s 窗并做
    窗级二级滤（|v_med−带目标|≤WIN_V_TOL），返回 walk_segments.json 条目
    （D057 同构：stem/npz_path/i0/i1/dur_s/v_med/v_p90；i0/i1=窗在局内帧
    区间，npz 按整局落——build_wbt_vae_inputs 的窗口切分照常工作）。"""
    segs = []
    for fn in sorted(os.listdir(ep_dir)):
        if not fn.endswith(".npz"):
            continue
        path = os.path.join(ep_dir, fn)
        with np.load(path) as z:      # 显式关句柄（Windows 删除依赖此）
            meta = json.loads(str(z["meta"]))
            if not bool(meta.get("admitted", False)):
                continue
            tok, trans = z["tokens"], z["trans_m"]
            quat, jp = z["quat_wxyz"], z["jp_isaac"]
            n = len(tok)
        if not (len(trans) == len(quat) == len(jp) == n):
            raise RuntimeError(f"{fn} tokens/trans_m/quat_wxyz/jp_isaac 行数不一致")
        target = float(meta["band_target"])
        stem = os.path.splitext(fn)[0]
        for i0, i1 in window_bounds(n):
            v_med, v_p90 = window_v_stats(trans[i0:i1])
            if abs(v_med - target) > WIN_V_TOL:
                continue
            segs.append({"stem": stem, "npz_path": os.path.abspath(path),
                         "i0": int(i0), "i1": int(i1),
                         "dur_s": round((i1 - i0) / FPS, 4),
                         "v_med": round(v_med, 4), "v_p90": round(v_p90, 4)})
    return segs


def write_segments_json(ep_dir, out_json):
    segs = build_segments_entries(ep_dir)
    save_json(out_json, segs)
    return segs


# --------------------------------------------------------------------- CLI


def build_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--latent-vae-path", default="",
                    help="vae.pt 必填（token_window_vae DirSpeedPhaseTokenVAE，"
                         "env latent 分支同源加载；pca.npz 取同目录；"
                         "selftest/segments 模式豁免，其余模式缺省即 FAIL）")
    ap.add_argument("--arm", default="warmup",
                    choices=["selftest", "warmup", "fixed", "controller",
                             "harvest", "segments"],
                    help="六模式：selftest=纯 numpy 单测；warmup=CEM 池；"
                         "fixed=池中最贴带目标者常 z 开环；controller=闭环"
                         "切换；harvest=滚收割；segments=离线打包（不启 Isaac）")
    ap.add_argument("--selftest", action="store_true",
                    help="等价 --arm selftest（纯 numpy，不 import torch/isaac）")
    ap.add_argument("--n-vbins", type=int, default=3)
    ap.add_argument("--n-dbins", type=int, default=8)
    ap.add_argument("--vb-list", default="0,1,2",
                    help="合法 vb 档集合（band_vb_map 取值须在其中）")
    ap.add_argument("--force-dbin", type=int, default=4,
                    help="db 常量档（默认 4 = +x 前向，E35 bin 公式）")
    ap.add_argument("--rollout-s", type=float, default=20.0,
                    help="每局秒数（§5r 预注册 20s = 1000 步 @50Hz）")
    ap.add_argument("--pop", type=int, default=64,
                    help="num_envs（warmup=CEM 候选数；采集臂=每批局数，多余"
                         " env 的局丢弃不计；G2 建议 --pop 20 = 每批即配额）")
    ap.add_argument("--iters", type=int, default=3, help="warmup CEM 轮数")
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--std-floor", type=float, default=0.1)
    ap.add_argument("--pool-per-band", type=int, default=8,
                    help="warmup 每带入池精英数（按 fitness 降序取 top）")
    ap.add_argument("--z-init-file", default="",
                    help="可选 D048r z_sweep_summary.json（per_vb[vb].best.z "
                         "按带映射档混入 warmup 首轮分布做种子）")
    ap.add_argument("--band-targets", default=BAND_TARGETS_DEF,
                    help="速度带目标 m/s，逗号分隔（§5r 预注册 0.35,0.70,1.00）")
    ap.add_argument("--band-vb-map", default=BAND_VB_MAP_DEF,
                    help="带目标→vb 解码档（§5r 冒烟默认 0.35:1,0.70:1,1.00:2）")
    ap.add_argument("--retarget-tol", type=float, default=RETARGET_TOL,
                    help="G2 改宗判据 |v_prior−目标| 阈（就近改宗并落账）")
    ap.add_argument("--pool-json", default="",
                    help="fixed/controller/harvest 必填：warmup 产物 pool.json")
    ap.add_argument("--episodes-per-band", type=int, default=20,
                    help="fixed/controller：每带每 seed 局数（§5r G2 N=20）")
    ap.add_argument("--seeds", default="0",
                    help="采集臂 jitter seed 列表（同 seed 跨臂初态配对）")
    ap.add_argument("--ctrl-every", type=int, default=25,
                    help="controller 控制周期（步；25 步=0.5s）")
    ap.add_argument("--fade-steps", type=int, default=12,
                    help="z 切换线性交叉淡入步数（α=j/fade_steps 端点精确）")
    ap.add_argument("--harvest-arm", default="controller",
                    choices=["fixed", "controller"],
                    help="harvest 使用哪条臂（§5r G2 判据后择优）")
    ap.add_argument("--min-windows-per-band", type=int, default=300,
                    help="harvest 每带准入窗数停线阈值（§5r G3 ≥300）")
    ap.add_argument("--max-episodes", type=int, default=60,
                    help="harvest 每带局数上限")
    ap.add_argument("--segments-from", default="controller",
                    help="segments 模式来源：arm 名（读 <tag>/<arm>/episodes）"
                         "或直接给目录路径")
    ap.add_argument("--cem-seed", type=int, default=0,
                    help="z 采样 numpy 种子 + torch 全局种子（落盘可复现）")
    ap.add_argument("--jitter-seed", type=int, default=0,
                    help="局初态 jitter_and_reset 种子（同 seed 跨臂配对）")
    ap.add_argument("--router-model-dir", default=DEFAULT_ROUTER_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT_BASE, help="输出根目录")
    ap.add_argument("--tag", default="vae", help="输出子目录 = d058_<tag>")
    return ap


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _r(v, nd=4):
    return round(float(v), nd)


# ------------------------------------------------------------------ selftest


def selftest():
    """纯 numpy 七用例（不 import torch/isaac，§5r G0 入库门）。"""
    n_ok = 0
    rng = np.random.default_rng(0)

    # ① 触发规则：单拍超标不切、连续两拍才切、断拍清零、淡入期不切
    c, fire = trigger_update(0, True, False)
    assert (c, fire) == (1, False), "单拍超标不得切"
    c, fire = trigger_update(c, True, False)
    assert (c, fire) == (0, True), "连续两拍才切"
    c, fire = trigger_update(1, False, False)
    assert (c, fire) == (0, False), "断拍计数清零"
    c, fire = trigger_update(1, True, True)
    assert (c, fire) == (0, False), "淡入期不响应新触发"
    seq = 0
    fired = []
    for cond in (True, False, True, True):
        seq, f = trigger_update(seq, cond, False)
        fired.append(f)
    assert fired == [False, False, False, True], "T,F,T,T 序列只末拍切"
    n_ok += 1

    # ② 候选评分：yaw 漂移反侧候选无惩罚、同号吃惩罚（方向正确）
    pool_v = [0.35, 0.70, 0.70, 0.35]
    pool_yr = [0.20, -0.15, 0.15, -0.20]
    sc = score_candidates(pool_v, pool_yr, 0.70, +0.10)
    assert int(np.argmin(sc)) == 1, "正 yaw 漂移应偏向 yaw_rate 反侧候选"
    assert abs(sc[2] - (0.0 + 0.3 * 0.15)) < 1e-12, "同号候选应吃 0.3·|yr| 惩罚"
    assert sc[1] == 0.0, "反侧同速候选零惩罚"
    sc0 = score_candidates(pool_v, pool_yr, 0.70, 0.0)
    assert np.allclose(sc0, np.abs(np.asarray(pool_v) - 0.70)), \
        "yaw_err=0 无偏置"
    n_ok += 1

    # ③ 交叉淡入调度：12 步线性、端点精确
    z_from, z_to = np.zeros(Z_DIM), np.ones(Z_DIM)
    assert np.allclose(fade_blend(z_from, z_to, 0, 12), z_from), "j=0 端点=from"
    assert np.allclose(fade_blend(z_from, z_to, 12, 12), z_to), "j=12 端点=to"
    assert np.allclose(fade_blend(z_from, z_to, 13, 12), z_to), "j>F 夹紧 to"
    assert np.allclose(fade_blend(z_from, z_to, 6, 12), 0.5 * z_to), "中点 0.5"
    for j in (1, 4, 7, 11):
        a = j / 12.0
        assert np.allclose(fade_blend(z_from, z_to, j, 12), a * z_to), \
            f"j={j} 线性 α=j/12"
    n_ok += 1

    # ④ trailing 统计：合成序列的净前向速度与 yaw wrap
    v = trailing_v_fwd((0.0, 0.0), (0.5, 0.2), (1.0, 0.0), 25)
    assert abs(v - 1.0) < 1e-12, "25 步 0.5m 净前向 = 1.0 m/s"
    v2 = trailing_v_fwd((0.0, 0.0), (-0.25, 0.1), (1.0, 0.0), 25)
    assert abs(v2 - (-0.5)) < 1e-12, "净后退为负"
    f0d = np.array([math.cos(1.0), math.sin(1.0)])
    v3 = trailing_v_fwd((0.0, 0.0), (0.4 * f0d), f0d, 50)
    assert abs(v3 - 0.4) < 1e-12, "任意航向系投影自洽"
    assert abs(float(wrap_pi(math.pi + 0.2)) - (-math.pi + 0.2)) < 1e-12, \
        "wrap 上界"
    assert abs(float(wrap_pi(-3.2)) - (2 * math.pi - 3.2)) < 1e-12, "wrap 下界"
    yaw_err = float(wrap_pi(3.10 - (-3.10)))
    assert abs(yaw_err - (6.2 - 2 * math.pi)) < 1e-12, "跨 ±π yaw 误差"
    n_ok += 1

    # ⑤ 直度与窗 v_med 计算（frame_speed_bwd 口径）
    n_frames = 1000
    x = np.arange(n_frames, dtype=np.float64) * (0.4 / FPS)
    trans = np.stack([x, np.zeros(n_frames), np.zeros(n_frames)], axis=1)
    assert abs(straightness_of(trans[:, :2]) - 1.0) < 1e-12, "直线直度=1"
    th = np.linspace(0.0, math.pi, 400)
    arc = np.stack([np.cos(th), np.sin(th)], axis=1)   # 半圆：net 2 / path≈π
    assert abs(straightness_of(arc) - 2.0 / math.pi) < 1e-4, \
        "半圆直度≈2/π（弦长离散容差 1e-4）"
    vm, vp = window_v_stats(trans[:200])
    assert abs(vm - 0.4) < 1e-12 and abs(vp - 0.4) < 1e-12, "匀速窗 v_med/v_p90"
    wb = window_bounds(n_frames)
    assert len(wb) == 9 and wb[0] == (0, 200) and wb[-1] == (800, 1000), \
        "20s 局 4s/2s 窗 = 9 窗（起点 0,2,...,16s）"
    assert window_bounds(199) == [], "不足一整窗丢弃"
    n_ok += 1

    # ⑥ pool JSON 读写 roundtrip
    entries = [{"z": [round(float(v), 6) for v in rng.normal(size=Z_DIM)],
                "v_fwd_med": 0.69 + 0.01 * i, "yaw_rate_med": -0.05 + 0.02 * i,
                "survival": 1.0, "iter": 1, "env_idx": i, "fitness": 0.05}
               for i in range(3)]
    pool_doc = {"experiment": "D058", "arm": "warmup",
                "targets_effective": {"0.70": 0.70},
                "pool": {"0.70": entries}}
    with tempfile.TemporaryDirectory() as td:
        pj = os.path.join(td, "pool.json")
        save_json(pj, pool_doc)
        pool_back, teff = load_pool_file(pj)
        ent2, zs2, vs2, yr2 = pool_band_arrays(pool_back, 0.70)
        assert len(ent2) == 3 and zs2.shape == (3, Z_DIM)
        assert np.allclose(zs2, np.asarray([e["z"] for e in entries], float))
        assert np.allclose(vs2, [e["v_fwd_med"] for e in entries])
        assert abs(teff["0.70"] - 0.70) < 1e-12
        # 改宗规则：目标 1.00 距池最近 0.71 > 0.20 → 改宗 0.71
        t_eff, retg = retarget_band(1.00, vs2)
        assert retg and abs(t_eff - 0.71) < 1e-12, "就近改宗实际可达值"
        t_ok, retg2 = retarget_band(0.70, vs2)
        assert (not retg2) and abs(t_ok - 0.70) < 1e-12, "带内候选不改宗"
    n_ok += 1

    # ⑦ walk_segments.json 与整局 npz 行数自洽（build_wbt_vae_inputs 同构校验）
    n_ep = 1000
    tok = rng.standard_normal((n_ep, 64)).astype(np.float32)
    tr = np.stack([np.arange(n_ep) * (0.7 / FPS), np.zeros(n_ep),
                   np.zeros(n_ep)], axis=1)
    qt = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_ep, 1))
    jp = rng.standard_normal((n_ep, 29))
    meta_ok = {"experiment": "D058", "stem": "ep_b1_s0_000", "band": 0.70,
               "band_target": 0.70, "arm": "controller", "seed": 0,
               "straightness": 0.97, "v_net_fwd": 0.69, "h_min": 0.71,
               "admitted": True}
    meta_bad = dict(meta_ok, admitted=False)
    with tempfile.TemporaryDirectory() as td:
        ep_dir = os.path.join(td, "episodes")
        os.makedirs(ep_dir)
        p_ok = os.path.join(ep_dir, "ep_b1_s0_000.npz")
        p_bad = os.path.join(ep_dir, "ep_b1_s0_001.npz")
        save_episode_npz(p_ok, tokens=tok, trans_m=tr, quat_wxyz=qt,
                         jp_isaac=jp, z_t=rng.standard_normal((n_ep, Z_DIM)),
                         meta=meta_ok)
        save_episode_npz(p_bad, tokens=tok[:100], trans_m=tr[:100],
                         quat_wxyz=qt[:100], jp_isaac=jp[:100],
                         z_t=rng.standard_normal((100, Z_DIM)), meta=meta_bad)
        segs = build_segments_entries(ep_dir)
        assert segs, "admitted 局应产出窗"
        with np.load(p_ok) as z_chk:
            for s in segs:
                i0, i1 = int(s["i0"]), int(s["i1"])
                n = i1 - i0
                assert s["stem"] == "ep_b1_s0_000" and s["npz_path"] == \
                    os.path.abspath(p_ok)
                assert 0 <= i0 < i1 <= n_ep and abs(s["dur_s"] - 4.0) < 1e-9
                # build_wbt_vae_inputs.load_windows 同款行数校验
                assert len(z_chk["tokens"][i0:i1]) == \
                    len(z_chk["trans_m"][i0:i1]) == \
                    len(z_chk["quat_wxyz"][i0:i1]) == \
                    len(z_chk["jp_isaac"][i0:i1]) == n
                vm_chk, _ = window_v_stats(z_chk["trans_m"][i0:i1])
                assert abs(vm_chk - s["v_med"]) < 1e-3, "segments v_med 可复算"
                assert abs(vm_chk - 0.70) <= WIN_V_TOL, "窗级二级滤口径一致"
        assert len(segs) == 9, "0.7 m/s 直线局 9 窗全过二级滤"
    n_ok += 1

    print(f"SELFTEST OK {n_ok} cases")


# --------------------------------------------------------------------- main


def main():
    ap = build_args()
    cli = ap.parse_args()
    if cli.selftest or cli.arm == "selftest":
        selftest()
        return
    if cli.arm == "segments":
        run_segments_mode(cli)
        return
    if not cli.latent_vae_path:
        raise SystemExit("[args] FAIL: --latent-vae-path 必填"
                         "（selftest/segments 模式豁免）")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    # server has no display: viewport/hydra init segfaults (repo gotcha；
    # z_sweep_cem_d048r 同款 headless 默认)
    ap.set_defaults(headless=True)
    cli = ap.parse_args()          # 带 launcher 参数重解析

    # ---- Isaac 启动前 fail-fast 校验
    vb_list = [int(v) for v in cli.vb_list.split(",") if v.strip() != ""]
    bands = [round(float(v), 2) for v in cli.band_targets.split(",")
             if v.strip() != ""]
    band_vb = parse_band_vb_map(cli.band_vb_map)
    if not bands:
        raise SystemExit("[args] FAIL: --band-targets 为空")
    if cli.arm in ("fixed", "controller", "harvest") and not cli.pool_json:
        raise SystemExit(f"[args] FAIL: --arm {cli.arm} 需 --pool-json"
                         "（warmup 产物）")
    for b in bands:
        if b not in band_vb:
            raise SystemExit(f"[args] FAIL: 带目标 {b} 不在 --band-vb-map")
        if band_vb[b] not in vb_list:
            raise SystemExit(f"[args] FAIL: band_vb_map[{b}]={band_vb[b]} "
                             f"不在 --vb-list {vb_list}")
        if not (0 <= band_vb[b] < cli.n_vbins):
            raise SystemExit(f"[args] FAIL: vb {band_vb[b]} 出值域 "
                             f"[0,{cli.n_vbins})")
    if not (0 <= cli.force_dbin < cli.n_dbins):
        raise SystemExit(f"[args] FAIL: force_dbin {cli.force_dbin} 出值域 "
                         f"[0,{cli.n_dbins})")
    steps = int(round(cli.rollout_s * 50.0))
    if cli.pop < 1 or cli.iters < 1 or cli.ctrl_every < 1 or \
            cli.fade_steps < 1 or cli.episodes_per_band < 1 or \
            cli.pool_per_band < 1:
        raise SystemExit("[args] FAIL: pop/iters/ctrl_every/fade_steps/"
                         "episodes_per_band/pool_per_band 必须 >=1")

    # 种子记录与复现：z 采样 = numpy default_rng(cem_seed)；env 初态/噪声 =
    # torch.manual_seed(cem_seed) + jitter_and_reset(seed)（内部
    # default_rng(1000+seed)）。同 seed 跨臂（fixed vs controller）初态配对。
    import torch
    torch.manual_seed(cli.cem_seed)
    torch.cuda.manual_seed_all(cli.cem_seed)
    rng = np.random.default_rng(cli.cem_seed)

    app_launcher = AppLauncher(cli)
    sim_app = app_launcher.app

    from isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from isaac.eval_apt_isaac import jitter_and_reset

    class SteerEnv(AptFlatG1Env):
        """D058 闭环 z 导引 oracle：CemEnv（z_sweep_cem_d048r）同款旁路——
        _compute_q_des 不走策略，直接 tokens = VAE.decode(z, sc, vb, db) →
        冻结 decoder。区别：_steer_z 是 (N,16) 张量、每步读当前值（闭环切换
        由驱动循环在 step 之间原地更新它实现，env 侧不懂切换/淡入）；
        _last_tokens 缓存本步 decode 输出供逐局落盘。canonical
        apt_flat_env.py 零改动。"""

        _steer_z: torch.Tensor | None = None
        _steer_vb: int = 1
        _steer_db: int = 4
        _last_tokens: torch.Tensor | None = None

        def _compute_q_des(self, phase, aux, res=None):
            with torch.no_grad():
                phi = self._latent_phase
                sc = torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)
                vb = torch.full((self.num_envs,), self._steer_vb,
                                dtype=torch.long, device=self.device)
                db = torch.full((self.num_envs,), self._steer_db,
                                dtype=torch.long, device=self.device)
                tokens = self._vae.decode(self._steer_z, sc, vb, db).detach()
                self._last_tokens = tokens
                # walk clock 固定步频推进（env latent 分支 E27 路径同式）
                self._latent_phase = (phi + self._latent_phase_rate) % math.tau
            action_t = self._decoder.decode(
                tokens,
                self._hist_ang_vel,
                self._hist_joint_pos,
                self._hist_joint_vel,
                self._hist_last_actions,
                self._hist_gravity,
            )
            return self._sonic_default_t + action_t * self._sonic_scale_t

    cfg = AptFlatG1EnvCfg()
    cfg.scene.num_envs = cli.pop
    cfg.episode_length_s = cli.rollout_s + 30.0   # 预算内不触发 trunc
    cfg.router_model_dir = cli.router_model_dir
    cfg.latent_mode = True
    cfg.latent_dir_bins = True
    cfg.latent_vae_path = cli.latent_vae_path
    cfg.latent_vae_n_bins = cli.n_vbins
    cfg.latent_vae_n_dbins = cli.n_dbins
    # D048r hotfix3 同款：latent obs 105 vs 缺省 91 → _get_observations assert
    cfg.observation_space += 14
    cfg.action_space = 16                          # latent z only（train 同款）
    env = SteerEnv(cfg)
    env._steer_db = int(cli.force_dbin)
    device = env.device

    def _yaw_vec(q):
        """(N,4) w-first quat → (N,) yaw（eval_apt_isaac._yaw_of 向量化）。"""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    tag_dir = os.path.join(cli.out, f"d058_{cli.tag}")
    arm_dir = os.path.join(tag_dir, cli.arm)
    ep_dir = os.path.join(arm_dir, "episodes")
    seeds = [int(s) for s in cli.seeds.split(",") if s.strip() != ""]
    if not seeds:
        raise SystemExit("[args] FAIL: --seeds 为空")

    def run_batch(*, vb, seed, z_init, band_target, pool_arrays=None,
                  start_idx=None, controller=False, record=False):
        """一局：num_envs 并行 --rollout-s。PRE-step 纪律：每步在 env.step
        返回后、下一 action 提交前读 robot.data（=下一拍的 PRE 状态）；
        done 步（term|trunc，auto-reset 污染，b3p 口径）数据一律丢弃，且
        摔倒 env 冻结录制（不收 auto-reset 后的"重生"数据）。

        controller=True 时每 --ctrl-every 步一拍：trailing 净前向（初始航向
        系投影）+ yaw 误差 → 双阈值连续两拍触发 → 池内 argmin 换 z →
        --fade-steps 线性交叉淡入（混合 z 原地写 env._steer_z）。

        返回 per-env dict 列表（n=cli.pop；含标量指标与 windows；record=True
        时附 "_arr" 整局数组）。"""
        n = env.num_envs
        zs_pool = vs_pool = yrs_pool = None
        if controller:
            assert pool_arrays is not None and start_idx is not None, \
                "controller 臂需要 pool_arrays/start_idx"
            zs_pool, vs_pool, yrs_pool = pool_arrays
        env._steer_z = torch.from_numpy(
            np.asarray(z_init, dtype=np.float32)).to(device)
        env._steer_vb = int(vb)
        jitter_and_reset(env, seed)
        env._latent_phase.zero_()      # 时钟起点钉 0（D048r 同款，可复现）
        xy0 = env.robot.data.root_pos_w[:, :2].detach().cpu().numpy().copy()
        yaw0 = _yaw_vec(env.robot.data.root_quat_w.detach()).cpu().numpy().copy()
        f0 = np.stack([np.cos(yaw0), np.sin(yaw0)], axis=1)

        rows_pos = [[] for _ in range(n)]
        rows_quat = [[] for _ in range(n)]
        rows_jp = [[] for _ in range(n)]
        rows_tok = [[] for _ in range(n)]
        rows_z = [[] for _ in range(n)]
        h_min = np.full(n, np.inf)
        fall_step = np.full(n, -1, dtype=np.int64)
        alive_t = torch.ones(n, dtype=torch.bool, device=device)
        if controller:
            cur_idx = np.full(n, int(start_idx), dtype=np.int64)
            pend_idx = np.full(n, -1, dtype=np.int64)   # >=0 = 淡入中
            fade_j = np.zeros(n, dtype=np.int64)
            consec = np.zeros(n, dtype=np.int64)
            beat_xy = xy0.copy()
            beat_step = np.zeros(n, dtype=np.int64)
            switches = [[] for _ in range(n)]

        for t in range(steps):
            # ---- 淡入推进（驱动侧算好混合 z 原地写 _steer_z；env 不懂淡入）
            if controller:
                for e in np.nonzero(fade_j > 0)[0]:
                    zb = fade_blend(zs_pool[cur_idx[e]], zs_pool[pend_idx[e]],
                                    int(fade_j[e]), cli.fade_steps)
                    env._steer_z[e] = torch.as_tensor(zb.astype(np.float32),
                                                      device=device)
                    fade_j[e] += 1
                    if fade_j[e] > cli.fade_steps:
                        fade_j[e] = 0
                        cur_idx[e] = pend_idx[e]
                        pend_idx[e] = -1
            action = torch.zeros(n, env.cfg.action_space,
                                 dtype=torch.float32, device=device)
            obs, rew, term, trunc, _ = env.step(action)
            done = term.bool() | trunc.bool()
            live_t = (~done) & alive_t
            term_new = term.bool() & alive_t
            live_np = live_t.cpu().numpy()
            if bool(live_t.any()):
                # done 步 robot.data 已被 auto-reset 污染（b3p 口径）→ 只收 live
                pos_w = env.robot.data.root_pos_w.detach().cpu().numpy()
                quat_w = env.robot.data.root_quat_w.detach().cpu().numpy()
                jp = env.robot.data.joint_pos[:, env._body_idx].detach() \
                    .cpu().numpy()
                tok = (env._last_tokens.detach().cpu().numpy()
                       if env._last_tokens is not None
                       else np.zeros((n, 64), dtype=np.float32))
                z_now = env._steer_z.detach().cpu().numpy()
                h_min[live_np] = np.minimum(h_min[live_np], pos_w[live_np, 2])
                for e in np.nonzero(live_np)[0]:
                    rows_pos[e].append(pos_w[e])
                    rows_quat[e].append(quat_w[e])
                    rows_jp[e].append(jp[e])
                    rows_tok[e].append(tok[e])
                    rows_z[e].append(z_now[e])
            newly = term_new.cpu().numpy()
            fall_step[newly] = t
            alive_t &= ~term.bool()
            if not bool(alive_t.any()):
                break
            # ---- 控制拍（每 ctrl_every 步；摔倒 env 不再评估）
            if controller and (t + 1) % cli.ctrl_every == 0:
                for e in np.nonzero(live_np)[0]:
                    if not rows_pos[e]:
                        continue
                    n_el = (t + 1) - int(beat_step[e])
                    if n_el <= 0:
                        continue
                    xy_now = rows_pos[e][-1][:2]
                    v_trail = trailing_v_fwd(beat_xy[e], xy_now, f0[e], n_el)
                    yaw_now = float(yaw_of_np(rows_quat[e][-1][None, :])[0])
                    yaw_err = float(wrap_pi(yaw_now - float(yaw0[e])))
                    cond = (abs(yaw_err) > YAW_TRIG_RAD) or \
                        (abs(v_trail - float(band_target)) > V_TRIG)
                    consec[e], fire = trigger_update(
                        int(consec[e]), cond, bool(pend_idx[e] >= 0))
                    beat_xy[e] = xy_now
                    beat_step[e] = t + 1
                    if fire:
                        sc = score_candidates(vs_pool, yrs_pool,
                                              float(band_target), yaw_err)
                        to_i = int(np.argmin(sc))
                        if to_i != int(cur_idx[e]):
                            switches[e].append({
                                "step": t + 1,
                                "from_idx": int(cur_idx[e]),
                                "to_idx": to_i,
                                "trigger": {"yaw_err": _r(yaw_err),
                                            "v_trail": _r(v_trail),
                                            "target": _r(band_target)}})
                            pend_idx[e] = to_i
                            fade_j[e] = 1   # 下一步起 α=1/fade_steps

        # ---- 逐局收账（PRE/POST 口径见 run_batch docstring）
        episodes = []
        for e in range(n):
            rows = rows_pos[e]
            rec = {"env_idx": e, "n_steps": len(rows),
                   "fall_step": int(fall_step[e]), "h_min": _r(h_min[e], 3),
                   "band_target": _r(band_target), "vb": int(vb),
                   "seed": int(seed)}
            if controller:
                rec["switches"] = switches[e]
                rec["pool_idx_start"] = int(start_idx)
                rec["pool_idx_final"] = int(cur_idx[e])
            else:
                rec["switches"] = []
            if not rows:
                rec.update({"v_net_fwd": None, "net_disp": None,
                            "path_len": None, "straightness": None,
                            "upright_final": None, "survived": False,
                            "admitted": False, "v_fwd_med": None,
                            "yaw_rate_med": None, "windows": [],
                            "n_windows_kept": 0})
                episodes.append(rec)
                continue
            pos = np.stack(rows)
            quat = np.stack(rows_quat[e])
            xy = np.vstack([xy0[e][None, :], pos[:, :2]])
            path_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            net_vec = xy[-1] - xy[0]
            net_disp = float(np.linalg.norm(net_vec))
            v_net = float(np.dot(net_vec, f0[e])) / (len(rows) / FPS)
            strg = straightness_of(xy)
            up_final = float(upright_np(quat[-1][None, :])[0])
            survived = bool(fall_step[e] < 0) and up_final >= UPRIGHT_MIN
            v_ser = (np.diff(xy, axis=0) @ f0[e]) * FPS
            v_fwd_med = float(np.median(v_ser))
            yaw_all = np.concatenate(([float(yaw0[e])], yaw_of_np(quat)))
            yr_med = float(np.median(wrap_pi(np.diff(yaw_all)) * FPS))
            admitted = bool(survived and h_min[e] >= EP_H_MIN
                            and strg >= STRAIGHT_MIN
                            and abs(v_net - float(band_target)) <= EP_V_TOL)
            wins = []
            if fall_step[e] < 0 and len(rows) == steps:
                for (i0, i1) in window_bounds(len(rows)):
                    v_med_w, v_p90_w = window_v_stats(pos[i0:i1, :3])
                    wins.append({"i0": int(i0), "i1": int(i1),
                                 "dur_s": _r((i1 - i0) / FPS),
                                 "v_med": _r(v_med_w), "v_p90": _r(v_p90_w),
                                 "kept": bool(abs(v_med_w - float(band_target))
                                              <= WIN_V_TOL)})
            rec.update({"v_net_fwd": _r(v_net), "net_disp": _r(net_disp),
                        "path_len": _r(path_len), "straightness": _r(strg),
                        "upright_final": _r(up_final), "survived": survived,
                        "admitted": admitted, "v_fwd_med": _r(v_fwd_med),
                        "yaw_rate_med": _r(yr_med), "windows": wins,
                        "n_windows_kept": int(sum(w["kept"] for w in wins))})
            if record:
                rec["_arr"] = {"pos": pos, "quat": quat, "jp": np.stack(rows_jp[e]),
                               "tok": np.stack(rows_tok[e]),
                               "z": np.stack(rows_z[e])}
            episodes.append(rec)
        return episodes

    def fitness_of(e, target):
        """warmup CEM fitness（§5r）：−(|净前向−带目标|)；摔/终末 upright<0.9
        → −1（D048r 判负同式）。"""
        if e["v_net_fwd"] is None or e["fall_step"] >= 0 \
                or e["upright_final"] is None or e["upright_final"] < UPRIGHT_MIN:
            return -1.0
        return -abs(e["v_net_fwd"] - float(target))

    def save_episode(e):
        """整局 npz 落盘（D044 同构五数组 + meta JSON 串）并回填 npz_path；
        零行局（开局即摔、无任何干净步）无数据可落，跳过只留 summary 行。"""
        arr = e.pop("_arr", None)
        if arr is None:
            return
        path = os.path.join(ep_dir, e["stem"] + ".npz")
        meta = {"experiment": "D058", "stem": e["stem"],
                "band": e["band"], "band_idx": e["band_idx"],
                "band_target": e["band_target"], "arm": e["arm"],
                "seed": e["seed"], "vb": e["vb"],
                "straightness": e["straightness"],
                "v_net_fwd": e["v_net_fwd"], "h_min": e["h_min"],
                "admitted": e["admitted"], "n_steps": e["n_steps"],
                "fall_step": e["fall_step"],
                "upright_final": e["upright_final"],
                "path_len": e["path_len"],
                "pool_idx_start": e.get("pool_idx_start"),
                "pool_idx_final": e.get("pool_idx_final"),
                "n_switches": len(e["switches"]),
                "created_at": now_iso()}
        save_episode_npz(path, tokens=arr["tok"], trans_m=arr["pos"],
                         quat_wxyz=arr["quat"], jp_isaac=arr["jp"],
                         z_t=arr["z"], meta=meta)
        e["npz_path"] = os.path.abspath(path)

    def write_arm_summary(arm_label, all_eps, extra):
        """<arm>_summary.json：逐局记录（PRE-step 口径）+ 带级聚合（直度中位/
        带命中率/准入率——G2 三臂判据的直接读数）。"""
        per_band = {}
        for bi, b in enumerate(bands):
            eps_b = [e for e in all_eps if e["band_idx"] == bi]
            wins = [w for e in eps_b for w in e["windows"]]
            strg = [e["straightness"] for e in eps_b if e["straightness"]]
            vnet = [e["v_net_fwd"] for e in eps_b if e["v_net_fwd"] is not None]
            per_band[band_key(b)] = {
                "episodes": len(eps_b),
                "admitted": int(sum(bool(e["admitted"]) for e in eps_b)),
                "straightness_med": _r(np.median(strg)) if strg else None,
                "v_net_fwd_med": _r(np.median(vnet)) if vnet else None,
                "windows_total": len(wins),
                "windows_kept": int(sum(w["kept"] for w in wins)),
                "band_hit_rate": _r(sum(w["kept"] for w in wins) / len(wins))
                if wins else None,
            }
        out = {"experiment": "D058", "arm": arm_label,
               "created_at": now_iso(),
               "args": {k: v for k, v in vars(cli).items()},
               "bands": [band_key(b) for b in bands],
               "band_vb_map": {band_key(b): int(band_vb[b]) for b in bands},
               "band_targets": {band_key(b): float(b) for b in bands},
               "rollout": {"steps": steps, "dur_s": steps / FPS,
                           "pop": cli.pop,
                           "clock_rule": "walk clock 起步钉 0、固定 pca rate 推进"},
               "metric_rule": "净前向=初始航向系投影位移/时长（D048r 口径）；"
                              "直度=net_disp/path_len；准入=不摔∧h_min≥0.40∧"
                              "直度≥0.90∧|净前向−目标|≤0.15；窗级滤 "
                              "|v_med−目标|≤0.25（frame_speed_bwd XY@50Hz）",
               "window_stats": per_band,
               "episodes": all_eps,
               **extra}
        save_json(os.path.join(arm_dir, f"{arm_label}_summary.json"), out)
        return per_band

    # ------------------------------------------------------------ 模式驱动

    if cli.arm == "warmup":
        os.makedirs(arm_dir, exist_ok=True)
        z_init = load_z_init(cli.z_init_file) if cli.z_init_file else {}
        pool_all, targets_eff, retargeted, diag = {}, {}, {}, {}
        for bi, b in enumerate(bands):
            vb = band_vb[b]
            target = float(b)
            z_star = z_init.get(int(vb))
            cand, it_diag = [], []
            mu, var = np.zeros(Z_DIM), np.ones(Z_DIM)
            for it in range(cli.iters):
                if it == 0 and z_star is not None:
                    zs = mix_z_init(rng, z_star, cli.pop)
                else:
                    zs = cem_sample(rng, mu, var, cli.pop)
                eps = run_batch(vb=vb, seed=cli.jitter_seed + it, z_init=zs,
                                band_target=target, record=False)
                fit = np.asarray([fitness_of(e, target) for e in eps])
                for zrow, e in zip(zs, eps):
                    # 零行局（开局即摔）无逐帧统计 → 0.0 占位（fitness 判 −1，
                    # 排序自然沉底，不会入池成为有效候选）
                    cand.append({"z": [round(float(v), 6) for v in zrow],
                                 "v_fwd_med": e["v_fwd_med"]
                                 if e["v_fwd_med"] is not None else 0.0,
                                 "yaw_rate_med": e["yaw_rate_med"]
                                 if e["yaw_rate_med"] is not None else 0.0,
                                 "survival": 1.0 if e["survived"] else 0.0,
                                 "v_net": e["v_net_fwd"],
                                 "iter": it, "env_idx": e["env_idx"],
                                 "fitness": _r(fitness_of(e, target))})
                n_fall = int(sum(e["fall_step"] >= 0 for e in eps))
                it_diag.append({"iter": it,
                                "fitness_med": _r(np.median(fit)),
                                "fitness_max": _r(fit.max()),
                                "n_fall": n_fall,
                                "survival_rate": _r(
                                    sum(e["fall_step"] < 0 for e in eps)
                                    / len(eps)),
                                "seed_jitter": cli.jitter_seed + it})
                print(f"[cem] b{bi}({band_key(b)},vb{vb}) it{it}: "
                      f"med={it_diag[-1]['fitness_med']} "
                      f"max={it_diag[-1]['fitness_max']} fall={n_fall}",
                      flush=True)
                mu, var, _ = cem_next_distribution(zs, fit, cli.elite_frac,
                                                   cli.std_floor)
            cand.sort(key=lambda c: -c["fitness"])
            pool_entries = cand[:cli.pool_per_band]
            pool_all[band_key(b)] = pool_entries
            vs_b = np.asarray([c["v_fwd_med"] for c in pool_entries],
                              dtype=np.float64)
            t_eff, retg = retarget_band(target, vs_b, cli.retarget_tol)
            targets_eff[band_key(b)] = _r(t_eff)
            retargeted[band_key(b)] = bool(retg)
            diag[band_key(b)] = it_diag
            if retg:
                print(f"[retarget] 带 {band_key(b)} 无候选 "
                      f"|v−{target}|≤{cli.retarget_tol} → 就近改宗 {t_eff} "
                      "（G2 规则，落账不静默）", flush=True)
        pool_doc = {"experiment": "D058", "arm": "warmup",
                    "created_at": now_iso(),
                    "args": {k: v for k, v in vars(cli).items()},
                    "z_init_file": cli.z_init_file or "",
                    "z_init_rule": "首轮前半=赢家 z+N(0,0.1²) 局部探索、"
                                   "后半=N(0,I)" if z_init else "",
                    "band_vb_map": {band_key(b): int(band_vb[b]) for b in bands},
                    "band_targets": {band_key(b): float(b) for b in bands},
                    "targets_effective": targets_eff,
                    "retargeted": retargeted,
                    "retarget_rule": f"warmup 后带内无候选 |v−目标|≤"
                                     f"{cli.retarget_tol} → 就近改宗（G2）",
                    "pool": pool_all,
                    "cem_diag": diag}
        save_json(os.path.join(tag_dir, "warmup", "pool.json"), pool_doc)
        save_json(os.path.join(tag_dir, "warmup", "warmup_summary.json"),
                  pool_doc)
        print(f"saved {os.path.join(tag_dir, 'warmup', 'pool.json')}",
              flush=True)

    elif cli.arm in ("fixed", "controller"):
        controller = cli.arm == "controller"
        os.makedirs(ep_dir, exist_ok=True)
        pool, teff_file = load_pool_file(cli.pool_json)
        all_eps = []
        targets_eff_run = {}
        for bi, b in enumerate(bands):
            vb = band_vb[b]
            entries, zs, vs, yrs = pool_band_arrays(pool, b)
            target = float(teff_file.get(band_key(b), float(b)))
            target, _ = retarget_band(target, vs, cli.retarget_tol)
            targets_eff_run[band_key(b)] = _r(target)
            start_idx = pick_fixed_idx(vs, target)
            print(f"[{cli.arm}] 带 {band_key(b)}(vb{vb}) 目标 {target} "
                  f"start_idx={start_idx} pool_z={zs[start_idx].round(3).tolist()}"
                  , flush=True)
            for seed in seeds:
                made, k = 0, 0
                while made < cli.episodes_per_band:
                    eps = run_batch(vb=vb, seed=seed,
                                    z_init=np.tile(zs[start_idx], (cli.pop, 1)),
                                    band_target=target, pool_arrays=(zs, vs, yrs),
                                    start_idx=start_idx, controller=controller,
                                    record=True)
                    for e in eps[:cli.episodes_per_band - made]:
                        e.update({"stem": f"ep_b{bi}_s{seed}_{k:03d}",
                                  "band": float(b), "band_idx": bi,
                                  "arm": cli.arm})
                        save_episode(e)
                        all_eps.append(e)
                        made += 1
                        k += 1
        per_band = write_arm_summary(cli.arm, all_eps, {
            "pool_json": os.path.abspath(cli.pool_json),
            "seeds": seeds,
            "episodes_per_band": cli.episodes_per_band,
            "targets_effective": targets_eff_run,
        })
        print(f"[{cli.arm}] window_stats={json.dumps(per_band)}", flush=True)

    elif cli.arm == "harvest":
        controller = cli.harvest_arm == "controller"
        os.makedirs(ep_dir, exist_ok=True)
        pool, teff_file = load_pool_file(cli.pool_json)
        all_eps = []
        targets_eff_run = {}
        for bi, b in enumerate(bands):
            vb = band_vb[b]
            entries, zs, vs, yrs = pool_band_arrays(pool, b)
            target = float(teff_file.get(band_key(b), float(b)))
            target, _ = retarget_band(target, vs, cli.retarget_tol)
            targets_eff_run[band_key(b)] = _r(target)
            start_idx = pick_fixed_idx(vs, target)
            made, kept_w, k = 0, 0, 0
            while kept_w < cli.min_windows_per_band and made < cli.max_episodes:
                seed = seeds[made % len(seeds)]
                eps = run_batch(vb=vb, seed=seed,
                                z_init=np.tile(zs[start_idx], (cli.pop, 1)),
                                band_target=target, pool_arrays=(zs, vs, yrs),
                                start_idx=start_idx, controller=controller,
                                record=True)
                for e in eps:
                    if made >= cli.max_episodes:
                        break
                    e.update({"stem": f"ep_b{bi}_s{seed}_{k:03d}",
                              "band": float(b), "band_idx": bi,
                              "arm": "harvest"})
                    save_episode(e)
                    kept_w += e["n_windows_kept"]
                    made += 1
                    k += 1
                    all_eps.append(e)
                print(f"[harvest/{cli.harvest_arm}] 带 {band_key(b)}: 局 "
                      f"{made}/{cli.max_episodes} 准入窗 {kept_w}/"
                      f"{cli.min_windows_per_band}", flush=True)
        write_arm_summary("harvest", all_eps, {
            "harvest_arm": cli.harvest_arm,
            "pool_json": os.path.abspath(cli.pool_json),
            "seeds": seeds,
            "targets_effective": targets_eff_run,
            "min_windows_per_band": cli.min_windows_per_band,
            "max_episodes": cli.max_episodes,
            "stop_rule": "滚至每带准入窗≥--min-windows-per-band 或 局数≥"
                         "--max-episodes；滚 3× 预算窗数仍 <150 窗/带 → "
                         "材料不足落账收线（§5r G3 停线，人工判读）",
        })
        segs = write_segments_json(ep_dir,
                                   os.path.join(tag_dir, "walk_segments.json"))
        print(f"[harvest] walk_segments.json: {len(segs)} windows "
              f"({os.path.join(tag_dir, 'walk_segments.json')})", flush=True)

    # Isaac 有时在解释器退出时挂死（repo gotcha）：daemon close + 硬退出
    # （z_sweep/oracle 同款；结果已落盘）。
    import threading

    def _close():
        try:
            sim_app.close()
        except Exception:
            pass

    closer = threading.Thread(target=_close, daemon=True)
    closer.start()
    closer.join(timeout=30)
    os._exit(0)


def run_segments_mode(cli):
    """--arm segments：离线打包已准入局（不启 Isaac）。扫描
    <tag>/<segments-from>/episodes/ 中 meta.admitted=True 的整局 npz，重算
    4s/2s 窗 + 窗级二级滤，写 <tag>/walk_segments.json（D057 同构）。"""
    tag_dir = os.path.join(cli.out, f"d058_{cli.tag}")
    src = cli.segments_from
    arm_dir = os.path.join(tag_dir, src) if src in ("fixed", "controller",
                                                    "harvest") else src
    ep_dir = os.path.join(arm_dir, "episodes")
    if not os.path.isdir(ep_dir):
        raise SystemExit(f"[segments] FAIL: episodes 目录不存在 {ep_dir}"
                         "（先跑 fixed/controller/harvest 采集局）")
    segs = write_segments_json(ep_dir, os.path.join(tag_dir,
                                                    "walk_segments.json"))
    n_ep = len({s["stem"] for s in segs})
    print(f"[segments] {len(segs)} windows / {n_ep} episodes -> "
          f"{os.path.join(tag_dir, 'walk_segments.json')}（npz 整局落、"
          "i0/i1=窗区间；build_wbt_vae_inputs --segments-json 可消费）",
          flush=True)
    if not segs:
        print("[segments] WARN: 无准入窗产出（检查 episodes meta.admitted / "
              "窗级滤 |v_med−目标|≤0.25）", flush=True)


if __name__ == "__main__":
    main()
