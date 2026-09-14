"""D057 G4: WBT 行走窗 → VAE v1/v2 双兼容训练输入 + speedA 式 vb 速度标签。

读 mine_wbt_walk_segments.py 产物 walk_segments.json（G3），逐窗切源 npz
（tokens / trans_m / quat_wxyz / jp_isaac 切窗并做行数一致性校验；quat/jp
仅校验不落盘——v1/v2 训练器均不消费），拼装输出目录（默认
<conv-dir>/vae_inputs_wbt）：

  token.npy (N,64) float32   mode.npy & mode_id.npy（同内容 int64 全 2=WALK；
                             v1 train_token_vae_e39 读 mode.npy、v2
                             train_token_vae_e39_v2 读 mode_id.npy，双兼容）
  ul.npy int64 全 0（none）  angle_bin.npy（import build_b4lite_vae_inputs
                             .angle_bins 逐窗算；参数抄 v2 builder 默认
                             dt=0.02 / smooth=5 / v_thresh=0.05 / theta0_k=10）
  segment_bounds.npy (M,2)   每窗=一段，连续无缝隙覆盖 [0,N)
  window_keep_mask.npy 全 1（v2 builder 同款窗口枚举长度 = sum(n_i-(window-1))，
                             无 intermediate 材料，全部保留）
  segment_source.npy identity 0..M-1（无过采样，每窗自成源段）
  build_meta.json            mode_table / oversample_plan(identity) /
                             segment_detail / provenance / generated_at

vb 速度标签（speedA-D048n 同口径，XY@50Hz）：
  拼装后逐窗 frame_speed_bwd(trans_m[i0:i1], 50.0, "xy")（窗内 v[0]=v[1]）
  -> 全部拼装帧速度取 1/3、2/3 分位为 edges
  -> vb = clip(digitize(v, edges), 0, 2)（与 build_d048n L18 口径一致）
  落 vb_speed.npy (N,) int64 + vb_speed_meta.json（edges / counts /
  label_policy / ref_frames / n_frames / source_npz_md5 逐 npz 文件 md5）。

用法（服务器 .venv_isaac python；numpy-only，不依赖 torch）：
  python build_wbt_vae_inputs.py --segments-json <conv-dir>/walk_segments.json
  python build_wbt_vae_inputs.py --selftest            # 本机 numpy-only 自测

结构说明（D057 重构）：selftest 脚手架来自 wbt_common.CheckLog；main 拆为
load_windows -> write_outputs -> write_metas 三阶段函数（均在本文件），
两个 meta dict 字面量键顺序逐字保留。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np

try:                      # 服务器执行根平铺 import / 仓库根包 import 双兼容
    from build_d048n_vb_speed_labels import frame_speed_bwd, md5_file
except ImportError:
    from apt_g1.build_d048n_vb_speed_labels import frame_speed_bwd, md5_file

try:                      # v2 builder 窗口枚举单一实现（keep 掩码长度与训练侧一致）
    from build_b4lite_vae_inputs_v2 import window_keep_mask
except ImportError:
    from apt_g1.build_b4lite_vae_inputs_v2 import window_keep_mask

try:                      # selftest 脚手架（wbt_common，D057 重构）；双兼容同上
    from wbt_common import CheckLog
except ImportError:
    from apt_g1.wbt_common import CheckLog


def _import_angle_bins():
    """angle_bins/frame_speed 双兼容 import；失败返回 (None, None, err)。

    仅 selftest 允许降级 SKIP（按 D057 G4 指令）；main 入口拦截。
    """
    try:
        from build_b4lite_vae_inputs import angle_bins, frame_speed
        return angle_bins, frame_speed, None
    except ImportError:
        pass
    try:
        from apt_g1.build_b4lite_vae_inputs import angle_bins, frame_speed
        return angle_bins, frame_speed, None
    except ImportError as exc:
        return None, None, exc


ANGLE_BINS, FRAME_SPEED, ANGLE_BINS_ERR = _import_angle_bins()

FPS = 50.0        # 上游契约：转换 npz 统一 50Hz（口径勿改）
MODE_ID = 2       # WALK embed idx（v2.1 表 WALK=2）；mode.npy/mode_id.npy 全 2
MODE_HPP_ID = 2   # SONIC hpp id：WALK=2（build_b4lite_vae_inputs_v2.MODE_HPPO）
LABEL_POLICY = "wbt_walk_tercile_xy_bwd_diff_50hz"


def pack_windows(wins: list[dict], dt: float, smooth: int, v_thresh: float,
                 theta0_k: int) -> dict:
    """逐窗拼装四件套 + 逐窗 vb 速度序列。

    wins 元素需带 {stem, npz_path, i0, i1, v_med, tok_w (n,64), trans_w (n,3)}。
    ANGLE_BINS 为 None 时（仅 selftest 降级路径）angle_bin 置全 0 占位。
    """
    toks, modes, uls, bins, v_all, bounds, detail = [], [], [], [], [], [], []
    off = 0
    for w in wins:
        tok_w, trans_w = w["tok_w"], w["trans_w"]
        n = len(tok_w)
        assert n >= 2, f"{w['stem']}[{w['i0']}:{w['i1']}] 窗长 {n} < 2"
        if ANGLE_BINS is None:  # selftest 降级：angle_bins 依赖缺失
            ab = np.zeros(n, dtype=np.int64)
        else:
            spd = FRAME_SPEED(trans_w, dt, smooth)
            ab, _ = ANGLE_BINS(trans_w, spd, v_thresh, theta0_k)
        v_all.append(frame_speed_bwd(trans_w, FPS, "xy"))
        toks.append(tok_w.astype(np.float32))
        modes.append(np.full(n, MODE_ID, dtype=np.int64))
        uls.append(np.zeros(n, dtype=np.int64))
        bins.append(ab)
        bounds.append([off, off + n])
        detail.append({"stem": w["stem"], "npz_path": w["npz_path"],
                       "i0": int(w["i0"]), "i1": int(w["i1"]),
                       "n_frames": int(n), "v_med": w.get("v_med")})
        off += n
    return {"token": np.concatenate(toks), "mode": np.concatenate(modes),
            "mode_id": np.concatenate(modes), "ul": np.concatenate(uls),
            "angle_bin": np.concatenate(bins), "v_all": np.concatenate(v_all),
            "bounds": np.asarray(bounds, dtype=np.int64),
            "segment_detail": detail}


def vb_tercile(v_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """全部拼装帧速度 -> 三分位 edges + vb（build_d048n 同构口径）。"""
    edges = np.quantile(v_all, [1.0 / 3.0, 2.0 / 3.0])
    vb = np.clip(np.digitize(v_all, edges), 0, 2).astype(np.int64)
    return edges, vb


def verify_packed(res: dict, vb: np.ndarray) -> None:
    """长度一致 + bounds 连续无缝隙不重叠覆盖 [0,N) + vb 值域。"""
    n = len(res["token"])
    assert n == len(res["mode"]) == len(res["mode_id"]) == len(res["ul"]) \
        == len(res["angle_bin"]) == len(vb), "拼装长度不一致"
    b = res["bounds"]
    assert b.ndim == 2 and b.shape[1] == 2, "segment_bounds 形状异常"
    assert int(b[0, 0]) == 0 and int(b[-1, 1]) == n, "bounds 未覆盖 [0,N)"
    assert np.array_equal(b[1:, 0], b[:-1, 1]), "bounds 有缝隙/重叠"
    assert int(vb.min()) >= 0 and int(vb.max()) <= 2, "vb 越界 [0,2]"


def _ramp_trans(speeds) -> np.ndarray:
    """n 帧分段匀速 +x 直线轨迹（z=0）：v[i]=speeds[i]（i>=1，v[0]=v[1]，
    speeds[0]==speeds[1] 时逐帧速度与 speeds 恒等）。"""
    speeds = np.asarray(speeds, dtype=np.float64)
    x = np.concatenate([[0.0], np.cumsum(speeds[1:] / FPS)])
    z = np.zeros(len(speeds))
    return np.stack([x, z, z], axis=1)


def load_windows(segs: list[dict],
                 segments_json: str) -> tuple[list[dict], dict, dict]:
    """逐窗切源 npz 装载 + 逐 npz md5 留档 + 行数一致性校验（原 main 装载段拆出，
    数值与校验逻辑零漂移）。

    返回 (wins, cache, md5s)：wins 元素带 {stem, npz_path, i0, i1, v_med,
    tok_w (n,64) float32, trans_w (n,3) float64}；cache 为 stem -> npz 句柄
    （同 stem 只读一次盘）；md5s 为 stem -> 源文件 md5（vb_speed_meta.json 用）。
    """
    cache: dict[str, object] = {}
    md5s: dict[str, str] = {}
    wins: list[dict] = []
    for s in segs:
        stem, i0, i1 = s["stem"], int(s["i0"]), int(s["i1"])
        if stem not in cache:
            if not os.path.isfile(s["npz_path"]):
                raise FileNotFoundError(f"源 npz 缺失: {s['npz_path']}")
            cache[stem] = np.load(s["npz_path"])
            md5s[stem] = md5_file(s["npz_path"])  # 逐 npz 文件 md5（仿 D048n）
        z = cache[stem]
        n = i1 - i0
        tok_w = z["tokens"][i0:i1].astype(np.float32)
        trans_w = z["trans_m"][i0:i1].astype(np.float64)
        quat_w = z["quat_wxyz"][i0:i1]   # 切窗仅做行一致性校验，不落盘
        jp_w = z["jp_isaac"][i0:i1]
        assert n > 0 and len(tok_w) == len(trans_w) == len(quat_w) == len(jp_w) == n, \
            f"{stem}[{i0}:{i1}] tokens/trans_m/quat_wxyz/jp_isaac 行数不一致"
        wins.append({"stem": stem, "npz_path": s["npz_path"], "i0": i0, "i1": i1,
                     "v_med": s.get("v_med"), "tok_w": tok_w, "trans_w": trans_w})
    print(f"[load] {len(segs)} windows from {len(cache)} npz "
          f"({segments_json})")
    return wins, cache, md5s


def write_outputs(out_dir: str, res: dict, vb: np.ndarray, keep: np.ndarray,
                  seg_src: np.ndarray) -> None:
    """九件 np.save 落盘（原 main 落盘段拆出；文件名与保存顺序逐字未动）。"""
    np.save(os.path.join(out_dir, "token.npy"), res["token"])
    np.save(os.path.join(out_dir, "mode.npy"), res["mode"])
    np.save(os.path.join(out_dir, "mode_id.npy"), res["mode_id"])
    np.save(os.path.join(out_dir, "ul.npy"), res["ul"])
    np.save(os.path.join(out_dir, "angle_bin.npy"), res["angle_bin"])
    np.save(os.path.join(out_dir, "segment_bounds.npy"), res["bounds"])
    np.save(os.path.join(out_dir, "window_keep_mask.npy"), keep)
    np.save(os.path.join(out_dir, "segment_source.npy"), seg_src)
    np.save(os.path.join(out_dir, "vb_speed.npy"), vb)


def write_metas(out_dir: str, args: argparse.Namespace, n: int, m: int,
                edges: np.ndarray, counts: dict, md5s: dict, keep: np.ndarray,
                segment_detail: list) -> None:
    """build_meta.json 与 vb_speed_meta.json 落盘（原 main meta 段拆出）。

    两个 meta dict 字面量键顺序逐字保留（json.dump 产物字节可比）；
    generated_at 在本函数内取一次、两文件共用（与原实现同语义）。
    """
    now = datetime.datetime.now().isoformat(timespec="seconds")
    meta = {
        "experiment": "D057",
        "generated_at": now,
        "provenance": (
            "WBT 转换器 npz（manifest.json）-> mine_wbt_walk_segments.py 滑窗挖掘 "
            f"-> walk_segments.json({args.segments_json}) 逐窗原样拼装；"
            "无过采样（identity），每窗=一段（segment_bounds=窗界）"),
        "segments_json": args.segments_json,
        "mode_table": [{"embed_idx": MODE_ID, "mode_name": "WALK",
                        "mode_id_hpp": MODE_HPP_ID}],
        "oversample_plan": {"mode": "identity", "copies": []},
        "segment_detail": segment_detail,
        "bin_rules": {"dt_s": args.dt, "speed_smooth_frames": args.smooth,
                      "v_thresh_mps": args.v_thresh, "theta0_k": args.theta0_k,
                      "n_dbins": 8, "bin0": "段首 theta0 ±22.5° = 前向（v1/v2 同款）",
                      "low_speed_rule": "继承前一有效 bin，段首无有效帧则 bin0",
                      "angle_bin_source": "build_b4lite_vae_inputs.angle_bins 逐窗算（零漂移 import）"},
        "window": args.window,
        "n_windows_mask": int(len(keep)),
        "fps": FPS,
        "total_frames": int(n),
        "n_segments": int(m),
        "dual_mode_files": "mode.npy 与 mode_id.npy 同内容（int64 全 2=WALK；"
                           "v1/v2 训练器双兼容）",
        "vb_labels": {"files": ["vb_speed.npy", "vb_speed_meta.json"],
                      "policy": LABEL_POLICY},
        "outputs": ["token.npy", "mode.npy", "mode_id.npy", "ul.npy",
                    "angle_bin.npy", "segment_bounds.npy", "window_keep_mask.npy",
                    "segment_source.npy", "vb_speed.npy", "vb_speed_meta.json"],
    }
    with open(os.path.join(out_dir, "build_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    vb_meta = {
        "experiment": "D057",
        "generated_at": now,
        "label_policy": LABEL_POLICY,
        "ref_frames": "all_packed_walk_windows",
        "n_frames": int(n),
        "edges": [float(e) for e in edges],
        "counts": counts,
        "fps": FPS,
        "speed_rule": "v[t]=||trans[t]-trans[t-1]||×fps（XY 后向差分，"
                      "build_d048n frame_speed_bwd 零漂移 import；逐窗算，窗内 v[0]=v[1]）",
        "digitize_rule": "vb=clip(digitize(v,edges),0,2)（build_d048n L18 同口径）",
        "segments_json": args.segments_json,
        "source_npz_md5": md5s,
    }
    with open(os.path.join(out_dir, "vb_speed_meta.json"), "w", encoding="utf-8") as f:
        json.dump(vb_meta, f, ensure_ascii=False, indent=1)
    print(f"[write] {out_dir}: 四件套 + bounds/mask/source + vb_speed + 双 meta")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="D057 G4: WBT 行走窗 → VAE v1/v2 输入四件套 + vb 速度标签")
    ap.add_argument("--segments-json", default="",
                    help="mine_wbt_walk_segments.py 产物 walk_segments.json")
    ap.add_argument("--out-dir", default="",
                    help="输出目录（默认 = segments-json 所在目录/vae_inputs_wbt）")
    ap.add_argument("--window", type=int, default=10,
                    help="窗口长（与训练侧 window=10 一致，keep 掩码长度用）")
    ap.add_argument("--dt", type=float, default=0.02, help="v2 builder 同款默认")
    ap.add_argument("--smooth", type=int, default=5, help="速度滑动平均帧数")
    ap.add_argument("--v-thresh", type=float, default=0.05, help="有效方向速度阈值 m/s")
    ap.add_argument("--theta0-k", type=int, default=10, help="theta0 圆均值取前 K 个有效帧")
    ap.add_argument("--selftest", action="store_true", help="numpy-only 自测（不读文件）")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if ANGLE_BINS is None:
        raise SystemExit(f"angle_bins/frame_speed import 失败: {ANGLE_BINS_ERR}")
    if not args.segments_json:
        ap.error("--segments-json 必填（或用 --selftest）")

    with open(args.segments_json, encoding="utf-8") as f:
        segs = json.load(f)
    assert isinstance(segs, list) and segs, \
        f"walk_segments.json 应为非空列表: {args.segments_json}"
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.segments_json)), "vae_inputs_wbt")
    os.makedirs(out_dir, exist_ok=True)

    wins, cache, md5s = load_windows(segs, args.segments_json)

    res = pack_windows(wins, args.dt, args.smooth, args.v_thresh, args.theta0_k)
    edges, vb = vb_tercile(res.pop("v_all"))
    verify_packed(res, vb)
    n, m = len(res["token"]), len(res["bounds"])
    keep = window_keep_mask(res["bounds"], np.zeros(m, dtype=np.int64),
                            args.window, 1.0, 0)
    assert keep.all() and len(keep) == sum(
        max(int(b[1] - b[0]) - (args.window - 1), 0) for b in res["bounds"]), \
        "keep 掩码长度与窗口枚举不符"
    counts = {int(b): int(c) for b, c in zip(*np.unique(vb, return_counts=True))}
    print(f"[pack] N={n} M={m} edges={edges.tolist()} vb_counts={counts}")

    seg_src = np.arange(m, dtype=np.int64)  # identity：无过采样，每窗自成源段
    write_outputs(out_dir, res, vb, keep, seg_src)
    write_metas(out_dir, args, n, m, edges, counts, md5s, keep,
                res["segment_detail"])


def selftest() -> None:
    """numpy-only 自测：合成两窗 token/trans 测拼装/bounds/edges 三分位/vb 计数。

    两窗各含两段匀速（0.15/0.30 与 0.30/0.45 m/s），拼装后全帧速度恰为
    0.15×134 / 0.30×133 / 0.45×134——三分位边界严格落在层间空隙（0.2/0.4），
    digitize 结果对浮点抖动稳健。angle_bins import 失败时该断言 SKIP。

    selftest 脚手架来自 wbt_common（D057 重构）。
    """
    if ANGLE_BINS is None:
        print(f"[selftest] SKIP angle_bin check: angle_bins import 失败 "
              f"({ANGLE_BINS_ERR})")
    log = CheckLog()

    rng = np.random.default_rng(0)
    tok_a = rng.standard_normal((200, 64)).astype(np.float32)
    tok_b = rng.standard_normal((201, 64)).astype(np.float32)
    trans_a = _ramp_trans([0.15] * 134 + [0.30] * 66)   # 0.15×134, 0.30×66
    trans_b = _ramp_trans([0.30] * 67 + [0.45] * 134)   # 0.30×67, 0.45×134
    wins = [
        {"stem": "synA_ep01_s1", "npz_path": "syn/a.npz", "i0": 0, "i1": 200,
         "v_med": 0.2, "tok_w": tok_a, "trans_w": trans_a},
        {"stem": "synB_ep02_s1", "npz_path": "syn/b.npz", "i0": 0, "i1": 201,
         "v_med": 0.3, "tok_w": tok_b, "trans_w": trans_b},
    ]

    res = pack_windows(wins, 0.02, 5, 0.05, 10)
    edges, vb = vb_tercile(res["v_all"])
    try:
        verify_packed(res, vb)
        log.check("verify_packed assertions", True)
    except AssertionError as exc:
        log.check(f"verify_packed assertions ({exc})", False)

    log.check("pack N=401", len(res["token"]) == 401)
    log.check("pack token concat order",
          np.array_equal(res["token"], np.concatenate([tok_a, tok_b])))
    log.check("pack bounds [[0,200],[200,401]]",
          res["bounds"].tolist() == [[0, 200], [200, 401]])
    log.check("mode all-2 int64", res["mode"].dtype == np.int64
          and int(res["mode"].min()) == int(res["mode"].max()) == MODE_ID)
    log.check("mode_id identical to mode", np.array_equal(res["mode"], res["mode_id"]))
    log.check("ul all-0 int64", res["ul"].dtype == np.int64 and int(res["ul"].max()) == 0)
    log.check("edges tercile ~[0.2, 0.4]", np.allclose(edges, [0.2, 0.4], atol=1e-6))
    log.check("vb counts [134, 133, 134]",
          np.bincount(vb, minlength=3).tolist() == [134, 133, 134])
    if ANGLE_BINS is not None:
        log.check("angle_bin straight-line all 0", int(res["angle_bin"].max()) == 0)
    km = window_keep_mask(res["bounds"], np.zeros(2, dtype=np.int64), 10, 1.0, 0)
    log.check("keep mask all-1 len 383", len(km) == 383 and bool(km.all()))

    if log.failures:
        raise SystemExit(f"selftest FAILED: {log.failures}")
    print("[selftest] build_wbt_vae_inputs: ALL PASS")


if __name__ == "__main__":
    main()
