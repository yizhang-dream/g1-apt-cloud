"""D046b-R3 回归测试：owner 评审三轮（2026-09-07）三个反例入现役函数覆盖。

R2 单测教训：复制实现逻辑的测试抓不住真实过采样身份——本轮全部直接打现役
函数（构建器 oversample_by_mode 真复制 + holdout_ident + playback_window），
无实现复制。

覆盖：
  T1 过采样身份重放：合成 5 原始段（含 intermediate、整段+截段副本）→ 现役
     oversample_by_mode 真复制 → derive_segment_sources 重放身份：副本 token
     与源段前缀逐位一致、copies 计数与 plan 双向一致、副本继承源段组别；
     篡改 plan / n_frames 必须 raise（checksum 反例）
  T2 v3 缺陷性质：同一材料按过采样后段序号奇偶分半必有源段跨两侧（v3 泄漏
     机制），按源原始段分组零跨侧
  T3 缺类反例（owner）：8 类仅 2 类双侧有窗、token 全同 → holdout 宏识别
     0.5（v3 会以 0.5>=2/8 误放行）但 v3.1 必须 identifiable=False；
     全覆盖 + 可分点对照 → True
  T4 同窗比值三态：匀速合成——完整播放 ratio=1.0（v2 口径 100/99≈1.0101
     作废）、播放相提前终止 ratio=1.0、hold 相不入比值（playback_t_used）；
     退化 n_pb<=1 → None；播放步数超参考行数 → ValueError

用法（仓库根目录，纯 numpy）：python apt_g1/d046b_r3_regression_test.py
全部 PASS 时 exit 0。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from apt_g1.build_b4lite_vae_inputs_v2 import oversample_by_mode
    from apt_g1.holdout_ident import (derive_segment_sources, holdout_splits,
                                      nearest_centroid_holdout)
    from apt_g1.isaac.playback_window import playback_same_window_ratio
except ImportError:                    # 服务器执行根平铺布局
    from build_b4lite_vae_inputs_v2 import oversample_by_mode
    from holdout_ident import (derive_segment_sources, holdout_splits,
                               nearest_centroid_holdout)
    from playback_window import playback_same_window_ratio

FAILS = []


def check(name: str, cond: bool, msg: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {msg}" if msg and not cond else ""))
    if not cond:
        FAILS.append(name)


def expect_raises(name: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
        check(name, False, "未抛异常")
    except Exception:
        check(name, True)


# ---- T1/T2 材料：3 mode（M0/M1/intermediate），5 原始段，M0 需复制 550 帧 ----
SEGS = [  # (stem, embed_idx, n_frames, material)
    ("a_seg", 0, 300, 0),
    ("b_seg", 0, 150, 0),
    ("c_seg", 1, 800, 0),
    ("d_seg", 1, 200, 0),
    ("e_inter", 2, 120, 1),
]
rng = np.random.default_rng(0)
toks, modes, uls, abs_, bnds, mats = [], [], [], [], [], []
off = 0
for i, (st, mi, n, ma) in enumerate(SEGS):
    toks.append(rng.normal(size=(n, 3)) + 100.0 * i)   # 逐段可辨识，token 等价可查
    modes.append(np.full(n, mi, dtype=np.int64))
    uls.append(np.zeros(n, dtype=np.int64))
    abs_.append(np.zeros(n, dtype=np.int64))
    bnds.append([off, off + n])
    mats.append(ma)
    off += n
arrs = {"token": np.concatenate(toks).astype(np.float32),
        "mode_id": np.concatenate(modes), "ul": np.concatenate(uls),
        "angle_bin": np.concatenate(abs_), "bounds": np.asarray(bnds, np.int64),
        "stems": [s[0] for s in SEGS], "materials": np.asarray(mats, np.int64)}

os_res, plan = oversample_by_mode(arrs)
POST = os_res["bounds"]
PLAN_META = {
    "mode_table": [{"embed_idx": 0, "mode_name": "M0"},
                   {"embed_idx": 1, "mode_name": "M1"},
                   {"embed_idx": 2, "mode_name": "intermediate"}],
    "oversample_plan": plan,
    # 故意乱序（镜像构建器 segment_detail 不按 stem 排序的现状）
    "segment_detail": {"train": [
        {"stem": st, "target": ["M0", "M1", "intermediate"][mi], "n_frames": n}
        for st, mi, n, _ in sorted(SEGS, reverse=True)]},
}


def t12_identity_and_split() -> None:
    print("[T1/T2] 过采样身份重放 + 源段分组不跨侧")
    src, stems = derive_segment_sources(POST, PLAN_META)
    check("副本数=3（2 整段 a/b + 1 截段 a）", len(src) == len(SEGS) + 3,
          f"got {len(src)}")
    check("原始段前缀映射自身", (src[:len(SEGS)] == np.arange(len(SEGS))).all())
    check("copies 计数与 plan 一致（a_seg2/b_seg1）",
          plan["0"]["copies"] == {"a_seg": 2, "b_seg": 1}, str(plan["0"]["copies"]))
    ok_tok = True
    for j in range(len(SEGS), len(src)):
        i, a, b = int(src[j]), int(POST[j, 0]), int(POST[j, 1])
        ia = int(POST[i, 0])
        # 副本侧用过采样后数组（POST 坐标），源段侧用原始数组（原始坐标）
        ok_tok &= np.array_equal(os_res["token"][a:b],
                                 arrs["token"][ia:ia + (b - a)])
    check("每个副本 token 与源段前缀逐位一致", ok_tok)
    expect_raises("篡改 copies 计数 -> raise",
                  derive_segment_sources, POST,
                  {**PLAN_META, "oversample_plan":
                   {**plan, "0": {**plan["0"], "copies": {"a_seg": 3, "b_seg": 1}}}})
    bad_detail = {**PLAN_META, "segment_detail": {"train": [
        {"stem": st, "target": ["M0", "M1", "intermediate"][mi],
         "n_frames": n + (1 if st == "c_seg" else 0)}
        for st, mi, n, _ in SEGS]}}
    expect_raises("篡改 n_frames -> raise", derive_segment_sources, POST, bad_detail)

    # T2：v3 旧口径（过采样后段序号奇偶）a_seg 原段 idx0(A) 与其副本 idx5(B) 跨侧
    old_half = np.arange(len(src)) % 2 == 0
    check("v3 旧口径下 a_seg 跨两侧（缺陷性质成立）",
          len(set(old_half[src == 0])) == 2)
    sp = holdout_splits(src, stems)
    check("源段分组下所有原始段零跨侧",
          all(len(set(sp["stem_parity"][src == i])) == 1 for i in range(len(SEGS))))
    check("副本继承源段组别（3 个副本逐一核对）",
          all(sp["stem_parity"][j] == sp["stem_parity"][int(src[j])]
              for j in range(len(SEGS), len(src))))
    check("对照分组 stem_hash/stem_halves 同样零跨侧",
          all(len(set(sp[nm][src == i])) == 1
              for nm in ("stem_hash", "stem_halves") for i in range(len(SEGS))))


def t3_coverage_gate() -> None:
    print("[T3] 缺类覆盖门（owner 反例：8 类仅 2 类可留出、token 全同）")
    pts = np.zeros((40, 4))
    labels = np.repeat(np.arange(8), 5)
    half = np.zeros(40, dtype=bool)
    half[0:3] = True                       # c0: 双侧
    half[5:8] = True                       # c1: 双侧
    half[10:] = True                       # c2..c7: 只在 A 半（无留出窗）
    r = nearest_centroid_holdout(pts, labels, list(range(8)), half,
                                 [f"c{i}" for i in range(8)])
    check("coverage=partial 且缺类单列 c2..c7",
          r["coverage"] == "partial"
          and r["classes_missing_holdout"] == [f"c{i}" for i in range(2, 8)])
    check("缩小任务宏识别=0.5（复现 owner 数字）", r["holdout_macro_recall"] == 0.5)
    check("v3 随机线确实会放行（0.5>=2/8，缺陷性质）", 0.5 >= 2 * (1 / 8))
    check("v3.1 判 identifiable=False（覆盖门拦截）", r["readout_identifiable"] is False)
    check("reason 指向覆盖不足", "coverage_insufficient" in (r["reason"] or ""))

    pts2 = np.eye(8)[np.repeat(np.arange(8), 6)]      # 全覆盖 + 完全可分对照
    labels2 = np.repeat(np.arange(8), 6)
    half2 = np.tile([True, True, True, False, False, False], 8)
    r2 = nearest_centroid_holdout(pts2, labels2, list(range(8)), half2,
                                  [f"c{i}" for i in range(8)])
    check("全覆盖可分对照 -> identifiable=True",
          r2["coverage"] == "full" and r2["readout_identifiable"] is True)


def t4_playback_window() -> None:
    print("[T4] 同窗路径比：完整播放 / 提前终止 / hold / 退化 / 防御")
    n = 100
    ref = np.stack([0.1 * (np.arange(n) + 1), np.zeros(n)], 1)   # 行 t = (0.1(t+1),0)
    def robot(n_pb: int) -> np.ndarray:
        # step 后状态=参考行 t（门内 q_track_mae 同口径）；行 0 为抖动重置点
        return np.stack([0.1 * np.arange(n_pb + 1), np.zeros(n_pb + 1)], 1)

    pw = playback_same_window_ratio(robot(n), ref)
    check("完整播放 ratio=1.0（v2 口径为 100/99≈1.0101）", pw["ratio"] == 1.0,
          str(pw))
    old = (n * 0.1) / ((n - 1) * 0.1)
    check("v2 旧口径确实 1.0101（对照，证明修的就是它）",
          abs(old - n / (n - 1)) < 1e-12 and abs(old - 1.0101) < 1e-4)
    check("双侧间隔数 = n_pb-1 = 99", pw["n_intervals"] == n - 1)

    pw2 = playback_same_window_ratio(robot(50), ref)          # step 50 摔倒（done 步剔除）
    check("播放相提前终止 ratio=1.0、t_used=50", pw2["ratio"] == 1.0
          and pw2["playback_t_used"] == 50, str(pw2))
    check("hold 相不入比值（本函数只见播放相点位）",
          pw2["playback_t_used"] != n)

    check("n_pb=1 -> ratio None", playback_same_window_ratio(robot(1), ref)["ratio"] is None)
    check("n_pb=0 -> ratio None", playback_same_window_ratio(robot(0), ref)["ratio"] is None)
    expect_raises("播放步数超参考行数 -> ValueError",
                  playback_same_window_ratio, robot(101), ref)


if __name__ == "__main__":
    t12_identity_and_split()
    t3_coverage_gate()
    t4_playback_window()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("\nALL PASS")
