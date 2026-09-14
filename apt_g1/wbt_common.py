"""D057 WBT 线共享底层：LeRobot 双 schema parquet 读取、根 XY 速度、selftest
脚手架。scan_wbt_datasets / convert_wbt_g1_parquet / mine_wbt_walk_segments /
build_wbt_vae_inputs 四脚本零漂移共用；数值路径逐字搬运自各脚本原实现
（convert_wbt_g1_parquet.py @d4f88a0 / scan_wbt_datasets.py @617e2f0）。等价性
纪律：改本模块任何数值口径前，必须重跑四脚本 --selftest 并在服务器做产物
md5 对拍（先例见 12b7a68 的 --batch-encode 等价验证）。"""
from __future__ import annotations

import numpy as np

# 新款 grouped schema 的关节分组列与拼接顺序（= 契约：腿+腰 15 -> L 臂 7 -> R 臂 7）
GROUPED_JOINT_COLS = (
    "observation.state.lower_body",
    "observation.state.left_arm",
    "observation.state.right_arm",
)


def xy_speed_series(pos_xy, fps):
    """根 XY 后向差分速度（单一实现）：v[t]=‖p[t]−p[t−1]‖×fps，长度 N−1。

    N<2（无法差分）返回空数组。与 build_d048n_vb_speed_labels.frame_speed_bwd
    的 N 长度、v[0]=v[1] 口径不同，勿混用。
    """
    pos_xy = np.asarray(pos_xy, dtype=np.float64)
    if len(pos_xy) < 2:
        return np.empty(0)
    return np.linalg.norm(np.diff(pos_xy, axis=0), axis=1) * float(fps)


def xy_speed_stats(trans_xy, fps):
    """(v_med, v_p90) m/s：XY 后向差分 × fps 口径（源帧率，wbt 块专用，区别于
    D044 b4lite 的 @50Hz kinematics_stats）。内部取 [:, :2]，数值实现基于
    xy_speed_series（公式相同、数值相同）。"""
    xy = np.asarray(trans_xy, dtype=np.float64)[:, :2]
    v = xy_speed_series(xy, fps)
    if not len(v):
        return 0.0, 0.0
    return float(np.median(v)), float(np.percentile(v, 90))


def read_wbt_parquet(path, need_joints=True):
    """单 parquet 文件双 schema 自适应读取（仿 tmp/wbt_probe.py read_file）。

    合并 convert read_parquet_file（全字段）与 scan read_file（省 IO）两口径：
    返回 {schema, ep, frame, root7, j29}——schema 为 "q36"|"grouped"；ep 为
    int64 数组；frame 优先 frame_index、次选 index、皆缺为 None（按行序）；
    root7 为 (n,7) float64（q36 时=整列前 7 维，grouped 时=state_base_pose
    列）；j29 为 (n,29) float64。need_joints=False 时 j29 恒为 None，且
    grouped 分支不读三个关节分组列（省 IO；列名齐全性校验仍做，q36 分支本就
    整列读，仅不返回关节）。pyarrow 延迟装载，模块导入/selftest 不触发。
    两种 schema 皆缺（grouped 列不齐）→ ValueError。"""
    import pyarrow.parquet as pq  # 惰性 import：--selftest 分支不得触发

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
        root7, j29 = q36[:, :7], (q36[:, 7:36] if need_joints else None)
    else:
        schema = "grouped"
        missing = [g for g in GROUPED_JOINT_COLS if g not in names]
        if missing or "observation.state.state_base_pose" not in names:
            raise ValueError(f"{path}: neither q36 nor complete grouped schema "
                             f"(missing {missing})")
        root7 = col("observation.state.state_base_pose")
        # 拼接顺序即契约：lower_body[15] + left_arm[7] + right_arm[7] = 29
        j29 = (np.concatenate([col(g) for g in GROUPED_JOINT_COLS], axis=1)
               if need_joints else None)
    if (root7.ndim != 2 or root7.shape[1] != 7
            or (j29 is not None and j29.shape[1] != 29)):
        raise ValueError(f"{path}: root7 {root7.shape} / j29 {j29.shape} not (n,7)/(n,29)")
    return {"schema": schema, "ep": ep, "frame": frame, "root7": root7, "j29": j29}


class CheckLog:
    """--selftest 共用脚手架：逐项打印 + 失败项收集；最终 ALL PASS/FAILED
    汇总与退出码仍由各脚本自理。"""

    def __init__(self):
        self.failures: list[str] = []

    def check(self, name, cond, detail=""):
        print(f"[selftest] {'PASS' if cond else 'FAIL'} {name}"
              + ("" if cond or not detail else f"  ({detail})"))
        if not cond:
            self.failures.append(name)
