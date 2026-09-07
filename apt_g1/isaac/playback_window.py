"""播放相同窗路径比（D046b-R3，owner 评审三轮 P2）——纯 numpy，便于无 Isaac 回归测试。

时间对应约定（b3p_gate_isaac.py 全门统一，与 q_track_mae_vs_ref_rad 同口径）：
token 行 t 在第 t 步被消费（t 从 0 计），step 后状态 ↔ 参考行 t；重置位姿
（jitter 站姿起步）没有参考对应帧——参考在 t=0 时就在行 0，机器人却要先走一步
沉降。因此同窗比值两侧都不含第一间隔：实际侧取 traj_pb[1:]（step 后位姿序列，
不含重置点）的相邻差分，参考侧取 trans_m[:n_pb]（行 0..n_pb-1）的相邻差分，
两侧均为 n_pb-1 个间隔、覆盖同一参考行窗。

v2 旧口径缺陷：实际 n_pb 个间隔（含重置→首步）vs 参考最多 n_pb-1 个间隔——
完整播放（n_pb=n_rows）时系统性偏高 ~1/(n_rows-1)（100 步匀速例 100/99≈1.0101）；
提前终止时恰好同窗（n_pb<n_rows 未触参考截断），故该缺陷只在完整播放显形。
历史 run 不回刷；hold 相不入本比值（hold_disp_m 单列）。
"""
from __future__ import annotations

import numpy as np


def playback_same_window_ratio(traj_pb: np.ndarray, trans_m_xy: np.ndarray) -> dict:
    """播放相同窗路径比。

    traj_pb: (n_pb+1, 2) 重置点 + 逐播放 step 后 xy 位姿（done 步已剔除，
    与门主循环口径一致）；trans_m_xy: (n_rows, 2) 参考 xy 轨迹。
    返回 dict(ratio, robot_path_m, ref_path_m, n_intervals, playback_t_used)；
    有效间隔数 <1 或参考路径 <=0.5 m 时 ratio=None（沿用门内 guard）；
    n_pb 超出参考行数属调用方状态错位 → ValueError。
    """
    traj_pb = np.asarray(traj_pb, dtype=np.float64)
    trans_m_xy = np.asarray(trans_m_xy, dtype=np.float64)
    n_pb = len(traj_pb) - 1
    if n_pb > len(trans_m_xy):
        raise ValueError(f"播放步数 {n_pb} 超出参考行数 {len(trans_m_xy)}")
    rob = traj_pb[1:]
    robot_path = (float(np.linalg.norm(np.diff(rob, axis=0), axis=1).sum())
                  if len(rob) > 1 else 0.0)
    ref_pts = trans_m_xy[:n_pb]
    ref_path = (float(np.linalg.norm(np.diff(ref_pts, axis=0), axis=1).sum())
                if len(ref_pts) > 1 else 0.0)
    return {
        "ratio": (round(robot_path / ref_path, 6) if ref_path > 0.5 else None),
        "robot_path_m": robot_path,
        "ref_path_m": ref_path,
        "n_intervals": max(n_pb - 1, 0),
        "playback_t_used": int(n_pb),
    }
