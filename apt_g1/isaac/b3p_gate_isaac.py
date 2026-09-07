"""D044 / DS official-data line B3' gate: class-level Isaac oracle replay
survival over CSV-converted BONES-SEED G1 tokens.

Gate (DS_OFFICIAL_DATA_PLAN §3, unchanged by the D043 data-source switch):
per core class >= 10 segments x 500-step Isaac oracle replay (D034 mechanism),
segments cross-sampled over actors; class PASSES iff survival >= 95%.

Input = the manifest.json written by convert_bones_g1_csv.py (one npz per
segment: tokens (n,64) @50 Hz + jp_isaac reference + trans_m). Replay loop is
the D037 rehearsal pattern verbatim (token -> frozen SonicTorchDecoder ->
q_des, env-owned closed-loop history, policy/VAE/router bypassed,
jitter_and_reset standing start, zero aux action). Canonical apt_flat_env.py
stays untouched.

Step budget: max(n_rows, --steps) per segment -- the pre-registered 500-step
budget, never truncating longer segments; the token index clamps at the last
row exactly like D034/D037 (hold phase after token end). fall_step is
recorded so falls during playback vs hold can be separated post hoc.

v2 aggregation (owner 评审二轮修正 2026-09-07，D047 教训落实；历史 run 不回刷):
  - 类级聚合改按唯一 stem 计片段：n_segments=唯一片段数，n_replays=回放条数
    分列。旧版把 {stem}__seed{seed} 每条回放当一个片段（look 4 段 x3 seed 曾被
    记 n_segments=12 gate=true），加 seed 即可跨过 >=10 段门槛，违反预注册。
  - 主判 seed = min(seeds)（预注册缺省 seeds="0" 即 seed0 主判）：类级存活率
    只按片段在主判 seed 下是否 completed 计；全 seed 存活片段数与逐 seed 回放
    存活另列（描述性，不入门）。
  - playback_path_ratio 改同窗比值：分子=播放相实际有效步数路径（done 步不计），
    分母=参考轨迹截断到同一有效步数；两侧步数 playback_t_used/playback_t_total
    落盘。旧口径（分子截断/分母全长）在提前摔倒时系统性低估，不再可比。

Usage (lab-ts, Isaac wrapper, cwd=GR00T-WholeBodyControl):
  nohup bash /tmp/run_apt_isaac.sh \
    /home/cvgluser/ros2_data/apt_g1/isaac/b3p_gate_isaac.py \
    --manifest /home/cvgluser/ros2_data/apt_g1/data/ds_bones/g1_b3p/manifest.json \
    --out /home/cvgluser/ros2_data/apt_g1/data/ds_bones/g1_b3p/gate_result.json \
    > /home/cvgluser/ros2_data/apt_g1/data/ds_bones/g1_b3p/gate.log 2>&1 < /dev/null & disown
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

DEFAULT_DIR = "/home/cvgluser/ros2_data/apt_g1/data/ds_bones/g1_b3p"


def build_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=f"{DEFAULT_DIR}/manifest.json",
                    help="manifest.json from convert_bones_g1_csv.py --sample")
    ap.add_argument("--out", default=f"{DEFAULT_DIR}/gate_result.json")
    ap.add_argument("--steps", type=int, default=500,
                    help="minimum replay steps per segment (B3' budget)")
    ap.add_argument("--seeds", default="0", help="jitter seed(s), comma separated")
    ap.add_argument("--only-class", default=None,
                    help="replay only this class from the manifest")
    ap.add_argument("--min-rows", type=int, default=100,
                    help="skip segments shorter than this many @50Hz rows (2 s)")
    ap.add_argument("--router-model-dir",
                    default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_final")
    return ap


def load_segments(manifest_path, only_class, min_rows):
    entries = json.load(open(manifest_path))
    segs = []
    for e in entries:
        if "error" in e:
            continue
        if only_class and e["class"] != only_class:
            continue
        npz = np.load(e["npz_path"], allow_pickle=False)
        meta = json.loads(str(npz["meta"]))
        if meta["n_rows_enc"] < min_rows:
            print(f"[skip] {meta['stem']}: {meta['n_rows_enc']} rows < {min_rows}")
            continue
        segs.append({
            "stem": meta["stem"], "class": meta["class"], "actor": meta["actor"],
            "tokens": npz["tokens"].astype(np.float32),
            "jp_isaac": npz["jp_isaac"].astype(np.float64),
            "trans_m": npz["trans_m"].astype(np.float64),
            "roundtrip_mae": meta.get("roundtrip_mae"),
        })
    print(f"[load] {len(segs)} segments from {manifest_path}")
    return segs


def main():
    from isaaclab.app import AppLauncher

    ap = build_args()
    AppLauncher.add_app_launcher_args(ap)
    # server has no display: viewport/hydra init segfaults (repo gotcha)
    ap.set_defaults(headless=True)
    cli = ap.parse_args()
    seeds = [int(s) for s in cli.seeds.split(",")]

    segs = load_segments(cli.manifest, cli.only_class, cli.min_rows)
    if not segs:
        raise SystemExit("no replayable segments in manifest")
    totals = {s["stem"]: max(len(s["tokens"]), cli.steps) for s in segs}
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(cli)
    sim_app = app_launcher.app

    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.eval_apt_isaac import jitter_and_reset

    class BonesReplayEnv(AptFlatG1Env):
        """Token oracle (D034 pattern, same subclass shape as
        replay_bones_tokens_isaac.py): _compute_q_des bypasses policy/VAE/
        router and decodes the offline token stream with the env-owned
        closed-loop history."""

        _oracle_tokens: torch.Tensor | None = None
        _oracle_idx: int = 0

        def _compute_q_des(self, phase, aux, res=None):
            i = min(self._oracle_idx, self._oracle_tokens.shape[0] - 1)
            tokens = self._oracle_tokens[i].unsqueeze(0).expand(self.num_envs, -1)
            self._oracle_idx += 1
            action_t = self._decoder.decode(
                tokens,
                self._hist_ang_vel,
                self._hist_joint_pos,
                self._hist_joint_vel,
                self._hist_last_actions,
                self._hist_gravity,
            )
            return self._sonic_default_t + action_t * self._sonic_scale_t

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            self._oracle_idx = 0

    cfg = AptFlatG1EnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = max(totals.values()) / 50.0 + 30.0
    cfg.router_model_dir = cli.router_model_dir
    env = BonesReplayEnv(cfg)

    results, all_ok = {}, True
    for seg in segs:
        tokens, jp_ref = seg["tokens"], seg["jp_isaac"]
        n_rows = len(tokens)
        total = totals[seg["stem"]]
        env._oracle_tokens = torch.from_numpy(tokens).to(env.device)
        for seed in seeds:
            jitter_and_reset(env, seed)
            env._oracle_idx = 0
            xy0 = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy()
            traj = [xy0]
            traj_pb = [xy0]  # D047 口径注记：播放相路径单列（见 playback_path_m）
            h_min, h_end, fall_step, steps_done = float("inf"), None, None, 0
            q_err_ref, q_err_pd = [], []
            for t in range(total):
                action = torch.zeros(
                    env.num_envs, env.cfg.action_space, dtype=torch.float32, device=env.device
                )
                obs, reward, term, trunc, _ = env.step(action)
                steps_done = t + 1
                done = bool(term[0]) or bool(trunc[0])
                # DirectRLEnv auto-resets inside step() on the done step ->
                # exclude it from metrics (D037 note)
                if not done:
                    h = float(env.robot.data.root_pos_w[0, 2].item())
                    h_min = min(h_min, h)
                    h_end = h
                    xy_now = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy()
                    traj.append(xy_now)
                    if t < n_rows:  # token playback phase
                        traj_pb.append(xy_now)
                        q_des_s = env._q_des[0].detach().cpu().numpy()
                        q_act_s = env.robot.data.joint_pos[0, env._body_idx].detach().cpu().numpy()
                        q_err_pd.append(float(np.abs(q_act_s - q_des_s).mean()))
                        q_err_ref.append(float(np.abs(q_act_s - jp_ref[t]).mean()))
                if bool(term[0]):
                    fall_step = t
                    break
                if bool(trunc[0]):
                    break
            traj = np.asarray(traj)
            seg_len = np.linalg.norm(np.diff(traj, axis=0), axis=1)
            path_len = float(seg_len.sum())
            disp_vec = traj[-1] - traj[0]
            dur = steps_done / 50.0
            completed = fall_step is None and steps_done >= total
            ref_path = float(np.linalg.norm(np.diff(seg["trans_m"][:, :2], axis=0), axis=1).sum())
            # D047 path-ratio 口径注记：realized_path_ratio 的分子累计到播放+hold
            # 结束/摔倒，与完整参考不同时间窗，不可当同窗比值读；不回刷历史 run。
            traj_pb = np.asarray(traj_pb)
            playback_path = (float(np.linalg.norm(np.diff(traj_pb, axis=0), axis=1).sum())
                             if len(traj_pb) > 1 else 0.0)
            # v2 同窗比值（owner 评审二轮 P2-4）：播放相实际有效步数 n_pb（done 步
            # 不计入），参考轨迹截断到 trans_m[:n_pb+1]；两侧步数落盘。
            n_pb = len(traj_pb) - 1
            ref_pb_pts = seg["trans_m"][:min(n_pb + 1, n_rows), :2]
            ref_path_pb = (float(np.linalg.norm(np.diff(ref_pb_pts, axis=0), axis=1).sum())
                           if len(ref_pb_pts) > 1 else 0.0)
            hold_disp = (float(np.linalg.norm(traj[-1] - traj_pb[-1]))
                         if steps_done > n_rows else None)
            key = f"{seg['stem']}__seed{seed}"
            results[key] = {
                "class": seg["class"], "actor": seg["actor"], "stem": seg["stem"],
                "seed": seed, "n_rows_tokens": n_rows, "total_steps": total,
                "completed": completed,
                "fall_step": fall_step,
                "fall_during_playback": fall_step is not None and fall_step < n_rows,
                "steps": steps_done,
                "h_min": round(h_min, 3),
                "h_end": round(h_end, 3) if h_end is not None else None,
                "path_len_m": round(path_len, 2),
                "ref_path_len_m": round(ref_path, 2),
                "disp_norm_m": round(float(np.linalg.norm(disp_vec)), 2),
                "mean_speed_mps": round(path_len / dur, 3) if dur > 0 else None,
                "realized_path_ratio": round(path_len / ref_path, 3) if ref_path > 0.5 else None,
                "playback_path_m": round(playback_path, 2),
                "playback_ref_path_m": round(ref_path_pb, 2),
                "playback_t_used": int(n_pb),
                "playback_t_total": int(n_rows),
                "playback_path_ratio": (round(playback_path / ref_path_pb, 3)
                                        if ref_path_pb > 0.5 else None),
                "hold_disp_m": round(hold_disp, 2) if hold_disp is not None else None,
                "q_track_mae_vs_ref_rad": round(float(np.mean(q_err_ref)), 4) if q_err_ref else None,
                "q_track_mae_pd_rad": round(float(np.mean(q_err_pd)), 4) if q_err_pd else None,
            }
            all_ok &= completed
            r = results[key]
            print(f"[gate] {seg['class']:6s} {seg['stem'][:44]:44s} seed{seed} "
                  f"completed={completed} fall_step={fall_step} h_min={r['h_min']} "
                  f"path={r['path_len_m']}/{r['ref_path_len_m']}m "
                  f"qMAE={r['q_track_mae_vs_ref_rad']}", flush=True)

    # ---- class-level gate aggregation
    # v2（owner 评审二轮 P1-3，D047 教训）：回放条数≠片段数——look 4 段×3 seed
    # 曾被旧版记 n_segments=12 gate=true。片段数只按唯一 stem 计；类级存活只看
    # 主判 seed（min(seeds)，预注册缺省 seed0）；加 seed 不能跨 >=10 段门槛。
    primary_seed = min(seeds)
    by_class = {}
    for r in results.values():
        by_class.setdefault(r["class"], []).append(r)
    classes = {}
    for cls in sorted(by_class):
        rs = by_class[cls]
        seg_map = {}
        for r in rs:
            seg_map.setdefault(r["stem"], {})[r["seed"]] = r
        n_seg = len(seg_map)
        prim = [sd.get(primary_seed) for sd in seg_map.values()]
        prim_missing = sum(1 for p in prim if p is None)
        n_ok = sum(1 for p in prim if p is not None and p["completed"])
        n_all_seed_ok = sum(1 for sd in seg_map.values()
                            if sd and all(v["completed"] for v in sd.values()))
        per_seed = {}
        for s in seeds:
            rs_s = [sd[s] for sd in seg_map.values() if s in sd]
            per_seed[str(s)] = {"n_replays": len(rs_s),
                                "n_survived": sum(1 for r in rs_s if r["completed"])}
        classes[cls] = {
            "primary_seed": primary_seed,
            "n_segments": n_seg,
            "n_replays": len(rs),
            "n_survived_segments_primary": n_ok,
            "n_segments_primary_seed_missing": prim_missing,
            "survival_rate_segments_primary": round(n_ok / n_seg, 4) if n_seg else None,
            "n_segments_survived_all_seeds": n_all_seed_ok,
            "per_seed_survival_replays": per_seed,
            "gate": (n_seg >= 10 and prim_missing == 0 and n_ok / n_seg >= 0.95)
                    if n_seg else False,
            "n_fall_during_playback": sum(1 for r in rs if r["fall_during_playback"]),
            "q_track_mae_median_rad": round(float(np.median(
                [r["q_track_mae_vs_ref_rad"] for r in rs if r["q_track_mae_vs_ref_rad"] is not None])), 4),
            "realized_path_ratio_median": round(float(np.median(
                [r["realized_path_ratio"] for r in rs if r["realized_path_ratio"] is not None])), 3),
            "playback_path_ratio_median": (round(float(np.median(
                [r["playback_path_ratio"] for r in rs
                 if r.get("playback_path_ratio") is not None])), 3)
                if any(r.get("playback_path_ratio") is not None for r in rs) else None),
        }
    out = {
        "exp": "D044 B3' gate (BONES-SEED G1 native CSV -> obs -> tokens -> Isaac oracle replay)",
        "manifest": cli.manifest,
        "steps_budget": cli.steps,
        "seeds": seeds,
        "primary_seed": min(seeds),
        "n_segments": len(segs),
        "n_segments_unique": len({r["stem"] for r in results.values()}),
        "n_replays": len(results),
        "all_completed": all_ok,
        "classes": classes,
        "gate_overall_pass": all(c["gate"] for c in classes.values()) if classes else False,
        "per_segment": results,
    }
    with open(cli.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(classes, indent=1), flush=True)
    print(f"saved {cli.out}")

    # Isaac sometimes hangs on interpreter exit (D037 gotcha): bound close()
    # to a daemon thread with a timeout and hard-exit regardless.
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


if __name__ == "__main__":
    main()
