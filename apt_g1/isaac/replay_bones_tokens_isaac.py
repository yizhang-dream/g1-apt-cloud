"""D037 / DS plan-B line B3 rehearsal: closed-loop Isaac replay of OUR
offline-encoded BONES tokens.

B2 encoded the official GEAR-SONIC sample_data G1 walk pkl into 2003x64 SONIC
tokens offline (encode_bones_smoke.py; lattice-clean, but the offline decoder
roundtrip MAE was 0.564 rad vs the reference). Before building any B3 gate on
top, replay those tokens inside AptFlatG1Env exactly like D034's official-token
oracle replay (oracle_token_replay_isaac.py): recorded token -> frozen
SonicTorchDecoder -> q_des, env-owned closed-loop 10-frame history,
policy/VAE/router bypassed, jitter_and_reset initial standing pose.

This is a REHEARSAL, not a gate: it validates the chain "offline encode ->
Isaac closed-loop execution" end to end and collects the first real numbers
(fall step, heights, realized path, reference tracking). A fall on a single
slow amateur walk segment does not by itself condemn line B (the B3 gate will
require >=10 segments per gait class).

Reference tracking: per-frame |q_act - jp_ref| against the B2 reference
trajectory (pkl jp, 30->50 Hz resampled by the SAME encode_bones_smoke code
and mapped to Isaac joint order via B2's recorded order decision), plus the
D035-style PD tracking MAE (|q_act - q_des|) for continuity. The reference is
a LOOP: path_len ~17 m, first-to-last disp ~0.4 m, mean speed ~0.42 m/s --
the realized PATH LENGTH is the comparison target, not displacement.

Run on lab-ts via the Isaac wrapper (cwd=GR00T-WholeBodyControl):
  cd /home/cvgluser/ros2_data && nohup bash /tmp/run_apt_isaac.sh \
    /home/cvgluser/ros2_data/apt_g1/isaac/replay_bones_tokens_isaac.py \
    --out /home/cvgluser/ros2_data/apt_g1/data/ds_bones/b3_rehearsal/rehearsal.json \
    > /home/cvgluser/ros2_data/apt_g1/data/ds_bones/b3_rehearsal/rehearsal.log \
    2>&1 < /dev/null & disown; echo OK
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

B2_DIR = "/home/cvgluser/ros2_data/apt_g1/data/ds_bones/b2"
B1_PKL = ("/home/cvgluser/ros2_data/apt_g1/data/ds_bones/b1/sample_data/"
          "robot_filtered/210531/walk_forward_amateur_001__A001.pkl")


def build_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--tokens-npy",
        default=f"{B2_DIR}/tokens_walk_forward_amateur_001__A001.npy",
        help="B2 offline-encoded tokens (n, 64) float32, 50 Hz rows",
    )
    ap.add_argument("--ref-pkl", default=B1_PKL, help="B2 source motion pkl")
    ap.add_argument(
        "--smoke-json", default=f"{B2_DIR}/smoke_result.json",
        help="B2 result JSON: reuses its joint-order decision + fps_source "
        "(one source of truth; no re-derivation here)",
    )
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument(
        "--router-model-dir",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_final",
    )
    ap.add_argument(
        "--out",
        default="/home/cvgluser/ros2_data/apt_g1/data/ds_bones/b3_rehearsal/rehearsal.json",
    )
    return ap


def extract(cli):
    """Fail-fast extraction BEFORE paying the Isaac startup: tokens + the B2
    reference joint rows (AS LOADED order, 50 Hz, len == n_tokens). The
    joint-order mapping is applied only after AppLauncher: gear_sonic's g1
    module imports isaaclab.actuators, which needs the sim app up (same
    import order as oracle_token_replay_isaac.py / apt_flat_env)."""
    tokens = np.load(cli.tokens_npy).astype(np.float32)
    assert tokens.ndim == 2 and tokens.shape[1] == 64, f"tokens shape {tokens.shape}"
    smoke = json.load(open(cli.smoke_json))
    order = smoke["joint_order"]["decision"]
    fps_src = float(smoke["fps_source"])

    # encode_bones_smoke is the B2 module (read-only): reuse its pkl loader and
    # resampler so the reference here is identical to what B2 encoded.
    from encode_bones_smoke import load_pkl, resample, unwrap

    obj = unwrap(load_pkl(cli.ref_pkl)[0])
    dof = obj["dof"].astype(np.float64)
    assert dof.ndim == 2 and dof.shape[1] == 29, f"dof shape {dof.shape}"
    dof_rs = resample(dof, fps_src, 50.0) if fps_src != 50.0 else dof
    trans = obj["root_trans_offset"].astype(np.float64)
    trans_rs = resample(trans, fps_src, 50.0) if fps_src != 50.0 else trans

    n = min(len(tokens), len(dof_rs))
    if len(tokens) != len(dof_rs):
        print(f"[extract] WARN tokens={len(tokens)} ref={len(dof_rs)} -> using {n}")
    tokens, dof_rs, trans_rs = tokens[:n], dof_rs[:n], trans_rs[:n]
    lat = tokens.astype(np.float64) * 16.0
    lattice_viol = int(np.sum(np.abs(lat - np.round(lat)) > 0.05))
    path_len = float(np.linalg.norm(np.diff(trans_rs[:, :2], axis=0), axis=1).sum())
    disp = float(np.linalg.norm(trans_rs[-1, :2] - trans_rs[0, :2]))
    dur = n / 50.0
    ref = {
        "pkl": cli.ref_pkl,
        "joint_order": order,
        "fps_source": fps_src,
        "n_frames": int(n),
        "path_len_m": round(path_len, 2),
        "first_to_last_horiz_disp_m": round(disp, 3),
        "mean_speed_mps": round(path_len / dur, 4),
        "duration_s": round(dur, 2),
    }
    print(
        f"[extract] tokens {tokens.shape} lattice_viol={lattice_viol} "
        f"ref(order={order}) path_len={path_len:.1f} m disp={disp:.2f} m "
        f"mean speed={path_len / dur:.3f} m/s over {dur:.1f} s",
        flush=True,
    )
    return tokens, dof_rs, ref, lattice_viol


def main():
    from isaaclab.app import AppLauncher

    ap = build_args()
    AppLauncher.add_app_launcher_args(ap)
    # server has no display: viewport/hydra init segfaults (repo gotcha)
    ap.set_defaults(headless=True)
    cli = ap.parse_args()
    seeds = [int(s) for s in cli.seeds.split(",")]

    tokens, dof_ref, ref_info, lattice_viol = extract(cli)
    n_rows = len(tokens)
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(cli)
    sim_app = app_launcher.app

    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.eval_apt_isaac import jitter_and_reset
    from gear_sonic.envs.manager_env.robots.g1 import (
        G1_ISAACLAB_TO_MUJOCO_DOF,
        G1_MUJOCO_TO_ISAACLAB_DOF,
    )

    # joint-order mapping (post-AppLauncher, see extract docstring):
    # reference rows -> MuJoCo order (per B2's recorded decision) -> IsaacLab.
    i2m = np.asarray(G1_ISAACLAB_TO_MUJOCO_DOF)
    m2i = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)
    jp_ref_mj = dof_ref if ref_info["joint_order"] == "mujoco" else dof_ref[:, i2m]
    jp_ref_isaac = jp_ref_mj[:, m2i]

    class BonesReplayEnv(AptFlatG1Env):
        """Token oracle for the B2 offline-encoded BONES tokens (D034 pattern:
        _compute_q_des bypasses policy/VAE/router entirely and decodes the
        offline token stream with the env-owned closed-loop history.
        Deliberately a subclass -- canonical apt_flat_env.py stays untouched)."""

        _oracle_tokens: torch.Tensor | None = None
        _oracle_idx: int = 0

        def _compute_q_des(self, phase, aux, res=None):
            i = min(self._oracle_idx, self._oracle_tokens.shape[0] - 1)
            tokens = self._oracle_tokens[i].unsqueeze(0).expand(self.num_envs, -1)
            self._oracle_idx += 1
            # _decoder_obs_parts expects a numpy token array (torch.from_numpy);
            # oracle tokens are already a device tensor -> pass parts directly
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
    cfg.episode_length_s = n_rows / 50.0 + 30.0
    cfg.router_model_dir = cli.router_model_dir
    env = BonesReplayEnv(cfg)
    env._oracle_tokens = torch.from_numpy(tokens).to(env.device)

    results = {}
    for seed in seeds:
        jitter_and_reset(env, seed)  # D034 initial standing pose (SONIC default + jitter)
        env._oracle_idx = 0
        xy0 = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy()
        traj = [xy0]
        h_trace = []
        h_min = float("inf")
        h_end = None
        fall_step = None
        steps_done = 0
        q_err_ref, q_err_pd = [], []
        q_act_last = None
        quat_last = None
        for t in range(n_rows):
            action = torch.zeros(
                env.num_envs, env.cfg.action_space, dtype=torch.float32, device=env.device
            )
            obs, reward, term, trunc, _ = env.step(action)
            steps_done = t + 1
            done = bool(term[0]) or bool(trunc[0])
            # NOTE: DirectRLEnv auto-resets a terminated env INSIDE step(), so
            # on the done step the returned state is already the reset standing
            # pose -- exclude it from trajectory/height/tracking metrics.
            if not done:
                h = float(env.robot.data.root_pos_w[0, 2].item())
                h_trace.append(round(h, 3))
                h_min = min(h_min, h)
                h_end = h
                traj.append(env.robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy())
                q_des_s = env._q_des[0].detach().cpu().numpy()
                q_act_s = env.robot.data.joint_pos[0, env._body_idx].detach().cpu().numpy()
                q_act_last = q_act_s.copy()
                quat_last = env.robot.data.root_quat_w[0].detach().cpu().numpy().copy()
                q_err_pd.append(float(np.abs(q_act_s - q_des_s).mean()))
                q_err_ref.append(float(np.abs(q_act_s - jp_ref_isaac[t]).mean()))
            if bool(term[0]):
                fall_step = t
                break
            if bool(trunc[0]):
                break
        traj = np.asarray(traj)  # (recorded_frames+1, 2)
        seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        path_len = float(seg.sum())
        disp_vec = traj[-1] - traj[0]
        dur = steps_done / 50.0
        r = {
            "steps": steps_done,
            "completed": fall_step is None and steps_done >= n_rows,
            "fall_step": fall_step,
            "h_min": round(h_min, 3),
            "h_end": round(h_end, 3) if h_end is not None else None,
            "path_len_m": round(path_len, 2),
            "disp_norm_m": round(float(np.linalg.norm(disp_vec)), 2),
            "disp_x_m": round(float(disp_vec[0]), 2),
            "mean_speed_mps": round(path_len / dur, 3),
            "vx_x_mps": round(float(disp_vec[0]) / dur, 3),
            "duration_s": round(dur, 2),
            "q_track_mae_vs_ref_rad": round(float(np.mean(q_err_ref)), 4),
            "q_track_mae_pd_rad": round(float(np.mean(q_err_pd)), 4),
            "q_act_last_isaac": [round(float(x), 3) for x in q_act_last] if q_act_last is not None else None,
            "root_quat_last_wxyz": [round(float(x), 3) for x in quat_last] if quat_last is not None else None,
            "h_trace": h_trace,
        }
        results[f"seed{seed}"] = r
        print(
            f"seed{seed} completed={r['completed']} fall_step={fall_step} "
            f"h_min={r['h_min']} h_end={r['h_end']} path_len={r['path_len_m']} m "
            f"disp={r['disp_norm_m']} m mean_speed={r['mean_speed_mps']} m/s "
            f"qMAE_ref={r['q_track_mae_vs_ref_rad']} qMAE_pd={r['q_track_mae_pd_rad']}",
            flush=True,
        )

    n_falls = sum(1 for r in results.values() if r["fall_step"] is not None)
    n_completed = sum(1 for r in results.values() if r["completed"])
    out = {
        "exp": "D037",
        "line": "DS plan-B B3 rehearsal: offline-encoded BONES tokens -> Isaac closed loop",
        "note": ("rehearsal, not a gate: validates the offline-encode -> Isaac-execute "
                 "chain and collects first numbers; B3 gate needs >=10 segments per class"),
        "tokens_npy": cli.tokens_npy,
        "n_rows": int(n_rows),
        "lattice_violations": lattice_viol,
        "reference": ref_info,
        "seeds": results,
        "n_falls": n_falls,
        "n_completed": n_completed,
        "all_completed": n_completed == len(seeds),
    }
    with open(cli.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print("saved", cli.out)

    # Isaac sometimes hangs on interpreter exit (server gotcha). On this box
    # (D037, 2026-09-04) even sim_app.close() itself hung after the JSON was
    # written -- bound close() to a daemon thread with a timeout and hard-exit
    # regardless (results are already on disk at this point).
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
