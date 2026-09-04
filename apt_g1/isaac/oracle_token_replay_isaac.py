"""Phase 0 Isaac oracle-token replay calibration (DS_GAIT_MANIFOLD_PLAN §2).

Before any collection/training, verify the Isaac substrate can EXECUTE the
official-loop RUN gait. Replays the token stream recorded from the official
deploy loop (drive_run_probe -> policy_input.csv, D033) inside AptFlatG1Env
with the policy/VAE bypassed: recorded token -> frozen SonicTorchDecoder ->
q_des (D002 protocol, Isaac version). The env keeps its own closed-loop
10-frame decoder history, so this tests the Isaac execution stack (PD,
control frequency, decoder consumption path) -- not open-loop playback.

Gate (plan §2): mean realized vx over seeds / official-loop vx >= 0.9 => PASS.
< 0.9 => align execution params (<= 3 iterations); still < 0.9 => G3 speed
target downgrades to the measured Isaac ceiling (recorded, non-blocking).

Run on lab-ts (realization ~2 min Isaac startup + seeds x 60 s):
  cd /home/cvgluser/ros2_data && nohup bash /tmp/run_apt_isaac.sh \
    /home/cvgluser/ros2_data/apt_g1/isaac/oracle_token_replay_isaac.py \
    --out /home/cvgluser/ros2_data/apt_g1/outputs/ds_phase0/oracle_replay_isaac.json \
    > /home/cvgluser/ros2_data/apt_g1/outputs/ds_phase0/replay.log 2>&1 \
    < /dev/null & disown; echo OK
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def build_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--tokens-csv",
        default="/tmp/ds_smoke/policy_input.csv",
        help="official-loop recording; col 0-63 = 64-d token, 50 Hz rows",
    )
    ap.add_argument(
        "--motion-csv",
        default="/tmp/ds_smoke/target_motion.csv",
        help="official-loop recording; col 0 = root x (deploy frame)",
    )
    ap.add_argument("--run-rows", type=int, default=3000, help="60 s @ 50 Hz")
    ap.add_argument(
        "--official-vx",
        type=float,
        default=1.033,
        help="official-loop REALIZED vx for the gate ratio (D033 probe: "
        "62 m / 60 s RUN on the WBC sim loop; target_motion.csv is the "
        "planner reference ~2.1 m/s, not realized)",
    )
    ap.add_argument(
        "--min-run-vx",
        type=float,
        default=0.7,
        help="1 s-window displacement threshold that marks the RUN onset row",
    )
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument(
        "--router-model-dir",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_final",
    )
    ap.add_argument(
        "--out",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/ds_phase0/oracle_replay_isaac.json",
    )
    ap.add_argument(
        "--save-tokens",
        default="/home/cvgluser/ros2_data/apt_g1/data/ds_phase0/run_tokens.npz",
    )
    return ap


def extract_run_window(cli):
    """Locate the RUN onset in the recording and slice the token window.

    Row i of tokens_csv/motion_csv are the same 50 Hz control tick (deploy
    logs both at the policy rate; row counts must match). NOTE (D033):
    target_motion col 0 is the PLANNER reference root x (~2.1 m/s steady for
    RUN), NOT the realized WBC motion -- it is used only to find the motion
    onset; the official-loop REALIZED vx is the D033 probe measurement
    (--official-vx, 62 m / 60 s). RUN onset = first row whose trailing 1 s
    reference displacement clears --min-run-vx (idle/stand rows sit at ~0).
    """
    # deploy csv rows end with a trailing comma (995th field empty) ->
    # genfromtxt + filling_values tolerates it; token cols are 0-63 either way
    tokens_all = np.genfromtxt(
        cli.tokens_csv, delimiter=",", dtype=np.float32, filling_values=0.0
    )
    x = np.genfromtxt(
        cli.motion_csv, delimiter=",", dtype=np.float32, filling_values=0.0,
        usecols=(0,),
    )
    n = min(len(x), len(tokens_all))
    if len(x) != len(tokens_all):
        print(f"[extract] WARN row mismatch tokens={len(tokens_all)} motion={len(x)} -> {n}")
    tokens_all, x = tokens_all[:n], x[:n]
    vx1 = x[50:] - x[:-50]  # reference m per 1 s trailing window
    run_start = None
    for i in range(len(vx1)):
        if vx1[i] > cli.min_run_vx:
            run_start = i
            break
    if run_start is None:
        raise SystemExit(
            f"[extract] FAIL: no RUN onset found in {cli.tokens_csv} "
            f"(rows={n}, max 1 s disp={vx1.max():.2f} m)"
        )
    end = min(run_start + cli.run_rows, n)
    if end - run_start < 2000:
        raise SystemExit(
            f"[extract] FAIL: motion segment too short from onset {run_start} "
            f"(have {end - run_start} rows, need >= 2000)"
        )
    tokens = tokens_all[run_start:end, :64]
    lat = tokens * 16.0
    lattice_viol = int(np.sum(np.abs(lat - np.round(lat)) > 0.05))
    planner_ref_disp = float(x[end - 1] - x[run_start])
    planner_ref_vx = planner_ref_disp / ((end - run_start) / 50.0)
    window_vx1 = vx1[run_start : end - 50]
    print(
        f"[extract] rows={n} onset={run_start} window=[{run_start},{end}) "
        f"planner_ref_disp={planner_ref_disp:.1f} m ref_vx={planner_ref_vx:.3f} m/s "
        f"(NOT realized; official realized vx={cli.official_vx:.3f} from D033 probe) "
        f"window ref-vx mean={window_vx1.mean():.3f} min={window_vx1.min():.3f} "
        f"lattice_viol={lattice_viol}",
        flush=True,
    )
    return tokens, run_start, end - run_start, planner_ref_vx, lattice_viol


def main():
    from isaaclab.app import AppLauncher

    ap = build_args()
    AppLauncher.add_app_launcher_args(ap)
    # server has no display: viewport/hydra init segfaults (repo gotcha),
    # so headless must be the default; kit/experience flags still passable
    ap.set_defaults(headless=True)
    cli = ap.parse_args()
    seeds = [int(s) for s in cli.seeds.split(",")]

    # fail fast on extraction before paying the Isaac startup
    tokens, run_start, n_rows, planner_ref_vx, lattice_viol = extract_run_window(cli)
    Path(cli.save_tokens).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cli.save_tokens,
        tokens=tokens,
        run_start=run_start,
        n_rows=n_rows,
        official_vx=cli.official_vx,
        planner_ref_vx=planner_ref_vx,
        tokens_csv=cli.tokens_csv,
    )
    print(f"[extract] tokens saved -> {cli.save_tokens}", flush=True)

    app_launcher = AppLauncher(cli)
    sim_app = app_launcher.app

    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.eval_apt_isaac import jitter_and_reset

    class OracleReplayEnv(AptFlatG1Env):
        """Token oracle: _compute_q_des bypasses policy/VAE/router entirely and
        decodes the recorded official-loop token stream (env-owned history).
        Deliberately a subclass -- canonical apt_flat_env.py stays untouched."""

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
    env = OracleReplayEnv(cfg)
    env._oracle_tokens = torch.from_numpy(tokens).to(env.device)

    # D035 slip audit: G1 ankle-roll links terminate at the feet; contact is
    # proxied by foot-frame world height (plane terrain at z=0), calibrated to
    # the per-run minimum. Honest gait => contact-phase foot speed ~ 0; skating
    # => fast feet while "in contact". Pre-registered verdict: worse-foot
    # median contact-phase horizontal speed < 0.15 m/s => honest.
    foot_ids = []
    all_bodies = env.robot.body_names
    for pat in ("left_ankle_roll.*", "right_ankle_roll.*"):
        try:
            ids, names = env.robot.find_bodies(pat)
        except ValueError:
            raise SystemExit(
                f"[slip] FAIL: pattern {pat!r} matched none of {all_bodies}"
            )
        foot_ids.append(int(ids[0]))
        print(f"[slip] foot body: {names[0]}", flush=True)

    results = {}
    for seed in seeds:
        jitter_and_reset(env, seed)
        env._oracle_idx = 0
        xy0 = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy()
        h_min = float("inf")
        fall_step = None
        steps_done = 0
        foot_z, foot_v, q_err = [], [], []
        for t in range(n_rows):
            action = torch.zeros(
                env.num_envs, env.cfg.action_space, dtype=torch.float32, device=env.device
            )
            obs, reward, term, trunc, _ = env.step(action)
            steps_done = t + 1
            h = float(env.robot.data.root_pos_w[0, 2].item())
            h_min = min(h_min, h)
            bp = env.robot.data.body_pos_w[0, foot_ids, 2].detach().cpu().numpy()
            bv = env.robot.data.body_lin_vel_w[0, foot_ids, :2].detach().cpu().numpy()
            foot_z.append(bp.copy())
            foot_v.append(np.linalg.norm(bv, axis=1))
            q_des_s = env._q_des[0].detach().cpu().numpy()
            q_act_s = env.robot.data.joint_pos[0, env._body_idx].detach().cpu().numpy()
            q_err.append(float(np.abs(q_act_s - q_des_s).mean()))
            if bool(term[0]):
                fall_step = t
                break
            if bool(trunc[0]):
                break
        xy1 = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy()
        disp_x = float(xy1[0] - xy0[0])
        disp_norm = float(np.linalg.norm(xy1 - xy0))
        fz = np.asarray(foot_z)          # (T, 2)
        fv = np.asarray(foot_v)          # (T, 2) horizontal |v|
        min_z = float(fz.min())
        stance = fz < min_z + 0.02       # (T, 2) contact proxy
        contact_speed = np.where(stance, fv, np.nan)
        med = np.nanmedian(contact_speed, axis=0)
        p90 = np.nanpercentile(contact_speed, 90, axis=0)
        slip_frac = np.nanmean(contact_speed > 0.2, axis=0)
        onsets = (stance[1:] & ~stance[:-1]).sum(axis=0)
        worse = int(np.argmax(med))
        r = {
            "steps": steps_done,
            "completed": fall_step is None and steps_done >= n_rows,
            "fall_step": fall_step,
            "h_min": round(h_min, 3),
            "disp_x": round(disp_x, 2),
            "disp_norm": round(disp_norm, 2),
            "vx_x": round(disp_x / (n_rows / 50.0), 3),
            "foot_min_z": round(min_z, 4),
            "stance_frac": [round(float(v), 3) for v in stance.mean(axis=0)],
            "contact_speed_median": round(float(med[worse]), 4),
            "contact_speed_p90": round(float(p90[worse]), 4),
            "slip_frac_gt02": round(float(slip_frac[worse]), 4),
            "steps_per_sec": round(float(onsets[worse]) / (n_rows / 50.0), 2),
            "q_track_mae_rad": round(float(np.mean(q_err)), 4),
            "slip_verdict": (
                "HONEST" if med[worse] < 0.15 else
                "PARTIAL_SLIP" if med[worse] < 0.4 else "SKATING"
            ),
            "slip_verdict_rule": "worse-foot median contact-phase speed: <0.15 honest / <0.4 partial / >=0.4 skating",
        }
        results[f"seed{seed}"] = r
        print(
            f"seed{seed} completed={r['completed']} fall_step={fall_step} "
            f"h_min={r['h_min']} disp_x={r['disp_x']} vx={r['vx_x']} "
            f"slip={r['slip_verdict']} (med={r['contact_speed_median']} "
            f"p90={r['contact_speed_p90']} stance={r['stance_frac']} "
            f"step/s={r['steps_per_sec']} qMAE={r['q_track_mae_rad']})",
            flush=True,
        )

    mean_vx = float(np.mean([r["vx_x"] for r in results.values()]))
    ratio = mean_vx / cli.official_vx if cli.official_vx > 1e-6 else 0.0
    n_falls = sum(1 for r in results.values() if r["fall_step"] is not None)
    out = {
        "phase": "DS_GAIT_MANIFOLD_PLAN Phase 0 (Isaac oracle-token replay)",
        "slip_audit": "D035: contact-phase horizontal foot speed per seed (HONEST <0.15 < PARTIAL <0.4 <= SKATING m/s)",
        "tokens_csv": cli.tokens_csv,
        "run_start_row": int(run_start),
        "run_rows": int(n_rows),
        "lattice_violations": lattice_viol,
        "planner_ref_vx": round(float(planner_ref_vx), 4),
        "official_vx": cli.official_vx,
        "official_vx_source": "D033 base_sim probe: 62 m / 60 s RUN on official WBC sim loop",
        "seeds": results,
        "n_falls": n_falls,
        "mean_vx_x": round(mean_vx, 4),
        "realization_ratio": round(ratio, 4),
        "gate": "PASS" if ratio >= 0.9 else "FAIL",
        "gate_rule": "mean realized vx / official-loop realized vx >= 0.9 (plan §2)",
    }
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    with open(cli.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print("saved", cli.out)

    # Isaac sometimes hangs on interpreter exit (server gotcha). On this box
    # (D036/D037 session, 2026-09-04) even sim_app.close() itself hung after
    # the JSON was written -- bound close() to a daemon thread with a timeout
    # and hard-exit regardless (results are already on disk at this point).
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
