"""D048r R1/R2：快段 token 闭环回放——decoder 本体上限 / VAE 重建代价三角测量。

DS_CONTINUOUS_EXECUTION_PLAN §5i 预注册。R1（本体层）= pick_fast_segments_d048r.py
挑出的 top 快段原始 token 直喂冻结 SONIC decoder 闭环回放（20s×3 seed）；R2
（接口-被动层）= 同段 token 过 VAE encode→decode 重建后同回放。R1−R2 = VAE
重建的执行代价；decoder 本体上限 = 各段闭环段均 vx 的最大值。

与 oracle_token_replay_isaac.py 的骨架关系（静态 diff 说明）：
  【照抄】AptFlatG1Env 子类旁路 pattern（_compute_q_des 直喂 token→冻结
    decoder，env 自持 10 帧闭环 history；_reset_idx 复位帧指针）、
    jitter_and_reset 站姿起步、零 aux 动作、AppLauncher headless 默认 +
    Isaac 前 fail-fast、daemon 线程 close + os._exit(0) 防挂死、
    done 步不计指标（DirectRLEnv done 步 auto-reset 污染，b3p 口径）。
  【改动】① 输入 = d048r_fast_segments.json 段清单（或单 --npz），不再是
    官方回路 CSV；② 步预算固定 --dur-s 20s（token 末行夹紧 = hold 相，
    D034 同款）；③ 指标改双口径：mean_vx_fwd（净前向位移/时长，初始航向
    系投影）+ mean_speed_path（路径长/时长，b3p mean_speed_mps 同式）+
    realized_ratio（闭环 vs 参考 v_med，D048n 同口径段中位速）；播放相同窗
    比值复用 playback_window.playback_same_window_ratio（D046b-R3 v3 口径）；
    survived = 全程未摔且 h_min≥0.40；④ R2 增加 VAE 重建路径（token_window_vae
    DirSpeedPhaseTokenVAE，与 env latent 分支同源加载 strict=False；窗口
    build_windows、phase=pca.npz pmean/V2 投影〔train_token_vae_e39.
    walk_phase_rate 投影法、不自拟合〕、条件=段帧级 vb/db 标签经
    build_d048n.align_segments 确定性对齐 + 逐帧 md5 抽查，对不齐显式失败
    ——D046b-R3 纪律）；⑤ 聚合 = per 段跨 seed 中位 + per token_source 汇总。

Run on lab-ts (Isaac wrapper, cwd=GR00T-WholeBodyControl):
  nohup bash /tmp/run_apt_isaac.sh \
    /home/cvgluser/ros2_data/apt_g1/isaac/replay_token_speed_d048r.py \
    --token-source orig --tag r1_orig \
    > /home/cvgluser/ros2_data/apt_g1/outputs/d048r/r1.log 2>&1 < /dev/null & disown
  # R2: --token-source vae_recon --latent-vae-path <vae.pt> --tag r2_<vaeTag>
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

try:                      # 服务器执行根平铺 import / 仓库根包 import 双兼容
    from build_d048n_vb_speed_labels import (
        DEFAULT_INPUTS,
        DEFAULT_NPZ_DIRS,
        align_segments,
        frame_speed_bwd,
        md5_frame,
    )
    from playback_window import playback_same_window_ratio
    from train_token_vae_e39 import build_windows
except ImportError:
    from apt_g1.build_d048n_vb_speed_labels import (
        DEFAULT_INPUTS,
        DEFAULT_NPZ_DIRS,
        align_segments,
        frame_speed_bwd,
        md5_frame,
    )
    from apt_g1.isaac.playback_window import playback_same_window_ratio
    from apt_g1.train_token_vae_e39 import build_windows

HOME = os.path.expanduser("~")
DEFAULT_SEGMENTS_JSON = f"{HOME}/ros2_data/apt_g1/data/d048r/d048r_fast_segments.json"
DEFAULT_OUT_BASE = f"{HOME}/ros2_data/apt_g1/outputs/d048r"
DEFAULT_ROUTER_DIR = f"{HOME}/ros2_data/apt_g1/outputs/distill_final"


def build_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--segments-json", default=DEFAULT_SEGMENTS_JSON,
                    help="pick_fast_segments_d048r.py 产物（segments[].stem/"
                         "npz_path/v_med）")
    ap.add_argument("--npz", default="",
                    help="单段模式：直接给 npz 路径（覆盖 --segments-json）")
    ap.add_argument("--token-source", choices=["orig", "vae_recon"],
                    default="orig", help="orig=原始 token（R1）；vae_recon=VAE "
                                         "encode→decode 重建（R2）")
    ap.add_argument("--latent-vae-path", default="",
                    help="R2 必填：vae.pt（token_window_vae 类加载，env 同源 "
                         "strict=False）")
    ap.add_argument("--n-vbins", type=int, default=3)
    ap.add_argument("--n-dbins", type=int, default=8)
    ap.add_argument("--vae-inputs-dir", default=DEFAULT_INPUTS,
                    help="R2 条件标签目录（vb/db/segment_bounds/build_meta.json/"
                         "token.npy）")
    ap.add_argument("--npz-dir", nargs="+", default=DEFAULT_NPZ_DIRS,
                    help="R2 对齐用源 npz 目录（build_d048n 同款解析）")
    ap.add_argument("--vb-file", default="vb_speed.npy",
                    help="R2 vb 标签文件名（D048n/D048p 产物；旧口径指定旧文件名）")
    ap.add_argument("--db-file", default="angle_bin.npy",
                    help="R2 db 标签文件名（v2 builder 产物名）")
    ap.add_argument("--pca-npz", default="",
                    help="phase 投影 pmean/V2（默认 = vae.pt 同目录 pca.npz，env "
                         "latent 分支同源）")
    ap.add_argument("--z-mode", choices=["mu", "sample"], default="mu",
                    help="R2 encode 取值：mu=确定性（eval 探针同款，默认）；"
                         "sample=reparameterize（配 --z-seed）")
    ap.add_argument("--z-seed", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dur-s", type=float, default=20.0,
                    help="每 run 步预算秒（§5i 预注册 20s；50Hz）")
    ap.add_argument("--survive-hmin", type=float, default=0.40,
                    help="survived 高度门槛（b3p 口径）")
    ap.add_argument("--router-model-dir", default=DEFAULT_ROUTER_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT_BASE, help="输出根目录")
    ap.add_argument("--tag", default="",
                    help="输出子目录 = d048r_<tag>（默认 r1_orig / r2_vae）")
    return ap


# ----------------------------------------------------------------- 输入装载


def load_segment_list(cli):
    """段清单 → [{stem, npz_path, v_med}]，npz 内 tokens/trans_m 一并读入。"""
    entries = []
    if cli.npz:
        entries.append({"stem": Path(cli.npz).stem, "npz_path": cli.npz,
                        "v_med": None})
    else:
        with open(cli.segments_json, encoding="utf-8") as f:
            data = json.load(f)
        entries = [
            {"stem": s["stem"], "npz_path": s["npz_path"], "v_med": s["v_med"]}
            for s in data["segments"]
        ]
    if not entries:
        raise SystemExit("[load] FAIL: 段清单为空")
    segs = []
    for e in entries:
        if not os.path.isfile(e["npz_path"]):
            raise SystemExit(f"[load] FAIL: npz 缺失 {e['npz_path']}")
        z = np.load(e["npz_path"], allow_pickle=False)
        trans = z["trans_m"].astype(np.float64)
        tokens = z["tokens"].astype(np.float32)
        v_med = e["v_med"]
        if v_med is None:       # 单段模式：现算（D048n 同口径段中位）
            v_med = float(np.median(frame_speed_bwd(trans, 50.0, "xy")))
        segs.append({"stem": e["stem"], "tokens": tokens, "trans_m": trans,
                     "v_med": float(v_med), "frames": len(tokens)})
        print(f"[load] {e['stem']}: {len(tokens)} frames v_med={v_med:.3f} m/s",
              flush=True)
    return segs


def resolve_label_slices(cli, stems, frames_by_stem):
    """R2 条件标签对齐（Isaac 启动前 fail-fast）。

    走 build_d048n.align_segments（确定性重放三重 checksum 内建）；段 → 组装
    帧序的 kind=orig 区间（原始段每 stem 恰一个；kind=copy 副本段不算）。
    任何 stem 缺席/歧义/帧数不符/逐帧 md5 抽查不过 → SystemExit（宁可退出
    不要错标，D046b-R3 纪律）。
    """
    (bounds, _src_idx, _orig_stems, seg_records, provenance, trans_cache,
     _npz_paths, _build_meta) = align_segments(cli.vae_inputs_dir, cli.npz_dir)
    token_all = np.load(os.path.join(cli.vae_inputs_dir, "token.npy")).astype(
        np.float32)
    vb_all = np.load(os.path.join(cli.vae_inputs_dir, cli.vb_file)).astype(np.int64)
    db_all = np.load(os.path.join(cli.vae_inputs_dir, cli.db_file)).astype(np.int64)
    n = len(token_all)
    if not (len(vb_all) == len(db_all) == n):
        raise SystemExit(f"[align] FAIL: vb/db 长度 ({len(vb_all)}/{len(db_all)}) "
                         f"!= token N {n}")
    if int(bounds[-1, 1]) != n:
        raise SystemExit(f"[align] FAIL: bounds 总帧 {int(bounds[-1, 1])} != N {n}")

    slices, problems = {}, []
    for stem in stems:
        cand = [j for j, r in enumerate(seg_records)
                if r["stem"] == stem and r["kind"] == "orig"]
        if len(cand) != 1:
            problems.append(f"{stem}: 组装语料中原始段命中 {len(cand)} 个（0=不在"
                            "该 inputs 目录（如 dev/test 划分段），>1=歧义）")
            continue
        j = cand[0]
        s0, s1 = int(bounds[j, 0]), int(bounds[j, 1])
        if frames_by_stem[stem] != s1 - s0:
            problems.append(f"{stem}: npz 帧数 {frames_by_stem[stem]} != 组装区间 "
                            f"{s1 - s0}")
            continue
        toks_src = trans_cache[stem][1]
        for k in {0, (s1 - s0) // 2, s1 - s0 - 1}:   # 首/中/末帧 md5 抽查
            if md5_frame(token_all[s0 + k]) != md5_frame(toks_src[k]):
                problems.append(f"{stem}: 组装帧 {s0 + k} token md5 与源 npz 不一致")
                break
        slices[stem] = (s0, s1)
    if problems:
        raise SystemExit("[align] FAIL 段标签对齐失败（显式退出，不静默错标）:\n  "
                         + "\n  ".join(problems))
    print(f"[align] {len(slices)} 段 vb/db 标签对齐通过（{provenance}）；"
          f"vb_file={cli.vb_file} db_file={cli.db_file}", flush=True)
    return slices


def load_vae(cli, device):
    """D048r hotfix2：R2 重建须 encode→decode 全链，改用训练侧完整模型类
    （token_window_vae 运行时镜像是 decode-only，strict=False 会丢 encoder
    权重——首跑 AttributeError: no attribute 'encode'）。维度从 meta.json 读，
    strict=True 全量加载防权重错位。"""
    from train_token_vae_e39 import DirSpeedPhaseTokenVAE as FullVAE
    import json as _json
    meta_p = Path(cli.latent_vae_path).parent / "meta.json"
    dims = _json.load(open(meta_p))
    vae = FullVAE(token_dim=dims["token_dim"], window=dims["window"],
                  latent_dim=dims["latent_dim"], hidden_dim=dims["hidden"],
                  phase_dim=dims["phase_dim"], n_vbins=cli.n_vbins,
                  n_dbins=cli.n_dbins).to(device)
    vae.load_state_dict(
        torch.load(cli.latent_vae_path, map_location=device),
        strict=True,
    )
    vae.eval()
    pca_path = cli.pca_npz or str(Path(cli.latent_vae_path).parent / "pca.npz")
    pca = np.load(pca_path)
    print(f"[vae] {cli.latent_vae_path} loaded; pca={pca_path}", flush=True)
    return vae, pca


def reconstruct_segment_tokens(vae, pca, tok, vb_seg, db_seg, device,
                               z_mode: str, z_seed: int) -> np.ndarray:
    """段 token → VAE 重建 token（确定性 mu 默认）。

    窗口 = build_windows（向后 10 帧，段首零填充，train 同一实现）；phase =
    pca.npz pmean/V2 投影（walk_phase_rate 投影法：phi=atan2(proj1,proj0)，
    不重拟合 PCA）；条件 = 段帧级 vb/db（对齐见 resolve_label_slices）。
    """
    pmean, V2 = pca["pmean"].astype(np.float32), pca["V2"].astype(np.float32)
    proj = (tok.astype(np.float32) - pmean) @ V2
    phi = np.arctan2(proj[:, 1], proj[:, 0]).astype(np.float32)
    sc = np.stack([np.sin(phi), np.cos(phi)], axis=1).astype(np.float32)
    x = torch.from_numpy(build_windows(tok, vae.window)).to(device)
    sc_t = torch.from_numpy(sc).to(device)
    vb_t = torch.from_numpy(vb_seg).long().to(device)
    db_t = torch.from_numpy(db_seg).long().to(device)
    with torch.no_grad():
        mu, lv = vae.encode(x)
        z = mu if z_mode == "mu" else vae.reparameterize(mu, lv)
        tok_rec = vae.decode(z, sc_t, vb_t, db_t)
    return tok_rec.cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------- 回放环境


def main():
    from isaaclab.app import AppLauncher

    ap = build_args()
    AppLauncher.add_app_launcher_args(ap)
    # server has no display: viewport/hydra init segfaults (repo gotcha)
    ap.set_defaults(headless=True)
    cli = ap.parse_args()
    seeds = [int(s) for s in cli.seeds.split(",")]
    steps_budget = int(round(cli.dur_s * 50.0))
    tag = cli.tag or ("r1_orig" if cli.token_source == "orig" else "r2_vae")
    out_dir = os.path.join(cli.out, f"d048r_{tag}")

    if cli.token_source == "vae_recon" and not cli.latent_vae_path:
        raise SystemExit("[args] FAIL: --token-source vae_recon 需要 --latent-vae-path")

    # Isaac 前 fail-fast：段装载 + R2 标签对齐（numpy-only）
    segs = load_segment_list(cli)
    label_slices = None
    if cli.token_source == "vae_recon":
        label_slices = resolve_label_slices(
            cli, [s["stem"] for s in segs], {s["stem"]: s["frames"] for s in segs})

    app_launcher = AppLauncher(cli)
    sim_app = app_launcher.app

    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.eval_apt_isaac import jitter_and_reset

    class SpeedReplayEnv(AptFlatG1Env):
        """Token oracle（D034/D037 同构，骨架抄 oracle_token_replay_isaac.py）：
        _compute_q_des 旁路 policy/VAE/router，逐帧 token 直喂冻结 decoder，
        env 自持 10 帧闭环 history。canonical apt_flat_env.py 零改动。"""

        _oracle_tokens: torch.Tensor | None = None
        _oracle_idx: int = 0

        def _compute_q_des(self, phase, aux, res=None):
            i = min(self._oracle_idx, self._oracle_tokens.shape[0] - 1)
            tokens = self._oracle_tokens[i].unsqueeze(0).expand(self.num_envs, -1)
            self._oracle_idx += 1
            # _decoder_obs_parts 期望 numpy token；此处已是 device tensor ->
            # 六参直传（oracle 同款零拷贝路径）
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
    cfg.episode_length_s = steps_budget / 50.0 + 30.0   # 预算内不触发 trunc
    cfg.router_model_dir = cli.router_model_dir
    env = SpeedReplayEnv(cfg)

    # R2：VAE 装载 + 逐段重建（env.device 已知；重建失败的段显式退出）
    if cli.token_source == "vae_recon":
        vae, pca = load_vae(cli, env.device)
        if cli.z_mode == "sample":
            torch.manual_seed(cli.z_seed)
            np.random.seed(cli.z_seed)
        for seg in segs:
            s0, s1 = label_slices[seg["stem"]]
            vb_seg = np.load(os.path.join(cli.vae_inputs_dir, cli.vb_file),
                             ).astype(np.int64)[s0:s1]
            db_seg = np.load(os.path.join(cli.vae_inputs_dir, cli.db_file),
                             ).astype(np.int64)[s0:s1]
            seg["tokens_recon"] = reconstruct_segment_tokens(
                vae, pca, seg["tokens"], vb_seg, db_seg, env.device,
                cli.z_mode, cli.z_seed)
            rmse = float(np.sqrt(np.mean(
                (seg["tokens_recon"] - seg["tokens"]) ** 2)))
            seg["recon_token_rmse"] = round(rmse, 5)
            print(f"[recon] {seg['stem']}: token RMSE={rmse:.4f}", flush=True)

    def _yaw_of(q):
        # 标准 quat→yaw（Isaac root_quat_w 为 w-first；eval_apt_isaac 同式）
        w, x, y, z = q[0], q[1], q[2], q[3]
        return float(
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).item()
        )

    per_run, per_seg = {}, {}
    for seg in segs:
        tokens = (seg["tokens_recon"] if cli.token_source == "vae_recon"
                  else seg["tokens"])
        env._oracle_tokens = torch.from_numpy(tokens).to(env.device)
        runs = {}
        for seed in seeds:
            jitter_and_reset(env, seed)
            env._oracle_idx = 0
            xy0 = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy()
            yaw0 = _yaw_of(env.robot.data.root_quat_w[0].detach())
            f0 = np.array([math.cos(yaw0), math.sin(yaw0)])  # 初始航向系前向
            traj_pb = [xy0]          # 播放相轨迹（首点=重置位姿，同窗比值基点）
            h_min, h_end, fall_step, steps_done = float("inf"), None, None, 0
            traj = [xy0]             # 全程 live 轨迹（done 步不计，b3p 口径）
            for t in range(steps_budget):
                action = torch.zeros(
                    env.num_envs, env.cfg.action_space,
                    dtype=torch.float32, device=env.device)
                obs, reward, term, trunc, _ = env.step(action)
                steps_done = t + 1
                done = bool(term[0]) or bool(trunc[0])
                if not done:
                    h = float(env.robot.data.root_pos_w[0, 2].item())
                    h_min = min(h_min, h)
                    h_end = h
                    xy_now = env.robot.data.root_pos_w[0, :2].detach() \
                        .cpu().numpy().copy()
                    traj.append(xy_now)
                    if t < seg["frames"]:
                        traj_pb.append(xy_now)
                if bool(term[0]):
                    fall_step = t
                    break
                if bool(trunc[0]):
                    break
            traj_a = np.asarray(traj)
            path_len = float(np.linalg.norm(np.diff(traj_a, axis=0),
                                            axis=1).sum())
            disp_vec = traj_a[-1] - traj_a[0]
            dur = steps_done / 50.0
            disp_fwd = float(disp_vec @ f0)
            mean_vx_fwd = disp_fwd / dur if dur > 0 else None
            mean_speed_path = path_len / dur if dur > 0 else None
            realized_ratio = (mean_speed_path / seg["v_med"]
                              if seg["v_med"] > 1e-6 else None)
            realized_ratio_fwd = (mean_vx_fwd / seg["v_med"]
                                  if seg["v_med"] > 1e-6 else None)
            traj_pb_a = np.asarray(traj_pb)
            first_step_m = (float(np.linalg.norm(traj_pb_a[1] - traj_pb_a[0]))
                            if len(traj_pb_a) > 1 else 0.0)
            pw = playback_same_window_ratio(traj_pb_a, seg["trans_m"][:, :2])
            hold_disp = (float(np.linalg.norm(traj_a[-1] - traj_pb_a[-1]))
                         if steps_done > seg["frames"] else None)
            survived = fall_step is None and h_min >= cli.survive_hmin
            key = f"{seg['stem']}__seed{seed}"
            per_run[key] = {
                "stem": seg["stem"], "seed": seed,
                "token_source": cli.token_source,
                "n_rows_tokens": seg["frames"], "steps_budget": steps_budget,
                "steps": steps_done,
                "completed": fall_step is None and steps_done >= steps_budget,
                "fall_step": fall_step,
                "survived": survived,
                "survive_rule": f"未摔且 h_min>={cli.survive_hmin}",
                "h_min": round(h_min, 3),
                "h_end": round(h_end, 3) if h_end is not None else None,
                "disp_fwd_m": round(disp_fwd, 2),
                "dur_s": round(dur, 2),
                "mean_vx_fwd": round(mean_vx_fwd, 4) if mean_vx_fwd is not None else None,
                "path_len_m": round(path_len, 2),
                "mean_speed_path": round(mean_speed_path, 4) if mean_speed_path is not None else None,
                "v_med_ref": round(seg["v_med"], 4),
                "realized_ratio": round(realized_ratio, 4) if realized_ratio is not None else None,
                "realized_ratio_fwd": round(realized_ratio_fwd, 4) if realized_ratio_fwd is not None else None,
                "playback_first_step_m": round(first_step_m, 2),
                "playback_path_m": round(pw["robot_path_m"], 2),
                "playback_ref_path_m": round(pw["ref_path_m"], 2),
                "playback_path_ratio": (round(pw["ratio"], 4)
                                        if pw["ratio"] is not None else None),
                "playback_t_used": int(pw["playback_t_used"]),
                "hold_disp_m": round(hold_disp, 2) if hold_disp is not None else None,
                **({"recon_token_rmse": seg.get("recon_token_rmse")}
                   if cli.token_source == "vae_recon" else {}),
            }
            runs[seed] = per_run[key]
            r = per_run[key]
            print(f"[run] {seg['stem'][:40]:40s} seed{seed} "
                  f"survived={survived} fall={fall_step} h_min={r['h_min']} "
                  f"vx_fwd={r['mean_vx_fwd']} v_path={r['mean_speed_path']} "
                  f"ratio={r['realized_ratio']} pb_ratio={r['playback_path_ratio']}",
                  flush=True)

        def _med(field):
            vals = [runs[s][field] for s in seeds if runs[s][field] is not None]
            return round(float(np.median(vals)), 4) if vals else None

        per_seg[seg["stem"]] = {
            "v_med_ref": round(seg["v_med"], 4), "frames": seg["frames"],
            "n_seeds": len(seeds),
            "n_survived": sum(1 for r in runs.values() if r["survived"]),
            "n_falls": sum(1 for r in runs.values() if r["fall_step"] is not None),
            "mean_vx_fwd_med": _med("mean_vx_fwd"),
            "mean_speed_path_med": _med("mean_speed_path"),
            "realized_ratio_med": _med("realized_ratio"),
            "realized_ratio_fwd_med": _med("realized_ratio_fwd"),
            "h_min_min": round(min(runs[s]["h_min"] for s in seeds), 3),
            **({"recon_token_rmse": seg.get("recon_token_rmse")}
               if cli.token_source == "vae_recon" else {}),
        }
        print(f"[seg] {seg['stem']}: med vx_fwd={per_seg[seg['stem']]['mean_vx_fwd_med']} "
              f"survived={per_seg[seg['stem']]['n_survived']}/{len(seeds)}", flush=True)

    # ---- per token_source 汇总
    seg_meds = [v["mean_vx_fwd_med"] for v in per_seg.values()
                if v["mean_vx_fwd_med"] is not None]
    seg_meds_path = [v["mean_speed_path_med"] for v in per_seg.values()
                     if v["mean_speed_path_med"] is not None]
    n_runs = len(per_run)
    summary = {
        "token_source": cli.token_source,
        "n_segments": len(per_seg),
        "n_runs": n_runs,
        "n_survived_runs": sum(1 for r in per_run.values() if r["survived"]),
        "n_falls": sum(1 for r in per_run.values() if r["fall_step"] is not None),
        "survival_rate": round(sum(1 for r in per_run.values()
                                   if r["survived"]) / n_runs, 4) if n_runs else None,
        "ceiling_vx_fwd": round(max(seg_meds), 4) if seg_meds else None,
        "ceiling_vx_fwd_rule": "max over segments of 跨 seed 中位 mean_vx_fwd"
                               "（§5i：decoder 本体上限 = 最大闭环段均 vx）",
        "ceiling_vx_path": round(max(seg_meds_path), 4) if seg_meds_path else None,
        "vx_fwd_med_over_segments": round(float(np.median(seg_meds)), 4)
            if seg_meds else None,
        "realized_ratio_med": (round(float(np.median(
            [r["realized_ratio"] for r in per_run.values()
             if r["realized_ratio"] is not None])), 4)
            if any(r["realized_ratio"] is not None for r in per_run.values())
            else None),
    }

    out = {
        "experiment": "D048r",
        "layer": "R1" if cli.token_source == "orig" else "R2",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": {k: v for k, v in vars(cli).items()},
        "seeds": seeds,
        "steps_budget": steps_budget,
        "label_alignment_provenance": (
            "build_d048n.align_segments + 逐帧 md5 抽查"
            if label_slices else "n/a (token_source=orig)"),
        "summary": summary,
        "per_segment": per_seg,
        "per_run": per_run,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "replay_runs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print("saved", out_path)

    # Isaac 有时在解释器退出时挂死（repo gotcha）：close() 绑 daemon 线程带
    # 超时，结果已落盘后无条件硬退出（oracle/b3p 同款）。
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
