"""D048r R3：z 常向量 CEM 优化——接口-主动层 decoder 上限包络三角测量。

DS_CONTINUOUS_EXECUTION_PLAN §5i 预注册。覆写 env `_compute_q_des`：不走策略，
直接 tokens = VAE.decode(z_const, sin/cos(phase), vb_const, db_const) → 冻结
SONIC decoder → q_des（phase 推进 = env 既有 `_latent_phase` walk clock 固定
步频；db_const 默认 4 = +x 前向档；vb_const 由 --force-vbin）。CEM：--pop 128
（=num_envs，每 env 一个独立 z 候选，z∈ℝ16）、--iters 5、精英 top --elite-frac
0.16；首轮 z~N(0,I) 逐 env 独立采样（种子记录可复现），每轮 20s（1000 步）
rollout，fitness = 净前向位移/20s（初始航向系投影；摔倒 env fitness=-1 并记
fall_step，终末 upright<0.9 也判负——upright=exp(-g_xy²/0.1)，eval 同式）；下一
轮 z~N(mean_elite, (std_elite+0.1²)I)（协方差对角 = std_elite+0.01：std_elite
塌缩到 0 时 std=0.1，即防塌缩 floor）。外层 for vb in --vb-list（默认 0,1,2）
每档独立 CEM。停机规则（预注册）：某档首轮存活率 < --min-first-survival（默认
0.5）→ 停该档，落盘首轮 z 采样分布事实后进入下一档。

CEM 更新核心 = 本文件顶部纯 numpy 函数（torch 延迟到 main() 内 import，本机
无 torch 可 import 本模块单测）：cem_elite_indices / cem_next_distribution /
cem_sample。

Run on lab-ts (Isaac wrapper, cwd=GR00T-WholeBodyControl):
  nohup bash /tmp/run_apt_isaac.sh \
    /home/cvgluser/ros2_data/apt_g1/isaac/z_sweep_cem_d048r.py \
    --latent-vae-path <vae.pt> --tag <vaeTag> \
    > /home/cvgluser/ros2_data/apt_g1/outputs/d048r/r3.log 2>&1 < /dev/null & disown
产物 <out>/d048r_r3_<tag>/{z_sweep_summary.json, z_history.npz}。
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path

import numpy as np

HOME = os.path.expanduser("~")
DEFAULT_OUT_BASE = f"{HOME}/ros2_data/apt_g1/outputs/d048r"
DEFAULT_ROUTER_DIR = f"{HOME}/ros2_data/apt_g1/outputs/distill_final"


# ----------------------------------------------------------------- CEM 核心
# （纯 numpy；本机单测 = tmp/d048r_test/cem_selftest.py，不依赖 torch/Isaac）


def cem_elite_indices(fitness: np.ndarray, elite_frac: float) -> np.ndarray:
    """精英下标：fitness 降序 top round(elite_frac*pop)（≥1；稳定排序确定性）。"""
    fitness = np.asarray(fitness, dtype=np.float64)
    k = max(1, int(round(elite_frac * len(fitness))))
    return np.argsort(-fitness, kind="stable")[:k]


def cem_next_distribution(zs: np.ndarray, fitness: np.ndarray, elite_frac: float,
                          std_floor: float = 0.1):
    """下一轮分布 (mu, var, elite_idx)。

    mu = 精英均值；var = 精英逐维 std + std_floor²（§5i 字面式
    N(mean_elite, (std_elite+0.1²)I)：std_elite→0 时采样 std→std_floor，
    防塌缩）。返回 var（协方差对角），采样方用 cem_sample。
    """
    zs = np.asarray(zs, dtype=np.float64)
    elite_idx = cem_elite_indices(fitness, elite_frac)
    elite = zs[elite_idx]
    mu = elite.mean(axis=0)
    std_elite = elite.std(axis=0)          # ddof=0
    var = std_elite + std_floor ** 2
    return mu, var, elite_idx


def cem_sample(rng: np.random.Generator, mu: np.ndarray, var: np.ndarray,
               n: int) -> np.ndarray:
    """z ~ N(mu, diag(var))，逐候选独立。"""
    return rng.normal(loc=mu, scale=np.sqrt(np.asarray(var)),
                      size=(n, len(mu)))


# ----------------------------------------------------------------- CLI


def build_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--latent-vae-path", required=True,
                    help="vae.pt 必填（token_window_vae DirSpeedPhaseTokenVAE，"
                         "env latent 分支同源加载；pca.npz 取同目录）")
    ap.add_argument("--n-vbins", type=int, default=3)
    ap.add_argument("--n-dbins", type=int, default=8)
    ap.add_argument("--vb-list", default="0,1,2",
                    help="外层 vb 档循环（§5i 预注册 0,1,2，每档独立 CEM）")
    ap.add_argument("--force-dbin", type=int, default=4,
                    help="db 常量档（默认 4 = +x 前向，E35 bin 公式）")
    ap.add_argument("--pop", type=int, default=128,
                    help="候选数 = num_envs（§5i 预注册 128）")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--elite-frac", type=float, default=0.16)
    ap.add_argument("--std-floor", type=float, default=0.1)
    ap.add_argument("--rollout-s", type=float, default=20.0,
                    help="每轮 rollout 秒（§5i 预注册 20s = 1000 步 @50Hz）")
    ap.add_argument("--min-first-survival", type=float, default=0.5,
                    help="首轮存活率低于该值停该档（§5i 停机规则；0=关闭）")
    ap.add_argument("--cem-seed", type=int, default=0,
                    help="z 采样 numpy 种子 + torch 全局种子（落盘可复现）")
    ap.add_argument("--jitter-seed", type=int, default=0,
                    help="第 it 轮 rollout 的 jitter seed = jitter_seed + it")
    ap.add_argument("--router-model-dir", default=DEFAULT_ROUTER_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT_BASE, help="输出根目录")
    ap.add_argument("--tag", default="vae",
                    help="输出子目录 = d048r_r3_<tag>")
    return ap


def main():
    from isaaclab.app import AppLauncher

    ap = build_args()
    AppLauncher.add_app_launcher_args(ap)
    # server has no display: viewport/hydra init segfaults (repo gotcha)
    ap.set_defaults(headless=True)
    cli = ap.parse_args()

    vb_list = [int(v) for v in cli.vb_list.split(",")]
    steps = int(round(cli.rollout_s * 50.0))
    dur = steps / 50.0
    if cli.iters < 1:
        raise SystemExit("[args] FAIL: --iters 必须 >=1")
    out_dir = os.path.join(cli.out, f"d048r_r3_{cli.tag}")
    os.makedirs(out_dir, exist_ok=True)

    # 种子记录与复现：z 采样 = numpy default_rng(cem_seed)；env 初态/噪声 =
    # torch.manual_seed(cem_seed) + 每轮 jitter_and_reset(seed=jitter_seed+it)
    # （内部 default_rng(1000+seed)）。三条种子全部落盘。
    import torch
    torch.manual_seed(cli.cem_seed)
    torch.cuda.manual_seed_all(cli.cem_seed)
    rng = np.random.default_rng(cli.cem_seed)

    app_launcher = AppLauncher(cli)
    sim_app = app_launcher.app

    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.eval_apt_isaac import jitter_and_reset

    class CemEnv(AptFlatG1Env):
        """z 常向量 oracle：_compute_q_des 旁路策略，直接
        tokens = VAE.decode(z_const, sc, vb_const, db_const)（clock 推进抄 env
        latent 分支固定步频；decoder 尾部抄 oracle_token_replay 六参直传）。
        canonical apt_flat_env.py 零改动。"""

        _cem_z: torch.Tensor | None = None
        _cem_vb: int = 1
        _cem_db: int = 4

        def _compute_q_des(self, phase, aux, res=None):
            with torch.no_grad():
                phi = self._latent_phase
                sc = torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)
                vb = torch.full((self.num_envs,), self._cem_vb,
                                dtype=torch.long, device=self.device)
                db = torch.full((self.num_envs,), self._cem_db,
                                dtype=torch.long, device=self.device)
                tokens = self._vae.decode(self._cem_z, sc, vb, db).detach()
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
    # D048r hotfix3：observation_space 由调用方按 eval 同式 bump（latent_mode
    # 分支 _last_phase 2 -> 16 in observation = +14；缺省 91 与 latent obs 105
    # 不符 → _get_observations assert 崩，smoke 首跑复现）
    cfg.observation_space += 14
    cfg.action_space = 16                          # latent z only（train 同款）
    env = CemEnv(cfg)
    env._cem_db = int(cli.force_dbin)
    if not (0 <= env._cem_db < cli.n_dbins):
        raise SystemExit(f"[args] FAIL: force_dbin {env._cem_db} 出值域 "
                         f"[0,{cli.n_dbins})")
    for vb in vb_list:
        if not (0 <= vb < cli.n_vbins):
            raise SystemExit(f"[args] FAIL: vb {vb} 出值域 [0,{cli.n_vbins})")
    device = env.device

    def _yaw_vec(q):
        """(N,4) w-first quat → (N,) yaw（eval_apt_isaac._yaw_of 向量化）。"""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _upright_final_vec(q):
        """(N,4) quat → (N,) upright = exp(-g_xy²/0.1)（eval 同式：直立 1）。"""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        g_xy = 2.0 * torch.sqrt((x * z - w * y) ** 2 + (y * z + w * x) ** 2)
        return torch.exp(-(g_xy ** 2) / 0.1)

    def rollout(z_np: np.ndarray, vb: int, seed: int) -> dict:
        """一轮 20s：全 env 并行，返回 fitness/诊断（numpy，逐 env）。"""
        env._cem_z = torch.from_numpy(z_np).float().to(device)
        env._cem_vb = int(vb)
        jitter_and_reset(env, seed)
        env._latent_phase.zero_()      # 时钟起点钉 0（可复现；落盘注记）
        n = env.num_envs
        xy = env.robot.data.root_pos_w[:, :2].detach()
        q0 = env.robot.data.root_quat_w.detach()
        xy0 = xy.clone()
        yaw0 = _yaw_vec(q0)
        f0 = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], dim=1)
        xy_last = xy.clone()
        quat_last = q0.clone()
        h_min = torch.full((n,), float("inf"), device=device)
        fall_step = np.full(n, -1, dtype=np.int64)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        for t in range(steps):
            action = torch.zeros(n, env.cfg.action_space,
                                 dtype=torch.float32, device=device)
            obs, rew, term, trunc, _ = env.step(action)
            done = term.bool() | trunc.bool()
            live = ~done
            if live.any():     # done 步 robot.data 已被 auto-reset 污染（b3p 口径）
                xy_last[live] = env.robot.data.root_pos_w[live, :2].detach()
                quat_last[live] = env.robot.data.root_quat_w[live].detach()
                h_min[live] = torch.minimum(
                    h_min[live], env.robot.data.root_pos_w[live, 2].detach())
            newly = (term.bool() & alive)
            fall_step[newly.cpu().numpy()] = t
            alive &= ~term.bool()
            if not bool(alive.any()):
                break
        fwd = ((xy_last - xy0) * f0).sum(dim=1)     # 初始航向系净前向位移
        fit = (fwd / dur).cpu().numpy().copy()
        upright_final = _upright_final_vec(quat_last).cpu().numpy()
        n_fall = int((fall_step >= 0).sum())
        fit[fall_step >= 0] = -1.0                  # 摔倒判负（预注册）
        fit[upright_final < 0.9] = -1.0             # 终末姿态判负（预注册）
        return {
            "fitness": fit,
            "fall_step": fall_step,
            "n_fall": n_fall,
            "n_upright_fail": int(((upright_final < 0.9) & (fall_step < 0)).sum()),
            "survival_rate": float((fall_step < 0).mean()),
            "h_min": h_min.cpu().numpy(),
        }

    per_vb, z_dump = {}, {}
    for vb in vb_list:
        hist = {"vb": vb, "iterations": [], "stopped_early": False}
        mu = np.zeros(16)
        var = np.ones(16)
        last = None
        for it in range(cli.iters):
            if it == 0:
                z = cem_sample(rng, np.zeros(16), np.ones(16), cli.pop)
                # 首轮 z~N(0,1)：与后续 N(mu,var) 同式（mu=0, var=1），逐 env 独立
            else:
                z = cem_sample(rng, mu, var, cli.pop)
            roll = rollout(z, vb, seed=cli.jitter_seed + it)
            fit = roll["fitness"]
            best_i = int(np.argmax(fit))
            k = max(1, int(round(cli.elite_frac * cli.pop)))
            it_rec = {
                "iter": it,
                "k_elite": k,
                "fitness_med": round(float(np.median(fit)), 4),
                "fitness_max": round(float(fit.max()), 4),
                "survival_rate": round(roll["survival_rate"], 4),
                "n_fall": roll["n_fall"],
                "n_upright_fail": roll["n_upright_fail"],
                "h_min_min": round(float(roll["h_min"].min()), 3),
                "best_env_idx": best_i,
                "seed_jitter": cli.jitter_seed + it,
            }
            hist["iterations"].append(it_rec)
            print(f"[cem] vb{vb} it{it}: med={it_rec['fitness_med']} "
                  f"max={it_rec['fitness_max']} surv={it_rec['survival_rate']} "
                  f"fall={roll['n_fall']}", flush=True)
            z_dump[f"z_vb{vb}_it{it}"] = z
            z_dump[f"fit_vb{vb}_it{it}"] = fit
            z_dump[f"fall_vb{vb}_it{it}"] = roll["fall_step"]
            last = (z, fit)
            if it == 0 and cli.min_first_survival > 0 and \
                    roll["survival_rate"] < cli.min_first_survival:
                # §5i 停机规则：首轮存活率 <50% → 停该档，落盘 z 采样分布事实
                hist["stopped_early"] = True
                hist["stop_rule"] = (f"首轮存活率 {roll['survival_rate']:.3f} < "
                                     f"{cli.min_first_survival}（§5i 停该档）")
                print(f"[cem] vb{vb}: {hist['stop_rule']}", flush=True)
                break
            mu, var, elite_idx = cem_next_distribution(
                z, fit, cli.elite_frac, cli.std_floor)
            hist["iterations"][-1]["elite_idx"] = elite_idx.tolist()
            z_dump[f"mu_vb{vb}_it{it}"] = mu
            z_dump[f"var_vb{vb}_it{it}"] = var
        z_best, fit_best = last
        bi = int(np.argmax(fit_best))
        per_vb[str(vb)] = {
            **hist,
            "best": {"z": [round(float(v), 6) for v in z_best[bi]],
                     "fitness": round(float(fit_best[bi]), 4),
                     "env_idx": bi},
            "best_iter_overall": int(np.argmax(
                [m2["fitness_max"] for m2 in hist["iterations"]])),
            "vx_star": round(float(fit_best.max()), 4),
            "z_dim": int(z_best.shape[1]),
        }
    z_best_all = {v: d["vx_star"] for v, d in per_vb.items()}
    top_vb = max(z_best_all, key=z_best_all.get)
    envelope = {
        "vx_star_per_vb": z_best_all,
        "interface_ceiling_vx": z_best_all[top_vb],
        "interface_ceiling_rule": "max over vb of 终轮 best fitness（§5i 接口上限"
                                  "= 三档包络最大值）",
        "argmax_vb": int(top_vb),
    }
    print(json.dumps(envelope, indent=1), flush=True)

    out = {
        "experiment": "D048r",
        "layer": "R3",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": {k: v for k, v in vars(cli).items()},
        "seeds": {"cem_numpy": cli.cem_seed, "torch": cli.cem_seed,
                  "jitter_rule": "jitter_and_reset(seed=jitter_seed+iter)，"
                                 "内部 default_rng(1000+seed)"},
        "rollout": {"steps": steps, "dur_s": dur, "pop": cli.pop,
                    "clock_start": 0.0,
                    "clock_rule": "walk clock 起步钉 0、固定 pca rate 推进"},
        "fitness_rule": "净前向位移(初始航向系)/20s；摔倒或终末 upright<0.9 → -1",
        "envelope": envelope,
        "per_vb": per_vb,
    }
    out_path = os.path.join(out_dir, "z_sweep_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(out_dir, "z_history.npz"), **z_dump)
    print("saved", out_path)

    # Isaac 有时在解释器退出时挂死（repo gotcha）：daemon close + 硬退出
    # （oracle/b3p 同款；结果已落盘）。
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
