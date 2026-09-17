"""D059 G0-③：爬障段 oracle 回放上限卡（地形场景）——fork 自 D048r R1。

fork 来源：`apt_g1/isaac/replay_token_speed_d048r.py`（D048r R1 现役，518 行）。
本文件与其**同源分叉**，语义除下面七处增量外逐字一致（FORK 纪律：分叉后不回并，
父本后续改动不自动继承；父本 docstring 原文完整保留于本 docstring 下半段）。

七处增量（全量清单，无第八处）：
  ① 地形参数 `--terrain {plane,rough_paper,rough_sym}`（缺省 plane = 平地对照，
     行为与父本完全一致）：`cfg.terrain` 走 `isaac.terrain_cfg.
     make_terrain_importer_cfg` —— 接线范式抄 `apt_g1/isaac/eval_apt_isaac.py`
     （导入见 L798、赋值见 L855-857）；args 块（`vars(cli)` 自动携带）与输出
     JSON 顶层记 terrain 类型。
  ② 路径直度 `straightness = XY 净位移模长 / 路径长`（口径沿 `apt_g1/_archive/
     steer_harvest_d058.py:202-211 straightness_of` 的 net/path；净位移复用父本
     traj_a、路径长复用父本已算 path_len，不另算一套；退化路径返 0.0）：per_run
     加 straightness，per_segment 加 straightness_med（跨 seed 中位），summary
     加 straightness_med（全 run 中位）。
  ③ 本 docstring（fork 声明 + 上述增量清单 + D059 G0-③ 用途）。
  ④ 无 vae_inputs 守卫：D059 爬障段**没有** D048r 的 vae_inputs 链，父本
     resolve_label_slices 只在 `--token-source vae_recon` 分支内调用（orig
     主臂本就不触 label 切片）——本文件把该隐式约定写成**显式守卫**：orig 时
     跳过 label 切片 + 打印 SKIP + 输出 JSON 顶层 `labels=None`；vae_recon 而
     `--vae-inputs-dir` 不是存在目录时在 AppLauncher 之前 SystemExit（Isaac
     前 fail-fast，不静默错标）。
  ⑤ D059 身份串：输出 JSON `"experiment": "D059"`（父本 "D048r"）、输出子目录
     前缀 `d059_terrain_<tag>`（父本 `d048r_<tag>`）、缺省 segments-json 指向
     D059 爬障选样产物、缺省输出根 outputs/d059_terrain——防 D059 产物混入
     D048r 命名空间。
  ⑥ 地形噪声 `--terrain-noise FLOAT`（缺省 0.04 = 现行为零漂移）：噪声幅度不再
     吃 make_terrain_importer_cfg 的缺省——直传 `noise=cli.terrain_noise`
     （terrain_cfg.py:16-20 本就有该可选参数 → **共享文件零改动**，未走"加参数
     / 自建 cfg"两条退路）。plane 分支不消费 noise（平地对照逐字节不变）；
     rough/rough_paper/rough_sym 生效；值与 terrain 类型同经 args 块（vars(cli)
     自动携带）入输出 JSON，并逐 run 落 ⑦ 的 npz meta。
  ⑦ 逐 50Hz 步状态记录 `--record-state OUT_DIR`（缺省 "" = 关，零行为漂移；
     循环内零磁盘 IO）：每 run 一个纯 numpy 缓冲（StateRecorder），run 结束
     一次性落 `<OUT_DIR>/<stem>__seed<k>.npz`，键 =
       q_des (n,29)      当步下发的 29 体关节位置目标（env._q_des，即
                         _apply_action 直喂 set_joint_position_target 的量，
                         非 full 46 维；含手/腰的补零不属本记录口径）
       root_pos (n,3)    step 后 root 世界位置
       root_quat (n,4)   step 后 root 世界四元数（Isaac w-first）
       tokens_exec (n,64) 当步喂冻结 decoder 的 token 行（vae_recon 时为重建
                         token）；索引取 env 解码路径真实执行的 `_last_token_i`
                         （不在循环里重推 min(t, n-1)），值取 numpy 侧 tokens
       root_pos_pre (n,3) / root_quat_pre (n,4)  step **前**（= 执行 tokens_exec[t]
                         之前）的 root 位姿，即第 t 步的（状态, 动作）因果对；t=0
                         行 = jitter 站姿起步位姿（与指标 xy0 同源）
       fall_step         标量，未摔 = -1
       meta              JSON 串（stem/seed/terrain/terrain_noise/v_med/
                         token_source/steps/steps_budget/dur_s/completed/survived）
     体位姿为什么给两套：`root_pos/root_quat` = **step 后**，沿本文件指标路径与
     playback_window "step 后状态 ↔ 参考行 t" 口径（post[t] 与 tokens_exec[t] 同
     行即"该 token 执行完的结果"）；`*_pre` = **step 前**，对齐 D060 源 1 生成器
     （build/gen_planner_scripts_d060.py docstring："PRE-step state at token index
     t"）的 (state_t → token_t) 语料口径。两者同源（pre[t] ≡ post[t-1]，仅 t=0
     行需起步位姿），同时落盘使 D060 四源拼接不必重跑本卡、也不必猜口径。
     n = 未 done 的步数：done 步**不可记**（DirectRLEnv 在该步内 auto-reset，
     step 后 root_pos/_q_des 已是复位值 = 污染，与指标"done 步不计"同一理由；
     该步步前值可由 pre 对补出，本记录不含）。缓冲 = 列表累积 + 结束 np.stack，
     循环内只做小张量→numpy 拷贝；落盘失败只告警不中断（长 run 里指标 JSON
     优先）。本机自测见 record_state_selftest / `--selftest`（纯 numpy，无 IsaacLab）。

D059 G0-③ 用途（`refine-logs/ds/DS_CONTINUOUS_EXECUTION_PLAN.md` §5s 预注册，
09-16 立项）："冻结 SONIC decoder 是否具备地形剧本执行能力"从未测过（D048r R1
上限卡只测平地语料）；本卡 = ds_bones 爬障段分层选段（障碍类型×段速，K≥12）
× 3 seeds × 20s，场景 = 平地对照（--terrain plane）+ 地形。指标沿 R1：存活率 /
h_min / realized_ratio / 路径直度。**预注册偏差登记（09-17）**：§5s 原文"mjlab
地形（D030 驱动复用）"经查不可序列化复用（sim/ds_mode_terrain.py:8-11，MjSpec
内存组装、无 mesh 资产导出）→ 场景改用 isaac/terrain_cfg.py 的 IsaacLab 生成器
按同一组论文参数重造（rough_paper），偏离照 D058 先例落账（tracker D059 行）。
判读：甲 = 存活 ≥2/3 ∧ realized_ratio 不显著
低于平地 R1 带 → 地形剧本可演、全线开；乙 = 可演但跟踪显著劣化 → 余量问题登记，
作者设计须含纠偏预算；丙 = 大面积摔 → 冻结 decoder 地形线根死、纲领级收线报
owner。

Run on lab-ts (Isaac wrapper, cwd=GR00T-WholeBodyControl):
  nohup bash /tmp/run_apt_isaac.sh \
    /home/cvgluser/ros2_data/apt_g1/isaac/replay_token_terrain_d059.py \
    --terrain plane --token-source orig --tag flat \
    > /home/cvgluser/ros2_data/apt_g1/outputs/d059_terrain/flat.log 2>&1 < /dev/null & disown
  # 地形臂：--terrain rough_paper（论文形状对称 rough）/ rough_sym（0.1m 对称对照）
  # R1 主臂不读 --vae-inputs-dir（增量④）：D059 爬障段无需 D048r label 切片链，
  # 输出 JSON 的 labels 落 null；只有 R2（--token-source vae_recon）才要求该目录存在。
  # R2 路径同父本：--token-source vae_recon --latent-vae-path <vae.pt> --tag r2_<vaeTag>
  # 地形噪声（增量⑥，缺省 0.04 = rough_paper 论文形状参数）：--terrain-noise 0.06
  # 闭环收割（增量⑦，D060 源 4）：--record-state <OUT_DIR>（逐 run 落
  # <stem>__seed<k>.npz；不回写指标路径）
  # 本机无 IsaacLab 自测（纯 numpy）：PYTHONPATH=apt_g1 python <本文件> --selftest

---------- 以下为父本 D048r docstring 原文（未改动，仅去掉原首行三引号） ----------

D048r R1/R2：快段 token 闭环回放——decoder 本体上限 / VAE 重建代价三角测量。

DS_CONTINUOUS_EXECUTION_PLAN §5i 预注册。R1（本体层）= probe/pick_fast_segments_d048r.py
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
import sys
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
    from build_d048n_vb_speed_labels import (
        DEFAULT_INPUTS,
        DEFAULT_NPZ_DIRS,
        align_segments,
        frame_speed_bwd,
        md5_frame,
    )
    from isaac.playback_window import playback_same_window_ratio
    from train_token_vae_e39 import build_windows

HOME = os.path.expanduser("~")
DEFAULT_SEGMENTS_JSON = (f"{HOME}/ros2_data/apt_g1/data/ds_bones/"
                         f"g1_d059_climb_smoke/d059_climb_segments.json")
DEFAULT_OUT_BASE = f"{HOME}/ros2_data/apt_g1/outputs/d059_terrain"
DEFAULT_ROUTER_DIR = f"{HOME}/ros2_data/apt_g1/outputs/distill_final"


def build_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--segments-json", default=DEFAULT_SEGMENTS_JSON,
                    help="段清单（D059 select_climb_d059 --segments-json-out 产物，"
                         "D048r pick_fast_segments 同构；segments[].stem/npz_path/"
                         "v_med）")
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
    ap.add_argument("--terrain", choices=["plane", "rough_paper", "rough_sym"],
                    default="plane",
                    help="D059：场景地形（plane=平地对照 = 父本行为；"
                         "rough_paper=论文形状对称 rough；rough_sym=0.1m 对称对照）；"
                         "noise 见 --terrain-noise（缺省 0.04），seed 用 "
                         "make_terrain_importer_cfg 缺省 0")
    ap.add_argument("--terrain-noise", type=float, default=0.04, dest="terrain_noise",
                    help="D059 增量⑥：地形噪声幅度（直传 make_terrain_importer_cfg"
                         " 的 noise；缺省 0.04 = 现行为零漂移；plane 分支不消费）")
    ap.add_argument("--out", default=DEFAULT_OUT_BASE, help="输出根目录")
    ap.add_argument("--tag", default="",
                    help="输出子目录 = d059_terrain_<tag>（缺省 tag 沿父本 r1_orig/r2_vae；G0-③ 惯例 tag=flat/rough_paper）")
    ap.add_argument("--record-state", default="", dest="record_state",
                    help="D059 增量⑦：逐 50Hz 步状态记录输出目录（缺省空 = 关，"
                         "零行为漂移）→ <OUT_DIR>/<stem>__seed<k>.npz；键见 docstring"
                         " 增量⑦（q_des/root_pos/root_quat/tokens_exec/fall_step/meta）")
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


# ------------------------------------- D059 增量②：路径直度（纯函数，可独立自测）


def path_straightness(xy_a, path_len_m=None) -> float:
    """直度 = XY 净位移模长 / 路径长（口径沿 _archive/steer_harvest_d058.py:
    202-211 straightness_of 的 net/path；退化路径返 0.0）。

    path_len_m=None 时自算（独立调用/自测）；生产调用点传入父本已算的 path_len
    （与 mean_speed_path 同源量，不另算一套）。净位移 = 首末点 XY 距离。
    """
    xy_a = np.asarray(xy_a, dtype=np.float64)
    if len(xy_a) < 2:
        return 0.0
    if path_len_m is None:
        path_len_m = float(np.linalg.norm(np.diff(xy_a, axis=0), axis=1).sum())
    if path_len_m <= 0.0:
        return 0.0
    return float(np.linalg.norm(xy_a[-1] - xy_a[0])) / path_len_m


# ------------------- D059 增量⑦：逐 50Hz 步状态记录（纯 numpy，可独立自测）


RECORD_KEYS = ("q_des", "root_pos", "root_quat", "tokens_exec",
               "root_pos_pre", "root_quat_pre")
RECORD_DIMS = {"q_des": 29, "root_pos": 3, "root_quat": 4, "tokens_exec": 64,
               "root_pos_pre": 3, "root_quat_pre": 4}


class StateRecorder:
    """逐 50Hz 控制步状态缓冲：列表累积，run 结束 np.stack 一次性落盘。

    口径见 docstring 增量⑦。调用约定：调用方只在"未 done 的 step"（step 后
    状态未被 auto-reset 污染）调 add_step，且只传当步各量的 numpy 视图
    （`enabled=False` 时 add_step 空转，调用方应另行判 .enabled 以免白做
    张量→numpy 拷贝）。finalize 严格校验：逐键行数一致 + 宽度 == RECORD_DIMS，
    不符即 ValueError（生产路径 save_state_record 捕获后只告警，不让记录问题
    毁掉已跑完的 run）。
    """

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._buf: dict[str, list[np.ndarray]] = {k: [] for k in RECORD_KEYS}

    def add_step(self, q_des, root_pos, root_quat, tokens_exec,
                 root_pos_pre, root_quat_pre):
        if not self.enabled:
            return
        for k, v in zip(RECORD_KEYS, (q_des, root_pos, root_quat, tokens_exec,
                                      root_pos_pre, root_quat_pre)):
            self._buf[k].append(np.asarray(v, dtype=np.float32).reshape(-1).copy())

    def __len__(self) -> int:
        return len(self._buf["q_des"])

    def finalize(self) -> dict[str, np.ndarray]:
        n = len(self)
        arrays = {}
        for k in RECORD_KEYS:
            rows = self._buf[k]
            if len(rows) != n:
                raise ValueError(f"[record] {k} 行数 {len(rows)} != {n}")
            if n == 0:
                arrays[k] = np.zeros((0, RECORD_DIMS[k]), dtype=np.float32)
                continue
            a = np.stack(rows, axis=0)
            if a.shape[1] != RECORD_DIMS[k]:
                raise ValueError(f"[record] {k} 宽度 {a.shape[1]} != "
                                 f"RECORD_DIMS {RECORD_DIMS[k]}")
            arrays[k] = a
        return arrays


def save_state_record(out_root: str, stem: str, seed: int, recorder: StateRecorder,
                      meta: dict, fall_step) -> str | None:
    """recorder → `<out_root>/<stem>__seed<seed>.npz`（+ fall_step 标量 + meta 串）。

    fall_step=None（未摔）落 -1（0 是合法步号，不能当哨兵）。落盘失败只告警
    返回 None：指标 JSON 优先，记录问题不得中断长 run。n 与 steps-(有摔?1:0)
    不符时告警（提示缓冲与指标口径错位，如误记了 done 步）——D059 配置下
    episode_length_s = 预算 + 30s，trunc 不触发，故"未 done 步数"就是这个式子。
    """
    path = os.path.join(out_root, f"{stem}__seed{seed}.npz")
    n_expect = int(meta["steps"]) - (1 if fall_step is not None else 0)
    try:
        os.makedirs(out_root, exist_ok=True)
        arrays = recorder.finalize()
        arrays["fall_step"] = np.int64(-1 if fall_step is None else fall_step)
        arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
        np.savez(path, **arrays)
    except Exception as e:      # 记录失败不致命：指标 JSON 必须能落盘
        print(f"[record] FAIL {path}: {type(e).__name__}: {e}"
              f"（指标 JSON 不受影响）", flush=True)
        return None
    note = "" if len(recorder) == n_expect else (
        f" [!] 记录 {len(recorder)} 步 != 预期未 done 步 {n_expect}")
    print(f"[record] saved {path} n={len(recorder)} "
          f"fall_step={int(arrays['fall_step'])}{note}", flush=True)
    return path


def record_state_selftest() -> int:
    """增量⑦ 记录缓冲自测（纯 numpy，5 步合成数据→落盘→读回逐键断言）。

    用法（本机，无 IsaacLab）：PYTHONPATH=apt_g1 python <本文件> --selftest
    """
    import inspect
    import tempfile

    fails: list[str] = []
    n, rng = 5, np.random.default_rng(0)
    meta = {"stem": "seg_demo", "seed": 3, "terrain": "rough_paper",
            "terrain_noise": 0.06, "v_med": 0.42, "token_source": "orig",
            "steps": 6, "steps_budget": 1000, "dur_s": 20.0,
            "completed": False, "survived": False}

    def check(name, cond, extra=""):
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{extra}", flush=True)
        if not cond:
            fails.append(name)

    print("=== D059 增量⑦ StateRecorder selftest（纯 numpy，无 IsaacLab）===",
          flush=True)
    data = {k: rng.standard_normal((n, RECORD_DIMS[k])).astype(np.float32)
            for k in RECORD_KEYS}
    check("RECORD_KEYS 顺序 == add_step 形参顺序（位置 zip 依赖此不变量）",
          list(RECORD_KEYS) == list(
              inspect.signature(StateRecorder.add_step).parameters)[1:])
    buf = StateRecorder(enabled=True)
    for t in range(n):
        buf.add_step(*[data[k][t] for k in RECORD_KEYS])
    check("len == 5", len(buf) == n, f" (got {len(buf)})")
    arrays = buf.finalize()
    for k in RECORD_KEYS:
        check(f"finalize[{k}] shape == {(n, RECORD_DIMS[k])}",
              arrays[k].shape == (n, RECORD_DIMS[k]), f" (got {arrays[k].shape})")
        check(f"finalize[{k}] 与输入逐位相同", np.array_equal(arrays[k], data[k]))
    check("finalize 为 copy（不共享输入内存）",
          not np.shares_memory(arrays["q_des"], data["q_des"]))

    off = StateRecorder(enabled=False)
    off.add_step(*[data[k][0] for k in RECORD_KEYS])
    check("enabled=False 不累积", len(off) == 0)
    check("enabled=False finalize → (0,dim) 空表",
          all(off.finalize()[k].shape == (0, RECORD_DIMS[k]) for k in RECORD_KEYS))

    bad = StateRecorder(enabled=True)
    bad.add_step(np.zeros(29, np.float32), np.zeros(3, np.float32),
                 np.zeros(4, np.float32), np.zeros(63, np.float32),
                 np.zeros(3, np.float32), np.zeros(4, np.float32))
    try:
        bad.finalize()
        check("宽度不符 → ValueError", False)
    except ValueError as e:
        check("宽度不符 → ValueError", "宽度" in str(e), f" ({e})")

    ragged = StateRecorder(enabled=True)
    ragged.add_step(*[data[k][0] for k in RECORD_KEYS])
    ragged._buf["root_pos"].clear()      # 伪造逐键行数错位
    try:
        ragged.finalize()
        check("行数不一致 → ValueError", False)
    except ValueError as e:
        check("行数不一致 → ValueError", "行数" in str(e), f" ({e})")

    with tempfile.TemporaryDirectory() as td:
        p = save_state_record(td, "seg_demo", 3, buf, meta, fall_step=5)
        check("落盘名 = <stem>__seed<k>.npz",
              p is not None and os.path.basename(p) == "seg_demo__seed3.npz",
              f" ({p})")
        check("文件存在", bool(p) and os.path.isfile(p))
        with np.load(p) as z:      # 上下文管理：Windows 上不及时关句柄会卡住
            files = set(z.files)   # TemporaryDirectory 清理
            got = {k: z[k].copy() for k in RECORD_KEYS}
            got_fall, got_ndim = int(z["fall_step"]), z["fall_step"].ndim
            got_meta = json.loads(str(z["meta"]))
        check("npz 键齐（6 量 + fall_step + meta）",
              set(RECORD_KEYS) | {"fall_step", "meta"} <= files, f" ({sorted(files)})")
        for k in RECORD_KEYS:
            check(f"读回 {k} 与输入逐位相同", np.array_equal(got[k], data[k]))
        check("读回 fall_step == 5", got_fall == 5, f" (got {got_fall})")
        check("fall_step 为 0-d 标量", got_ndim == 0)
        check("读回 meta 整字典相等", got_meta == meta, f" ({got_meta})")
        check("meta 含增量⑥⑦必备键 stem/seed/terrain/terrain_noise/v_med",
              all(kk in got_meta for kk in ("stem", "seed", "terrain",
                                            "terrain_noise", "v_med")))

    with tempfile.TemporaryDirectory() as td:
        meta_nofall = {**meta, "seed": 0, "steps": n}   # 未摔：n == steps（自洽）
        p = save_state_record(td, "seg_demo", 0, buf, meta_nofall, fall_step=None)
        with np.load(p) as z:
            got_fall = int(z["fall_step"])
        check("未摔 fall_step == -1", got_fall == -1)

    empty = StateRecorder(enabled=True)
    with tempfile.TemporaryDirectory() as td:
        # 真路径边界：第 0 步就摔（无未 done 步）→ n=0 仍须可落可读
        p = save_state_record(td, "seg_fall0", 7, empty,
                              {**meta, "stem": "seg_fall0", "steps": 1},
                              fall_step=0)
        with np.load(p) as z:
            got = {k: z[k].shape for k in RECORD_KEYS}
            got_fall = int(z["fall_step"])
        check("n=0（首步即摔）npz 键形状 (0,dim)",
              got == {k: (0, RECORD_DIMS[k]) for k in RECORD_KEYS}, f" ({got})")
        check("n=0 时 fall_step == 0（0 不当哨兵）", got_fall == 0)

    with tempfile.TemporaryDirectory() as td:
        p = save_state_record(td, "seg_demo", 1, buf, {**meta, "steps": 3},
                              fall_step=None)
        check("n 与 steps 不符只告警、仍返回路径", p is not None)

    print(f"=== selftest {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}"
          f"（{len(fails)} 项失败）===", flush=True)
    return 0 if not fails else 1


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
    out_dir = os.path.join(cli.out, f"d059_terrain_{tag}")

    if cli.token_source == "vae_recon" and not cli.latent_vae_path:
        raise SystemExit("[args] FAIL: --token-source vae_recon 需要 --latent-vae-path")

    # D059 增量④：无 vae_inputs 守卫（把父本的隐式约定显式化）。D059 爬障段没有
    # D048r 的 vae_inputs 链：orig 主臂不触 label 切片（父本 resolve_label_slices
    # 只在 vae_recon 分支内调用），输出 JSON 顶层记 labels=None；vae_recon 而
    # --vae-inputs-dir 不是存在目录 → Isaac 启动前 fail-fast（不烧 GPU 静默错标）。
    slice_labels = cli.token_source == "vae_recon"
    if slice_labels and not (cli.vae_inputs_dir
                             and os.path.isdir(cli.vae_inputs_dir)):
        raise SystemExit("[args] FAIL: --token-source vae_recon 需要存在的 "
                         f"--vae-inputs-dir（当前 {cli.vae_inputs_dir!r}）")
    if not slice_labels:      # orig：显式跳过 label 切片（labels=None 落 JSON）
        print("[labels] SKIP: token_source=orig（D059 无 vae_inputs 链）"
              "→ labels=None", flush=True)

    # Isaac 前 fail-fast：段装载 + R2 标签对齐（numpy-only）
    segs = load_segment_list(cli)
    label_slices = None
    if cli.token_source == "vae_recon":
        label_slices = resolve_label_slices(
            cli, [s["stem"] for s in segs], {s["stem"]: s["frames"] for s in segs})

    app_launcher = AppLauncher(cli)
    sim_app = app_launcher.app

    from isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from isaac.eval_apt_isaac import jitter_and_reset
    from isaac.terrain_cfg import make_terrain_importer_cfg  # D059 增量①

    class SpeedReplayEnv(AptFlatG1Env):
        """Token oracle（D034/D037 同构，骨架抄 oracle_token_replay_isaac.py）：
        _compute_q_des 旁路 policy/VAE/router，逐帧 token 直喂冻结 decoder，
        env 自持 10 帧闭环 history。canonical apt_flat_env.py 零改动。"""

        _oracle_tokens: torch.Tensor | None = None
        _oracle_idx: int = 0
        _last_token_i: int = 0   # D059 增量⑦：当步真实执行的 token 行索引

        def _compute_q_des(self, phase, aux, res=None):
            i = min(self._oracle_idx, self._oracle_tokens.shape[0] - 1)
            self._last_token_i = i
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
    # D059 增量①：地形接线（plane 与父本 cfg.terrain 缺省等值 = 平地对照零变化）；
    # 范式 = eval_apt_isaac.py:798 import + L855-857 make_terrain_importer_cfg
    # D059 增量⑥：noise 显式直传（terrain_cfg.py 本就有 noise 可选参数，共享文件
    # 零改动）；缺省 0.04 == 函数缺省 → 与父本/增量①现行为逐字节等价（plane 分支
    # 不消费 noise）。
    cfg.terrain = make_terrain_importer_cfg(cli.terrain, noise=cli.terrain_noise)
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
            # D059 增量⑦：每 run 独立缓冲（关时 add_step 空转，不走下面的拷贝）
            recorder = StateRecorder(enabled=bool(cli.record_state))
            xy0 = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy()
            yaw0 = _yaw_of(env.robot.data.root_quat_w[0].detach())
            f0 = np.array([math.cos(yaw0), math.sin(yaw0)])  # 初始航向系前向
            traj_pb = [xy0]          # 播放相轨迹（首点=重置位姿，同窗比值基点）
            h_min, h_end, fall_step, steps_done = float("inf"), None, None, 0
            traj = [xy0]             # 全程 live 轨迹（done 步不计，b3p 口径）
            for t in range(steps_budget):
                if recorder.enabled:   # D059 增量⑦：步前位姿（t=0 = 起步位姿）
                    pre_pos = env.robot.data.root_pos_w[0].detach().cpu().numpy()
                    pre_quat = env.robot.data.root_quat_w[0].detach().cpu().numpy()
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
                    if recorder.enabled:   # D059 增量⑦：未 done 步记录（step 后状态；
                        # done 步不可记：auto-reset 后 root_pos/_q_des 已是复位值）
                        recorder.add_step(
                            env._q_des[0].detach().cpu().numpy(),          # (29,)
                            env.robot.data.root_pos_w[0].detach().cpu().numpy(),
                            env.robot.data.root_quat_w[0].detach().cpu().numpy(),
                            tokens[env._last_token_i],                     # (64,)
                            pre_pos, pre_quat,     # 步前位姿（D060 源1 口径）
                        )
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
            straightness = path_straightness(traj_a, path_len)  # D059 增量②
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
                "straightness": round(straightness, 4),
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
            if recorder.enabled:   # D059 增量⑦：run 末一次性落盘（循环内零 IO）
                save_state_record(
                    cli.record_state, seg["stem"], seed, recorder,
                    meta={
                        "stem": seg["stem"], "seed": seed,
                        "terrain": cli.terrain,
                        "terrain_noise": cli.terrain_noise,
                        "v_med": round(seg["v_med"], 6),
                        "token_source": cli.token_source,
                        "steps": steps_done, "steps_budget": steps_budget,
                        "dur_s": cli.dur_s,
                        "completed": r["completed"], "survived": survived,
                    },
                    fall_step=fall_step,
                )

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
            "straightness_med": _med("straightness"),
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
        "straightness_med": (round(float(np.median(
            [r["straightness"] for r in per_run.values()
             if r["straightness"] is not None])), 4)
            if any(r["straightness"] is not None for r in per_run.values())
            else None),
    }

    out = {
        "experiment": "D059",
        "layer": "R1" if cli.token_source == "orig" else "R2",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": {k: v for k, v in vars(cli).items()},
        "seeds": seeds,
        "terrain": cli.terrain,  # D059 增量①：场景地形（args 块亦自动携带）
        "steps_budget": steps_budget,
        "label_alignment_provenance": (
            "build_d048n.align_segments + 逐帧 md5 抽查"
            if label_slices else "n/a (token_source=orig)"),
        "labels": ({"vae_inputs_dir": cli.vae_inputs_dir, "vb_file": cli.vb_file,
                    "db_file": cli.db_file} if label_slices else None),
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
    if "--selftest" in sys.argv[1:]:      # 增量⑦自测：纯 numpy，Isaac 前短路
        sys.exit(record_state_selftest())
    main()
