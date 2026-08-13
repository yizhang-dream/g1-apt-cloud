"""kNN memory-based distillation: closed-loop no-band eval."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, '/home/cvgluser/ros2_data')
sys.path.insert(0, '/home/cvgluser/ros2_data/apt_g1')
sys.path.insert(0, '/home/cvgluser/ros2_data/GR00T-WholeBodyControl')
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from eval_distill import NoQuantDecoder, SCEN, build_cmd_feature, hist_to_proprio

D = '/home/cvgluser/ros2_data/apt_g1/data/exp1'
proprio = np.load(D + '/proprio.npy'); cmd = np.load(D + '/cmd.npy'); token = np.load(D + '/token.npy')
mode = np.load(D + '/mode.npy'); speed = np.load(D + '/speed.npy'); modes = np.load(D + '/meta_modes.npy')
mdir = cmd[:, 4:7]
n = len(proprio)
val = np.zeros(n, dtype=bool); val[15606:17938] = True; val[18722:20308] = True
tr = ~val
tr_idx = np.where(tr)[0]
pmean = proprio[tr_idx].mean(0, keepdims=True).astype(np.float32)
pstd = proprio[tr_idx].std(0, keepdims=True).astype(np.float32) + 1e-6
P = ((proprio - pmean) / pstd).astype(np.float32)

def knn_tokens_for(c, k=3):
    m = int(c['mode'])
    s = c['speed']
    d0 = c['mdir'][0]
    cand = np.where((mode == m) & (np.abs(speed - s) < 1e-6))[0]
    if d0 > 0:
        cand = cand[mdir[cand, 0] > 0]
    elif d0 < 0:
        cand = cand[mdir[cand, 0] <= 0]
    cand = cand[tr[cand]]
    return cand, k

repo = '/home/cvgluser/ros2_data/GR00T-WholeBodyControl'
env = MujocoG1FlatEnv(NoQuantDecoder(repo + '/gear_sonic_deploy/policy/release/model_decoder.onnx'), repo, use_elastic_band=False, stand_only=True)
env.command = np.zeros(3, dtype=np.float32)

out = {}
for name, c in SCEN:
    cand, k = knn_tokens_for(c)
    print(name, 'candidates', len(cand), flush=True)
    if len(cand) == 0:
        out[name] = {'fall_step': None, 'note': 'no candidates'}
        continue
    C = P[cand]  # (M,930)
    env.reset()
    heights, vxs = [], []
    fall = None
    x0 = float(env.data.xpos[env.model.body('pelvis').id][0])
    t0 = time.time()
    for t in range(400):
        prop = hist_to_proprio(env._get_sonic_history())
        q = (prop - pmean.ravel()) / pstd.ravel()
        dist = np.einsum('ij,ij->i', C, np.broadcast_to(q[None], C.shape))
        nn = np.argpartition(-dist, -k)[-k:]  # largest dot = nearest
        tok = np.round(token[cand[nn]].mean(0) * 16) / 16
        tok = np.clip(tok, -1.0, 1.0).astype(np.float32)
        obs, reward, terminated, info = env.step({'token': tok, 'aux': np.zeros(12, dtype=np.float32)})
        heights.append(float(env.data.qpos[2])); vxs.append(float(env._get_base_linear_velocity()[0]))
        if terminated:
            fall = t; break
    out[name] = dict(fall_step=fall, h_mean=round(float(np.mean(heights)), 3), h_min=round(float(np.min(heights)), 3),
                     vx=round(float(np.mean(vxs)), 3), x_end=round(float(env.data.xpos[env.model.body('pelvis').id][0]) - x0, 2),
                     time_s=round(time.time() - t0, 1))
    print(' ', out[name], flush=True)
json.dump(out, open('/home/cvgluser/ros2_data/apt_g1/outputs/distill/eval_knn.json', 'w'), indent=1)