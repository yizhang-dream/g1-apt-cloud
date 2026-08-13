"""kNN with temporal window tracking (motion-matching style)."""
import json, sys, time
import numpy as np
sys.path.insert(0, '/home/cvgluser/ros2_data')
sys.path.insert(0, '/home/cvgluser/ros2_data/apt_g1')
sys.path.insert(0, '/home/cvgluser/ros2_data/GR00T-WholeBodyControl')
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from eval_distill import NoQuantDecoder, hist_to_proprio

D = '/home/cvgluser/ros2_data/apt_g1/data/exp1'
proprio = np.load(D + '/proprio.npy'); cmd = np.load(D + '/cmd.npy'); token = np.load(D + '/token.npy')
mode = np.load(D + '/mode.npy'); speed = np.load(D + '/speed.npy')
mdir = cmd[:, 4:7]
n = len(proprio)
val = np.zeros(n, dtype=bool); val[15606:17938] = True; val[18722:20308] = True
tr = ~val
tr_idx = np.where(tr)[0]
pmean = proprio[tr_idx].mean(0, keepdims=True).astype(np.float32)
pstd = proprio[tr_idx].std(0, keepdims=True).astype(np.float32) + 1e-6
P = ((proprio - pmean) / pstd).astype(np.float32)

repo = '/home/cvgluser/ros2_data/GR00T-WholeBodyControl'
env = MujocoG1FlatEnv(NoQuantDecoder(repo + '/gear_sonic_deploy/policy/release/model_decoder.onnx'), repo, use_elastic_band=False, stand_only=True)
env.command = np.zeros(3, dtype=np.float32)

def run(name, c, k, win):
    m = int(c['mode']); s = c['speed']
    cand = np.where((mode == m) & (np.abs(speed - s) < 1e-6))[0]
    if c['mdir'][0] > 0:
        cand = cand[mdir[cand, 0] > 0]
    cand = cand[tr[cand]]
    C = P[cand]
    env.reset()
    heights, vxs = [], []
    fall = None
    last_match = 0
    for t in range(600):
        prop = hist_to_proprio(env._get_sonic_history())
        q = ((prop - pmean.ravel()) / pstd.ravel()).astype(np.float32)
        if win is None:
            idxs = np.arange(len(cand))
        else:
            lo = last_match - win; hi = last_match + win + 1
            idxs = np.arange(max(lo, 0), min(hi, len(cand)))
        dist = np.einsum('ij,ij->i', C[idxs], np.broadcast_to(q[None], C[idxs].shape))
        if k == 1:
            best = idxs[int(np.argmax(dist))]
            tok = token[cand[best]]
        else:
            nn = idxs[np.argpartition(-dist, -k)[-k:]]
            wts = np.exp((dist[np.argsort(-dist)][:k] - dist.max()) * 5)
            tok = np.average(token[cand[nn]], axis=0, weights=wts)
        tok = np.clip(np.round(tok * 16) / 16, -1.0, 1.0).astype(np.float32)
        obs, reward, terminated, info = env.step({'token': tok, 'aux': np.zeros(12, dtype=np.float32)})
        heights.append(float(env.data.qpos[2])); vxs.append(float(env._get_base_linear_velocity()[0]))
        if terminated:
            fall = t; break
        last_match = int(np.where(idxs == (nn[0] if k > 1 else best))[0][0]) if k > 1 else int(np.where(idxs == best)[0][0])
    print(f'{name:10s} k={k} win={win} fall={fall} h_min={min(heights):.3f} vx={np.mean(vxs):.3f}', flush=True)
    return dict(fall_step=fall, h_min=round(float(min(heights)), 3), vx=round(float(np.mean(vxs)), 3))

scen = {
    'idle': dict(mode=0, speed=-1.0, mdir=[0, 0, 0]),
    'slow_walk': dict(mode=1, speed=0.2, mdir=[1, 0, 0]),
    'walk': dict(mode=2, speed=-1.0, mdir=[1, 0, 0]),
    'jump': dict(mode=17, speed=-1.0, mdir=[1, 0, 0]),
}
out = {}
for name, c in scen.items():
    for k, win in [(1, None), (3, None), (1, 100), (3, 100), (1, 50), (3, 50)]:
        out[f'{name}_k{k}_w{win}'] = run(name, c, k, win)
json.dump(out, open('/home/cvgluser/ros2_data/apt_g1/outputs/distill/eval_knn_win.json', 'w'), indent=1)