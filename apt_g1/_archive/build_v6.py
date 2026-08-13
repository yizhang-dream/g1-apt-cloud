"""v6: per-group best prototype config; full battery + oracle stealth check."""
import json, os, shutil, sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, '/home/cvgluser/ros2_data')
sys.path.insert(0, '/home/cvgluser/ros2_data/apt_g1')
sys.path.insert(0, '/home/cvgluser/ros2_data/GR00T-WholeBodyControl')

D = '/home/cvgluser/ros2_data/apt_g1/data/exp_all'
token = np.load(D+'/token.npy'); mode = np.load(D+'/mode.npy'); speed = np.load(D+'/speed.npy'); ab = np.load(D+'/angle_bin.npy')
odir5 = '/home/cvgluser/ros2_data/apt_g1/outputs/distill_v5'
odir6 = '/home/cvgluser/ros2_data/apt_g1/outputs/distill_v6'
os.makedirs(odir6, exist_ok=True)
meta5 = json.load(open(odir5+'/phase_meta.json'))

# best config per group (how, B)
BEST = {
    (17, -1.0, 4): ('median', 40),     # jump
    (1, 0.2, 5): ('mean', 64),         # turn_right 60deg
    (1, 0.2, 2): ('nearest', 40),      # turn_left / strafe_left
    (1, 0.2, 6): ('nearest', 40),      # strafe_right (v5 mean also ok)
}
def build_proto(gi, B, how):
    md = meta5[str(gi)]; g = tuple(md['group'])
    rows = np.where((mode == g[0]) & (np.abs(speed - g[1]) < 1e-6) & (ab == g[2]))[0]
    T = token[rows]; mu = np.array(md['mu'], dtype=np.float32); V2 = np.array(md['V2'], dtype=np.float32)
    proj = (T - mu) @ V2.T
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    bi = np.floor((phi + np.pi) / (2 * np.pi) * B).astype(int) % B
    proto = np.zeros((B, 64), dtype=np.float32)
    centers = np.linspace(-np.pi, np.pi, B, endpoint=False)
    for b in range(B):
        sel = np.where(bi == b)[0]
        if len(sel) == 0:
            proto[b] = proto[b-1] if b > 0 else T[0]
            continue
        if how == 'mean':
            proto[b] = T[sel].mean(0)
        elif how == 'median':
            proto[b] = np.median(T[sel], axis=0)
        else:
            d = np.abs((phi[sel] - centers[b] + np.pi) % (2*np.pi) - np.pi)
            proto[b] = T[sel[int(np.argmin(d))]]
    return np.clip(np.round(proto * 16) / 16, -1, 1).astype(np.float32)

for gi, md in meta5.items():
    gi = int(gi)
    shutil.copy(f'{odir5}/phase_g{gi}.pt', f'{odir6}/phase_g{gi}.pt')
    g = tuple(md['group'])
    how, B = BEST.get(g, ('mean', 40))
    proto = build_proto(gi, B, how)
    np.save(f'{odir6}/proto_g{gi}.npy', proto)
    print('group', g, '->', how, B, flush=True)
shutil.copy(f'{odir5}/phase_norm.npz', f'{odir6}/phase_norm.npz')
json.dump(meta5, open(f'{odir6}/phase_meta.json', 'w'), indent=1)
print('v6 ready')