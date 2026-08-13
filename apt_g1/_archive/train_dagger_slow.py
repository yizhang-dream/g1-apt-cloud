"""DAgger-lite for slow_fwd: student closed-loop states + kNN phase relabel, retrain slow phase net."""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, '/home/cvgluser/ros2_data')
sys.path.insert(0, '/home/cvgluser/ros2_data/apt_g1')
sys.path.insert(0, '/home/cvgluser/ros2_data/GR00T-WholeBodyControl')
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from eval_distill import NoQuantDecoder, hist_to_proprio

class PhaseNet(nn.Module):
    def __init__(self, d_in, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, 2))
    def forward(self, x): return self.net(x)

D = '/home/cvgluser/ros2_data/apt_g1/data/exp_all'
proprio = np.load(D+'/proprio.npy'); cmd = np.load(D+'/cmd.npy'); token = np.load(D+'/token.npy')
mode = np.load(D+'/mode.npy'); speed = np.load(D+'/speed.npy'); ab = np.load(D+'/angle_bin.npy')
modes_list = np.load(D+'/meta_modes.npy')
odir6 = '/home/cvgluser/ros2_data/apt_g1/outputs/distill_v6'
odir7 = '/home/cvgluser/ros2_data/apt_g1/outputs/distill_v7'
os.makedirs(odir7, exist_ok=True)
meta6 = json.load(open(odir6+'/phase_meta.json'))
norm = np.load(odir6+'/phase_norm.npz')
pmean = norm['pmean'].ravel(); pstd = norm['pstd'].ravel()
gmap = {tuple(md['group']): int(gi) for gi, md in meta6.items()}
gi_slow = gmap[(1, 0.2, 4)]
print('slow gi', gi_slow)

# official slow rows (exp1 phase1) + their PCA phase labels
slow_rows = np.where((mode == 1) & (np.abs(speed - 0.2) < 1e-6) & (ab == 4) & (np.arange(len(mode)) >= 1831) & (np.arange(len(mode)) < 8250))[0]
md = meta6[str(gi_slow)]
mu = np.array(md['mu'], dtype=np.float32); V2 = np.array(md['V2'], dtype=np.float32)
T = token[slow_rows]
proj = (T - mu) @ V2.T
phi_off = np.arctan2(proj[:, 1], proj[:, 0])
# official training X/labels
P_off = ((proprio[slow_rows] - pmean) / pstd).astype(np.float32)
X_off = np.concatenate([P_off, cmd[slow_rows]], axis=1).astype(np.float32)
y_off = np.stack([np.sin(phi_off), np.cos(phi_off)], axis=1).astype(np.float32)

# student rollout: collect (x, kNN phase label)
net = PhaseNet(930 + cmd.shape[1]).cuda()
net.load_state_dict(torch.load(f'{odir6}/phase_g{gi_slow}.pt', map_location='cuda'))
net.eval()
proto = np.load(f'{odir6}/proto_g{gi_slow}.npy')
env = MujocoG1FlatEnv(NoQuantDecoder('/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx'),
                      '/home/cvgluser/ros2_data/GR00T-WholeBodyControl', use_elastic_band=False, stand_only=True)
env.command = np.zeros(3, dtype=np.float32)
oh = np.zeros(len(modes_list), dtype=np.float32); oh[1] = 1.0
feat = np.concatenate([oh, np.array([1,0,0],np.float32), np.array([1,0,0],np.float32), np.array([0.2,-1,1],np.float32)]).astype(np.float32)
P_slow = ((proprio[slow_rows] - pmean) / pstd).astype(np.float32)

X_stu = []
y_stu = []
import mujoco
t0 = time.time()
for seed in [0, 1, 2]:
    env.reset(); rng = np.random.default_rng(100 + seed)
    env.data.qpos[2] += float(rng.normal(0, 0.005))
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(np.float32)
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco.mj_forward(env.model, env.data)
    env._reset_history(); env._fill_history_from_state()
    sc_prev = None
    for t in range(1000):
        prop = hist_to_proprio(env._get_sonic_history())
        x = np.concatenate([((prop - pmean) / pstd), feat]).astype(np.float32)
        # kNN phase label from official slow rows
        q = ((prop - pmean) / pstd).astype(np.float32)
        dist = np.einsum('ij,ij->i', P_slow, np.broadcast_to(q[None], P_slow.shape))
        nb = int(np.argmax(dist))
        phi_lab = float(phi_off[nb])
        X_stu.append(x); y_stu.append([np.sin(phi_lab), np.cos(phi_lab)])
        with torch.no_grad():
            sc = net(torch.from_numpy(x[None]).cuda())[0].cpu().numpy().astype(np.float32)
        if sc_prev is not None:
            sc = 0.3 * sc_prev + 0.7 * sc
        sc_prev = sc
        phi = float(np.arctan2(sc[0], sc[1]))
        b = int(np.floor((phi + np.pi) / (2 * np.pi) * 40) % 40)
        obs, reward, terminated, info = env.step({'token': proto[b], 'aux': np.zeros(12, dtype=np.float32)})
        if terminated:
            break
print('collected student samples:', len(X_stu), round(time.time()-t0, 1), 's')

X_all = np.concatenate([X_off, np.array(X_stu, dtype=np.float32)])
y_all = np.concatenate([y_off, np.array(y_stu, dtype=np.float32)])
print('train set', X_all.shape)
net2 = PhaseNet(930 + cmd.shape[1]).cuda()
opt = torch.optim.AdamW(net2.parameters(), lr=5e-4, weight_decay=1e-5)
lossf = nn.MSELoss()
ds = TensorDataset(torch.from_numpy(X_all), torch.from_numpy(y_all))
ld = DataLoader(ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
best = 1e9
for ep in range(60):
    net2.train(); tl = 0.0; tb = 0
    for x, yy in ld:
        x = x.cuda(non_blocking=True); yy = yy.cuda(non_blocking=True)
        opt.zero_grad(); loss = lossf(net2(x), yy); loss.backward(); opt.step()
        tl += loss.item() * len(yy); tb += len(yy)
    net2.eval()
    with torch.no_grad():
        pv = net2(torch.from_numpy(X_off[3200:]).cuda()).cpu().numpy()
    err = float(np.mean(1 - (pv[:, 0] * y_off[3200:, 0] + pv[:, 1] * y_off[3200:, 1])))
    if err < best:
        best = err
        torch.save(net2.state_dict(), f'{odir7}/phase_g{gi_slow}.pt')
    if (ep + 1) % 10 == 0:
        print(f'ep {ep+1} loss {tl/tb:.5f} val_ang_err {err:.4f}', flush=True)
print('best', best)
# copy remaining v6 artifacts
import shutil
for gi, md in meta6.items():
    gi = int(gi)
    if gi == gi_slow:
        continue
    shutil.copy(f'{odir6}/phase_g{gi}.pt', f'{odir7}/phase_g{gi}.pt')
    shutil.copy(f'{odir6}/proto_g{gi}.npy', f'{odir7}/proto_g{gi}.npy')
shutil.copy(f'{odir6}/proto_g{gi_slow}.npy', f'{odir7}/proto_g{gi_slow}.npy')
shutil.copy(f'{odir6}/phase_norm.npz', f'{odir7}/phase_norm.npz')
json.dump(meta6, open(f'{odir7}/phase_meta.json', 'w'), indent=1)
print('v7 ready')