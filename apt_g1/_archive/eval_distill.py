"""Closed-loop no-band MuJoCo eval: learned token encoder vs oracle tokens vs zero token."""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/cvgluser/ros2_data')
sys.path.insert(0, '/home/cvgluser/ros2_data/apt_g1')
sys.path.insert(0, '/home/cvgluser/ros2_data/GR00T-WholeBodyControl')
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from apt_g1.sonic.sonic_wrapper import SonicOnnxDecoder
from train_distill import MLP

class NoQuantDecoder(SonicOnnxDecoder):
    def decode(self, token, history):
        obs = self.build_decoder_obs(token, history)
        return self.session.run([self.output_name], {self.input_name: obs})[0]

SCEN = [
    ('idle',        dict(mode=0,  speed=-1.0, mdir=[0, 0, 0],   fdir=[1, 0, 0])),
    ('slow_walk',   dict(mode=1,  speed=0.2,  mdir=[1, 0, 0],   fdir=[1, 0, 0])),
    ('walk',        dict(mode=2,  speed=-1.0, mdir=[1, 0, 0],   fdir=[1, 0, 0])),
    ('jump',        dict(mode=17, speed=-1.0, mdir=[1, 0, 0],   fdir=[1, 0, 0])),
    ('backward',    dict(mode=1,  speed=0.2,  mdir=[-1, 0, 0],  fdir=[1, 0, 0])),
]

def build_cmd_feature(c, modes):
    oh = np.zeros(len(modes), dtype=np.float32)
    oh[int(np.where(modes == c['mode'])[0][0])] = 1.0
    return np.concatenate([oh, np.array(c['mdir'], dtype=np.float32),
                           np.array(c['fdir'], dtype=np.float32),
                           np.array([c['speed'], -1.0, 1.0], dtype=np.float32)]).astype(np.float32)

def hist_to_proprio(hist):
    parts = [hist['base_angular_velocity'], hist['body_joint_positions'],
             hist['body_joint_velocities'], hist['last_actions'], hist['gravity_dir']]
    return np.concatenate([p.reshape(-1) for p in parts]).astype(np.float32)

def run_rollout(env, token_source, cmd_feat, steps, norm=None, model=None, oracle=None, seed=0, model_forward=None):
    env.reset()
    heights, vxs = [], []
    fall_step = None
    prev = np.zeros(64, dtype=np.float32)
    x0 = float(env.data.xpos[env.model.body('pelvis').id][0])
    for t in range(steps):
        hist = env._get_sonic_history()
        proprio = hist_to_proprio(hist)
        if token_source == 'learned':
            x = np.concatenate([(proprio - norm['pmean'].ravel()) / norm['pstd'].ravel(), cmd_feat]).astype(np.float32)
            if model_forward is not None:
                tok, prev = model_forward(x, prev)
            else:
                with torch.no_grad():
                    tok = model(torch.from_numpy(x[None]).cuda()).cpu().numpy()[0]
                tok = np.clip(np.round(tok * 16) / 16, -1.0, 1.0).astype(np.float32)
        elif token_source == 'zero':
            tok = np.zeros(64, dtype=np.float32)
        else:
            tok = oracle[t % len(oracle)]
        obs, reward, terminated, info = env.step({'token': tok, 'aux': np.zeros(12, dtype=np.float32)})
        heights.append(float(env.data.qpos[2]))
        vxs.append(float(env._get_base_linear_velocity()[0]))
        if terminated:
            fall_step = t
            break
    heights = np.array(heights); vxs = np.array(vxs)
    x_end = float(env.data.xpos[env.model.body('pelvis').id][0]) - x0
    return dict(fall_step=fall_step, height_mean=float(heights.mean()), height_min=float(heights.min()),
                vx_mean=float(vxs.mean()), height_last=float(heights[-1]), x_end=x_end)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--model-path', default='/home/cvgluser/ros2_data/apt_g1/outputs/distill/model_mlp.pt')
    ap.add_argument('--norm-path', default='/home/cvgluser/ros2_data/apt_g1/outputs/distill/norm_mlp.npz')
    ap.add_argument('--arch', default='mlp', choices=['mlp', 'deep', 'ar', 'ar_deep', 'ar_delta', 'tf_ar'])
    ap.add_argument('--ema', type=float, default=0.0)
    ap.add_argument('--out', default='/home/cvgluser/ros2_data/apt_g1/outputs/distill/eval_mlp.json')
    ap.add_argument('--sources', default='learned,oracle,zero')
    args = ap.parse_args()

    D = '/home/cvgluser/ros2_data/apt_g1/data/exp1'
    modes = np.load(D + '/meta_modes.npy')
    token = np.load(D + '/token.npy'); mode = np.load(D + '/mode.npy'); speed = np.load(D + '/speed.npy'); cmd = np.load(D + '/cmd.npy')

    norm = np.load(args.norm_path)
    model_forward = None
    if args.arch == 'mlp':
        from train_distill import MLP
        model = MLP(930 + 13).cuda()
        model.load_state_dict(torch.load(args.model_path, map_location='cuda'))
        model.eval()
    else:
        ema = args.ema
        if args.arch in ('ar_delta', 'tf_ar'):
            from train_distill3 import MLPBlock, TF
            pred_delta = True
            if args.arch == 'ar_delta':
                model = MLPBlock(930 + 13 + 64, 64, layers=3).cuda()
            else:
                model = TF(13, ar=True).cuda()
            model.load_state_dict(torch.load(args.model_path, map_location='cuda'))
            model.eval()
            def model_forward(x, prev):
                if args.arch == 'ar_delta':
                    xin = np.concatenate([x, prev]).astype(np.float32)
                    with torch.no_grad():
                        d = model(torch.from_numpy(xin[None]).cuda()).cpu().numpy()[0]
                else:
                    with torch.no_grad():
                        d = model(torch.from_numpy(x[:930].reshape(1, 10, 93)).cuda(),
                                  torch.from_numpy(x[930:][None]).cuda(),
                                  torch.from_numpy(prev[None]).cuda()).cpu().numpy()[0]
                raw = prev + d
                tok = np.clip(np.round(raw * 16) / 16, -1.0, 1.0).astype(np.float32)
                if ema > 0:
                    tok = (ema * prev + (1 - ema) * tok).astype(np.float32)
                return tok, tok
        else:
            from train_distill2 import MLPDeep
            use_ar = args.arch.startswith('ar')
            d_in = 930 + 13 + (64 if use_ar else 0)
            model = MLPDeep(d_in, layers=3 if 'deep' in args.arch else 2).cuda()
            model.load_state_dict(torch.load(args.model_path, map_location='cuda'))
            model.eval()
            def model_forward(x, prev):
                if use_ar:
                    xin = np.concatenate([x, prev]).astype(np.float32)
                else:
                    xin = x
                with torch.no_grad():
                    tok = model(torch.from_numpy(xin[None]).cuda()).cpu().numpy()[0]
                tok = np.clip(np.round(tok * 16) / 16, -1.0, 1.0).astype(np.float32)
                if ema > 0:
                    tok = (ema * prev + (1 - ema) * tok).astype(np.float32)
                return tok, tok

    repo = '/home/cvgluser/ros2_data/GR00T-WholeBodyControl'
    decoder = NoQuantDecoder(repo + '/gear_sonic_deploy/policy/release/model_decoder.onnx')
    env = MujocoG1FlatEnv(decoder, repo, use_elastic_band=False, stand_only=True)
    env.command = np.zeros(3, dtype=np.float32)

    oracle = {}
    mdir = cmd[:, 4:7]
    for name, c in SCEN:
        sel = (mode == c['mode']) & (np.abs(speed - c['speed']) < 1e-6)
        if name == 'backward':
            sel &= mdir[:, 0] < 0
        idx = np.where(sel)[0]
        oracle[name] = token[idx].astype(np.float32)

    results = {}
    for name, c in SCEN:
        cmd_feat = build_cmd_feature(c, modes)
        results[name] = {}
        for src in args.sources.split(','):
            t0 = time.time()
            r = run_rollout(env, src, cmd_feat, args.steps, norm=norm, model=model,
                            oracle=oracle.get(name), seed=0, model_forward=model_forward)
            r['n_oracle'] = int(len(oracle.get(name, [])))
            r['time_s'] = round(time.time() - t0, 1)
            results[name][src] = r
            print(f'{name:10s} {src:8s} fall={r["fall_step"]} h_mean={r["height_mean"]:.3f} '
                  f'h_min={r["height_min"]:.3f} vx={r["vx_mean"]:.3f} x_end={r["x_end"]:.2f} ({r["time_s"]}s)', flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, 'w'), indent=1)
    print('saved', args.out)

if __name__ == '__main__':
    main()