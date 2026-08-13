"""Isolation: walk_back 60s standalone; walk40->idle->jump sequence."""
import json, sys
import numpy as np
sys.path.insert(0, '/home/cvgluser/ros2_data')
sys.path.insert(0, '/home/cvgluser/ros2_data/apt_g1')
sys.path.insert(0, '/home/cvgluser/ros2_data/GR00T-WholeBodyControl')
from apt_g1.encoder import Command, PhaseRouterEncoder
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from eval_distill import NoQuantDecoder
from stress_test import make_env, run_episode

MODEL_DIR = '/home/cvgluser/ros2_data/apt_g1/outputs/distill_final'
encoder = PhaseRouterEncoder(MODEL_DIR)
out = {}
# walk_back 60s standalone
env = make_env(70.0)
cmd_back = Command.from_vxvy(-0.8, 0.0)
out['walk_back_60s'] = {}
for seed in [0, 1, 2]:
    r = run_episode(encoder, env, [(cmd_back, 60)], 3000, seed)
    out['walk_back_60s'][f'seed{seed}'] = r
    print(f'walk_back60 seed{seed} done={r["completed"]} h_min={r["h_min"]} vx={r["vx"]} disp={r["displacement"]}', flush=True)
# walk40 -> idle3 -> jump5
env = make_env(55.0)
cmd_walk = Command.from_vxvy(0.8, 0.0)
cmd_idle = Command.from_vxvy(0.0, 0.0)
cmd_jump = Command(mode=17, speed=-1.0, mdir=np.array([1.0, 0, 0], dtype=np.float32), fdir=np.array([1.0, 0, 0], dtype=np.float32))
sched = [(cmd_walk, 40), (cmd_idle, 3), (cmd_jump, 5)]
out['walk40_idle_jump'] = {}
for seed in [0, 1, 2]:
    r = run_episode(encoder, env, sched, 2400, seed)
    out['walk40_idle_jump'][f'seed{seed}'] = r
    print(f'walk40_idle_jump seed{seed} done={r["completed"]} fall={r["fall_step"]} h_min={r["h_min"]} disp={r["displacement"]}', flush=True)
json.dump(out, open(f'{MODEL_DIR}/stress_isolate.json', 'w'), indent=1)