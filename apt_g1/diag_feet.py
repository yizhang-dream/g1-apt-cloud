"""Diagnostic: check feet contact + CoM at the default reset pose (base z=0.76)."""
import mujoco
import numpy as np

REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
MODEL = REPO + "/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"

DEFAULT = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float32)

model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)
model.opt.timestep = 0.005

body_qpos_adr = []
for act_id in range(model.nu):
    joint_id = model.actuator_trnid[act_id, 0]
    name = model.joint(joint_id).name
    if "hand" in name:
        continue
    body_qpos_adr.append(model.jnt_qposadr[joint_id])
body_qpos_adr = np.asarray(body_qpos_adr, dtype=int)

mujoco.mj_resetData(model, data)
data.qpos[:3] = np.array([0.0, 0.0, 0.76])
data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
data.qpos[body_qpos_adr] = DEFAULT
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

# feet + base + com positions
for name in ["left_foot", "right_foot", "pelvis", "left_ankle", "right_ankle"]:
    try:
        bid = model.body(name).id
        print(f"{name:12s} pos = {data.xpos[bid]}")
    except Exception as e:
        print(f"{name:12s} NOT FOUND ({e})")

print("COM (subtree) =", data.subtree_com[model.body('pelvis').id])

# simulate 100 free steps (no control) and see if it falls
z0 = float(data.qpos[2])
zmin = z0
for _ in range(200):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)
    zmin = min(zmin, float(data.qpos[2]))
print(f"free-fall: base z0={z0:.3f} -> zmin={zmin:.3f} over 1s")
