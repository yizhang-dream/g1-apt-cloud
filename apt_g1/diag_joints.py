"""Diagnostic: print the G1 actuator->joint mapping (MuJoCo order) for the legs."""
import mujoco

REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
model = mujoco.MjModel.from_xml_path(
    REPO + "/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
)
data = mujoco.MjData(model)
print("nu =", model.nu)
for act_id in range(model.nu):
    joint_id = model.actuator_trnid[act_id, 0]
    name = model.joint(joint_id).name
    if "hand" in name:
        continue
    qposadr = model.jnt_qposadr[joint_id]
    dofadr = model.jnt_dofadr[joint_id]
    print(f"act {act_id:3d}  joint {joint_id:3d}  {name:28s} qposadr {qposadr:3d} dofadr {dofadr:3d}")
