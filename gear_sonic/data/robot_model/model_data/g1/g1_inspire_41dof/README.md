# G1 Inspire 41DoF asset

This asset is based on Unitree's official G1 29DoF robot with Inspire DFQ hands:

- Source repository: https://github.com/unitreerobotics/unitree_ros
- Source file: `robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf`
- Source commit used here: `d6f13aad60320ce1d60a07b82a76b5a553f2a0a9`

The URDF is copied as:

- `g1_inspire_41dof.urdf`

Only meshes referenced by this URDF are copied into `meshes/`.

The Inspire hand model contains mimic joints. The independent hand command/state
interface is 6 DoF per hand, while the URDF still includes additional mimic
joints for visualization and kinematics. Important mimic coefficients from the
URDF:

- `thumb_intermediate_joint = 1.6 * thumb_proximal_pitch_joint`
- `thumb_distal_joint = 2.4 * thumb_proximal_pitch_joint`
- each non-thumb `*_intermediate_joint = 1.0 * *_proximal_joint`
