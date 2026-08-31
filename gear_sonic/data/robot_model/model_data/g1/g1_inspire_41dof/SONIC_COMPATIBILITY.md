# SONIC compatibility

## Files

- `g1_inspire_41dof.urdf`: original 41-DoF URDF.
- `g1_inspire_41dof.xml`: original generated MJCF. It has 53 actuators because
  every mimic joint is actuated independently.
- `scene.xml`: original MuJoCo scene.
- `g1_inspire_41dof_sonic.xml`: SONIC-adapted MJCF with 41 actuators.
- `scene_sonic.xml`: scene selected by the SONIC Inspire WBC configuration.

The adapted model retains all visual and kinematic joints, removes actuators
from the 12 URDF mimic joints, and restores their coupling with MuJoCo equality
constraints. Each Inspire hand therefore exposes six active joints in hardware
order: pinky, ring, middle, index, thumb pitch, thumb yaw.

## Launch

From the repository root:

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
  --interface sim \
  --wbc-version sonic_model12_inspire
```

The SONIC body policy remains 29 DoF. Start the PICO hybrid hand script with
`--hand-type inspire_dfx --inspire-interface sim`; it publishes the six-value
Inspire command independently of the fixed seven-joint Dex3 ZMQ fields. The
MuJoCo bridge subscribes to `rt/inspire/cmd` for this model configuration.
