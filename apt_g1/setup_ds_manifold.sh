#!/bin/bash
# ds_manifold setup: MuJoCo sim loop only; deploy spawned by drive_ds_manifold.py.
# Mirror of /tmp/setup_ds_smoke.sh with record paths -> /tmp/ds_manifold
# (DS_GAIT_MANIFOLD_PLAN Phase 1). Canonical copy: apt_g1/setup_ds_manifold.sh;
# deploy as /tmp/setup_ds_manifold.sh on lab-ts before driving.
set -e
cd /home/cvgluser/ros2_data/GR00T-WholeBodyControl
pkill -f 'run_sim_loop.py' 2>/dev/null || true
pkill -f 'g1_deploy_onnx_ref' 2>/dev/null || true
sleep 2
mkdir -p /tmp/ds_manifold/logs

if ! grep -q 'self.elastic_band = None' gear_sonic/utils/mujoco_sim/base_sim.py; then
  python3 /tmp/patch_base_sim.py
fi
if grep -q 'ENABLE_ELASTIC_BAND: True' gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_inspire.yaml; then
  python3 /tmp/patch_g1band.py
fi

cat > /tmp/g1sim_ds_manifold.sh <<'EOSIM'
#!/bin/bash
cd /home/cvgluser/ros2_data/GR00T-WholeBodyControl
source .venv_sim/bin/activate
source /opt/ros/humble/setup.bash
unset CYCLONEDDS_URI
if [ -d /home/cvgluser/cyclonedds/install ]; then
  export CYCLONEDDS_HOME=/home/cvgluser/cyclonedds/install
fi
export TensorRT_ROOT=/home/cvgluser/TensorRT
export onnxruntime_DIR=/home/cvgluser/.local/onnxruntime/lib/cmake/onnxruntime
export CMAKE_PREFIX_PATH=/home/cvgluser/.local/onnxruntime
export LD_LIBRARY_PATH=/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib:/home/cvgluser/.local/onnxruntime/lib:/home/cvgluser/TensorRT/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib
exec python gear_sonic/scripts/run_sim_loop.py --interface sim --wbc-version sonic_model12_inspire --no-enable-onscreen
EOSIM

cat > /tmp/ds_manifold/run_deploy.sh <<'EODEP'
#!/bin/bash
cd /home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy
source /opt/ros/humble/setup.bash
unset CYCLONEDDS_URI
export TensorRT_ROOT=/home/cvgluser/TensorRT
export onnxruntime_DIR=/home/cvgluser/.local/onnxruntime/lib/cmake/onnxruntime
export CMAKE_PREFIX_PATH=/home/cvgluser/.local/onnxruntime
export LD_LIBRARY_PATH=/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib:/home/cvgluser/.local/onnxruntime/lib:/home/cvgluser/TensorRT/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib
exec ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ --obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx --planner-file planner/target_vel/V2/planner_sonic.onnx --input-type keyboard --output-type all --zmq-host localhost --disable-crc-check --record-input-file /tmp/ds_manifold/commands.csv --policy-input-logfile /tmp/ds_manifold/policy_input.csv --target-motion-logfile /tmp/ds_manifold/target_motion.csv --logs-dir /tmp/ds_manifold/logs --enable-csv-logs
EODEP

chmod +x /tmp/g1sim_ds_manifold.sh /tmp/ds_manifold/run_deploy.sh
grep -n ENABLE_ELASTIC_BAND gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_inspire.yaml || true
nohup /tmp/g1sim_ds_manifold.sh > /tmp/ds_manifold/sim.log 2>&1 &
echo SIM_STARTED
sleep 12
ps aux | grep run_sim_loop | grep -v grep | head -2
tail -3 /tmp/ds_manifold/sim.log
