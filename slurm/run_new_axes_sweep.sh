#!/bin/bash
# Sweep the 8 new axes (perception/action/dynamics noise + impulses) on
# P0 + P1 checkpoints. Run from inside the holder via srun.
set -euo pipefail
cd /ihub/homedirs/svs_ald/sudhir/manipulation
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

P0_CKPT=logs/rsl_rl/open_drawer_osc_phase0/2026-04-29_00-25-49_baseline/model_3800.pt
P1_CKPT=logs/rsl_rl/open_drawer_osc_phase1/2026-04-29_00-34-48_init_pose_dr/model_3800.pt

NEW_AXES='("obs_noise_object_m","obs_noise_ee","action_noise_pct","gravity_scale","torque_offset_Nm","torque_noise_Nm","drawer_impulse","ee_impulse")'

for tag in p0 p1; do
  case "$tag" in
    p0) ckpt=$P0_CKPT ;;
    p1) ckpt=$P1_CKPT ;;
  esac
  echo "=== new-axes sweep on $tag: $ckpt ==="
  uv run python -m kinova_tasks.eval_sweep \
      --checkpoint-file "$ckpt" \
      --output-dir "docs/results/new_axes_${tag}" \
      --num-envs 64 \
      --episodes-per-setting 64 \
      --only-axes "$NEW_AXES"
done
