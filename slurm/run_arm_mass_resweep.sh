#!/bin/bash
# Re-sweep arm_link_mass_pct only, on all three phase checkpoints, after
# fixing the regex bug that NaN'd the original sweeps.
set -euo pipefail
cd /ihub/homedirs/svs_ald/sudhir/manipulation
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

P0_CKPT=logs/rsl_rl/open_drawer_osc_phase0/2026-04-29_00-25-49_baseline/model_3800.pt
P1_CKPT=logs/rsl_rl/open_drawer_osc_phase1/2026-04-29_00-34-50_init_pose_dr/model_3800.pt
P2_CKPT=logs/rsl_rl/open_drawer_osc_phase2/2026-04-29_08-55-14_drawer_dr/model_2700.pt

for tag in p0 p1 p2; do
  case "$tag" in
    p0) ckpt=$P0_CKPT ;;
    p1) ckpt=$P1_CKPT ;;
    p2) ckpt=$P2_CKPT ;;
  esac
  echo "=== arm_mass resweep on $tag: $ckpt ==="
  uv run python -m kinova_tasks.eval_sweep \
      --checkpoint-file "$ckpt" \
      --output-dir "docs/results/arm_mass_resweep_${tag}" \
      --num-envs 64 \
      --episodes-per-setting 64 \
      --only-axes arm_link_mass_pct
done
