#!/bin/bash
# Lane A OOD sweep on Phase 1 init-pose-DR checkpoint (model_3800).
# Run as job-step inside holder.
set -euo pipefail
cd /ihub/homedirs/svs_ald/sudhir/manipulation
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
exec uv run python -m kinova_tasks.eval_sweep \
    --checkpoint-file logs/rsl_rl/open_drawer_osc_phase1/2026-04-29_00-34-48_init_pose_dr/model_3800.pt \
    --output-dir docs/results/open_drawer_osc_phase1 \
    --num-envs 64 \
    --episodes-per-setting 64
