#!/bin/bash
# Lane A OOD sweep on Phase 0 baseline checkpoint (model_3800).
# Run as job-step inside holder.
set -euo pipefail
cd /ihub/homedirs/svs_ald/sudhir/manipulation
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
exec uv run python -m kinova_tasks.eval_sweep \
    --checkpoint-file logs/rsl_rl/open_drawer_osc_phase0/2026-04-29_00-25-49_baseline/model_3800.pt \
    --output-dir docs/results/open_drawer_osc_phase0 \
    --num-envs 64 \
    --episodes-per-setting 64
