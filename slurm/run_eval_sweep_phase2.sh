#!/bin/bash
# Lane A OOD sweep on Phase 2 drawer-DR checkpoint.
set -euo pipefail
cd /ihub/homedirs/svs_ald/sudhir/manipulation
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
exec uv run python -m kinova_tasks.eval_sweep \
    --checkpoint-file logs/rsl_rl/open_drawer_osc_phase2/2026-04-29_08-55-14_drawer_dr/model_2700.pt \
    --output-dir docs/results/open_drawer_osc_phase2 \
    --num-envs 64 \
    --episodes-per-setting 64
