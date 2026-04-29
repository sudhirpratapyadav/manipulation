#!/bin/bash
# End-to-end check that eval_sweep numbers match a hand-computed ground truth.
set -euo pipefail
cd /ihub/homedirs/svs_ald/sudhir/manipulation
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
exec uv run python -m kinova_tasks.eval_sweep_handcheck \
    --checkpoint logs/rsl_rl/open_drawer_osc_phase0/2026-04-29_00-25-49_baseline/model_400.pt \
    --num-envs 16 --episodes 32 --axis goal_depth
