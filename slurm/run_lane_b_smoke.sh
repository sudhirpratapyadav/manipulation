#!/bin/bash
# Smoke test for the Lane B async sim driver. Uses an early checkpoint of
# the in-flight Phase 0 run; we only check the driver runs end-to-end
# without crashing — the success rate at iter 400 will be ~0 by design.
set -euo pipefail
cd /ihub/homedirs/svs_ald/sudhir/manipulation
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
exec uv run python nn_policy/sim_open_drawer_osc_async.py \
    --checkpoint logs/rsl_rl/open_drawer_osc_phase0/2026-04-29_00-25-49_baseline/model_400.pt \
    --output docs/results/_lane_b_smoke.json \
    --episodes-per-setting 1
