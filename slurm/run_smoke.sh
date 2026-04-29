#!/bin/bash
# Smoke test for the eval-sweep override plumbing. See
# src/kinova_tasks/eval_sweep_smoketest.py for what it checks.
set -euo pipefail
cd /ihub/homedirs/svs_ald/sudhir/manipulation
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
exec uv run python -m kinova_tasks.eval_sweep_smoketest
