#!/bin/bash
# Verify each registered phase task builds without error at small num_envs.
set -euo pipefail
cd /ihub/homedirs/svs_ald/sudhir/manipulation
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
exec uv run python -c "
import sys, traceback
import mjlab, kinova_tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
ids = [
    'Mjlab-Open-Drawer-Osc-Kinova',         # Phase 0 (sanity)
    'Mjlab-Open-Drawer-Osc-Kinova-Eval',
    'Mjlab-Open-Drawer-Osc-Kinova-Phase1',
    'Mjlab-Open-Drawer-Osc-Kinova-Phase2',
    'Mjlab-Open-Drawer-Osc-Kinova-Phase4',
]
fails = 0
for tid in ids:
    try:
        cfg = load_env_cfg(tid)
        cfg.scene.num_envs = 4
        env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
        ev = list(env.cfg.events.keys())
        ndr = sum(1 for k in ev if k.startswith('dr_'))
        delta = env.cfg.events['reset_robot_joints'].params.get('joint_delta_deg', '?')
        pose = env.cfg.events['reset_base'].params.get('pose_range', {})
        print(f'  [OK] {tid}: joint_delta_deg={delta} pose_range_keys={sorted(pose.keys())} dr_events={ndr}')
        env.close()
    except Exception as e:
        fails += 1
        print(f'  [FAIL] {tid}: {type(e).__name__}: {e}')
        traceback.print_exc()
print(f'phase-check: {len(ids)-fails}/{len(ids)} OK')
sys.exit(0 if fails == 0 else 1)
"
