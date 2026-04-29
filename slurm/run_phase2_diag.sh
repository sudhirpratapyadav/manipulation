#!/bin/bash
# Headless diagnosis of Phase 2 NaN at iter 0.
# Builds Phase 2 env at training-realistic num_envs=1024 on GPU and steps it
# 30 times with zero actions, reporting NaN counts each step.
set -euo pipefail
cd /ihub/homedirs/svs_ald/sudhir/manipulation
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
exec uv run python - <<'PY'
import torch, mjlab, kinova_tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv

cfg = load_env_cfg('Mjlab-Open-Drawer-Osc-Kinova-Phase2', play=False)
cfg.scene.num_envs = 1024
device = 'cuda:0'
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
obs, _ = env.reset()
n_envs = cfg.scene.num_envs
nan_envs_reset = int(torch.isnan(obs['actor']).any(dim=1).sum())
print(f'[diag] post-reset nan_envs (obs)={nan_envs_reset}/{n_envs}')

def nan_state(d):
    out = {}
    for fname in ['qpos','qvel','qacc','qacc_warmstart','sensordata']:
        try:
            t = getattr(d, fname)
            if t is None: continue
            n = int((torch.isnan(t) | torch.isinf(t)).any(dim=-1).sum())
            out[fname] = n
        except Exception as e:
            out[fname] = f'err:{e}'
    return out

print(f'[diag] post-reset physics state NaN counts: {nan_state(env.sim.data)}')

# Inspect the per-env model fields after the startup-mode DR has fired.
m = env.sim.model
print('[diag] post-reset model state samples (first 5 envs):')
print('  body_mass (drawer/drawer_base):',
      m.body_mass[:, env.scene._entities['drawer'].indexing.body_ids[1]][:5].tolist()
      if m.body_mass.ndim == 2 else 'not vectorized')
# arm-link mass after DR
robot_idx = env.scene._entities['robot'].indexing
arm_links = robot_idx.body_ids[:8]  # base..bracelet
if m.body_mass.ndim == 2:
    print('  arm_link masses (env 0):', m.body_mass[0, arm_links].tolist())
    print('  arm_link masses (env 1):', m.body_mass[1, arm_links].tolist())
    # check for inertia issues
    if hasattr(m, 'body_inertia'):
        bi = m.body_inertia
        if bi.ndim == 3:
            for ei in [0, 1]:
                vals = bi[ei, arm_links].cpu().numpy()
                negs = (vals < 0).sum()
                zeros = (vals == 0).sum()
                nans = float('nan') if not torch.isfinite(torch.tensor(vals)).all() else 0
                print(f'  env {ei} arm_link inertia: any<0={int(negs)} any==0={int(zeros)} nan={nans}')

# Step with zero actions
action_dim = env.action_manager.total_action_dim
zero = torch.zeros((n_envs, action_dim), device=device)
print(f'[diag] stepping with zeros, action_dim={action_dim} ...')
for i in range(30):
    out = env.step(zero)
    nan_envs = int(torch.isnan(out[0]['actor']).any(dim=1).sum())
    nan_rew = int(torch.isnan(out[1]).sum())
    rew_min = float(out[1].min())
    rew_max = float(out[1].max())
    if i < 5 or nan_envs > 0 or i % 10 == 0:
        print(f'  step {i:3d}  nan_envs={nan_envs:>4}/{n_envs}  nan_rew={nan_rew:>4}  rew=[{rew_min:.3f},{rew_max:.3f}]')

# Reproduce PPO iter-0 conditions: random actions sampled from a unit-std
# Gaussian, clamped to [-1, 1].
print(f'[diag] resetting and stepping with PPO-style random actions ...')
obs, _ = env.reset()
torch.manual_seed(0)
for i in range(60):
    act = torch.randn((n_envs, action_dim), device=device).clamp(-1.0, 1.0)
    out = env.step(act)
    nan_envs = int(torch.isnan(out[0]['actor']).any(dim=1).sum())
    nan_rew = int(torch.isnan(out[1]).sum())
    rew_min = float(out[1].min())
    rew_max = float(out[1].max())
    state_nans = nan_state(env.sim.data)
    print(f'  step {i:3d}  nan_envs={nan_envs:>4}/{n_envs}  nan_rew={nan_rew:>4}  rew=[{rew_min:.3f},{rew_max:.3f}]  state={state_nans}')
    if nan_envs == n_envs:
        print('[diag] all-envs-NaN reached; reproducing training condition.')
        break
PY
