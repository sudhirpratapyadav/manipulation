"""Debug script: test cartesian task with zero actions."""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_from_matrix

# Load cartesian task config
cfg = load_env_cfg("Mjlab-Lift-Cube-Kinova-Cartesian", play=True)
cfg.scene.num_envs = 1

device = "cuda:0" if torch.cuda.is_available() else "cpu"
env = ManagerBasedRlEnv(cfg=cfg, device=device)

# Get action dim
action_dim = env.action_space.shape[1]
print(f"Action dim: {action_dim}")
print(f"Action terms: {list(env.action_manager._terms.keys())}")
for name, term in env.action_manager._terms.items():
    print(f"  {name}: dim={term.action_dim}, type={type(term).__name__}")

# Get IK action term for inspection
ik_term = env.action_manager._terms["ik_pose"]
print(f"\nHome pos: {ik_term._home_pos[0].cpu().numpy()}")
print(f"Home quat: {ik_term._home_quat[0].cpu().numpy()}")

# Reset
obs, _ = env.reset()
print("\n=== After reset ===")

# Get EE pose
robot = env.scene["robot"]
site_ids, _ = robot.find_sites("pinch_site")
site_id = robot.indexing.site_ids[site_ids[0]].item()

def print_state(step_i):
    ee_pos = env.sim.data.site_xpos[0, site_id].cpu().numpy()
    ee_quat = quat_from_matrix(env.sim.data.site_xmat[0, site_id]).cpu().numpy()
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    desired_pos = ik_term._desired_pos[0].cpu().numpy()
    desired_quat = ik_term._desired_quat[0].cpu().numpy()

    print(f"\n--- Step {step_i} ---")
    print(f"  Joint pos:     {joint_pos[:7]}")
    print(f"  Default pos:   {default_pos[:7]}")
    print(f"  EE pos:        {ee_pos}")
    print(f"  EE quat:       {ee_quat}")
    print(f"  Desired pos:   {desired_pos}")
    print(f"  Desired quat:  {desired_quat}")
    print(f"  Home pos:      {ik_term._home_pos[0].cpu().numpy()}")
    print(f"  Pos error:     {ee_pos - desired_pos}")

print_state(0)

# Step with zero actions
zero_action = torch.zeros(1, action_dim, device=device)
for i in range(1, 11):
    obs, rew, terminated, truncated, info = env.step(zero_action)
    print_state(i)

env.close()
