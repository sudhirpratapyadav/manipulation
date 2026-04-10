"""Exact copy of what test_cube_vs_cylinder does for cylinder."""

import torch
import xml.etree.ElementTree as ET
from pathlib import Path

disc_xml_path = Path(__file__).parent / "xmls" / "disc.xml"

# Read and parse XML
tree = ET.parse(disc_xml_path)
root = tree.getroot()
geom = root.find(".//geom[@name='disc_geom']")
original_type = geom.get('type')
original_size = geom.get('size')

# Set to cylinder
geom.set('type', 'cylinder')
geom.set('size', '0.02 0.02')
tree.write(disc_xml_path)

try:
    from mjlab.tasks.manipulation.config.franka.env_cfgs import franka_push_disc_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    
    # Force reload
    import importlib
    import mjlab.asset_zoo.objects.free.disc.disc_constants
    importlib.reload(mjlab.asset_zoo.objects.free.disc.disc_constants)
    
    cfg = franka_push_disc_env_cfg(play=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    
    disc = env.scene.entities["disc"]
    
    pose = torch.tensor([[0.7, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], device=env.device)
    velocity = torch.zeros((1, 6), device=env.device)
    
    disc.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.device))
    disc.write_root_link_velocity_to_sim(velocity, env_ids=torch.tensor([0], device=env.device))
    
    initial_z = disc.data.root_link_pos_w[0, 2].item()
    print(f"Initial z: {initial_z:.4f}")
    
    action_dim = env.action_manager.total_action_dim
    
    for step in range(500):
        action = torch.zeros((env.num_envs, action_dim), device=env.device)
        obs, rew, terminated, truncated, info = env.step(action)
        
        if step % 100 == 0:
            z = disc.data.root_link_pos_w[0, 2].item()
            print(f"Step {step:3d}: z = {z:.6f}")
    
    final_z = disc.data.root_link_pos_w[0, 2].item()
    print(f"Final z: {final_z:.6f}")
    
    del env
    torch.cuda.empty_cache()
    
    if final_z < -0.1:
        print("❌ FAILED")
    else:
        print("✓ SUCCESS")
        
finally:
    geom.set('type', original_type)
    geom.set('size', original_size)
    tree.write(disc_xml_path)
