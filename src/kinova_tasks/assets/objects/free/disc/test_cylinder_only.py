"""Test ONLY cylinder in full task environment."""

import torch


def main():
    print("\n" + "="*60)
    print("TEST: Cylinder in Full Task Environment")
    print("="*60)

    # Make sure XML is cylinder
    import xml.etree.ElementTree as ET
    from pathlib import Path

    disc_xml_path = Path(__file__).parent / "xmls" / "disc.xml"
    tree = ET.parse(disc_xml_path)
    root = tree.getroot()
    geom = root.find(".//geom[@name='disc_geom']")

    print(f"  Current geom type: {geom.get('type')}")
    print(f"  Current geom size: {geom.get('size')}")

    # Force cylinder
    geom.set('type', 'cylinder')
    geom.set('size', '0.02 0.02')
    tree.write(disc_xml_path)

    print("  Set to cylinder geometry\n")

    from mjlab.tasks.manipulation.config.franka.env_cfgs import franka_push_disc_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    # Create environment
    cfg = franka_push_disc_env_cfg(play=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")

    print(f"  Environment created")

    # Get disc
    disc = env.scene.entities["disc"]

    # Check initial position
    print(f"  Initial disc z: {disc.data.root_link_pos_w[0, 2].item():.6f}")

    # Set disc to HIGH position (like in test that passed)
    pose = torch.tensor([[0.7, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], device=env.device)
    velocity = torch.zeros((1, 6), device=env.device)

    print(f"  Setting disc to z=0.5")
    disc.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.device))
    disc.write_root_link_velocity_to_sim(velocity, env_ids=torch.tensor([0], device=env.device))

    # Check if write worked
    z_after_write = disc.data.root_link_pos_w[0, 2].item()
    print(f"  Disc z after write: {z_after_write:.6f}")

    if abs(z_after_write - 0.3) < 0.01:
        print("  ⚠️  WARNING: Write didn't take effect! Still at initial position.")

    # Run simulation
    print(f"\n  Running 500 steps...")
    action_dim = env.action_manager.total_action_dim

    for step in range(500):
        action = torch.zeros((env.num_envs, action_dim), device=env.device)
        obs, rew, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            z = disc.data.root_link_pos_w[0, 2].item()
            print(f"  Step {step:3d}: z = {z:.6f}")

    final_z = disc.data.root_link_pos_w[0, 2].item()
    print(f"\n  Final z: {final_z:.6f}")

    if final_z < -0.1:
        print(f"\n❌ FAILED: Cylinder fell through ground!")
        return False
    else:
        print(f"\n✓ SUCCESS: Cylinder stayed on ground")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
