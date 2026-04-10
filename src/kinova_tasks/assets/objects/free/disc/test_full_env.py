"""Test the ACTUAL task environment."""

import torch

def main():
    print("\n" + "="*60)
    print("TEST: Running actual task environment")
    print("="*60)

    from mjlab.tasks.manipulation.config.franka.env_cfgs import franka_push_disc_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    # Create the actual task environment
    cfg = franka_push_disc_env_cfg(play=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")

    print(f"  Environment created with {env.num_envs} environments")
    print(f"  Scene entities: {list(env.scene.entities.keys())}")

    # Get disc entity
    disc = env.scene.entities["disc"]
    print(f"  Disc body name: {disc.body_names[0] if hasattr(disc, 'body_names') else 'disc'}")

    # Set disc to high position
    pose = torch.tensor([[0.7, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], device=env.device)
    velocity = torch.zeros((1, 6), device=env.device)

    print(f"\n  Setting disc to z=0.5 (will fall to ground)")
    disc.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.device))
    disc.write_root_link_velocity_to_sim(velocity, env_ids=torch.tensor([0], device=env.device))

    # Get initial position
    initial_z = disc.data.root_link_pos_w[0, 2].item()
    print(f"  Initial z: {initial_z:.4f}")

    # Run simulation for 500 steps (10 seconds at 0.02s control dt)
    print(f"\n  Running simulation...")
    action_dim = env.action_manager.total_action_dim
    print(f"  Action dimension: {action_dim}")
    for step in range(500):
        # Zero action (no robot movement)
        action = torch.zeros((env.num_envs, action_dim), device=env.device)
        obs, rew, terminated, truncated, info = env.step(action)

        if step % 50 == 0:
            z = disc.data.root_link_pos_w[0, 2].item()
            print(f"  Step {step:3d}: z = {z:.4f}")

    final_z = disc.data.root_link_pos_w[0, 2].item()
    print(f"\n  Final z: {final_z:.4f}")

    if final_z < -0.1:
        print(f"\n❌ FAILED: Disc fell through ground!")
        print(f"   This confirms the issue is in the full task environment.")
        return False
    else:
        print(f"\n✓ SUCCESS: Disc stayed on ground")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
