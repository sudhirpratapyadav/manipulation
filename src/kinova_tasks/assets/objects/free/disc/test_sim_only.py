"""Test just the full simulation loop (Test 7 only)."""

import torch
import mujoco

from mjlab.asset_zoo.objects.free.disc import get_disc_cfg, get_mocap_goal_cfg
from mjlab.asset_zoo.robots import get_franka_robot_cfg
from mjlab.terrains.terrain_importer import TerrainImporterCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg, MujocoCfg


def main():
    print("\n" + "="*60)
    print("TEST: Full sim loop with entity updates + CUDA graphs")
    print("="*60)

    scene_cfg = SceneCfg(
        num_envs=1,
        env_spacing=1.5,
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={
            "robot": get_franka_robot_cfg(),
            "disc": get_disc_cfg(),
            "mocap_goal": get_mocap_goal_cfg(),
        }
    )

    scene = Scene(scene_cfg, device="cuda:0")

    # Create simulation with EXACT task settings
    sim_cfg = SimulationCfg(
        nconmax=200,
        njmax=1000,
        mujoco=MujocoCfg(
            timestep=0.005,
            iterations=10,
            ls_iterations=20,
            impratio=10,
            cone="elliptic",
        ),
    )

    mj_model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=mj_model, device="cuda:0")
    scene.initialize(mj_model=sim.mj_model, model=sim.wp_model, data=sim.wp_data)

    disc_entity = scene.entities["disc"]

    print(f"  Disc root link id: {disc_entity.cfg.root_link_id}")

    # Set initial high position to simulate falling
    pose = torch.tensor([[0.7, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], device="cuda:0")
    velocity = torch.zeros((1, 6), device="cuda:0")

    print(f"  Setting disc to z=0.5 (will fall to ground)")
    disc_entity.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([0], device="cuda:0"))
    disc_entity.write_root_link_velocity_to_sim(velocity, env_ids=torch.tensor([0], device="cuda:0"))

    # Run full simulation loop (like task does)
    print(f"  Running simulation for 500 steps...")
    for step in range(500):
        # Step simulation (uses CUDA graph if available)
        sim.step()

        # Update entities (like task does)
        scene.update(dt=0.02)  # decimation=4, timestep=0.005 -> 0.02

        if step % 50 == 0:
            z = disc_entity.data.root_link_pos_w[0, 2].item()
            print(f"  Step {step:3d}: z = {z:.4f}")

    final_z = disc_entity.data.root_link_pos_w[0, 2].item()
    print(f"\n  Final z: {final_z:.4f}")

    if final_z < -0.1:
        print(f"❌ FAILED: Disc fell through ground!")
        return False
    else:
        print(f"✓ SUCCESS: Disc stayed on ground")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
