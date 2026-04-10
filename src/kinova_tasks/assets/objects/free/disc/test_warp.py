"""Minimal test with mujoco_warp to debug cylinder-plane collision."""

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.asset_zoo.objects.free.disc import get_disc_spec
from mjlab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg


def main():
    # Create terrain with ground plane
    terrain_cfg = TerrainImporterCfg(terrain_type="plane", num_envs=1)
    terrain = TerrainImporter(terrain_cfg, device="cpu")

    # Get the spec with ground plane
    spec = terrain.spec

    # Create disc and attach it
    disc_spec = get_disc_spec()
    frame = spec.worldbody.add_frame(pos=(0, 0, 0.5))
    spec.attach(disc_spec, prefix="disc/", frame=frame)

    # Compile with standard MuJoCo first
    mj_model = spec.compile()

    # Apply TASK simulation settings to test if they cause the issue
    print("Applying TASK simulation settings...")
    mj_model.opt.timestep = 0.005  # Task uses 0.005 instead of 0.002
    mj_model.opt.iterations = 10  # Task uses 10 instead of 100
    mj_model.opt.impratio = 10  # Task uses 10 instead of 1.0
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC  # Task uses elliptic
    mj_model.opt.ls_iterations = 20

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    print(f"Model info:")
    print(f"  nconmax: {mj_model.nconmax}")
    print(f"  nbody: {mj_model.nbody}")
    print(f"  ngeom: {mj_model.ngeom}")

    # Create mujoco_warp model and data
    import warp as wp
    device = "cuda:0"
    wp_device = wp.get_device(device)

    with wp.ScopedDevice(wp_device):
        model = mjwarp.put_model(mj_model)
        data = mjwarp.put_data(mj_model, mj_data, nworld=1, nconmax=200, njmax=1000)

    print(f"\nRunning simulation with mujoco_warp on {device}...")
    qpos_np = data.qpos.numpy()
    print(f"Initial disc z position: {qpos_np[0, 2]:.4f}")

    # Run simulation for 2 seconds (200 steps at 0.01s timestep)
    with wp.ScopedDevice(wp_device):
        for step in range(2000):
            mjwarp.step(model, data)

            if step % 50 == 0:
                qpos_np = data.qpos.numpy()
                z_pos = qpos_np[0, 2]
                print(f"Step {step:3d}: disc z = {z_pos:.4f}")

        qpos_np = data.qpos.numpy()
        final_z = qpos_np[0, 2]
        print(f"\nFinal disc z position: {final_z:.4f}")

        if final_z < -0.1:
            print("❌ FAILED: Disc fell through ground!")
        else:
            print("✓ SUCCESS: Disc stayed on ground")


if __name__ == "__main__":
    main()
