"""Incremental test to find what breaks cylinder collision."""

import mujoco
import mujoco_warp as mjwarp
import warp as wp

from mjlab.asset_zoo.objects.free.disc import get_disc_cfg, get_disc_spec
from mjlab.asset_zoo.robots import get_franka_robot_cfg
from mjlab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg
from mjlab.scene import Scene, SceneCfg


def test_disc_only():
    """Test 1: Just disc + ground (baseline - should work)"""
    print("\n" + "="*60)
    print("TEST 1: Disc + Ground only")
    print("="*60)

    terrain_cfg = TerrainImporterCfg(terrain_type="plane", num_envs=1)
    terrain = TerrainImporter(terrain_cfg, device="cpu")
    spec = terrain.spec

    disc_spec = get_disc_spec()
    frame = spec.worldbody.add_frame(pos=(0, 0, 0.5))
    spec.attach(disc_spec, prefix="disc/", frame=frame)

    return run_simulation(spec, "disc/disc", test_name="Disc only")


def test_disc_with_robot():
    """Test 2: Disc + Ground + Franka robot"""
    print("\n" + "="*60)
    print("TEST 2: Disc + Ground + Franka")
    print("="*60)

    scene_cfg = SceneCfg(
        num_envs=1,
        env_spacing=1.5,
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={
            "robot": get_franka_robot_cfg(),
            "disc": get_disc_cfg(),
        }
    )

    scene = Scene(scene_cfg, device="cpu")
    return run_simulation(scene.spec, "disc/disc", test_name="Disc + Robot")


def test_disc_with_robot_and_mocap():
    """Test 3: Disc + Ground + Franka + Mocap goal"""
    print("\n" + "="*60)
    print("TEST 3: Disc + Ground + Franka + Mocap")
    print("="*60)

    from mjlab.asset_zoo.objects.free.disc import get_mocap_goal_cfg

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

    scene = Scene(scene_cfg, device="cpu")
    return run_simulation(scene.spec, "disc/disc", test_name="Disc + Robot + Mocap")


def test_multi_env():
    """Test 4: Multiple environments (4 envs)"""
    print("\n" + "="*60)
    print("TEST 4: 4 Environments")
    print("="*60)

    from mjlab.asset_zoo.objects.free.disc import get_mocap_goal_cfg

    scene_cfg = SceneCfg(
        num_envs=4,
        env_spacing=1.5,
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={
            "robot": get_franka_robot_cfg(),
            "disc": get_disc_cfg(),
            "mocap_goal": get_mocap_goal_cfg(),
        }
    )

    scene = Scene(scene_cfg, device="cpu")
    return run_simulation(scene.spec, "disc/disc", test_name="4 Environments", num_envs=4)


def test_with_sensors():
    """Test 5: Add contact sensors"""
    print("\n" + "="*60)
    print("TEST 5: With contact sensors")
    print("="*60)

    from mjlab.asset_zoo.objects.free.disc import get_mocap_goal_cfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    ee_ground_collision_cfg = ContactSensorCfg(
        name="ee_ground_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern="link7",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    scene_cfg = SceneCfg(
        num_envs=1,
        env_spacing=1.5,
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={
            "robot": get_franka_robot_cfg(),
            "disc": get_disc_cfg(),
            "mocap_goal": get_mocap_goal_cfg(),
        },
        sensors=(ee_ground_collision_cfg,),
    )

    scene = Scene(scene_cfg, device="cpu")
    return run_simulation(scene.spec, "disc/disc", test_name="With Sensors")


def run_simulation(spec, disc_body_name, test_name="Test", num_envs=1):
    """Run simulation and check if disc falls through ground."""
    # Compile model
    mj_model = spec.compile()

    # Apply task simulation settings
    mj_model.opt.timestep = 0.005
    mj_model.opt.iterations = 10
    mj_model.opt.impratio = 10
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    mj_model.opt.ls_iterations = 20

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    # Find disc body
    disc_body_id = None
    for i in range(mj_model.nbody):
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and disc_body_name in body_name:
            disc_body_id = i
            break

    if disc_body_id is None:
        print(f"❌ ERROR: Could not find disc body '{disc_body_name}'")
        return False

    # Get qpos offset for disc
    disc_qpos_adr = mj_model.body_jntadr[disc_body_id]
    if disc_qpos_adr < 0:
        print(f"❌ ERROR: Disc has no joint")
        return False

    disc_z_idx = mj_model.jnt_qposadr[disc_qpos_adr] + 2  # +2 for z position

    print(f"  Model: {mj_model.nbody} bodies, {mj_model.ngeom} geoms")
    print(f"  Disc body id: {disc_body_id}, z index: {disc_z_idx}")

    # Create mujoco_warp model and data
    device = "cuda:0"
    wp_device = wp.get_device(device)

    with wp.ScopedDevice(wp_device):
        model = mjwarp.put_model(mj_model)
        data = mjwarp.put_data(mj_model, mj_data, nworld=num_envs, nconmax=200, njmax=1000)

    # Run simulation
    qpos_np = data.qpos.numpy()
    initial_z = qpos_np[0, disc_z_idx]
    print(f"  Initial z: {initial_z:.4f}")

    with wp.ScopedDevice(wp_device):
        # Run for 400 steps (2 seconds at 0.005 timestep)
        for step in range(400):
            mjwarp.step(model, data)

        qpos_np = data.qpos.numpy()
        final_z = qpos_np[0, disc_z_idx]

    print(f"  Final z:   {final_z:.4f}")

    # Check all environments if multi-env
    if num_envs > 1:
        for env_id in range(num_envs):
            env_z = qpos_np[env_id, disc_z_idx]
            if env_z < -0.1:
                print(f"  ❌ Env {env_id}: z={env_z:.4f} - FELL THROUGH!")
                return False

    if final_z < -0.1:
        print(f"❌ FAILED: Disc fell through ground!")
        return False
    else:
        print(f"✓ SUCCESS: Disc stayed on ground")
        return True


def test_with_pose_reset():
    """Test 6: Simulate command system pose reset"""
    print("\n" + "="*60)
    print("TEST 6: With pose reset (simulating command)")
    print("="*60)

    import torch
    from mjlab.asset_zoo.objects.free.disc import get_mocap_goal_cfg
    from mjlab.entity import Entity

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
    mj_model = scene.compile()

    # Apply task simulation settings
    mj_model.opt.timestep = 0.005
    mj_model.opt.iterations = 10
    mj_model.opt.impratio = 10
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    mj_model.opt.ls_iterations = 20

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    # Create warp model
    device = "cuda:0"
    wp_device = wp.get_device(device)

    with wp.ScopedDevice(wp_device):
        model = mjwarp.put_model(mj_model)
        data = mjwarp.put_data(mj_model, mj_data, nworld=1, nconmax=200, njmax=1000)

    # Initialize entities
    disc_entity = scene.entities["disc"]
    disc_entity.initialize(mj_model, model, data, device)

    # Simulate command reset: set disc pose like PushingCommand does
    pose = torch.tensor([[0.7, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]], device=device)  # x, y, z, qw, qx, qy, qz
    velocity = torch.zeros((1, 6), device=device)

    print(f"  Resetting disc to z=0.05 (like command system)")
    disc_entity.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([0], device=device))
    disc_entity.write_root_link_velocity_to_sim(velocity, env_ids=torch.tensor([0], device=device))

    # Read back position using numpy
    xpos_np = data.xpos.numpy()
    initial_z = xpos_np[0, disc_entity.cfg.root_link_id, 2]
    print(f"  Initial z after reset: {initial_z:.4f}")

    # Run simulation
    with wp.ScopedDevice(wp_device):
        for step in range(400):
            mjwarp.step(model, data)
            if step % 100 == 0:
                xpos_np = data.xpos.numpy()
                z = xpos_np[0, disc_entity.cfg.root_link_id, 2]
                print(f"  Step {step}: z = {z:.4f}")

        xpos_np = data.xpos.numpy()
        final_z = xpos_np[0, disc_entity.cfg.root_link_id, 2]

    print(f"  Final z: {final_z:.4f}")

    if final_z < -0.1:
        print(f"❌ FAILED: Disc fell through ground!")
        return False
    else:
        print(f"✓ SUCCESS: Disc stayed on ground")
        return True


def test_with_full_sim():
    """Test 7: Full simulation loop with entity updates (like actual task)"""
    print("\n" + "="*60)
    print("TEST 7: Full sim loop with entity updates + CUDA graphs")
    print("="*60)

    import torch
    from mjlab.asset_zoo.objects.free.disc import get_mocap_goal_cfg
    from mjlab.sim import Simulation, SimulationCfg, MujocoCfg

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
    scene.initialize(mj_model, sim.model, sim.data, "cuda:0")

    disc_entity = scene.entities["disc"]

    # Set initial high position to simulate falling
    pose = torch.tensor([[0.7, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], device="cuda:0")
    velocity = torch.zeros((1, 6), device="cuda:0")
    disc_entity.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([0], device="cuda:0"))
    disc_entity.write_root_link_velocity_to_sim(velocity, env_ids=torch.tensor([0], device="cuda:0"))

    print(f"  Starting disc at z=0.5 (will fall to ground)")

    # Run full simulation loop (like task does)
    for step in range(500):
        # Step simulation (uses CUDA graph if available)
        sim.step()

        # Update entities (like task does)
        scene.update(dt=0.02)  # decimation=4, timestep=0.005 -> 0.02

        if step % 50 == 0:
            disc_entity.data.root_link_pos_w.copy_(sim.data.xpos[:, disc_entity.cfg.root_link_id, :])
            z = disc_entity.data.root_link_pos_w[0, 2].item()
            print(f"  Step {step}: z = {z:.4f}")

    disc_entity.data.root_link_pos_w.copy_(sim.data.xpos[:, disc_entity.cfg.root_link_id, :])
    final_z = disc_entity.data.root_link_pos_w[0, 2].item()
    print(f"  Final z: {final_z:.4f}")

    if final_z < -0.1:
        print(f"❌ FAILED: Disc fell through ground!")
        return False
    else:
        print(f"✓ SUCCESS: Disc stayed on ground")
        return True


def main():
    """Run all tests in sequence."""
    print("\n" + "="*60)
    print("INCREMENTAL TEST: Finding what breaks cylinder collision")
    print("="*60)

    tests = [
        ("Disc only", test_disc_only),
        ("Disc + Robot", test_disc_with_robot),
        ("Disc + Robot + Mocap", test_disc_with_robot_and_mocap),
        ("Multi-env (4)", test_multi_env),
        ("With Sensors", test_with_sensors),
        ("With pose reset", test_with_pose_reset),
        ("Full sim loop", test_with_full_sim),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
            if not success:
                print(f"\n⚠️  PROBLEM FOUND at: {test_name}")
                print("This is where cylinder collision breaks!")
                break
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
            break

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")


if __name__ == "__main__":
    main()
