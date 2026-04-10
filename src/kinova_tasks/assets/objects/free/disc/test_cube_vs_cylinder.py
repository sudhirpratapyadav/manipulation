"""Compare cube vs cylinder in actual task environment."""

import torch


def test_with_geometry(geom_type):
    """Test with specific geometry type."""
    print("\n" + "="*60)
    print(f"TEST: Running with {geom_type.upper()} geometry")
    print("="*60)

    # Temporarily modify disc XML
    import xml.etree.ElementTree as ET
    from pathlib import Path

    if geom_type == "cube":
        disc_xml_path = "/media/cvlab/EXTDRIVE/sudhir/continual_learning/mjlab/src/mjlab/asset_zoo/objects/free/cube/xmls/cube.xml"
    else:
        disc_xml_path = Path(__file__).parent / "xmls" / "disc.xml"

    # Read and parse XML
    # tree = ET.parse(disc_xml_path)
    # root = tree.getroot()

    # Find geom element
    # geom = root.find(".//geom[@name='disc_geom']")
    # original_type = geom.get('type')
    # original_size = geom.get('size')

    # # Modify geometry
    # if geom_type == "cylinder":
    #     geom.set('type', 'cylinder')
    #     geom.set('size', '0.02 0.02')  # radius, half-height
    # elif geom_type == "box":
    #     geom.set('type', 'box')
    #     geom.set('size', '0.02 0.02 0.02')  # half-extents

    # Write modified XML
    # tree.write(disc_xml_path)

    try:
        # Now test
        from mjlab.tasks.manipulation.config.franka.env_cfgs import franka_push_disc_env_cfg
        from mjlab.envs import ManagerBasedRlEnv

        # Force reload the module to pick up XML changes
        import importlib
        import mjlab.asset_zoo.objects.free.disc.disc_constants
        importlib.reload(mjlab.asset_zoo.objects.free.disc.disc_constants)

        # Create environment
        cfg = franka_push_disc_env_cfg(play=True)
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")

        print(f"  Environment created with {env.num_envs} environments")

        # Get disc entity
        disc = env.scene.entities["disc"]

        # Set disc to high position
        pose = torch.tensor([[0.7, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], device=env.device)
        velocity = torch.zeros((1, 6), device=env.device)

        print(f"  Setting disc to z=0.5 (will fall to ground)")
        disc.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.device))
        disc.write_root_link_velocity_to_sim(velocity, env_ids=torch.tensor([0], device=env.device))

        initial_z = disc.data.root_link_pos_w[0, 2].item()
        print(f"  Initial z: {initial_z:.4f}")

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

        # Clean up
        del env
        torch.cuda.empty_cache()

        if final_z < -0.1:
            print(f"  ❌ FAILED: {geom_type.upper()} fell through ground!")
            return False
        else:
            print(f"  ✓ SUCCESS: {geom_type.upper()} stayed on ground")
            return True

    finally:
        # Restore original XML
        geom.set('type', original_type)
        geom.set('size', original_size)
        tree.write(disc_xml_path)


def main():
    """Run tests with both geometries."""
    print("\n" + "="*60)
    print("COMPARISON TEST: Cube vs Cylinder in Full Task Environment")
    print("="*60)

    results = {}

    

    # # Small delay to ensure cleanup
    # import time
    # time.sleep(2)

    # Test 2: Cylinder
    try:
        print("\n" + "-"*60)
        print("Testing with CYLINDER geometry...")
        print("-"*60)
        results["cylinder"] = test_with_geometry("cylinder")
    except Exception as e:
        print(f"❌ ERROR with cylinder: {e}")
        import traceback
        traceback.print_exc()
        results["cylinder"] = False

    # Test 1: Box (cube)
    try:
        print("\n" + "-"*60)
        print("Testing with BOX geometry...")
        print("-"*60)
        results["box"] = test_with_geometry("box")
    except Exception as e:
        print(f"❌ ERROR with box: {e}")
        import traceback
        traceback.print_exc()
        results["box"] = False

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for geom_type, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {status}: {geom_type.upper()}")

    print("\n" + "="*60)
    if results.get("box") and not results.get("cylinder"):
        print("CONCLUSION: Cylinder-specific collision issue confirmed!")
        print("Cube works, but cylinder falls through ground in task env.")
    elif results.get("box") and results.get("cylinder"):
        print("CONCLUSION: Both work! Issue may have been fixed.")
    elif not results.get("box") and not results.get("cylinder"):
        print("CONCLUSION: Both fail! General collision issue.")
    else:
        print("CONCLUSION: Unexpected result pattern.")
    print("="*60)

    return results.get("box", False) or results.get("cylinder", False)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
