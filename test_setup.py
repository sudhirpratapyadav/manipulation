#!/usr/bin/env python3
"""Quick test to verify Kinova lift task setup."""

import sys

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from kinova_lift import kinova_gen3
        from kinova_lift.env_cfgs import kinova_lift_cube_env_cfg
        from kinova_lift.rl_cfg import kinova_lift_cube_ppo_runner_cfg
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_task_registration():
    """Test that task is registered with mjlab."""
    print("\nTesting task registration...")
    try:
        # Simply check if we can import the registered task
        # The fact that list_envs shows it means it's registered
        import subprocess
        result = subprocess.run(
            ["uv", "run", "list_envs"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if "Mjlab-Lift-Cube-Kinova" in result.stdout:
            print("✓ Task registered successfully (verified with list_envs)")
            return True
        else:
            print("✗ Task not found in list_envs output")
            return False
    except Exception as e:
        print(f"✗ Registration test failed: {e}")
        return False

def test_env_creation():
    """Test that environment can be created."""
    print("\nTesting environment creation...")
    try:
        from kinova_lift.env_cfgs import kinova_lift_cube_env_cfg
        cfg = kinova_lift_cube_env_cfg()
        print(f"✓ Environment config created")
        print(f"  - Num envs: {cfg.scene.num_envs}")
        print(f"  - Episode length: {cfg.episode_length_s}s")
        print(f"  - Decimation: {cfg.decimation}")
        return True
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_robot_config():
    """Test robot configuration."""
    print("\nTesting robot configuration...")
    try:
        from kinova_lift.kinova_gen3 import get_kinova_robot_cfg, KINOVA_ACTION_SCALE
        robot_cfg = get_kinova_robot_cfg()
        spec = robot_cfg.spec_fn()
        model = spec.compile()
        print(f"✓ Robot model compiled successfully")
        print(f"  - DOF: {model.nv}")
        print(f"  - Bodies: {model.nbody}")
        print(f"  - Actuators: {model.nu}")
        print(f"  - Action scale: {KINOVA_ACTION_SCALE}")
        return True
    except Exception as e:
        print(f"✗ Robot config failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Kinova Lift Task Setup Test")
    print("="*60)

    tests = [
        test_imports,
        test_task_registration,
        test_env_creation,
        test_robot_config,
    ]

    results = [test() for test in tests]

    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)

    if all(results):
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Train: uv run train Mjlab-Lift-Cube-Kinova")
        print("  2. Play: uv run play Mjlab-Lift-Cube-Kinova --agent zero")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
