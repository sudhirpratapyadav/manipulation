#!/usr/bin/env python3
"""Test visualization of pinch_site."""

import mujoco
import mujoco.viewer
from kinova_lift.kinova_gen3 import get_kinova_robot_cfg

def main():
    print("Loading Kinova robot with pinch_site visualization...")

    # Load robot
    cfg = get_kinova_robot_cfg()
    spec = cfg.spec_fn()
    model = spec.compile()
    data = mujoco.MjData(model)

    # Check site properties
    pinch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pinch_site')
    print(f"\n✓ pinch_site configuration:")
    print(f"  - ID: {pinch_id}")
    print(f"  - Type: {model.site_type[pinch_id]} (2=ELLIPSOID)")
    print(f"  - Size: {model.site_size[pinch_id]}")
    print(f"  - RGBA: {model.site_rgba[pinch_id]} (red, semi-transparent)")
    print(f"  - Group: {model.site_group[pinch_id]}")

    print("\n📺 Launching viewer...")
    print("Look for a red semi-transparent sphere/ellipsoid at the gripper pinch point!")
    print("\nViewer controls:")
    print("  - In Viser: Check the 'Site' checkboxes in the left panel")
    print("  - In Native: Press 'V' to cycle through visualization groups")
    print("  - Use mouse to rotate/zoom the view")
    print("\nPress Ctrl+C to exit")

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Step simulation
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
