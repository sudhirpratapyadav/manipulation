"""Door constants and configuration."""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

##
# MJCF paths.
##

_HERE = Path(__file__).parent
DOOR_XML: Path = _HERE / "xmls" / "door.xml"
assert DOOR_XML.exists(), f"XML not found: {DOOR_XML}"

MOCAP_TARGET_XML: Path = _HERE / "xmls" / "mocap_target.xml"


##
# Spec functions.
##

def get_door_spec() -> mujoco.MjSpec:
    """Load Door MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(DOOR_XML))


def get_mocap_target_spec() -> mujoco.MjSpec:
    """Load Mocap Target MjSpec from XML."""
    if not MOCAP_TARGET_XML.exists():
        # Fallback: create mocap target programmatically
        spec = mujoco.MjSpec()
        mocap_target = spec.worldbody.add_body(name="mocap_target")
        mocap_target.mocap = True
        mocap_target.pos = [0, 0, 0]
        mocap_target.add_geom(
            name="mocap_target_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.01, 0.01, 0.08],
            rgba=[1, 0.5, 0, 1],
            contype=0,
            conaffinity=0,
        )
        return spec
    return mujoco.MjSpec.from_file(str(MOCAP_TARGET_XML))


##
# Initial state.
##

# Door starts at 45 degrees (0.785398 rad) to prevent drift to 0
DOOR_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"door_hinge": 0.785398},  # 45 degrees
    joint_vel={".*": 0.0},
)


##
# Entity configs.
##

def get_door_cfg() -> EntityCfg:
    """Get a fresh door configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues.
    """
    return EntityCfg(
        spec_fn=get_door_spec,
        init_state=DOOR_INIT_STATE,
    )


def get_mocap_target_cfg() -> EntityCfg:
    """Get a fresh mocap target configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_target_spec,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    # Test door entity
    door = Entity(get_door_cfg())
    print(f"Door entity created with {len(door.joint_names)} joints: {door.joint_names}")

    viewer.launch(door.spec.compile())
