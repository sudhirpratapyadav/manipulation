"""Drawer constants and configuration."""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

##
# MJCF paths.
##

_HERE = Path(__file__).parent
DRAWER_XML: Path = _HERE / "xmls" / "drawer.xml"
assert DRAWER_XML.exists(), f"XML not found: {DRAWER_XML}"

MOCAP_TARGET_XML: Path = _HERE / "xmls" / "mocap_target.xml"


##
# Spec functions.
##

def get_drawer_spec() -> mujoco.MjSpec:
    """Load Drawer MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(DRAWER_XML))


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
            size=[0.01, 0.08, 0.01],  # Matches handle size
            rgba=[1, 0.5, 0, 1],
            contype=0,
            conaffinity=0,
        )
        return spec
    return mujoco.MjSpec.from_file(str(MOCAP_TARGET_XML))


##
# Initial state.
##

# Drawer starts at 0.0 (closed position)
DRAWER_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"drawer_slide": 0.0},  # Closed
    joint_vel={".*": 0.0},
)


##
# Entity configs.
##

def get_drawer_cfg() -> EntityCfg:
    """Get a fresh drawer configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues.
    """
    return EntityCfg(
        spec_fn=get_drawer_spec,
        init_state=DRAWER_INIT_STATE,
    )


def get_mocap_target_cfg() -> EntityCfg:
    """Get a fresh mocap target configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_target_spec,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    # Test drawer entity
    drawer = Entity(get_drawer_cfg())
    print(f"Drawer entity created with {len(drawer.joint_names)} joints: {drawer.joint_names}")

    viewer.launch(drawer.spec.compile())
