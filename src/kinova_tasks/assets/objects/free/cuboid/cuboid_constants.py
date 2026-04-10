"""Cuboid constants and configuration."""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

##
# MJCF paths.
##

_HERE = Path(__file__).parent
CUBOID_XML: Path = _HERE / "xmls" / "cuboid.xml"
assert CUBOID_XML.exists(), f"XML not found: {CUBOID_XML}"


##
# Spec functions.
##

def get_cuboid_spec() -> mujoco.MjSpec:
    """Load Cuboid MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(CUBOID_XML))


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Create mocap goal (orange box matching cuboid) for visualization."""
    spec = mujoco.MjSpec()
    mocap_goal = spec.worldbody.add_body(name="mocap_goal")
    mocap_goal.mocap = True
    mocap_goal.pos = [0, 0, 0]
    mocap_goal.add_geom(
        name="mocap_goal_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.04, 0.04, 0.015],  # Matches cuboid size
        rgba=[1, 0.5, 0, 0.1],  # Orange (same as other mocap goals)
        contype=0,
        conaffinity=0,
    )
    return spec


##
# Entity configs.
##

def get_cuboid_cfg() -> EntityCfg:
    """Get a fresh cuboid configuration instance."""
    return EntityCfg(
        spec_fn=get_cuboid_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
