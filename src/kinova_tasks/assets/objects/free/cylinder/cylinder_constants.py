"""Cylinder constants and configuration."""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

##
# MJCF paths.
##

_HERE = Path(__file__).parent
CYLINDER_XML: Path = _HERE / "xmls" / "cylinder.xml"
assert CYLINDER_XML.exists(), f"XML not found: {CYLINDER_XML}"


##
# Spec functions.
##

def get_cylinder_spec() -> mujoco.MjSpec:
    """Load Cylinder MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(CYLINDER_XML))


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Create mocap goal (orange cylinder matching object) for visualization."""
    spec = mujoco.MjSpec()
    mocap_goal = spec.worldbody.add_body(name="mocap_goal")
    mocap_goal.mocap = True
    mocap_goal.pos = [0, 0, 0]
    mocap_goal.add_geom(
        name="mocap_goal_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.02, 0.02, 0.0],  # Matches cylinder size (radius, half-height, unused)
        rgba=[1, 0.5, 0, 1],  # Orange (same as other mocap goals)
        contype=0,
        conaffinity=0,
    )
    return spec


##
# Entity configs.
##

def get_cylinder_cfg() -> EntityCfg:
    """Get a fresh cylinder configuration instance."""
    return EntityCfg(
        spec_fn=get_cylinder_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
