"""Disc constants and configuration."""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

##
# MJCF paths.
##

_HERE = Path(__file__).parent
DISC_XML: Path = _HERE / "xmls" / "disc.xml"
assert DISC_XML.exists(), f"XML not found: {DISC_XML}"


##
# Spec functions.
##

def get_disc_spec() -> mujoco.MjSpec:
    """Load Disc MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(DISC_XML))


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Create mocap goal (orange disc matching disc) for visualization."""
    spec = mujoco.MjSpec()
    mocap_goal = spec.worldbody.add_body(name="mocap_goal")
    mocap_goal.mocap = True
    mocap_goal.pos = [0, 0, 0]
    mocap_goal.add_geom(
        name="mocap_goal_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.02, 0.02, 0.0],  # EXACT same as working cylinder (radius, half-height, unused)
        rgba=[1, 0.5, 0, 1],  # Orange (same as other mocap goals)
        contype=0,
        conaffinity=0,
    )
    return spec


##
# Entity configs.
##

def get_disc_cfg() -> EntityCfg:
    """Get a fresh disc configuration instance."""
    return EntityCfg(
        spec_fn=get_disc_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
