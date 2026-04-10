"""Cube constants and configuration."""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

##
# MJCF paths.
##

_HERE = Path(__file__).parent
CUBE_XML: Path = _HERE / "xmls" / "cube.xml"
assert CUBE_XML.exists(), f"XML not found: {CUBE_XML}"


##
# Spec functions.
##

def get_cube_spec() -> mujoco.MjSpec:
    """Load Cube MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(CUBE_XML))


##
# Entity configs.
##

def get_cube_cfg() -> EntityCfg:
    """Get a fresh cube configuration instance."""
    return EntityCfg(
        spec_fn=get_cube_spec,
    )
