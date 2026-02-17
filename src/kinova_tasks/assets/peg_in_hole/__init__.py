"""Peg-in-hole object configuration."""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

##
# MJCF paths.
##

_HERE = Path(__file__).parent
PEG_XML: Path = _HERE / "xmls" / "peg.xml"
HOLE_XML: Path = _HERE / "xmls" / "hole.xml"


##
# Spec functions.
##

def get_peg_spec() -> mujoco.MjSpec:
    """Load Peg MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(PEG_XML))


def get_hole_spec() -> mujoco.MjSpec:
    """Load Hole MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(HOLE_XML))


##
# Entity configs.
##

def get_peg_cfg() -> EntityCfg:
    """Get a fresh peg configuration instance."""
    return EntityCfg(spec_fn=get_peg_spec)


def get_hole_cfg() -> EntityCfg:
    """Get a fresh hole configuration instance."""
    return EntityCfg(spec_fn=get_hole_spec)
