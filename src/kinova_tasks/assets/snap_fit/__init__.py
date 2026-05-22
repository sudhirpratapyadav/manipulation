"""Snap-fit object configuration."""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

##
# MJCF paths.
##

_HERE = Path(__file__).parent
SNAP_PEG_XML: Path = _HERE / "xmls" / "snap_peg.xml"
SNAP_SOCKET_XML: Path = _HERE / "xmls" / "socket.xml"


##
# Spec functions.
##

def get_snap_peg_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(SNAP_PEG_XML))


def get_snap_socket_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(SNAP_SOCKET_XML))


##
# Entity configs.
##

def get_snap_peg_cfg() -> EntityCfg:
    return EntityCfg(spec_fn=get_snap_peg_spec)


def get_snap_socket_cfg() -> EntityCfg:
    return EntityCfg(spec_fn=get_snap_socket_spec)
