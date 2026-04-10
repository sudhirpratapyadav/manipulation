"""Drawer constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg

##
# MJCF paths.
##

DOOR_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "drawer" / "xmls" / "drawer.xml"
)
assert DOOR_XML.exists(), f"XML not found: {DOOR_XML}"


def get_drawer_spec() -> mujoco.MjSpec:
    """Load Drawer MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(DOOR_XML))

DOOR_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"drawer_slide": 0.2}, 
    joint_vel={".*": 0.0},
)

def get_drawer_cfg() -> EntityCfg:
    """Get a fresh drawer configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues.
    """
    return EntityCfg(
        spec_fn=get_drawer_spec,
        init_state=DOOR_INIT_STATE,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity
    drawer = Entity(get_drawer_cfg())
    viewer.launch(drawer.spec.compile())
