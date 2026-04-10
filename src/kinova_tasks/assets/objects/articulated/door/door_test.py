"""Door constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg

##
# MJCF paths.
##

DOOR_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "door" / "xmls" / "door.xml"
)
assert DOOR_XML.exists(), f"XML not found: {DOOR_XML}"


def get_door_spec() -> mujoco.MjSpec:
    """Load Door MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(DOOR_XML))

DOOR_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"door_hinge": 0.785398},  # 45 degrees
    joint_vel={".*": 0.0},
)

def get_door_cfg() -> EntityCfg:
    """Get a fresh door configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues.
    """
    return EntityCfg(
        spec_fn=get_door_spec,
        init_state=DOOR_INIT_STATE,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity
    door = Entity(get_door_cfg())
    viewer.launch(door.spec.compile())
