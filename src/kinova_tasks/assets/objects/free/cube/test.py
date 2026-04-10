"""Cube constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg

##
# MJCF paths.
##

CUBE_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "cube" / "xmls" / "cube.xml"
)
assert CUBE_XML.exists(), f"XML not found: {CUBE_XML}"


def get_cube_spec() -> mujoco.MjSpec:
    """Load Cube MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(CUBE_XML))

# CUBE_INIT_STATE = EntityCfg.InitialStateCfg(
#     joint_pos={"cube_hinge": 0.785398},  # 45 degrees
#     joint_vel={".*": 0.0},
# )

def get_cube_cfg() -> EntityCfg:
    """Get a fresh cube configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues.
    """
    return EntityCfg(
        spec_fn=get_cube_spec,
        # init_state=CUBE_INIT_STATE,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity
    cube = Entity(get_cube_cfg())
    viewer.launch(cube.spec.compile())
