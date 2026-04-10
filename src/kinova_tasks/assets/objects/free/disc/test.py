"""Disc constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg

##
# MJCF paths.
##

DISC_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "disc" / "xmls" / "disc.xml"
)
assert DISC_XML.exists(), f"XML not found: {DISC_XML}"


def get_disc_spec() -> mujoco.MjSpec:
    """Load Disc MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(DISC_XML))

# DISC_INIT_STATE = EntityCfg.InitialStateCfg(
#     joint_pos={"disc_hinge": 0.785398},  # 45 degrees
#     joint_vel={".*": 0.0},
# )

def get_disc_cfg() -> EntityCfg:
    """Get a fresh disc configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues.
    """
    return EntityCfg(
        spec_fn=get_disc_spec,
        # init_state=DISC_INIT_STATE,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity
    from mjlab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg

    # Create terrain with ground plane
    terrain_cfg = TerrainImporterCfg(terrain_type="plane", num_envs=1)
    terrain = TerrainImporter(terrain_cfg, device="cpu")

    # Get the spec with ground plane
    spec = terrain.spec

    # Create disc entity and add it to the spec
    disc = Entity(get_disc_cfg())

    # Attach disc to the worldbody with some initial position above ground
    frame = spec.worldbody.add_frame(pos=(0, 0, 0.5))
    spec.attach(disc.spec, prefix="disc/", frame=frame)

    viewer.launch(spec.compile())
