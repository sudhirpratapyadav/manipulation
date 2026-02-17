"""Common base environment configuration for Kinova Gen3 tasks."""

import mujoco

from kinova_tasks.assets.kinova_gen3 import get_kinova_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg


def get_cube_spec(cube_size: float = 0.025, mass: float = 0.06) -> mujoco.MjSpec:
    """Create a cube object specification."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_joint")
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(cube_size,) * 3,
        mass=mass,
        rgba=(0.2, 0.6, 0.9, 1.0),
    )
    return spec


def kinova_base_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create base Kinova Gen3 lift cube environment configuration.

    Sets up scene entities, fingertip friction, collision sensor, viewer,
    and play mode overrides. Does NOT configure actions (task-specific).

    Args:
        play: If True, configure for evaluation/play mode.

    Returns:
        Manager-based RL environment configuration.
    """
    cfg = make_lift_cube_env_cfg()

    # Scene entities
    cfg.scene.entities = {
        "robot": get_kinova_robot_cfg(),
        "cube": EntityCfg(spec_fn=get_cube_spec),
    }

    # EE site for observations and rewards
    cfg.observations["actor"].terms["ee_to_cube"].params[
        "asset_cfg"
    ].site_names = ("pinch_site",)
    cfg.rewards["lift"].params["asset_cfg"].site_names = ("pinch_site",)

    # Fingertip friction randomization (Robotiq 2F-85 pads)
    fingertip_geoms = r"(left|right)_pad[12]"
    cfg.events["fingertip_friction_slide"].params[
        "asset_cfg"
    ].geom_names = fingertip_geoms
    cfg.events["fingertip_friction_spin"].params[
        "asset_cfg"
    ].geom_names = fingertip_geoms
    cfg.events["fingertip_friction_roll"].params[
        "asset_cfg"
    ].geom_names = fingertip_geoms

    # Collision sensor for end-effector ground contact
    assert cfg.scene.sensors is not None
    for sensor in cfg.scene.sensors:
        if sensor.name == "ee_ground_collision":
            assert isinstance(sensor, ContactSensorCfg)
            sensor.primary.pattern = "bracelet_link"

    # Viewer
    cfg.viewer.body_name = "base_link"

    # Parallel environments
    cfg.scene.num_envs = 4096

    # Play mode overrides
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
        assert cfg.commands is not None
        cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

    return cfg
