"""Kinova Gen3 pick-and-lift task with OSC + RGB wrist camera observations.

Same task as pick_cube_osc.py but replaces privileged state observations
(ee_to_object, object_pos, object_to_goal) with RGB images from the wrist
and a static D455 camera.  Goal position is kept as a proprioceptive input
for goal conditioning.

The wrist camera is already defined in gen3_gripper_torque.xml on bracelet_link.
Camera name in MuJoCo: "robot/wrist".
"""

from __future__ import annotations

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp

from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import MaterialCfg, TextureCfg

from kinova_tasks.tasks.base_rl_cfg import kinova_ppo_runner_cfg
from kinova_tasks.tasks.pick_cube_osc import kinova_pick_cube_osc_env_cfg

_BROWN_TEXTURE = TextureCfg(
    name="groundplane",
    type="2d",
    builtin="flat",
    rgb1=(0.47, 0.32, 0.18),
    rgb2=(0.47, 0.32, 0.18),
    width=300,
    height=300,
)

_BROWN_MATERIAL = MaterialCfg(
    name="groundplane",
    texuniform=True,
    texrepeat=(4.0, 4.0),
    reflectance=0.1,
    texture="groundplane",
    geom_names_expr=("terrain$",),
)


_VISION_CNN_CFG = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
}
_VISION_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"

# Kinova wrist camera lives on bracelet_link — pos/quat already set in XML.
# The camera faces forward/downward toward the workspace, similar to YAM's D405.
_WRIST_CAM_NAME = "robot/wrist"

# Static D455 camera: centered laterally (x=0), in front of robot, low elevation.
# Quat aims the camera at workspace center (0, -0.5, 0.05).
_D455_POS = (0.0, -0.9, 0.3)
_D455_QUAT = (0.874642, 0.484769, 0.0, 0.0)  # w x y z, looking at (0, -0.5, 0.05)
_D455_FOVY = 58.0  # RealSense D455 vertical FoV


def kinova_pick_cube_vision_osc_env_cfg(play: bool = False):
    """Kinova Gen3 pick-and-lift with OSC + wrist cam + static D455 cam.

    Builds on kinova_pick_cube_osc_env_cfg and adds:
      - Wrist CameraSensor wrapping the existing robot/wrist camera in the XML
      - Static D455 CameraSensor attached to robot/base_link, framing the
        workspace (per-env replication via parent_body)
      - "camera" observation group with both RGB images concatenated
      - Object color randomization for visual diversity
      - Removes privileged state obs (ee_to_object, object_pos, object_to_goal)
    """
    cfg = kinova_pick_cube_osc_env_cfg(play=play)

    # --- Brown ground plane ---
    assert isinstance(cfg.scene.terrain, TerrainEntityCfg)
    cfg.scene.terrain.textures = (_BROWN_TEXTURE,)
    cfg.scene.terrain.materials = (_BROWN_MATERIAL,)

    # --- Add wrist camera sensor ---
    wrist_cam_cfg = CameraSensorCfg(
        name="wrist",
        camera_name=_WRIST_CAM_NAME,
        height=64,
        width=64,
        data_types=("rgb",),
        enabled_geom_groups=(0, 3),
        use_shadows=False,
        use_textures=True,
    )

    # --- Static D455 camera, attached to robot/base_link so each parallel
    # env gets its own copy framing its own workspace.  pos/quat are now
    # relative to base_link (which sits at the env origin per INIT_STATE),
    # so the same numerical values that worked when this was a worldbody
    # cam still frame the same workspace per-env.
    d455_cam_cfg = CameraSensorCfg(
        name="d455",
        camera_name=None,
        parent_body="robot/base_link",
        pos=_D455_POS,
        quat=_D455_QUAT,
        fovy=_D455_FOVY,
        height=64,
        width=64,
        data_types=("rgb",),
        enabled_geom_groups=(0, 3),
        use_shadows=False,
        use_textures=True,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (wrist_cam_cfg, d455_cam_cfg)

    # --- Camera observation group (wrist + d455 concatenated) ---
    camera_obs = ObservationGroupCfg(
        terms={
            "wrist_rgb": ObservationTermCfg(
                func=manipulation_mdp.camera_rgb,
                params={"sensor_name": "wrist"},
            ),
            "d455_rgb": ObservationTermCfg(
                func=manipulation_mdp.camera_rgb,
                params={"sensor_name": "d455"},
            ),
        },
        enable_corruption=False,
        concatenate_terms=True,
    )
    cfg.observations["camera"] = camera_obs

    # --- Object color randomization (visual diversity for RGB training) ---
    cfg.events["object_color"] = EventTermCfg(
        func=dr.geom_rgba,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
            "operation": "abs",
            "distribution": "uniform",
            "axes": [0, 1, 2],
            "ranges": (0.0, 1.0),
        },
    )

    # --- Remove privileged state from actor observations ---
    # Term names follow the upstream cube->object rename in pick_cube_osc.py.
    actor_obs = cfg.observations["actor"]
    removed = []
    for key in ("ee_to_object", "object_pos", "object_to_goal"):
        if actor_obs.terms.pop(key, None) is not None:
            removed.append(key)
    assert len(removed) == 3, (
        f"Expected to remove 3 privileged terms from actor obs, removed {removed}. "
        f"Did pick_cube_osc.py rename them? Available terms: "
        f"{list(actor_obs.terms.keys())}"
    )
    # goal_pos is kept — gives the policy the target without revealing object state.

    return cfg


def kinova_pick_cube_vision_osc_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for pick-cube vision OSC task (CNN + spatial softmax)."""
    base = kinova_ppo_runner_cfg(experiment_name="kinova_pick_cube_vision_osc")
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=_VISION_CNN_CFG,
            class_name=_VISION_MODEL_CLS,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=_VISION_CNN_CFG,
            class_name=_VISION_MODEL_CLS,
        ),
        algorithm=base.algorithm,
        experiment_name="kinova_pick_cube_vision_osc",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=3_000,
        obs_groups={
            "actor": ("actor", "camera"),
            "critic": ("critic", "camera"),
        },
    )
