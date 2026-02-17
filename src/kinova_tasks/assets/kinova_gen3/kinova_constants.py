"""Kinova Gen3 robot constants and configuration for lift task."""

from pathlib import Path
import mujoco

from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

_HERE = Path(__file__).parent
KINOVA_GEN3_GRIPPER_XML: Path = _HERE / "xmls" / "gen3_gripper.xml"


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Load Kinova Gen3 mesh assets."""
    assets: dict[str, bytes] = {}
    update_assets(assets, KINOVA_GEN3_GRIPPER_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    """Load Kinova Gen3 with Robotiq 2F-85 gripper for position control.

    Includes the 7-DOF arm with position actuators plus the parallel gripper mechanism.
    """
    spec = mujoco.MjSpec.from_file(str(KINOVA_GEN3_GRIPPER_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Joint names.
##

ARM_JOINTS = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]

GRIPPER_JOINTS = [
    "right_driver_joint",
    "right_coupler_joint",
    "right_spring_link_joint",
    "right_follower_joint",
    "left_driver_joint",
    "left_coupler_joint",
    "left_spring_link_joint",
    "left_follower_joint",
]

##
# Initial state for lift task.
##

# Initial state with gripper open ready to grasp
# Driver joints at lower value for open position (range is 0-0.8, 0=open, 0.8=fully closed)
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        # Arm joints - ready pose for lifting
        "joint_1": 0.0,               # 0°
        "joint_2": 0.3490658504,      # 20°
        "joint_3": 0.0,               # 0°
        "joint_4": 1.7453292519,      # 100°
        "joint_5": 0.0,               # 0°
        "joint_6": -0.5235987756,     # -30°
        "joint_7": -1.5707963268,     # -90°
        # Gripper joints - open position
        "right_driver_joint": 0.0,
        "left_driver_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

# Initial state with gripper closed (for tasks where robot holds an object)
INIT_STATE_GRIPPER_CLOSED = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        # Arm joints - same ready pose
        "joint_1": 0.0,               # 0°
        "joint_2": 0.3490658504,      # 20°
        "joint_3": 0.0,               # 0°
        "joint_4": 1.7453292519,      # 100°
        "joint_5": 0.0,               # 0°
        "joint_6": -0.5235987756,     # -30°
        "joint_7": -1.5707963268,     # -90°
        # Gripper joints - closed position (0.8 = fully closed)
        "right_driver_joint": 0.8,
        "left_driver_joint": 0.8,
    },
    joint_vel={".*": 0.0},
)

# Initial state for peg-in-hole task (from old mjlab)
# Arm rotated 90° at joint_1, gripper fully closed (all 8 joints consistent
# with 4-bar linkage equality constraints at ctrl=255 equilibrium).
INIT_STATE_PEGINHOLE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        # Arm joints
        "joint_1": 1.5707963268,      # 90°
        "joint_2": 0.5235987756,      # 30°
        "joint_3": 0.0,               # 0°
        "joint_4": 1.5707963268,      # 90°
        "joint_5": 0.0,               # 0°
        "joint_6": 1.0471975512,      # 60°
        "joint_7": -1.5707963268,     # -90°
        # Gripper joints - 80% closed (consistent 4-bar linkage state)
        "right_driver_joint": 0.671,
        "right_coupler_joint": 0.001,
        "right_spring_link_joint": 0.673,
        "right_follower_joint": -0.646,
        "left_driver_joint": 0.671,
        "left_coupler_joint": 0.001,
        "left_spring_link_joint": 0.673,
        "left_follower_joint": -0.646,
    },
    joint_vel={".*": 0.0},
)

##
# Articulation config.
##

# XmlPositionActuatorCfg automatically finds and uses actuators defined in the XML
KINOVA_ACTUATORS = XmlPositionActuatorCfg(
    target_names_expr=(".*",),  # Match all joints (arm + gripper)
)

KINOVA_GRIPPER_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(KINOVA_ACTUATORS,),
    soft_joint_pos_limit_factor=0.9,
)


def get_kinova_robot_cfg() -> EntityCfg:
    """Get a fresh Kinova Gen3 robot configuration instance for lift task.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(),  # Use collisions from XML
        spec_fn=get_spec,
        articulation=KINOVA_GRIPPER_ARTICULATION,
    )


def get_kinova_robot_cfg_closed_gripper() -> EntityCfg:
    """Get Kinova Gen3 config with gripper closed (for holding objects).

    The gripper starts closed and the fingers_actuator ctrl defaults to 255
    (closed) via the init state. Used for tasks like peg-in-hole where the
    robot holds an object throughout the episode.
    """
    return EntityCfg(
        init_state=INIT_STATE_GRIPPER_CLOSED,
        collisions=(),
        spec_fn=get_spec,
        articulation=KINOVA_GRIPPER_ARTICULATION,
    )


def get_kinova_robot_cfg_peginhole() -> EntityCfg:
    """Get Kinova Gen3 config for peg-in-hole task.

    Arm is rotated 90° at joint_1 to face the workspace side.
    Gripper is closed (driver joints at 1.0) to hold the peg.
    """
    return EntityCfg(
        init_state=INIT_STATE_PEGINHOLE,
        collisions=(),
        spec_fn=get_spec,
        articulation=KINOVA_GRIPPER_ARTICULATION,
    )


# Action scale for delta control
KINOVA_ACTION_SCALE = 0.04
