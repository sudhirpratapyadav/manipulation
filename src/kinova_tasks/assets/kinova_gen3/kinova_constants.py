"""Kinova Gen3 robot constants and configuration for lift task."""

from pathlib import Path
import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

_HERE = Path(__file__).parent
KINOVA_GEN3_GRIPPER_XML: Path = _HERE / "xmls" / "gen3_gripper.xml"
KINOVA_GEN3_GRIPPER_TORQUE_XML: Path = _HERE / "xmls" / "gen3_gripper_torque.xml"
KINOVA_GEN3_NO_GRIPPER_XML: Path = _HERE / "xmls" / "gen3_no_gripper_torque.xml"


def get_spec() -> mujoco.MjSpec:
    """Load Kinova Gen3 with Robotiq 2F-85 gripper for position control.

    Includes the 7-DOF arm with position actuators plus the parallel gripper mechanism.
    """
    return mujoco.MjSpec.from_file(str(KINOVA_GEN3_GRIPPER_XML))


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
        # Gripper joints - 60% closed (consistent 4-bar linkage state)
        "right_driver_joint": 0.503,
        "right_coupler_joint": 0.001,
        "right_spring_link_joint": 0.505,
        "right_follower_joint": -0.485,
        "left_driver_joint": 0.503,
        "left_coupler_joint": 0.001,
        "left_spring_link_joint": 0.505,
        "left_follower_joint": -0.485,
    },
    joint_vel={".*": 0.0},
)

##
# Articulation config.
##

# gen3_gripper.xml declares the 7 arm <position> actuators; fingers_actuator
# is a <general> tendon actuator and is intentionally not part of articulation.
# (target_names_expr matches joint names, so the fingers tendon is filtered out.)
KINOVA_ACTUATORS = XmlActuatorCfg(
    target_names_expr=(".*",),  # Match all joints (arm + gripper finger joints)
    command_field="position",
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

##
# No-gripper arm (native torque actuators, for OSC control).
##

# Initial state for the arm-only (no gripper) configuration
INIT_STATE_NO_GRIPPER = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "joint_1": 0.0,
        "joint_2": 0.3490658504,   # 20°
        "joint_3": 0.0,
        "joint_4": 1.7453292519,   # 100°
        "joint_5": 0.0,
        "joint_6": -0.5235987756,  # -30°
        "joint_7": -1.5707963268,  # -90°
    },
    joint_vel={".*": 0.0},
)


def get_no_gripper_spec() -> mujoco.MjSpec:
    """Load Kinova Gen3 arm without gripper using native torque actuators."""
    return mujoco.MjSpec.from_file(str(KINOVA_GEN3_NO_GRIPPER_XML))


def get_gripper_torque_spec() -> mujoco.MjSpec:
    """Load Kinova Gen3 with Robotiq 2F-85 gripper using torque actuators for the arm."""
    return mujoco.MjSpec.from_file(str(KINOVA_GEN3_GRIPPER_TORQUE_XML))


# gen3_no_gripper_torque.xml declares the 7 arm joints as <motor> torque actuators.
KINOVA_NO_GRIPPER_ACTUATORS = XmlActuatorCfg(
    target_names_expr=("joint_.*",),
    command_field="effort",
)

KINOVA_NO_GRIPPER_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(KINOVA_NO_GRIPPER_ACTUATORS,),
    soft_joint_pos_limit_factor=0.9,
)


def get_kinova_no_gripper_robot_cfg() -> EntityCfg:
    """Get Kinova Gen3 arm-only config with native torque actuators (for OSC control).

    Returns a fresh EntityCfg each time to avoid mutation issues.
    """
    return EntityCfg(
        init_state=INIT_STATE_NO_GRIPPER,
        collisions=(),
        spec_fn=get_no_gripper_spec,
        articulation=KINOVA_NO_GRIPPER_ARTICULATION,
    )


##
# Gripper arm with torque actuators (for OSC + gripper tasks).
##

# gen3_gripper_torque.xml declares the 7 arm joints as <motor> torque actuators.
# fingers_actuator is a tendon-based <general> actuator controlled separately
# via write_ctrl (SceneEntityCfg with actuator_names), not through articulation.
KINOVA_GRIPPER_TORQUE_ARM_ACTUATORS = XmlActuatorCfg(
    target_names_expr=("joint_.*",),
    command_field="effort",
)

KINOVA_GRIPPER_TORQUE_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(KINOVA_GRIPPER_TORQUE_ARM_ACTUATORS,),
    soft_joint_pos_limit_factor=0.9,
)


def get_kinova_robot_cfg_peginhole_osc() -> EntityCfg:
    """Get Kinova Gen3 + gripper config for peg-in-hole with OSC torque control.

    Uses gen3_gripper_torque.xml: arm joints are native torque (motor) actuators
    for OSC, gripper uses the tendon-based fingers_actuator (position).
    Arm is rotated 90° at joint_1 to face the workspace side.
    """
    return EntityCfg(
        init_state=INIT_STATE_PEGINHOLE,
        collisions=(),
        spec_fn=get_gripper_torque_spec,
        articulation=KINOVA_GRIPPER_TORQUE_ARTICULATION,
    )


##
# Snap-fit horizontal-reach init state (gripper closed, EE faces +X).
##

# Horizontal home: same arm joints as reach_osc (EE at (0.734, -0.025, 0.523)
# with tool axis pointing world +X) but gripper closed to hold the snap peg.
INIT_STATE_SNAP_FIT = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        # Arm joints — horizontal reach pose (matches reach_osc _HOME_JOINT_POS)
        "joint_1": 0.0,                # 0°
        "joint_2": 0.3490658504,       # 20°
        "joint_3": 0.0,                # 0°
        "joint_4": 1.7453292519,       # 100°
        "joint_5": 0.0,                # 0°
        "joint_6": -0.5235987756,      # -30°
        "joint_7": -1.5707963268,      # -90°
        # Gripper joints — closed (consistent 4-bar linkage equilibrium)
        "right_driver_joint": 0.503,
        "right_coupler_joint": 0.001,
        "right_spring_link_joint": 0.505,
        "right_follower_joint": -0.485,
        "left_driver_joint": 0.503,
        "left_coupler_joint": 0.001,
        "left_spring_link_joint": 0.505,
        "left_follower_joint": -0.485,
    },
    joint_vel={".*": 0.0},
)


def get_kinova_robot_cfg_snap_fit_osc() -> EntityCfg:
    """Kinova Gen3 + gripper for snap-fit (horizontal push, OSC torque control).

    EE faces +X (forward) with gripper closed to hold the snap peg. Arm pose
    is the same as reach_osc; gripper is at the closed equilibrium used by
    peg-in-hole.
    """
    return EntityCfg(
        init_state=INIT_STATE_SNAP_FIT,
        collisions=(),
        spec_fn=get_gripper_torque_spec,
        articulation=KINOVA_GRIPPER_TORQUE_ARTICULATION,
    )
