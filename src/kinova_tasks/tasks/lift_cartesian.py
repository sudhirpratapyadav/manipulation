"""Kinova Gen3 lift task with cartesian (differential IK) actions."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from kinova_tasks.tasks.base_env_cfg import kinova_base_env_cfg
from kinova_tasks.tasks.base_rl_cfg import kinova_ppo_runner_cfg
from kinova_tasks.tasks.actions import HomeRelativeIKActionCfg


def kinova_lift_cartesian_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Kinova lift cube with cartesian (IK) control for arm + joint control for gripper.

    Actions:
        - ik_pose (6D): position + axis-angle deltas relative to home EE pose
        - gripper (1D): gripper driver joint position
    """
    cfg = kinova_base_env_cfg(play=play)

    # Replace joint_pos action with IK action for arm + separate gripper action
    cfg.actions = {
        "ik_pose": HomeRelativeIKActionCfg(
            entity_name="robot",
            actuator_names=("joint_.*",),  # Arm joints only
            frame_name="pinch_site",
            frame_type="site",
            # Home EE pose from default joint config (pinch_site FK)
            home_pos=(0.733607, -0.024850, 0.523015),
            home_quat=(0.5, 0.5, 0.5, 0.5),
            damping=0.05,
            max_dq=0.5,
            position_weight=1.0,
            orientation_weight=1.0,
            joint_limit_weight=0.1,
            posture_weight=0.02,
            pos_scale=0.1,
            ori_scale=0.1,
        ),
        "gripper": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("fingers_actuator",),  # Tendon-based gripper
            scale=1.0,
            use_default_offset=True,
        ),
    }

    return cfg


def kinova_lift_cartesian_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for cartesian-space lift task."""
    return kinova_ppo_runner_cfg(experiment_name="kinova_lift_cube_cartesian")
