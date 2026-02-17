"""Kinova Gen3 lift task with joint-space actions."""

from kinova_tasks.assets.kinova_gen3 import KINOVA_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from kinova_tasks.tasks.base_env_cfg import kinova_base_env_cfg
from kinova_tasks.tasks.base_rl_cfg import kinova_ppo_runner_cfg


def kinova_lift_joint_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Kinova lift cube with joint-space position control."""
    cfg = kinova_base_env_cfg(play=play)

    # Joint position actions for arm + gripper
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KINOVA_ACTION_SCALE

    return cfg


def kinova_lift_joint_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for joint-space lift task."""
    return kinova_ppo_runner_cfg(experiment_name="kinova_lift_cube_joint")
