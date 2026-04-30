"""Kinova Gen3 reach task using DifferentialIK action (arm only, no gripper)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from kinova_tasks.tasks.base_rl_cfg import kinova_ppo_runner_cfg
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
    compute_pose_error,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Robot asset — no-gripper arm with PD position actuators
# ---------------------------------------------------------------------------

_NO_GRIPPER_XML = (
    Path(__file__).parent.parent / "assets/kinova_gen3/xmls/gen3_no_gripper_torque.xml"
)
_TARGET_MARKER_XML = (
    Path(__file__).parent.parent / "assets/reach/xmls/target_marker.xml"
)
_EE_MARKER_XML = (
    Path(__file__).parent.parent / "assets/reach/xmls/ee_marker.xml"
)

_INIT_STATE = EntityCfg.InitialStateCfg(
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


def _get_no_gripper_spec() -> mujoco.MjSpec:
    """Load no-gripper arm XML; remove built-in motor actuators.

    Existing <motor> actuators are cleared so IdealPdActuatorCfg can
    add its own (position-target-based) motor actuators without conflicts.
    """
    spec = mujoco.MjSpec.from_file(str(_NO_GRIPPER_XML))
    for act in list(spec.actuators):
        spec.delete(act)
    return spec


def _get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=_INIT_STATE,
        collisions=(),
        spec_fn=_get_no_gripper_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                # Joints 1-4: large links, higher torque
                IdealPdActuatorCfg(
                    target_names_expr=("joint_[1-4]",),
                    stiffness=300.0,
                    damping=20.0,
                    effort_limit=39.0,
                    armature=5.0,
                    frictionloss=1.0,
                ),
                # Joints 5-7: wrist links, lower torque
                IdealPdActuatorCfg(
                    target_names_expr=("joint_[5-7]",),
                    stiffness=100.0,
                    damping=10.0,
                    effort_limit=9.0,
                    armature=5.5,
                    frictionloss=2.0,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


def _get_target_marker_cfg() -> EntityCfg:
    def _spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_TARGET_MARKER_XML))
    return EntityCfg(spec_fn=_spec)


def _get_ee_marker_cfg() -> EntityCfg:
    def _spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_EE_MARKER_XML))
    return EntityCfg(spec_fn=_spec)


# ---------------------------------------------------------------------------
# Home EE pose (pinch_site FK at default joint config)
# ---------------------------------------------------------------------------

_HOME_POS = (0.733607, -0.024850, 0.523015)
_HOME_QUAT = (0.5, 0.5, 0.5, 0.5)  # w, x, y, z


# ---------------------------------------------------------------------------
# Custom MDP functions
# ---------------------------------------------------------------------------


def reset_reach_target(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pos_range: tuple[float, float] = (-0.15, 0.15),
    ori_range: tuple[float, float] = (-0.3, 0.3),
) -> None:
    """Sample a random 6D target pose and store it on the env.

    Position is sampled within [home ± pos_range] in the robot local frame.
    Orientation is a small Euler-angle perturbation around home quat.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)

    home = torch.tensor(_HOME_POS, device=env.device)
    r = pos_range
    target_pos_local = sample_uniform(
        home + r[0], home + r[1], (n, 3), device=env.device
    )
    target_pos = target_pos_local + env.scene.env_origins[env_ids]

    euler = sample_uniform(
        torch.tensor([ori_range[0]] * 3, device=env.device),
        torch.tensor([ori_range[1]] * 3, device=env.device),
        (n, 3),
        device=env.device,
    )
    delta_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
    home_quat = torch.tensor(_HOME_QUAT, device=env.device).unsqueeze(0).expand(n, -1)
    target_quat = quat_mul(home_quat, delta_quat)

    if not hasattr(env, "_reach_target_pos"):
        env._reach_target_pos = torch.zeros(env.num_envs, 3, device=env.device)
        env._reach_target_quat = torch.zeros(env.num_envs, 4, device=env.device)
        env._reach_target_quat[:, 0] = 1.0

    env._reach_target_pos[env_ids] = target_pos
    env._reach_target_quat[env_ids] = target_quat

    # Move the visual marker to the sampled target pose
    marker: Entity = env.scene["target"]
    pose = torch.cat([target_pos, target_quat], dim=-1)  # (n, 7)
    marker.write_mocap_pose_to_sim(pose, env_ids=env_ids)


def ee_to_target(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> torch.Tensor:
    """6D error [pos_error(3), rot_error_axis_angle(3)] from EE to target."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    ee_quat = robot.data.site_quat_w[:, site_ids].squeeze(1)

    target_pos = getattr(env, "_reach_target_pos", torch.zeros_like(ee_pos))
    target_quat = getattr(env, "_reach_target_quat", ee_quat.clone())

    pos_err, rot_err = compute_pose_error(ee_pos, ee_quat, target_pos, target_quat)
    return torch.cat([pos_err, rot_err], dim=-1)  # (N, 6)


def reach_reward(
    env: ManagerBasedRlEnv,
    std: float = 0.1,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> torch.Tensor:
    """Gaussian reward on EE-to-target position distance."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    target_pos = getattr(env, "_reach_target_pos", ee_pos.clone())
    dist_sq = torch.sum(torch.square(ee_pos - target_pos), dim=-1)
    return torch.exp(-dist_sq / (std ** 2))


def update_ee_marker(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> None:
    """Move the EE marker mocap body to the current pinch_site pose every step."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)   # (N, 3)
    ee_quat = robot.data.site_quat_w[:, site_ids].squeeze(1)  # (N, 4)

    marker: Entity = env.scene["ee_marker"]
    pose = torch.cat([ee_pos, ee_quat], dim=-1)  # (N, 7)
    marker.write_mocap_pose_to_sim(pose)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


def kinova_reach_diff_ik_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Kinova Gen3 reach task with 6D differential IK control (arm only, no gripper).

    Goal: move pinch_site to a randomised 6D target pose each episode.

    Actions (6D):
        relative pos + axis-angle delta applied to current EE pose
    Observations (26D):
        joint_pos_rel(7) + joint_vel_rel(7) + ee_to_target(6) + last_action(6)
    """
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "ee_to_target": ObservationTermCfg(func=ee_to_target),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    actions = {
        "ik_pose": DifferentialIKActionCfg(
            entity_name="robot",
            actuator_names=("joint_[1-7]",),
            frame_name="pinch_site",
            frame_type="site",
            use_relative_mode=True,
            delta_pos_scale=0.05,   # 5 cm / step
            delta_ori_scale=0.1,    # ~6° / step
            kp_task=1.0,
            damping=0.05,
            max_dq=0.5,
            position_weight=1.0,
            orientation_weight=1.0,
            joint_limit_weight=0.1,
            posture_weight=0.02,
        ),
    }

    events = {
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.05, 0.05),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "reset_reach_target": EventTermCfg(
            func=reset_reach_target,
            mode="reset",
        ),
        "update_ee_marker": EventTermCfg(
            func=update_ee_marker,
            mode="step",
        ),
    }

    rewards = {
        "reach": RewardTermCfg(
            func=reach_reward,
            weight=1.0,
            params={"std": 0.1},
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.01,
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_detection": TerminationTermCfg(func=nan_detection, time_out=False),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=1.5,
            entities={
                "robot": _get_robot_cfg(),
                "target": _get_target_marker_cfg(),
                "ee_marker": _get_ee_marker_cfg(),
            },
        ),
        observations={"actor": ObservationGroupCfg(actor_terms, enable_corruption=True)},
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=1.5,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.002,  # 500 Hz physics
                iterations=4,
                ls_iterations=10,
            ),
        ),
        decimation=10,        # 50 Hz policy rate
        episode_length_s=10.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg


def kinova_reach_diff_ik_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for reach task."""
    return kinova_ppo_runner_cfg(experiment_name="kinova_reach_diff_ik")
