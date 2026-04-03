"""Kinova Gen3 reach task using Operational Space Control (OSC, arm only, no gripper)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from kinova_tasks.assets.kinova_gen3.kinova_constants import get_assets
from kinova_tasks.tasks.base_rl_cfg import kinova_ppo_runner_cfg
from kinova_tasks.tasks.actions.osc import OperationalSpaceActionCfg
from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
import mjlab.envs.mdp as mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
    compute_pose_error,
    matrix_from_quat,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# ---------------------------------------------------------------------------
# Robot asset — no-gripper arm with native torque actuators
# ---------------------------------------------------------------------------

_NO_GRIPPER_XML = (
    Path(__file__).parent.parent / "assets/kinova_gen3/xmls/gen3_no_gripper_torque.xml"
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
    spec = mujoco.MjSpec.from_file(str(_NO_GRIPPER_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec  # keep native <motor> actuators


def _get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=_INIT_STATE,
        collisions=(),
        spec_fn=_get_no_gripper_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlMotorActuatorCfg(target_names_expr=("joint_.*",)),),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


# ---------------------------------------------------------------------------
# Home EE pose (pinch_site FK at default joint config)
# ---------------------------------------------------------------------------

_HOME_POS = (0.733607, -0.024850, 0.523015)
_HOME_QUAT = (0.5, 0.5, 0.5, 0.5)  # w, x, y, z

# Distinct axis colors for target frame: slightly desaturated RGB
_TARGET_FRAME_COLORS = (
    (1.0, 0.4, 0.4),  # X — coral red
    (0.4, 1.0, 0.4),  # Y — light green
    (0.4, 0.4, 1.0),  # Z — light blue
)


# ---------------------------------------------------------------------------
# Custom MDP functions
# ---------------------------------------------------------------------------


def joint_pos(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Absolute joint positions (rad)."""
    robot: Entity = env.scene[entity_name]
    return robot.data.joint_pos


def joint_vel(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Absolute joint velocities (rad/s)."""
    robot: Entity = env.scene[entity_name]
    return robot.data.joint_vel


def reset_reach_target(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pos_range: tuple[float, float] = (-0.15, 0.15),   # ±15 cm from home
    ori_range: tuple[float, float] = (-0.3, 0.3),     # ±0.3 rad (~17°) per axis
) -> None:
    """Sample a random 6D target pose and store it on the env."""
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
    return torch.cat([pos_err, rot_err], dim=-1)


def ee_pos_vec(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> torch.Tensor:
    """3D position error vector from EE to target."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    target_pos = getattr(env, "_reach_target_pos", torch.zeros_like(ee_pos))
    return target_pos - ee_pos


def ee_pos_error(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> torch.Tensor:
    """Euclidean distance (m) from EE to target position."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    target_pos = getattr(env, "_reach_target_pos", ee_pos.clone())
    return torch.norm(ee_pos - target_pos, dim=-1)


def ee_ori_error(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> torch.Tensor:
    """Orientation error magnitude (rad) from EE to target orientation."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    ee_quat = robot.data.site_quat_w[:, site_ids].squeeze(1)
    target_pos = getattr(env, "_reach_target_pos", ee_pos.clone())
    target_quat = getattr(env, "_reach_target_quat", ee_quat.clone())
    _, rot_err = compute_pose_error(ee_pos, ee_quat, target_pos, target_quat)
    return torch.norm(rot_err, dim=-1)


def pos_success(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
    pos_threshold: float = 0.02,
) -> torch.Tensor:
    """Binary: EE within pos_threshold (m) of target position (matches reward signal)."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    target_pos = getattr(env, "_reach_target_pos", ee_pos.clone())
    return (torch.norm(ee_pos - target_pos, dim=-1) < pos_threshold).float()


def pose_success(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
    pos_threshold: float = 0.02,
    ori_threshold: float = 0.1,
) -> torch.Tensor:
    """Binary: EE within pos_threshold (m) AND ori_threshold (rad) of target pose."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    ee_quat = robot.data.site_quat_w[:, site_ids].squeeze(1)
    target_pos = getattr(env, "_reach_target_pos", ee_pos.clone())
    target_quat = getattr(env, "_reach_target_quat", ee_quat.clone())
    pos_err, rot_err = compute_pose_error(ee_pos, ee_quat, target_pos, target_quat)
    pos_ok = torch.norm(pos_err, dim=-1) < pos_threshold
    ori_ok = torch.norm(rot_err, dim=-1) < ori_threshold
    return (pos_ok & ori_ok).float()


# ---------------------------------------------------------------------------
# Class-based reward term — computes reward + draws EE/target frames in viser
# ---------------------------------------------------------------------------


class reach_reward:
    """Gaussian reach reward with debug_vis drawing EE and target frames."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._entity_name: str = cfg.params.get("entity_name", "robot")
        self._site_name: str = cfg.params.get("site_name", "pinch_site")
        self._debug_vis_enabled: bool = True
        self._delta_pos_scale: float = env.action_manager._terms["osc_pose"].cfg.delta_pos_scale

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float = 0.1,
        entity_name: str = "robot",
        site_name: str = "pinch_site",
    ) -> torch.Tensor:
        robot: Entity = env.scene[entity_name]
        site_ids = robot.find_sites(site_name)[0]
        ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
        target_pos = getattr(env, "_reach_target_pos", ee_pos.clone())
        dist_sq = torch.sum(torch.square(ee_pos - target_pos), dim=-1)
        return torch.exp(-dist_sq / (std ** 2))

    def reset(self, env_ids: torch.Tensor) -> None:
        pass

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        if not self._debug_vis_enabled:
            return
        env = self._env
        env_indices = list(visualizer.get_env_indices(env.num_envs))
        if not env_indices:
            return

        robot: Entity = env.scene[self._entity_name]
        site_ids = robot.find_sites(self._site_name)[0]

        # Last action: shape (num_envs, 6), first 3 dims are delta_ee_pos (unit scale)
        last_action = env.action_manager.action  # (num_envs, 6)

        for i in env_indices:
            # EE frame (default RGB axes)
            ee_pos = robot.data.site_pos_w[i, site_ids].squeeze(0)
            ee_quat = robot.data.site_quat_w[i, site_ids].squeeze(0)
            ee_rotm = matrix_from_quat(ee_quat.unsqueeze(0)).squeeze(0).cpu().numpy()
            visualizer.add_frame(
                position=ee_pos.cpu().numpy(),
                rotation_matrix=ee_rotm,
                scale=0.12,
                label=f"ee_frame_{i}",
            )

            # Action arrow: delta_ee_pos scaled by delta_pos_scale
            delta_pos = last_action[i, :3] * self._delta_pos_scale
            arrow_end = (ee_pos + delta_pos).cpu().numpy()
            visualizer.add_arrow(
                start=ee_pos.cpu().numpy(),
                end=arrow_end,
                color=(1.0, 0.8, 0.0, 1.0),  # yellow
                width=0.01,
                label=f"action_arrow_{i}",
            )

            # Pose error arrow: EE → target position (cyan)
            if not hasattr(env, "_reach_target_pos"):
                continue
            tgt_pos = env._reach_target_pos[i]
            visualizer.add_arrow(
                start=ee_pos.cpu().numpy(),
                end=tgt_pos.cpu().numpy(),
                color=(0.0, 1.0, 1.0, 1.0),  # cyan
                width=0.01,
                label=f"pos_error_arrow_{i}",
            )

            # Target frame (desaturated axes to distinguish from EE)
            tgt_quat = env._reach_target_quat[i]
            tgt_rotm = matrix_from_quat(tgt_quat.unsqueeze(0)).squeeze(0).cpu().numpy()
            visualizer.add_frame(
                position=tgt_pos.cpu().numpy(),
                rotation_matrix=tgt_rotm,
                scale=0.12,
                label=f"target_frame_{i}",
                axis_colors=_TARGET_FRAME_COLORS,
            )


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


def kinova_reach_osc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Kinova Gen3 reach task with OSC torque control (arm only, no gripper).

    Goal: move pinch_site to a randomised 6D target pose each episode.
    EE and target frames are visualised in viser via the debug_vis system.

    Actions (6D):
        relative pos + axis-angle delta applied to current EE pose
    Observations (26D):
        joint_pos_rel(7) + joint_vel_rel(7) + ee_to_target(6) + last_action(6)
    """
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=joint_pos),
        "joint_vel": ObservationTermCfg(func=joint_vel),
        "ee_pos_vec": ObservationTermCfg(func=ee_pos_vec),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    actions = {
        "osc_pose": OperationalSpaceActionCfg(
            entity_name="robot",
            actuator_names=("joint_.*",),
            frame_name="pinch_site",
            frame_type="site",
            use_relative_mode=True,
            delta_pos_scale=0.03,       # 3 cm per unit action
            delta_ori_scale=0.02,       # ~1.1° per unit action
            position_weight=1.0,
            orientation_weight=0.0,
            kp_pos=50.0,
            kd_pos=10.0,
            kp_ori=50.0,
            kd_ori=10.0,
            max_torque=[39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0],  # Nm, joints 1-4 / 5-7
            posture_weight=0.0,         # null-space posture restore
            posture_kp=10.0,
            posture_kd=2.0,
        ),
    }

    events = {
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.05, 0.05),  # ±0.05 rad per joint
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "reset_reach_target": EventTermCfg(
            func=reset_reach_target,
            mode="reset",
        ),
    }

    rewards = {
        "reach": RewardTermCfg(
            func=reach_reward,
            weight=1.0,
            params={"std": 0.5},        # coarse reward, gradient even when far
        ),
        "reach_precise": RewardTermCfg(
            func=reach_reward,
            weight=1.0,
            params={"std": 0.1},        # fine reward, incentivises getting close
        ),
        # "action_rate_l2": RewardTermCfg(
        #     func=mdp.action_rate_l2,
        #     weight=-0.01,               # penalise jerky actions
        # ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_detection": TerminationTermCfg(func=mdp.nan_detection, time_out=False),
    }

    metrics = {
        "pos_error": MetricsTermCfg(func=ee_pos_error),
        "ori_error": MetricsTermCfg(func=ee_ori_error),
        "pos_success": MetricsTermCfg(func=pos_success),    # matches reward signal
        "pose_success": MetricsTermCfg(func=pose_success),  # pos + ori within threshold
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=1.5,
            entities={"robot": _get_robot_cfg()},
        ),
        observations={
            "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
            "critic": ObservationGroupCfg(actor_terms, enable_corruption=False),
        },
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
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
                timestep=0.002,         # 500 Hz physics
                iterations=4,
                ls_iterations=10,
            ),
        ),
        decimation=50,                  # policy runs at 10 Hz
        episode_length_s=3.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.observations["critic"].enable_corruption = False

    return cfg


def kinova_reach_osc_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for OSC reach task."""
    return kinova_ppo_runner_cfg(experiment_name="kinova_reach_osc")
