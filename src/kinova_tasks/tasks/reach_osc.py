"""Kinova Gen3 reach task using Operational Space Control (OSC, arm only, no gripper)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from kinova_tasks.assets.kinova_gen3.kinova_constants import get_kinova_no_gripper_robot_cfg
from kinova_tasks.tasks.base_rl_cfg import kinova_ppo_runner_cfg
from kinova_tasks.tasks.actions.osc import OperationalSpaceActionCfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
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
# Home EE pose (pinch_site FK at default joint config)
# ---------------------------------------------------------------------------

_HOME_POS = (0.733607, -0.024850, 0.523015)
_HOME_QUAT = (0.5, 0.5, 0.5, 0.5)  # w, x, y, z

_TARGET_FRAME_COLORS = (
    (1.0, 0.4, 0.4),  # X — coral red
    (0.4, 1.0, 0.4),  # Y — light green
    (0.4, 0.4, 1.0),  # Z — light blue
)


# ---------------------------------------------------------------------------
# Reach pose command (CommandTerm / CommandTermCfg)
# ---------------------------------------------------------------------------


class ReachPoseCommand(CommandTerm):
    """Samples and maintains a 6D target pose for the EE to reach.

    Target is expressed in local/physics frame (same as ``site_pos_w``).
    Each env's MuJoCo simulation starts at its own local origin — env_origins
    are a visualisation-only grid offset and must NOT be mixed into physics
    coordinates.
    """

    cfg: ReachPoseCommandCfg

    def __init__(self, cfg: ReachPoseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        robot: Entity = env.scene[cfg.entity_name]
        self._site_ids = robot.find_sites(cfg.site_name)[0]

        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_quat[:, 0] = 1.0  # identity quaternion

        self.metrics["pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ori_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pos_success"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Concatenated [target_pos(3), target_quat(4)] in world frame, shape (B, 7)."""
        return torch.cat([self.target_pos, self.target_quat], dim=-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        r = self.cfg.pos_range
        ori_r = self.cfg.ori_range

        home = torch.tensor(_HOME_POS, device=self.device)
        target_pos_local = sample_uniform(
            home + r[0], home + r[1], (n, 3), device=self.device
        )
        self.target_pos[env_ids] = target_pos_local

        euler = sample_uniform(
            torch.tensor([ori_r[0]] * 3, device=self.device),
            torch.tensor([ori_r[1]] * 3, device=self.device),
            (n, 3),
            device=self.device,
        )
        delta_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        home_quat = torch.tensor(_HOME_QUAT, device=self.device).unsqueeze(0).expand(n, -1)
        self.target_quat[env_ids] = quat_mul(home_quat, delta_quat)

    def _update_metrics(self) -> None:
        robot: Entity = self._env.scene[self.cfg.entity_name]
        ee_pos = robot.data.site_pos_w[:, self._site_ids].squeeze(1)
        ee_quat = robot.data.site_quat_w[:, self._site_ids].squeeze(1)
        pos_err, rot_err = compute_pose_error(
            ee_pos, ee_quat, self.target_pos, self.target_quat
        )
        pos_norm = torch.norm(pos_err, dim=-1)
        ori_norm = torch.norm(rot_err, dim=-1)
        self.metrics["pos_error"] = pos_norm
        self.metrics["ori_error"] = ori_norm
        self.metrics["pos_success"] = (pos_norm < self.cfg.pos_threshold).float()

    def _update_command(self) -> None:
        pass  # Target is static between resamples.

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self._env.num_envs)
        if not env_indices:
            return
        for i in env_indices:
            tgt_pos = self.target_pos[i]
            tgt_quat = self.target_quat[i]
            tgt_rotm = matrix_from_quat(tgt_quat.unsqueeze(0)).squeeze(0).cpu().numpy()
            visualizer.add_frame(
                position=tgt_pos.cpu().numpy(),
                rotation_matrix=tgt_rotm,
                scale=0.12,
                label=f"target_frame_{i}",
                axis_colors=_TARGET_FRAME_COLORS,
            )


@dataclass(kw_only=True)
class ReachPoseCommandCfg(CommandTermCfg):
    """Configuration for the reach-pose command term."""

    entity_name: str = "robot"
    site_name: str = "pinch_site"
    pos_range: tuple[float, float] = (-0.15, 0.15)
    ori_range: tuple[float, float] = (-0.3, 0.3)
    pos_threshold: float = 0.02

    def build(self, env: ManagerBasedRlEnv) -> ReachPoseCommand:
        return ReachPoseCommand(self, env)


# ---------------------------------------------------------------------------
# Custom MDP functions
# ---------------------------------------------------------------------------


def _get_reach_command(env: ManagerBasedRlEnv, command_name: str) -> ReachPoseCommand:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, ReachPoseCommand), (
        f"Expected ReachPoseCommand for '{command_name}', got {type(term)}"
    )
    return term


def joint_pos(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Absolute joint positions (rad)."""
    robot: Entity = env.scene[entity_name]
    return robot.data.joint_pos


def joint_vel(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Absolute joint velocities (rad/s)."""
    robot: Entity = env.scene[entity_name]
    return robot.data.joint_vel


def ee_to_target(
    env: ManagerBasedRlEnv,
    command_name: str = "reach_pose",
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> torch.Tensor:
    """6D error [pos_error(3), rot_error_axis_angle(3)] from EE to target."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    ee_quat = robot.data.site_quat_w[:, site_ids].squeeze(1)
    cmd = _get_reach_command(env, command_name)
    pos_err, rot_err = compute_pose_error(ee_pos, ee_quat, cmd.target_pos, cmd.target_quat)
    return torch.cat([pos_err, rot_err], dim=-1)


def ee_pos_vec(
    env: ManagerBasedRlEnv,
    command_name: str = "reach_pose",
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> torch.Tensor:
    """3D position error vector from EE to target.

    Both target and EE are in local/physics frame (no env_origin offset),
    so their difference is a consistent error vector matching what the
    real robot would observe.
    """
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    cmd = _get_reach_command(env, command_name)
    return cmd.target_pos - ee_pos


def ee_pos_error(
    env: ManagerBasedRlEnv,
    command_name: str = "reach_pose",
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> torch.Tensor:
    """Euclidean distance (m) from EE to target position."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    cmd = _get_reach_command(env, command_name)
    return torch.norm(ee_pos - cmd.target_pos, dim=-1)


def ee_ori_error(
    env: ManagerBasedRlEnv,
    command_name: str = "reach_pose",
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> torch.Tensor:
    """Orientation error magnitude (rad) from EE to target orientation."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    ee_quat = robot.data.site_quat_w[:, site_ids].squeeze(1)
    cmd = _get_reach_command(env, command_name)
    _, rot_err = compute_pose_error(ee_pos, ee_quat, cmd.target_pos, cmd.target_quat)
    return torch.norm(rot_err, dim=-1)


def pos_success(
    env: ManagerBasedRlEnv,
    command_name: str = "reach_pose",
    entity_name: str = "robot",
    site_name: str = "pinch_site",
    pos_threshold: float = 0.02,
) -> torch.Tensor:
    """Binary: EE within pos_threshold (m) of target position."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    cmd = _get_reach_command(env, command_name)
    return (torch.norm(ee_pos - cmd.target_pos, dim=-1) < pos_threshold).float()


def pose_success(
    env: ManagerBasedRlEnv,
    command_name: str = "reach_pose",
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
    cmd = _get_reach_command(env, command_name)
    pos_err, rot_err = compute_pose_error(ee_pos, ee_quat, cmd.target_pos, cmd.target_quat)
    pos_ok = torch.norm(pos_err, dim=-1) < pos_threshold
    ori_ok = torch.norm(rot_err, dim=-1) < ori_threshold
    return (pos_ok & ori_ok).float()


# ---------------------------------------------------------------------------
# Class-based reward term — computes reward + draws EE/action frames in viser
# ---------------------------------------------------------------------------


def ori_reward(
    env: ManagerBasedRlEnv,
    std: float = 0.3,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
    command_name: str = "reach_pose",
) -> torch.Tensor:
    """Gaussian orientation reward based on axis-angle error magnitude."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    ee_quat = robot.data.site_quat_w[:, site_ids].squeeze(1)
    cmd = _get_reach_command(env, command_name)
    _, rot_err = compute_pose_error(ee_pos, ee_quat, cmd.target_pos, cmd.target_quat)
    rot_sq = torch.sum(torch.square(rot_err), dim=-1)
    return torch.exp(-rot_sq / (std ** 2))


class reach_reward:
    """Gaussian reach reward. Draws EE frame and action arrow via debug_vis.
    Target frame is drawn by ReachPoseCommand._debug_vis_impl.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._entity_name: str = cfg.params.get("entity_name", "robot")
        self._site_name: str = cfg.params.get("site_name", "pinch_site")
        self._command_name: str = cfg.params.get("command_name", "reach_pose")
        self._debug_vis_enabled: bool = True
        self._delta_pos_scale: float = env.action_manager._terms["osc_pose"].cfg.delta_pos_scale

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float = 0.1,
        entity_name: str = "robot",
        site_name: str = "pinch_site",
        command_name: str = "reach_pose",
    ) -> torch.Tensor:
        robot: Entity = env.scene[entity_name]
        site_ids = robot.find_sites(site_name)[0]
        ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
        cmd = _get_reach_command(env, command_name)
        # print(f"cmd:{cmd.target_pos}")
        dist_sq = torch.sum(torch.square(ee_pos - cmd.target_pos), dim=-1)
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
        last_action = env.action_manager.action  # (num_envs, 6)

        for i in env_indices:
            ee_pos = robot.data.site_pos_w[i, site_ids].squeeze(0)
            ee_quat = robot.data.site_quat_w[i, site_ids].squeeze(0)
            ee_rotm = matrix_from_quat(ee_quat.unsqueeze(0)).squeeze(0).cpu().numpy()
            visualizer.add_frame(
                position=ee_pos.cpu().numpy(),
                rotation_matrix=ee_rotm,
                scale=0.12,
                label=f"ee_frame_{i}",
            )
            # Action arrow: delta_ee_pos scaled to metres
            delta_pos = last_action[i, :3] * self._delta_pos_scale
            arrow_end = (ee_pos + delta_pos).cpu().numpy()
            visualizer.add_arrow(
                start=ee_pos.cpu().numpy(),
                end=arrow_end,
                color=(1.0, 0.8, 0.0, 1.0),  # yellow
                width=0.01,
                label=f"action_arrow_{i}",
            )


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


def kinova_reach_osc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Kinova Gen3 reach task with OSC torque control (arm only, no gripper).

    Goal: move pinch_site to a randomised 6D target pose. The target is
    resampled every 3 s (mid-episode) via the command manager, following
    the standard mjlab pattern.

    Actions (6D):
        relative pos + axis-angle delta applied to current EE pose
    Observations (26D):
        joint_pos(7) + joint_vel(7) + ee_to_target(6) + last_action(6)
    """
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=joint_pos),
        "joint_vel": ObservationTermCfg(func=joint_vel),
        "ee_to_target": ObservationTermCfg(
            func=ee_to_target,
            params={"command_name": "reach_pose"},
        ),
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
            orientation_weight=1.0,
            kp_pos=50.0,
            kd_pos=10.0,
            kp_ori=50.0,
            kd_ori=10.0,
            max_torque=[39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0],
            posture_weight=0.0,
            posture_kp=10.0,
            posture_kd=2.0,
        ),
    }

    commands = {
        "reach_pose": ReachPoseCommandCfg(
            resampling_time_range=(3.0, 3.0),  # resample every 3 s mid-episode
            debug_vis=True,
            pos_range=(-0.15, 0.15),
            ori_range=(-3.14159, 3.14159),
            pos_threshold=0.02,
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
    }

    rewards = {
        "reach": RewardTermCfg(
            func=reach_reward,
            weight=1.0,
            params={"std": 0.3, "command_name": "reach_pose"},
        ),
        "reach_precise": RewardTermCfg(
            func=reach_reward,
            weight=1.0,
            params={"std": 0.1, "command_name": "reach_pose"},
        ),
        "ori": RewardTermCfg(
            func=ori_reward,
            weight=1.0,
            params={"std": 1.0, "command_name": "reach_pose"},
        ),
        "ori_precise": RewardTermCfg(
            func=ori_reward,
            weight=1.0,
            params={"std": 0.3, "command_name": "reach_pose"},
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_detection": TerminationTermCfg(func=mdp.nan_detection, time_out=False),
    }

    metrics = {
        "pos_error": MetricsTermCfg(
            func=ee_pos_error,
            params={"command_name": "reach_pose"},
        ),
        "ori_error": MetricsTermCfg(
            func=ee_ori_error,
            params={"command_name": "reach_pose"},
        ),
        "pos_success": MetricsTermCfg(
            func=pos_success,
            params={"command_name": "reach_pose"},
        ),
        "pose_success": MetricsTermCfg(
            func=pose_success,
            params={"command_name": "reach_pose"},
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=1.5,
            entities={"robot": get_kinova_no_gripper_robot_cfg()},
        ),
        observations={
            "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
            "critic": ObservationGroupCfg(actor_terms, enable_corruption=False),
        },
        actions=actions,
        commands=commands,
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
        episode_length_s=10.0,
    )

    if play:
        cfg.episode_length_s = 10.0
        cfg.observations["actor"].enable_corruption = False
        cfg.observations["critic"].enable_corruption = False

    return cfg


def kinova_reach_osc_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for OSC reach task."""
    return kinova_ppo_runner_cfg(experiment_name="kinova_reach_osc")
