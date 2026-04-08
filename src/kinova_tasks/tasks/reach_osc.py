"""Kinova Gen3 reach task using Operational Space Control (OSC, arm only, no gripper)."""

from __future__ import annotations

import math as _math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

_DEG_TO_RAD = _math.pi / 180.0

from kinova_tasks.assets.kinova_gen3.kinova_constants import get_kinova_no_gripper_robot_cfg
from kinova_tasks.tasks.base_rl_cfg import kinova_ppo_runner_cfg
from kinova_tasks.tasks.actions.osc import OperationalSpaceActionCfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
import mjlab.envs.mdp as mdp
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    compute_pose_error,
    matrix_from_quat,
    sample_uniform,
)
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


_HOME_JOINT_POS = (0.0, 0.3490658504, 0.0, 1.7453292519, 0.0, -0.5235987756, -1.5707963268)
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

        # FK lag buffer: stores the post-reset EE pose from the most recent resample.
        # On the next resample for an env this becomes the new target, guaranteeing
        # the target is always a pose the robot can physically reach.
        home_local = torch.tensor(_HOME_POS, device=self.device)
        self._fk_buf_pos = (home_local + env.scene.env_origins).clone()   # (num_envs, 3)
        home_quat = torch.tensor(_HOME_QUAT, device=self.device)
        self._fk_buf_quat = home_quat.unsqueeze(0).expand(self.num_envs, -1).clone()

    @property
    def command(self) -> torch.Tensor:
        """Concatenated [target_pos(3), target_quat(4)] in world frame, shape (B, 7)."""
        return torch.cat([self.target_pos, self.target_quat], dim=-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        robot: Entity = self._env.scene[self.cfg.entity_name]

        # site_pos_w is valid here: sim.forward() is called before command_manager.compute()
        cur_pos = robot.data.site_pos_w[:, self._site_ids].squeeze(1)    # (num_envs, 3) world
        cur_quat = robot.data.site_quat_w[:, self._site_ids].squeeze(1)  # (num_envs, 4)

        # Serve the buffered FK pose from the PREVIOUS reset as this episode's target.
        # The robot starts at a freshly randomised joint config, so target ≠ start.
        self.target_pos[env_ids] = self._fk_buf_pos[env_ids]
        self.target_quat[env_ids] = self._fk_buf_quat[env_ids]

        # Store THIS episode's starting EE pose for use as the next episode's target.
        self._fk_buf_pos[env_ids] = cur_pos[env_ids].clone()
        self._fk_buf_quat[env_ids] = cur_quat[env_ids].clone()

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass  # Target is static between resamples.

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        import numpy as np
        env_indices = visualizer.get_env_indices(self._env.num_envs)
        if not env_indices:
            return

        lo = np.array(self.cfg.pos_lo, dtype=np.float32)
        hi = np.array(self.cfg.pos_hi, dtype=np.float32)
        local_corners = np.array([
            [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
            [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
            [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]],
        ])
        edges = [
            (0,1),(2,3),(4,5),(6,7),  # along x
            (0,2),(1,3),(4,6),(5,7),  # along y
            (0,4),(1,5),(2,6),(3,7),  # along z
        ]

        for i in env_indices:
            origin = self._env.scene.env_origins[i].cpu().numpy()
            corners = local_corners + origin  # offset to world frame

            for idx, (a, b) in enumerate(edges):
                visualizer.add_cylinder(
                    start=corners[a],
                    end=corners[b],
                    radius=0.005,
                    color=(0.8, 0.8, 0.2, 0.6),
                    label=f"workspace_box_edge_{i}_{idx}",
                )

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
    pos_lo: tuple[float, float, float] = (-1.0, -1.0, -0.3)
    pos_hi: tuple[float, float, float] = ( 1.0,  1.0,  1.2)
    ori_range: tuple[float, float] = (-0.3, 0.3)
    target_pos_delta_max: float | None = None
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


def ee_pose(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> torch.Tensor:
    """Current EE pose [pos(3), axis_angle(3)] in local (robot-base) frame."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    pos = robot.data.site_pos_w[:, site_ids].squeeze(1) - env.scene.env_origins
    quat = robot.data.site_quat_w[:, site_ids].squeeze(1)
    return torch.cat([pos, axis_angle_from_quat(quat)], dim=-1)


def target_pose(
    env: ManagerBasedRlEnv,
    command_name: str = "reach_pose",
) -> torch.Tensor:
    """Target pose [pos(3), axis_angle(3)] in local (robot-base) frame."""
    cmd = _get_reach_command(env, command_name)
    local_pos = cmd.target_pos - env.scene.env_origins
    return torch.cat([local_pos, axis_angle_from_quat(cmd.target_quat)], dim=-1)


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


def gated_ori_reward(
    env: ManagerBasedRlEnv,
    pos_std: float = 0.3,
    ori_std: float = 1.0,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
    command_name: str = "reach_pose",
) -> torch.Tensor:
    """Orientation reward gated by position proximity: reach_reward * (1 + ori_reward)."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    ee_pos = robot.data.site_pos_w[:, site_ids].squeeze(1)
    ee_quat = robot.data.site_quat_w[:, site_ids].squeeze(1)
    cmd = _get_reach_command(env, command_name)
    dist_sq = torch.sum(torch.square(ee_pos - cmd.target_pos), dim=-1)
    reach = torch.exp(-dist_sq / (pos_std ** 2))
    _, rot_err = compute_pose_error(ee_pos, ee_quat, cmd.target_pos, cmd.target_quat)
    rot_sq = torch.sum(torch.square(rot_err), dim=-1)
    ori = torch.exp(-rot_sq / (ori_std ** 2))
    return reach * (1.0 + ori)


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
# Custom event: fully random joint-space reset
# ---------------------------------------------------------------------------


def reset_joints_random(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    entity_name: str = "robot",
    joint_delta_max: float | None = None,
) -> None:
    """Reset joints to home pose ± uniform delta, clipped to soft limits.

    If joint_delta_max is None, falls back to uniform sampling over full soft limits.
    """
    robot: Entity = env.scene[entity_name]
    soft_limits = robot.data.soft_joint_pos_limits  # (num_envs, num_joints, 2)
    lo = soft_limits[env_ids, :, 0]
    hi = soft_limits[env_ids, :, 1]

    if joint_delta_max is None:
        joint_pos = lo + (hi - lo) * torch.rand_like(lo)
    else:
        n = len(env_ids)
        home = torch.tensor(_HOME_JOINT_POS, device=env.device).unsqueeze(0).expand(n, -1)
        delta = sample_uniform(
            torch.full((len(_HOME_JOINT_POS),), -joint_delta_max, device=env.device),
            torch.full((len(_HOME_JOINT_POS),), joint_delta_max, device=env.device),
            (n, len(_HOME_JOINT_POS)),
            device=env.device,
        )
        joint_pos = torch.max(torch.min(home + delta, hi), lo)

    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


def reach_ori_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str = "reach_pose",
    start_deg: float = 10.0,
    hold_iters: int = 200,
    step_iters: int = 100,
    step_deg: float = 10.0,
    max_deg: float = 180.0,
    num_steps_per_env: int = 24,
) -> dict[str, torch.Tensor]:
    """Grow ori_range from start_deg → max_deg as training progresses.

    Schedule (training iterations = common_step_counter // num_steps_per_env):
      [0, hold_iters)                       → ±start_deg
      [hold_iters, hold_iters+step_iters)   → ±(start_deg + step_deg)
      every further step_iters              → +step_deg … until ±max_deg
    """
    del env_ids  # unused — curriculum applies globally
    command_term = env.command_manager.get_term(command_name)
    cfg: ReachPoseCommandCfg = command_term.cfg  # type: ignore[assignment]

    train_iter = env.common_step_counter // num_steps_per_env
    max_rad = max_deg * _DEG_TO_RAD

    if train_iter < hold_iters:
        current_rad = start_deg * _DEG_TO_RAD
    else:
        increments = (train_iter - hold_iters) // step_iters + 1
        current_rad = min(
            start_deg * _DEG_TO_RAD + increments * step_deg * _DEG_TO_RAD,
            max_rad,
        )

    cfg.ori_range = (-current_rad, current_rad)
    return {"ori_range_deg": torch.tensor(current_rad / _DEG_TO_RAD)}


def reach_pos_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str = "reach_pose",
    start_m: float = 0.05,
    hold_iters: int = 200,
    step_iters: int = 100,
    step_m: float = 0.05,
    max_m: float = 0.5,
    num_steps_per_env: int = 24,
) -> dict[str, torch.Tensor]:
    """Grow target_pos_delta_max from start_m → max_m as training progresses.

    Schedule (training iterations = common_step_counter // num_steps_per_env):
      [0, hold_iters)                       → start_m
      [hold_iters, hold_iters+step_iters)   → start_m + step_m
      every further step_iters              → +step_m … until max_m
    """
    del env_ids
    command_term = env.command_manager.get_term(command_name)
    cfg: ReachPoseCommandCfg = command_term.cfg  # type: ignore[assignment]

    train_iter = env.common_step_counter // num_steps_per_env

    if train_iter < hold_iters:
        current = start_m
    else:
        increments = (train_iter - hold_iters) // step_iters + 1
        current = min(start_m + increments * step_m, max_m)

    cfg.target_pos_delta_max = current
    return {"target_pos_delta_max_m": torch.tensor(current)}


def reach_joint_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str = "reset_robot_joints",
    start_deg: float = 10.0,
    hold_iters: int = 200,
    step_iters: int = 100,
    step_deg: float = 10.0,
    max_deg: float = 180.0,
    num_steps_per_env: int = 24,
) -> dict[str, torch.Tensor]:
    """Grow joint_delta_max from start_deg → max_deg as training progresses.

    Schedule (training iterations = common_step_counter // num_steps_per_env):
      [0, hold_iters)                       → start_deg
      [hold_iters, hold_iters+step_iters)   → start_deg + step_deg
      every further step_iters              → +step_deg … until max_deg
    """
    del env_ids
    event_cfg = env.event_manager.get_term_cfg(event_name)

    train_iter = env.common_step_counter // num_steps_per_env
    max_rad = max_deg * _DEG_TO_RAD

    if train_iter < hold_iters:
        current_rad = start_deg * _DEG_TO_RAD
    else:
        increments = (train_iter - hold_iters) // step_iters + 1
        current_rad = min(
            start_deg * _DEG_TO_RAD + increments * step_deg * _DEG_TO_RAD,
            max_rad,
        )

    event_cfg.params["joint_delta_max"] = current_rad
    return {"joint_delta_max_deg": torch.tensor(current_rad / _DEG_TO_RAD)}


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


def kinova_reach_osc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Kinova Gen3 reach task with OSC torque control (arm only, no gripper).

    Goal: move pinch_site to a randomised 6D target pose. The target is
    resampled only on episode reset (resampling_time_range > episode_length_s).

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
        "ee_pose": ObservationTermCfg(func=ee_pose),
        "target_pose": ObservationTermCfg(
            func=target_pose,
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
            delta_pos_scale=0.02,      # 2 cm per unit action
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
            resampling_time_range=(1e9, 1e9),  # only resample on reset
            debug_vis=True,
            pos_lo=(-1.0, -1.0, -0.3),
            pos_hi=( 1.0,  1.0,  1.2),
            ori_range=(-3.14159, 3.14159),
            target_pos_delta_max=0.20,  # seeded; grown by reach_pos_curriculum
            pos_threshold=0.02,
        ),
    }

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        ),
        "reset_robot_joints": EventTermCfg(
            func=reset_joints_random,
            mode="reset",
            params={"entity_name": "robot", "joint_delta_max": 10.0 * _DEG_TO_RAD},  # seeded; grown by reach_joint_curriculum
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
            params={"std": 0.05, "command_name": "reach_pose"},
        ),
        "ori": RewardTermCfg(
            func=gated_ori_reward,
            weight=1.0,
            params={"pos_std": 0.3, "ori_std": 0.4, "command_name": "reach_pose"},
        ),
        "ori_precise": RewardTermCfg(
            func=gated_ori_reward,
            weight=1.0,
            params={"pos_std": 0.3, "ori_std": 0.1, "command_name": "reach_pose"},
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_detection": TerminationTermCfg(func=mdp.nan_detection, time_out=False),
    }

    curriculum = {
        "ori_range": CurriculumTermCfg(
            func=reach_ori_curriculum,
            params={
                "command_name": "reach_pose",
                "start_deg": 10.0,
                "hold_iters": 200,
                "step_iters": 100,
                "step_deg": 5.0,
                "max_deg": 20.0,
                "num_steps_per_env": 24,
            },
        ),
        "target_pos_delta": CurriculumTermCfg(
            func=reach_pos_curriculum,
            params={
                "command_name": "reach_pose",
                "start_m": 0.20,
                "hold_iters": 200,
                "step_iters": 100,
                "step_m": 0.10,
                "max_m": 1.0,
                "num_steps_per_env": 24,
            },
        ),
        "joint_delta": CurriculumTermCfg(
            func=reach_joint_curriculum,
            params={
                "event_name": "reset_robot_joints",
                "start_deg": 10.0,
                "hold_iters": 200,
                "step_iters": 100,
                "step_deg": 5.0,
                "max_deg": 20.0,
                "num_steps_per_env": 24,
            },
        ),
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
            spec_fn=lambda spec: [
                setattr(g, "contype", 0) or setattr(g, "conaffinity", 0)
                for g in spec.geoms if g.name == "terrain"
            ],
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
        curriculum=curriculum,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=2.2,
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
