"""Kinova Gen3 peg-in-hole task with cartesian (differential IK) actions.

Task: Insert peg (held in gripper) into hole (fixed on ground).
The robot uses cartesian home-relative IK control to position the EE
above and into the hole. The gripper stays closed throughout the episode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from kinova_tasks.assets.kinova_gen3 import get_kinova_robot_cfg_peginhole
from kinova_tasks.assets.peg_in_hole import get_hole_cfg, get_peg_cfg
from kinova_tasks.tasks.actions import HomeRelativeIKActionCfg
from kinova_tasks.tasks.base_rl_cfg import kinova_ppo_runner_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.lab_api.math import sample_uniform
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Custom reset events for peg-in-hole
# ---------------------------------------------------------------------------


def reset_peg_in_gripper(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    peg_entity_name: str = "peg",
    pinch_pos_local: tuple[float, float, float] = (-0.024850, -0.482624, 0.174564),
) -> None:
    """Place the peg at the known pinch_site position (between gripper fingers).

    Uses the pre-computed FK position of pinch_site at the init joint config
    rather than reading site_pos_w, which is stale during reset (no mj_forward
    has been called yet after joint positions are written).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    peg: Entity = env.scene[peg_entity_name]
    n = len(env_ids)

    # Known pinch_site position (local frame) + env origin → world frame
    pos = torch.tensor(pinch_pos_local, device=env.device).unsqueeze(0).expand(n, -1).clone()
    pos = pos + env.scene.env_origins[env_ids]

    # Identity quaternion (peg upright)
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0  # w=1

    pose = torch.cat([pos, quat], dim=-1)  # (n, 7)
    vel = torch.zeros(n, 6, device=env.device)

    peg.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    peg.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


def reset_hole_position(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    hole_entity_name: str = "hole",
    x_range: tuple[float, float] = (-0.02, -0.02),
    y_range: tuple[float, float] = (-0.5, -0.5),
    z_range: tuple[float, float] = (0.02, 0.02),
) -> None:
    """Randomize hole mocap position on reset."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    hole: Entity = env.scene[hole_entity_name]
    n = len(env_ids)

    lower = torch.tensor([x_range[0], y_range[0], z_range[0]], device=env.device)
    upper = torch.tensor([x_range[1], y_range[1], z_range[1]], device=env.device)
    pos = sample_uniform(lower, upper, (n, 3), device=env.device)
    pos = pos + env.scene.env_origins[env_ids]

    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0  # identity

    pose = torch.cat([pos, quat], dim=-1)
    hole.write_mocap_pose_to_sim(pose, env_ids=env_ids)


def site_pos_relative_to_home(
    env: ManagerBasedRlEnv,
    entity_name: str,
    site_names: tuple[str, ...],
    home_pos: tuple[float, float, float] = (-0.024850, -0.482624, 0.174564),
) -> torch.Tensor:
    """Return entity site positions relative to the IK home pose.

    Returns a flat vector of (num_sites * 3) values: [site0_xyz, site1_xyz, ...].
    """
    entity: Entity = env.scene[entity_name]
    # home_pos is in local (robot-base) frame; convert to world
    home_w = (
        torch.tensor(home_pos, device=env.device)
        .unsqueeze(0)
        .expand(env.num_envs, -1)
        + env.scene.env_origins
    )

    parts = []
    for sname in site_names:
        sid = entity.find_sites(sname)[0]
        pos_w = entity.data.site_pos_w[:, sid].squeeze(1)  # (num_envs, 3)
        parts.append(pos_w - home_w)

    return torch.cat(parts, dim=-1)  # (num_envs, num_sites * 3)


def ee_to_sites_distance(
    env: ManagerBasedRlEnv,
    target_entity: str,
    target_site_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Vector from EE (pinch_site) to each target site. Returns (num_sites * 3)."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)  # (N, 3)

    target: Entity = env.scene[target_entity]
    parts = []
    for sname in target_site_names:
        sid = target.find_sites(sname)[0]
        site_pos = target.data.site_pos_w[:, sid].squeeze(1)  # (N, 3)
        parts.append(site_pos - ee_pos)

    return torch.cat(parts, dim=-1)


def peg_to_hole_sites_distance(
    env: ManagerBasedRlEnv,
    peg_entity: str = "peg",
    hole_entity: str = "hole",
    peg_site_names: tuple[str, ...] = ("cylinder_start", "cylinder_end"),
    hole_site_names: tuple[str, ...] = ("hole_top", "hole_bottom"),
) -> torch.Tensor:
    """Vector from each peg site to corresponding hole site. Returns (num_pairs * 3)."""
    peg: Entity = env.scene[peg_entity]
    hole: Entity = env.scene[hole_entity]

    parts = []
    for pname, hname in zip(peg_site_names, hole_site_names):
        pid = peg.find_sites(pname)[0]
        hid = hole.find_sites(hname)[0]
        peg_pos = peg.data.site_pos_w[:, pid].squeeze(1)
        hole_pos = hole.data.site_pos_w[:, hid].squeeze(1)
        parts.append(hole_pos - peg_pos)

    return torch.cat(parts, dim=-1)


def peg_to_hole_reward(
    env: ManagerBasedRlEnv,
    std: float,
    peg_entity: str = "peg",
    hole_entity: str = "hole",
    peg_site_names: tuple[str, ...] = ("cylinder_start", "cylinder_end"),
    hole_site_names: tuple[str, ...] = ("hole_top", "hole_bottom"),
) -> torch.Tensor:
    """Gaussian reward for peg sites proximity to corresponding hole sites."""
    peg: Entity = env.scene[peg_entity]
    hole: Entity = env.scene[hole_entity]

    error = torch.zeros(env.num_envs, device=env.device)
    for pname, hname in zip(peg_site_names, hole_site_names):
        pid = peg.find_sites(pname)[0]
        hid = hole.find_sites(hname)[0]
        peg_pos = peg.data.site_pos_w[:, pid].squeeze(1)
        hole_pos = hole.data.site_pos_w[:, hid].squeeze(1)
        error += torch.sum(torch.square(peg_pos - hole_pos), dim=-1)

    return torch.nan_to_num(torch.exp(-error / std**2), nan=0.0)


GRIPPER_CLOSED_JOINT_POS = {
    "right_driver_joint": 0.503,
    "right_coupler_joint": 0.001,
    "right_spring_link_joint": 0.505,
    "right_follower_joint": -0.485,
    "left_driver_joint": 0.503,
    "left_coupler_joint": 0.001,
    "left_spring_link_joint": 0.505,
    "left_follower_joint": -0.485,
}

GRIPPER_JOINT_NAMES = tuple(GRIPPER_CLOSED_JOINT_POS.keys())


def peg_out_of_bounds(
    env: ManagerBasedRlEnv,
    peg_entity: str = "peg",
    hole_entity: str = "hole",
    peg_site_name: str = "cylinder_end",
    home_pos: tuple[float, float, float] = (-0.024850, -0.482624, 0.174564),
    workspace_half: tuple[float, float, float] = (0.12, 0.12, 0.12),
    hole_half: tuple[float, float, float] = (0.03, 0.03, 0.04),
) -> torch.Tensor:
    """Terminate if peg lower site is outside both workspace and hole boxes.

    The peg must be inside at least one box to stay alive.

    Box 1 (workspace): centered on home_pos (static per env), defined by
        half-extents workspace_half.
    Box 2 (hole): centered on current hole position (moves per env),
        defined by half-extents hole_half.
    """
    peg: Entity = env.scene[peg_entity]
    hole: Entity = env.scene[hole_entity]

    sid = peg.find_sites(peg_site_name)[0]
    pos_w = peg.data.site_pos_w[:, sid].squeeze(1)  # (N, 3)
    pos_local = pos_w - env.scene.env_origins  # to local frame

    # Workspace box: centered on home pose
    home = torch.tensor(home_pos, device=env.device)
    ws_half = torch.tensor(workspace_half, device=env.device)
    in_workspace = ((pos_local >= home - ws_half) & (pos_local <= home + ws_half)).all(dim=-1)

    # Hole box: centered on current hole position (per env)
    hole_pos_local = hole.data.root_link_pos_w - env.scene.env_origins  # (N, 3)
    h_half = torch.tensor(hole_half, device=env.device)
    in_hole = ((pos_local >= hole_pos_local - h_half) & (pos_local <= hole_pos_local + h_half)).all(dim=-1)

    return ~(in_workspace | in_hole)


def peg_to_hole_error(
    env: ManagerBasedRlEnv,
    peg_entity: str = "peg",
    hole_entity: str = "hole",
    peg_site_names: tuple[str, ...] = ("cylinder_start", "cylinder_end"),
    hole_site_names: tuple[str, ...] = ("hole_top", "hole_bottom"),
) -> torch.Tensor:
    """Mean Euclidean distance from peg sites to corresponding hole sites."""
    peg: Entity = env.scene[peg_entity]
    hole: Entity = env.scene[hole_entity]

    total = torch.zeros(env.num_envs, device=env.device)
    for pname, hname in zip(peg_site_names, hole_site_names):
        pid = peg.find_sites(pname)[0]
        hid = hole.find_sites(hname)[0]
        peg_pos = peg.data.site_pos_w[:, pid].squeeze(1)
        hole_pos = hole.data.site_pos_w[:, hid].squeeze(1)
        total += torch.norm(peg_pos - hole_pos, dim=-1)

    result = total / len(peg_site_names)
    return result


def reset_workspace_bounds(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    entity_name: str = "workspace_bounds",
    home_pos: tuple[float, float, float] = (-0.024850, -0.482624, 0.174564),
) -> None:
    """Position workspace bounds visualization at home pose."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    entity: Entity = env.scene[entity_name]
    n = len(env_ids)

    pos = torch.tensor(home_pos, device=env.device).unsqueeze(0).expand(n, -1).clone()
    pos = pos + env.scene.env_origins[env_ids]

    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0

    pose = torch.cat([pos, quat], dim=-1)
    entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)


def reset_gripper_joints_closed(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=GRIPPER_JOINT_NAMES
    ),
) -> None:
    """Write consistent closed-position values for all 8 gripper joints.

    These values were computed from the 4-bar linkage equilibrium at ctrl=255.
    Written directly (no soft-limit clamping) so the gripper starts closed.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    robot: Entity = env.scene[asset_cfg.name]
    n = len(env_ids)
    nj = len(GRIPPER_CLOSED_JOINT_POS)

    pos = torch.tensor(
        list(GRIPPER_CLOSED_JOINT_POS.values()), device=env.device
    ).unsqueeze(0).expand(n, -1).clone()
    vel = torch.zeros(n, nj, device=env.device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    robot.write_joint_state_to_sim(pos, vel, joint_ids=joint_ids, env_ids=env_ids)


def reset_gripper_ctrl_closed(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", actuator_names=("fingers_actuator",)
    ),
    closed_ctrl: float = 191.25,
) -> None:
    """Set the fingers_actuator ctrl to 60% closed (191.25).

    The Robotiq 2F-85 uses a single tendon-based actuator:
    ctrl=0 → open, ctrl=255 → closed (driver joints at 0.8 rad).
    This must be called on reset so the actuator actively holds the
    gripper closed (joint init state alone is not enough since ctrl
    defaults to 0).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    robot: Entity = env.scene[asset_cfg.name]
    n = len(env_ids)
    num_actuators = len(asset_cfg.actuator_ids) if isinstance(asset_cfg.actuator_ids, list) else 1
    ctrl = torch.full((n, num_actuators), closed_ctrl, device=env.device)
    robot.data.write_ctrl(ctrl, ctrl_ids=asset_cfg.actuator_ids, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


def kinova_peg_in_hole_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Kinova peg-in-hole with cartesian (IK) control.

    The peg is placed in the gripper on reset. The gripper stays closed
    throughout the episode (no gripper action). The policy only controls
    the 6D IK pose (position + orientation offsets from home).

    Actions:
        - ik_pose (6D): position + axis-angle deltas relative to home EE pose
    """

    # --- Observations ---
    actor_terms = {
        # Peg start→hole top, peg end→hole bottom (6D)
        "peg_to_hole": ObservationTermCfg(
            func=peg_to_hole_sites_distance,
            params={
                "peg_entity": "peg",
                "hole_entity": "hole",
                "peg_site_names": ("cylinder_start", "cylinder_end"),
                "hole_site_names": ("hole_top", "hole_bottom"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
    }

    critic_terms = {
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[1-7]",)),
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        # EE to peg cylinder_start (3D)
        "ee_to_peg": ObservationTermCfg(
            func=ee_to_sites_distance,
            params={
                "target_entity": "peg",
                "target_site_names": ("cylinder_start",),
                "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        # EE to hole top + bottom sites (6D)
        "ee_to_hole": ObservationTermCfg(
            func=ee_to_sites_distance,
            params={
                "target_entity": "hole",
                "target_site_names": ("hole_top", "hole_bottom"),
                "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        # Peg start→hole top, peg end→hole bottom (6D)
        "peg_to_hole": ObservationTermCfg(
            func=peg_to_hole_sites_distance,
            params={
                "peg_entity": "peg",
                "hole_entity": "hole",
                "peg_site_names": ("cylinder_start", "cylinder_end"),
                "hole_site_names": ("hole_top", "hole_bottom"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        # Hole top + bottom site positions relative to home (6D)
        "hole_pos_home": ObservationTermCfg(
            func=site_pos_relative_to_home,
            params={
                "entity_name": "hole",
                "site_names": ("hole_top", "hole_bottom"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        # Peg start + end site positions relative to home (6D)
        "peg_pos_home": ObservationTermCfg(
            func=site_pos_relative_to_home,
            params={
                "entity_name": "peg",
                "site_names": ("cylinder_start", "cylinder_end"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    # --- Actions: Position-only IK (no rotation control, no gripper action) ---
    actions: dict[str, ActionTermCfg] = {
        "ik_pose": HomeRelativeIKActionCfg(
            entity_name="robot",
            actuator_names=("joint_.*",),  # Arm joints only
            frame_name="pinch_site",
            frame_type="site",
            # Home EE pose from default joint config (pinch_site FK)
            home_pos=(-0.024850, -0.482624, 0.174564),
            home_quat=(0.0, 1.0, 0.0, 0.0),
            damping=0.05,
            max_dq=0.5,
            position_weight=1.0,
            orientation_weight=0.0,  # No orientation control
            joint_limit_weight=0.1,
            posture_weight=0.02,
            pos_scale=1.0,  # 1:1 action to position mapping (meters)
            ori_scale=0.0,  # Orientation actions ignored (3D action space)
            max_pos_delta=0.5,  # Clip action delta to 0.5m between steps
        ),
    }

    # --- Events ---
    # Order matters: reset robot joints first, then close gripper, then place peg
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[1-7]",)),
            },
        ),
        # Write all 8 gripper joints to consistent closed state
        "reset_gripper_joints": EventTermCfg(
            func=reset_gripper_joints_closed,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=GRIPPER_JOINT_NAMES
                ),
            },
        ),
        # Set fingers_actuator ctrl to 60% closed (191.25) so gripper stays shut
        "reset_gripper_ctrl": EventTermCfg(
            func=reset_gripper_ctrl_closed,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", actuator_names=("fingers_actuator",)
                ),
                "closed_ctrl": 191.25,
            },
        ),
        # Position hole (mocap) on ground
        "reset_hole_position": EventTermCfg(
            func=reset_hole_position,
            mode="reset",
            params={
                "hole_entity_name": "hole",
                "x_range": (-0.04, 0.04),
                "y_range": (-0.52, -0.48),
                "z_range": (0.02, 0.02),
            },
        ),
        # Place peg at known pinch_site position
        "reset_peg_in_gripper": EventTermCfg(
            func=reset_peg_in_gripper,
            mode="reset",
            params={
                "peg_entity_name": "peg",
                "pinch_pos_local": (-0.024850, -0.482624, 0.174564),
            },
        ),
        # Fingertip friction randomization (Robotiq 2F-85 pads)
        "fingertip_friction_slide": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", geom_names=r"(left|right)_pad[12]"
                ),
                "operation": "abs",
                "field": "geom_friction",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.3, 1.5),
            },
        ),
        "fingertip_friction_spin": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", geom_names=r"(left|right)_pad[12]"
                ),
                "operation": "abs",
                "field": "geom_friction",
                "distribution": "log_uniform",
                "axes": [1],
                "ranges": (1e-4, 2e-2),
            },
        ),
        "fingertip_friction_roll": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", geom_names=r"(left|right)_pad[12]"
                ),
                "operation": "abs",
                "field": "geom_friction",
                "distribution": "log_uniform",
                "axes": [2],
                "ranges": (1e-5, 5e-3),
            },
        ),
    }

    # --- Rewards ---
    rewards = {
        # Peg→hole approach (Gaussian, std=0.1)
        "peg_to_hole": RewardTermCfg(
            func=peg_to_hole_reward,
            weight=1.0,
            params={"std": 0.1},
        ),
        # Tight insertion bonus (Gaussian, std=0.02)
        "insertion_precise": RewardTermCfg(
            func=peg_to_hole_reward,
            weight=2.0,
            params={"std": 0.02},
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.01,  # Will be increased via curriculum
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[1-7]",)),
            },
        ),
        "joint_vel_hinge": RewardTermCfg(
            func=manipulation_mdp.joint_velocity_hinge_penalty,
            weight=-0.01,  # Will be increased via curriculum
            params={
                "max_vel": 0.5,
                "asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[1-7]",)),
            },
        ),
    }

    # --- Terminations ---
    ee_ground_collision_cfg = ContactSensorCfg(
        name="ee_ground_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern="bracelet_link",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_detection": TerminationTermCfg(func=nan_detection, time_out=False),
        "ee_ground_collision": TerminationTermCfg(
            func=manipulation_mdp.illegal_contact,
            params={"sensor_name": "ee_ground_collision"},
        ),
        "peg_out_of_bounds": TerminationTermCfg(
            func=peg_out_of_bounds,
            params={
                "peg_entity": "peg",
                "hole_entity": "hole",
                "peg_site_name": "cylinder_end",
                "home_pos": (-0.024850, -0.482624, 0.174564),
                "workspace_half": (0.12, 0.12, 0.12),  # 12cm workspace
                "hole_half": (0.015, 0.015, 0.06),
            },
        ),
    }

    # --- Metrics ---
    metrics = {
        "peg_to_hole_error": MetricsTermCfg(
            func=peg_to_hole_error,
            params={
                "peg_entity": "peg",
                "hole_entity": "hole",
                "peg_site_names": ("cylinder_start", "cylinder_end"),
                "hole_site_names": ("hole_top", "hole_bottom"),
            },
        ),
    }

    # --- Curriculum ---
    curriculum = {
        "action_rate_l2_weight": CurriculumTermCfg(
            func=manipulation_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.01},
                    {"step": 2400, "weight": -0.04},
                    {"step": 4800, "weight": -0.07},
                    {"step": 7200, "weight": -0.10},
                ],
            },
        ),
        "joint_vel_hinge_weight": CurriculumTermCfg(
            func=manipulation_mdp.reward_weight,
            params={
                "reward_name": "joint_vel_hinge",
                "weight_stages": [
                    {"step": 0, "weight": -0.01},
                    {"step": 2400, "weight": -0.04},
                    {"step": 4800, "weight": -0.07},
                    {"step": 7200, "weight": -0.10},
                ],
            },
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(terrain_type="plane"),
            num_envs=4096,
            env_spacing=1.0,
            entities={
                "robot": get_kinova_robot_cfg_peginhole(),
                "peg": get_peg_cfg(),
                "hole": get_hole_cfg(),
            },
            sensors=(ee_ground_collision_cfg,),
        ),
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=1.5,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=55,
            njmax=600,
            mujoco=MujocoCfg(
                timestep=0.01,  # 100 Hz simulation/IK rate
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=10,  # 10 Hz policy rate (100/10)
        episode_length_s=10.0,
    )

    # Play mode overrides
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

    return cfg


def kinova_peg_in_hole_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for peg-in-hole task."""
    return kinova_ppo_runner_cfg(experiment_name="kinova_peg_in_hole")
