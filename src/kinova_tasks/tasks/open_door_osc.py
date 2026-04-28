"""Kinova Gen3 door-opening task with Operational Space Control (OSC, relative mode).

Task: Grasp the door handle and open the door to a randomly sampled target angle.
The policy controls the 6D EE pose (OSC relative mode) plus a 1D gripper open/close.

Actions (7D):
    - osc_pose (6D): relative position + axis-angle delta applied to current EE pose
    - gripper (1D): -1 = fully open, +1 = fully closed
"""

from __future__ import annotations

import math as _math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import mjlab.envs.mdp as mdp
from kinova_tasks.assets.kinova_gen3 import get_kinova_robot_cfg_peginhole_osc
from kinova_tasks.assets.objects.articulated.door.door_constants import get_door_cfg
from kinova_tasks.tasks.actions.osc import OperationalSpaceActionCfg
from kinova_tasks.tasks.base_rl_cfg import kinova_ppo_runner_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
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
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import axis_angle_from_quat, sample_uniform
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Home joint positions for open-door task: joint_1 = 0° (arm facing forward)
_HOME_JOINT_POS = (
    0.0,            # joint_1   0°
    0.5235987756,   # joint_2  30°
    0.0,            # joint_3   0°
    1.5707963268,   # joint_4  90°
    0.0,            # joint_5   0°
    1.0471975512,   # joint_6  60°
    -1.5707963268,  # joint_7 -90°
)
_DEG_TO_RAD = _math.pi / 180.0

# Gripper driver joint range [0, 0.8] (0=open, 0.8=closed)
_GRIPPER_DRIVER_MAX = 0.8

GRIPPER_OPEN_JOINT_POS = {
    "right_driver_joint": 0.0,
    "right_coupler_joint": 0.0,
    "right_spring_link_joint": 0.0,
    "right_follower_joint": 0.0,
    "left_driver_joint": 0.0,
    "left_coupler_joint": 0.0,
    "left_spring_link_joint": 0.0,
    "left_follower_joint": 0.0,
}
GRIPPER_JOINT_NAMES = tuple(GRIPPER_OPEN_JOINT_POS.keys())

# Door spawn range (local frame) — door_base mocap body position
# x: ±2 cm, y: 10 cm range (matching mjlab_old randomisation tightness)
_DOOR_POS_LO = (0.7, -0.05, 0.45)
_DOOR_POS_HI = ( 0.9, 0.15, 0.65)

# Door initial hinge angle range at reset (radians): 0° – 80°
_DOOR_INIT_ANGLE_LO =  0.0 * _DEG_TO_RAD
_DOOR_INIT_ANGLE_HI = 10.0 * _DEG_TO_RAD

# Door goal angle range (radians): 60° – 90°
_DOOR_GOAL_LO = 60.0 * _DEG_TO_RAD
_DOOR_GOAL_HI = 90.0 * _DEG_TO_RAD

# Door hinge geometry (from door.xml, used for target-handle vis)
# Hinge pivot in door_base frame: (0, -0.3, 0)
# Object_site relative to hinge pivot: (-0.04, 0.55, 0)
_HINGE_OFFSET = (0.0, -0.3, 0.0)          # pivot in door_base frame
_HANDLE_FROM_HINGE = (-0.04, 0.55, 0.0)   # handle relative to hinge at θ=0


# ---------------------------------------------------------------------------
# Gripper action term (identical to pick_cube_osc)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class GripperCtrlActionCfg(ActionTermCfg):
    """1D gripper action: maps policy output to fingers_actuator ctrl."""

    actuator_name: str = "fingers_actuator"
    ctrl_min: float = 0.0    # fully open
    ctrl_max: float = 255.0  # fully closed

    def build(self, env: ManagerBasedRlEnv) -> GripperCtrlAction:
        return GripperCtrlAction(self, env)


class GripperCtrlAction(ActionTerm):
    """Maps a 1D policy action to the Robotiq fingers_actuator ctrl signal."""

    cfg: GripperCtrlActionCfg

    def __init__(self, cfg: GripperCtrlActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # Resolve actuator IDs the same way SceneEntityCfg does in event functions,
        # so write_ctrl receives proper local ctrl indices (not raw find_actuators output).
        self._gripper_asset_cfg = SceneEntityCfg(
            cfg.entity_name, actuator_names=(cfg.actuator_name,)
        )
        self._gripper_asset_cfg.resolve(env.scene)
        self._raw_actions = torch.zeros(env.num_envs, 1, device=env.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone()

    def apply_actions(self) -> None:
        robot: Entity = self._env.scene[self.cfg.entity_name]
        ctrl_mid = (self.cfg.ctrl_min + self.cfg.ctrl_max) * 0.5
        ctrl_half = (self.cfg.ctrl_max - self.cfg.ctrl_min) * 0.5
        # Map [-1, 1] → [ctrl_min, ctrl_max]
        ctrl = ctrl_mid + self._raw_actions.clamp(-1.0, 1.0) * ctrl_half
        robot.data.write_ctrl(ctrl, ctrl_ids=self._gripper_asset_cfg.actuator_ids)


# ---------------------------------------------------------------------------
# Door goal command term
# ---------------------------------------------------------------------------


class DoorGoalCommand(CommandTerm):
    """Samples and maintains a target door hinge angle for the open-door task."""

    cfg: DoorGoalCommandCfg

    def __init__(self, cfg: DoorGoalCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.goal_angle = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Target door hinge angle (rad), shape (num_envs, 1)."""
        return self.goal_angle

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        lo = torch.full((len(env_ids), 1), self.cfg.angle_lo, device=self.device)
        hi = torch.full((len(env_ids), 1), self.cfg.angle_hi, device=self.device)
        self.goal_angle[env_ids] = lo + (hi - lo) * torch.rand_like(lo)

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        import numpy as np

        env_indices = visualizer.get_env_indices(self._env.num_envs)
        if not env_indices:
            return

        door: Entity = self._env.scene["door"]
        hx, hy, hz = _HINGE_OFFSET
        dx, dy, dz = _HANDLE_FROM_HINGE

        # Door spawn range edges (blue rectangle, drawn once per env)
        # Corners in local frame; z lifted slightly above floor for visibility
        spawn_lo = np.array([_DOOR_POS_LO[0], _DOOR_POS_LO[1], 0.02], dtype=np.float32)
        spawn_hi = np.array([_DOOR_POS_HI[0], _DOOR_POS_HI[1], 0.02], dtype=np.float32)
        spawn_corners = np.array([
            [spawn_lo[0], spawn_lo[1], spawn_lo[2]],
            [spawn_hi[0], spawn_lo[1], spawn_lo[2]],
            [spawn_lo[0], spawn_hi[1], spawn_lo[2]],
            [spawn_hi[0], spawn_hi[1], spawn_lo[2]],
        ], dtype=np.float32)
        spawn_edges = [(0, 1), (0, 2), (1, 3), (2, 3)]

        for i in env_indices:
            origin = self._env.scene.env_origins[i].cpu().numpy()

            # Door spawn bounding box (blue)
            for idx, (a, b) in enumerate(spawn_edges):
                visualizer.add_cylinder(
                    start=spawn_corners[a] + origin,
                    end=spawn_corners[b] + origin,
                    radius=0.004, color=(0.2, 0.5, 1.0, 0.5),
                    label=f"door_spawn_edge_{i}_{idx}",
                )

            door_pos = door.data.root_link_pos_w[i].cpu().numpy()

            # Current handle position (blue sphere, same as object_at_goal_reward vis)
            handle_ids = door.find_sites("object_site")[0]
            handle_pos_np = door.data.site_pos_w[i, handle_ids].squeeze(0).cpu().numpy()
            visualizer.add_sphere(
                center=handle_pos_np, radius=0.022,
                color=(0.2, 0.5, 1.0, 0.6), label=f"handle_current_{i}",
            )

            # Goal handle position (orange sphere)
            hinge_w = door_pos + np.array([hx, hy, hz], dtype=np.float32)
            goal_θ = float(self.goal_angle[i, 0].item())
            cos_θ, sin_θ = _math.cos(goal_θ), _math.sin(goal_θ)
            tx = dx * cos_θ - dy * sin_θ
            ty = dx * sin_θ + dy * cos_θ
            target_handle_w = hinge_w + np.array([tx, ty, dz], dtype=np.float32)

            visualizer.add_sphere(
                center=target_handle_w, radius=0.03,
                color=(1.0, 0.5, 0.0, 0.4), label=f"door_goal_{i}",
            )
            visualizer.add_sphere(
                center=target_handle_w, radius=0.008,
                color=(1.0, 0.6, 0.0, 0.9), label=f"door_goal_dot_{i}",
            )



@dataclass(kw_only=True)
class DoorGoalCommandCfg(CommandTermCfg):
    """Configuration for the door goal command term."""

    angle_lo: float = _DOOR_GOAL_LO
    angle_hi: float = _DOOR_GOAL_HI

    def build(self, env: ManagerBasedRlEnv) -> DoorGoalCommand:
        return DoorGoalCommand(self, env)


def _get_door_goal_command(env: ManagerBasedRlEnv, command_name: str = "door_goal") -> DoorGoalCommand:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, DoorGoalCommand)
    return term


# ---------------------------------------------------------------------------
# Reset functions
# ---------------------------------------------------------------------------


def reset_joints_with_delta(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    entity_name: str = "robot",
    joint_delta_deg: float = 5.0,
) -> None:
    """Reset arm joints to home ± uniform delta (degrees)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    robot: Entity = env.scene[entity_name]
    n = len(env_ids)
    home = torch.tensor(_HOME_JOINT_POS, device=env.device).unsqueeze(0).expand(n, -1)
    delta_rad = joint_delta_deg * _DEG_TO_RAD
    delta = (torch.rand(n, len(_HOME_JOINT_POS), device=env.device) * 2 - 1) * delta_rad

    soft_limits = robot.data.soft_joint_pos_limits
    lo = soft_limits[env_ids, :7, 0]
    hi = soft_limits[env_ids, :7, 1]
    joint_pos = torch.clamp(home + delta, lo, hi)
    joint_vel = torch.zeros_like(joint_pos)

    arm_joint_ids = torch.arange(7, device=env.device)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=arm_joint_ids, env_ids=env_ids)


def reset_gripper_open(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    joint_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES),
    ctrl_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", actuator_names=("fingers_actuator",)),
) -> None:
    """Reset gripper to fully open."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    robot: Entity = env.scene[joint_asset_cfg.name]
    n = len(env_ids)
    nj = len(GRIPPER_OPEN_JOINT_POS)

    pos = torch.zeros(n, nj, device=env.device)
    vel = torch.zeros(n, nj, device=env.device)

    joint_ids = joint_asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)
    robot.write_joint_state_to_sim(pos, vel, joint_ids=joint_ids, env_ids=env_ids)

    num_actuators = len(ctrl_asset_cfg.actuator_ids) if isinstance(ctrl_asset_cfg.actuator_ids, list) else 1
    ctrl = torch.zeros(n, num_actuators, device=env.device)
    robot.data.write_ctrl(ctrl, ctrl_ids=ctrl_asset_cfg.actuator_ids, env_ids=env_ids)


def reset_door(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    door_entity_name: str = "door",
    x_range: tuple[float, float] = (_DOOR_POS_LO[0], _DOOR_POS_HI[0]),
    y_range: tuple[float, float] = (_DOOR_POS_LO[1], _DOOR_POS_HI[1]),
    z: float = _DOOR_POS_LO[2],
    init_angle_range: tuple[float, float] = (_DOOR_INIT_ANGLE_LO, _DOOR_INIT_ANGLE_HI),
) -> None:
    """Reset door base position and hinge joint to a randomly sampled initial angle."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    door: Entity = env.scene[door_entity_name]
    n = len(env_ids)

    # Sample door base position (mocap body — use write_mocap_pose_to_sim)
    lower = torch.tensor([x_range[0], y_range[0], z], device=env.device)
    upper = torch.tensor([x_range[1], y_range[1], z], device=env.device)
    pos = sample_uniform(lower, upper, (n, 3), device=env.device)
    pos = pos + env.scene.env_origins[env_ids]

    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0  # identity orientation
    pose = torch.cat([pos, quat], dim=-1)  # (N, 7)
    door.write_mocap_pose_to_sim(pose, env_ids=env_ids)

    # Sample hinge initial angle uniformly from init_angle_range
    jpos = sample_uniform(
        torch.tensor([init_angle_range[0]], device=env.device),
        torch.tensor([init_angle_range[1]], device=env.device),
        (n, 1),
        device=env.device,
    )
    jvel = torch.zeros(n, 1, device=env.device)
    door.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Observation functions
# Mirrors pick_cube_osc: joint_vel, ee_pose, gripper_state,
#   ee_to_object, object_pos, object_to_goal, goal_pos, actions
# The door handle (object_site) is treated as the manipulated "object".
# Its position is read from site kinematics — MuJoCo propagates the hinge
# joint through FK so door.data.site_pos_w already reflects the current angle.
# ---------------------------------------------------------------------------


def joint_vel(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Absolute arm joint velocities (7D, rad/s)."""
    robot: Entity = env.scene[entity_name]
    return robot.data.joint_vel[:, :7]


def ee_pose(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> torch.Tensor:
    """Current EE pose [pos(3), axis_angle(3)] in local frame (6D)."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    pos = robot.data.site_pos_w[:, site_ids].squeeze(1) - env.scene.env_origins
    quat = robot.data.site_quat_w[:, site_ids].squeeze(1)
    return torch.cat([pos, axis_angle_from_quat(quat)], dim=-1)


def gripper_state(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=("right_driver_joint",)),
) -> torch.Tensor:
    """Gripper openness normalized to [0=open, 1=closed] (1D)."""
    robot: Entity = env.scene[asset_cfg.name]
    pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    return pos / _GRIPPER_DRIVER_MAX


def ee_to_object(
    env: ManagerBasedRlEnv,
    door_entity_name: str = "door",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Vector from EE (pinch_site) to door handle (object_site) in world frame (3D).

    object_site lives on the articulated handle body, so its world position is
    automatically updated by MuJoCo FK whenever the hinge angle changes.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    door: Entity = env.scene[door_entity_name]
    handle_ids = door.find_sites("object_site")[0]
    handle_pos_w = door.data.site_pos_w[:, handle_ids].squeeze(1)
    return handle_pos_w - ee_pos


def object_pos(
    env: ManagerBasedRlEnv,
    door_entity_name: str = "door",
) -> torch.Tensor:
    """Handle (object_site) position in local (robot-base) frame (3D)."""
    door: Entity = env.scene[door_entity_name]
    handle_ids = door.find_sites("object_site")[0]
    return door.data.site_pos_w[:, handle_ids].squeeze(1) - env.scene.env_origins


def _goal_handle_pos_w(
    env: ManagerBasedRlEnv,
    door_entity_name: str = "door",
    command_name: str = "door_goal",
) -> torch.Tensor:
    """Compute goal handle position in world frame via forward kinematics (N, 3).

    Uses the hinge geometry from door.xml:
      - Hinge pivot in door_base frame: _HINGE_OFFSET = (0, -0.3, 0)
      - Handle offset from hinge at θ=0: _HANDLE_FROM_HINGE = (-0.04, 0.55, 0)
    Rotation is around the z-axis (door_base has identity orientation after reset).
    """
    door: Entity = env.scene[door_entity_name]
    door_base_w = door.data.root_link_pos_w  # (N, 3)

    hx, hy, hz = _HINGE_OFFSET
    dx, dy, dz = _HANDLE_FROM_HINGE

    hinge_w = door_base_w + torch.tensor([hx, hy, hz], device=env.device)  # (N, 3)

    goal_angle = _get_door_goal_command(env, command_name).goal_angle  # (N, 1)
    cos_a = torch.cos(goal_angle)  # (N, 1)
    sin_a = torch.sin(goal_angle)  # (N, 1)

    # Rotate handle-from-hinge vector by goal_angle around z
    tx = dx * cos_a - dy * sin_a          # (N, 1)
    ty = dx * sin_a + dy * cos_a          # (N, 1)
    tz = torch.full_like(tx, dz)          # (N, 1)

    return hinge_w + torch.cat([tx, ty, tz], dim=-1)  # (N, 3)


def goal_pos(
    env: ManagerBasedRlEnv,
    door_entity_name: str = "door",
    command_name: str = "door_goal",
) -> torch.Tensor:
    """Goal handle position in local (robot-base) frame (3D)."""
    return _goal_handle_pos_w(env, door_entity_name, command_name) - env.scene.env_origins


def object_to_goal(
    env: ManagerBasedRlEnv,
    door_entity_name: str = "door",
    command_name: str = "door_goal",
) -> torch.Tensor:
    """Vector from current handle position to goal handle position in world frame (3D)."""
    door: Entity = env.scene[door_entity_name]
    handle_ids = door.find_sites("object_site")[0]
    handle_pos_w = door.data.site_pos_w[:, handle_ids].squeeze(1)
    target_w = _goal_handle_pos_w(env, door_entity_name, command_name)
    return target_w - handle_pos_w


# ---------------------------------------------------------------------------
# Reward functions
# Mirrors pick_cube_osc: position-based Gaussians for both reach and goal phases.
# ---------------------------------------------------------------------------


def ee_to_object_reward(
    env: ManagerBasedRlEnv,
    std: float,
    door_entity_name: str = "door",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Gaussian reward for EE proximity to door handle (reach phase)."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    door: Entity = env.scene[door_entity_name]
    handle_ids = door.find_sites("object_site")[0]
    handle_pos_w = door.data.site_pos_w[:, handle_ids].squeeze(1)
    dist_sq = torch.sum(torch.square(handle_pos_w - ee_pos), dim=-1)
    return torch.nan_to_num(torch.exp(-dist_sq / std**2), nan=0.0)


class object_at_goal_reward:
    """Gaussian reward for handle proximity to goal position (open phase).

    Goal position is the FK-computed world position of object_site at goal_angle.
    This is a 3-D position error, identical in structure to cube_at_goal_reward.

    Debug vis draws:
      - Green sphere at current handle position
      - EE coordinate frame at pinch_site
    (Goal sphere + ray are drawn by DoorGoalCommand._debug_vis_impl)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._door_entity_name: str = cfg.params.get("door_entity_name", "door")
        self._command_name: str = cfg.params.get("command_name", "door_goal")
        self._ee_site_name: str = "pinch_site"
        self._debug_vis_enabled: bool = True
        robot: Entity = env.scene["robot"]
        self._ee_site_ids = robot.find_sites(self._ee_site_name)[0]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float,
        door_entity_name: str = "door",
        command_name: str = "door_goal",
    ) -> torch.Tensor:
        door: Entity = env.scene[door_entity_name]
        handle_ids = door.find_sites("object_site")[0]
        handle_pos_w = door.data.site_pos_w[:, handle_ids].squeeze(1)
        target_w = _goal_handle_pos_w(env, door_entity_name, command_name)
        dist_sq = torch.sum(torch.square(target_w - handle_pos_w), dim=-1)
        return torch.nan_to_num(torch.exp(-dist_sq / std**2), nan=0.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        pass

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        if not self._debug_vis_enabled:
            return

        env = self._env
        env_indices = list(visualizer.get_env_indices(env.num_envs))
        if not env_indices:
            return

        door: Entity = env.scene[self._door_entity_name]
        robot: Entity = env.scene["robot"]
        handle_ids = door.find_sites("object_site")[0]

        for i in env_indices:
            # Current handle: green sphere
            handle_pos_np = door.data.site_pos_w[i, handle_ids].squeeze(0).cpu().numpy()
            visualizer.add_sphere(
                center=handle_pos_np, radius=0.022,
                color=(0.2, 1.0, 0.3, 0.7), label=f"object_pos_{i}",
            )

            # EE coordinate frame
            from mjlab.utils.lab_api.math import matrix_from_quat
            ee_pos = robot.data.site_pos_w[i, self._ee_site_ids].squeeze(0)
            ee_quat = robot.data.site_quat_w[i, self._ee_site_ids].squeeze(0)
            ee_rotm = matrix_from_quat(ee_quat.unsqueeze(0)).squeeze(0).cpu().numpy()
            visualizer.add_frame(
                position=ee_pos.cpu().numpy(),
                rotation_matrix=ee_rotm,
                scale=0.08, label=f"ee_frame_{i}",
            )


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def object_to_goal_error(
    env: ManagerBasedRlEnv,
    door_entity_name: str = "door",
    command_name: str = "door_goal",
) -> torch.Tensor:
    """Euclidean distance from current handle position to goal handle position."""
    door: Entity = env.scene[door_entity_name]
    handle_ids = door.find_sites("object_site")[0]
    handle_pos_w = door.data.site_pos_w[:, handle_ids].squeeze(1)
    target_w = _goal_handle_pos_w(env, door_entity_name, command_name)
    return torch.norm(target_w - handle_pos_w, dim=-1)


def ee_to_object_error(
    env: ManagerBasedRlEnv,
    door_entity_name: str = "door",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Euclidean distance from EE to door handle."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    door: Entity = env.scene[door_entity_name]
    handle_ids = door.find_sites("object_site")[0]
    handle_pos_w = door.data.site_pos_w[:, handle_ids].squeeze(1)
    return torch.norm(handle_pos_w - ee_pos, dim=-1)


# ---------------------------------------------------------------------------
# Termination functions
# ---------------------------------------------------------------------------


def door_out_of_range(
    env: ManagerBasedRlEnv,
    door_entity_name: str = "door",
) -> torch.Tensor:
    """Terminate if the door hinge angle goes out of its valid range [0, π/2]."""
    door: Entity = env.scene[door_entity_name]
    angle = door.data.joint_pos.squeeze(-1)
    return (angle < -0.1) | (angle > _math.pi / 2 + 0.1)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


def kinova_open_door_osc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Kinova Gen3 door-opening task with OSC (relative mode) + gripper control.

    Door starts closed. The policy must grasp the handle and open it to a
    randomly sampled target angle in [60°, 90°].

    Actions (7D):
        - osc_pose (6D): relative EE position + axis-angle delta
        - gripper (1D): -1=open, +1=closed
    """

    # --- Observations ---
    # Mirrors pick_cube_osc. Handle = "object"; goal = FK-computed target handle pos.
    # Total: 7 + 6 + 1 + 3 + 3 + 3 + 3 + 7 = 33D
    actor_terms = {
        "joint_vel":     ObservationTermCfg(func=joint_vel,  noise=Unoise(n_min=-1.5,  n_max=1.5)),
        "ee_pose":       ObservationTermCfg(func=ee_pose,    noise=Unoise(n_min=-0.01, n_max=0.01)),
        "gripper_state": ObservationTermCfg(
            func=gripper_state,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("right_driver_joint",))},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "ee_to_object": ObservationTermCfg(
            func=ee_to_object,
            params={
                "door_entity_name": "door",
                "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "object_pos": ObservationTermCfg(
            func=object_pos,
            params={"door_entity_name": "door"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "object_to_goal": ObservationTermCfg(
            func=object_to_goal,
            params={"door_entity_name": "door", "command_name": "door_goal"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "goal_pos": ObservationTermCfg(
            func=goal_pos,
            params={"door_entity_name": "door", "command_name": "door_goal"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    critic_terms = {**actor_terms}

    observations = {
        "actor":  ObservationGroupCfg(actor_terms,  enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    # --- Actions: 6D OSC + 1D gripper ---
    actions = {
        "osc_pose": OperationalSpaceActionCfg(
            entity_name="robot",
            actuator_names=("joint_.*",),
            frame_name="pinch_site",
            frame_type="site",
            use_relative_mode=True,
            delta_pos_scale=0.01,
            delta_ori_scale=0.02,
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
        "gripper": GripperCtrlActionCfg(
            entity_name="robot",
            actuator_name="fingers_actuator",
            ctrl_min=0.0,
            ctrl_max=255.0,
        ),
    }

    # --- Events ---
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        ),
        "reset_robot_joints": EventTermCfg(
            func=reset_joints_with_delta,
            mode="reset",
            params={"entity_name": "robot", "joint_delta_deg": 5.0},
        ),
        "reset_gripper_open": EventTermCfg(
            func=reset_gripper_open,
            mode="reset",
            params={
                "joint_asset_cfg": SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES),
                "ctrl_asset_cfg": SceneEntityCfg("robot", actuator_names=("fingers_actuator",)),
            },
        ),
        "reset_door": EventTermCfg(
            func=reset_door,
            mode="reset",
            params={
                "door_entity_name": "door",
                "x_range": (_DOOR_POS_LO[0], _DOOR_POS_HI[0]),
                "y_range": (_DOOR_POS_LO[1], _DOOR_POS_HI[1]),
                "z": _DOOR_POS_LO[2],
                "init_angle_range": (_DOOR_INIT_ANGLE_LO, _DOOR_INIT_ANGLE_HI),
            },
        ),
        # Fingertip friction randomization (Robotiq 2F-85 pads)
        "fingertip_friction_slide": EventTermCfg(
            mode="startup",
            func=mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=r"(left|right)_pad[12]"),
                "operation": "abs",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.3, 1.5),
            },
        ),
        "fingertip_friction_spin": EventTermCfg(
            mode="startup",
            func=mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=r"(left|right)_pad[12]"),
                "operation": "abs",
                "distribution": "log_uniform",
                "axes": [1],
                "ranges": (1e-4, 2e-2),
            },
        ),
        "fingertip_friction_roll": EventTermCfg(
            mode="startup",
            func=mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=r"(left|right)_pad[12]"),
                "operation": "abs",
                "distribution": "log_uniform",
                "axes": [2],
                "ranges": (1e-5, 5e-3),
            },
        ),
    }

    # --- Rewards ---
    # Mirrors pick_cube_osc: position-based Gaussians for reach and goal phases.
    rewards = {
        # Phase 1: EE reaches handle (Gaussian, std=0.15)
        "reach_object": RewardTermCfg(
            func=ee_to_object_reward,
            weight=1.0,
            params={
                "std": 0.15,
                "door_entity_name": "door",
                "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
            },
        ),
        # Phase 2: Handle reaches goal position (Gaussian, std=0.10 m)
        "move_to_goal": RewardTermCfg(
            func=object_at_goal_reward,
            weight=1.0,
            params={"std": 0.10},
        ),
        # Tight placement bonus (Gaussian, std=0.05 m ~5 cm)
        "goal_precise": RewardTermCfg(
            func=object_at_goal_reward,
            weight=2.0,
            params={"std": 0.05},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[1-7]",))},
        ),
        "joint_vel_hinge": RewardTermCfg(
            func=manipulation_mdp.joint_velocity_hinge_penalty,
            weight=-0.01,
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
        "time_out":            TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_detection":       TerminationTermCfg(func=mdp.nan_detection, time_out=False),
        "ee_ground_collision": TerminationTermCfg(
            func=manipulation_mdp.illegal_contact,
            params={"sensor_name": "ee_ground_collision"},
        ),
    }

    # --- Metrics ---
    metrics = {
        "object_to_goal_error": MetricsTermCfg(
            func=object_to_goal_error,
            params={"door_entity_name": "door", "command_name": "door_goal"},
        ),
        "ee_to_object_error": MetricsTermCfg(
            func=ee_to_object_error,
            params={
                "door_entity_name": "door",
                "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
            },
        ),
    }

    # --- Curriculum ---
    curriculum = {
        "action_rate_l2_weight": CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={
                "reward_name": "action_rate_l2",
                "stages": [
                    {"step": 0,    "weight": -0.01},
                    {"step": 2400, "weight": -0.04},
                    {"step": 4800, "weight": -0.07},
                    {"step": 7200, "weight": -0.10},
                ],
            },
        ),
        "joint_vel_hinge_weight": CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={
                "reward_name": "joint_vel_hinge",
                "stages": [
                    {"step": 0,    "weight": -0.01},
                    {"step": 2400, "weight": -0.04},
                    {"step": 4800, "weight": -0.07},
                    {"step": 7200, "weight": -0.10},
                ],
            },
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=4096,
            env_spacing=2.5,
            entities={
                "robot": get_kinova_robot_cfg_peginhole_osc(),
                "door":  get_door_cfg(),
            },
            sensors=(ee_ground_collision_cfg,),
        ),
        observations=observations,
        actions=actions,
        commands={
            "door_goal": DoorGoalCommandCfg(
                resampling_time_range=(1e9, 1e9),  # resample only on episode reset
                debug_vis=True,
            ),
        },
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
                timestep=0.002,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=50,  # 10 Hz policy rate (500/50)
        episode_length_s=10.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

    return cfg


def kinova_open_door_osc_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for open-door OSC task."""
    return kinova_ppo_runner_cfg(experiment_name="kinova_open_door_osc")
