"""Kinova Gen3 pick-and-lift cube task with Operational Space Control (OSC, relative mode).

Task: Grasp a cube from the ground and lift it to a randomly sampled aerial goal.
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
from kinova_tasks.assets.objects.free.cube.cube_constants import get_cube_cfg
from kinova_tasks.tasks.actions.osc import OperationalSpaceActionCfg
from kinova_tasks.tasks.base_rl_cfg import kinova_ppo_runner_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
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
# Curriculum schedule: linear ramp from start to end values between
# _RAMP_START_ITER and _RAMP_END_ITER (PPO iters).
# Mirrors the open_drawer baseline_dr schedule.
# ---------------------------------------------------------------------------
_RAMP_START_ITER = 500
_RAMP_END_ITER = 3000


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Home joint positions (INIT_STATE_PEGINHOLE): arm rotated 90° at joint_1
_HOME_JOINT_POS = (
    1.5707963268,   # joint_1  90°
    0.5235987756,   # joint_2  30°
    0.0,            # joint_3   0°
    1.5707963268,   # joint_4  90°
    0.0,            # joint_5   0°
    1.0471975512,   # joint_6  60°
    -1.5707963268,  # joint_7 -90°
)
_DEG_TO_RAD = _math.pi / 180.0

# FK position of pinch_site at INIT_STATE_PEGINHOLE (local frame)
_HOME_POS = (-0.024850, -0.482624, 0.174564)

# Spawn / goal bounding boxes (local frame, used in events + debug vis)
_OBJECT_SPAWN_LO = (-0.1, -0.6,  0.02)
_OBJECT_SPAWN_HI = ( 0.1, -0.4,  0.03)
# _OBJECT_SPAWN_LO = (-0.3, -0.7,  0.02)
# _OBJECT_SPAWN_HI = ( 0.3, -0.3,  0.03)

_GOAL_LO = (-0.10, -0.60,  0.10)
_GOAL_HI = ( 0.10, -0.40,  0.30)
# _GOAL_LO = (-0.30, -0.70,  0.05)
# _GOAL_HI = ( 0.30, -0.30,  0.40)

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


# ---------------------------------------------------------------------------
# Goal command term
# ---------------------------------------------------------------------------


class ObjectGoalCommand(CommandTerm):
    """Samples and maintains a 3-D aerial goal position for the pick-and-lift task."""

    cfg: ObjectGoalCommandCfg

    def __init__(self, cfg: ObjectGoalCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Goal position in world frame, shape (num_envs, 3)."""
        return self.goal_pos

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        lo = torch.tensor([self.cfg.x_range[0], self.cfg.y_range[0], self.cfg.z_range[0]], device=self.device)
        hi = torch.tensor([self.cfg.x_range[1], self.cfg.y_range[1], self.cfg.z_range[1]], device=self.device)
        n = len(env_ids)
        pos = sample_uniform(lo, hi, (n, 3), device=self.device)
        self.goal_pos[env_ids] = pos + self._env.scene.env_origins[env_ids]

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        import numpy as np

        env_indices = visualizer.get_env_indices(self._env.num_envs)
        if not env_indices:
            return

        _edges = [
            (0,1),(2,3),(4,5),(6,7),  # along x
            (0,2),(1,3),(4,6),(5,7),  # along y
            (0,4),(1,5),(2,6),(3,7),  # along z
        ]

        def _box_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
            return np.array([
                [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
                [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]],
            ], dtype=np.float32)

        cube_corners_local = _box_corners(np.array(_OBJECT_SPAWN_LO), np.array(_OBJECT_SPAWN_HI))
        goal_corners_local = _box_corners(np.array(_GOAL_LO), np.array(_GOAL_HI))

        for i in env_indices:
            origin = self._env.scene.env_origins[i].cpu().numpy()

            # Cube spawn bounding box (blue)
            cube_corners = cube_corners_local + origin
            for idx, (a, b) in enumerate(_edges):
                visualizer.add_cylinder(
                    start=cube_corners[a], end=cube_corners[b],
                    radius=0.004, color=(0.2, 0.5, 1.0, 0.5),
                    label=f"cube_spawn_box_edge_{i}_{idx}",
                )

            # Goal bounding box (orange)
            goal_corners = goal_corners_local + origin
            for idx, (a, b) in enumerate(_edges):
                visualizer.add_cylinder(
                    start=goal_corners[a], end=goal_corners[b],
                    radius=0.004, color=(1.0, 0.5, 0.0, 0.5),
                    label=f"goal_box_edge_{i}_{idx}",
                )

            # Goal marker sphere
            goal_pos_np = self.goal_pos[i].cpu().numpy()
            visualizer.add_sphere(
                center=goal_pos_np, radius=0.04,
                color=(1.0, 0.5, 0.0, 0.35), label=f"pick_goal_{i}",
            )
            visualizer.add_sphere(
                center=goal_pos_np, radius=0.008,
                color=(1.0, 0.6, 0.0, 0.9), label=f"pick_goal_dot_{i}",
            )


@dataclass(kw_only=True)
class ObjectGoalCommandCfg(CommandTermCfg):
    """Configuration for the pick goal command term."""

    x_range: tuple[float, float] = (_GOAL_LO[0], _GOAL_HI[0])
    y_range: tuple[float, float] = (_GOAL_LO[1], _GOAL_HI[1])
    z_range: tuple[float, float] = (_GOAL_LO[2], _GOAL_HI[2])

    def build(self, env: ManagerBasedRlEnv) -> ObjectGoalCommand:
        return ObjectGoalCommand(self, env)


def _get_object_goal_command(env: ManagerBasedRlEnv, command_name: str = "object_goal") -> ObjectGoalCommand:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, ObjectGoalCommand)
    return term


# ---------------------------------------------------------------------------
# Gripper action term
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class GripperCtrlActionCfg(ActionTermCfg):
    """1D gripper action: maps policy output to fingers_actuator ctrl.

    Action: tanh-squashed, then scaled.
      -1 → ctrl_min (fully open)
      +1 → ctrl_max (fully closed)
    """

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
# Reset events
# ---------------------------------------------------------------------------


def reset_joints_with_delta(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    entity_name: str = "robot",
    joint_delta_deg: float = 5.0,
) -> None:
    """Reset arm joints (1-7) to home pose ± uniform delta, clipped to soft limits."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    robot: Entity = env.scene[entity_name]
    soft_limits = robot.data.soft_joint_pos_limits  # (num_envs, num_joints, 2)

    lo = soft_limits[env_ids, :7, 0]
    hi = soft_limits[env_ids, :7, 1]

    n = len(env_ids)
    delta_rad = joint_delta_deg * _DEG_TO_RAD
    home = torch.tensor(_HOME_JOINT_POS, device=env.device).unsqueeze(0).expand(n, -1)
    delta = sample_uniform(
        torch.full((7,), -delta_rad, device=env.device),
        torch.full((7,), delta_rad, device=env.device),
        (n, 7),
        device=env.device,
    )
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
    """Reset gripper to fully open: zero all 8 gripper joints and set ctrl=0."""
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

    # Set fingers_actuator ctrl to 0 (open)
    num_actuators = len(ctrl_asset_cfg.actuator_ids) if isinstance(ctrl_asset_cfg.actuator_ids, list) else 1
    ctrl = torch.zeros(n, num_actuators, device=env.device)
    robot.data.write_ctrl(ctrl, ctrl_ids=ctrl_asset_cfg.actuator_ids, env_ids=env_ids)


def reset_object_position(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    object_entity_name: str = "cube",
    x_range: tuple[float, float] = (-0.08, 0.03),
    y_range: tuple[float, float] = (-0.55, -0.42),
    z_range: tuple[float, float] = (0.19, 0.21),
) -> None:
    """Spawn the cube at a random position within the workspace."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    cube: Entity = env.scene[object_entity_name]
    n = len(env_ids)

    lower = torch.tensor([x_range[0], y_range[0], z_range[0]], device=env.device)
    upper = torch.tensor([x_range[1], y_range[1], z_range[1]], device=env.device)
    pos = sample_uniform(lower, upper, (n, 3), device=env.device)
    pos = pos + env.scene.env_origins[env_ids]

    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0  # identity (flat on ground)

    pose = torch.cat([pos, quat], dim=-1)
    vel = torch.zeros(n, 6, device=env.device)
    cube.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    cube.write_root_link_velocity_to_sim(vel, env_ids=env_ids)



# ---------------------------------------------------------------------------
# Observation functions (absolute values, local frame — style of reach_osc)
# ---------------------------------------------------------------------------


def joint_pos(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Absolute arm joint positions (7D, rad)."""
    robot: Entity = env.scene[entity_name]
    return robot.data.joint_pos[:, :7]


def joint_vel(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Absolute arm joint velocities (7D, rad/s)."""
    robot: Entity = env.scene[entity_name]
    return robot.data.joint_vel[:, :7]


def ee_pose(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    site_name: str = "pinch_site",
) -> torch.Tensor:
    """Current EE pose [pos(3), axis_angle(3)] in local (robot-base) frame (6D)."""
    robot: Entity = env.scene[entity_name]
    site_ids = robot.find_sites(site_name)[0]
    pos = robot.data.site_pos_w[:, site_ids].squeeze(1) - env.scene.env_origins
    quat = robot.data.site_quat_w[:, site_ids].squeeze(1)
    return torch.cat([pos, axis_angle_from_quat(quat)], dim=-1)


def gripper_state(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=("right_driver_joint",)),
) -> torch.Tensor:
    """Gripper openness: right_driver_joint normalized to [0=open, 1=closed] (1D)."""
    robot: Entity = env.scene[asset_cfg.name]
    pos = robot.data.joint_pos[:, asset_cfg.joint_ids]  # (N, 1)
    return pos / _GRIPPER_DRIVER_MAX


def ee_to_object(
    env: ManagerBasedRlEnv,
    object_entity_name: str = "cube",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Vector from EE (pinch_site) to cube center in world frame (3D)."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)  # (N, 3)
    cube: Entity = env.scene[object_entity_name]
    return cube.data.root_link_pos_w - ee_pos


def object_pos(
    env: ManagerBasedRlEnv,
    object_entity_name: str = "cube",
) -> torch.Tensor:
    """Cube position in local (robot-base) frame (3D)."""
    cube: Entity = env.scene[object_entity_name]
    return cube.data.root_link_pos_w - env.scene.env_origins


def object_to_goal(
    env: ManagerBasedRlEnv,
    object_entity_name: str = "cube",
    command_name: str = "object_goal",
) -> torch.Tensor:
    """Vector from cube center to goal position in world frame (3D)."""
    cube: Entity = env.scene[object_entity_name]
    goal_w = _get_object_goal_command(env, command_name).goal_pos
    return goal_w - cube.data.root_link_pos_w


def goal_pos(env: ManagerBasedRlEnv, command_name: str = "object_goal") -> torch.Tensor:
    """Goal position in local (robot-base) frame (3D)."""
    goal_w = _get_object_goal_command(env, command_name).goal_pos
    return goal_w - env.scene.env_origins


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def ee_to_object_reward(
    env: ManagerBasedRlEnv,
    std: float,
    object_entity_name: str = "cube",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Gaussian reward for EE proximity to cube (reach/approach phase)."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    cube: Entity = env.scene[object_entity_name]
    dist_sq = torch.sum(torch.square(cube.data.root_link_pos_w - ee_pos), dim=-1)
    return torch.nan_to_num(torch.exp(-dist_sq / std**2), nan=0.0)


class object_at_goal_reward:
    """Gaussian reward for cube proximity to goal (lift phase).

    Debug vis draws:
      - Blue sphere at cube center
      - EE coordinate frame at pinch_site
    (Goal sphere + bounding boxes are drawn by ObjectGoalCommand._debug_vis_impl)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._object_entity_name: str = cfg.params.get("object_entity_name", "cube")
        self._command_name: str = cfg.params.get("command_name", "object_goal")
        self._ee_site_name: str = "pinch_site"
        self._debug_vis_enabled: bool = True
        robot: Entity = env.scene["robot"]
        self._ee_site_ids = robot.find_sites(self._ee_site_name)[0]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float,
        object_entity_name: str = "cube",
        command_name: str = "object_goal",
    ) -> torch.Tensor:
        cube: Entity = env.scene[object_entity_name]
        goal_w = _get_object_goal_command(env, command_name).goal_pos
        dist_sq = torch.sum(torch.square(cube.data.root_link_pos_w - goal_w), dim=-1)
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

        cube: Entity = env.scene[self._object_entity_name]
        robot: Entity = env.scene["robot"]

        for i in env_indices:
            # Cube: small blue sphere
            cube_pos_np = cube.data.root_link_pos_w[i].cpu().numpy()
            visualizer.add_sphere(
                center=cube_pos_np, radius=0.022,
                color=(0.2, 0.5, 1.0, 0.6), label=f"cube_pos_{i}",
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
    object_entity_name: str = "cube",
    command_name: str = "object_goal",
) -> torch.Tensor:
    """Euclidean distance from cube center to goal position."""
    cube: Entity = env.scene[object_entity_name]
    goal_w = _get_object_goal_command(env, command_name).goal_pos
    return torch.norm(cube.data.root_link_pos_w - goal_w, dim=-1)


def ee_to_object_error(
    env: ManagerBasedRlEnv,
    object_entity_name: str = "cube",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Euclidean distance from EE to cube center."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    cube: Entity = env.scene[object_entity_name]
    return torch.norm(cube.data.root_link_pos_w - ee_pos, dim=-1)


# ---------------------------------------------------------------------------
# Contact-force penalty
# ---------------------------------------------------------------------------


class ee_ground_force_penalty:
    """Penalize EE-to-ground contact force magnitude and draw debug arrows.

    The sensor must be configured with fields that include ``"force"`` and
    ``"pos"`` (and ``"normal"``, ``"tangent"`` for ``global_frame=True``).
    ``global_frame=True`` on the sensor puts the force vector in world
    coordinates so the arrow direction is correct in the viser overlay.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._sensor_name: str = cfg.params.get("sensor_name", "ee_ground_collision")
        self._debug_vis_enabled: bool = True  # toggled by viser checkbox

    def reset(self, env_ids: torch.Tensor) -> None:
        pass  # required for reward_manager to register this as a class term

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str = "ee_ground_collision",
        scale: float = 10.0,
    ) -> torch.Tensor:
        """Exponential penalty for EE-ground contact force.

        Returns exp(‖f‖ / scale) - 1, which is ~0 for forces well below
        ``scale`` (N) and grows exponentially above it.  Use with a negative
        weight in rewards.
        """
        sensor: ContactSensor = env.scene[sensor_name]
        force = sensor.data.force  # [B, N, 3]
        if force is None:
            return torch.zeros(env.num_envs, device=env.device)
        f_mag = torch.norm(force, dim=-1)           # [B, N]
        f_mag = f_mag.max(dim=-1).values            # [B]  — worst slot
        return torch.exp(f_mag / scale) - 1.0       # [B]

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        """Draw a force arrow at each active EE-ground contact point."""
        if not self._debug_vis_enabled:
            return
        env = self._env
        sensor: ContactSensor = env.scene[self._sensor_name]
        data = sensor.data

        if data.force is None or data.pos is None or data.found is None:
            return

        # found: [B, N], force: [B, N, 3], pos: [B, N, 3]  (global frame)
        found = data.found  # [B, N]  — already 2D, no squeeze
        force = data.force  # [B, N, 3]
        pos   = data.pos    # [B, N, 3]

        # Ensure 2D regardless of whether found is [B] or [B, N]
        if found.dim() == 1:
            found = found.unsqueeze(-1)
            force = force.unsqueeze(1)
            pos   = pos.unsqueeze(1)

        scale = 0.002  # metres per Newton — tune to taste

        for i in visualizer.get_env_indices(env.num_envs):
            for slot in range(found.shape[1]):
                if found[i, slot].item() <= 0:
                    continue
                f_vec = force[i, slot]          # [3]
                f_mag = torch.norm(f_vec).item()
                if f_mag < 0.1:                 # skip near-zero noise
                    continue
                start = pos[i, slot]            # [3]
                end   = start + f_vec * scale
                # colour: green → red by magnitude (capped at 100 N)
                t = min(f_mag / 100.0, 1.0)
                color = (t, 1.0 - t, 0.0, 0.9)
                visualizer.add_arrow(
                    start=start,
                    end=end,
                    color=color,
                    width=0.008,
                    label=f"EE-gnd {f_mag:.1f}N",
                )


# ---------------------------------------------------------------------------
# Termination functions
# ---------------------------------------------------------------------------


def object_out_of_bounds(
    env: ManagerBasedRlEnv,
    object_entity_name: str = "cube",
    home_pos: tuple[float, float, float] = _HOME_POS,
    workspace_half: tuple[float, float, float] = (0.35, 0.25, 0.40),
) -> torch.Tensor:
    """Terminate if the cube leaves the workspace box centered on the home EE pose."""
    cube: Entity = env.scene[object_entity_name]
    pos_local = cube.data.root_link_pos_w - env.scene.env_origins  # (N, 3)
    home = torch.tensor(home_pos, device=env.device)
    ws_half = torch.tensor(workspace_half, device=env.device)
    in_workspace = (
        (pos_local >= home - ws_half) & (pos_local <= home + ws_half)
    ).all(dim=-1)
    return ~in_workspace


# ---------------------------------------------------------------------------
# Curriculum functions
# ---------------------------------------------------------------------------


def _ramp_by_iter(
    env: ManagerBasedRlEnv,
    start_val: float,
    end_val: float,
    num_steps_per_env: int = 24,
) -> float:
    """Linear ramp on PPO iter (= common_step_counter // num_steps_per_env).

    Held at ``start_val`` below ``_RAMP_START_ITER`` and at ``end_val`` above
    ``_RAMP_END_ITER``.
    """
    train_iter = env.common_step_counter // num_steps_per_env
    if train_iter <= _RAMP_START_ITER:
        return start_val
    if train_iter >= _RAMP_END_ITER:
        return end_val
    t = (train_iter - _RAMP_START_ITER) / (_RAMP_END_ITER - _RAMP_START_ITER)
    return start_val + t * (end_val - start_val)


def object_spawn_extent_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    start_x: float = 0.10, end_x: float = 0.30,
    start_y: float = 0.10, end_y: float = 0.20,
    center_x: float = 0.0, center_y: float = -0.5,
    z_lo: float = 0.02, z_hi: float = 0.03,
    num_steps_per_env: int = 24,
) -> dict[str, torch.Tensor]:
    """Widen cube spawn x/y extents over training (z fixed: cube on ground)."""
    del env_ids
    hx = _ramp_by_iter(env, start_x, end_x, num_steps_per_env)
    hy = _ramp_by_iter(env, start_y, end_y, num_steps_per_env)
    term = env.event_manager.get_term_cfg("reset_object_position")
    term.params["x_range"] = (center_x - hx, center_x + hx)
    term.params["y_range"] = (center_y - hy, center_y + hy)
    term.params["z_range"] = (z_lo, z_hi)
    return {
        "object_spawn_x_extent": torch.tensor(hx),
        "object_spawn_y_extent": torch.tensor(hy),
    }


def goal_extent_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    start_x: float = 0.10, end_x: float = 0.30,
    start_y: float = 0.10, end_y: float = 0.20,
    start_z_lo: float = 0.10, end_z_lo: float = 0.025,
    start_z_hi: float = 0.30, end_z_hi: float = 0.40,
    center_x: float = 0.0, center_y: float = -0.5,
    num_steps_per_env: int = 24,
) -> dict[str, torch.Tensor]:
    """Widen goal x/y extents and z range over training.

    Goal z minimum ramps down to cube resting height (0.025) so the policy
    sees both "lift high" and "place at ground" goals at full curriculum.
    """
    del env_ids
    hx = _ramp_by_iter(env, start_x, end_x, num_steps_per_env)
    hy = _ramp_by_iter(env, start_y, end_y, num_steps_per_env)
    z_lo = _ramp_by_iter(env, start_z_lo, end_z_lo, num_steps_per_env)
    z_hi = _ramp_by_iter(env, start_z_hi, end_z_hi, num_steps_per_env)
    term = env.command_manager.get_term("object_goal")
    term.cfg.x_range = (center_x - hx, center_x + hx)
    term.cfg.y_range = (center_y - hy, center_y + hy)
    term.cfg.z_range = (z_lo, z_hi)
    return {
        "goal_x_extent": torch.tensor(hx),
        "goal_y_extent": torch.tensor(hy),
        "goal_z_lo": torch.tensor(z_lo),
        "goal_z_hi": torch.tensor(z_hi),
    }


def joint_delta_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    start_deg: float = 5.0,
    end_deg: float = 20.0,
    num_steps_per_env: int = 24,
) -> dict[str, torch.Tensor]:
    """Widen per-joint reset half-extent (degrees) over training."""
    del env_ids
    d = _ramp_by_iter(env, start_deg, end_deg, num_steps_per_env)
    term = env.event_manager.get_term_cfg("reset_robot_joints")
    term.params["joint_delta_deg"] = float(d)
    return {"joint_delta_deg": torch.tensor(d)}


def base_pose_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    start_xy_m: float = 0.0,
    end_xy_m: float = 0.05,
    start_yaw_deg: float = 0.0,
    end_yaw_deg: float = 10.0,
    num_steps_per_env: int = 24,
) -> dict[str, torch.Tensor]:
    """Widen robot base XY (m) and yaw (deg) reset half-extents over training."""
    del env_ids
    xy = _ramp_by_iter(env, start_xy_m, end_xy_m, num_steps_per_env)
    yaw_deg = _ramp_by_iter(env, start_yaw_deg, end_yaw_deg, num_steps_per_env)
    yaw_rad = yaw_deg * _DEG_TO_RAD
    term = env.event_manager.get_term_cfg("reset_base")
    term.params["pose_range"] = {
        "x":   (-xy, xy),
        "y":   (-xy, xy),
        "yaw": (-yaw_rad, yaw_rad),
    }
    return {
        "base_xy_extent_m": torch.tensor(xy),
        "base_yaw_extent_deg": torch.tensor(yaw_deg),
    }


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


def kinova_pick_cube_osc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Kinova Gen3 pick-and-lift cube with OSC (relative mode) + gripper control.

    Cube starts on the ground within the arm's workspace. The policy must grasp
    the cube and lift it to a randomly sampled aerial goal position.

    Actions (7D):
        - osc_pose (6D): relative EE position + axis-angle delta
        - gripper (1D): -1=open, +1=closed
    """

    # --- Observations ---
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
                "object_entity_name": "cube",
                "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "object_pos": ObservationTermCfg(
            func=object_pos,
            params={"object_entity_name": "cube"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "object_to_goal": ObservationTermCfg(
            func=object_to_goal,
            params={"object_entity_name": "cube"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "goal_pos": ObservationTermCfg(
            func=goal_pos,
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
            actuator_names=("joint_.*",),  # arm joints only
            frame_name="pinch_site",
            frame_type="site",
            use_relative_mode=True,
            delta_pos_scale=0.01,     # 2 cm per unit action
            delta_ori_scale=0.02,     # ~1.1° per unit action
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
    # Reset events use the *start-of-curriculum* values; the curriculum
    # functions overwrite these terms' params each step to widen the ranges
    # between iter 500 and 3000.
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
        "reset_object_position": EventTermCfg(
            func=reset_object_position,
            mode="reset",
            params={
                "object_entity_name": "cube",
                "x_range": (-0.10, 0.10),
                "y_range": (-0.60, -0.40),
                "z_range": (_OBJECT_SPAWN_LO[2], _OBJECT_SPAWN_HI[2]),
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
        # Cube contact-surface friction (matte plastic vs glossy painted vs metal)
        "dr_object_friction_slide": EventTermCfg(
            mode="startup",
            func=mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("cube", geom_names=("cube_geom",)),
                "operation": "abs",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.3, 1.5),
            },
        ),
        # Cube mass (~50 g nominal): scale ∈ [0.5, 2.0] log-uniform via pseudo_inertia
        "dr_object_mass": EventTermCfg(
            mode="startup",
            func=mdp.dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("cube", body_names=("cube",)),
                "alpha_range": (-0.5 * _math.log(2.0), 0.5 * _math.log(2.0)),
                "distribution": "uniform",
            },
        ),
    }

    # --- Rewards ---
    rewards = {
        # Phase 1: EE reaches cube (Gaussian, std=0.15)
        "reach_object": RewardTermCfg(
            func=ee_to_object_reward,
            weight=1.0,
            params={
                "std": 0.15,
                "object_entity_name": "cube",
                "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
            },
        ),
        # Phase 2: Cube reaches aerial goal (Gaussian, std=0.1)
        "move_to_goal": RewardTermCfg(
            func=object_at_goal_reward,
            weight=1.0,
            params={"std": 0.10},
        ),
        # Tight placement bonus (Gaussian, std=0.05)
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
        # EE-ground force penalty disabled while we focus on widening the
        # operating envelope. Re-enable (and its curriculum below) if EE
        # slamming into the table becomes a problem on the trained policy.
        # "ee_ground_force": RewardTermCfg(
        #     func=ee_ground_force_penalty,
        #     weight=0.0,
        #     params={"sensor_name": "ee_ground_collision", "scale": 10.0},
        # ),
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
        fields=("found", "force", "pos", "normal", "tangent"),
        reduce="maxforce",      # pick strongest contact per slot
        num_slots=1,
        global_frame=True,      # rotate force into world frame for debug arrows
    )

    terminations = {
        "time_out":         TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_detection":    TerminationTermCfg(func=mdp.nan_detection, time_out=False),
        # "ee_ground_collision": TerminationTermCfg(
        #     func=manipulation_mdp.illegal_contact,
        #     params={"sensor_name": "ee_ground_collision"},
        # ),
        "object_out_of_bounds": TerminationTermCfg(
            func=object_out_of_bounds,
            params={
                "object_entity_name": "cube",
                "home_pos": _HOME_POS,
                # workspace_half left at function default (0.35, 0.25, 0.40)
                # so it envelopes the curriculum-end cube spawn ranges.
                # See pick_cube_osc.py git history (commit 845197d) where
                # the default was widened.
            },
        ),
    }

    # --- Metrics ---
    metrics = {
        "object_to_goal_error": MetricsTermCfg(
            func=object_to_goal_error,
            params={"object_entity_name": "cube"},
        ),
        "ee_to_object_error": MetricsTermCfg(
            func=ee_to_object_error,
            params={
                "object_entity_name": "cube",
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
        # EE-ground force curriculum disabled with the reward term itself.
        # "ee_ground_force_weight": CurriculumTermCfg(
        #     func=mdp.reward_curriculum,
        #     params={
        #         "reward_name": "ee_ground_force",
        #         "stages": [
        #             {"step":    0, "weight":  0.000},
        #             {"step": 2400, "weight": -0.010},
        #             {"step": 3600, "weight": -0.025},
        #             {"step": 4800, "weight": -0.050},
        #             {"step": 6000, "weight": -0.075},
        #             {"step": 7200, "weight": -0.100},
        #         ],
        #     },
        # ),
        # Init-distribution curricula: linearly widen between iter 500 and 3000.
        "object_spawn_extent": CurriculumTermCfg(
            func=object_spawn_extent_curriculum,
            params={
                "start_x": 0.10, "end_x": 0.30,
                "start_y": 0.10, "end_y": 0.20,
                "center_x": 0.0, "center_y": -0.5,
                "z_lo": _OBJECT_SPAWN_LO[2], "z_hi": _OBJECT_SPAWN_HI[2],
            },
        ),
        "goal_extent": CurriculumTermCfg(
            func=goal_extent_curriculum,
            params={
                "start_x": 0.10, "end_x": 0.30,
                "start_y": 0.10, "end_y": 0.20,
                "start_z_lo": 0.10, "end_z_lo": 0.025,
                "start_z_hi": 0.30, "end_z_hi": 0.40,
                "center_x": 0.0, "center_y": -0.5,
            },
        ),
        "joint_delta": CurriculumTermCfg(
            func=joint_delta_curriculum,
            params={"start_deg": 5.0, "end_deg": 20.0},
        ),
        "base_pose": CurriculumTermCfg(
            func=base_pose_curriculum,
            params={
                "start_xy_m": 0.0, "end_xy_m": 0.05,
                "start_yaw_deg": 0.0, "end_yaw_deg": 10.0,
            },
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=4096,
            env_spacing=1.0,
            entities={
                "robot": get_kinova_robot_cfg_peginhole_osc(),
                "cube":  get_cube_cfg(),
            },
            sensors=(ee_ground_collision_cfg,),
        ),
        observations=observations,
        actions=actions,
        commands={
            "object_goal": ObjectGoalCommandCfg(
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
                timestep=0.002,  # 500 Hz physics / OSC rate
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
        # Drop the curriculum and pin reset/goal ranges to the curriculum
        # end-state (full-width distribution) so play mode shows the policy
        # under the widest training conditions, not the narrow start.
        cfg.curriculum = {}
        cfg.events["reset_robot_joints"].params["joint_delta_deg"] = 20.0
        cfg.events["reset_base"].params["pose_range"] = {
            "x":   (-0.05, 0.05),
            "y":   (-0.05, 0.05),
            "yaw": (-10.0 * _DEG_TO_RAD, 10.0 * _DEG_TO_RAD),
        }
        cfg.events["reset_object_position"].params["x_range"] = (-0.30, 0.30)
        cfg.events["reset_object_position"].params["y_range"] = (-0.70, -0.30)
        cfg.events["reset_object_position"].params["z_range"] = (
            _OBJECT_SPAWN_LO[2], _OBJECT_SPAWN_HI[2],
        )
        cfg.commands["object_goal"].x_range = (-0.30, 0.30)
        cfg.commands["object_goal"].y_range = (-0.70, -0.30)
        cfg.commands["object_goal"].z_range = (0.025, 0.40)

    return cfg


def kinova_pick_cube_osc_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for pick-cube OSC task."""
    return kinova_ppo_runner_cfg(experiment_name="kinova_pick_cube_osc")
