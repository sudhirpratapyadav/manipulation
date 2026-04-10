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
_CUBE_SPAWN_LO = (-0.08, -0.55,  0.02)
_CUBE_SPAWN_HI = ( 0.03, -0.42,  0.03)

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


class PickGoalCommand(CommandTerm):
    """Samples and maintains a 3-D aerial goal position for the pick-and-lift task."""

    cfg: PickGoalCommandCfg

    def __init__(self, cfg: PickGoalCommandCfg, env: ManagerBasedRlEnv):
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

        cube_corners_local = _box_corners(np.array(_CUBE_SPAWN_LO), np.array(_CUBE_SPAWN_HI))
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
class PickGoalCommandCfg(CommandTermCfg):
    """Configuration for the pick goal command term."""

    x_range: tuple[float, float] = (_GOAL_LO[0], _GOAL_HI[0])
    y_range: tuple[float, float] = (_GOAL_LO[1], _GOAL_HI[1])
    z_range: tuple[float, float] = (_GOAL_LO[2], _GOAL_HI[2])

    def build(self, env: ManagerBasedRlEnv) -> PickGoalCommand:
        return PickGoalCommand(self, env)


def _get_pick_goal_command(env: ManagerBasedRlEnv, command_name: str = "pick_goal") -> PickGoalCommand:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, PickGoalCommand)
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


def reset_cube_position(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    cube_entity_name: str = "cube",
    x_range: tuple[float, float] = (-0.08, 0.03),
    y_range: tuple[float, float] = (-0.55, -0.42),
    z_range: tuple[float, float] = (0.19, 0.21),
) -> None:
    """Spawn the cube at a random position within the workspace."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    cube: Entity = env.scene[cube_entity_name]
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


def ee_to_cube(
    env: ManagerBasedRlEnv,
    cube_entity_name: str = "cube",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Vector from EE (pinch_site) to cube center in world frame (3D)."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)  # (N, 3)
    cube: Entity = env.scene[cube_entity_name]
    return cube.data.root_link_pos_w - ee_pos


def cube_pos(
    env: ManagerBasedRlEnv,
    cube_entity_name: str = "cube",
) -> torch.Tensor:
    """Cube position in local (robot-base) frame (3D)."""
    cube: Entity = env.scene[cube_entity_name]
    return cube.data.root_link_pos_w - env.scene.env_origins


def cube_to_goal(
    env: ManagerBasedRlEnv,
    cube_entity_name: str = "cube",
    command_name: str = "pick_goal",
) -> torch.Tensor:
    """Vector from cube center to goal position in world frame (3D)."""
    cube: Entity = env.scene[cube_entity_name]
    goal_w = _get_pick_goal_command(env, command_name).goal_pos
    return goal_w - cube.data.root_link_pos_w


def goal_pos(env: ManagerBasedRlEnv, command_name: str = "pick_goal") -> torch.Tensor:
    """Goal position in local (robot-base) frame (3D)."""
    goal_w = _get_pick_goal_command(env, command_name).goal_pos
    return goal_w - env.scene.env_origins


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def ee_to_cube_reward(
    env: ManagerBasedRlEnv,
    std: float,
    cube_entity_name: str = "cube",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Gaussian reward for EE proximity to cube (reach/approach phase)."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    cube: Entity = env.scene[cube_entity_name]
    dist_sq = torch.sum(torch.square(cube.data.root_link_pos_w - ee_pos), dim=-1)
    return torch.nan_to_num(torch.exp(-dist_sq / std**2), nan=0.0)


class cube_at_goal_reward:
    """Gaussian reward for cube proximity to goal (lift phase).

    Debug vis draws:
      - Blue sphere at cube center
      - EE coordinate frame at pinch_site
    (Goal sphere + bounding boxes are drawn by PickGoalCommand._debug_vis_impl)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._cube_entity_name: str = cfg.params.get("cube_entity_name", "cube")
        self._command_name: str = cfg.params.get("command_name", "pick_goal")
        self._ee_site_name: str = "pinch_site"
        self._debug_vis_enabled: bool = True
        robot: Entity = env.scene["robot"]
        self._ee_site_ids = robot.find_sites(self._ee_site_name)[0]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float,
        cube_entity_name: str = "cube",
        command_name: str = "pick_goal",
    ) -> torch.Tensor:
        cube: Entity = env.scene[cube_entity_name]
        goal_w = _get_pick_goal_command(env, command_name).goal_pos
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

        cube: Entity = env.scene[self._cube_entity_name]
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


def cube_to_goal_error(
    env: ManagerBasedRlEnv,
    cube_entity_name: str = "cube",
    command_name: str = "pick_goal",
) -> torch.Tensor:
    """Euclidean distance from cube center to goal position."""
    cube: Entity = env.scene[cube_entity_name]
    goal_w = _get_pick_goal_command(env, command_name).goal_pos
    return torch.norm(cube.data.root_link_pos_w - goal_w, dim=-1)


def ee_to_cube_error(
    env: ManagerBasedRlEnv,
    cube_entity_name: str = "cube",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Euclidean distance from EE to cube center."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    cube: Entity = env.scene[cube_entity_name]
    return torch.norm(cube.data.root_link_pos_w - ee_pos, dim=-1)


# ---------------------------------------------------------------------------
# Termination functions
# ---------------------------------------------------------------------------


def cube_out_of_bounds(
    env: ManagerBasedRlEnv,
    cube_entity_name: str = "cube",
    home_pos: tuple[float, float, float] = _HOME_POS,
    workspace_half: tuple[float, float, float] = (0.20, 0.20, 0.30),
) -> torch.Tensor:
    """Terminate if the cube leaves the workspace box centered on the home EE pose."""
    cube: Entity = env.scene[cube_entity_name]
    pos_local = cube.data.root_link_pos_w - env.scene.env_origins  # (N, 3)
    home = torch.tensor(home_pos, device=env.device)
    ws_half = torch.tensor(workspace_half, device=env.device)
    in_workspace = (
        (pos_local >= home - ws_half) & (pos_local <= home + ws_half)
    ).all(dim=-1)
    return ~in_workspace


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
    # Total: 7 + 7 + 6 + 1 + 3 + 3 + 3 + 3 + 7 = 40D
    actor_terms = {
        "joint_pos":     ObservationTermCfg(func=joint_pos,  noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel":     ObservationTermCfg(func=joint_vel,  noise=Unoise(n_min=-1.5,  n_max=1.5)),
        "ee_pose":       ObservationTermCfg(func=ee_pose,    noise=Unoise(n_min=-0.01, n_max=0.01)),
        "gripper_state": ObservationTermCfg(
            func=gripper_state,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("right_driver_joint",))},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "ee_to_cube": ObservationTermCfg(
            func=ee_to_cube,
            params={
                "cube_entity_name": "cube",
                "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "cube_pos": ObservationTermCfg(
            func=cube_pos,
            params={"cube_entity_name": "cube"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "cube_to_goal": ObservationTermCfg(
            func=cube_to_goal,
            params={"cube_entity_name": "cube"},
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
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        ),
        "reset_robot_joints": EventTermCfg(
            func=reset_joints_with_delta,
            mode="reset",
            params={"entity_name": "robot", "joint_delta_deg": 2.0},
        ),
        "reset_gripper_open": EventTermCfg(
            func=reset_gripper_open,
            mode="reset",
            params={
                "joint_asset_cfg": SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES),
                "ctrl_asset_cfg": SceneEntityCfg("robot", actuator_names=("fingers_actuator",)),
            },
        ),
        "reset_cube_position": EventTermCfg(
            func=reset_cube_position,
            mode="reset",
            params={
                "cube_entity_name": "cube",
                "x_range": (_CUBE_SPAWN_LO[0], _CUBE_SPAWN_HI[0]),
                "y_range": (_CUBE_SPAWN_LO[1], _CUBE_SPAWN_HI[1]),
                "z_range": (_CUBE_SPAWN_LO[2], _CUBE_SPAWN_HI[2]),
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
    rewards = {
        # Phase 1: EE reaches cube (Gaussian, std=0.15)
        "reach_cube": RewardTermCfg(
            func=ee_to_cube_reward,
            weight=1.0,
            params={
                "std": 0.15,
                "cube_entity_name": "cube",
                "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
            },
        ),
        # Phase 2: Cube reaches aerial goal (Gaussian, std=0.1)
        "lift_to_goal": RewardTermCfg(
            func=cube_at_goal_reward,
            weight=1.0,
            params={"std": 0.10},
        ),
        # Tight placement bonus (Gaussian, std=0.05)
        "goal_precise": RewardTermCfg(
            func=cube_at_goal_reward,
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
        "time_out":         TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_detection":    TerminationTermCfg(func=mdp.nan_detection, time_out=False),
        "ee_ground_collision": TerminationTermCfg(
            func=manipulation_mdp.illegal_contact,
            params={"sensor_name": "ee_ground_collision"},
        ),
        "cube_out_of_bounds": TerminationTermCfg(
            func=cube_out_of_bounds,
            params={
                "cube_entity_name": "cube",
                "home_pos": _HOME_POS,
                "workspace_half": (0.20, 0.20, 0.30),
            },
        ),
    }

    # --- Metrics ---
    metrics = {
        "cube_to_goal_error": MetricsTermCfg(
            func=cube_to_goal_error,
            params={"cube_entity_name": "cube"},
        ),
        "ee_to_cube_error": MetricsTermCfg(
            func=ee_to_cube_error,
            params={
                "cube_entity_name": "cube",
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
            "pick_goal": PickGoalCommandCfg(
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
        cfg.curriculum = {}

    return cfg


def kinova_pick_cube_osc_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for pick-cube OSC task."""
    return kinova_ppo_runner_cfg(experiment_name="kinova_pick_cube_osc")
