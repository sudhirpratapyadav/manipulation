"""Kinova Gen3 drawer-opening task with Operational Space Control (OSC, relative mode).

Task: Grasp the drawer handle and pull it open to a randomly sampled target slide distance.
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
from kinova_tasks.assets.objects.articulated.drawer.drawer_constants import get_drawer_cfg
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

# Home joint positions for open-drawer task: joint_1 = 0° (arm facing forward)
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

# Drawer spawn range (local frame) — drawer_base mocap body position
_DRAWER_POS_LO = (0.7, -0.05, 0.45)
_DRAWER_POS_HI = (0.9,  0.15, 0.65)

# Drawer initial slide range at reset (meters): 0 = closed, -0.25 = fully open
# Start near closed with a small random offset
_DRAWER_INIT_SLIDE_LO = -0.02
_DRAWER_INIT_SLIDE_HI =  0.0

# Drawer goal slide range (meters): target is a pulled-open position.
# drawer_slide joint range: [-0.25, 0]  (negative = pulled out / open)
_DRAWER_GOAL_LO = -0.25  # fully open
_DRAWER_GOAL_HI = -0.15  # partially open

# Drawer handle geometry (from drawer.xml):
#   object_site is at pos="-0.04 0 0" in the handle body frame.
#   handle body slides along x-axis (axis="1 0 0") within drawer_base.
#   At slide joint value q, object_site world pos =
#       drawer_base_pos_w + (q + _HANDLE_OFFSET_X, 0, 0)
_HANDLE_OFFSET_X = -0.044
_HANDLE_OFFSET_Y =  0.0
_HANDLE_OFFSET_Z =  0.0


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
# Drawer goal command term
# ---------------------------------------------------------------------------


class DrawerGoalCommand(CommandTerm):
    """Samples and maintains a target drawer slide position for the open-drawer task."""

    cfg: DrawerGoalCommandCfg

    def __init__(self, cfg: DrawerGoalCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.goal_slide = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Target drawer slide position (m), shape (num_envs, 1).

        Negative values mean pulled out (open). Range: [_DRAWER_GOAL_LO, _DRAWER_GOAL_HI].
        """
        return self.goal_slide

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        lo = torch.full((len(env_ids), 1), self.cfg.slide_lo, device=self.device)
        hi = torch.full((len(env_ids), 1), self.cfg.slide_hi, device=self.device)
        self.goal_slide[env_ids] = lo + (hi - lo) * torch.rand_like(lo)

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        import numpy as np

        env_indices = visualizer.get_env_indices(self._env.num_envs)
        if not env_indices:
            return

        drawer: Entity = self._env.scene["drawer"]

        # Spawn range bounding box edges (blue)
        spawn_lo = np.array([_DRAWER_POS_LO[0], _DRAWER_POS_LO[1], 0.02], dtype=np.float32)
        spawn_hi = np.array([_DRAWER_POS_HI[0], _DRAWER_POS_HI[1], 0.02], dtype=np.float32)
        spawn_corners = np.array([
            [spawn_lo[0], spawn_lo[1], spawn_lo[2]],
            [spawn_hi[0], spawn_lo[1], spawn_lo[2]],
            [spawn_lo[0], spawn_hi[1], spawn_lo[2]],
            [spawn_hi[0], spawn_hi[1], spawn_lo[2]],
        ], dtype=np.float32)
        spawn_edges = [(0, 1), (0, 2), (1, 3), (2, 3)]

        for i in env_indices:
            origin = self._env.scene.env_origins[i].cpu().numpy()

            # Spawn bounding box (blue)
            for idx, (a, b) in enumerate(spawn_edges):
                visualizer.add_cylinder(
                    start=spawn_corners[a] + origin,
                    end=spawn_corners[b] + origin,
                    radius=0.004, color=(0.2, 0.5, 1.0, 0.5),
                    label=f"drawer_spawn_edge_{i}_{idx}",
                )

            # Current handle position (blue sphere)
            handle_ids = drawer.find_sites("object_site")[0]
            handle_pos_np = drawer.data.site_pos_w[i, handle_ids].squeeze(0).cpu().numpy()
            visualizer.add_sphere(
                center=handle_pos_np, radius=0.022,
                color=(0.2, 0.5, 1.0, 0.6), label=f"handle_current_{i}",
            )

            # Goal handle position (orange sphere): linear FK along x
            drawer_pos = drawer.data.root_link_pos_w[i].cpu().numpy()
            q_goal = float(self.goal_slide[i, 0].item())
            target_handle_w = drawer_pos + np.array(
                [q_goal + _HANDLE_OFFSET_X, _HANDLE_OFFSET_Y, _HANDLE_OFFSET_Z],
                dtype=np.float32,
            )
            visualizer.add_sphere(
                center=target_handle_w, radius=0.03,
                color=(1.0, 0.5, 0.0, 0.4), label=f"drawer_goal_{i}",
            )
            visualizer.add_sphere(
                center=target_handle_w, radius=0.008,
                color=(1.0, 0.6, 0.0, 0.9), label=f"drawer_goal_dot_{i}",
            )


@dataclass(kw_only=True)
class DrawerGoalCommandCfg(CommandTermCfg):
    """Configuration for the drawer goal command term."""

    slide_lo: float = _DRAWER_GOAL_LO
    slide_hi: float = _DRAWER_GOAL_HI

    def build(self, env: ManagerBasedRlEnv) -> DrawerGoalCommand:
        return DrawerGoalCommand(self, env)


def _get_drawer_goal_command(env: ManagerBasedRlEnv, command_name: str = "drawer_goal") -> DrawerGoalCommand:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, DrawerGoalCommand)
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


def reset_drawer(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    drawer_entity_name: str = "drawer",
    x_range: tuple[float, float] = (_DRAWER_POS_LO[0], _DRAWER_POS_HI[0]),
    y_range: tuple[float, float] = (_DRAWER_POS_LO[1], _DRAWER_POS_HI[1]),
    z: float = _DRAWER_POS_LO[2],
    init_slide_range: tuple[float, float] = (_DRAWER_INIT_SLIDE_LO, _DRAWER_INIT_SLIDE_HI),
) -> None:
    """Reset drawer base position and slide joint to a randomly sampled initial position."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    drawer: Entity = env.scene[drawer_entity_name]
    n = len(env_ids)

    # Sample drawer base position (mocap body)
    lower = torch.tensor([x_range[0], y_range[0], z], device=env.device)
    upper = torch.tensor([x_range[1], y_range[1], z], device=env.device)
    pos = sample_uniform(lower, upper, (n, 3), device=env.device)
    pos = pos + env.scene.env_origins[env_ids]

    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0  # identity orientation
    pose = torch.cat([pos, quat], dim=-1)  # (N, 7)
    drawer.write_mocap_pose_to_sim(pose, env_ids=env_ids)

    # Sample slide initial position uniformly from init_slide_range
    jpos = sample_uniform(
        torch.tensor([init_slide_range[0]], device=env.device),
        torch.tensor([init_slide_range[1]], device=env.device),
        (n, 1),
        device=env.device,
    )
    jvel = torch.zeros(n, 1, device=env.device)
    drawer.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Observation functions
# Mirrors open_door_osc: joint_vel, ee_pose, gripper_state,
#   ee_to_object, object_pos, object_to_goal, goal_pos, actions
# The drawer handle (object_site) is treated as the manipulated "object".
# Its position is read from site kinematics — MuJoCo propagates the slide
# joint through FK so drawer.data.site_pos_w already reflects the current pos.
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
    drawer_entity_name: str = "drawer",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Vector from EE (pinch_site) to drawer handle (object_site) in world frame (3D).

    object_site lives on the articulated handle body, so its world position is
    automatically updated by MuJoCo FK whenever the slide joint changes.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    drawer: Entity = env.scene[drawer_entity_name]
    handle_ids = drawer.find_sites("object_site")[0]
    handle_pos_w = drawer.data.site_pos_w[:, handle_ids].squeeze(1)
    return handle_pos_w - ee_pos


def object_pos(
    env: ManagerBasedRlEnv,
    drawer_entity_name: str = "drawer",
) -> torch.Tensor:
    """Handle (object_site) position in local (robot-base) frame (3D)."""
    drawer: Entity = env.scene[drawer_entity_name]
    handle_ids = drawer.find_sites("object_site")[0]
    return drawer.data.site_pos_w[:, handle_ids].squeeze(1) - env.scene.env_origins


def _goal_handle_pos_w(
    env: ManagerBasedRlEnv,
    drawer_entity_name: str = "drawer",
    command_name: str = "drawer_goal",
) -> torch.Tensor:
    """Compute goal handle position in world frame via linear FK (N, 3).

    Drawer slide joint moves along x-axis in drawer_base frame (identity orientation).
    At goal slide value q_goal, object_site world pos is:
        drawer_base_pos_w + (q_goal + _HANDLE_OFFSET_X, 0, 0)
    """
    drawer: Entity = env.scene[drawer_entity_name]
    drawer_base_w = drawer.data.root_link_pos_w  # (N, 3)

    q_goal = _get_drawer_goal_command(env, command_name).goal_slide  # (N, 1)

    tx = q_goal + _HANDLE_OFFSET_X                         # (N, 1)
    ty = torch.full_like(tx, _HANDLE_OFFSET_Y)             # (N, 1)
    tz = torch.full_like(tx, _HANDLE_OFFSET_Z)             # (N, 1)

    return drawer_base_w + torch.cat([tx, ty, tz], dim=-1)  # (N, 3)


def goal_pos(
    env: ManagerBasedRlEnv,
    drawer_entity_name: str = "drawer",
    command_name: str = "drawer_goal",
) -> torch.Tensor:
    """Goal handle position in local (robot-base) frame (3D)."""
    return _goal_handle_pos_w(env, drawer_entity_name, command_name) - env.scene.env_origins


def object_to_goal(
    env: ManagerBasedRlEnv,
    drawer_entity_name: str = "drawer",
    command_name: str = "drawer_goal",
) -> torch.Tensor:
    """Vector from current handle position to goal handle position in world frame (3D)."""
    drawer: Entity = env.scene[drawer_entity_name]
    handle_ids = drawer.find_sites("object_site")[0]
    handle_pos_w = drawer.data.site_pos_w[:, handle_ids].squeeze(1)
    target_w = _goal_handle_pos_w(env, drawer_entity_name, command_name)
    return target_w - handle_pos_w


# ---------------------------------------------------------------------------
# Reward functions
# Mirrors open_door_osc: position-based Gaussians for both reach and goal phases.
# ---------------------------------------------------------------------------


def ee_to_object_reward(
    env: ManagerBasedRlEnv,
    std: float,
    drawer_entity_name: str = "drawer",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Gaussian reward for EE proximity to drawer handle (reach phase)."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    drawer: Entity = env.scene[drawer_entity_name]
    handle_ids = drawer.find_sites("object_site")[0]
    handle_pos_w = drawer.data.site_pos_w[:, handle_ids].squeeze(1)
    dist_sq = torch.sum(torch.square(handle_pos_w - ee_pos), dim=-1)
    return torch.nan_to_num(torch.exp(-dist_sq / std**2), nan=0.0)


class object_at_goal_reward:
    """Gaussian reward for handle proximity to goal position (open phase).

    Goal position is the FK-computed world position of object_site at goal_slide.
    This is a 3-D position error, identical in structure to cube_at_goal_reward.

    Debug vis draws:
      - Green sphere at current handle position
      - EE coordinate frame at pinch_site
    (Goal sphere is drawn by DrawerGoalCommand._debug_vis_impl)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._drawer_entity_name: str = cfg.params.get("drawer_entity_name", "drawer")
        self._command_name: str = cfg.params.get("command_name", "drawer_goal")
        self._ee_site_name: str = "pinch_site"
        self._debug_vis_enabled: bool = True
        robot: Entity = env.scene["robot"]
        self._ee_site_ids = robot.find_sites(self._ee_site_name)[0]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float,
        drawer_entity_name: str = "drawer",
        command_name: str = "drawer_goal",
    ) -> torch.Tensor:
        drawer: Entity = env.scene[drawer_entity_name]
        handle_ids = drawer.find_sites("object_site")[0]
        handle_pos_w = drawer.data.site_pos_w[:, handle_ids].squeeze(1)
        target_w = _goal_handle_pos_w(env, drawer_entity_name, command_name)
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

        drawer: Entity = env.scene[self._drawer_entity_name]
        robot: Entity = env.scene["robot"]
        handle_ids = drawer.find_sites("object_site")[0]

        for i in env_indices:
            # Current handle: green sphere
            handle_pos_np = drawer.data.site_pos_w[i, handle_ids].squeeze(0).cpu().numpy()
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
    drawer_entity_name: str = "drawer",
    command_name: str = "drawer_goal",
) -> torch.Tensor:
    """Euclidean distance from current handle position to goal handle position."""
    drawer: Entity = env.scene[drawer_entity_name]
    handle_ids = drawer.find_sites("object_site")[0]
    handle_pos_w = drawer.data.site_pos_w[:, handle_ids].squeeze(1)
    target_w = _goal_handle_pos_w(env, drawer_entity_name, command_name)
    return torch.norm(target_w - handle_pos_w, dim=-1)


def ee_to_object_error(
    env: ManagerBasedRlEnv,
    drawer_entity_name: str = "drawer",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    """Euclidean distance from EE to drawer handle."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    drawer: Entity = env.scene[drawer_entity_name]
    handle_ids = drawer.find_sites("object_site")[0]
    handle_pos_w = drawer.data.site_pos_w[:, handle_ids].squeeze(1)
    return torch.norm(handle_pos_w - ee_pos, dim=-1)


# success_rate: per-step indicator that handle is within tolerance of goal.
# MetricsManager averages across the episode, so the logged value is the
# *fraction of steps* the handle stayed within tolerance — i.e. a "dwell"
# success. Strictly harder than terminal success (a policy that briefly
# touches the band but drifts off scores below 1.0). Mjlab's MetricsTermCfg
# does not currently expose a `reduce="last"` mode; using the dwell
# interpretation is the closest faithful Phase-0 metric without patching
# mjlab.
def success_rate(
    env: ManagerBasedRlEnv,
    threshold: float = 0.02,
    drawer_entity_name: str = "drawer",
    command_name: str = "drawer_goal",
) -> torch.Tensor:
    err = object_to_goal_error(env, drawer_entity_name, command_name)
    return (err < threshold).float()


# ---------------------------------------------------------------------------
# Termination functions
# ---------------------------------------------------------------------------


def drawer_out_of_range(
    env: ManagerBasedRlEnv,
    drawer_entity_name: str = "drawer",
) -> torch.Tensor:
    """Terminate if the drawer slide joint goes out of its valid range [-0.25, 0]."""
    drawer: Entity = env.scene[drawer_entity_name]
    slide = drawer.data.joint_pos.squeeze(-1)
    return (slide > 0.05) | (slide < -0.30)


# ---------------------------------------------------------------------------
# Phase knobs — robustness improvement plan deltas vs. the Phase 0 baseline.
# ---------------------------------------------------------------------------


@dataclass
class PhaseKnobs:
    """Per-phase config deltas applied on top of the Phase 0 baseline.

    Each field is the *new* value; ``None`` means "leave the Phase 0
    default in place." See ``docs/open_drawer_improvement_plan.md``.
    """

    # Phase 1 — initial-pose randomization.
    joint_delta_deg: float | None = None  # Phase 0: 5.0 → Phase 1: 15.0
    base_pose_range: dict | None = None   # Phase 0: {} → Phase 1: ±2cm/±2°yaw

    # Phase 2 — targeted DR (drawer + arm).
    drawer_friction_range: tuple[float, float] | None = None   # frictionloss
    drawer_damping_range:  tuple[float, float] | None = None
    drawer_mass_alpha_range: tuple[float, float] | None = None  # pseudo_inertia α
    arm_link_mass_alpha_range: tuple[float, float] | None = None

    # Phase 3 — slow execution.
    osc_delta_pos_clip: float | None = None  # 3a (None disables clip)
    quadratic_vel_penalty_weight: float | None = None  # 3b
    max_vel_curriculum: list[dict] | None = None  # 3c stages
    add_processed_action_obs: bool = False  # 3d
    action_scale_dr_range: tuple[float, float] | None = None  # 3e


def kinova_open_drawer_osc_env_cfg(
    play: bool = False,
    phase_knobs: PhaseKnobs | None = None,
) -> ManagerBasedRlEnvCfg:
    """Kinova Gen3 drawer-opening task with OSC (relative mode) + gripper control.

    Drawer starts closed. The policy must grasp the handle and pull it open to a
    randomly sampled target slide distance in [-0.25 m, -0.15 m].

    Actions (7D):
        - osc_pose (6D): relative EE position + axis-angle delta
        - gripper (1D): -1=open, +1=closed
    """

    # --- Observations ---
    # Mirrors open_door_osc. Handle = "object"; goal = FK-computed target handle pos.
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
                "drawer_entity_name": "drawer",
                "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "object_pos": ObservationTermCfg(
            func=object_pos,
            params={"drawer_entity_name": "drawer"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "object_to_goal": ObservationTermCfg(
            func=object_to_goal,
            params={"drawer_entity_name": "drawer", "command_name": "drawer_goal"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "goal_pos": ObservationTermCfg(
            func=goal_pos,
            params={"drawer_entity_name": "drawer", "command_name": "drawer_goal"},
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
        "reset_drawer": EventTermCfg(
            func=reset_drawer,
            mode="reset",
            params={
                "drawer_entity_name": "drawer",
                "x_range": (_DRAWER_POS_LO[0], _DRAWER_POS_HI[0]),
                "y_range": (_DRAWER_POS_LO[1], _DRAWER_POS_HI[1]),
                "z": _DRAWER_POS_LO[2],
                "init_slide_range": (_DRAWER_INIT_SLIDE_LO, _DRAWER_INIT_SLIDE_HI),
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
        # Phase 1: EE reaches handle (Gaussian, std=0.15)
        "reach_object": RewardTermCfg(
            func=ee_to_object_reward,
            weight=1.0,
            params={
                "std": 0.15,
                "drawer_entity_name": "drawer",
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
            params={"drawer_entity_name": "drawer", "command_name": "drawer_goal"},
        ),
        "ee_to_object_error": MetricsTermCfg(
            func=ee_to_object_error,
            params={
                "drawer_entity_name": "drawer",
                "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
            },
        ),
        # Phase 0 primary metric — see comment on success_rate(): dwell-fraction
        # of steps within 2 cm of the goal handle position.
        "success_rate": MetricsTermCfg(
            func=success_rate,
            params={
                "threshold": 0.02,
                "drawer_entity_name": "drawer",
                "command_name": "drawer_goal",
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
                "robot":  get_kinova_robot_cfg_peginhole_osc(),
                "drawer": get_drawer_cfg(),
            },
            sensors=(ee_ground_collision_cfg,),
        ),
        observations=observations,
        actions=actions,
        commands={
            "drawer_goal": DrawerGoalCommandCfg(
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

    if phase_knobs is not None:
        _apply_phase_knobs(cfg, phase_knobs)

    return cfg


def _apply_phase_knobs(cfg: ManagerBasedRlEnvCfg, k: PhaseKnobs) -> None:
    """Apply Phase 1+ deltas on top of the Phase 0 baseline cfg, in place."""

    # --- Phase 1: initial-pose randomization ---
    if k.joint_delta_deg is not None:
        cfg.events["reset_robot_joints"].params["joint_delta_deg"] = k.joint_delta_deg
    if k.base_pose_range is not None:
        cfg.events["reset_base"].params["pose_range"] = dict(k.base_pose_range)

    # --- Phase 2: targeted DR ---
    if k.drawer_friction_range is not None:
        cfg.events["dr_drawer_friction"] = EventTermCfg(
            mode="startup",
            func=mdp.dr.dof_frictionloss,
            params={
                "asset_cfg": SceneEntityCfg("drawer", joint_names=("drawer_slide",)),
                "operation": "abs",
                "distribution": "uniform",
                "ranges": tuple(k.drawer_friction_range),
            },
        )
    if k.drawer_damping_range is not None:
        cfg.events["dr_drawer_damping"] = EventTermCfg(
            mode="startup",
            func=mdp.dr.dof_damping,
            params={
                "asset_cfg": SceneEntityCfg("drawer", joint_names=("drawer_slide",)),
                "operation": "abs",
                "distribution": "uniform",
                "ranges": tuple(k.drawer_damping_range),
            },
        )
    if k.drawer_mass_alpha_range is not None:
        cfg.events["dr_drawer_mass"] = EventTermCfg(
            mode="startup",
            func=mdp.dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("drawer", body_names=("drawer_base",)),
                "alpha_range": tuple(k.drawer_mass_alpha_range),
                "distribution": "uniform",
            },
        )
    if k.arm_link_mass_alpha_range is not None:
        # Arm links only — excludes the massless `end_effector_link` (mass=0
        # → log-uniform pseudo_inertia produces NaN) and the gripper
        # `*_spring_link` bodies (gripper internals, not load-bearing for
        # arm dynamics). The earlier `.*_link` match caught both and NaN'd
        # every env at iter 0.
        _ARM_LINK_RE = (
            r"(base|shoulder|half_arm_[12]|forearm|"
            r"spherical_wrist_[12]|bracelet)_link"
        )
        cfg.events["dr_arm_link_mass"] = EventTermCfg(
            mode="startup",
            func=mdp.dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=_ARM_LINK_RE),
                "alpha_range": tuple(k.arm_link_mass_alpha_range),
                "distribution": "uniform",
            },
        )

    # --- Phase 3: slow execution ---
    if k.osc_delta_pos_clip is not None:
        # Symmetric clip on the OSC delta-position channel (first 3 of 6 OSC dims).
        c = float(k.osc_delta_pos_clip)
        cfg.actions["osc_pose"].clip = {
            ".*": (-1.0, 1.0),  # default
            "osc_pose": [(-c, c)] * 3 + [(-1.0, 1.0)] * 3,
        }
    if k.quadratic_vel_penalty_weight is not None:
        # Replace the linear hinge with a quadratic vel penalty.
        from mjlab.tasks.manipulation import mdp as manipulation_mdp_local  # noqa: F401
        cfg.rewards.pop("joint_vel_hinge", None)
        cfg.rewards["joint_vel_quadratic"] = RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=k.quadratic_vel_penalty_weight,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[1-7]",))},
        )
    if k.max_vel_curriculum is not None:
        cfg.curriculum["joint_vel_hinge_max_vel"] = CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={"reward_name": "joint_vel_hinge", "stages": list(k.max_vel_curriculum)},
        )
    if k.add_processed_action_obs:
        # Augment actor obs with the previous *processed* action.
        # mjlab's last_action returns the raw action; we expose processed via
        # a custom hook on the env. Defer until mjlab 1.4 lands the API; for
        # now this knob is a no-op placeholder so the wiring is in place.
        pass
    if k.action_scale_dr_range is not None:
        # Per-env OSC delta_pos_scale DR. Mjlab's mdp.dr.* doesn't expose this
        # because delta_pos_scale is a config field, not a model field.
        # Wiring deferred to a Phase 3e custom OSC wrapper.
        pass


# ---------------------------------------------------------------------------
# Phase preset factory functions — registered as separate task IDs so each
# phase's deltas vs. Phase 0 are explicit and reproducible. See plan §3.
# ---------------------------------------------------------------------------


def _phase1_knobs() -> PhaseKnobs:
    """Phase 1: wider initial-pose randomization."""
    return PhaseKnobs(
        joint_delta_deg=15.0,
        base_pose_range={
            "x":   (-0.02, 0.02),
            "y":   (-0.02, 0.02),
            "yaw": (-_DEG_TO_RAD * 2.0, _DEG_TO_RAD * 2.0),
        },
    )


def _phase2_knobs() -> PhaseKnobs:
    """Phase 2: targeted DR on top of Phase 1 keepers (init pose stays from P1)."""
    k = _phase1_knobs()
    # Drawer slide friction: nominal ~0.01; widen to [0.005, 0.02] (½ × .. 2×).
    k.drawer_friction_range = (0.005, 0.02)
    # Drawer slide damping: nominal 1.0; widen to [0.5, 2.0].
    k.drawer_damping_range = (0.5, 2.0)
    # Drawer base mass: scale ∈ [0.5, 2.0] → α = 0.5·ln(scale) ∈ [-0.347, +0.347].
    a_drawer = 0.5 * _math.log(2.0)
    k.drawer_mass_alpha_range = (-a_drawer, a_drawer)
    # Arm link mass: ±10% → α ∈ [-0.0477, +0.0477] (narrow; widen in Phase 4).
    a_arm = 0.5 * _math.log(1.10)
    k.arm_link_mass_alpha_range = (-a_arm, a_arm)
    return k


def _phase4_knobs() -> PhaseKnobs:
    """Phase 4 placeholder: starts as Phase 2 + (Phase 3 keeper if any).

    Final ranges and Phase 3 keeper choice are decided after Phases 1-3
    finish, so this is a sensible default that can be tuned at Phase 4
    submission time.
    """
    k = _phase2_knobs()
    # Widen drawer mass to [0.5, 5.0] → α ∈ [-0.347, +0.805].
    k.drawer_mass_alpha_range = (-0.5 * _math.log(2.0), 0.5 * _math.log(5.0))
    # Widen arm link mass to ±25%.
    a_arm = 0.5 * _math.log(1.25)
    k.arm_link_mass_alpha_range = (-a_arm, a_arm)
    # Phase 3 keeper inserted here at submit time. Default: no slow-exec change.
    return k


def kinova_open_drawer_osc_phase1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return kinova_open_drawer_osc_env_cfg(play=play, phase_knobs=_phase1_knobs())


def kinova_open_drawer_osc_phase2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return kinova_open_drawer_osc_env_cfg(play=play, phase_knobs=_phase2_knobs())


def kinova_open_drawer_osc_phase4_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return kinova_open_drawer_osc_env_cfg(play=play, phase_knobs=_phase4_knobs())


def kinova_open_drawer_osc_eval_env_cfg() -> ManagerBasedRlEnvCfg:
    """OOD-evaluation variant of the open-drawer OSC task.

    Phase 0: skeleton only. Inherits the training config but disables
    observation corruption and curriculum so eval numbers are deterministic.
    Episode length is kept at the training value (10 s) so success_rate is
    comparable to training-time numbers. Sweep CLI overrides (drawer
    friction, base mass, init slide, etc.) will be wired up in a later
    phase alongside the sweep harness.
    """
    cfg = kinova_open_drawer_osc_env_cfg(play=False)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    return cfg


def kinova_open_drawer_osc_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for open-drawer OSC task."""
    return kinova_ppo_runner_cfg(experiment_name="kinova_open_drawer_osc")
