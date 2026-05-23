"""Kinova Gen3 snap-fit task with Operational Space Control (OSC, relative mode).

Task: Insert a snap peg (held closed in the gripper) horizontally (+X) into a
fixed-in-air socket. The peg starts pre-aligned, with its tip just at the
socket entrance (lip plane). The socket has two cantilever jaws on torsion
springs that splay open as the peg's bulge passes through, then snap shut
behind it. Success = bulge past the lip plane AND peg seated near back-stop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from kinova_tasks.assets.kinova_gen3 import get_kinova_robot_cfg_snap_fit_osc
from kinova_tasks.assets.snap_fit import get_snap_peg_cfg, get_snap_socket_cfg
from kinova_tasks.tasks.actions.osc import OperationalSpaceActionCfg
from kinova_tasks.tasks.base_rl_cfg import kinova_ppo_runner_cfg
from kinova_tasks.tasks.peg_in_hole_osc import (
    GRIPPER_JOINT_NAMES,
    reset_gripper_ctrl_closed,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.managers.action_manager import ActionTermCfg
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
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    sample_uniform,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Home pose (matches reach_osc): EE points world +X, mid-height, gripper closed
# ---------------------------------------------------------------------------

_HOME_JOINT_POS = (
    0.0,            # joint_1   0°
    0.3490658504,   # joint_2  20°
    0.0,            # joint_3   0°
    1.7453292519,   # joint_4 100°
    0.0,            # joint_5   0°
    -0.5235987756,  # joint_6 -30°
    -1.5707963268,  # joint_7 -90°
)
# FK pinch_site at this pose (local frame), copied from reach_osc.
_HOME_POS = (0.733607, -0.024850, 0.523015)
_DEG_TO_RAD = 3.14159265358979 / 180.0

# Socket placement.
# Peg-local frame: base half-extent in X = 0.015, prong pivots at peg-local x=+0.015
# (right at the base front face). Prong arms 40 mm long, splayed at 14° → sphere
# tips at peg-local (~+0.0538, 0, ±0.0237) at rest.
# Socket-local frame: flap_inner at x=0 (entrance plane), flap_outer at x=+0.065,
# back_wall at x=+0.077.
# Place socket so the prong sphere tips sit 1 mm before the flap inner plane — the
# very first +X push sends the splayed sphere tips into the flap leading edges.
_PRONG_TIP_X_AT_REST = 0.0506  # peg-local x of prong-sphere center at 14° splay
# Stand-off between prong tips and socket flap_inner at reset (m). Robot must
# travel +X by roughly this distance before the snap engages.
_SOCKET_STANDOFF_X = 0.050
_SOCKET_DEFAULT_POS = (
    _HOME_POS[0] + _PRONG_TIP_X_AT_REST + _SOCKET_STANDOFF_X,
    _HOME_POS[1],
    _HOME_POS[2],
)


# ---------------------------------------------------------------------------
# Tighter closed-gripper pose for snap-fit (driver ~94% closed, matching ctrl=240).
# Robotiq 2F-85 driver joint range is [0, 0.8]; 0.753 ≈ 240/255.
# Other joints scaled from the 60%-closed reference (driver=0.503).
# ---------------------------------------------------------------------------
_SNAP_FIT_GRIPPER_CLOSED = {
    "right_driver_joint":      0.753,
    "right_coupler_joint":     0.0015,
    "right_spring_link_joint": 0.756,
    "right_follower_joint":   -0.726,
    "left_driver_joint":       0.753,
    "left_coupler_joint":      0.0015,
    "left_spring_link_joint":  0.756,
    "left_follower_joint":    -0.726,
}


def reset_gripper_joints_snap_fit_closed(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=GRIPPER_JOINT_NAMES
    ),
) -> None:
    """Write the snap-fit tighter closed-pose values for all 8 gripper joints."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    robot: Entity = env.scene[asset_cfg.name]
    n = len(env_ids)
    nj = len(_SNAP_FIT_GRIPPER_CLOSED)

    pos = torch.tensor(
        list(_SNAP_FIT_GRIPPER_CLOSED.values()), device=env.device
    ).unsqueeze(0).expand(n, -1).clone()
    vel = torch.zeros(n, nj, device=env.device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    robot.write_joint_state_to_sim(pos, vel, joint_ids=joint_ids, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Reset events
# ---------------------------------------------------------------------------


def reset_joints_with_delta(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    entity_name: str = "robot",
    joint_delta_deg: float = 2.0,
) -> None:
    """Reset arm joints (1-7) to horizontal home pose ± uniform delta."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    robot: Entity = env.scene[entity_name]
    soft_limits = robot.data.soft_joint_pos_limits

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


def reset_peg_in_gripper(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    peg_entity_name: str = "peg",
    robot_entity_name: str = "robot",
    ee_site_name: str = "pinch_site",
) -> None:
    """Place the peg at the actual pinch_site position with identity orientation.

    With the horizontal home pose, the EE tool axis points world +X, and the
    peg's local +X axis (shaft direction) aligned to world +X gives the peg
    shaft pointing forward into the socket. Identity quaternion is correct.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    env.scene.write_data_to_sim()
    env.sim.forward()

    robot: Entity = env.scene[robot_entity_name]
    peg: Entity = env.scene[peg_entity_name]

    ee_site_id = robot.find_sites(ee_site_name)[0]
    pos = robot.data.site_pos_w[env_ids, ee_site_id].squeeze(1).clone()  # (n, 3)

    n = len(env_ids)
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0  # identity (world-aligned)

    pose = torch.cat([pos, quat], dim=-1)
    vel = torch.zeros(n, 6, device=env.device)

    peg.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    peg.write_root_link_velocity_to_sim(vel, env_ids=env_ids)

    if not getattr(env, "_snap_fit_debug_printed", False):
        env._snap_fit_debug_printed = True
        env.scene.write_data_to_sim()
        env.sim.forward()
        try:
            right_pad_id = robot.find_geoms("right_pad1")[0]
            left_pad_id = robot.find_geoms("left_pad1")[0]
            base_pos = peg.data.root_link_pos_w[0]
            r_pad_pos = robot.data.geom_pos_w[0, right_pad_id]
            l_pad_pos = robot.data.geom_pos_w[0, left_pad_id]
            r_d = (r_pad_pos - base_pos).cpu().tolist()
            l_d = (l_pad_pos - base_pos).cpu().tolist()
            print(
                f"[snap_fit debug] env0 reset: peg_base_w={base_pos.cpu().tolist()} "
                f"right_pad-base={r_d} left_pad-base={l_d}",
                flush=True,
            )
        except Exception as exc:
            print(f"[snap_fit debug] pad-distance probe failed: {exc}", flush=True)


def reset_socket_position(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    socket_entity_name: str = "socket",
    x_range: tuple[float, float] = (_SOCKET_DEFAULT_POS[0], _SOCKET_DEFAULT_POS[0]),
    y_range: tuple[float, float] = (_SOCKET_DEFAULT_POS[1], _SOCKET_DEFAULT_POS[1]),
    z_range: tuple[float, float] = (_SOCKET_DEFAULT_POS[2], _SOCKET_DEFAULT_POS[2]),
) -> None:
    """Pose the socket mocap body in front of the EE."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    socket: Entity = env.scene[socket_entity_name]
    n = len(env_ids)

    lower = torch.tensor([x_range[0], y_range[0], z_range[0]], device=env.device)
    upper = torch.tensor([x_range[1], y_range[1], z_range[1]], device=env.device)
    pos = sample_uniform(lower, upper, (n, 3), device=env.device)
    pos = pos + env.scene.env_origins[env_ids]

    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0

    pose = torch.cat([pos, quat], dim=-1)
    socket.write_mocap_pose_to_sim(pose, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def ee_to_sites_distance(
    env: ManagerBasedRlEnv,
    target_entity: str,
    target_site_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("pinch_site",)),
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)

    target: Entity = env.scene[target_entity]
    parts = []
    for sname in target_site_names:
        sid = target.find_sites(sname)[0]
        site_pos = target.data.site_pos_w[:, sid].squeeze(1)
        parts.append(site_pos - ee_pos)

    return torch.cat(parts, dim=-1)


def peg_to_socket_sites_distance(
    env: ManagerBasedRlEnv,
    peg_entity: str = "peg",
    socket_entity: str = "socket",
    peg_site_names: tuple[str, ...] = ("lip_top_pivot", "lip_top_tip"),
    socket_site_names: tuple[str, ...] = ("flap_inner", "flap_outer"),
) -> torch.Tensor:
    peg: Entity = env.scene[peg_entity]
    socket: Entity = env.scene[socket_entity]

    parts = []
    for pname, sname in zip(peg_site_names, socket_site_names):
        pid = peg.find_sites(pname)[0]
        sid = socket.find_sites(sname)[0]
        peg_pos = peg.data.site_pos_w[:, pid].squeeze(1)
        socket_pos = socket.data.site_pos_w[:, sid].squeeze(1)
        parts.append(socket_pos - peg_pos)

    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Reward / metric / termination helpers
# ---------------------------------------------------------------------------


def peg_to_socket_reward(
    env: ManagerBasedRlEnv,
    std: float,
    peg_entity: str = "peg",
    socket_entity: str = "socket",
    peg_site_names: tuple[str, ...] = ("lip_top_pivot", "lip_top_tip"),
    socket_site_names: tuple[str, ...] = ("flap_inner", "flap_outer"),
) -> torch.Tensor:
    """Gaussian reward for peg-site → socket-site proximity (mirrors peg_in_hole)."""
    peg: Entity = env.scene[peg_entity]
    socket: Entity = env.scene[socket_entity]
    error = torch.zeros(env.num_envs, device=env.device)
    for pname, sname in zip(peg_site_names, socket_site_names):
        pid = peg.find_sites(pname)[0]
        sid = socket.find_sites(sname)[0]
        peg_pos = peg.data.site_pos_w[:, pid].squeeze(1)
        socket_pos = socket.data.site_pos_w[:, sid].squeeze(1)
        error += torch.sum(torch.square(peg_pos - socket_pos), dim=-1)
    return torch.nan_to_num(torch.exp(-error / std**2), nan=0.0)


def lips_past_flap(
    env: ManagerBasedRlEnv,
    peg_entity: str = "peg",
    socket_entity: str = "socket",
) -> torch.Tensor:
    """1.0 if BOTH lip pivots have passed the flap outer-edge plane (along world +X).

    Once the pivots are past flap_outer, the lip arms are no longer constrained
    by the flap walls — they spring back to their rest splayed position and the
    snap is achieved.
    """
    peg: Entity = env.scene[peg_entity]
    socket: Entity = env.scene[socket_entity]
    top_pivot_id = peg.find_sites("lip_top_pivot")[0]
    bot_pivot_id = peg.find_sites("lip_bot_pivot")[0]
    flap_outer_id = socket.find_sites("flap_outer")[0]
    top_x = peg.data.site_pos_w[:, top_pivot_id, 0].squeeze(-1)
    bot_x = peg.data.site_pos_w[:, bot_pivot_id, 0].squeeze(-1)
    flap_x = socket.data.site_pos_w[:, flap_outer_id, 0].squeeze(-1)
    return ((top_x > flap_x) & (bot_x > flap_x)).float()


def snap_fit_success(
    env: ManagerBasedRlEnv,
    peg_entity: str = "peg",
    socket_entity: str = "socket",
    seated_threshold: float = 0.01,
) -> torch.Tensor:
    """Lips popped through (both pivots past flap outer edge) AND peg seated.

    Seated = peg origin within `seated_threshold` of the socket's seated_target
    site (which is positioned so that when the peg origin is there, the peg base
    back face is touching the back wall).
    """
    peg: Entity = env.scene[peg_entity]
    socket: Entity = env.scene[socket_entity]

    top_pivot_id = peg.find_sites("lip_top_pivot")[0]
    bot_pivot_id = peg.find_sites("lip_bot_pivot")[0]
    flap_outer_id = socket.find_sites("flap_outer")[0]
    object_id = peg.find_sites("object_site")[0]
    seated_id = socket.find_sites("seated_target")[0]

    top_x = peg.data.site_pos_w[:, top_pivot_id, 0].squeeze(-1)
    bot_x = peg.data.site_pos_w[:, bot_pivot_id, 0].squeeze(-1)
    flap_x = socket.data.site_pos_w[:, flap_outer_id, 0].squeeze(-1)
    past_flap = (top_x > flap_x) & (bot_x > flap_x)

    object_pos = peg.data.site_pos_w[:, object_id].squeeze(1)
    seated_pos = socket.data.site_pos_w[:, seated_id].squeeze(1)
    seated = torch.norm(object_pos - seated_pos, dim=-1) < seated_threshold
    return (past_flap & seated).float()


def peg_to_seated_error(
    env: ManagerBasedRlEnv,
    peg_entity: str = "peg",
    socket_entity: str = "socket",
) -> torch.Tensor:
    peg: Entity = env.scene[peg_entity]
    socket: Entity = env.scene[socket_entity]
    object_id = peg.find_sites("object_site")[0]
    seated_id = socket.find_sites("seated_target")[0]
    object_pos = peg.data.site_pos_w[:, object_id].squeeze(1)
    seated_pos = socket.data.site_pos_w[:, seated_id].squeeze(1)
    return torch.norm(object_pos - seated_pos, dim=-1)


def peg_slip_termination(
    env: ManagerBasedRlEnv,
    peg_entity: str = "peg",
    robot_entity: str = "robot",
    ee_site_name: str = "pinch_site",
    pos_threshold: float = 0.05,
    angle_threshold: float = 2.618,
) -> torch.Tensor:
    """Terminate if peg slips relative to gripper (mirrors peg_in_hole logic)."""
    peg: Entity = env.scene[peg_entity]
    robot: Entity = env.scene[robot_entity]

    peg_pos_w = peg.data.root_link_pos_w
    peg_quat_w = peg.data.root_link_quat_w

    ee_site_id = robot.find_sites(ee_site_name)[0]
    ee_pos_w = robot.data.site_pos_w[:, ee_site_id].squeeze(1)
    ee_quat_w = robot.data.site_quat_w[:, ee_site_id].squeeze(1)

    peg_pos_wrt_ee = quat_apply_inverse(ee_quat_w, peg_pos_w - ee_pos_w)
    peg_quat_wrt_ee = quat_mul(quat_conjugate(ee_quat_w), peg_quat_w)

    if not hasattr(env, "_snap_peg_slip_init_pos"):
        env._snap_peg_slip_init_pos = peg_pos_wrt_ee.clone()
        env._snap_peg_slip_init_quat = peg_quat_wrt_ee.clone()

    just_reset = env.episode_length_buf == 1
    if just_reset.any():
        env._snap_peg_slip_init_pos[just_reset] = peg_pos_wrt_ee[just_reset].clone()
        env._snap_peg_slip_init_quat[just_reset] = peg_quat_wrt_ee[just_reset].clone()

    delta_pos = peg_pos_wrt_ee - env._snap_peg_slip_init_pos
    pos_norm = torch.linalg.norm(delta_pos, dim=-1)

    delta_quat = quat_mul(quat_conjugate(env._snap_peg_slip_init_quat), peg_quat_wrt_ee)
    angle_norm = torch.linalg.norm(axis_angle_from_quat(delta_quat), dim=-1)

    return (pos_norm > pos_threshold) | (angle_norm > angle_threshold)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


def kinova_snap_fit_osc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Kinova snap-fit with horizontal OSC push (relative mode).

    The peg is placed at the EE pinch_site on reset (identity orientation, so
    peg-X aligns with world-X). The socket mocap is placed in front of the EE
    so the peg tip starts at the socket entrance (lip plane).
    """

    # --- Observations ---
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[1-7]",))},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "peg_to_socket": ObservationTermCfg(
            func=peg_to_socket_sites_distance,
            params={
                "peg_entity": "peg",
                "socket_entity": "socket",
                "peg_site_names": ("lip_top_pivot", "lip_top_tip"),
                "socket_site_names": ("flap_inner", "flap_outer"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "ee_to_socket": ObservationTermCfg(
            func=ee_to_sites_distance,
            params={
                "target_entity": "socket",
                "target_site_names": ("flap_inner", "seated_target"),
                "asset_cfg": SceneEntityCfg("robot", site_names=("pinch_site",)),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(actor_terms, enable_corruption=False),
    }

    # --- Actions: same OSC config as peg_in_hole_osc ---
    actions: dict[str, ActionTermCfg] = {
        "osc_pose": OperationalSpaceActionCfg(
            entity_name="robot",
            actuator_names=("joint_.*",),
            frame_name="pinch_site",
            frame_type="site",
            use_relative_mode=True,
            delta_pos_scale=0.02,
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
        "reset_gripper_joints": EventTermCfg(
            func=reset_gripper_joints_snap_fit_closed,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES),
            },
        ),
        "reset_gripper_ctrl": EventTermCfg(
            func=reset_gripper_ctrl_closed,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=("fingers_actuator",)),
                "closed_ctrl": 240.0,
            },
        ),
        "reset_socket_position": EventTermCfg(
            func=reset_socket_position,
            mode="reset",
            params={"socket_entity_name": "socket"},
        ),
        "reset_peg_in_gripper": EventTermCfg(
            func=reset_peg_in_gripper,
            mode="reset",
            params={
                "peg_entity_name": "peg",
                "robot_entity_name": "robot",
                "ee_site_name": "pinch_site",
            },
        ),
    }

    # --- Rewards ---
    rewards = {
        # Approach: peg sites toward socket sites (broad Gaussian)
        "approach": RewardTermCfg(
            func=peg_to_socket_reward,
            weight=1.0,
            params={"std": 0.05},
        ),
        # Tight seated bonus (narrow Gaussian)
        "seated": RewardTermCfg(
            func=peg_to_socket_reward,
            weight=2.0,
            params={"std": 0.01},
        ),
        # Sparse: both lip pivots past flap outer edge (snap achieved)
        "past_flap": RewardTermCfg(
            func=lips_past_flap,
            weight=1.0,
        ),
    }

    # --- Terminations ---
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_detection": TerminationTermCfg(func=nan_detection, time_out=False),
        "peg_slip": TerminationTermCfg(
            func=peg_slip_termination,
            params={
                "peg_entity": "peg",
                "robot_entity": "robot",
                "ee_site_name": "pinch_site",
                "pos_threshold": 0.05,
                "angle_threshold": 2.618,
            },
        ),
    }

    # --- Metrics ---
    metrics = {
        "peg_to_seated_error": MetricsTermCfg(func=peg_to_seated_error),
        "success": MetricsTermCfg(func=snap_fit_success),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=4096,
            env_spacing=1.0,
            entities={
                "robot": get_kinova_robot_cfg_snap_fit_osc(),
                "peg": get_snap_peg_cfg(),
                "socket": get_snap_socket_cfg(),
            },
            sensors=(),
        ),
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        curriculum={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=1.5,
            elevation=-10.0,
            azimuth=180.0,
        ),
        sim=SimulationCfg(
            nconmax=80,
            njmax=800,
            mujoco=MujocoCfg(
                timestep=0.002,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=50,
        episode_length_s=5.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

    return cfg


def kinova_snap_fit_osc_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for snap-fit OSC task."""
    return kinova_ppo_runner_cfg(experiment_name="kinova_snap_fit_osc")
