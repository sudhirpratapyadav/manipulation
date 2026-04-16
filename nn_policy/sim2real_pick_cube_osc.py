"""Kinova Gen3 sim2real for pick-and-lift cube task with OSC.

An NN policy (RSL-RL PPO checkpoint) drives both sim and real arm simultaneously.
A single policy forward pass with batch=2 produces actions for both at once.

Policy observations (33-D, same as pick_cube_osc training checkpoint):
    joint_vel(7) + ee_pose(6) + gripper_state(1) + ee_to_cube(3) +
    cube_pos(3) + cube_to_goal(3) + goal_pos(3) + last_action(7)

  joint_vel      : arm joint velocities [rad/s]
  ee_pose        : [ee_pos(3), ee_axis_angle(3)] in robot-local frame
  gripper_state  : right_driver_joint / 0.8  (normalized to [0=open, 1=closed])
  ee_to_cube     : cube_pos_world - pinch_site_world  (3D)
  cube_pos       : cube_pos_world - env_origin  (local frame, 3D)
  cube_to_goal   : goal_pos_world - cube_pos_world  (3D)
  goal_pos       : goal_pos_world - env_origin  (local frame, 3D)
  last_action    : previous 7D policy action

Policy action (7D):
    [delta_pos(3), delta_ori_axis_angle(3), gripper(1)]
    OSC part scaled by DELTA_POS_SCALE / DELTA_ORI_SCALE.
    gripper: -1=fully open, +1=fully closed → mapped to [0, 255] ctrl.

Cube pose:
    For sim: comes from MuJoCo physics (freejoint).
    For real: randomised on each reset (will be replaced by perception later).

Processes / Threads:
  - Policy process  (mp.Process): batch=2 inference at TARGET_HZ Hz
  - Real process    (mp.Process): 500 Hz OSC → hardware torques + gripper
  - OSC thread      (threading):  500 Hz OSC + MuJoCo physics (sim)
  - Viz thread      (threading):  30 Hz scene update
  - Main thread:                  viser GUI + polling

Run:
  python sim2real_pick_cube_osc.py --checkpoint weights/model_499.pt --ip 192.168.1.10
  python sim2real_pick_cube_osc.py --checkpoint weights/model_499.pt --sim-only
"""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import threading
import time

import mujoco
import numpy as np
import pinocchio as pin
import torch
import viser
from scipy.spatial.transform import Rotation

from hardware import KinovaHardware
from kinova_tasks.assets.kinova_gen3.kinova_constants import (
    KINOVA_GEN3_GRIPPER_TORQUE_XML,
    get_assets,
)
from kinova_tasks.assets.objects.free.cube.cube_constants import CUBE_XML
from policy import PolicyAgent
from viewer import ViserMujocoScene

# ── Model ─────────────────────────────────────────────────────────────────────
_TORQUE_XML      = KINOVA_GEN3_GRIPPER_TORQUE_XML
_ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]

_GRIPPER_JOINT_NAMES = [
    "right_driver_joint", "right_coupler_joint",
    "right_spring_link_joint", "right_follower_joint",
    "left_driver_joint",  "left_coupler_joint",
    "left_spring_link_joint",  "left_follower_joint",
]
_GRIPPER_OPEN_QPOS   = np.zeros(8)
_GRIPPER_OPEN_CTRL   = 0.0
_GRIPPER_DRIVER_MAX  = 0.8   # right_driver_joint range: [0=open, 0.8=closed]

# ── Home joint positions — must match pick_cube_osc.py _HOME_JOINT_POS ────────
# [90°, 30°, 0°, 90°, 0°, 60°, -90°] in radians
HOME_DEG = np.array([90.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])

# ── Cube spawn range — matches _CUBE_SPAWN_LO / _CUBE_SPAWN_HI from pick_cube_osc.py ──
# Robot-local frame.  Real robot: randomised for now, will be replaced by perception.
CUBE_SPAWN_X = (-0.05,  0.05)
CUBE_SPAWN_Y = (-0.48, -0.52)
CUBE_SPAWN_Z = ( 0.02, 0.03)

# CUBE_SPAWN_X = (-0.3,  0.3)
# CUBE_SPAWN_Y = (-0.7, -0.3)
# CUBE_SPAWN_Z = ( 0.02, 0.03)

# ── Goal position range (robot-local frame, matches _GOAL_LO / _GOAL_HI) ──────
GOAL_LO = np.array([-0.05, -0.48, 0.1])
GOAL_HI = np.array([ 0.05, -0.52, 0.30])

# GOAL_LO = np.array([-0.30, -0.70, 0.05])
# GOAL_HI = np.array([ 0.30, -0.30, 0.40])


# ── OSC gains — must match pick_cube_osc.py ───────────────────────────────────
KP_POS         =  50.0
KD_POS         =  2.0
KP_ORI         =  50.0
KD_ORI         =  2.0
POSTURE_KP     =  10.0
POSTURE_KD     =   2.0
POSTURE_WEIGHT =   0.0

MAX_JOINT_TORQUE    = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])
TAU_OFFSETS_DEFAULT = np.array([0.0,  0.0,  -0.5, 0.0,  0.0, 1.0, 0.0])

# ── Action scales — must match pick_cube_osc.py ───────────────────────────────
DELTA_POS_SCALE = 0.005   # metres per unit action (half of training 0.01)
DELTA_ORI_SCALE = 0.01    # rad per unit action (half of training 0.02)

# ── Gripper ctrl mapping: policy [-1, +1] → fingers_actuator [0, 255] ─────────
GRIPPER_CTRL_MIN = 0.0
GRIPPER_CTRL_MAX = 255.0

# ── Workspace bounds for OSC target clipping ──────────────────────────────────
_WS_RADIUS = 0.25   # metres

# ── Timing ────────────────────────────────────────────────────────────────────
PHYSICS_DT   = 0.002
TARGET_HZ    = 10
OSC_HZ       = 500
OSC_SUBSTEPS = OSC_HZ // TARGET_HZ
VIZ_HZ       = 30

# ── Gains key layout ──────────────────────────────────────────────────────────
GAINS_KEYS = ["kp_pos", "kd_pos", "kp_ori", "kd_ori",
              "posture_kp", "posture_kd", "posture_weight"]


# ── Shared memory helpers ─────────────────────────────────────────────────────
def _np(shm: mp.Array) -> np.ndarray:
    """Zero-copy numpy view of a multiprocessing double Array."""
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


def _pack_gains(kp_pos, kd_pos, kp_ori, kd_ori,
                posture_kp, posture_kd, posture_weight) -> np.ndarray:
    return np.array([kp_pos, kd_pos, kp_ori, kd_ori,
                     posture_kp, posture_kd, posture_weight])


def _gains_dict(arr: np.ndarray) -> dict:
    return dict(zip(GAINS_KEYS, arr))


# ── Helpers ───────────────────────────────────────────────────────────────────

def kinova_deg_to_rad(deg: np.ndarray) -> np.ndarray:
    s = deg.copy()
    s[s > 180.0] -= 360.0
    return np.deg2rad(s)


class PinocchioArm:
    def __init__(self, mjcf_path: str, ee_frame: str) -> None:
        self.model = pin.buildModelFromMJCF(mjcf_path)
        self.model.gravity.linear = np.array([0.0, 0.0, -9.81])
        self.data = self.model.createData()

        self.ee_frame_id = self.model.getFrameId(ee_frame)
        if self.ee_frame_id >= self.model.nframes:
            raise ValueError(f"Frame '{ee_frame}' not found")

        self._v_idx = np.array(
            [self.model.joints[self.model.getJointId(n)].idx_v for n in _ARM_JOINT_NAMES],
            dtype=np.intp,
        )
        self._q_idx = np.array(
            [self.model.joints[self.model.getJointId(n)].idx_q for n in _ARM_JOINT_NAMES],
            dtype=np.intp,
        )
        self._q_full  = pin.neutral(self.model)
        self._dq_full = np.zeros(self.model.nv)

    def _set_q(self, q, dq=None):
        self._q_full[self._q_idx] = q
        if dq is not None:
            self._dq_full[self._v_idx] = dq

    def fk(self, q):
        self._set_q(q)
        pin.framesForwardKinematics(self.model, self.data, self._q_full)
        oMf = self.data.oMf[self.ee_frame_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def jacobian(self, q):
        self._set_q(q)
        pin.computeJointJacobians(self.model, self.data, self._q_full)
        pin.framesForwardKinematics(self.model, self.data, self._q_full)
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id,
                                 pin.LOCAL_WORLD_ALIGNED)
        return J[:, self._v_idx]

    def dynamics(self, q, dq):
        """Returns (M_arm (7,7), nle_arm (7,), J_dot_dq (6,))."""
        self._set_q(q, dq)

        pin.crba(self.model, self.data, self._q_full)
        M_sub = self.data.M[np.ix_(self._v_idx, self._v_idx)]
        M_arm = 0.5 * (M_sub + M_sub.T)

        pin.nonLinearEffects(self.model, self.data, self._q_full, self._dq_full)
        nle_arm = self.data.nle[self._v_idx].copy()

        pin.computeJointJacobians(self.model, self.data, self._q_full)
        pin.framesForwardKinematics(self.model, self.data, self._q_full)
        pin.computeJointJacobiansTimeVariation(self.model, self.data,
                                               self._q_full, self._dq_full)
        J_dot = pin.getFrameJacobianTimeVariation(
            self.model, self.data, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        J_dot_dq = J_dot[:, self._v_idx] @ dq
        return M_arm, nle_arm, J_dot_dq


def pose_error_6d(target_pos, target_quat_xyzw, cur_pos, cur_rot):
    pos_err = target_pos - cur_pos
    ori_err = Rotation.from_matrix(
        Rotation.from_quat(target_quat_xyzw).as_matrix() @ cur_rot.T
    ).as_rotvec()
    return np.concatenate([pos_err, ori_err])


def compute_osc_torques(robot, target_pos, target_quat_xyzw, q, dq, *,
                        gains, posture_target):
    """Operational Space Control → joint torques."""
    ee_pos, ee_rot = robot.fk(q)
    error = pose_error_6d(target_pos, target_quat_xyzw, ee_pos, ee_rot)

    J      = robot.jacobian(q)
    ee_vel = J @ dq

    ddx_des = np.empty(6)
    ddx_des[:3] = gains["kp_pos"] * error[:3] + gains["kd_pos"] * (0.0 - ee_vel[:3])
    ddx_des[3:] = gains["kp_ori"] * error[3:] + gains["kd_ori"] * (0.0 - ee_vel[3:])

    M, nle, J_dot_dq = robot.dynamics(q, dq)
    M_inv     = np.linalg.inv(M)
    Lambda    = np.linalg.inv(J @ M_inv @ J.T)
    J_dyn_inv = M_inv @ J.T @ Lambda

    F = Lambda @ (ddx_des - J_dot_dq)
    N = np.eye(7) - J.T @ J_dyn_inv.T
    tau_posture = (gains["posture_kp"] * (posture_target - q)
                   + gains["posture_kd"] * (0.0 - dq))

    tau = J.T @ F + nle + gains["posture_weight"] * (N @ tau_posture)
    return np.clip(tau, -MAX_JOINT_TORQUE, MAX_JOINT_TORQUE)


# ── Arrow mesh helper ─────────────────────────────────────────────────────────

def _make_arrow_mesh(start: np.ndarray, end: np.ndarray,
                     shaft_r: float = 0.003, head_r: float = 0.008,
                     head_frac: float = 0.30, n: int = 10):
    """Triangle mesh (vertices, faces) for an arrow from start → end."""
    d      = end - start
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        end    = start + np.array([0., 0., 1e-4])
        d      = end - start
        length = 1e-4

    shaft_len = length * (1.0 - head_frac)
    d_norm    = d / length

    ref  = np.array([1., 0., 0.]) if abs(d_norm[0]) < 0.9 else np.array([0., 1., 0.])
    perp = np.cross(d_norm, ref);  perp /= np.linalg.norm(perp)
    tang = np.cross(d_norm, perp)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s   = np.cos(angles), np.sin(angles)

    ring = lambda r, along: (start + d_norm * along)[None] + r * (
        c[:, None] * perp + s[:, None] * tang)

    shaft_bot = ring(shaft_r, 0.0)
    shaft_top = ring(shaft_r, shaft_len)
    head_base = ring(head_r,  shaft_len)
    head_tip  = end[None]
    ctr_bot   = start[None]
    ctr_stop  = (start + d_norm * shaft_len)[None]

    verts = np.vstack([shaft_bot, shaft_top, head_base,
                       head_tip, ctr_bot, ctr_stop]).astype(np.float32)
    tip_i  = 3 * n
    cbot_i = 3 * n + 1
    csto_i = 3 * n + 2

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([cbot_i, j,      i     ])
        faces.append([i,      n + i,  j     ])
        faces.append([j,      n + i,  n + j ])
        faces.append([csto_i, n + i,  n + j ])
        faces.append([2*n+i,  tip_i,  2*n+j ])

    return verts, np.array(faces, dtype=np.uint32)


# ── Observation builder ───────────────────────────────────────────────────────

def build_obs(
    q: np.ndarray,
    dq: np.ndarray,
    ee_pos: np.ndarray,
    ee_rot: np.ndarray,
    gripper_driver_pos: float,
    cube_pos_world: np.ndarray,
    goal_pos_world: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """Build 33-D observation matching the pick_cube_osc training checkpoint.

    Layout (33D):
        joint_vel(7)      — arm joint velocities [rad/s]
        ee_pose(6)        — [ee_pos_local(3), ee_axis_angle(3)]
        gripper_state(1)  — right_driver_joint / 0.8
        ee_to_cube(3)     — cube_pos_world - ee_pos_world
        cube_pos(3)       — cube_pos_world (local frame = world for single env)
        cube_to_goal(3)   — goal_pos_world - cube_pos_world
        goal_pos(3)       — goal_pos_world (local frame)
        last_action(7)

    Args:
        q                 : (7,) arm joint positions [rad]  (used for FK only, not in obs)
        dq                : (7,) arm joint velocities [rad/s]
        ee_pos            : (3,) EE world position [m]
        ee_rot            : (3,3) EE world rotation matrix
        gripper_driver_pos: right_driver_joint position (0=open, 0.8=closed)
        cube_pos_world    : (3,) cube center world position [m]
        goal_pos_world    : (3,) goal world position [m]
        last_action       : (7,) previous policy action
    """
    # ee_pose: local frame = world frame for single env (env_origin = 0)
    ee_axis_angle = Rotation.from_matrix(ee_rot).as_rotvec()

    joint_vel_obs    = dq.astype(np.float32)                          # (7,)
    ee_pose_obs      = np.concatenate([ee_pos, ee_axis_angle])        # (6,)
    gripper_obs      = np.array([gripper_driver_pos / _GRIPPER_DRIVER_MAX], dtype=np.float32)  # (1,)
    ee_to_cube_obs   = cube_pos_world - ee_pos                        # (3,)
    cube_pos_obs     = cube_pos_world.copy()                          # (3,) local = world
    cube_to_goal_obs = goal_pos_world - cube_pos_world                # (3,)
    goal_pos_obs     = goal_pos_world.copy()                          # (3,) local = world

    return np.concatenate([
        joint_vel_obs,    # 7
        ee_pose_obs,      # 6
        gripper_obs,      # 1
        ee_to_cube_obs,   # 3
        cube_pos_obs,     # 3
        cube_to_goal_obs, # 3
        goal_pos_obs,     # 3
        last_action,      # 7
    ]).astype(np.float32)  # total: 33


def _gripper_action_to_ctrl(action_1d: float) -> float:
    """Map policy gripper action [-1, +1] → fingers_actuator ctrl [0, 255]."""
    ctrl_mid  = (GRIPPER_CTRL_MIN + GRIPPER_CTRL_MAX) * 0.5
    ctrl_half = (GRIPPER_CTRL_MAX - GRIPPER_CTRL_MIN) * 0.5
    return float(ctrl_mid + np.clip(action_1d, -1.0, 1.0) * ctrl_half)


# ── Policy process ────────────────────────────────────────────────────────────

def policy_process_fn(checkpoint_path, device_str, home_rad_arr,
                      shm_q_sim,
                      shm_dq_sim,
                      shm_q_real,
                      shm_dq_real,
                      shm_osc_target_sim, shm_osc_target_real,
                      shm_action_sim, shm_action_real,
                      shm_ws_lo, shm_ws_hi,
                      shm_cube_pos_sim,
                      shm_cube_pos_real,
                      shm_goal_pos,
                      shm_gripper_driver_sim,
                      shm_gripper_driver_real,
                      shm_gripper_ctrl_sim,
                      shm_gripper_ctrl_real,
                      shm_policy_hz,
                      policy_reset_event, reset_in_progress, stop_event):
    """Single policy forward pass with batch=2 (sim + real) at TARGET_HZ."""
    policy_agent     = PolicyAgent(checkpoint_path, device=device_str)
    if policy_agent.obs_dim != 33:
        raise ValueError(
            f"Checkpoint obs_dim={policy_agent.obs_dim} but pick_cube_osc expects 33. "
            f"Did you pass the wrong checkpoint?"
        )
    if policy_agent.action_dim != 7:
        raise ValueError(
            f"Checkpoint action_dim={policy_agent.action_dim} but pick_cube_osc expects 7."
        )
    robot            = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    home_rad         = home_rad_arr.copy()
    last_action_sim  = np.zeros(7, dtype=np.float32)
    last_action_real = np.zeros(7, dtype=np.float32)
    period           = 1.0 / TARGET_HZ
    iters            = 0
    t_rate           = time.time()

    while not stop_event.is_set():
        t0 = time.time()

        if reset_in_progress.is_set():
            _np(shm_action_sim)[:]  = 0.0
            _np(shm_action_real)[:] = 0.0
            last_action_sim[:]  = 0.0
            last_action_real[:] = 0.0
            time.sleep(0.05)
            continue

        if policy_reset_event.is_set():
            policy_reset_event.clear()
            last_action_sim[:]  = 0.0
            last_action_real[:] = 0.0
            iters  = 0
            t_rate = time.time()

        ws_lo = _np(shm_ws_lo).copy()
        ws_hi = _np(shm_ws_hi).copy()

        # ── Read cube and goal positions ───────────────────────────────────────
        cube_pos_sim  = _np(shm_cube_pos_sim).copy()
        cube_pos_real = _np(shm_cube_pos_real).copy()
        goal_pos      = _np(shm_goal_pos).copy()

        # ── Read gripper driver joint positions ────────────────────────────────
        gripper_driver_sim  = float(_np(shm_gripper_driver_sim)[0])
        gripper_driver_real = float(_np(shm_gripper_driver_real)[0])

        # ── Read shared joint state ────────────────────────────────────────────
        q_sim   = _np(shm_q_sim).copy()
        dq_sim  = _np(shm_dq_sim).copy()
        q_real  = _np(shm_q_real).copy()
        dq_real = _np(shm_dq_real).copy()

        # ── FK ─────────────────────────────────────────────────────────────────
        ee_pos_sim,  ee_rot_sim  = robot.fk(q_sim)
        ee_pos_real, ee_rot_real = robot.fk(q_real)

        # ── Build batch observations ───────────────────────────────────────────
        obs_sim  = build_obs(q_sim,  dq_sim,  ee_pos_sim,  ee_rot_sim,
                             gripper_driver_sim,  cube_pos_sim,  goal_pos, last_action_sim)
        obs_real = build_obs(q_real, dq_real, ee_pos_real, ee_rot_real,
                             gripper_driver_real, cube_pos_real, goal_pos, last_action_real)

        obs_batch    = np.stack([obs_sim, obs_real], axis=0)  # (2, 33)
        action_batch = policy_agent.get_action(obs_batch)     # (2, 7)
        action_sim   = action_batch[0]
        action_real  = action_batch[1]

        last_action_sim[:]  = action_sim
        last_action_real[:] = action_real

        # ── Sim: action[:6] → OSC target ──────────────────────────────────────
        osc_tgt_pos_sim = np.clip(ee_pos_sim + action_sim[:3] * DELTA_POS_SCALE,
                                  ws_lo, ws_hi)
        _np(shm_action_sim)[:3] = action_sim[:3] * DELTA_POS_SCALE   # raw delta
        _np(shm_action_sim)[3:6] = osc_tgt_pos_sim - ee_pos_sim      # effective delta
        _np(shm_action_sim)[6]  = action_sim[6]                       # gripper raw

        delta_rot_sim    = Rotation.from_rotvec(action_sim[3:6] * DELTA_ORI_SCALE)
        osc_tgt_quat_sim = (delta_rot_sim * Rotation.from_matrix(ee_rot_sim)).as_quat()
        _np(shm_osc_target_sim)[:3] = osc_tgt_pos_sim
        _np(shm_osc_target_sim)[3:] = osc_tgt_quat_sim

        # Gripper ctrl for sim
        _np(shm_gripper_ctrl_sim)[0] = _gripper_action_to_ctrl(action_sim[6])

        # ── Real: action[:6] → OSC target ─────────────────────────────────────
        osc_tgt_pos_real = np.clip(ee_pos_real + action_real[:3] * DELTA_POS_SCALE,
                                   ws_lo, ws_hi)
        _np(shm_action_real)[:3] = action_real[:3] * DELTA_POS_SCALE  # raw delta
        _np(shm_action_real)[3:6] = osc_tgt_pos_real - ee_pos_real    # effective delta
        _np(shm_action_real)[6]  = action_real[6]                      # gripper raw

        delta_rot_real    = Rotation.from_rotvec(action_real[3:6] * DELTA_ORI_SCALE)
        osc_tgt_quat_real = (delta_rot_real * Rotation.from_matrix(ee_rot_real)).as_quat()
        _np(shm_osc_target_real)[:3] = osc_tgt_pos_real
        _np(shm_osc_target_real)[3:] = osc_tgt_quat_real

        # Gripper ctrl for real: hardware send_torques expects [0.0=open, 1.0=closed]
        _np(shm_gripper_ctrl_real)[0] = (np.clip(action_real[6], -1.0, 1.0) + 1.0) * 0.5

        iters += 1
        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            _np(shm_policy_hz)[0] = iters / dt_rate
            g_ctrl = float(_np(shm_gripper_ctrl_sim)[0])
            print(f"[policy] gripper_action={action_sim[6]:+.3f}  "
                  f"ctrl={g_ctrl:.0f}/255  "
                  f"ee→cube_sim={np.linalg.norm(cube_pos_sim - ee_pos_sim)*100:.1f}cm",
                  flush=True)
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Real process ──────────────────────────────────────────────────────────────

def real_process_fn(ip, home_rad_arr,
                    shm_q_real, shm_dq_real,
                    shm_osc_target_real, shm_gains,
                    shm_gripper_driver_real,
                    shm_gripper_ctrl_real,
                    shm_real_hz,
                    stop_event, reset_event, reset_done_event,
                    ready_event=None):
    """Connects to hardware, runs OSC at 500 Hz, publishes state back."""
    inner_dt       = 1.0 / OSC_HZ
    robot          = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    home_rad       = home_rad_arr.copy()
    posture_target = home_rad.copy()

    hw = KinovaHardware(ip)
    try:
        print("[real] Connecting…")
        hw.connect()
        hw.clear_faults()
        if not hw.wait_until_ready():
            print("[real] Robot not ready")
            return

        hw.set_servoing_mode(low_level=False)
        print("[real] Going to home…")
        hw.go_to_joints(HOME_DEG)
        time.sleep(1.0)
        hw.set_servoing_mode(low_level=True)
        time.sleep(0.5)
        hw.set_torque_mode(True)

        state   = hw.read_state()
        pos_deg = state.positions_deg.copy()
        vel_deg = state.velocities_deg.copy()

        q  = kinova_deg_to_rad(pos_deg)
        dq = np.deg2rad(vel_deg)
        _np(shm_q_real)[:]  = q
        _np(shm_dq_real)[:] = dq

        # Signal to main that initial homing is complete
        if ready_event is not None:
            ready_event.set()

        iters  = 0
        t_rate = time.time()

        while not stop_event.is_set():

            if reset_event.is_set():
                reset_event.clear()
                print("[real] Going home…")
                hw.set_torque_mode(False)
                hw.set_servoing_mode(low_level=False)
                hw.go_to_joints(HOME_DEG)
                time.sleep(0.5)
                hw.set_servoing_mode(low_level=True)
                time.sleep(0.3)
                hw.set_torque_mode(True)
                state   = hw.read_state()
                pos_deg = state.positions_deg.copy()
                vel_deg = state.velocities_deg.copy()
                _np(shm_q_real)[:]  = kinova_deg_to_rad(pos_deg)
                _np(shm_dq_real)[:] = np.deg2rad(vel_deg)
                iters  = 0
                t_rate = time.time()
                reset_done_event.set()
                continue

            target         = _np(shm_osc_target_real).copy()
            gains          = _gains_dict(_np(shm_gains).copy())
            gripper_ctrl   = float(_np(shm_gripper_ctrl_real)[0])

            # Read actual gripper position from hardware once per policy cycle
            # read_gripper_position() returns [0.0=open, 1.0=closed]
            gripper_pos_01 = hw.read_gripper_position()
            _np(shm_gripper_driver_real)[0] = gripper_pos_01 * _GRIPPER_DRIVER_MAX

            for _ in range(OSC_SUBSTEPS):
                t_inner = time.time()

                q  = kinova_deg_to_rad(pos_deg)
                dq = np.deg2rad(vel_deg)

                _np(shm_q_real)[:]  = q
                _np(shm_dq_real)[:] = dq

                tau = compute_osc_torques(
                    robot, target[:3], target[3:], q, dq,
                    gains=gains, posture_target=posture_target,
                )
                tau += TAU_OFFSETS_DEFAULT

                state   = hw.send_torques(tau, pos_deg, gripper_position=gripper_ctrl)
                pos_deg = state.positions_deg.copy()
                vel_deg = state.velocities_deg.copy()

                iters += 1
                elapsed_inner = time.time() - t_inner
                if elapsed_inner < inner_dt:
                    time.sleep(inner_dt - elapsed_inner)

            dt_rate = time.time() - t_rate
            if dt_rate >= 1.0:
                _np(shm_real_hz)[0] = iters / dt_rate
                iters  = 0
                t_rate = time.time()

    finally:
        try:
            if hw.in_torque_mode:
                hw.set_torque_mode(False)
                time.sleep(0.5)
            hw.set_servoing_mode(low_level=False)
            time.sleep(1.0)
            hw.clear_faults()
            if hw.wait_until_ready(timeout=5.0):
                hw.go_to_joints(HOME_DEG)
        except Exception as e:
            print(f"[real] Shutdown warning: {e}")
        hw.disconnect()
        print("[real] Done.")


# ── OSC thread (sim) ──────────────────────────────────────────────────────────

def osc_thread_fn(mj_model, mj_data, robot,
                  shm_q_sim, shm_dq_sim, shm_osc_target_sim, shm_gains,
                  shm_gripper_ctrl_sim,
                  shm_gripper_driver_sim,
                  shm_cube_pos_sim,
                  shm_osc_hz, osc_reset_event, osc_reset_done, stop_event,
                  posture_target, home_qpos,
                  arm_q_idxs, arm_dq_idxs, arm_ctrl_idxs,
                  gripper_q_idxs, gripper_dq_idxs, gripper_ctrl_idx,
                  cube_qpos_adr, cube_dofadr, cube_site_id,
                  cube_spawn_x, cube_spawn_y, cube_spawn_z):
    """Thread in main process: 500 Hz OSC + MuJoCo physics (sim).

    On reset: arm to home, gripper open, cube at random spawn position.
    During episode: gripper ctrl from policy, cube pos published from physics.
    """
    inner_dt = 1.0 / OSC_HZ
    iters    = 0
    t_rate   = time.time()

    while not stop_event.is_set():
        t0 = time.time()

        if osc_reset_event.is_set():
            osc_reset_event.clear()
            mj_data.qpos[arm_q_idxs]      = home_qpos
            mj_data.qvel[arm_dq_idxs]     = 0.0
            mj_data.ctrl[arm_ctrl_idxs]   = 0.0
            # Gripper open
            mj_data.qpos[gripper_q_idxs]  = _GRIPPER_OPEN_QPOS
            mj_data.qvel[gripper_dq_idxs] = 0.0
            mj_data.ctrl[gripper_ctrl_idx] = _GRIPPER_OPEN_CTRL
            # Cube at random spawn position
            cx = np.random.uniform(*cube_spawn_x)
            cy = np.random.uniform(*cube_spawn_y)
            cz = np.random.uniform(*cube_spawn_z)
            mj_data.qpos[cube_qpos_adr:cube_qpos_adr+3] = np.array([cx, cy, cz])
            mj_data.qpos[cube_qpos_adr+3:cube_qpos_adr+7] = np.array([1, 0, 0, 0])  # wxyz identity
            mj_data.qvel[cube_dofadr:cube_dofadr+6] = 0.0
            mujoco.mj_forward(mj_model, mj_data)
            # Publish new cube pos immediately so _on_reset can read it
            _np(shm_cube_pos_sim)[:] = np.array([cx, cy, cz])
            # Reset gripper ctrl to open
            _np(shm_gripper_ctrl_sim)[0] = _GRIPPER_OPEN_CTRL
            iters  = 0
            t_rate = time.time()
            osc_reset_done.set()
            print("[osc] Reset done")

        q_sim  = mj_data.qpos[arm_q_idxs].copy()
        dq_sim = mj_data.qvel[arm_dq_idxs].copy()

        _np(shm_q_sim)[:]  = q_sim
        _np(shm_dq_sim)[:] = dq_sim

        # Publish cube position from physics
        cube_pos = mj_data.qpos[cube_qpos_adr:cube_qpos_adr+3].copy()
        _np(shm_cube_pos_sim)[:] = cube_pos

        # Publish gripper driver joint pos (first gripper joint = right_driver_joint)
        _np(shm_gripper_driver_sim)[0] = mj_data.qpos[gripper_q_idxs[0]]

        tgt     = _np(shm_osc_target_sim).copy()
        gains   = _gains_dict(_np(shm_gains).copy())
        tau_sim = compute_osc_torques(
            robot, tgt[:3], tgt[3:], q_sim, dq_sim,
            gains=gains, posture_target=posture_target,
        )

        gripper_ctrl = float(_np(shm_gripper_ctrl_sim)[0])
        mj_data.ctrl[arm_ctrl_idxs]   = tau_sim
        mj_data.ctrl[gripper_ctrl_idx] = gripper_ctrl
        mujoco.mj_step(mj_model, mj_data)

        iters += 1
        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            _np(shm_osc_hz)[0] = iters / dt_rate
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - t0
        if elapsed < inner_dt:
            time.sleep(inner_dt - elapsed)


# ── Viz thread ────────────────────────────────────────────────────────────────

def viz_thread_fn(sim_view, real_view, mj_model, mj_data_phys, mj_data_sim, mj_data_real,
                  arm_q_idxs, gripper_q_idxs, cube_qpos_adr,
                  shm_q_sim, shm_q_real, shm_gripper_driver_real, shm_viz_hz, stop_event):
    period = 1.0 / VIZ_HZ
    iters  = 0
    t_rate = time.time()

    while not stop_event.is_set():
        t = time.time()

        # Sim: copy arm + gripper joints + cube qpos from physics
        mj_data_sim.qpos[arm_q_idxs]                        = _np(shm_q_sim)
        mj_data_sim.qpos[gripper_q_idxs]                    = mj_data_phys.qpos[gripper_q_idxs]
        mj_data_sim.qpos[cube_qpos_adr:cube_qpos_adr+7]     = mj_data_phys.qpos[cube_qpos_adr:cube_qpos_adr+7]
        # Real ghost: arm joints + gripper scaled from actual driver position
        mj_data_real.qpos[arm_q_idxs] = _np(shm_q_real)
        real_driver = float(_np(shm_gripper_driver_real)[0])  # [0, 0.8]
        sim_driver  = float(mj_data_phys.qpos[gripper_q_idxs[0]])
        if sim_driver > 1e-4:
            scale = real_driver / sim_driver
            mj_data_real.qpos[gripper_q_idxs] = mj_data_phys.qpos[gripper_q_idxs] * scale
        else:
            # sim gripper open — set real proportionally from driver pos directly
            mj_data_real.qpos[gripper_q_idxs] = mj_data_phys.qpos[gripper_q_idxs] * (
                real_driver / _GRIPPER_DRIVER_MAX)

        mujoco.mj_kinematics(mj_model, mj_data_sim)
        mujoco.mj_kinematics(mj_model, mj_data_real)
        sim_view.update(mj_data_sim)
        real_view.update(mj_data_real)

        iters += 1
        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            _np(shm_viz_hz)[0] = iters / dt_rate
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - t
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/pick_cube/model_998.pt",
                        help="Path to RSL-RL PPO checkpoint (.pt)")
    parser.add_argument("--ip",       default="192.168.1.10",
                        help="Kinova robot IP address")
    parser.add_argument("--sim-only", action="store_true",
                        help="Skip real robot (sim only, no hardware required)")
    args = parser.parse_args()

    policy_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── Compute home FK before any CUDA ───────────────────────────────────────
    print("Loading Pinocchio…")
    robot_main = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    home_rad   = kinova_deg_to_rad(HOME_DEG)
    ee_pos0, ee_rot0 = robot_main.fk(home_rad)
    q_xyzw0          = Rotation.from_matrix(ee_rot0).as_quat()  # xyzw

    # Workspace bounds centred on home EE pos
    ws_lo = ee_pos0 - _WS_RADIUS
    ws_hi = ee_pos0 + _WS_RADIUS
    print(f"  home EE: {ee_pos0}  ws [{ws_lo} … {ws_hi}]")
    print("  OK")

    # ── Initial cube and goal positions ───────────────────────────────────────
    # Cube: random spawn (will be replaced by perception on real)
    cube_init = np.array([
        np.random.uniform(*CUBE_SPAWN_X),
        np.random.uniform(*CUBE_SPAWN_Y),
        np.random.uniform(*CUBE_SPAWN_Z),
    ])
    goal_init = np.array([
        np.random.uniform(GOAL_LO[0], GOAL_HI[0]),
        np.random.uniform(GOAL_LO[1], GOAL_HI[1]),
        np.random.uniform(GOAL_LO[2], GOAL_HI[2]),
    ])
    print(f"  cube init: {cube_init}")
    print(f"  goal init: {goal_init}")

    # ── Shared memory ──────────────────────────────────────────────────────────
    shm_gains               = mp.Array(ctypes.c_double, len(GAINS_KEYS))
    shm_q_sim               = mp.Array(ctypes.c_double, 7)
    shm_dq_sim              = mp.Array(ctypes.c_double, 7)
    shm_osc_target_sim      = mp.Array(ctypes.c_double, 7)   # pos(3) + quat_xyzw(4)
    shm_action_sim          = mp.Array(ctypes.c_double, 7)   # [:3] raw pos delta, [3:6] eff delta, [6] gripper
    shm_q_real              = mp.Array(ctypes.c_double, 7)
    shm_dq_real             = mp.Array(ctypes.c_double, 7)
    shm_osc_target_real     = mp.Array(ctypes.c_double, 7)
    shm_action_real         = mp.Array(ctypes.c_double, 7)
    shm_ws_lo               = mp.Array(ctypes.c_double, 3)
    shm_ws_hi               = mp.Array(ctypes.c_double, 3)
    shm_cube_pos_sim        = mp.Array(ctypes.c_double, 3)   # from MuJoCo physics
    shm_cube_pos_real       = mp.Array(ctypes.c_double, 3)   # randomised / from perception
    shm_goal_pos            = mp.Array(ctypes.c_double, 3)   # shared goal for both
    shm_gripper_driver_sim  = mp.Array(ctypes.c_double, 1)   # right_driver_joint sim
    shm_gripper_driver_real = mp.Array(ctypes.c_double, 1)   # right_driver_joint real (approx)
    shm_gripper_ctrl_sim    = mp.Array(ctypes.c_double, 1)   # fingers_actuator ctrl sim
    shm_gripper_ctrl_real   = mp.Array(ctypes.c_double, 1)   # fingers_actuator ctrl real
    shm_osc_hz              = mp.Array(ctypes.c_double, 1)
    shm_policy_hz           = mp.Array(ctypes.c_double, 1)
    shm_viz_hz              = mp.Array(ctypes.c_double, 1)
    shm_real_hz             = mp.Array(ctypes.c_double, 1)

    _np(shm_gains)[:] = _pack_gains(KP_POS, KD_POS, KP_ORI, KD_ORI,
                                     POSTURE_KP, POSTURE_KD, POSTURE_WEIGHT)
    _np(shm_q_sim)[:]           = home_rad
    _np(shm_dq_sim)[:]          = 0.0
    _np(shm_q_real)[:]          = home_rad
    _np(shm_dq_real)[:]         = 0.0
    _np(shm_osc_target_sim)[:3] = ee_pos0
    _np(shm_osc_target_sim)[3:] = q_xyzw0
    _np(shm_osc_target_real)[:3] = ee_pos0
    _np(shm_osc_target_real)[3:] = q_xyzw0
    _np(shm_ws_lo)[:] = ws_lo
    _np(shm_ws_hi)[:] = ws_hi
    _np(shm_cube_pos_sim)[:]    = cube_init
    _np(shm_cube_pos_real)[:]   = cube_init
    _np(shm_goal_pos)[:]        = goal_init
    _np(shm_gripper_driver_sim)[0]  = 0.0   # start open
    _np(shm_gripper_driver_real)[0] = 0.0
    _np(shm_gripper_ctrl_sim)[0]    = GRIPPER_CTRL_MIN
    _np(shm_gripper_ctrl_real)[0]   = GRIPPER_CTRL_MIN

    stop_event         = mp.Event()
    osc_reset_event    = mp.Event()
    osc_reset_done     = threading.Event()
    policy_reset_event = mp.Event()
    reset_in_progress  = mp.Event()
    real_reset_event   = mp.Event()
    real_reset_done    = mp.Event()
    real_ready_event   = mp.Event()

    # Hold policy paused until startup reset completes
    reset_in_progress.set()

    # ── Spawn real process BEFORE any CUDA init ────────────────────────────────
    real_proc = None
    if not args.sim_only:
        real_proc = mp.Process(
            target=real_process_fn,
            args=(args.ip, home_rad,
                  shm_q_real, shm_dq_real,
                  shm_osc_target_real, shm_gains,
                  shm_gripper_driver_real,
                  shm_gripper_ctrl_real,
                  shm_real_hz,
                  stop_event, real_reset_event, real_reset_done,
                  real_ready_event),
            daemon=True,
        )
        real_proc.start()
        print(f"[main] Real process PID: {real_proc.pid}")

    # ── Spawn policy process BEFORE any CUDA init ──────────────────────────────
    policy_proc = mp.Process(
        target=policy_process_fn,
        args=(args.checkpoint, policy_device, home_rad,
              shm_q_sim,
              shm_dq_sim,
              shm_q_real,
              shm_dq_real,
              shm_osc_target_sim, shm_osc_target_real,
              shm_action_sim, shm_action_real,
              shm_ws_lo, shm_ws_hi,
              shm_cube_pos_sim,
              shm_cube_pos_real,
              shm_goal_pos,
              shm_gripper_driver_sim,
              shm_gripper_driver_real,
              shm_gripper_ctrl_sim,
              shm_gripper_ctrl_real,
              shm_policy_hz,
              policy_reset_event, reset_in_progress, stop_event),
        daemon=True,
    )
    policy_proc.start()
    print(f"[main] Policy process PID: {policy_proc.pid}")

    # ── MuJoCo sim ─────────────────────────────────────────────────────────────
    def _compile_model():
        spec = mujoco.MjSpec.from_file(str(_TORQUE_XML))
        spec.assets = get_assets(spec.meshdir)
        # Add a floor plane so the cube has something to rest on
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = np.array([0.0, 0.0, 0.01])   # infinite plane (size[0,1]=0)
        floor.pos  = np.array([0.0, 0.0, 0.0])
        floor.rgba = np.array([0.8, 0.8, 0.8, 1.0])
        floor.contype  = 1
        floor.conaffinity = 1
        # Attach cube (freejoint body)
        cube_spec = mujoco.MjSpec.from_file(str(CUBE_XML))
        spec.attach(cube_spec, prefix="cube/", suffix="", frame=spec.worldbody.add_frame())
        return spec.compile()

    mj_model     = _compile_model()
    mj_data_phys = mujoco.MjData(mj_model)
    mj_data_sim  = mujoco.MjData(mj_model)   # viz thread (sim ghost)
    mj_data_real = mujoco.MjData(mj_model)   # viz thread (real ghost)

    arm_q_idxs    = np.array([mj_model.joint(n).qposadr.item() for n in _ARM_JOINT_NAMES])
    arm_dq_idxs   = np.array([mj_model.joint(n).dofadr.item()  for n in _ARM_JOINT_NAMES])
    arm_ctrl_idxs = np.array([mj_model.actuator(n).id          for n in _ARM_JOINT_NAMES])

    gripper_q_idxs   = np.array([mj_model.joint(n).qposadr.item() for n in _GRIPPER_JOINT_NAMES])
    gripper_dq_idxs  = np.array([mj_model.joint(n).dofadr.item()  for n in _GRIPPER_JOINT_NAMES])
    gripper_ctrl_idx = mj_model.actuator("fingers_actuator").id

    cube_jnt      = mj_model.joint("cube/cube_joint")
    cube_qpos_adr = cube_jnt.qposadr.item()   # freejoint: 7 qpos entries
    cube_dofadr   = cube_jnt.dofadr.item()    # freejoint: 6 dof entries
    cube_site_id  = mj_model.site("cube/object_site").id

    # Initial state: arm at home, gripper open, cube at random spawn
    for d_ in (mj_data_phys, mj_data_sim, mj_data_real):
        d_.qpos[arm_q_idxs]      = home_rad
        d_.qpos[gripper_q_idxs]  = _GRIPPER_OPEN_QPOS
        d_.ctrl[gripper_ctrl_idx] = _GRIPPER_OPEN_CTRL
        d_.qpos[cube_qpos_adr:cube_qpos_adr+3] = cube_init
        d_.qpos[cube_qpos_adr+3:cube_qpos_adr+7] = np.array([1, 0, 0, 0])  # wxyz

    mujoco.mj_forward(mj_model, mj_data_phys)
    mujoco.mj_kinematics(mj_model, mj_data_sim)
    mujoco.mj_kinematics(mj_model, mj_data_real)

    robot_osc = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")

    # ── Viser ──────────────────────────────────────────────────────────────────
    server    = viser.ViserServer(label="Kinova Pick Cube Sim2Real")
    scene     = ViserMujocoScene.create(server, mj_model)
    sim_view  = scene.add_robot("sim",  color=(0.75, 0.75, 0.75, 1.00))
    real_view = scene.add_robot("real", color=(0.20, 0.55, 0.90, 0.65))

    # In sim-only mode hide the entire real ghost; otherwise hide just the cube
    _cube_body_id = mj_model.body("cube/cube").id
    for (body_id, _g, _s), handle in real_view._handles.items():
        if args.sim_only or body_id == _cube_body_id:
            handle.visible = False
    scene.create_visualization_gui(camera_distance=1.2, camera_azimuth=135.0,
                                   camera_elevation=30.0)

    # EE frames (sim + real)
    def _add_ee_frame(name):
        return server.scene.add_frame(
            f"/{name}_ee_frame",
            position=tuple(float(v) for v in ee_pos0),
            wxyz=(float(q_xyzw0[3]), float(q_xyzw0[0]),
                  float(q_xyzw0[1]), float(q_xyzw0[2])),
            axes_length=0.08, axes_radius=0.003,
        )
    frame_ee_sim  = _add_ee_frame("sim")
    frame_ee_real = _add_ee_frame("real")
    if args.sim_only:
        frame_ee_real.visible = False

    # Goal marker sphere
    goal_marker = server.scene.add_icosphere(
        "/goal_marker", radius=0.04,
        position=tuple(float(v) for v in goal_init),
        color=(1.0, 0.5, 0.0),
    )

    # Real cube marker (randomised position, not from physics) — hidden in sim-only
    real_cube_marker = server.scene.add_box(
        "/real_cube_marker",
        dimensions=(0.04, 0.04, 0.04),
        position=tuple(float(v) for v in cube_init),
        color=(0.2, 0.5, 1.0),
    )
    if args.sim_only:
        real_cube_marker.visible = False

    # Action arrows: sim (orange=raw, yellow=eff), real (red=raw, green=eff)
    _av0, _af0 = _make_arrow_mesh(ee_pos0, ee_pos0 + np.array([0.02, 0., 0.]))
    raw_arrow_sim  = server.scene.add_mesh_simple(
        "/sim_raw_action",  vertices=_av0,        faces=_af0,
        color=(1.0, 0.45, 0.0), side="double")
    eff_arrow_sim  = server.scene.add_mesh_simple(
        "/sim_eff_action",  vertices=_av0.copy(),  faces=_af0,
        color=(1.0, 0.85, 0.0), side="double")
    raw_arrow_real = server.scene.add_mesh_simple(
        "/real_raw_action", vertices=_av0.copy(),  faces=_af0,
        color=(1.0, 0.20, 0.2), side="double")
    eff_arrow_real = server.scene.add_mesh_simple(
        "/real_eff_action", vertices=_av0.copy(),  faces=_af0,
        color=(0.20, 0.90, 0.3), side="double")
    if args.sim_only:
        raw_arrow_real.visible = False
        eff_arrow_real.visible = False

    # Workspace bounding box
    _lo, _hi = ws_lo, ws_hi
    _corners = np.array([
        [_lo[0], _lo[1], _lo[2]], [_hi[0], _lo[1], _lo[2]],
        [_lo[0], _hi[1], _lo[2]], [_hi[0], _hi[1], _lo[2]],
        [_lo[0], _lo[1], _hi[2]], [_hi[0], _lo[1], _hi[2]],
        [_lo[0], _hi[1], _hi[2]], [_hi[0], _hi[1], _hi[2]],
    ], dtype=np.float32)
    _edges = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    ws_bbox = server.scene.add_line_segments(
        "/workspace_bounds",
        points=np.array([[_corners[a], _corners[b]] for a, b in _edges],
                        dtype=np.float32),
        colors=np.array([0.8, 0.8, 0.2], dtype=np.float32),
        line_width=1.5,
    )

    # Cube spawn bounding box (blue)
    def _make_bbox_lines(lo: np.ndarray, hi: np.ndarray, path: str,
                         color: tuple, line_width: float = 1.5):
        corners = np.array([
            [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
            [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
            [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]],
        ], dtype=np.float32)
        edges = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
        return server.scene.add_line_segments(
            path,
            points=np.array([[corners[a], corners[b]] for a, b in edges], dtype=np.float32),
            colors=np.array(color, dtype=np.float32),
            line_width=line_width,
        )

    cube_spawn_lo = np.array([CUBE_SPAWN_X[0], CUBE_SPAWN_Y[0], CUBE_SPAWN_Z[0]])
    cube_spawn_hi = np.array([CUBE_SPAWN_X[1], CUBE_SPAWN_Y[1], CUBE_SPAWN_Z[1]])
    cube_spawn_bbox = _make_bbox_lines(cube_spawn_lo, cube_spawn_hi,
                                       "/cube_spawn_bounds",
                                       color=(0.2, 0.5, 1.0), line_width=1.5)

    # Goal bounding box (orange)
    goal_bbox = _make_bbox_lines(GOAL_LO, GOAL_HI,
                                 "/goal_bounds",
                                 color=(1.0, 0.5, 0.0), line_width=1.5)

    # ── GUI ────────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Visualizations"):
        cb_all            = server.gui.add_checkbox("Show all",              initial_value=True)
        cb_bbox           = server.gui.add_checkbox("Workspace bbox",        initial_value=True)
        cb_cube_spawn_box = server.gui.add_checkbox("Cube spawn box",        initial_value=True)
        cb_goal_box       = server.gui.add_checkbox("Goal box",              initial_value=True)
        cb_ee_frames      = server.gui.add_checkbox("EE frames",             initial_value=True)
        cb_goal           = server.gui.add_checkbox("Goal marker",           initial_value=True)
        cb_real_cube      = server.gui.add_checkbox("Real cube marker",      initial_value=True)
        cb_sim_raw        = server.gui.add_checkbox("Sim raw action arrow",  initial_value=True)
        cb_sim_eff        = server.gui.add_checkbox("Sim eff action arrow",  initial_value=True)
        cb_real_raw       = server.gui.add_checkbox("Real raw action arrow", initial_value=True)
        cb_real_eff       = server.gui.add_checkbox("Real eff action arrow", initial_value=True)

    _all_vis = [
        (cb_bbox,           ws_bbox),
        (cb_cube_spawn_box, cube_spawn_bbox),
        (cb_goal_box,       goal_bbox),
        (cb_goal,           goal_marker),
        (cb_real_cube,      real_cube_marker),
        (cb_sim_raw,        raw_arrow_sim),
        (cb_sim_eff,        eff_arrow_sim),
        (cb_real_raw,       raw_arrow_real),
        (cb_real_eff,       eff_arrow_real),
    ]

    def _on_show_all(_):
        v = cb_all.value
        for cb, obj in _all_vis:
            cb.value    = v
            obj.visible = v
        cb_ee_frames.value    = v
        frame_ee_sim.visible  = v
        frame_ee_real.visible = v

    cb_all.on_update(_on_show_all)
    cb_bbox.on_update(          lambda _: setattr(ws_bbox,         "visible", cb_bbox.value))
    cb_cube_spawn_box.on_update(lambda _: setattr(cube_spawn_bbox, "visible", cb_cube_spawn_box.value))
    cb_goal_box.on_update(      lambda _: setattr(goal_bbox,       "visible", cb_goal_box.value))
    cb_goal.on_update(          lambda _: setattr(goal_marker,     "visible", cb_goal.value))
    cb_real_cube.on_update(     lambda _: setattr(real_cube_marker,"visible", cb_real_cube.value))
    cb_ee_frames.on_update(lambda _: [setattr(o, "visible", cb_ee_frames.value)
                                      for o in (frame_ee_sim, frame_ee_real)])
    cb_sim_raw.on_update(  lambda _: setattr(raw_arrow_sim,   "visible", cb_sim_raw.value))
    cb_sim_eff.on_update(  lambda _: setattr(eff_arrow_sim,   "visible", cb_sim_eff.value))
    cb_real_raw.on_update( lambda _: setattr(raw_arrow_real,  "visible", cb_real_raw.value))
    cb_real_eff.on_update( lambda _: setattr(eff_arrow_real,  "visible", cb_real_eff.value))

    with server.gui.add_folder("Policy"):
        txt_policy_hz  = server.gui.add_text(
            f"Policy rate (target {TARGET_HZ} Hz)", initial_value="— Hz")
        txt_osc_hz     = server.gui.add_text(
            f"Sim OSC rate (target {OSC_HZ} Hz)",   initial_value="— Hz")
        txt_real_hz    = server.gui.add_text(
            f"Real rate (target {OSC_HZ} Hz)",      initial_value="— Hz")
        txt_viz_hz     = server.gui.add_text(
            f"Viz rate (target {VIZ_HZ} Hz)",       initial_value="— Hz")
        txt_ee_sim     = server.gui.add_text("Sim EE XYZ",           initial_value="—")
        txt_ee_real    = server.gui.add_text("Real EE XYZ",          initial_value="—")
        txt_cube_sim   = server.gui.add_text("Sim cube pos",         initial_value="—")
        txt_cube_real  = server.gui.add_text("Real cube pos",        initial_value="—")
        txt_goal       = server.gui.add_text("Goal pos",             initial_value="—")
        txt_ee_cube    = server.gui.add_text("Sim EE→cube dist",     initial_value="—")
        txt_cube_goal  = server.gui.add_text("Sim cube→goal dist",   initial_value="—")
        txt_gripper_s  = server.gui.add_text("Gripper ctrl sim",     initial_value="—")
        txt_gripper_r  = server.gui.add_text("Gripper ctrl real",    initial_value="—")
        reset_btn      = server.gui.add_button("Reset")

    with server.gui.add_folder("OSC Gains"):
        sl_kp_pos = server.gui.add_slider("Kp pos", min=0.0, max=1000.0, step=1.0,
                                          initial_value=KP_POS)
        sl_kd_pos = server.gui.add_slider("Kd pos", min=0.0, max=200.0,  step=0.5,
                                          initial_value=KD_POS)
        sl_kp_ori = server.gui.add_slider("Kp ori", min=0.0, max=1000.0, step=1.0,
                                          initial_value=KP_ORI)
        sl_kd_ori = server.gui.add_slider("Kd ori", min=0.0, max=200.0,  step=0.5,
                                          initial_value=KD_ORI)

    with server.gui.add_folder("Posture"):
        sl_post_kp = server.gui.add_slider("Posture Kp",     min=0.0, max=100.0,
                                           step=0.1,  initial_value=POSTURE_KP)
        sl_post_kd = server.gui.add_slider("Posture Kd",     min=0.0, max=20.0,
                                           step=0.1,  initial_value=POSTURE_KD)
        sl_post_w  = server.gui.add_slider("Posture weight", min=0.0, max=1.0,
                                           step=0.01, initial_value=POSTURE_WEIGHT)

    with server.gui.add_folder("Joint Angles (deg)"):
        txt_joints = [
            server.gui.add_text(f"J{i+1}  real | sim | err", initial_value="— | — | —")
            for i in range(7)
        ]

    def _on_reset(_):
        reset_in_progress.set()
        osc_reset_done.clear()
        real_reset_done.clear()

        # Sample new goal position
        new_goal = np.array([
            np.random.uniform(GOAL_LO[0], GOAL_HI[0]),
            np.random.uniform(GOAL_LO[1], GOAL_HI[1]),
            np.random.uniform(GOAL_LO[2], GOAL_HI[2]),
        ])
        _np(shm_goal_pos)[:] = new_goal
        goal_marker.position = tuple(float(v) for v in new_goal)
        print(f"[reset] goal pos: {new_goal}")

        # Trigger all resets (sim OSC thread samples cube position internally)
        osc_reset_event.set()
        policy_reset_event.set()
        _np(shm_osc_target_sim)[:3]  = ee_pos0
        _np(shm_osc_target_sim)[3:]  = q_xyzw0
        _np(shm_osc_target_real)[:3] = ee_pos0
        _np(shm_osc_target_real)[3:] = q_xyzw0

        # Wait for sim OSC reset — cube_pos_sim written by OSC thread before done
        osc_reset_done.wait()

        # Real cube pos = sim cube pos (will be replaced by perception later)
        sim_cube_pos = _np(shm_cube_pos_sim).copy()
        _np(shm_cube_pos_real)[:] = sim_cube_pos
        real_cube_marker.position = tuple(float(v) for v in sim_cube_pos)
        print(f"[reset] cube pos (sim=real): {sim_cube_pos}")

        # Reset gripper to open on both sim and real
        _np(shm_gripper_ctrl_real)[0] = 0.0   # 0.0 = open for hardware [0,1]
        _np(shm_gripper_driver_sim)[0]  = 0.0
        _np(shm_gripper_driver_real)[0] = 0.0

        if real_proc is not None:
            real_reset_event.set()
            print("[reset] Waiting for real robot to finish homing…")
            real_reset_done.wait()
            print("[reset] Real robot homed.")

        reset_in_progress.clear()
        print("[reset] All done, policy resuming.")

    reset_btn.on_click(_on_reset)

    # ── Start threads ──────────────────────────────────────────────────────────
    threading.Thread(target=osc_thread_fn, daemon=True, args=(
        mj_model, mj_data_phys, robot_osc,
        shm_q_sim, shm_dq_sim, shm_osc_target_sim, shm_gains,
        shm_gripper_ctrl_sim,
        shm_gripper_driver_sim,
        shm_cube_pos_sim,
        shm_osc_hz, osc_reset_event, osc_reset_done, stop_event,
        home_rad, home_rad,
        arm_q_idxs, arm_dq_idxs, arm_ctrl_idxs,
        gripper_q_idxs, gripper_dq_idxs, gripper_ctrl_idx,
        cube_qpos_adr, cube_dofadr, cube_site_id,
        CUBE_SPAWN_X, CUBE_SPAWN_Y, CUBE_SPAWN_Z,
    )).start()

    threading.Thread(target=viz_thread_fn, daemon=True, args=(
        sim_view, real_view, mj_model, mj_data_phys, mj_data_sim, mj_data_real,
        arm_q_idxs, gripper_q_idxs, cube_qpos_adr, shm_q_sim, shm_q_real,
        shm_gripper_driver_real, shm_viz_hz, stop_event,
    )).start()

    # ── Startup reset — wait for real robot to home before starting policy ────
    print("[main] Performing startup reset…")
    osc_reset_done.clear()
    osc_reset_event.set()   # reset sim arm + cube
    osc_reset_done.wait()
    # Sync real cube to sim
    sim_cube_pos = _np(shm_cube_pos_sim).copy()
    _np(shm_cube_pos_real)[:] = sim_cube_pos
    if not args.sim_only:
        print("[main] Waiting for real robot to finish initial homing…")
        real_ready_event.wait()
        print("[main] Real robot ready.")
    reset_in_progress.clear()
    print("[main] Startup reset done — policy running.")

    print("Running. Ctrl+C to stop.")

    def _wxyz(q): return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

    try:
        _print_step = 0
        while True:
            # ── Check policy process is alive ──────────────────────────────────
            if not policy_proc.is_alive():
                print(f"\n[main] ERROR: policy process died (exitcode={policy_proc.exitcode}). "
                      f"Check checkpoint path and obs_dim.")
                break

            # ── Sync gains ─────────────────────────────────────────────────────
            _np(shm_gains)[:] = _pack_gains(
                sl_kp_pos.value, sl_kd_pos.value,
                sl_kp_ori.value, sl_kd_ori.value,
                sl_post_kp.value, sl_post_kd.value, sl_post_w.value,
            )

            # ── FK for EE positions ─────────────────────────────────────────────
            q_sim_cur  = _np(shm_q_sim).copy()
            q_real_cur = _np(shm_q_real).copy()
            ee_pos_sim,  ee_rot_sim  = robot_main.fk(q_sim_cur)
            ee_pos_real, ee_rot_real = robot_main.fk(q_real_cur)
            ee_quat_sim  = Rotation.from_matrix(ee_rot_sim).as_quat()
            ee_quat_real = Rotation.from_matrix(ee_rot_real).as_quat()

            # ── EE frames ──────────────────────────────────────────────────────
            if cb_ee_frames.value:
                frame_ee_sim.position  = tuple(float(v) for v in ee_pos_sim)
                frame_ee_sim.wxyz      = _wxyz(ee_quat_sim)
                frame_ee_real.position = tuple(float(v) for v in ee_pos_real)
                frame_ee_real.wxyz     = _wxyz(ee_quat_real)

            # ── Mirror sim cube pos → real continuously ────────────────────────
            if not reset_in_progress.is_set():
                _np(shm_cube_pos_real)[:] = _np(shm_cube_pos_sim)

            # ── Real cube marker ───────────────────────────────────────────────
            if cb_real_cube.value:
                real_cube_marker.position = tuple(
                    float(v) for v in _np(shm_cube_pos_real))

            # ── Action arrows ──────────────────────────────────────────────────
            raw_delta_sim  = _np(shm_action_sim)[:3].copy()
            eff_delta_sim  = _np(shm_action_sim)[3:6].copy()
            raw_delta_real = _np(shm_action_real)[:3].copy()
            eff_delta_real = _np(shm_action_real)[3:6].copy()
            if cb_sim_raw.value:
                raw_arrow_sim.vertices  = _make_arrow_mesh(
                    ee_pos_sim, ee_pos_sim + raw_delta_sim)[0]
            if cb_sim_eff.value:
                eff_arrow_sim.vertices  = _make_arrow_mesh(
                    ee_pos_sim, ee_pos_sim + eff_delta_sim)[0]
            if cb_real_raw.value:
                raw_arrow_real.vertices = _make_arrow_mesh(
                    ee_pos_real, ee_pos_real + raw_delta_real)[0]
            if cb_real_eff.value:
                eff_arrow_real.vertices = _make_arrow_mesh(
                    ee_pos_real, ee_pos_real + eff_delta_real)[0]

            # ── GUI text ───────────────────────────────────────────────────────
            cube_pos_sim_cur  = _np(shm_cube_pos_sim).copy()
            cube_pos_real_cur = _np(shm_cube_pos_real).copy()
            goal_pos_cur      = _np(shm_goal_pos).copy()
            gripper_ctrl_sim  = float(_np(shm_gripper_ctrl_sim)[0])
            gripper_ctrl_real = float(_np(shm_gripper_ctrl_real)[0])

            txt_policy_hz.value = f"{_np(shm_policy_hz)[0]:.0f} Hz"
            txt_osc_hz.value    = f"{_np(shm_osc_hz)[0]:.0f} Hz"
            txt_real_hz.value   = f"{_np(shm_real_hz)[0]:.0f} Hz"
            txt_viz_hz.value    = f"{_np(shm_viz_hz)[0]:.0f} Hz"
            txt_ee_sim.value  = (f"x={ee_pos_sim[0]:+.3f}  "
                                 f"y={ee_pos_sim[1]:+.3f}  "
                                 f"z={ee_pos_sim[2]:+.3f}")
            txt_ee_real.value = (f"x={ee_pos_real[0]:+.3f}  "
                                 f"y={ee_pos_real[1]:+.3f}  "
                                 f"z={ee_pos_real[2]:+.3f}")
            txt_cube_sim.value  = (f"x={cube_pos_sim_cur[0]:+.3f}  "
                                   f"y={cube_pos_sim_cur[1]:+.3f}  "
                                   f"z={cube_pos_sim_cur[2]:+.3f}")
            txt_cube_real.value = (f"x={cube_pos_real_cur[0]:+.3f}  "
                                   f"y={cube_pos_real_cur[1]:+.3f}  "
                                   f"z={cube_pos_real_cur[2]:+.3f}")
            txt_goal.value = (f"x={goal_pos_cur[0]:+.3f}  "
                              f"y={goal_pos_cur[1]:+.3f}  "
                              f"z={goal_pos_cur[2]:+.3f}")
            txt_ee_cube.value  = (f"{np.linalg.norm(cube_pos_sim_cur - ee_pos_sim)*100:.1f} cm")
            txt_cube_goal.value = (f"{np.linalg.norm(goal_pos_cur - cube_pos_sim_cur)*100:.1f} cm")
            txt_gripper_s.value = f"{gripper_ctrl_sim:.0f} / {GRIPPER_CTRL_MAX:.0f}"
            txt_gripper_r.value = f"{gripper_ctrl_real:.0f} / {GRIPPER_CTRL_MAX:.0f}"

            # Joint angle table
            q_real_deg = np.rad2deg(q_real_cur)
            q_sim_deg  = np.rad2deg(q_sim_cur)
            for i in range(7):
                err = q_real_deg[i] - q_sim_deg[i]
                txt_joints[i].value = (f"{q_real_deg[i]:+6.1f} | "
                                       f"{q_sim_deg[i]:+6.1f} | "
                                       f"{err:+5.1f}")

            _print_step += 1
            if _print_step % 10 == 0:
                print(
                    f"\rsim_ee [{ee_pos_sim[0]:+.3f} {ee_pos_sim[1]:+.3f} "
                    f"{ee_pos_sim[2]:+.3f}]  "
                    f"real_ee [{ee_pos_real[0]:+.3f} {ee_pos_real[1]:+.3f} "
                    f"{ee_pos_real[2]:+.3f}]  "
                    f"cube [{cube_pos_sim_cur[0]:+.3f} {cube_pos_sim_cur[1]:+.3f} "
                    f"{cube_pos_sim_cur[2]:+.3f}]  "
                    f"ee→cube {np.linalg.norm(cube_pos_sim_cur-ee_pos_sim)*100:.1f}cm  "
                    f"policy {_np(shm_policy_hz)[0]:.0f} Hz  "
                    f"osc {_np(shm_osc_hz)[0]:.0f} Hz",
                    end="", flush=True,
                )

            time.sleep(1.0 / TARGET_HZ)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        stop_event.set()
        policy_proc.join(timeout=2.0)
        if real_proc is not None:
            real_proc.join(timeout=5.0)
        server.stop()
        print("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
