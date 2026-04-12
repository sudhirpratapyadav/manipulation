"""Kinova Gen3 sim2real for peg-in-hole task.

An NN policy (RSL-RL PPO checkpoint) drives both sim and real arm simultaneously.
A single policy forward pass with batch=2 produces actions for both at once.
No interactive command target — the policy drives to the fixed hole position.

Policy observations (28-D, same as peg_in_hole_osc training):
    joint_pos_rel(7) + peg_to_hole(6) + ee_to_peg(3) + ee_to_hole(6) + last_action(6)

  joint_pos_rel : q - HOME_JOINT_RAD  (matches mjlab joint_pos_rel with default_joint_pos=home)
  peg_to_hole   : [hole_top  - peg_cylinder_start (3),
                   hole_bottom - peg_cylinder_end  (3)]
  ee_to_peg     : peg_cylinder_start - ee_pos  (fixed offset, peg rigid in gripper)
  ee_to_hole    : [hole_top  - ee_pos (3),
                   hole_bottom - ee_pos (3)]

Policy action (6-D):
    [delta_pos(3), delta_ori_axis_angle(3)]
    scaled by DELTA_POS_SCALE / DELTA_ORI_SCALE before applying to current EE pose.

Processes / Threads:
  - Policy process  (mp.Process): batch=2 inference at TARGET_HZ Hz
  - Real process    (mp.Process): 500 Hz OSC → hardware torques
  - OSC thread      (threading):  500 Hz OSC + MuJoCo physics (sim)
  - Viz thread      (threading):  30 Hz scene update
  - Main thread:                  viser GUI + polling

Run:
  python sim2real_peg_in_hole.py --checkpoint weights/model_499.pt --ip 192.168.1.10
  python sim2real_peg_in_hole.py --checkpoint weights/model_499.pt --sim-only
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
from kinova_tasks.assets.peg_in_hole import HOLE_XML, PEG_XML
from policy import PolicyAgent
from viewer import ViserMujocoScene

# ── Model ─────────────────────────────────────────────────────────────────────
_TORQUE_XML      = KINOVA_GEN3_GRIPPER_TORQUE_XML
_ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]

# Gripper joint names and closed-position values (from peg_in_hole_osc.py)
_GRIPPER_JOINT_NAMES = [
    "right_driver_joint", "right_coupler_joint",
    "right_spring_link_joint", "right_follower_joint",
    "left_driver_joint",  "left_coupler_joint",
    "left_spring_link_joint",  "left_follower_joint",
]
_GRIPPER_CLOSED_QPOS = np.array([0.503, 0.001, 0.505, -0.485,
                                  0.503, 0.001, 0.505, -0.485])
_GRIPPER_CTRL_CLOSED = 191.25   # fingers_actuator ctrl value (60% closed)

# ── Home joint positions — must match peg_in_hole_osc.py _HOME_JOINT_POS ──────
# These are the default_joint_pos used by joint_pos_rel in training.
# [90°, 30°, 0°, 90°, 0°, 60°, -90°] in radians
HOME_DEG = np.array([90.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])

# ── Peg geometry — offsets in peg body frame (rigidly attached to EE) ────────
# Peg is assumed rigidly attached to the gripper and rotates with it.
# World-frame peg site positions:
#   peg_site_world = ee_pos + ee_rot @ EE_TO_PEG_ROT @ offset
#
# EE_TO_PEG_ROT corrects for any rotation between pinch_site frame and peg body
# frame. pinch_site has quat="0 1 0 0" (180° around X) in the XML, so its
# z-axis may be flipped relative to the peg body z-axis.
# Set to identity if peg body frame == pinch_site frame.
# Set to R_x180 = diag(1,-1,-1) if peg body z is opposite to pinch_site z.
#
# From peg.xml: cylinder_start at pos="0 0 -0.02", cylinder_end at "0 0 -0.07"
# relative to peg body origin. Peg hangs downward → offsets are in -z direction.
EE_TO_PEG_ROT = np.diag([1.0, -1.0, -1.0])   # 180° around X: corrects pinch_site flip

PEG_CYLINDER_START_OFFSET = np.array([0.0, 0.0, -0.02])   # in peg body frame [m]
PEG_CYLINDER_END_OFFSET   = np.array([0.0, 0.0, -0.07])   # in peg body frame [m]

# ── Hole position and site offsets ────────────────────────────────────────────
# From reset_hole_position: x=-0.02, y=-0.5, z=0.02 in robot-local frame.
# For a single-environment sim (no env_origins offset) this equals world frame.
# Adjust HOLE_POS_WORLD to match the real-world hole placement.
HOLE_POS_WORLD = np.array([-0.02, -0.5, 0.02])   # world-frame hole body origin

# On each reset, hole is randomized by sampling (dx, dy) uniformly from this range
# and adding to HOLE_POS_WORLD. Z is kept fixed.
HOLE_RAND_X = (-0.05, 0.05)   # metres, relative to HOLE_POS_WORLD
HOLE_RAND_Y = (-0.05, 0.05)   # metres, relative to HOLE_POS_WORLD

# From hole.xml: hole_top at pos="0 0 0.07", hole_bottom at pos="0 0 0.02"
# relative to the hole body — hole body is world-upright (identity quat on reset).
_HOLE_SITE_OFFSETS = {
    "hole_top":    np.array([0.0, 0.0, 0.07]),
    "hole_bottom": np.array([0.0, 0.0, 0.02]),
}
HOLE_TOP_WORLD    = HOLE_POS_WORLD + _HOLE_SITE_OFFSETS["hole_top"]
HOLE_BOTTOM_WORLD = HOLE_POS_WORLD + _HOLE_SITE_OFFSETS["hole_bottom"]

# ── OSC gains — must match peg_in_hole_osc.py ─────────────────────────────────
KP_POS         =  50.0
KD_POS         =  10.0
KP_ORI         =  50.0
KD_ORI         =  10.0
POSTURE_KP     =  10.0
POSTURE_KD     =   2.0
POSTURE_WEIGHT =   0.0   # zero in training env

MAX_JOINT_TORQUE    = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])
TAU_OFFSETS_DEFAULT = np.array([0.0,  0.0,  -0.5, 0.0,  0.0, 1.0, 0.0])

# ── Action scales — must match peg_in_hole_osc.py ─────────────────────────────
DELTA_POS_SCALE = 0.01   # metres per unit action
DELTA_ORI_SCALE = 0.02   # rad per unit action

# ── Workspace bounds for OSC target clipping ──────────────────────────────────
# Centred on home EE position (computed at startup from FK), ±WS_RADIUS per axis.
# Adjust WS_RADIUS if needed; the true home EE pos is computed from Pinocchio FK.
_WS_RADIUS = 0.20   # metres — generous for peg-in-hole reach

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

def build_obs(q, ee_pos, ee_rot, home_rad, hole_top, hole_bottom, last_action) -> np.ndarray:
    """Build 28-D observation matching the peg_in_hole_osc training environment.

    Peg is assumed rigidly attached to the gripper — peg site world positions are
    computed by rotating the EE-frame offsets by the current EE orientation:
        peg_site_world = ee_pos + ee_rot @ PEG_OFFSET_IN_EE

    Layout (same order as training actor_terms dict):
        joint_pos_rel(7)  — q minus home joint positions
        peg_to_hole(6)    — [hole_top - peg_cylinder_start (3),
                              hole_bottom - peg_cylinder_end (3)]
        ee_to_peg(3)      — peg_cylinder_start - ee_pos  (rotates with EE)
        ee_to_hole(6)     — [hole_top - ee_pos (3),
                              hole_bottom - ee_pos (3)]
        last_action(6)

    Args:
        q          : (7,) current arm joint positions [rad]
        ee_pos     : (3,) current EE world position [m]
        ee_rot     : (3,3) current EE world rotation matrix
        home_rad   : (7,) home joint positions [rad]  (for joint_pos_rel)
        hole_top   : (3,) hole_top site world position [m]  (fixed)
        hole_bottom: (3,) hole_bottom site world position [m]  (fixed)
        last_action: (6,) previous policy action
    """
    # Peg site world positions — rigid attachment, rotates with EE
    peg_start_w = ee_pos + ee_rot @ EE_TO_PEG_ROT @ PEG_CYLINDER_START_OFFSET   # (3,)
    peg_end_w   = ee_pos + ee_rot @ EE_TO_PEG_ROT @ PEG_CYLINDER_END_OFFSET     # (3,)

    joint_pos_rel = q - home_rad                                   # (7,)
    peg_to_hole   = np.concatenate([hole_top    - peg_start_w,    # (3,)
                                    hole_bottom - peg_end_w])     # (3,) → (6,)
    ee_to_peg_obs = peg_start_w - ee_pos                          # (3,) = ee_rot @ EE_TO_PEG_ROT @ offset
    ee_to_hole    = np.concatenate([hole_top    - ee_pos,         # (3,)
                                    hole_bottom - ee_pos])        # (3,) → (6,)

    return np.concatenate([
        joint_pos_rel,  # 7
        peg_to_hole,    # 6
        ee_to_peg_obs,  # 3
        ee_to_hole,     # 6
        last_action,    # 6
    ]).astype(np.float32)  # total: 28


# ── Policy process ────────────────────────────────────────────────────────────

def policy_process_fn(checkpoint_path, device_str, home_rad_arr,
                      shm_q_sim,
                      shm_q_real,
                      shm_osc_target_sim, shm_osc_target_real,
                      shm_action_sim, shm_action_real,
                      shm_ws_lo, shm_ws_hi,
                      shm_hole_pos,
                      shm_policy_hz,
                      policy_reset_event, reset_in_progress, stop_event):
    """Single policy forward pass with batch=2 (sim + real) at TARGET_HZ."""
    policy_agent      = PolicyAgent(checkpoint_path, device=device_str)
    robot             = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    home_rad          = home_rad_arr.copy()
    last_action_sim   = np.zeros(6, dtype=np.float32)
    last_action_real  = np.zeros(6, dtype=np.float32)
    period            = 1.0 / TARGET_HZ
    iters             = 0
    t_rate            = time.time()

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

        # ── Read hole position (updated on reset) ──────────────────────────
        hole_pos    = _np(shm_hole_pos).copy()
        hole_top    = hole_pos + _HOLE_SITE_OFFSETS["hole_top"]
        hole_bottom = hole_pos + _HOLE_SITE_OFFSETS["hole_bottom"]

        # ── Read shared state ──────────────────────────────────────────────
        q_sim  = _np(shm_q_sim).copy()
        q_real = _np(shm_q_real).copy()

        # ── FK ─────────────────────────────────────────────────────────────
        ee_pos_sim,  ee_rot_sim  = robot.fk(q_sim)
        ee_pos_real, ee_rot_real = robot.fk(q_real)

        # ── Build batch observations ───────────────────────────────────────
        obs_sim  = build_obs(q_sim,  ee_pos_sim,  ee_rot_sim,
                             home_rad, hole_top, hole_bottom, last_action_sim)
        obs_real = build_obs(q_real, ee_pos_real, ee_rot_real,
                             home_rad, hole_top, hole_bottom, last_action_real)

        obs_batch    = np.stack([obs_sim, obs_real], axis=0)  # (2, 28)
        action_batch = policy_agent.get_action(obs_batch)     # (2, 6)
        action_sim   = action_batch[0]
        action_real  = action_batch[1]

        last_action_sim[:]  = action_sim
        last_action_real[:] = action_real

        # ── Sim: action → OSC target ───────────────────────────────────────
        osc_tgt_pos_sim = np.clip(ee_pos_sim + action_sim[:3] * DELTA_POS_SCALE,
                                  ws_lo, ws_hi)
        _np(shm_action_sim)[:3] = action_sim[:3] * DELTA_POS_SCALE   # raw delta
        _np(shm_action_sim)[3:] = osc_tgt_pos_sim - ee_pos_sim       # effective delta

        delta_rot_sim    = Rotation.from_rotvec(action_sim[3:] * DELTA_ORI_SCALE)
        osc_tgt_quat_sim = (delta_rot_sim * Rotation.from_matrix(ee_rot_sim)).as_quat()
        _np(shm_osc_target_sim)[:3] = osc_tgt_pos_sim
        _np(shm_osc_target_sim)[3:] = osc_tgt_quat_sim

        # ── Real: action → OSC target ──────────────────────────────────────
        osc_tgt_pos_real = np.clip(ee_pos_real + action_real[:3] * DELTA_POS_SCALE,
                                   ws_lo, ws_hi)
        _np(shm_action_real)[:3] = action_real[:3] * DELTA_POS_SCALE  # raw delta
        _np(shm_action_real)[3:] = osc_tgt_pos_real - ee_pos_real     # effective delta

        delta_rot_real    = Rotation.from_rotvec(action_real[3:] * DELTA_ORI_SCALE)
        osc_tgt_quat_real = (delta_rot_real * Rotation.from_matrix(ee_rot_real)).as_quat()
        _np(shm_osc_target_real)[:3] = osc_tgt_pos_real
        _np(shm_osc_target_real)[3:] = osc_tgt_quat_real

        iters += 1
        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            _np(shm_policy_hz)[0] = iters / dt_rate
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Real process ──────────────────────────────────────────────────────────────

def real_process_fn(ip, home_rad_arr,
                    shm_q_real, shm_dq_real,
                    shm_osc_target_real, shm_gains,
                    shm_real_hz, shm_gripper,
                    stop_event, reset_event, reset_done_event):
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

        _np(shm_q_real)[:]  = kinova_deg_to_rad(pos_deg)
        _np(shm_dq_real)[:] = np.deg2rad(vel_deg)

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

            target = _np(shm_osc_target_real).copy()
            gains  = _gains_dict(_np(shm_gains).copy())

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

                state   = hw.send_torques(tau, pos_deg, gripper_position=float(_np(shm_gripper)[0]))
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
                  shm_osc_hz, osc_reset_event, osc_reset_done, stop_event,
                  posture_target, home_qpos,
                  arm_q_idxs, arm_dq_idxs, arm_ctrl_idxs,
                  gripper_q_idxs, gripper_dq_idxs, gripper_ctrl_idx,
                  peg_qpos_adr, pinch_site_id):
    """Thread in main process: 500 Hz OSC + MuJoCo physics (sim).

    Exactly mirrors training: peg is a free body held by gripper friction/contacts.
    On reset: place peg at pinch_site (world-upright) and close gripper — same as
    reset_peg_in_gripper + reset_gripper_joints_closed in peg_in_hole_osc.py.
    During episode: just step physics, gripper ctrl stays closed.
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
            mj_data.qpos[gripper_q_idxs]  = _GRIPPER_CLOSED_QPOS
            mj_data.qvel[gripper_dq_idxs] = 0.0
            mj_data.ctrl[gripper_ctrl_idx] = _GRIPPER_CTRL_CLOSED
            mujoco.mj_forward(mj_model, mj_data)
            # Place peg at pinch_site, world-upright — same as reset_peg_in_gripper
            mj_data.qpos[peg_qpos_adr:peg_qpos_adr+3] = mj_data.site_xpos[pinch_site_id]
            mj_data.qpos[peg_qpos_adr+3:peg_qpos_adr+7] = np.array([1, 0, 0, 0])  # wxyz
            mujoco.mj_forward(mj_model, mj_data)
            iters  = 0
            t_rate = time.time()
            osc_reset_done.set()
            print("[osc] Reset done")

        q_sim  = mj_data.qpos[arm_q_idxs].copy()
        dq_sim = mj_data.qvel[arm_dq_idxs].copy()

        _np(shm_q_sim)[:]  = q_sim
        _np(shm_dq_sim)[:] = dq_sim

        tgt     = _np(shm_osc_target_sim).copy()
        gains   = _gains_dict(_np(shm_gains).copy())
        tau_sim = compute_osc_torques(
            robot, tgt[:3], tgt[3:], q_sim, dq_sim,
            gains=gains, posture_target=posture_target,
        )

        mj_data.ctrl[arm_ctrl_idxs]   = tau_sim
        mj_data.ctrl[gripper_ctrl_idx] = _GRIPPER_CTRL_CLOSED
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
                  arm_q_idxs, peg_qpos_adr, shm_q_sim, shm_q_real, shm_viz_hz, stop_event):
    period = 1.0 / VIZ_HZ
    iters  = 0
    t_rate = time.time()

    while not stop_event.is_set():
        t = time.time()

        # Sim: copy arm joints from shared memory + peg qpos from physics (held by gripper)
        mj_data_sim.qpos[arm_q_idxs]              = _np(shm_q_sim)
        mj_data_sim.qpos[peg_qpos_adr:peg_qpos_adr+7] = mj_data_phys.qpos[peg_qpos_adr:peg_qpos_adr+7]
        # Real ghost: arm joints only (no peg body — peg markers drawn from EE pose in main loop)
        mj_data_real.qpos[arm_q_idxs]             = _np(shm_q_real)

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
    parser.add_argument("--checkpoint", default="weights/model_499.pt",
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

    # ── Shared memory ──────────────────────────────────────────────────────────
    shm_gains            = mp.Array(ctypes.c_double, len(GAINS_KEYS))
    shm_q_sim            = mp.Array(ctypes.c_double, 7)
    shm_dq_sim           = mp.Array(ctypes.c_double, 7)
    shm_osc_target_sim   = mp.Array(ctypes.c_double, 7)   # pos(3) + quat_xyzw(4)
    shm_action_sim       = mp.Array(ctypes.c_double, 6)   # [:3] raw delta, [3:] eff delta
    shm_q_real           = mp.Array(ctypes.c_double, 7)
    shm_dq_real          = mp.Array(ctypes.c_double, 7)
    shm_osc_target_real  = mp.Array(ctypes.c_double, 7)
    shm_action_real      = mp.Array(ctypes.c_double, 6)
    shm_ws_lo            = mp.Array(ctypes.c_double, 3)
    shm_ws_hi            = mp.Array(ctypes.c_double, 3)
    shm_gripper          = mp.Array(ctypes.c_double, 1)   # 0.0=open, 1.0=closed
    shm_hole_pos         = mp.Array(ctypes.c_double, 3)   # current hole body origin (world)
    shm_osc_hz           = mp.Array(ctypes.c_double, 1)
    shm_policy_hz        = mp.Array(ctypes.c_double, 1)
    shm_viz_hz           = mp.Array(ctypes.c_double, 1)
    shm_real_hz          = mp.Array(ctypes.c_double, 1)

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
    _np(shm_hole_pos)[:] = HOLE_POS_WORLD
    _np(shm_gripper)[0] = 1.0   # start closed

    stop_event         = mp.Event()
    osc_reset_event    = mp.Event()
    osc_reset_done     = threading.Event()  # set by OSC thread when sim reset complete
    policy_reset_event = mp.Event()
    reset_in_progress  = mp.Event()   # set during reset, cleared when all done
    real_reset_event   = mp.Event()
    real_reset_done    = mp.Event()


    # ── Spawn real process BEFORE any CUDA init ────────────────────────────────
    real_proc = None
    if not args.sim_only:
        real_proc = mp.Process(
            target=real_process_fn,
            args=(args.ip, home_rad,
                  shm_q_real, shm_dq_real,
                  shm_osc_target_real, shm_gains, shm_real_hz, shm_gripper,
                  stop_event, real_reset_event, real_reset_done),
            daemon=True,
        )
        real_proc.start()
        print(f"[main] Real process PID: {real_proc.pid}")

    # ── Spawn policy process BEFORE any CUDA init ──────────────────────────────
    policy_proc = mp.Process(
        target=policy_process_fn,
        args=(args.checkpoint, policy_device, home_rad,
              shm_q_sim,
              shm_q_real,
              shm_osc_target_sim, shm_osc_target_real,
              shm_action_sim, shm_action_real,
              shm_ws_lo, shm_ws_hi,
              shm_hole_pos,
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
        # Attach peg (freejoint body) and hole (mocap body) to worldbody
        peg_spec  = mujoco.MjSpec.from_file(str(PEG_XML))
        hole_spec = mujoco.MjSpec.from_file(str(HOLE_XML))
        spec.attach(peg_spec,  prefix="peg/",  suffix="", frame=spec.worldbody.add_frame())
        spec.attach(hole_spec, prefix="hole/", suffix="", frame=spec.worldbody.add_frame())
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

    peg_jnt       = mj_model.joint("peg/peg_joint")
    peg_qpos_adr  = int(peg_jnt.qposadr)
    pinch_site_id = mj_model.site("pinch_site").id

    # Hole mocap body — set once to fixed world position
    hole_mocap_id = int(mj_model.body("hole/hole").mocapid)
    for d_ in (mj_data_phys, mj_data_sim, mj_data_real):
        d_.mocap_pos[hole_mocap_id]  = HOLE_POS_WORLD
        d_.mocap_quat[hole_mocap_id] = np.array([1, 0, 0, 0])  # wxyz identity

    # Arm home + gripper closed initial state
    for d_ in (mj_data_phys, mj_data_sim, mj_data_real):
        d_.qpos[arm_q_idxs]     = home_rad
        d_.qpos[gripper_q_idxs] = _GRIPPER_CLOSED_QPOS
        d_.ctrl[gripper_ctrl_idx] = _GRIPPER_CTRL_CLOSED

    # Run forward to compute pinch_site position, then place peg there
    mujoco.mj_forward(mj_model, mj_data_phys)
    peg_init_pos = mj_data_phys.site_xpos[pinch_site_id].copy()
    for d_ in (mj_data_phys, mj_data_sim, mj_data_real):
        d_.qpos[peg_qpos_adr:peg_qpos_adr+3] = peg_init_pos
        d_.qpos[peg_qpos_adr+3:peg_qpos_adr+7] = np.array([1, 0, 0, 0])

    mujoco.mj_forward(mj_model, mj_data_phys)
    mujoco.mj_kinematics(mj_model, mj_data_sim)
    mujoco.mj_kinematics(mj_model, mj_data_real)

    robot_osc = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")

    # ── Viser ──────────────────────────────────────────────────────────────────
    server    = viser.ViserServer(label="Kinova Peg-in-Hole Sim2Real")
    scene     = ViserMujocoScene.create(server, mj_model)
    sim_view  = scene.add_robot("sim",  color=(0.75, 0.75, 0.75, 1.00))
    real_view = scene.add_robot("real", color=(0.20, 0.55, 0.90, 0.65))

    # Hide hole and peg from real ghost — they're static/sim-only objects
    _hole_body_id = mj_model.body("hole/hole").id
    _peg_body_id  = mj_model.body("peg/peg").id
    for (body_id, _g, _s), handle in real_view._handles.items():
        if body_id in (_hole_body_id, _peg_body_id):
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

    # Peg site markers (policy's assumed peg positions) — updated each loop
    # Hole sites are shown by the MuJoCo model itself (no manual markers needed)
    _peg_start0 = ee_pos0 + ee_rot0 @ EE_TO_PEG_ROT @ PEG_CYLINDER_START_OFFSET
    _peg_end0   = ee_pos0 + ee_rot0 @ EE_TO_PEG_ROT @ PEG_CYLINDER_END_OFFSET
    print(f"[peg markers] ee_pos0={ee_pos0}  peg_start0={_peg_start0}  peg_end0={_peg_end0}")
    peg_start_marker_sim  = server.scene.add_icosphere(
        "/sim_peg_start",  radius=0.01,
        position=tuple(float(v) for v in _peg_start0), color=(1.0, 0.2, 0.2),
    )
    peg_end_marker_sim    = server.scene.add_icosphere(
        "/sim_peg_end",    radius=0.01,
        position=tuple(float(v) for v in _peg_end0),   color=(0.2, 0.4, 1.0),
    )
    peg_start_marker_real = server.scene.add_icosphere(
        "/real_peg_start", radius=0.01,
        position=tuple(float(v) for v in _peg_start0), color=(1.0, 0.5, 0.5),
    )
    peg_end_marker_real   = server.scene.add_icosphere(
        "/real_peg_end",   radius=0.01,
        position=tuple(float(v) for v in _peg_end0),   color=(0.5, 0.6, 1.0),
    )

    # ── GUI ────────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Visualizations"):
        cb_all       = server.gui.add_checkbox("Show all",              initial_value=True)
        cb_bbox      = server.gui.add_checkbox("Workspace bbox",        initial_value=True)
        cb_ee_frames = server.gui.add_checkbox("EE frames",             initial_value=True)
        cb_peg       = server.gui.add_checkbox("Peg site markers",      initial_value=True)
        cb_sim_raw   = server.gui.add_checkbox("Sim raw action arrow",  initial_value=True)
        cb_sim_eff   = server.gui.add_checkbox("Sim eff action arrow",  initial_value=True)
        cb_real_raw  = server.gui.add_checkbox("Real raw action arrow", initial_value=True)
        cb_real_eff  = server.gui.add_checkbox("Real eff action arrow", initial_value=True)

    _all_vis = [
        (cb_bbox,     ws_bbox),
        (cb_peg,      peg_start_marker_sim),
        (cb_peg,      peg_end_marker_sim),
        (cb_peg,      peg_start_marker_real),
        (cb_peg,      peg_end_marker_real),
        (cb_sim_raw,  raw_arrow_sim),
        (cb_sim_eff,  eff_arrow_sim),
        (cb_real_raw, raw_arrow_real),
        (cb_real_eff, eff_arrow_real),
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
    cb_bbox.on_update(    lambda _: setattr(ws_bbox,       "visible", cb_bbox.value))
    cb_peg.on_update(     lambda _: [setattr(o, "visible", cb_peg.value)
                                     for o in (peg_start_marker_sim, peg_end_marker_sim,
                                               peg_start_marker_real, peg_end_marker_real)])
    cb_ee_frames.on_update(lambda _: [setattr(o, "visible", cb_ee_frames.value)
                                      for o in (frame_ee_sim, frame_ee_real)])
    cb_sim_raw.on_update( lambda _: setattr(raw_arrow_sim,  "visible", cb_sim_raw.value))
    cb_sim_eff.on_update( lambda _: setattr(eff_arrow_sim,  "visible", cb_sim_eff.value))
    cb_real_raw.on_update(lambda _: setattr(raw_arrow_real, "visible", cb_real_raw.value))
    cb_real_eff.on_update(lambda _: setattr(eff_arrow_real, "visible", cb_real_eff.value))

    with server.gui.add_folder("Policy"):
        txt_policy_hz = server.gui.add_text(
            f"Policy rate (target {TARGET_HZ} Hz)", initial_value="— Hz")
        txt_osc_hz    = server.gui.add_text(
            f"Sim OSC rate (target {OSC_HZ} Hz)",   initial_value="— Hz")
        txt_real_hz   = server.gui.add_text(
            f"Real rate (target {OSC_HZ} Hz)",      initial_value="— Hz")
        txt_viz_hz    = server.gui.add_text(
            f"Viz rate (target {VIZ_HZ} Hz)",       initial_value="— Hz")
        txt_ee_sim    = server.gui.add_text("Sim EE XYZ",      initial_value="—")
        txt_ee_real   = server.gui.add_text("Real EE XYZ",     initial_value="—")
        txt_peg_hole  = server.gui.add_text("Sim peg→hole dist", initial_value="—")
        reset_btn     = server.gui.add_button("Reset")
        sl_gripper    = server.gui.add_slider("Gripper", min=0.0, max=1.0, step=0.01,
                                              initial_value=1.0)

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
        # Pause policy immediately
        reset_in_progress.set()
        osc_reset_done.clear()
        real_reset_done.clear()

        # Sample new hole position
        dx = np.random.uniform(*HOLE_RAND_X)
        dy = np.random.uniform(*HOLE_RAND_Y)
        new_hole_pos = HOLE_POS_WORLD + np.array([dx, dy, 0.0])
        _np(shm_hole_pos)[:] = new_hole_pos
        for d_ in (mj_data_phys, mj_data_sim, mj_data_real):
            d_.mocap_pos[hole_mocap_id] = new_hole_pos
        print(f"[reset] hole pos: {new_hole_pos}  (dx={dx:+.3f}, dy={dy:+.3f})")

        # Trigger all resets
        osc_reset_event.set()
        policy_reset_event.set()
        _np(shm_osc_target_sim)[:3]  = ee_pos0
        _np(shm_osc_target_sim)[3:]  = q_xyzw0
        _np(shm_osc_target_real)[:3] = ee_pos0
        _np(shm_osc_target_real)[3:] = q_xyzw0

        # Wait for sim OSC reset to complete
        osc_reset_done.wait()
        print("[reset] Sim reset done.")

        if real_proc is not None:
            real_reset_event.set()
            print("[reset] Waiting for real robot to finish homing…")
            real_reset_done.wait()
            print("[reset] Real robot homed.")

        # Both sim and real are reset — unpause policy
        reset_in_progress.clear()
        print("[reset] All done, policy resuming.")

    reset_btn.on_click(_on_reset)

    # ── Start threads ──────────────────────────────────────────────────────────
    threading.Thread(target=osc_thread_fn, daemon=True, args=(
        mj_model, mj_data_phys, robot_osc,
        shm_q_sim, shm_dq_sim, shm_osc_target_sim, shm_gains,
        shm_osc_hz, osc_reset_event, osc_reset_done, stop_event,
        home_rad, home_rad,
        arm_q_idxs, arm_dq_idxs, arm_ctrl_idxs,
        gripper_q_idxs, gripper_dq_idxs, gripper_ctrl_idx,
        peg_qpos_adr, pinch_site_id,
    )).start()

    threading.Thread(target=viz_thread_fn, daemon=True, args=(
        sim_view, real_view, mj_model, mj_data_phys, mj_data_sim, mj_data_real,
        arm_q_idxs, peg_qpos_adr, shm_q_sim, shm_q_real,
        shm_viz_hz, stop_event,
    )).start()

    print("Running. Ctrl+C to stop.")

    def _wxyz(q): return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

    try:
        _print_step = 0
        while True:
            # ── Sync gains + gripper ───────────────────────────────────────────
            _np(shm_gains)[:] = _pack_gains(
                sl_kp_pos.value, sl_kd_pos.value,
                sl_kp_ori.value, sl_kd_ori.value,
                sl_post_kp.value, sl_post_kd.value, sl_post_w.value,
            )
            _np(shm_gripper)[0] = sl_gripper.value

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

            # ── Peg site markers (policy's assumed peg positions, not sim geom) ──
            if cb_peg.value:
                peg_start_marker_sim.position  = tuple(float(v) for v in
                    ee_pos_sim  + ee_rot_sim  @ EE_TO_PEG_ROT @ PEG_CYLINDER_START_OFFSET)
                peg_end_marker_sim.position    = tuple(float(v) for v in
                    ee_pos_sim  + ee_rot_sim  @ EE_TO_PEG_ROT @ PEG_CYLINDER_END_OFFSET)
                peg_start_marker_real.position = tuple(float(v) for v in
                    ee_pos_real + ee_rot_real @ EE_TO_PEG_ROT @ PEG_CYLINDER_START_OFFSET)
                peg_end_marker_real.position   = tuple(float(v) for v in
                    ee_pos_real + ee_rot_real @ EE_TO_PEG_ROT @ PEG_CYLINDER_END_OFFSET)

            # ── Action arrows ──────────────────────────────────────────────────
            raw_delta_sim  = _np(shm_action_sim)[:3].copy()
            eff_delta_sim  = _np(shm_action_sim)[3:].copy()
            raw_delta_real = _np(shm_action_real)[:3].copy()
            eff_delta_real = _np(shm_action_real)[3:].copy()
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

            # Peg-to-hole distance (sim) — policy's assumed cylinder_start → hole_top
            peg_start_sim = ee_pos_sim + ee_rot_sim @ EE_TO_PEG_ROT @ PEG_CYLINDER_START_OFFSET
            dist_sim = float(np.linalg.norm(HOLE_TOP_WORLD - peg_start_sim))
            txt_peg_hole.value = f"{dist_sim*100:.1f} cm"

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
                    f"peg→hole {dist_sim*100:.1f} cm  "
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
