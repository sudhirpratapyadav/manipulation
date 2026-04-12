"""Kinova Gen3 sim2real policy tester.

An NN policy (RSL-RL PPO checkpoint) drives both sim and real arm simultaneously.
A single policy forward pass with batch=2 produces actions for both at once.
The shared command target comes from a viser transform-controls handle.

Policy observations (38-D, same as reach_osc training):
    joint_pos(7) + joint_vel(7) + ee_to_target(6) + ee_pose(6) +
    target_pose(6) + last_action(6)

Policy action (6-D):
    [delta_pos(3), delta_ori_axis_angle(3)]
    scaled by DELTA_POS_SCALE / DELTA_ORI_SCALE before applying to EE pose.

Processes / Threads:
  - Policy process  (mp.Process): batch=2 inference at TARGET_HZ Hz
                                   reads shm_q_sim/dq_sim + shm_q_real/dq_real
                                   writes shm_osc_target_sim + shm_osc_target_real
  - Real process    (mp.Process): 500 Hz OSC → hardware torques
                                   reads shm_q_real/dq_real (published back),
                                   reads shm_osc_target_real
  - OSC thread      (threading):  500 Hz OSC + MuJoCo physics (sim)
                                   reads shm_osc_target_sim, writes shm_q_sim/dq_sim
  - Viz thread      (threading):  30 Hz — shows sim (grey) + real (blue ghost)
  - Main thread:                  viser polling + GUI (writes shm_command)

Startup order:
  1. Create shared memory + events
  2. Spawn real process        ← BEFORE any GPU/CUDA init (clean fork)
  3. Spawn policy process      ← BEFORE any CUDA init (clean fork)
  4. Init MuJoCo sim, Pinocchio, viser
  5. Start OSC + viz threads
  6. Main loop

Run:
  python sim2real_test_policy.py --checkpoint weights/model_4999.pt --ip 192.168.1.10
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
from kinova_tasks.assets.kinova_gen3.kinova_constants import KINOVA_GEN3_GRIPPER_XML, get_assets
from policy import PolicyAgent
from pynput import keyboard as pynput_kb
from viewer import ViserMujocoScene

# ── Model ─────────────────────────────────────────────────────────────────────
_TORQUE_XML_NO_GRIPPER = KINOVA_GEN3_GRIPPER_XML.parent / "gen3_no_gripper_torque.xml"
_TORQUE_XML_GRIPPER    = KINOVA_GEN3_GRIPPER_XML.parent / "gen3_gripper_torque.xml"
_ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]

# ── OSC gains — must match the training environment (reach_osc.py) ────────────
KP_POS         =  50.0
KD_POS         =  10.0
KP_ORI         =  50.0
KD_ORI         =  10.0
POSTURE_KP     =  10.0
POSTURE_KD     =   2.0
POSTURE_WEIGHT =   0.0   # zero in training env

MAX_JOINT_TORQUE = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])
TAU_OFFSETS_DEFAULT = np.array([0.0, 0.0, -0.5, 0.0, 0.0, 1.0, 0.0])

# ── Action scales — must match the training environment (reach_osc.py) ────────
DELTA_POS_SCALE = 0.02   # metres per unit action
DELTA_ORI_SCALE = 0.02   # rad per unit action

# ── Workspace bounds — centred on home EE pos from reach_osc.py ──────────────
_HOME_EE_POS = np.array([0.733607, -0.024850, 0.523015])
_WS_RADIUS   = 0.1   # ± 15 cm per axis
WS_LO = _HOME_EE_POS - _WS_RADIUS
WS_HI = _HOME_EE_POS + _WS_RADIUS

# ── Timing ────────────────────────────────────────────────────────────────────
PHYSICS_DT   = 0.002
TARGET_HZ    = 10
OSC_HZ       = 500
OSC_SUBSTEPS = OSC_HZ // TARGET_HZ
VIZ_HZ       = 30

HOME_DEG = np.array([0.0, 20.0, 0.0, 100.0, 0.0, -30.0, -90.0])

# ── Keyboard target control ───────────────────────────────────────────────────
KB_DELTA_POS = 0.005          # metres per main-loop tick while key held
KB_DELTA_ROT = np.deg2rad(0.5) # radians per main-loop tick while key held

KEY_DELTAS = {
    "w": np.array([ KB_DELTA_POS, 0, 0, 0, 0, 0]),
    "s": np.array([-KB_DELTA_POS, 0, 0, 0, 0, 0]),
    "a": np.array([0,  KB_DELTA_POS, 0, 0, 0, 0]),
    "d": np.array([0, -KB_DELTA_POS, 0, 0, 0, 0]),
    "e": np.array([0, 0,  KB_DELTA_POS, 0, 0, 0]),
    "q": np.array([0, 0, -KB_DELTA_POS, 0, 0, 0]),
    "i": np.array([0, 0, 0,  KB_DELTA_ROT, 0, 0]),
    "k": np.array([0, 0, 0, -KB_DELTA_ROT, 0, 0]),
    "j": np.array([0, 0, 0, 0,  KB_DELTA_ROT, 0]),
    "l": np.array([0, 0, 0, 0, -KB_DELTA_ROT, 0]),
    "u": np.array([0, 0, 0, 0, 0,  KB_DELTA_ROT]),
    "o": np.array([0, 0, 0, 0, 0, -KB_DELTA_ROT]),
}

_held_keys: set = set()
_held_keys_lock = threading.Lock()


def _on_key_press(key):
    try:
        with _held_keys_lock:
            _held_keys.add(key.char)
    except AttributeError:
        pass


def _on_key_release(key):
    try:
        with _held_keys_lock:
            _held_keys.discard(key.char)
    except AttributeError:
        pass


# ── Gains keys ────────────────────────────────────────────────────────────────
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
        self.joint_lower = self.model.lowerPositionLimit[self._q_idx].copy()
        self.joint_upper = self.model.upperPositionLimit[self._q_idx].copy()

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
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
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
        pin.computeJointJacobiansTimeVariation(self.model, self.data, self._q_full, self._dq_full)
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


def compute_osc_torques(robot, target_pos, target_quat_xyzw, q, dq, *, gains, posture_target):
    """Operational Space Control → joint torques.

    Λ = (J M⁻¹ Jᵀ)⁻¹
    F = Λ (ẍ_des − dJ·dq)
    τ = Jᵀ F + nle + w_posture · Nᵀ τ_posture
    """
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

    N           = np.eye(7) - J.T @ J_dyn_inv.T
    tau_posture = gains["posture_kp"] * (posture_target - q) + gains["posture_kd"] * (0.0 - dq)

    tau = J.T @ F + nle + gains["posture_weight"] * (N @ tau_posture)
    return np.clip(tau, -MAX_JOINT_TORQUE, MAX_JOINT_TORQUE)


# ── Arrow mesh helper ────────────────────────────────────────────────────────

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

    ring = lambda r, along: (start + d_norm * along)[None] + r * (c[:, None] * perp + s[:, None] * tang)

    shaft_bot  = ring(shaft_r, 0.0)
    shaft_top  = ring(shaft_r, shaft_len)
    head_base  = ring(head_r,  shaft_len)
    head_tip   = end[None]
    ctr_bot    = start[None]
    ctr_stop   = (start + d_norm * shaft_len)[None]

    verts = np.vstack([shaft_bot, shaft_top, head_base, head_tip, ctr_bot, ctr_stop]).astype(np.float32)
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

def build_obs(q, dq, ee_pos, ee_rot, cmd_pos, cmd_quat_xyzw, last_action) -> np.ndarray:
    """Build 38-D observation matching the reach_osc training environment."""
    ee_aa        = Rotation.from_matrix(ee_rot).as_rotvec()
    ee_pose_obs  = np.concatenate([ee_pos, ee_aa])

    tgt_aa       = Rotation.from_quat(cmd_quat_xyzw).as_rotvec()
    tgt_pose_obs = np.concatenate([cmd_pos, tgt_aa])

    pos_err = cmd_pos - ee_pos
    R_tgt   = Rotation.from_quat(cmd_quat_xyzw).as_matrix()
    rot_err = Rotation.from_matrix(R_tgt @ ee_rot.T).as_rotvec()
    ee_to_target_obs = np.concatenate([pos_err, rot_err])

    return np.concatenate([
        q, dq, ee_to_target_obs, ee_pose_obs, tgt_pose_obs, last_action,
    ]).astype(np.float32)


# ── Policy process (batch=2: sim + real) ─────────────────────────────────────

def policy_process_fn(checkpoint_path, device_str,
                      # sim state (written by OSC thread)
                      shm_q_sim, shm_dq_sim,
                      # real state (written by real process)
                      shm_q_real, shm_dq_real,
                      # shared command target (written by main thread)
                      shm_command,
                      # OSC targets (read by OSC thread / real process)
                      shm_osc_target_sim, shm_osc_target_real,
                      # action deltas for viz arrows
                      shm_action_sim, shm_action_real,
                      shm_policy_hz,
                      policy_reset_event, stop_event, torque_xml):
    """Single policy forward pass with batch=2 (sim + real) at TARGET_HZ."""
    policy_agent     = PolicyAgent(checkpoint_path, device=device_str)
    robot            = PinocchioArm(str(torque_xml), ee_frame="pinch_site")
    last_action_sim  = np.zeros(6, dtype=np.float32)
    last_action_real = np.zeros(6, dtype=np.float32)
    period           = 1.0 / TARGET_HZ
    iters            = 0
    t_rate           = time.time()

    while not stop_event.is_set():
        t0 = time.time()

        if policy_reset_event.is_set():
            policy_reset_event.clear()
            last_action_sim[:]  = 0.0
            last_action_real[:] = 0.0
            iters  = 0
            t_rate = time.time()

        # ── Read shared state ──────────────────────────────────────────────
        q_sim  = _np(shm_q_sim).copy()
        dq_sim = _np(shm_dq_sim).copy()
        q_real  = _np(shm_q_real).copy()
        dq_real = _np(shm_dq_real).copy()
        cmd = _np(shm_command).copy()
        cmd_pos, cmd_quat_xyzw = cmd[:3], cmd[3:]

        # ── FK for both ────────────────────────────────────────────────────
        ee_pos_sim,  ee_rot_sim  = robot.fk(q_sim)
        ee_pos_real, ee_rot_real = robot.fk(q_real)

        # ── Build batch of 2 observations ─────────────────────────────────
        obs_sim  = build_obs(q_sim,  dq_sim,  ee_pos_sim,  ee_rot_sim,
                             cmd_pos, cmd_quat_xyzw, last_action_sim)
        obs_real = build_obs(q_real, dq_real, ee_pos_real, ee_rot_real,
                             cmd_pos, cmd_quat_xyzw, last_action_real)

        # Single forward pass — policy_agent must support batch input
        obs_batch   = np.stack([obs_sim, obs_real], axis=0)   # (2, 38)
        action_batch = policy_agent.get_action(obs_batch)      # (2, 6)
        action_sim   = action_batch[0]
        action_real  = action_batch[1]

        last_action_sim[:]  = action_sim
        last_action_real[:] = action_real

        # ── Sim: action → OSC target ───────────────────────────────────────
        osc_tgt_pos_sim  = ee_pos_sim + action_sim[:3] * DELTA_POS_SCALE
        osc_tgt_pos_sim  = np.clip(osc_tgt_pos_sim, WS_LO, WS_HI)
        _np(shm_action_sim)[:3] = action_sim[:3] * DELTA_POS_SCALE        # raw delta
        _np(shm_action_sim)[3:] = osc_tgt_pos_sim - ee_pos_sim            # effective delta
        delta_rot_sim    = Rotation.from_rotvec(action_sim[3:] * DELTA_ORI_SCALE)
        osc_tgt_quat_sim = (delta_rot_sim * Rotation.from_matrix(ee_rot_sim)).as_quat()
        _np(shm_osc_target_sim)[:3] = osc_tgt_pos_sim
        _np(shm_osc_target_sim)[3:] = osc_tgt_quat_sim

        # ── Real: action → OSC target ──────────────────────────────────────
        osc_tgt_pos_real  = ee_pos_real + action_real[:3] * DELTA_POS_SCALE
        osc_tgt_pos_real  = np.clip(osc_tgt_pos_real, WS_LO, WS_HI)
        _np(shm_action_real)[:3] = action_real[:3] * DELTA_POS_SCALE      # raw delta
        _np(shm_action_real)[3:] = osc_tgt_pos_real - ee_pos_real         # effective delta
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


# ── Real process (separate process — no GIL sharing) ─────────────────────────

def real_process_fn(ip,
                    shm_q_real, shm_dq_real,
                    shm_osc_target_real, shm_gains,
                    shm_real_hz,
                    stop_event, reset_event, reset_done_event, torque_xml):
    """Connects to hardware, runs OSC at 500 Hz, publishes state back."""
    inner_dt       = 1.0 / OSC_HZ
    robot          = PinocchioArm(str(torque_xml), ee_frame="pinch_site")
    posture_target = kinova_deg_to_rad(HOME_DEG)

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

        # Publish initial state
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
                iters   = 0
                t_rate  = time.time()
                reset_done_event.set()
                continue

            # Snapshot target + gains
            target = _np(shm_osc_target_real).copy()
            gains  = _gains_dict(_np(shm_gains).copy())

            # ── OSC inner loop at 500 Hz ───────────────────────────────────
            for _ in range(OSC_SUBSTEPS):
                t_inner = time.time()

                q  = kinova_deg_to_rad(pos_deg)
                dq = np.deg2rad(vel_deg)

                # Publish current state for policy process
                _np(shm_q_real)[:]  = q
                _np(shm_dq_real)[:] = dq

                tau = compute_osc_torques(
                    robot, target[:3], target[3:], q, dq,
                    gains=gains, posture_target=posture_target,
                )
                tau += TAU_OFFSETS_DEFAULT

                state   = hw.send_torques(tau, pos_deg)
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
                  shm_osc_hz, osc_reset_event, stop_event,
                  posture_target, home_qpos, arm_q_idxs, arm_dq_idxs, arm_ctrl_idxs):
    """Thread in main process: 500 Hz OSC + MuJoCo physics (sim)."""
    inner_dt = 1.0 / OSC_HZ
    iters    = 0
    t_rate   = time.time()

    while not stop_event.is_set():
        t0 = time.time()

        if osc_reset_event.is_set():
            osc_reset_event.clear()
            mj_data.qpos[arm_q_idxs] = home_qpos
            mj_data.qvel[arm_dq_idxs] = 0.0
            mj_data.ctrl[arm_ctrl_idxs] = 0.0
            mujoco.mj_forward(mj_model, mj_data)
            iters  = 0
            t_rate = time.time()
            print("[osc] Reset done")

        q_sim  = mj_data.qpos[arm_q_idxs].copy()
        dq_sim = mj_data.qvel[arm_dq_idxs].copy()

        # Publish sim state for policy process
        _np(shm_q_sim)[:]  = q_sim
        _np(shm_dq_sim)[:] = dq_sim

        # Read latest OSC target written by policy process
        tgt               = _np(shm_osc_target_sim).copy()
        osc_tgt_pos       = tgt[:3]
        osc_tgt_quat_xyzw = tgt[3:]

        gains   = _gains_dict(_np(shm_gains).copy())
        tau_sim = compute_osc_torques(
            robot, osc_tgt_pos, osc_tgt_quat_xyzw, q_sim, dq_sim,
            gains=gains, posture_target=posture_target,
        )

        mj_data.ctrl[arm_ctrl_idxs] = tau_sim
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

def viz_thread_fn(sim_view, real_view, mj_model, mj_data_sim, mj_data_real,
                  arm_q_idxs, shm_q_sim, shm_q_real, shm_viz_hz, stop_event):
    period = 1.0 / VIZ_HZ
    iters  = 0
    t_rate = time.time()

    while not stop_event.is_set():
        t = time.time()

        mj_data_sim.qpos[arm_q_idxs]  = _np(shm_q_sim)
        mj_data_real.qpos[arm_q_idxs] = _np(shm_q_real)

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
    parser.add_argument("--checkpoint", default="weights/model_4999.pt")
    parser.add_argument("--ip",         default="192.168.1.10")
    parser.add_argument("--sim-only",   action="store_true",
                        help="Skip real robot (sim only, no hardware required)")
    parser.add_argument("--no-gripper", action="store_true")
    args = parser.parse_args()

    _TORQUE_XML = _TORQUE_XML_NO_GRIPPER if args.no_gripper else _TORQUE_XML_GRIPPER

    policy_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── Shared memory ──────────────────────────────────────────────────────
    shm_command          = mp.Array(ctypes.c_double, 7)   # pos(3) + quat_xyzw(4)
    shm_gains            = mp.Array(ctypes.c_double, len(GAINS_KEYS))
    # sim state
    shm_q_sim            = mp.Array(ctypes.c_double, 7)
    shm_dq_sim           = mp.Array(ctypes.c_double, 7)
    shm_osc_target_sim   = mp.Array(ctypes.c_double, 7)   # pos(3) + quat_xyzw(4)
    shm_action_sim       = mp.Array(ctypes.c_double, 6)   # [:3] raw delta, [3:] effective delta
    # real state
    shm_q_real           = mp.Array(ctypes.c_double, 7)
    shm_dq_real          = mp.Array(ctypes.c_double, 7)
    shm_osc_target_real  = mp.Array(ctypes.c_double, 7)
    shm_action_real      = mp.Array(ctypes.c_double, 6)
    # rates
    shm_osc_hz           = mp.Array(ctypes.c_double, 1)
    shm_policy_hz        = mp.Array(ctypes.c_double, 1)
    shm_viz_hz           = mp.Array(ctypes.c_double, 1)
    shm_real_hz          = mp.Array(ctypes.c_double, 1)

    _np(shm_gains)[:] = _pack_gains(KP_POS, KD_POS, KP_ORI, KD_ORI,
                                     POSTURE_KP, POSTURE_KD, POSTURE_WEIGHT)

    stop_event          = mp.Event()
    osc_reset_event     = mp.Event()
    policy_reset_event  = mp.Event()
    real_reset_event    = mp.Event()
    real_reset_done     = mp.Event()

    # ── Pinocchio (CPU only — before any CUDA) ─────────────────────────────
    print("Loading Pinocchio…")
    robot_main     = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    home_rad       = kinova_deg_to_rad(HOME_DEG)
    posture_target = home_rad.copy()
    ee_pos0, ee_rot0 = robot_main.fk(home_rad)
    q_xyzw0          = Rotation.from_matrix(ee_rot0).as_quat()   # xyzw

    # Initialise shared state to home
    _np(shm_q_sim)[:]           = home_rad
    _np(shm_dq_sim)[:]          = 0.0
    _np(shm_q_real)[:]          = home_rad
    _np(shm_dq_real)[:]         = 0.0
    _np(shm_command)[:3]        = ee_pos0
    _np(shm_command)[3:]        = q_xyzw0
    _np(shm_osc_target_sim)[:3] = ee_pos0
    _np(shm_osc_target_sim)[3:] = q_xyzw0
    _np(shm_osc_target_real)[:3]= ee_pos0
    _np(shm_osc_target_real)[3:]= q_xyzw0
    print("  OK")

    # ── Spawn real process BEFORE any CUDA init ────────────────────────────
    real_proc = None
    if not args.sim_only:
        real_proc = mp.Process(
            target=real_process_fn,
            args=(args.ip,
                  shm_q_real, shm_dq_real,
                  shm_osc_target_real, shm_gains, shm_real_hz,
                  stop_event, real_reset_event, real_reset_done, _TORQUE_XML),
            daemon=True,
        )
        real_proc.start()
        print(f"[main] Real process PID: {real_proc.pid}")

    # ── Spawn policy process BEFORE any CUDA init ──────────────────────────
    policy_proc = mp.Process(
        target=policy_process_fn,
        args=(args.checkpoint, policy_device,
              shm_q_sim, shm_dq_sim,
              shm_q_real, shm_dq_real,
              shm_command,
              shm_osc_target_sim, shm_osc_target_real,
              shm_action_sim, shm_action_real,
              shm_policy_hz,
              policy_reset_event, stop_event, _TORQUE_XML),
        daemon=True,
    )
    policy_proc.start()
    print(f"[main] Policy process PID: {policy_proc.pid}")

    # ── MuJoCo sim ─────────────────────────────────────────────────────────
    def _compile_model():
        spec = mujoco.MjSpec.from_file(str(_TORQUE_XML))
        spec.assets = get_assets(spec.meshdir)
        return spec.compile()

    mj_model     = _compile_model()
    mj_data_phys = mujoco.MjData(mj_model)
    mj_data_sim  = mujoco.MjData(mj_model)   # viz thread (sim ghost)
    mj_data_real = mujoco.MjData(mj_model)   # viz thread (real ghost)

    arm_q_idxs    = np.array([mj_model.joint(n).qposadr.item() for n in _ARM_JOINT_NAMES])
    arm_dq_idxs   = np.array([mj_model.joint(n).dofadr.item()  for n in _ARM_JOINT_NAMES])
    arm_ctrl_idxs = np.array([mj_model.actuator(n).id          for n in _ARM_JOINT_NAMES])

    mj_data_phys.qpos[arm_q_idxs] = home_rad
    mujoco.mj_forward(mj_model, mj_data_phys)
    mj_data_sim.qpos[arm_q_idxs]  = home_rad
    mj_data_real.qpos[arm_q_idxs] = home_rad
    mujoco.mj_kinematics(mj_model, mj_data_sim)
    mujoco.mj_kinematics(mj_model, mj_data_real)

    robot_osc = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")  # OSC thread

    # ── Viser ──────────────────────────────────────────────────────────────
    server    = viser.ViserServer(label="Kinova Sim2Real Policy")
    scene     = ViserMujocoScene.create(server, mj_model)
    sim_view  = scene.add_robot("sim",  color=(0.75, 0.75, 0.75, 1.00))
    real_view = scene.add_robot("real", color=(0.20, 0.55, 0.90, 0.65))
    scene.create_visualization_gui(camera_distance=1.2, camera_azimuth=135.0, camera_elevation=30.0)

    # EE frames for sim and real
    def _add_ee_frame(name):
        return server.scene.add_frame(
            f"/{name}_ee_frame",
            position=tuple(float(v) for v in ee_pos0),
            wxyz=(float(q_xyzw0[3]), float(q_xyzw0[0]), float(q_xyzw0[1]), float(q_xyzw0[2])),
            axes_length=0.08, axes_radius=0.003,
        )
    frame_ee_sim  = _add_ee_frame("sim")
    frame_ee_real = _add_ee_frame("real")

    # Action arrows for sim (orange=raw, yellow=clipped) and real (red=raw, green=clipped)
    _av0, _af0 = _make_arrow_mesh(ee_pos0, ee_pos0 + np.array([0.02, 0., 0.]))
    raw_arrow_sim  = server.scene.add_mesh_simple("/sim_raw_action",  vertices=_av0,       faces=_af0, color=(1.0, 0.45, 0.0), side="double")
    eff_arrow_sim  = server.scene.add_mesh_simple("/sim_eff_action",  vertices=_av0.copy(), faces=_af0, color=(1.0, 0.85, 0.0), side="double")
    raw_arrow_real = server.scene.add_mesh_simple("/real_raw_action", vertices=_av0.copy(), faces=_af0, color=(1.0, 0.20, 0.2), side="double")
    eff_arrow_real = server.scene.add_mesh_simple("/real_eff_action", vertices=_av0.copy(), faces=_af0, color=(0.20, 0.90, 0.3), side="double")

    # Workspace bounding box
    _lo, _hi = WS_LO, WS_HI
    _corners = np.array([
        [_lo[0], _lo[1], _lo[2]], [_hi[0], _lo[1], _lo[2]],
        [_lo[0], _hi[1], _lo[2]], [_hi[0], _hi[1], _lo[2]],
        [_lo[0], _lo[1], _hi[2]], [_hi[0], _lo[1], _hi[2]],
        [_lo[0], _hi[1], _hi[2]], [_hi[0], _hi[1], _hi[2]],
    ], dtype=np.float32)
    _edges = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    ws_bbox = server.scene.add_line_segments(
        "/workspace_bounds",
        points=np.array([[_corners[a], _corners[b]] for a, b in _edges], dtype=np.float32),
        colors=np.array([0.8, 0.8, 0.2], dtype=np.float32),
        line_width=1.5,
    )

    # Interactive transform-controls handle — shared command target
    transform_ctrl = server.scene.add_transform_controls(
        "/command_target",
        position=(float(ee_pos0[0]), float(ee_pos0[1]), float(ee_pos0[2])),
        wxyz=(float(q_xyzw0[3]), float(q_xyzw0[0]), float(q_xyzw0[1]), float(q_xyzw0[2])),
        scale=0.15,
    )

    with server.gui.add_folder("Visualizations"):
        cb_all        = server.gui.add_checkbox("Show all",             initial_value=True)
        cb_bbox       = server.gui.add_checkbox("Workspace bbox",       initial_value=True)
        cb_ee_frames  = server.gui.add_checkbox("EE frames",            initial_value=True)
        cb_sim_raw    = server.gui.add_checkbox("Sim raw action arrow",  initial_value=True)
        cb_sim_eff    = server.gui.add_checkbox("Sim eff action arrow",  initial_value=True)
        cb_real_raw   = server.gui.add_checkbox("Real raw action arrow", initial_value=True)
        cb_real_eff   = server.gui.add_checkbox("Real eff action arrow", initial_value=True)

    # "Show all" toggles every individual checkbox + scene object
    def _on_show_all(_event):
        v = cb_all.value
        cb_bbox.value      = v
        cb_ee_frames.value = v
        cb_sim_raw.value   = v
        cb_sim_eff.value   = v
        cb_real_raw.value  = v
        cb_real_eff.value  = v
        ws_bbox.visible        = v
        frame_ee_sim.visible   = v
        frame_ee_real.visible  = v
        raw_arrow_sim.visible  = v
        eff_arrow_sim.visible  = v
        raw_arrow_real.visible = v
        eff_arrow_real.visible = v

    cb_all.on_update(_on_show_all)
    cb_bbox.on_update(     lambda _: setattr(ws_bbox,        "visible", cb_bbox.value))
    cb_ee_frames.on_update(lambda _: (setattr(frame_ee_sim,  "visible", cb_ee_frames.value),
                                      setattr(frame_ee_real, "visible", cb_ee_frames.value)))
    cb_sim_raw.on_update(  lambda _: setattr(raw_arrow_sim,  "visible", cb_sim_raw.value))
    cb_sim_eff.on_update(  lambda _: setattr(eff_arrow_sim,  "visible", cb_sim_eff.value))
    cb_real_raw.on_update( lambda _: setattr(raw_arrow_real, "visible", cb_real_raw.value))
    cb_real_eff.on_update( lambda _: setattr(eff_arrow_real, "visible", cb_real_eff.value))

    with server.gui.add_folder("Policy"):
        txt_policy_hz = server.gui.add_text(f"Policy rate (target {TARGET_HZ} Hz)", initial_value="— Hz")
        txt_osc_hz    = server.gui.add_text(f"Sim OSC rate (target {OSC_HZ} Hz)",   initial_value="— Hz")
        txt_real_hz   = server.gui.add_text(f"Real rate (target {OSC_HZ} Hz)",      initial_value="— Hz")
        txt_viz_hz    = server.gui.add_text(f"Viz rate (target {VIZ_HZ} Hz)",       initial_value="— Hz")
        txt_cmd       = server.gui.add_text("Command XYZ",    initial_value="—")
        txt_ee_sim    = server.gui.add_text("Sim EE XYZ",     initial_value="—")
        txt_ee_real   = server.gui.add_text("Real EE XYZ",    initial_value="—")
        txt_err_sim   = server.gui.add_text("Sim pos error",  initial_value="—")
        txt_err_real  = server.gui.add_text("Real pos error", initial_value="—")
        reset_btn     = server.gui.add_button("Reset")

    with server.gui.add_folder("OSC Gains"):
        sl_kp_pos = server.gui.add_slider("Kp pos", min=0.0, max=1000.0, step=1.0,  initial_value=KP_POS)
        sl_kd_pos = server.gui.add_slider("Kd pos", min=0.0, max=200.0,  step=0.5,  initial_value=KD_POS)
        sl_kp_ori = server.gui.add_slider("Kp ori", min=0.0, max=1000.0, step=1.0,  initial_value=KP_ORI)
        sl_kd_ori = server.gui.add_slider("Kd ori", min=0.0, max=200.0,  step=0.5,  initial_value=KD_ORI)

    with server.gui.add_folder("Posture"):
        sl_post_kp = server.gui.add_slider("Posture Kp",     min=0.0, max=100.0, step=0.1,  initial_value=POSTURE_KP)
        sl_post_kd = server.gui.add_slider("Posture Kd",     min=0.0, max=20.0,  step=0.1,  initial_value=POSTURE_KD)
        sl_post_w  = server.gui.add_slider("Posture weight", min=0.0, max=1.0,   step=0.01, initial_value=POSTURE_WEIGHT)

    with server.gui.add_folder("Joint Angles (deg)"):
        txt_joints = [
            server.gui.add_text(f"J{i+1}  real | sim | err", initial_value="— | — | —")
            for i in range(7)
        ]

    def _on_reset(_):
        osc_reset_event.set()
        policy_reset_event.set()
        if real_proc is not None:
            real_reset_event.set()
        p = (float(ee_pos0[0]), float(ee_pos0[1]), float(ee_pos0[2]))
        w = (float(q_xyzw0[3]), float(q_xyzw0[0]), float(q_xyzw0[1]), float(q_xyzw0[2]))
        _np(shm_command)[:3]         = ee_pos0
        _np(shm_command)[3:]         = q_xyzw0
        _np(shm_osc_target_sim)[:3]  = ee_pos0
        _np(shm_osc_target_sim)[3:]  = q_xyzw0
        _np(shm_osc_target_real)[:3] = ee_pos0
        _np(shm_osc_target_real)[3:] = q_xyzw0
        transform_ctrl.position = p
        transform_ctrl.wxyz     = w

    reset_btn.on_click(_on_reset)

    threading.Thread(target=osc_thread_fn, daemon=True, args=(
        mj_model, mj_data_phys, robot_osc,
        shm_q_sim, shm_dq_sim, shm_osc_target_sim, shm_gains,
        shm_osc_hz, osc_reset_event, stop_event,
        posture_target, home_rad, arm_q_idxs, arm_dq_idxs, arm_ctrl_idxs,
    )).start()

    threading.Thread(target=viz_thread_fn, daemon=True, args=(
        sim_view, real_view, mj_model, mj_data_sim, mj_data_real,
        arm_q_idxs, shm_q_sim, shm_q_real, shm_viz_hz, stop_event,
    )).start()

    # ── Keyboard listener ──────────────────────────────────────────────────
    kb_listener = pynput_kb.Listener(on_press=_on_key_press, on_release=_on_key_release)
    kb_listener.start()

    print("Running — drag the viser transform handle OR use keyboard to move target:")
    print("  Translation  +X/−X: w/s   +Y/−Y: a/d   +Z/−Z: e/q")
    print("  Rotation     +Rx/−Rx: i/k  +Ry/−Ry: j/l  +Rz/−Rz: u/o")
    print("Ctrl+C to stop.\n")

    # Persistent target state — updated by keyboard or viser drag
    cmd_pos       = ee_pos0.copy()
    cmd_quat_xyzw = q_xyzw0.copy()

    try:
        _print_step = 0
        while True:
            # ── Keyboard: accumulate delta onto persistent target ───────────
            with _held_keys_lock:
                held = _held_keys.copy()

            kb_delta = np.zeros(6)
            for ch in held:
                if ch in KEY_DELTAS:
                    kb_delta += KEY_DELTAS[ch]

            if np.any(kb_delta != 0):
                cmd_pos = np.clip(cmd_pos + kb_delta[:3], WS_LO, WS_HI)
                if np.any(kb_delta[3:] != 0):
                    cmd_quat_xyzw = (
                        Rotation.from_rotvec(kb_delta[3:]) * Rotation.from_quat(cmd_quat_xyzw)
                    ).as_quat()
                # Push keyboard-driven target back to viser handle so they stay in sync
                transform_ctrl.position = tuple(float(v) for v in cmd_pos)
                transform_ctrl.wxyz     = (float(cmd_quat_xyzw[3]), float(cmd_quat_xyzw[0]),
                                            float(cmd_quat_xyzw[1]), float(cmd_quat_xyzw[2]))
            else:
                # No key held — read from viser handle (user may be dragging it)
                p = transform_ctrl.position
                w = transform_ctrl.wxyz
                cmd_pos       = np.array([p[0], p[1], p[2]])
                cmd_quat_xyzw = np.array([w[1], w[2], w[3], w[0]])   # wxyz → xyzw

            _np(shm_command)[:3] = cmd_pos
            _np(shm_command)[3:] = cmd_quat_xyzw

            # ── Sync gains ─────────────────────────────────────────────────
            _np(shm_gains)[:] = _pack_gains(
                sl_kp_pos.value, sl_kd_pos.value,
                sl_kp_ori.value, sl_kd_ori.value,
                sl_post_kp.value, sl_post_kd.value, sl_post_w.value,
            )

            # ── EE states ──────────────────────────────────────────────────
            q_sim_cur  = _np(shm_q_sim).copy()
            q_real_cur = _np(shm_q_real).copy()
            ee_pos_sim,  ee_rot_sim  = robot_main.fk(q_sim_cur)
            ee_pos_real, ee_rot_real = robot_main.fk(q_real_cur)
            ee_quat_sim  = Rotation.from_matrix(ee_rot_sim).as_quat()
            ee_quat_real = Rotation.from_matrix(ee_rot_real).as_quat()

            # EE frames
            def _wxyz(q): return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
            if cb_ee_frames.value:
                frame_ee_sim.position  = tuple(float(v) for v in ee_pos_sim)
                frame_ee_sim.wxyz      = _wxyz(ee_quat_sim)
                frame_ee_real.position = tuple(float(v) for v in ee_pos_real)
                frame_ee_real.wxyz     = _wxyz(ee_quat_real)

            # Action arrows — only recompute mesh when visible
            raw_delta_sim  = _np(shm_action_sim)[:3].copy()
            eff_delta_sim  = _np(shm_action_sim)[3:].copy()
            raw_delta_real = _np(shm_action_real)[:3].copy()
            eff_delta_real = _np(shm_action_real)[3:].copy()
            if cb_sim_raw.value:
                raw_arrow_sim.vertices  = _make_arrow_mesh(ee_pos_sim,  ee_pos_sim  + raw_delta_sim)[0]
            if cb_sim_eff.value:
                eff_arrow_sim.vertices  = _make_arrow_mesh(ee_pos_sim,  ee_pos_sim  + eff_delta_sim)[0]
            if cb_real_raw.value:
                raw_arrow_real.vertices = _make_arrow_mesh(ee_pos_real, ee_pos_real + raw_delta_real)[0]
            if cb_real_eff.value:
                eff_arrow_real.vertices = _make_arrow_mesh(ee_pos_real, ee_pos_real + eff_delta_real)[0]

            # Joint angles
            q_sim_deg  = np.rad2deg(q_sim_cur)
            q_real_deg = np.rad2deg(q_real_cur)
            for i, txt in enumerate(txt_joints):
                err = q_real_deg[i] - q_sim_deg[i]
                txt.value = f"{q_real_deg[i]:+7.2f}° | {q_sim_deg[i]:+7.2f}° | {err:+6.2f}°"

            # HUD
            pos_err_sim  = float(np.linalg.norm(cmd_pos - ee_pos_sim))
            pos_err_real = float(np.linalg.norm(cmd_pos - ee_pos_real))
            txt_policy_hz.value = f"{_np(shm_policy_hz)[0]:.0f} Hz"
            txt_osc_hz.value    = f"{_np(shm_osc_hz)[0]:.0f} Hz"
            txt_real_hz.value   = f"{_np(shm_real_hz)[0]:.0f} Hz"
            txt_viz_hz.value    = f"{_np(shm_viz_hz)[0]:.0f} Hz"
            txt_cmd.value       = f"x={cmd_pos[0]:.3f}  y={cmd_pos[1]:.3f}  z={cmd_pos[2]:.3f}"
            txt_ee_sim.value    = f"x={ee_pos_sim[0]:.3f}  y={ee_pos_sim[1]:.3f}  z={ee_pos_sim[2]:.3f}"
            txt_ee_real.value   = f"x={ee_pos_real[0]:.3f}  y={ee_pos_real[1]:.3f}  z={ee_pos_real[2]:.3f}"
            txt_err_sim.value   = f"{pos_err_sim*100:.1f} cm"
            txt_err_real.value  = f"{pos_err_real*100:.1f} cm"

            _print_step += 1
            if _print_step % 10 == 0:
                print(
                    f"\rcmd [{cmd_pos[0]:+.3f} {cmd_pos[1]:+.3f} {cmd_pos[2]:+.3f}]  "
                    f"sim  ee [{ee_pos_sim[0]:+.3f} {ee_pos_sim[1]:+.3f} {ee_pos_sim[2]:+.3f}] err {pos_err_sim*100:.1f}cm  "
                    f"real ee [{ee_pos_real[0]:+.3f} {ee_pos_real[1]:+.3f} {ee_pos_real[2]:+.3f}] err {pos_err_real*100:.1f}cm  "
                    f"policy {_np(shm_policy_hz)[0]:.0f}Hz  osc {_np(shm_osc_hz)[0]:.0f}Hz  real {_np(shm_real_hz)[0]:.0f}Hz",
                    end="", flush=True,
                )

            time.sleep(1.0 / TARGET_HZ)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        kb_listener.stop()
        stop_event.set()
        policy_proc.join(timeout=2.0)
        if real_proc is not None:
            real_proc.join(timeout=10.0)
        server.stop()
        print("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
