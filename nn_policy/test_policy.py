"""Kinova Gen3 sim policy tester.

An NN policy (RSL-RL PPO checkpoint) drives the arm; the target *command* is
set interactively via a viser transform-controls handle.  OSC converts the
policy's 6-D action into joint torques at 500 Hz.

Policy observations (38-D, same as reach_osc training):
    joint_pos(7) + joint_vel(7) + ee_to_target(6) + ee_pose(6) +
    target_pose(6) + last_action(6)

Policy action (6-D):
    [delta_pos(3), delta_ori_axis_angle(3)]
    scaled by DELTA_POS_SCALE / DELTA_ORI_SCALE before applying to EE pose.

Processes / Threads:
  - Policy process (mp.Process): loads policy, runs inference at TARGET_HZ Hz
  - OSC thread    (threading):   500 Hz torque computation + physics (main process)
  - Viz thread    (threading):   30 Hz scene update
  - Main thread:                 viser polling + GUI

Run:
  python test_policy.py --checkpoint weights/model_999.pt
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

from kinova_tasks.assets.kinova_gen3.kinova_constants import KINOVA_GEN3_GRIPPER_XML, get_assets
from policy import PolicyAgent
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

# ── Action scales — must match the training environment (reach_osc.py) ────────
DELTA_POS_SCALE = 0.02   # metres per unit action
DELTA_ORI_SCALE = 0.02   # rad per unit action

# ── Workspace bounds — centred on home EE pos from reach_osc.py ──────────────
_HOME_EE_POS = np.array([0.733607, -0.024850, 0.523015])
_WS_RADIUS   = 0.15   # ± 15 cm per axis
WS_LO = _HOME_EE_POS - _WS_RADIUS
WS_HI = _HOME_EE_POS + _WS_RADIUS

# ── Timing ────────────────────────────────────────────────────────────────────
PHYSICS_DT   = 0.002
TARGET_HZ    = 10
OSC_HZ       = 500
VIZ_HZ       = 30

HOME_DEG = np.array([0.0, 20.0, 0.0, 100.0, 0.0, -30.0, -90.0])

# ── Shared memory helpers ─────────────────────────────────────────────────────
def _np(shm: mp.Array) -> np.ndarray:
    """Zero-copy numpy view of a multiprocessing double Array."""
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


# ── Gains ─────────────────────────────────────────────────────────────────────
GAINS_KEYS = ["kp_pos", "kd_pos", "kp_ori", "kd_ori",
              "posture_kp", "posture_kd", "posture_weight"]


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

    # Orthonormal frame
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

    # vertex index layout: shaft_bot(n) | shaft_top(n) | head_base(n) | tip(1) | ctr_bot(1) | ctr_stop(1)
    verts = np.vstack([shaft_bot, shaft_top, head_base, head_tip, ctr_bot, ctr_stop]).astype(np.float32)
    tip_i  = 3 * n
    cbot_i = 3 * n + 1
    csto_i = 3 * n + 2

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([cbot_i, j,      i     ])           # bottom cap
        faces.append([i,      n + i,  j     ])           # shaft side
        faces.append([j,      n + i,  n + j ])
        faces.append([csto_i, n + i,  n + j ])           # shaft top cap
        faces.append([2*n+i,  tip_i,  2*n+j ])           # cone side

    return verts, np.array(faces, dtype=np.uint32)


# ── Observation builder ───────────────────────────────────────────────────────

def build_obs(q, dq, ee_pos, ee_rot, cmd_pos, cmd_quat_xyzw, last_action) -> np.ndarray:
    """Build 38-D observation matching the reach_osc training environment.

    Layout:
        joint_pos(7) + joint_vel(7) + ee_to_target(6) + ee_pose(6) +
        target_pose(6) + last_action(6)

    All quats in this file are xyzw (scipy convention).
    axis_angle = rotvec from scipy Rotation.
    """
    # ee_pose: [pos(3), axis_angle(3)]
    ee_aa        = Rotation.from_matrix(ee_rot).as_rotvec()
    ee_pose_obs  = np.concatenate([ee_pos, ee_aa])

    # target_pose: [pos(3), axis_angle(3)]
    tgt_aa       = Rotation.from_quat(cmd_quat_xyzw).as_rotvec()
    tgt_pose_obs = np.concatenate([cmd_pos, tgt_aa])

    # ee_to_target: [pos_error(3), rot_error_axis_angle(3)]
    pos_err = cmd_pos - ee_pos
    R_tgt   = Rotation.from_quat(cmd_quat_xyzw).as_matrix()
    rot_err = Rotation.from_matrix(R_tgt @ ee_rot.T).as_rotvec()
    ee_to_target_obs = np.concatenate([pos_err, rot_err])

    return np.concatenate([
        q, dq, ee_to_target_obs, ee_pose_obs, tgt_pose_obs, last_action,
    ]).astype(np.float32)


# ── Policy process ────────────────────────────────────────────────────────────

def policy_process_fn(checkpoint_path, device_str,
                      shm_q, shm_dq, shm_command,
                      shm_osc_target, shm_action, shm_policy_hz,
                      policy_reset_event, stop_event, torque_xml):
    """Separate process: loads policy, runs inference at TARGET_HZ."""
    policy_agent = PolicyAgent(checkpoint_path, device=device_str)
    robot        = PinocchioArm(str(torque_xml), ee_frame="pinch_site")
    last_action  = np.zeros(6, dtype=np.float32)
    period       = 1.0 / TARGET_HZ
    iters        = 0
    t_rate       = time.time()

    while not stop_event.is_set():
        t0 = time.time()

        if policy_reset_event.is_set():
            policy_reset_event.clear()
            last_action[:] = 0.0
            iters  = 0
            t_rate = time.time()

        q   = _np(shm_q).copy()
        dq  = _np(shm_dq).copy()
        cmd = _np(shm_command).copy()
        cmd_pos, cmd_quat_xyzw = cmd[:3], cmd[3:]

        ee_pos, ee_rot = robot.fk(q)
        obs    = build_obs(q, dq, ee_pos, ee_rot, cmd_pos, cmd_quat_xyzw, last_action)
        action = policy_agent.get_action(obs)
        last_action = action.copy()

        # Convert action → absolute OSC target (with workspace clipping)
        osc_tgt_pos  = ee_pos + action[:3] * DELTA_POS_SCALE
        osc_tgt_pos  = np.clip(osc_tgt_pos, WS_LO, WS_HI)
        _np(shm_action)[:3] = action[:3] * DELTA_POS_SCALE    # raw delta
        _np(shm_action)[3:] = osc_tgt_pos - ee_pos            # effective delta

        delta_rot    = Rotation.from_rotvec(action[3:] * DELTA_ORI_SCALE)
        osc_tgt_quat = (delta_rot * Rotation.from_matrix(ee_rot)).as_quat()  # xyzw

        _np(shm_osc_target)[:3] = osc_tgt_pos
        _np(shm_osc_target)[3:] = osc_tgt_quat

        iters += 1
        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            _np(shm_policy_hz)[0] = iters / dt_rate
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)


# ── OSC thread ────────────────────────────────────────────────────────────────

def osc_thread_fn(mj_model, mj_data, robot,
                  shm_q, shm_dq, shm_osc_target, shm_gains,
                  shm_osc_hz, osc_reset_event, stop_event,
                  posture_target, home_qpos, arm_q_idxs, arm_dq_idxs, arm_ctrl_idxs):
    """Thread in main process: 500 Hz OSC + physics via direct MuJoCo."""
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

        # Publish current state for policy process
        _np(shm_q)[:] = q_sim
        _np(shm_dq)[:] = dq_sim

        # Read latest OSC target written by policy process
        tgt               = _np(shm_osc_target).copy()
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

def viz_thread_fn(sim_view, mj_model_viz, mj_data_viz,
                  arm_q_idxs, shm_q, shm_viz_hz, stop_event):
    period = 1.0 / VIZ_HZ
    iters  = 0
    t_rate = time.time()

    while not stop_event.is_set():
        t = time.time()

        mj_data_viz.qpos[arm_q_idxs] = _np(shm_q)
        mujoco.mj_kinematics(mj_model_viz, mj_data_viz)
        sim_view.update(mj_data_viz)

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
    parser.add_argument("--checkpoint", default="weights/model_4999.pt",
                        help="Path to RSL-RL PPO checkpoint (.pt)")
    parser.add_argument("--no-gripper", action="store_true")
    args = parser.parse_args()

    _TORQUE_XML = _TORQUE_XML_NO_GRIPPER if args.no_gripper else _TORQUE_XML_GRIPPER

    policy_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── Shared memory ──────────────────────────────────────────────────────
    shm_command    = mp.Array(ctypes.c_double, 7)   # pos(3) + quat_xyzw(4)
    shm_gains      = mp.Array(ctypes.c_double, len(GAINS_KEYS))
    shm_q          = mp.Array(ctypes.c_double, 7)   # joint positions
    shm_dq         = mp.Array(ctypes.c_double, 7)   # joint velocities
    shm_osc_target = mp.Array(ctypes.c_double, 7)   # OSC target: pos(3)+quat_xyzw(4)
    shm_osc_hz     = mp.Array(ctypes.c_double, 1)
    shm_policy_hz  = mp.Array(ctypes.c_double, 1)
    shm_viz_hz     = mp.Array(ctypes.c_double, 1)
    shm_action     = mp.Array(ctypes.c_double, 6)   # [:3] raw delta, [3:] clipped delta

    _np(shm_gains)[:] = _pack_gains(KP_POS, KD_POS, KP_ORI, KD_ORI,
                                     POSTURE_KP, POSTURE_KD, POSTURE_WEIGHT)

    stop_event         = mp.Event()
    osc_reset_event    = mp.Event()
    policy_reset_event = mp.Event()

    # ── Home pose (CPU-only Pinocchio, must run before policy process spawn) ──
    print("Loading Pinocchio…")
    robot_main     = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    home_rad       = kinova_deg_to_rad(HOME_DEG)
    posture_target = home_rad
    ee_pos0, ee_rot0 = robot_main.fk(home_rad)
    q_xyzw0          = Rotation.from_matrix(ee_rot0).as_quat()   # xyzw
    _np(shm_q)[:] = home_rad
    _np(shm_command)[:3] = ee_pos0
    _np(shm_command)[3:] = q_xyzw0
    _np(shm_osc_target)[:3] = ee_pos0
    _np(shm_osc_target)[3:] = q_xyzw0
    print("  OK")

    # ── Spawn policy process (before any CUDA init in main process) ───────
    policy_proc = mp.Process(
        target=policy_process_fn,
        args=(args.checkpoint, policy_device,
              shm_q, shm_dq, shm_command,
              shm_osc_target, shm_action, shm_policy_hz,
              policy_reset_event, stop_event, _TORQUE_XML),
        daemon=True,
    )
    policy_proc.start()
    print(f"[main] Policy process PID: {policy_proc.pid}")

    # ── MuJoCo sim (physics) ───────────────────────────────────────────────
    def _compile_model():
        spec = mujoco.MjSpec.from_file(str(_TORQUE_XML))
        spec.assets = get_assets(spec.meshdir)
        return spec.compile()

    mj_model     = _compile_model()
    mj_data_phys = mujoco.MjData(mj_model)
    mj_data_viz  = mujoco.MjData(mj_model)   # separate data for viz thread

    arm_q_idxs    = np.array([mj_model.joint(n).qposadr.item() for n in _ARM_JOINT_NAMES])
    arm_dq_idxs   = np.array([mj_model.joint(n).dofadr.item()  for n in _ARM_JOINT_NAMES])
    arm_ctrl_idxs = np.array([mj_model.actuator(n).id          for n in _ARM_JOINT_NAMES])

    mj_data_phys.qpos[arm_q_idxs] = home_rad
    mujoco.mj_forward(mj_model, mj_data_phys)

    mj_data_viz.qpos[arm_q_idxs] = home_rad
    mujoco.mj_kinematics(mj_model, mj_data_viz)

    robot_sim = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")  # OSC thread

    # ── Viser ──────────────────────────────────────────────────────────────
    server   = viser.ViserServer(label="Kinova Sim Policy")
    scene    = ViserMujocoScene.create(server, mj_model)
    sim_view = scene.add_robot("sim", color=(0.75, 0.75, 0.75, 1.00))
    scene.create_visualization_gui(camera_distance=1.2, camera_azimuth=135.0, camera_elevation=30.0)

    # EE coordinate frame (updated each main loop tick)
    ee_q_xyzw0 = Rotation.from_matrix(ee_rot0).as_quat()
    frame_ee = server.scene.add_frame(
        "/ee_frame",
        position=(float(ee_pos0[0]), float(ee_pos0[1]), float(ee_pos0[2])),
        wxyz=(float(ee_q_xyzw0[3]), float(ee_q_xyzw0[0]),
              float(ee_q_xyzw0[1]), float(ee_q_xyzw0[2])),
        axes_length=0.10,
        axes_radius=0.004,
    )

    # Raw policy action arrow  (orange) — before workspace clipping
    _av0, _af0 = _make_arrow_mesh(ee_pos0, ee_pos0 + np.array([0.02, 0., 0.]))
    raw_action_arrow = server.scene.add_mesh_simple(
        "/raw_action_arrow",
        vertices=_av0,
        faces=_af0,
        color=(1.0, 0.45, 0.0),
        side="double",
    )
    # Clipped action arrow  (yellow) — after workspace clipping
    action_arrow = server.scene.add_mesh_simple(
        "/action_arrow",
        vertices=_av0.copy(),
        faces=_af0,
        color=(1.0, 0.85, 0.0),
        side="double",
    )

    # Workspace bounding box — 12 edges as line segments
    _lo, _hi = WS_LO, WS_HI
    _corners = np.array([
        [_lo[0], _lo[1], _lo[2]], [_hi[0], _lo[1], _lo[2]],
        [_lo[0], _hi[1], _lo[2]], [_hi[0], _hi[1], _lo[2]],
        [_lo[0], _lo[1], _hi[2]], [_hi[0], _lo[1], _hi[2]],
        [_lo[0], _hi[1], _hi[2]], [_hi[0], _hi[1], _hi[2]],
    ], dtype=np.float32)
    _edges = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    server.scene.add_line_segments(
        "/workspace_bounds",
        points=np.array([[_corners[a], _corners[b]] for a, b in _edges], dtype=np.float32),
        colors=np.array([0.8, 0.8, 0.2], dtype=np.float32),   # yellow-ish
        line_width=1.5,
    )

    # Interactive transform-controls handle: drag to set command target
    transform_ctrl = server.scene.add_transform_controls(
        "/command_target",
        position=(float(ee_pos0[0]), float(ee_pos0[1]), float(ee_pos0[2])),
        wxyz=(float(q_xyzw0[3]), float(q_xyzw0[0]), float(q_xyzw0[1]), float(q_xyzw0[2])),
        scale=0.15,
    )

    with server.gui.add_folder("Policy"):
        txt_policy_hz = server.gui.add_text(f"Policy rate (target {TARGET_HZ} Hz)", initial_value="— Hz")
        txt_osc_hz    = server.gui.add_text(f"OSC rate (target {OSC_HZ} Hz)",       initial_value="— Hz")
        txt_viz_hz    = server.gui.add_text(f"Viz rate (target {VIZ_HZ} Hz)",       initial_value="— Hz")
        txt_cmd       = server.gui.add_text("Command XYZ", initial_value="—")
        txt_ee        = server.gui.add_text("EE XYZ",      initial_value="—")
        txt_err       = server.gui.add_text("Pos error",   initial_value="—")
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

    def _on_reset(_):
        osc_reset_event.set()
        policy_reset_event.set()
        _np(shm_command)[:3] = ee_pos0
        _np(shm_command)[3:] = q_xyzw0
        _np(shm_osc_target)[:3] = ee_pos0
        _np(shm_osc_target)[3:] = q_xyzw0
        transform_ctrl.position = (float(ee_pos0[0]), float(ee_pos0[1]), float(ee_pos0[2]))
        transform_ctrl.wxyz     = (float(q_xyzw0[3]), float(q_xyzw0[0]),
                                    float(q_xyzw0[1]), float(q_xyzw0[2]))

    reset_btn.on_click(_on_reset)

    threading.Thread(target=osc_thread_fn, daemon=True, args=(
        mj_model, mj_data_phys, robot_sim,
        shm_q, shm_dq, shm_osc_target, shm_gains,
        shm_osc_hz, osc_reset_event, stop_event,
        posture_target, home_rad, arm_q_idxs, arm_dq_idxs, arm_ctrl_idxs,
    )).start()

    threading.Thread(target=viz_thread_fn, daemon=True, args=(
        sim_view, mj_model, mj_data_viz,
        arm_q_idxs, shm_q, shm_viz_hz, stop_event,
    )).start()

    print("Running — drag the viser transform handle to move the command target.")
    print("Ctrl+C to stop.\n")
    try:
        _print_step = 0
        while True:
            # ── Read command target from viser ─────────────────────────────
            p = transform_ctrl.position   # (x, y, z)
            w = transform_ctrl.wxyz       # (w, x, y, z)
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

            # ── EE state + action vis ──────────────────────────────────────
            q_sim = _np(shm_q).copy()
            ee_pos, ee_rot = robot_main.fk(q_sim)
            ee_quat_xyzw = Rotation.from_matrix(ee_rot).as_quat()
            pos_err_m = float(np.linalg.norm(cmd_pos - ee_pos))

            # EE frame axes
            frame_ee.position = (float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2]))
            frame_ee.wxyz     = (float(ee_quat_xyzw[3]), float(ee_quat_xyzw[0]),
                                 float(ee_quat_xyzw[1]), float(ee_quat_xyzw[2]))

            # Raw action arrow (orange): EE → EE + raw_delta
            raw_delta = _np(shm_action)[:3].copy()
            raw_action_arrow.vertices = _make_arrow_mesh(ee_pos, ee_pos + raw_delta)[0]
            # Clipped action arrow (yellow): EE → EE + effective_delta
            eff_delta = _np(shm_action)[3:].copy()
            action_arrow.vertices = _make_arrow_mesh(ee_pos, ee_pos + eff_delta)[0]

            txt_policy_hz.value = f"{_np(shm_policy_hz)[0]:.0f} Hz"
            txt_osc_hz.value    = f"{_np(shm_osc_hz)[0]:.0f} Hz"
            txt_viz_hz.value    = f"{_np(shm_viz_hz)[0]:.0f} Hz"
            txt_cmd.value    = f"x={cmd_pos[0]:.3f}  y={cmd_pos[1]:.3f}  z={cmd_pos[2]:.3f}"
            txt_ee.value     = f"x={ee_pos[0]:.3f}  y={ee_pos[1]:.3f}  z={ee_pos[2]:.3f}"
            txt_err.value    = f"{pos_err_m*100:.1f} cm"

            _print_step += 1
            if _print_step % 10 == 0:
                print(
                    f"\rcmd [{cmd_pos[0]:+.3f} {cmd_pos[1]:+.3f} {cmd_pos[2]:+.3f}]  "
                    f"ee  [{ee_pos[0]:+.3f} {ee_pos[1]:+.3f} {ee_pos[2]:+.3f}]  "
                    f"err {pos_err_m*100:.1f} cm  "
                    f"policy {_np(shm_policy_hz)[0]:.0f} Hz  osc {_np(shm_osc_hz)[0]:.0f} Hz",
                    end="", flush=True,
                )

            time.sleep(1.0 / TARGET_HZ)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        stop_event.set()
        policy_proc.join(timeout=2.0)
        server.stop()
        print("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
