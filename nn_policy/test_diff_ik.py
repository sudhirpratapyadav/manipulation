"""Kinova Gen3 sim2real OSC — multiprocessing edition.

Real robot runs in a separate process so the sim thread has the GIL to itself.
Communication via multiprocessing shared-memory arrays (no serialisation overhead).

Processes / threads:
  - Real process : outer loop (TARGET_HZ=50) snapshots target; inner loop (OSC_HZ=500)
                   recomputes OSC torques from current state + sends to hardware each 2ms
  - Main process:
      - Sim thread  : outer loop (TARGET_HZ=50) snapshots target; inner loop (OSC_HZ=500)
                      recomputes OSC torques from fresh sim state + steps physics each 2ms
      - Viz thread  (30 Hz)
      - Main thread : keyboard target + gains poll (TARGET_HZ=50)

Target is controlled via keyboard (relative increments):
  Translation : w/s (+/-Y)  a/d (+/-X)  q/e (+/-Z)
  Rotation    : i/k (+/-Rx) j/l (+/-Ry) u/o (+/-Rz)

Viser shows the current target position (read-only display).

Startup order in main():
  1. Create shared memory + events
  2. Spawn real process        ← BEFORE any GPU/CUDA init (clean fork)
  3. Init sim (GPU)
  4. Init Pinocchio, viser
  5. Start sim + viz threads
  6. Main loop

Run:
  python kinova_sim2real_osc_relative.py --ip 192.168.1.10
"""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pinocchio as pin
import torch
import viser
from pynput import keyboard as pynput_kb
from scipy.spatial.transform import Rotation

from hardware import KinovaHardware
from kinova_tasks.assets.kinova_gen3.kinova_constants import KINOVA_GEN3_GRIPPER_XML, get_assets
from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions.actions import JointEffortActionCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
from viewer import ViserMujocoScene

# ── Model ─────────────────────────────────────────────────────────────────────
_TORQUE_XML      = KINOVA_GEN3_GRIPPER_XML.parent / "gen3_torque_no_gripper.xml"
_ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]

# ── Sim initial state — matches HOME_DEG ──────────────────────────────────────
DEMO_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "joint_1": 0.0,
        "joint_2": 0.52359878,   # 30°
        "joint_3": 0.0,
        "joint_4": 1.57079633,   # 90°
        "joint_5": 0.0,
        "joint_6": 1.04719755,   # 60°
        "joint_7": -1.57079633,  # -90°
    },
    joint_vel={".*": 0.0},
)

# ── Default gains ─────────────────────────────────────────────────────────────
KP_POS         =  50.0
KD_POS         =   2.0
KP_ORI         =  10.0
KD_ORI         =   2.0
POSTURE_KP     =  10.0
POSTURE_KD     =   2.0
POSTURE_WEIGHT =   0.01

MAX_JOINT_TORQUE = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])
TAU_OFFSETS_DEFAULT = [0.0, 0.0, -0.5, 0.0, 0.0, 1.0, 0.0]

# ── Timing ────────────────────────────────────────────────────────────────────
PHYSICS_DT   = 0.002          # sim physics timestep (s)
TARGET_HZ    = 50             # outer loop: target update from keyboard
OSC_HZ       = 500            # inner loop: OSC torque compute + hardware send
OSC_SUBSTEPS = OSC_HZ // TARGET_HZ   # inner iterations per outer tick (10)
VIZ_HZ       = 30

HOME_DEG = np.array([0.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])

# ── Keyboard control ───────────────────────────────────────────────────────────
DELTA_POS = 0.02           # metres per main-loop tick while key is held
DELTA_ROT = np.deg2rad(1.0) # radians per main-loop tick while key is held

# 12 keys: ±X, ±Y, ±Z, ±Rx, ±Ry, ±Rz  (world frame)
KEY_DELTAS: dict[str, np.ndarray] = {
    "w": np.array([ DELTA_POS, 0, 0, 0, 0, 0]),
    "s": np.array([-DELTA_POS, 0, 0, 0, 0, 0]),
    "a": np.array([0,  DELTA_POS, 0, 0, 0, 0]),
    "d": np.array([0, -DELTA_POS, 0, 0, 0, 0]),
    "e": np.array([0, 0,  DELTA_POS, 0, 0, 0]),
    "q": np.array([0, 0, -DELTA_POS, 0, 0, 0]),
    "i": np.array([0, 0, 0,  DELTA_ROT, 0, 0]),
    "k": np.array([0, 0, 0, -DELTA_ROT, 0, 0]),
    "j": np.array([0, 0, 0, 0,  DELTA_ROT, 0]),
    "l": np.array([0, 0, 0, 0, -DELTA_ROT, 0]),
    "u": np.array([0, 0, 0, 0, 0,  DELTA_ROT]),
    "o": np.array([0, 0, 0, 0, 0, -DELTA_ROT]),
}

_held_keys: set[str] = set()
_held_keys_lock = threading.Lock()


def _on_key_press(key):
    try:
        with _held_keys_lock:
            _held_keys.add(key.char)
    except AttributeError:
        pass  # ignore special keys


def _on_key_release(key):
    try:
        with _held_keys_lock:
            _held_keys.discard(key.char)
    except AttributeError:
        pass

# ── Shared memory layout ──────────────────────────────────────────────────────
# shm_target : target_pos(3) + target_quat_xyzw(4)
# shm_gains  : gains in GAINS_KEYS order
GAINS_KEYS = ["kp_pos", "kd_pos", "kp_ori", "kd_ori",
              "posture_kp", "posture_kd", "posture_weight"]


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

        # Mass matrix (symmetrize — crba fills upper triangle only)
        pin.crba(self.model, self.data, self._q_full)
        M_sub = self.data.M[np.ix_(self._v_idx, self._v_idx)]
        M_arm = 0.5 * (M_sub + M_sub.T)

        # Nonlinear effects: C(q, dq)·dq + g(q)
        pin.nonLinearEffects(self.model, self.data, self._q_full, self._dq_full)
        nle_arm = self.data.nle[self._v_idx].copy()

        # Jacobian time derivative · dq  (acceleration bias)
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

    J      = robot.jacobian(q)        # (6, 7)
    ee_vel = J @ dq                   # (6,)

    ddx_des = np.empty(6)
    ddx_des[:3] = gains["kp_pos"] * error[:3] + gains["kd_pos"] * (0.0 - ee_vel[:3])
    ddx_des[3:] = gains["kp_ori"] * error[3:] + gains["kd_ori"] * (0.0 - ee_vel[3:])

    M, nle, J_dot_dq = robot.dynamics(q, dq)

    M_inv     = np.linalg.inv(M)
    Lambda    = np.linalg.inv(J @ M_inv @ J.T)   # (6, 6)
    J_dyn_inv = M_inv @ J.T @ Lambda              # (7, 6)

    F = Lambda @ (ddx_des - J_dot_dq)             # (6,)

    N           = np.eye(7) - J.T @ J_dyn_inv.T   # (7, 7) null-space projector
    tau_posture = gains["posture_kp"] * (posture_target - q) + gains["posture_kd"] * (0.0 - dq)

    tau = J.T @ F + nle + gains["posture_weight"] * (N @ tau_posture)
    return np.clip(tau, -MAX_JOINT_TORQUE, MAX_JOINT_TORQUE)


# ── Real process (separate process — no GIL sharing) ─────────────────────────

def real_process_fn(ip,
                    shm_q_real, shm_target_real, shm_gains, shm_real_hz,
                    stop_event, reset_event, reset_done_event):
    outer_dt   = 1.0 / TARGET_HZ
    inner_dt   = 1.0 / OSC_HZ
    robot      = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
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

        iters  = 0
        t_rate = time.time()

        while not stop_event.is_set():
            t_outer = time.time()

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
                iters   = 0
                t_rate  = time.time()
                reset_done_event.set()
                continue

            # Snapshot target + gains once per outer tick
            target = _np(shm_target_real).copy()
            gains  = _gains_dict(_np(shm_gains).copy())

            # ── Inner loop: OSC at 500 Hz ──────────────────────────────────
            for _ in range(OSC_SUBSTEPS):
                t_inner = time.time()

                q  = kinova_deg_to_rad(pos_deg)
                dq = np.deg2rad(vel_deg)
                _np(shm_q_real)[:] = q

                tau = compute_osc_torques(
                    robot, target[:3], target[3:], q, dq,
                    gains=gains, posture_target=posture_target,
                )
                tau += TAU_OFFSETS_DEFAULT  # feedforward offsets to help compensate gravity/friction

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


# ── Sim thread ────────────────────────────────────────────────────────────────

def sim_thread_fn(entity, sim, effort_action, robot, arm_ids, device,
                  shm_target_sim, shm_gains, shm_sim_hz,
                  stop_event, reset_done_event, posture_target):
    outer_dt = 1.0 / TARGET_HZ
    inner_dt = 1.0 / OSC_HZ
    iters    = 0
    t_rate   = time.time()

    while not stop_event.is_set():
        t_outer = time.time()

        if reset_done_event.is_set():
            reset_done_event.clear()
            entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
            sim.forward()
            effort_action.reset()
            iters  = 0
            t_rate = time.time()
            print("[sim] Reset done")

        # Snapshot target + gains once per outer tick
        target = _np(shm_target_sim).copy()
        gains  = _gains_dict(_np(shm_gains).copy())

        # ── Inner loop: OSC + physics at 500 Hz ───────────────────────────
        for _ in range(OSC_SUBSTEPS):
            t_inner = time.time()

            q_sim  = entity.data.joint_pos[0, arm_ids].cpu().numpy()
            dq_sim = entity.data.joint_vel[0, arm_ids].cpu().numpy()

            tau_sim = compute_osc_torques(
                robot, target[:3], target[3:], q_sim, dq_sim,
                gains=gains, posture_target=posture_target,
            )

            tau_t = torch.from_numpy(tau_sim).float().to(device).unsqueeze(0)  # (1, 7)
            effort_action.process_actions(tau_t)
            effort_action.apply_actions()
            entity.write_data_to_sim()
            sim.step()

            iters += 1
            elapsed_inner = time.time() - t_inner
            if elapsed_inner < inner_dt:
                time.sleep(inner_dt - elapsed_inner)

        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            _np(shm_sim_hz)[0] = iters / dt_rate
            iters  = 0
            t_rate = time.time()


# ── Viz thread ────────────────────────────────────────────────────────────────
# Two CPU mj_data objects: one for sim, one for real.
# mj_kinematics() is called at VIZ_HZ to update body xpos/xmat for each.

def viz_thread_fn(sim_view, real_view, mj_model_cpu, mj_data_sim, mj_data_real,
                  entity_ctrl, arm_ids, arm_q_idxs, shm_q_real, stop_event):
    period = 1.0 / VIZ_HZ

    while not stop_event.is_set():
        t = time.time()

        q_sim  = entity_ctrl.data.joint_pos[0, arm_ids].cpu().numpy()
        q_real = _np(shm_q_real).copy()

        mj_data_sim.qpos.flat[arm_q_idxs] = q_sim
        mujoco.mj_kinematics(mj_model_cpu, mj_data_sim)
        sim_view.update(mj_data_sim)

        mj_data_real.qpos.flat[arm_q_idxs] = q_real
        mujoco.mj_kinematics(mj_model_cpu, mj_data_real)
        real_view.update(mj_data_real)

        elapsed = time.time() - t
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",         default="192.168.1.10")
    args = parser.parse_args()

    # ── Shared memory ──────────────────────────────────────────────────────
    shm_q_real      = mp.Array(ctypes.c_double, 7)
    shm_target_sim  = mp.Array(ctypes.c_double, 7)  # pos(3) + quat_xyzw(4)
    shm_target_real = mp.Array(ctypes.c_double, 7)  # pos(3) + quat_xyzw(4)
    shm_gains       = mp.Array(ctypes.c_double, len(GAINS_KEYS))
    shm_real_hz     = mp.Array(ctypes.c_double, 1)
    shm_sim_hz      = mp.Array(ctypes.c_double, 1)

    _np(shm_q_real)[:] = kinova_deg_to_rad(HOME_DEG)
    _np(shm_gains)[:]  = _pack_gains(KP_POS, KD_POS, KP_ORI, KD_ORI,
                                     POSTURE_KP, POSTURE_KD, POSTURE_WEIGHT)

    stop_event       = mp.Event()
    reset_event      = mp.Event()
    reset_done_event = mp.Event()

    # ── Spawn real process BEFORE GPU init ─────────────────────────────────
    real_proc = mp.Process(
        target=real_process_fn,
        args=(args.ip,
              shm_q_real, shm_target_real, shm_gains, shm_real_hz,
              stop_event, reset_event, reset_done_event),
        daemon=True,
    )
    real_proc.start()

    # ── Sim (num_envs=1 — fast; real ghost rendered via CPU MuJoCo) ────────
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def get_spec():
        spec = mujoco.MjSpec.from_file(str(_TORQUE_XML))
        spec.assets = get_assets(spec.meshdir)
        return spec

    robot_cfg = EntityCfg(
        init_state=DEMO_INIT_STATE,
        collisions=(),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlMotorActuatorCfg(target_names_expr=("joint_.*",)),),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
    entity_ctrl = Entity(robot_cfg)
    model_ctrl  = entity_ctrl.compile()
    sim_ctrl    = Simulation(num_envs=1, cfg=SimulationCfg(mujoco=MujocoCfg(timestep=PHYSICS_DT, gravity=(0,0,-9.81))), model=model_ctrl, device=device)
    entity_ctrl.initialize(model_ctrl, sim_ctrl.model, sim_ctrl.data, device)
    entity_ctrl.write_joint_position_to_sim(entity_ctrl.data.default_joint_pos, joint_ids=None)
    sim_ctrl.forward()

    env_ns        = SimpleNamespace(num_envs=1, device=device, scene={"robot": entity_ctrl}, sim=sim_ctrl)
    effort_action = JointEffortActionCfg(entity_name="robot", actuator_names=("joint_.*",)).build(env_ns)
    arm_ids       = effort_action.target_ids

    # ── CPU MuJoCo model/data for visualization ────────────────────────────
    mj_model_cpu  = get_spec().compile()
    mj_data_sim   = mujoco.MjData(mj_model_cpu)
    mj_data_real  = mujoco.MjData(mj_model_cpu)
    arm_q_idxs    = np.array([mj_model_cpu.joint(n).qposadr for n in _ARM_JOINT_NAMES])

    home_rad = kinova_deg_to_rad(HOME_DEG)
    mj_data_sim.qpos.flat[arm_q_idxs]  = home_rad
    mj_data_real.qpos.flat[arm_q_idxs] = home_rad
    mujoco.mj_kinematics(mj_model_cpu, mj_data_sim)
    mujoco.mj_kinematics(mj_model_cpu, mj_data_real)

    print("Loading Pinocchio model…")
    robot_sim      = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")  # sim thread
    robot_main     = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")  # main loop FK only
    posture_target = kinova_deg_to_rad(HOME_DEG)
    print("  OK")

    # ── Viser — separate views for sim (grey) and real (blue ghost) ────────
    server    = viser.ViserServer(label="Kinova Sim2Real OSC")
    scene     = ViserMujocoScene.create(server, mj_model_cpu)
    sim_view  = scene.add_robot("sim",  color=(0.75, 0.75, 0.75, 1.00))
    real_view = scene.add_robot("real", color=(0.20, 0.55, 0.90, 0.65))
    scene.create_visualization_gui(camera_distance=1.2, camera_azimuth=135.0, camera_elevation=30.0)

    q_home_rad = kinova_deg_to_rad(HOME_DEG)
    ee_pos0, ee_rot0 = robot_main.fk(q_home_rad)
    q_xyzw0 = Rotation.from_matrix(ee_rot0).as_quat()

    _np(shm_target_sim)[:3]  = ee_pos0
    _np(shm_target_sim)[3:]  = q_xyzw0
    _np(shm_target_real)[:3] = ee_pos0
    _np(shm_target_real)[3:] = q_xyzw0

    # Display-only frames for sim and real targets (no user interaction)
    frame_tgt_sim  = server.scene.add_frame("/target_sim",  axes_length=0.08, axes_radius=0.005)
    frame_tgt_real = server.scene.add_frame("/target_real", axes_length=0.08, axes_radius=0.005)

    with server.gui.add_folder("Sim2Real"):
        txt_sim_hz  = server.gui.add_text("Sim rate",  initial_value="— Hz")
        txt_real_hz = server.gui.add_text("Real rate", initial_value="— Hz")
        txt_tgt_sim  = server.gui.add_text("Sim target XYZ",  initial_value="—")
        txt_tgt_real = server.gui.add_text("Real target XYZ", initial_value="—")
        reset_btn   = server.gui.add_button("Reset Both")

    with server.gui.add_folder("OSC Gains"):
        sl_kp_pos = server.gui.add_slider("Kp pos", min=0.0, max=1000.0, step=1.0,  initial_value=KP_POS)
        sl_kd_pos = server.gui.add_slider("Kd pos", min=0.0, max=200.0,  step=0.5,  initial_value=KD_POS)
        sl_kp_ori = server.gui.add_slider("Kp ori", min=0.0, max=1000.0, step=1.0,  initial_value=KP_ORI)
        sl_kd_ori = server.gui.add_slider("Kd ori", min=0.0, max=200.0,  step=0.5,  initial_value=KD_ORI)

    with server.gui.add_folder("Posture"):
        sl_post_kp = server.gui.add_slider("Posture Kp",     min=0.0, max=100.0, step=0.1,  initial_value=POSTURE_KP)
        sl_post_kd = server.gui.add_slider("Posture Kd",     min=0.0, max=20.0,  step=0.1,  initial_value=POSTURE_KD)
        sl_post_w  = server.gui.add_slider("Posture weight", min=0.0, max=1.0,   step=0.01, initial_value=POSTURE_WEIGHT)
        server.gui.add_text("Posture target (deg)", initial_value=", ".join(f"{v:.0f}" for v in HOME_DEG))

    th_stop = threading.Event()
    reset_btn.on_click(lambda _: reset_event.set())

    kb_listener = pynput_kb.Listener(on_press=_on_key_press, on_release=_on_key_release)
    kb_listener.start()

    print("\nKeyboard control active:")
    print("  Translation  +X/−X: d/a   +Y/−Y: w/s   +Z/−Z: e/q")
    print("  Rotation     +Rx/−Rx: i/k  +Ry/−Ry: j/l  +Rz/−Rz: u/o")
    print("  Ctrl+C to quit\n")

    threading.Thread(target=sim_thread_fn, daemon=True, args=(
        entity_ctrl, sim_ctrl, effort_action, robot_sim, arm_ids, device,
        shm_target_sim, shm_gains, shm_sim_hz,
        th_stop, reset_done_event, posture_target,
    )).start()

    threading.Thread(target=viz_thread_fn, daemon=True, args=(
        sim_view, real_view, mj_model_cpu, mj_data_sim, mj_data_real,
        entity_ctrl, arm_ids, arm_q_idxs, shm_q_real, th_stop,
    )).start()

    print("Running. Ctrl+C to stop.")
    try:
        _print_step = 0
        while True:
            # ── Keyboard delta ─────────────────────────────────────────────
            with _held_keys_lock:
                held = _held_keys.copy()

            delta = np.zeros(6)
            for ch in held:
                if ch in KEY_DELTAS:
                    delta += KEY_DELTAS[ch]

            # ── Sim target: sim_ee + delta ─────────────────────────────────
            q_sim = entity_ctrl.data.joint_pos[0, arm_ids].cpu().numpy()
            sim_ee_pos, sim_ee_rot = robot_main.fk(q_sim)
            sim_tgt_pos = sim_ee_pos + delta[:3]
            if np.any(delta[3:] != 0):
                sim_tgt_quat = (Rotation.from_rotvec(delta[3:]) * Rotation.from_matrix(sim_ee_rot)).as_quat()
            else:
                sim_tgt_quat = Rotation.from_matrix(sim_ee_rot).as_quat()
            _np(shm_target_sim)[:3] = sim_tgt_pos
            _np(shm_target_sim)[3:] = sim_tgt_quat

            # ── Real target: real_ee + delta ───────────────────────────────
            q_real = _np(shm_q_real).copy()
            real_ee_pos, real_ee_rot = robot_main.fk(q_real)
            real_tgt_pos = real_ee_pos + delta[:3]
            if np.any(delta[3:] != 0):
                real_tgt_quat = (Rotation.from_rotvec(delta[3:]) * Rotation.from_matrix(real_ee_rot)).as_quat()
            else:
                real_tgt_quat = Rotation.from_matrix(real_ee_rot).as_quat()
            _np(shm_target_real)[:3] = real_tgt_pos
            _np(shm_target_real)[3:] = real_tgt_quat

            # ── Viser: update target frames ────────────────────────────────
            def _wxyz(q_xyzw):
                return (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))

            frame_tgt_sim.position  = tuple(float(v) for v in sim_tgt_pos)
            frame_tgt_sim.wxyz      = _wxyz(sim_tgt_quat)
            frame_tgt_real.position = tuple(float(v) for v in real_tgt_pos)
            frame_tgt_real.wxyz     = _wxyz(real_tgt_quat)

            # ── Sync gains + HUD ───────────────────────────────────────────
            _np(shm_gains)[:] = _pack_gains(
                sl_kp_pos.value, sl_kd_pos.value,
                sl_kp_ori.value, sl_kd_ori.value,
                sl_post_kp.value, sl_post_kd.value, sl_post_w.value,
            )
            txt_sim_hz.value   = f"{_np(shm_sim_hz)[0]:.0f} Hz"
            txt_real_hz.value  = f"{_np(shm_real_hz)[0]:.0f} Hz"
            txt_tgt_sim.value  = f"x={sim_tgt_pos[0]:.3f}  y={sim_tgt_pos[1]:.3f}  z={sim_tgt_pos[2]:.3f}"
            txt_tgt_real.value = f"x={real_tgt_pos[0]:.3f}  y={real_tgt_pos[1]:.3f}  z={real_tgt_pos[2]:.3f}"

            # ── Terminal print (every 10 steps) ───────────────────────────
            _print_step += 1
            if _print_step % 10 != 0:
                time.sleep(1.0 / TARGET_HZ)
                continue
            delta_pos_mm  = delta[:3] * 1000.0
            delta_rot_deg = np.rad2deg(delta[3:])
            sim_ee_rpy    = Rotation.from_matrix(sim_ee_rot).as_euler("xyz", degrees=True)
            real_ee_rpy   = Rotation.from_matrix(real_ee_rot).as_euler("xyz", degrees=True)
            sim_tgt_rpy   = Rotation.from_quat(sim_tgt_quat).as_euler("xyz", degrees=True)
            real_tgt_rpy  = Rotation.from_quat(real_tgt_quat).as_euler("xyz", degrees=True)
            print(
                f"\r"
                f"Δ [{delta_pos_mm[0]:+5.1f} {delta_pos_mm[1]:+5.1f} {delta_pos_mm[2]:+5.1f}mm "
                f"{delta_rot_deg[0]:+4.1f} {delta_rot_deg[1]:+4.1f} {delta_rot_deg[2]:+4.1f}°]  |  "
                f"SIM  ee [{sim_ee_pos[0]:+.3f} {sim_ee_pos[1]:+.3f} {sim_ee_pos[2]:+.3f}] "
                f"rpy [{sim_ee_rpy[0]:+5.1f} {sim_ee_rpy[1]:+5.1f} {sim_ee_rpy[2]:+5.1f}°]  "
                f"tgt [{sim_tgt_pos[0]:+.3f} {sim_tgt_pos[1]:+.3f} {sim_tgt_pos[2]:+.3f}] "
                f"rpy [{sim_tgt_rpy[0]:+5.1f} {sim_tgt_rpy[1]:+5.1f} {sim_tgt_rpy[2]:+5.1f}°]  |  "
                f"REAL ee [{real_ee_pos[0]:+.3f} {real_ee_pos[1]:+.3f} {real_ee_pos[2]:+.3f}] "
                f"rpy [{real_ee_rpy[0]:+5.1f} {real_ee_rpy[1]:+5.1f} {real_ee_rpy[2]:+5.1f}°]  "
                f"tgt [{real_tgt_pos[0]:+.3f} {real_tgt_pos[1]:+.3f} {real_tgt_pos[2]:+.3f}] "
                f"rpy [{real_tgt_rpy[0]:+5.1f} {real_tgt_rpy[1]:+5.1f} {real_tgt_rpy[2]:+5.1f}°]",
                end="", flush=True,
            )

            time.sleep(1.0 / TARGET_HZ)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        kb_listener.stop()
        th_stop.set()
        stop_event.set()        # signals real process
        real_proc.join(timeout=10.0)
        server.stop()
        print("Done.")


if __name__ == "__main__":
    main()
