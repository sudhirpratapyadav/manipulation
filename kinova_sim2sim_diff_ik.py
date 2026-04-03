"""Kinova Gen3 sim2sim diff-IK — side-by-side comparison.

Sim 1 = inbuilt DifferentialIKAction  (from kinova_sim2real_diff_ik_inbuilt_sim.py)
Sim 2 = Pinocchio torque diff-IK      (from kinova_sim2real_diff_ik.py)

Sim 2 runs in a separate process so each sim gets a clean CUDA context.
The Sim 2 process loop mirrors sim_thread_fn from kinova_sim2real_diff_ik.py exactly,
with the addition of writing shm_q_sim2 for visualisation in the main process.

Processes / threads:
  - Sim 2 process (CONTROL_HZ) : mjlab GPU sim + Pinocchio torque diff-IK
  - Main process:
      - Sim 1 thread (CONTROL_HZ) : inbuilt DifferentialIKAction + sim step
      - Viz thread   (30 Hz)
      - Main thread  : viser target + gains poll (60 Hz)

Startup order:
  1. Shared memory + events
  2. Init shm_target via Pinocchio FK (valid quat before any process reads it)
  3. Spawn Sim 2 process   ← BEFORE any GPU/CUDA init in main process
  4. Init Sim 1 (GPU) + DifferentialIKAction
  5. Init viser
  6. Start Sim 1 + viz threads
  7. Main loop

Run:
  python kinova_sim2sim_diff_ik.py
"""

from __future__ import annotations

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
from scipy.spatial.transform import Rotation

from kinova_tasks.assets.kinova_gen3.kinova_constants import (
    KINOVA_GEN3_GRIPPER_XML,
    KINOVA_GRIPPER_ARTICULATION,
    get_assets,
)
from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions import DifferentialIKAction, DifferentialIKActionCfg
from mjlab.envs.mdp.actions.actions import JointEffortActionCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.utils.lab_api.math import quat_from_matrix
from viewer import ViserMujocoScene

# ── Model ─────────────────────────────────────────────────────────────────────
_TORQUE_XML      = KINOVA_GEN3_GRIPPER_XML.parent / "gen3_gripper_torque.xml"
_ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]


def _get_sim1_spec() -> mujoco.MjSpec:
    """Position-actuator spec for Sim 1 (inbuilt DifferentialIKAction)."""
    spec = mujoco.MjSpec.from_file(str(KINOVA_GEN3_GRIPPER_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


def _get_sim2_spec() -> mujoco.MjSpec:
    """Torque-actuator spec for Sim 2 (Pinocchio diff-IK)."""
    spec = mujoco.MjSpec.from_file(str(_TORQUE_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


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
KP_TASK          = 3.0
KP_JOINT         = 300.0
KD_JOINT         = 20.0
DAMPING          = 0.05
POS_WEIGHT       = 1.0
ORI_WEIGHT       = 1.0
POSTURE_WEIGHT   = 0.0
JNT_LIMIT_WEIGHT = 0.0
MAX_DQ           = 0.5

MAX_JOINT_TORQUE = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])

# ── Timing ────────────────────────────────────────────────────────────────────
PHYSICS_DT   = 0.002
CONTROL_HZ   = 100
SIM_SUBSTEPS = max(1, round(1.0 / (CONTROL_HZ * PHYSICS_DT)))
VIZ_HZ       = 30

HOME_DEG = np.array([0.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])

# ── Shared memory layout ──────────────────────────────────────────────────────
# shm_target  : target_pos(3) + target_quat_xyzw(4)
# shm_gains   : gains in GAINS_KEYS order
# shm_q_sim2  : Sim 2 joint positions (7) for visualisation
GAINS_KEYS = ["kp_task", "kp_joint", "kd_joint", "damping",
              "pos_weight", "ori_weight", "posture_weight", "jnt_limit_weight", "max_dq"]


def _np(shm: mp.Array) -> np.ndarray:
    """Zero-copy numpy view of a multiprocessing double Array."""
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


def _pack_gains(kp_task, kp_joint, kd_joint, damping,
                pos_weight, ori_weight, posture_weight, jnt_limit_weight, max_dq) -> np.ndarray:
    return np.array([kp_task, kp_joint, kd_joint, damping,
                     pos_weight, ori_weight, posture_weight, jnt_limit_weight, max_dq])


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
        self._q_full = pin.neutral(self.model)
        self.joint_lower = self.model.lowerPositionLimit[self._q_idx].copy()
        self.joint_upper = self.model.upperPositionLimit[self._q_idx].copy()

    def _set_q(self, q):
        self._q_full[self._q_idx] = q

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

    def gravity(self, q):
        self._set_q(q)
        pin.computeGeneralizedGravity(self.model, self.data, self._q_full)
        return self.data.g[self._v_idx].copy()


def pose_error_6d(target_pos, target_quat_xyzw, cur_pos, cur_rot):
    pos_err = target_pos - cur_pos
    ori_err = Rotation.from_matrix(
        Rotation.from_quat(target_quat_xyzw).as_matrix() @ cur_rot.T
    ).as_rotvec()
    return np.concatenate([pos_err, ori_err])


def compute_diff_ik_torques(robot, target_pos, target_quat_xyzw, q, dq, *, gains, posture_target,
                           _debug_state={"iters": 0}):
    ee_pos, ee_rot = robot.fk(q)
    error  = pose_error_6d(target_pos, target_quat_xyzw, ee_pos, ee_rot)
    pos_dx = gains["kp_task"] * error[:3]
    ori_dx = gains["kp_task"] * error[3:]

    J    = robot.jacobian(q)
    jacp, jacr = J[:3], J[3:]

    JTJ  = gains["pos_weight"] * (jacp.T @ jacp) + gains["ori_weight"] * (jacr.T @ jacr)
    JTdx = gains["pos_weight"] * (jacp.T @ pos_dx) + gains["ori_weight"] * (jacr.T @ ori_dx)

    r_limit  = np.clip(robot.joint_upper - q, None, 0) + np.clip(robot.joint_lower - q, 0, None)
    violated = (r_limit != 0).astype(float)
    JTJ  += np.diag(gains["jnt_limit_weight"] * violated)
    JTdx += gains["jnt_limit_weight"] * violated * r_limit

    JTJ  += gains["posture_weight"] * np.eye(7)
    JTdx += gains["posture_weight"] * (posture_target - q)

    JTJ  += gains["damping"] ** 2 * np.eye(7)

    dq_des = np.clip(np.linalg.solve(JTJ, JTdx), -gains["max_dq"], gains["max_dq"])

    gravity = robot.gravity(q)
    tau = gains["kp_joint"] * dq_des + gains["kd_joint"] * (0.0 - dq) + gravity
    tau_clipped = np.clip(tau, -MAX_JOINT_TORQUE, MAX_JOINT_TORQUE)

    _debug_state["iters"] += 1
    n = _debug_state["iters"]
    if n <= 3 or n % 300 == 0:
        print(f"[SIM2 IK  #{n:6d}] kp_task={gains['kp_task']:.2f}  "
              f"|pos_err|={np.linalg.norm(error[:3]):.4f}m  "
              f"|ori_err|={np.linalg.norm(error[3:]):.4f}rad")
        print(f"             EE_pos={np.round(ee_pos, 4)}  "
              f"target={np.round(target_pos, 4)}")
        print(f"             dq_des={np.round(dq_des, 4)}")
        print(f"             gravity={np.round(gravity, 3)}")
        print(f"             tau_net={np.round(tau_clipped, 3)}")
        clipped_mask = np.abs(tau) > MAX_JOINT_TORQUE
        if clipped_mask.any():
            print(f"  *** TORQUE CLIPPED on joints {np.where(clipped_mask)[0].tolist()}"
                  f"  tau_before={np.round(tau[clipped_mask], 2)}")

    return tau_clipped


# ── Sim 2 process — mirrors sim_thread_fn from kinova_sim2real_diff_ik.py ─────
# Extra: writes shm_q_sim2 each step so the main-process viz thread can render it.

def sim2_process_fn(control_hz,
                    shm_q_sim2, shm_target, shm_gains, shm_sim2_hz,
                    stop_event, reset_done_event):
    dt             = 1.0 / control_hz
    device         = "cuda:0" if torch.cuda.is_available() else "cpu"
    posture_target = kinova_deg_to_rad(HOME_DEG)

    robot_cfg = EntityCfg(
        init_state=DEMO_INIT_STATE,
        collisions=(),
        spec_fn=_get_sim2_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlMotorActuatorCfg(target_names_expr=("joint_.*",)),),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
    entity        = Entity(robot_cfg)
    model         = entity.compile()
    sim           = Simulation(
        num_envs=1,
        cfg=SimulationCfg(mujoco=MujocoCfg(timestep=PHYSICS_DT, gravity=(0, 0, -9.81))),
        model=model,
        device=device,
    )
    entity.initialize(model, sim.model, sim.data, device)
    entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
    sim.forward()

    env_ns        = SimpleNamespace(num_envs=1, device=device, scene={"robot": entity}, sim=sim)
    effort_action = JointEffortActionCfg(entity_name="robot", actuator_names=("joint_.*",)).build(env_ns)
    arm_ids       = effort_action.target_ids

    robot  = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    iters  = 0
    t_rate = time.time()
    print("[sim2] Ready.")

    while not stop_event.is_set():
        t = time.time()

        if reset_done_event.is_set():
            reset_done_event.clear()
            entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
            sim.forward()
            effort_action.reset()
            iters  = 0
            t_rate = time.time()
            print("[sim2] Reset done")

        q_sim  = entity.data.joint_pos[0, arm_ids].cpu().numpy()
        dq_sim = entity.data.joint_vel[0, arm_ids].cpu().numpy()
        _np(shm_q_sim2)[:] = q_sim

        target = _np(shm_target).copy()
        gains  = _gains_dict(_np(shm_gains).copy())

        tau_sim = compute_diff_ik_torques(
            robot, target[:3], target[3:], q_sim, dq_sim,
            gains=gains, posture_target=posture_target,
        )

        tau_t = torch.from_numpy(tau_sim).float().to(device).unsqueeze(0)  # (1, 7)
        effort_action.process_actions(tau_t)
        for _ in range(SIM_SUBSTEPS):
            effort_action.apply_actions()
            entity.write_data_to_sim()
            sim.step()

        iters += 1
        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            _np(shm_sim2_hz)[0] = iters / dt_rate
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - t
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print("[sim2] Done.")


# ── Sim 1 thread — mirrors sim_thread_fn from kinova_sim2real_diff_ik_inbuilt_sim.py ──

def sim1_thread_fn(entity, sim, ik_action, ik_cfg, arm_ids, device,
                   shm_target, shm_sim1_hz,
                   stop_event, reset_done_event):
    dt            = 1.0 / CONTROL_HZ
    iters         = 0
    t_rate        = time.time()
    target_action = torch.zeros(1, 7, device=device)
    _kp_prev      = -1.0   # track kp_task changes for debug

    while not stop_event.is_set():
        t = time.time()

        if reset_done_event.is_set():
            reset_done_event.clear()
            entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
            sim.forward()
            ik_action.reset()
            iters  = 0
            t_rate = time.time()
            print("[sim1] Reset done")

        target = _np(shm_target).copy()
        # shm_target: pos(3) + quat_xyzw(4) — diff-IK expects pos(3) + quat_wxyz(4)
        target_action[0, :3] = torch.tensor(target[:3], dtype=torch.float32, device=device)
        target_action[0, 3]  = float(target[6])   # w
        target_action[0, 4]  = float(target[3])   # x
        target_action[0, 5]  = float(target[4])   # y
        target_action[0, 6]  = float(target[5])   # z

        ik_action.process_actions(target_action)
        dq       = ik_action.compute_dq()
        q_target = entity.data.joint_pos[:, arm_ids] + dq
        entity.data.write_ctrl(q_target, ctrl_ids=arm_ids)
        for _ in range(SIM_SUBSTEPS):
            sim.step()

        # ── Debug prints ─────────────────────────────────────────────────────
        kp_now = ik_action.cfg.kp_task
        _kp_changed = (kp_now == 0.0 and _kp_prev != 0.0)
        if iters <= 3 or iters % 300 == 0 or _kp_changed:
            ee_pos  = sim.data.site_xpos[0, ik_action._frame_id].cpu().numpy()
            pos_err = target[:3] - ee_pos
            q_now   = entity.data.joint_pos[0, arm_ids].cpu().numpy()
            # ctrl that was written this step (local ctrl_ids -> global via indexing)
            _global_ctrl = entity.data.indexing.ctrl_ids[arm_ids].cpu().numpy()
            ctrl_vals    = sim.data.ctrl[0, _global_ctrl].cpu().numpy()
            act_force    = entity.data.actuator_force[0, :len(arm_ids)].cpu().numpy()
            dq_np        = dq[0].cpu().numpy()
            print(f"[SIM1 #{iters:6d}] kp_task={kp_now:.2f}  "
                  f"|pos_err|={np.linalg.norm(pos_err):.4f}m  "
                  f"ee_pos={np.round(ee_pos, 4)}")
            print(f"           target_pos={np.round(target[:3], 4)}")
            print(f"           dq={np.round(dq_np, 4)}")
            print(f"           ctrl(pos_target)={np.round(ctrl_vals, 4)}")
            print(f"           q_now          ={np.round(q_now, 4)}")
            print(f"           ctrl-q (error) ={np.round(ctrl_vals - q_now, 4)}")
            print(f"           actuator_force ={np.round(act_force, 3)}")
            if _kp_changed:
                print(f"  *** kp_task changed 0! Monitoring gravity hold...")
        _kp_prev = kp_now
        # ─────────────────────────────────────────────────────────────────────

        iters += 1
        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            _np(shm_sim1_hz)[0] = iters / dt_rate
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - t
        if elapsed < dt:
            time.sleep(dt - elapsed)


# ── Viz thread ────────────────────────────────────────────────────────────────

def viz_thread_fn(sim1_view, sim2_view, mj_model_cpu, mj_data_sim1, mj_data_sim2,
                  entity_sim1, arm_ids, arm_q_idxs, device, shm_q_sim2, stop_event):
    period = 1.0 / VIZ_HZ

    while not stop_event.is_set():
        t = time.time()

        q_sim1 = entity_sim1.data.joint_pos[0, arm_ids].cpu().numpy()
        q_sim2 = _np(shm_q_sim2).copy()

        mj_data_sim1.qpos.flat[arm_q_idxs] = q_sim1
        mujoco.mj_kinematics(mj_model_cpu, mj_data_sim1)
        sim1_view.update(mj_data_sim1)

        mj_data_sim2.qpos.flat[arm_q_idxs] = q_sim2
        mujoco.mj_kinematics(mj_model_cpu, mj_data_sim2)
        sim2_view.update(mj_data_sim2)

        elapsed = time.time() - t
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Shared memory ──────────────────────────────────────────────────────
    shm_q_sim2  = mp.Array(ctypes.c_double, 7)
    shm_target  = mp.Array(ctypes.c_double, 7)   # pos(3) + quat_xyzw(4)
    shm_gains   = mp.Array(ctypes.c_double, len(GAINS_KEYS))
    shm_sim1_hz = mp.Array(ctypes.c_double, 1)
    shm_sim2_hz = mp.Array(ctypes.c_double, 1)

    # ── Init shm_target with valid pose via Pinocchio FK (CPU, pre-GPU) ────
    # Must be done before spawning so Sim 2 never reads a zero quaternion.
    print("Computing initial EE pose…")
    _init_robot       = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    _q_home           = kinova_deg_to_rad(HOME_DEG)
    _ee_pos0, _ee_rot0 = _init_robot.fk(_q_home)
    _q_xyzw           = Rotation.from_matrix(_ee_rot0).as_quat()   # xyzw
    _np(shm_q_sim2)[:] = _q_home
    _np(shm_target)[:3] = _ee_pos0
    _np(shm_target)[3:] = _q_xyzw
    _np(shm_gains)[:]   = _pack_gains(KP_TASK, KP_JOINT, KD_JOINT, DAMPING,
                                      POS_WEIGHT, ORI_WEIGHT, POSTURE_WEIGHT,
                                      JNT_LIMIT_WEIGHT, MAX_DQ)
    print(f"  EE pos: {_ee_pos0}  quat_xyzw: {_q_xyzw}")

    stop_event            = mp.Event()
    reset_done_event_sim1 = mp.Event()   # for Sim 1 thread
    reset_done_event_sim2 = mp.Event()   # for Sim 2 process

    # ── Spawn Sim 2 process BEFORE GPU init ────────────────────────────────
    sim2_proc = mp.Process(
        target=sim2_process_fn,
        args=(CONTROL_HZ,
              shm_q_sim2, shm_target, shm_gains, shm_sim2_hz,
              stop_event, reset_done_event_sim2),
        daemon=True,
    )
    sim2_proc.start()

    # ── Sim 1 (inbuilt DifferentialIKAction, GPU) ──────────────────────────
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    robot_cfg = EntityCfg(
        init_state=DEMO_INIT_STATE,
        collisions=(),
        spec_fn=_get_sim1_spec,
        articulation=KINOVA_GRIPPER_ARTICULATION,
    )
    entity_sim1 = Entity(robot_cfg)
    model_sim1  = entity_sim1.compile()
    sim1        = Simulation(
        num_envs=1,
        cfg=SimulationCfg(mujoco=MujocoCfg(timestep=PHYSICS_DT, gravity=(0, 0, -9.81))),
        model=model_sim1,
        device=device,
    )
    entity_sim1.initialize(model_sim1, sim1.model, sim1.data, device)
    entity_sim1.write_joint_position_to_sim(entity_sim1.data.default_joint_pos, joint_ids=None)
    sim1.forward()

    env_ns  = SimpleNamespace(num_envs=1, device=device, scene={"robot": entity_sim1}, sim=sim1)
    ik_cfg  = DifferentialIKActionCfg(
        entity_name="robot",
        actuator_names=("joint_.*",),
        frame_name="pinch_site",
        frame_type="site",
        kp_task=KP_TASK,
        damping=DAMPING,
        position_weight=POS_WEIGHT,
        orientation_weight=ORI_WEIGHT,
        posture_weight=POSTURE_WEIGHT,
        joint_limit_weight=JNT_LIMIT_WEIGHT,
        use_relative_mode=False,
    )
    ik_action: DifferentialIKAction = ik_cfg.build(env_ns)
    arm_ids = ik_action._joint_ids

    # ── CPU MuJoCo model/data for visualisation ────────────────────────────
    mj_model_cpu = _get_sim1_spec().compile()
    mj_data_sim1 = mujoco.MjData(mj_model_cpu)
    mj_data_sim2 = mujoco.MjData(mj_model_cpu)
    arm_q_idxs   = np.array([mj_model_cpu.joint(n).qposadr for n in _ARM_JOINT_NAMES])

    home_rad = kinova_deg_to_rad(HOME_DEG)
    for d in (mj_data_sim1, mj_data_sim2):
        d.qpos.flat[arm_q_idxs] = home_rad
        mujoco.mj_kinematics(mj_model_cpu, d)

    # ── Viser — Sim 1 (grey) and Sim 2 (blue ghost) ────────────────────────
    server    = viser.ViserServer(label="Kinova Sim2Sim Diff-IK")
    scene     = ViserMujocoScene.create(server, mj_model_cpu)
    sim1_view = scene.add_robot("sim1_inbuilt",   color=(0.75, 0.75, 0.75, 1.00))
    sim2_view = scene.add_robot("sim2_pinocchio", color=(0.20, 0.55, 0.90, 0.65))
    scene.create_visualization_gui(camera_distance=1.2, camera_azimuth=135.0, camera_elevation=30.0)

    # Update shm_target from the actual Sim 1 GPU EE pose
    site_id   = ik_action._frame_id
    ee_pos0   = sim1.data.site_xpos[0, site_id].cpu().numpy()
    quat_wxyz = quat_from_matrix(sim1.data.site_xmat[0, site_id]).cpu().numpy()  # wxyz
    q_xyzw    = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # xyzw for shm
    _np(shm_target)[:3] = ee_pos0
    _np(shm_target)[3:] = q_xyzw

    # ── Startup diagnostics ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STARTUP DIAGNOSTICS")
    print("="*70)

    q_home = kinova_deg_to_rad(HOME_DEG)

    # 1. FK comparison: Pinocchio vs MuJoCo (Sim 1) at home
    pin_pos, pin_rot  = _init_robot.fk(q_home)
    mj_xmat = sim1.data.site_xmat[0, site_id].cpu().numpy().reshape(3, 3)
    pin_quat_xyzw = Rotation.from_matrix(pin_rot).as_quat()
    mj_quat_xyzw  = Rotation.from_matrix(mj_xmat).as_quat()
    print(f"\n[FK] Pinocchio  EE pos = {np.round(pin_pos, 6)}")
    print(f"[FK] MuJoCo(S1) EE pos = {np.round(ee_pos0, 6)}")
    print(f"[FK] pos diff          = {np.round(ee_pos0 - pin_pos, 6)}  "
          f"(norm={np.linalg.norm(ee_pos0 - pin_pos):.2e})")
    print(f"[FK] Pinocchio  quat(xyzw) = {np.round(pin_quat_xyzw, 6)}")
    print(f"[FK] MuJoCo(S1) quat(xyzw) = {np.round(mj_quat_xyzw, 6)}")
    print(f"[FK] quat diff norm         = {np.linalg.norm(mj_quat_xyzw - pin_quat_xyzw):.2e}")

    # 2. Gravity torques at home (Pinocchio reference)
    pin_grav = _init_robot.gravity(q_home)
    print(f"\n[GRAV] Pinocchio gravity torques at home:")
    for i, (g, lim) in enumerate(zip(pin_grav, MAX_JOINT_TORQUE)):
        print(f"  joint_{i+1}: {g:+.3f} Nm  (limit={lim:.0f} Nm, "
              f"fraction={abs(g)/lim*100:.1f}%)")

    # 3. Jacobian comparison at home
    J_pin = _init_robot.jacobian(q_home)
    # Trigger one compute_dq to populate mjwarp Jacobian buffers
    _target_tmp = torch.zeros(1, 7, device=device)
    _target_tmp[0, :3] = torch.tensor(ee_pos0, dtype=torch.float32, device=device)
    _target_tmp[0, 3:]  = torch.tensor(quat_wxyz, dtype=torch.float32, device=device)  # wxyz
    ik_action.process_actions(_target_tmp)
    ik_action.compute_dq()
    J_mj_p = ik_action._jacp_torch[0, :, ik_action._joint_dof_ids].cpu().numpy()  # (3, nj)
    J_mj_r = ik_action._jacr_torch[0, :, ik_action._joint_dof_ids].cpu().numpy()  # (3, nj)
    J_mj   = np.vstack([J_mj_p, J_mj_r])
    print(f"\n[JAC] Pinocchio Jacobian (6×7) at home — row norms:")
    print(f"  {[round(float(np.linalg.norm(J_pin[i, :])), 4) for i in range(6)]}")
    print(f"[JAC] MuJoCo(mjwarp) Jacobian (6×7) at home — row norms:")
    print(f"  {[round(float(np.linalg.norm(J_mj[i, :])), 4) for i in range(6)]}")
    J_diff = J_pin - J_mj
    print(f"[JAC] Max element-wise diff = {np.abs(J_diff).max():.4e}")
    print(f"[JAC] Frobenius norm diff   = {np.linalg.norm(J_diff):.4e}")

    # 4. Show what ctrl_ids arm_ids maps to (index sanity check)
    _global_ctrl_ids = entity_sim1.data.indexing.ctrl_ids[arm_ids].cpu().numpy()
    print(f"\n[CTRL] arm_ids (joint IDs)              = {arm_ids.cpu().numpy()}")
    print(f"[CTRL] -> global ctrl_ids (actuator IDs) = {_global_ctrl_ids}")
    print(f"[CTRL] initial ctrl values               = "
          f"{sim1.data.ctrl[0, _global_ctrl_ids].cpu().numpy()}")
    print(f"[CTRL] initial q (arm joints)            = "
          f"{entity_sim1.data.joint_pos[0, arm_ids].cpu().numpy()}")

    print("="*70 + "\n")

    transform_ctrl = server.scene.add_transform_controls(
        "/ik_target",
        position=tuple(float(v) for v in ee_pos0),
        wxyz=(float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
        scale=0.15,
    )

    with server.gui.add_folder("Sim2Sim"):
        txt_sim1_hz = server.gui.add_text("Sim1 (inbuilt) rate",   initial_value="— Hz")
        txt_sim2_hz = server.gui.add_text("Sim2 (pinocchio) rate", initial_value="— Hz")
        reset_btn   = server.gui.add_button("Reset Both")

    with server.gui.add_folder("IK Gains"):
        sl_kp_task  = server.gui.add_slider("Kp task",           min=0.0,  max=10.0, step=0.1,  initial_value=KP_TASK)
        sl_damping  = server.gui.add_slider("Damping",           min=1e-3, max=1.0,  step=1e-3, initial_value=DAMPING)
        sl_pos_w    = server.gui.add_slider("Pos weight",        min=0.0,  max=10.0, step=0.1,  initial_value=POS_WEIGHT)
        sl_ori_w    = server.gui.add_slider("Ori weight",        min=0.0,  max=10.0, step=0.1,  initial_value=ORI_WEIGHT)
        sl_max_dq   = server.gui.add_slider("Max dq (rad)",      min=0.01, max=2.0,  step=0.01, initial_value=MAX_DQ)
        sl_post_w   = server.gui.add_slider("Posture weight",    min=0.0,  max=1.0,  step=0.01, initial_value=POSTURE_WEIGHT)
        sl_jlim_w   = server.gui.add_slider("Jnt limit weight",  min=0.0,  max=1.0,  step=0.01, initial_value=JNT_LIMIT_WEIGHT)
        server.gui.add_text("Posture target (deg)", initial_value=", ".join(f"{v:.0f}" for v in HOME_DEG))

    with server.gui.add_folder("Sim2 Joint Gains"):
        sl_kp_joint = server.gui.add_slider("Kp joint", min=0.0, max=300.0, step=1.0, initial_value=KP_JOINT)
        sl_kd_joint = server.gui.add_slider("Kd joint", min=0.0, max=300.0, step=1.0, initial_value=KD_JOINT)

    th_stop = threading.Event()

    def on_reset(_):
        reset_done_event_sim1.set()
        reset_done_event_sim2.set()

    reset_btn.on_click(on_reset)

    threading.Thread(target=sim1_thread_fn, daemon=True, args=(
        entity_sim1, sim1, ik_action, ik_cfg, arm_ids, device,
        shm_target, shm_sim1_hz,
        th_stop, reset_done_event_sim1,
    )).start()

    threading.Thread(target=viz_thread_fn, daemon=True, args=(
        sim1_view, sim2_view, mj_model_cpu, mj_data_sim1, mj_data_sim2,
        entity_sim1, arm_ids, arm_q_idxs, device, shm_q_sim2, th_stop,
    )).start()

    print("Running. Ctrl+C to stop.")
    try:
        while True:
            p = transform_ctrl.position
            w = transform_ctrl.wxyz
            _np(shm_target)[:3] = [p[0], p[1], p[2]]
            _np(shm_target)[3:] = [w[1], w[2], w[3], w[0]]
            _np(shm_gains)[:]   = _pack_gains(
                sl_kp_task.value, sl_kp_joint.value, sl_kd_joint.value, sl_damping.value,
                sl_pos_w.value, sl_ori_w.value, sl_post_w.value, sl_jlim_w.value, sl_max_dq.value,
            )
            # Update Sim 1 IK cfg directly (shared with sim1 thread)
            ik_cfg.kp_task            = max(sl_kp_task.value, 0.0)
            ik_cfg.damping            = max(sl_damping.value,  1e-2)
            ik_cfg.position_weight    = max(sl_pos_w.value,    0.0)
            ik_cfg.orientation_weight = max(sl_ori_w.value,    0.0)
            ik_cfg.posture_weight     = max(sl_post_w.value,   0.0)
            ik_cfg.joint_limit_weight = max(sl_jlim_w.value,   0.0)
            txt_sim1_hz.value = f"{_np(shm_sim1_hz)[0]:.0f} Hz"
            txt_sim2_hz.value = f"{_np(shm_sim2_hz)[0]:.0f} Hz"
            time.sleep(1.0 / 60)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        th_stop.set()
        stop_event.set()
        sim2_proc.join(timeout=10.0)
        server.stop()
        print("Done.")


if __name__ == "__main__":
    main()
