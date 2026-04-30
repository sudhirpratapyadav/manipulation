"""Kinova Gen3 sim-only OSC (Pinocchio + mjlab).

Same structure as kinova_sim_diff_ik.py but uses Operational Space Control
instead of diff-IK: torques are computed directly from the full arm dynamics
(mass matrix, Coriolis, gravity) via pinocchio.

Threading:
  - Sim  thread : OSC torques computed at --control-hz; physics substeps
                  run every PHYSICS_DT to fill each control period
  - Viz  thread (30 Hz) : ViserMujocoScene update
  - Main thread : viser target + gains poll

Run:
  python kinova_osc.py [--control-hz 100]
"""

from __future__ import annotations

import argparse
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pinocchio as pin
import torch
import viser
from scipy.spatial.transform import Rotation

from kinova_tasks.assets.kinova_gen3.kinova_constants import KINOVA_GEN3_GRIPPER_XML
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions.actions import JointEffortActionCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.viewer.viser import ViserMujocoScene

# ── Model ─────────────────────────────────────────────────────────────────────
_TORQUE_XML      = KINOVA_GEN3_GRIPPER_XML.parent / "gen3_gripper_torque.xml"
_ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]

HOME_DEG = np.array([0.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])

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
KP_POS         = 400.0
KD_POS         =  40.0
KP_ORI         = 400.0
KD_ORI         =  40.0
POSTURE_KP     =  10.0
POSTURE_KD     =   2.0
POSTURE_WEIGHT =   0.01

MAX_JOINT_TORQUE = np.array([390.0, 390.0, 390.0, 390.0, 90.0, 90.0, 90.0])

# ── Timing ────────────────────────────────────────────────────────────────────
PHYSICS_DT   = 0.002
CONTROL_HZ   = 100
SIM_SUBSTEPS = max(1, round(1.0 / (CONTROL_HZ * PHYSICS_DT)))
VIZ_HZ       = 30


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


# ── Threads ───────────────────────────────────────────────────────────────────

def sim_thread_fn(entity, sim, effort_action, robot, arm_ids, device,
                  shared, stop_event, reset_event, posture_target,
                  sim_substeps, control_hz):
    dt     = 1.0 / control_hz
    iters  = 0
    t_rate = time.time()

    _t = dict(read_state=0.0, osc=0.0, process_actions=0.0,
              sub_apply=0.0, sub_write=0.0, sub_step=0.0, sleep=0.0)
    _timing_n = 0

    while not stop_event.is_set():
        t_loop = time.time()

        if reset_event.is_set():
            reset_event.clear()
            entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
            sim.forward()
            effort_action.reset()
            iters  = 0
            t_rate = time.time()
            _t = dict(read_state=0.0, osc=0.0, process_actions=0.0,
                      sub_apply=0.0, sub_write=0.0, sub_step=0.0, sleep=0.0)
            print("[sim] Reset")

        # ── read joint state ───────────────────────────────────────────────
        _t0 = time.perf_counter()
        q_sim  = entity.data.joint_pos[0, arm_ids].cpu().numpy()
        dq_sim = entity.data.joint_vel[0, arm_ids].cpu().numpy()
        _t["read_state"] += time.perf_counter() - _t0

        # ── OSC torque computation ─────────────────────────────────────────
        _t0 = time.perf_counter()
        tau_sim = compute_osc_torques(
            robot, shared["target_pos"], shared["target_quat_xyzw"], q_sim, dq_sim,
            gains=shared["gains"], posture_target=posture_target,
        )
        _t["osc"] += time.perf_counter() - _t0

        # ── process actions ────────────────────────────────────────────────
        _t0 = time.perf_counter()
        tau_t = torch.zeros(1, 7, dtype=torch.float32, device=device)
        tau_t[0] = torch.from_numpy(tau_sim).to(device)
        effort_action.process_actions(tau_t)
        _t["process_actions"] += time.perf_counter() - _t0

        # ── Physics substeps ───────────────────────────────────────────────
        for _ in range(sim_substeps):
            _t0 = time.perf_counter()
            effort_action.apply_actions()
            _t["sub_apply"] += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            entity.write_data_to_sim()
            _t["sub_write"] += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            sim.step()
            _t["sub_step"] += time.perf_counter() - _t0

        _timing_n += 1
        iters += 1
        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            shared["ctrl_hz"] = iters / dt_rate

            n     = _timing_n
            total = sum(_t.values())
            # lines = [f"[timing] {iters} iters/s  |  avg iter={total*1e3/n:.3f} ms  (compute-only Hz potential: {n/max(total-_t['sleep'],1e-9):.0f})"]
            # for name, acc in _t.items():
            #     pct = 100 * acc / total if total > 0 else 0
            #     lines.append(f"  {name:<18s}: {acc*1e3/n:6.3f} ms/iter  ({pct:4.1f}%)")
            # print("\n".join(lines))

            iters     = 0
            _timing_n = 0
            t_rate    = time.time()
            _t = dict(read_state=0.0, osc=0.0, process_actions=0.0,
                      sub_apply=0.0, sub_write=0.0, sub_step=0.0, sleep=0.0)

        elapsed = time.time() - t_loop
        remaining = dt - elapsed
        if remaining > 0:
            _t0 = time.perf_counter()
            time.sleep(remaining)
            _t["sleep"] += time.perf_counter() - _t0


def viz_thread_fn(scene, sim, stop_event):
    period = 1.0 / VIZ_HZ
    while not stop_event.is_set():
        t = time.time()
        scene.update(sim.wp_data)
        if scene.needs_update:
            scene.refresh_visualization()
        elapsed = time.time() - t
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-hz", type=int, default=CONTROL_HZ)
    args = parser.parse_args()

    sim_substeps = max(1, round(1.0 / (args.control_hz * PHYSICS_DT)))

    print(f"torch device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    device = "cuda:0"

    def get_spec():
        return mujoco.MjSpec.from_file(str(_TORQUE_XML))

    robot_cfg = EntityCfg(
        init_state=DEMO_INIT_STATE,
        collisions=(),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=("joint_.*",), command_field="effort"),),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
    entity = Entity(robot_cfg)
    model  = entity.compile()
    sim    = Simulation(num_envs=1, cfg=SimulationCfg(mujoco=MujocoCfg(timestep=PHYSICS_DT, gravity=(0,0,-9.81))), model=model, device=device)
    entity.initialize(model, sim.model, sim.data, device)
    entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
    sim.forward()

    env_ns        = SimpleNamespace(num_envs=1, device=device, scene={"robot": entity}, sim=sim)
    effort_action = JointEffortActionCfg(entity_name="robot", actuator_names=("joint_.*",)).build(env_ns)
    arm_ids       = effort_action.target_ids

    print("Loading Pinocchio model…")
    robot = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    posture_target = kinova_deg_to_rad(HOME_DEG)
    print("  OK")

    server = viser.ViserServer(label="Kinova Sim OSC")
    scene  = ViserMujocoScene.create(server, sim.mj_model, num_envs=1)
    scene.create_visualization_gui(camera_distance=1.2, camera_azimuth=135.0, camera_elevation=30.0)

    q_home_rad = kinova_deg_to_rad(HOME_DEG)
    ee_pos0, ee_rot0 = robot.fk(q_home_rad)
    q_xyzw = Rotation.from_matrix(ee_rot0).as_quat()
    transform_ctrl = server.scene.add_transform_controls(
        "/osc_target",
        position=tuple(float(v) for v in ee_pos0),
        wxyz=(float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])),
        scale=0.15,
    )

    with server.gui.add_folder("Sim"):
        txt_ctrl_hz = server.gui.add_text("Control rate", initial_value="— Hz")
        reset_btn   = server.gui.add_button("Reset")

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

    shared = {
        "target_pos":       ee_pos0.copy(),
        "target_quat_xyzw": np.array([q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]]),
        "ctrl_hz":          0.0,
        "gains": dict(
            kp_pos=KP_POS, kd_pos=KD_POS,
            kp_ori=KP_ORI, kd_ori=KD_ORI,
            posture_kp=POSTURE_KP, posture_kd=POSTURE_KD,
            posture_weight=POSTURE_WEIGHT,
        ),
    }

    stop_event  = threading.Event()
    reset_event = threading.Event()
    reset_btn.on_click(lambda _: reset_event.set())

    threading.Thread(target=sim_thread_fn, daemon=True, args=(
        entity, sim, effort_action, robot, arm_ids, device,
        shared, stop_event, reset_event, posture_target,
        sim_substeps, args.control_hz,
    )).start()

    threading.Thread(target=viz_thread_fn, daemon=True, args=(
        scene, sim, stop_event,
    )).start()

    print("Running. Ctrl+C to stop.")
    try:
        while True:
            p = transform_ctrl.position
            w = transform_ctrl.wxyz
            shared["target_pos"]       = np.array([p[0], p[1], p[2]])
            shared["target_quat_xyzw"] = np.array([w[1], w[2], w[3], w[0]])
            shared["gains"] = dict(
                kp_pos=sl_kp_pos.value, kd_pos=sl_kd_pos.value,
                kp_ori=sl_kp_ori.value, kd_ori=sl_kd_ori.value,
                posture_kp=sl_post_kp.value, posture_kd=sl_post_kd.value,
                posture_weight=sl_post_w.value,
            )
            txt_ctrl_hz.value = f"{shared['ctrl_hz']:.0f} Hz"
            time.sleep(1.0 / 60)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        stop_event.set()
        server.stop()
        print("Done.")


if __name__ == "__main__":
    main()
