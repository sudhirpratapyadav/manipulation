"""Kinova Gen3 sim-only diff-IK (Pinocchio + mjlab).

Same controller as kinova_sim2real_diff_ik.py but without hardware.

Threading:
  - Sim  thread : diff-IK torques computed at --control-hz; physics substeps
                  run every PHYSICS_DT to fill each control period
  - Viz  thread (30 Hz) : ViserMujocoScene update
  - Main thread : viser target + gains poll

Run:
  python kinova_sim_diff_ik.py [--control-hz 100]
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
PHYSICS_DT  = 0.002
CONTROL_HZ  = 100
SIM_SUBSTEPS = max(1, round(1.0 / (CONTROL_HZ * PHYSICS_DT)))  # physics steps per control step
VIZ_HZ      = 30


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


def compute_diff_ik_torques(robot, target_pos, target_quat_xyzw, q, dq, *, gains, posture_target):
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

    tau = gains["kp_joint"] * dq_des + gains["kd_joint"] * (0.0 - dq) + robot.gravity(q)
    return np.clip(tau, -MAX_JOINT_TORQUE, MAX_JOINT_TORQUE)


# ── Threads ───────────────────────────────────────────────────────────────────

def sim_thread_fn(entity, sim, effort_action, robot, arm_ids, device,
                  shared, stop_event, reset_event, posture_target,
                  sim_substeps, control_hz):
    dt     = 1.0 / control_hz
    iters  = 0
    t_rate = time.time()

    # Per-component accumulators (seconds); printed every second alongside Hz
    _t = dict(read_state=0.0, diff_ik=0.0, process_actions=0.0,
              sub_apply=0.0, sub_write=0.0, sub_step=0.0, sleep=0.0)
    _timing_n = 0  # never reset — counts iters within the current 1-s window

    while not stop_event.is_set():
        t_loop = time.time()

        if reset_event.is_set():
            reset_event.clear()
            entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
            sim.forward()
            effort_action.reset()
            iters  = 0
            t_rate = time.time()
            _t = dict(read_state=0.0, diff_ik=0.0, process_actions=0.0,
                      sub_apply=0.0, sub_write=0.0, sub_step=0.0, sleep=0.0)
            print("[sim] Reset")

        # ── read joint state ───────────────────────────────────────────────
        _t0 = time.perf_counter()
        q_sim  = entity.data.joint_pos[0, arm_ids].cpu().numpy()
        dq_sim = entity.data.joint_vel[0, arm_ids].cpu().numpy()
        _t["read_state"] += time.perf_counter() - _t0

        # ── diff-IK torque computation ─────────────────────────────────────
        _t0 = time.perf_counter()
        tau_sim = compute_diff_ik_torques(
            robot, shared["target_pos"], shared["target_quat_xyzw"], q_sim, dq_sim,
            gains=shared["gains"], posture_target=posture_target,
        )
        _t["diff_ik"] += time.perf_counter() - _t0

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

            # ── timing breakdown (once per second) ────────────────────────
            n = _timing_n
            total = sum(_t.values())
            lines = [f"[timing] {iters} iters/s  |  avg iter={total*1e3/n:.3f} ms  (compute-only Hz potential: {n/max(total-_t['sleep'],1e-9):.0f})"]
            for name, acc in _t.items():
                pct = 100 * acc / total if total > 0 else 0
                lines.append(f"  {name:<18s}: {acc*1e3/n:6.3f} ms/iter  ({pct:4.1f}%)")
            print("\n".join(lines))

            iters     = 0
            _timing_n = 0
            t_rate    = time.time()
            _t = dict(read_state=0.0, diff_ik=0.0, process_actions=0.0,
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

    # device = "cpu"  # GPU Warp overhead >> physics cost for num_envs=1; switch back for >32 envs
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

    server = viser.ViserServer(label="Kinova Sim Diff-IK")
    scene  = ViserMujocoScene.create(server, sim.mj_model, num_envs=1)
    scene.create_visualization_gui(camera_distance=1.2, camera_azimuth=135.0, camera_elevation=30.0)

    q_home_rad = kinova_deg_to_rad(HOME_DEG)
    ee_pos0, ee_rot0 = robot.fk(q_home_rad)
    q_xyzw = Rotation.from_matrix(ee_rot0).as_quat()
    transform_ctrl = server.scene.add_transform_controls(
        "/ik_target",
        position=tuple(float(v) for v in ee_pos0),
        wxyz=(float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])),
        scale=0.15,
    )

    with server.gui.add_folder("Sim"):
        txt_ctrl_hz = server.gui.add_text("Control rate", initial_value="— Hz")
        reset_btn   = server.gui.add_button("Reset")

    with server.gui.add_folder("Gains"):
        sl_kp_task  = server.gui.add_slider("Kp task",  min=0.0, max=10.0,  step=0.1,  initial_value=KP_TASK)
        sl_kp_joint = server.gui.add_slider("Kp joint", min=0.0, max=300.0, step=1.0,  initial_value=KP_JOINT)
        sl_kd_joint = server.gui.add_slider("Kd joint", min=0.0, max=300.0, step=1.0,  initial_value=KD_JOINT)
        sl_damping  = server.gui.add_slider("Damping",  min=1e-3, max=1.0,  step=1e-3, initial_value=DAMPING)

    with server.gui.add_folder("IK Weights"):
        sl_pos_w  = server.gui.add_slider("Pos weight",   min=0.0, max=1.0,  step=0.01, initial_value=POS_WEIGHT)
        sl_ori_w  = server.gui.add_slider("Ori weight",   min=0.0, max=1.0,  step=0.01, initial_value=ORI_WEIGHT)
        sl_max_dq = server.gui.add_slider("Max dq (rad)", min=0.01, max=2.0, step=0.01, initial_value=MAX_DQ)

    with server.gui.add_folder("Posture"):
        sl_post_w = server.gui.add_slider("Posture weight",   min=0.0, max=1.0, step=0.01, initial_value=POSTURE_WEIGHT)
        sl_jlim_w = server.gui.add_slider("Jnt limit weight", min=0.0, max=1.0, step=0.01, initial_value=JNT_LIMIT_WEIGHT)
        server.gui.add_text("Posture target (deg)", initial_value=", ".join(f"{v:.0f}" for v in HOME_DEG))

    shared = {
        "target_pos":       ee_pos0.copy(),
        "target_quat_xyzw": np.array([q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]]),
        "ctrl_hz":          0.0,
        "gains": dict(
            kp_task=KP_TASK, kp_joint=KP_JOINT,
            kd_joint=KD_JOINT, damping=DAMPING,
            pos_weight=POS_WEIGHT, ori_weight=ORI_WEIGHT,
            posture_weight=POSTURE_WEIGHT, jnt_limit_weight=JNT_LIMIT_WEIGHT,
            max_dq=MAX_DQ,
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
                kp_task=sl_kp_task.value, kp_joint=sl_kp_joint.value,
                kd_joint=sl_kd_joint.value, damping=sl_damping.value,
                pos_weight=sl_pos_w.value, ori_weight=sl_ori_w.value,
                posture_weight=sl_post_w.value, jnt_limit_weight=sl_jlim_w.value,
                max_dq=sl_max_dq.value,
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
