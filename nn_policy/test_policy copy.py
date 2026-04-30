"""Kinova Gen3 sim-only — neural network policy edition (multiprocess).

Three processes, communicating via shared memory (zero-copy, no queues):
  - Sim    (CONTROL_HZ): diff-IK + warp GPU step; writes body poses + joint state
  - Policy (CONTROL_HZ): Pinocchio FK → obs → NN inference; writes EE target + peg pose
  - Viz    (VIZ_HZ)    : viser render + GUI sliders; writes gain values

Shared memory layout:
  joint_pos  (7,)  f32   sim → policy
  joint_vel  (7,)  f32   sim → policy
  body_xpos  (N,3) f32   sim → viz
  body_xmat  (N,9) f32   sim → viz   (flat 3×3 rotation matrices)
  target     (7,)  f64   policy → sim   (EE pos[3] + quat_xyzw[4])
  peg_sim    (7,)  f64   policy → viz   (peg pos[3] + quat_wxyz[4])
  gains      (9,)  f64   viz → sim
  sim_hz     (1,)  f64   sim → viz
  policy_hz  (1,)  f64   policy → viz
  steps      (1,)  i64   sim → viz

Run:
  CUDA_VISIBLE_DEVICES=1 uv run python nn_policy/test_policy.py
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_NN_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR      = os.path.dirname(_NN_POLICY_DIR)
if _NN_POLICY_DIR not in sys.path:
    sys.path.insert(0, _NN_POLICY_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

import mujoco
import numpy as np
from kinova_tasks.assets.kinova_gen3.kinova_constants import KINOVA_GEN3_GRIPPER_XML

# ── Model paths ───────────────────────────────────────────────────────────────
_TORQUE_XML      = KINOVA_GEN3_GRIPPER_XML.parent / "gen3_gripper_torque.xml"
_ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
_PEG_XML  = Path(_ROOT_DIR) / "src/kinova_tasks/assets/peg_in_hole/xmls/peg.xml"
_HOLE_XML = Path(_ROOT_DIR) / "src/kinova_tasks/assets/peg_in_hole/xmls/hole.xml"

_DEFAULT_CHECKPOINT = os.path.join(_NN_POLICY_DIR, "weights", "model_999.pt")

# ── Scene poses ───────────────────────────────────────────────────────────────
HOLE_POS        = np.array([0.0, -0.5,  0.02])
HOLE_QUAT_XYZW  = np.array([0.00,  0.00,  0.00,  1.00])

EE_TO_PEG_POS       = np.array([0.00, 0.00, 0.0])
EE_TO_PEG_QUAT_XYZW = np.array([1.00, 0.00, 0.00, 0.00])

PEG_TOP_SITE_OFFSET    = np.array([0.00, 0.00, -0.02])
PEG_BOTTOM_SITE_OFFSET = np.array([0.00, 0.00, -0.07])
HOLE_TOP_SITE_OFFSET   = np.array([0.00, 0.00,  0.07])
HOLE_BOTTOM_SITE_OFFSET= np.array([0.00, 0.00,  0.02])

# ── Sim initial joint positions (plain dict — no mjlab import at module level) ─
DEMO_JOINT_POS = {
    "joint_1":  1.57079633, "joint_2":  0.52359878, "joint_3":  0.0,
    "joint_4":  1.57079633, "joint_5":  0.0,        "joint_6":  1.04719755,
    "joint_7": -1.57079633,
    "right_driver_joint": 0.8, "left_driver_joint": 0.8,
}

# ── Gains ─────────────────────────────────────────────────────────────────────
KP_TASK=1.0; KP_JOINT=100.0; KD_JOINT=20.0; DAMPING=0.05
POS_WEIGHT=1.0; ORI_WEIGHT=1.0; POSTURE_WEIGHT=0.0; JNT_LIMIT_WEIGHT=0.0; MAX_DQ=0.5
MAX_JOINT_TORQUE = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])
GAINS_KEYS = ["kp_task","kp_joint","kd_joint","damping",
              "pos_weight","ori_weight","posture_weight","jnt_limit_weight","max_dq"]

# ── Timing ────────────────────────────────────────────────────────────────────
PHYSICS_DT   = 0.002
CONTROL_HZ   = 50
SIM_SUBSTEPS = max(1, round(1.0 / (CONTROL_HZ * PHYSICS_DT)))
VIZ_HZ       = 30
EPISODE_SECS = 5.0
MAX_STEPS    = round(EPISODE_SECS * CONTROL_HZ)
HOME_DEG     = np.array([90.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])

TARGET_POS_BOUND = 0.25
DELTA_POS_BOUND  = 0.001


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _pack_gains(kp_task, kp_joint, kd_joint, damping,
                pos_weight, ori_weight, posture_weight, jnt_limit_weight, max_dq):
    return np.array([kp_task, kp_joint, kd_joint, damping,
                     pos_weight, ori_weight, posture_weight, jnt_limit_weight, max_dq])

def _gains_dict(arr): return dict(zip(GAINS_KEYS, arr))
def _xyzw_to_wxyz(q): return q[[3, 0, 1, 2]]

def kinova_deg_to_rad(deg):
    s = deg.copy(); s[s > 180.0] -= 360.0; return np.deg2rad(s)

def ee_to_peg_world(ee_pos, ee_rot, off_pos, off_rot):
    return ee_pos + ee_rot @ off_pos, ee_rot @ off_rot

def compute_peg_to_hole(peg_pos, peg_rot, hole_pos, hole_rot):
    return np.concatenate([
        hole_pos + hole_rot @ HOLE_TOP_SITE_OFFSET    - (peg_pos + peg_rot @ PEG_TOP_SITE_OFFSET),
        hole_pos + hole_rot @ HOLE_BOTTOM_SITE_OFFSET - (peg_pos + peg_rot @ PEG_BOTTOM_SITE_OFFSET),
    ]).astype(np.float32)

def pose_error_6d(target_pos, target_quat_xyzw, cur_pos, cur_rot):
    from scipy.spatial.transform import Rotation
    return np.concatenate([
        target_pos - cur_pos,
        Rotation.from_matrix(Rotation.from_quat(target_quat_xyzw).as_matrix() @ cur_rot.T).as_rotvec(),
    ])

def compute_diff_ik_torques(robot, target_pos, target_quat_xyzw, q, dq, *, gains, posture_target):
    ee_pos, ee_rot = robot.fk(q)
    err    = pose_error_6d(target_pos, target_quat_xyzw, ee_pos, ee_rot)
    pos_dx = gains["kp_task"] * err[:3]
    ori_dx = gains["kp_task"] * err[3:]
    J = robot.jacobian(q); jacp, jacr = J[:3], J[3:]
    JTJ  = gains["pos_weight"]*(jacp.T@jacp) + gains["ori_weight"]*(jacr.T@jacr)
    JTdx = gains["pos_weight"]*(jacp.T@pos_dx) + gains["ori_weight"]*(jacr.T@ori_dx)
    r_lim = np.clip(robot.joint_upper-q,None,0)+np.clip(robot.joint_lower-q,0,None)
    viol  = (r_lim!=0).astype(float)
    JTJ  += np.diag(gains["jnt_limit_weight"]*viol);   JTdx += gains["jnt_limit_weight"]*viol*r_lim
    JTJ  += gains["posture_weight"]*np.eye(7);          JTdx += gains["posture_weight"]*(posture_target-q)
    JTJ  += gains["damping"]**2*np.eye(7)
    dq_des = np.clip(np.linalg.solve(JTJ, JTdx), -gains["max_dq"], gains["max_dq"])
    return np.clip(gains["kp_joint"]*dq_des + gains["kd_joint"]*(0.0-dq) + robot.gravity(q),
                   -MAX_JOINT_TORQUE, MAX_JOINT_TORQUE)


# ── PinocchioArm ──────────────────────────────────────────────────────────────

class PinocchioArm:
    def __init__(self, mjcf_path: str, ee_frame: str) -> None:
        import pinocchio as pin
        self._pin = pin
        self.model = pin.buildModelFromMJCF(mjcf_path)
        self.model.gravity.linear = np.array([0.0, 0.0, -9.81])
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_frame)
        if self.ee_frame_id >= self.model.nframes:
            raise ValueError(f"Frame '{ee_frame}' not found")
        self._v_idx = np.array([self.model.joints[self.model.getJointId(n)].idx_v for n in _ARM_JOINT_NAMES], dtype=np.intp)
        self._q_idx = np.array([self.model.joints[self.model.getJointId(n)].idx_q for n in _ARM_JOINT_NAMES], dtype=np.intp)
        self._q_full = pin.neutral(self.model)
        self.joint_lower = self.model.lowerPositionLimit[self._q_idx].copy()
        self.joint_upper = self.model.upperPositionLimit[self._q_idx].copy()

    def _set_q(self, q): self._q_full[self._q_idx] = q

    def fk(self, q):
        self._set_q(q)
        self._pin.framesForwardKinematics(self.model, self.data, self._q_full)
        oMf = self.data.oMf[self.ee_frame_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def jacobian(self, q):
        self._set_q(q)
        self._pin.computeJointJacobians(self.model, self.data, self._q_full)
        self._pin.framesForwardKinematics(self.model, self.data, self._q_full)
        J = self._pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, self._pin.LOCAL_WORLD_ALIGNED)
        return J[:, self._v_idx]

    def gravity(self, q):
        self._set_q(q)
        self._pin.computeGeneralizedGravity(self.model, self.data, self._q_full)
        return self.data.g[self._v_idx].copy()


# ── Shared memory helpers ─────────────────────────────────────────────────────

def _attach_shm(shm_names: dict) -> tuple[dict, dict]:
    handles, arrays = {}, {}
    for key, (name, shape, dtype) in shm_names.items():
        shm = SharedMemory(name=name)
        handles[key] = shm
        arrays[key]  = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return handles, arrays

def _close_shm(handles: dict) -> None:
    for shm in handles.values():
        shm.close()


# ── Sim process ───────────────────────────────────────────────────────────────

def sim_process_fn(shm_names: dict, events: dict, device: str) -> None:
    """Owns warp GPU sim. Reads target+gains, writes body poses+joint state."""
    import torch
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs.mdp.actions.actions import JointEffortActionCfg
    from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg

    stop_ev   = events['stop']
    reset_ev  = events['reset']
    reset_pol = events['reset_policy']

    shm_handles, shm = _attach_shm(shm_names)
    nbody = shm['body_xpos'].shape[0]

    def get_spec():
        return mujoco.MjSpec.from_file(str(_TORQUE_XML))

    init_state = EntityCfg.InitialStateCfg(pos=(0,0,0), joint_pos=DEMO_JOINT_POS, joint_vel={".*": 0.0})
    robot_cfg  = EntityCfg(
        init_state=init_state, collisions=(), spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=("joint_.*",), command_field="effort"),),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
    entity = Entity(robot_cfg)
    model  = entity.compile()
    sim    = Simulation(
        num_envs=1,
        cfg=SimulationCfg(mujoco=MujocoCfg(timestep=PHYSICS_DT, gravity=(0, 0, -9.81))),
        model=model, device=device,
    )
    entity.initialize(model, sim.model, sim.data, device)
    entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
    sim.forward()

    fingers_ctrl_id = model.actuator("fingers_actuator").id
    gripper_ctrl    = torch.tensor([[255.0]], device=device)
    env_ns          = SimpleNamespace(num_envs=1, device=device, scene={"robot": entity}, sim=sim)
    effort_action   = JointEffortActionCfg(entity_name="robot", actuator_names=("joint_.*",)).build(env_ns)
    arm_ids         = effort_action.target_ids

    robot          = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    posture_target = kinova_deg_to_rad(HOME_DEG)

    # Seed shared target with home EE pose so sim has a valid target from step 1
    q_home             = kinova_deg_to_rad(HOME_DEG)
    home_ee_pos, home_ee_rot = robot.fk(q_home)
    shm['target'][:3] = home_ee_pos
    shm['target'][3:] = Rotation.from_matrix(home_ee_rot).as_quat()

    # Write initial body poses
    def _push_body_poses():
        shm['body_xpos'][:] = entity.data.data.xpos.cpu().numpy()[0]
        shm['body_xmat'][:] = entity.data.data.xmat.cpu().numpy()[0].reshape(nbody, 9)

    _push_body_poses()
    print("[sim] Ready")

    dt = 1.0 / CONTROL_HZ
    iters = 0; t_rate = time.time()

    while not stop_ev.is_set():
        t = time.time()

        if reset_ev.is_set():
            reset_ev.clear()
            entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
            entity.data.write_ctrl(gripper_ctrl, ctrl_ids=[fingers_ctrl_id])
            entity.write_data_to_sim()
            sim.forward()
            effort_action.reset()
            iters = 0; shm['steps'][0] = 0; t_rate = time.time()
            reset_pol.set()
            print("[sim] Reset done")

        q_sim  = entity.data.joint_pos[0, arm_ids].cpu().numpy()
        dq_sim = entity.data.joint_vel[0, arm_ids].cpu().numpy()

        tau = compute_diff_ik_torques(
            robot, shm['target'][:3].copy(), shm['target'][3:].copy(),
            q_sim, dq_sim, gains=_gains_dict(shm['gains'].copy()), posture_target=posture_target,
        )

        tau_t = torch.from_numpy(tau).float().to(device).unsqueeze(0)
        effort_action.process_actions(tau_t)
        for _ in range(SIM_SUBSTEPS):
            effort_action.apply_actions()
            entity.data.write_ctrl(gripper_ctrl, ctrl_ids=[fingers_ctrl_id])
            entity.write_data_to_sim()
            sim.step()

        # Write state to shared memory
        shm['joint_pos'][:] = entity.data.joint_pos[0, arm_ids].cpu().numpy()
        shm['joint_vel'][:] = entity.data.joint_vel[0, arm_ids].cpu().numpy()
        _push_body_poses()

        iters += 1; shm['steps'][0] += 1
        if shm['steps'][0] >= MAX_STEPS:
            reset_ev.set()   # self-reset; sim will handle on next iter and signal reset_pol

        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            shm['sim_hz'][0] = iters / dt_rate
            iters = 0; t_rate = time.time()

        elapsed = time.time() - t
        if elapsed < dt:
            time.sleep(dt - elapsed)

    _close_shm(shm_handles)


# ── Policy process ────────────────────────────────────────────────────────────

def policy_process_fn(shm_names: dict, events: dict, device: str, checkpoint: str) -> None:
    """Pinocchio FK → obs → NN policy → writes EE target + peg pose."""
    from scipy.spatial.transform import Rotation
    from policy import PolicyAgent

    stop_ev   = events['stop']
    reset_pol = events['reset_policy']

    shm_handles, shm = _attach_shm(shm_names)

    robot        = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    EE_TO_PEG_ROT = Rotation.from_quat(EE_TO_PEG_QUAT_XYZW).as_matrix()
    hole_rot      = Rotation.from_quat(HOLE_QUAT_XYZW).as_matrix()

    q_home                   = kinova_deg_to_rad(HOME_DEG)
    home_ee_pos, home_ee_rot = robot.fk(q_home)
    home_ee_quat_xyzw        = Rotation.from_matrix(home_ee_rot).as_quat()

    print(f"[policy] Loading: {checkpoint}")
    agent = PolicyAgent(checkpoint, device=device)
    print("[policy] Ready")

    last_action     = np.zeros(6, dtype=np.float32)
    prev_target_pos = home_ee_pos.copy()

    dt = 1.0 / CONTROL_HZ
    iters = 0; t_rate = time.time()

    while not stop_ev.is_set():
        t = time.time()

        if reset_pol.is_set():
            reset_pol.clear()
            prev_target_pos = home_ee_pos.copy()
            last_action[:]  = 0.0

        # FK
        q_sim            = shm['joint_pos'].copy().astype(np.float64)
        ee_pos, ee_rot   = robot.fk(q_sim)
        peg_pos, peg_rot = ee_to_peg_world(ee_pos, ee_rot, EE_TO_PEG_POS, EE_TO_PEG_ROT)
        peg_quat_xyzw    = Rotation.from_matrix(peg_rot).as_quat()

        # Peg pose → viz
        shm['peg_sim'][:3] = peg_pos
        shm['peg_sim'][3:] = _xyzw_to_wxyz(peg_quat_xyzw)

        # Obs + policy
        obs    = np.concatenate([compute_peg_to_hole(peg_pos, peg_rot, HOLE_POS, hole_rot), last_action])
        action = agent.get_action(obs)
        last_action = action.copy()

        # Action → EE target
        raw_pos    = home_ee_pos + action[:3].astype(np.float64)
        delta_pos  = np.clip(raw_pos - prev_target_pos, -DELTA_POS_BOUND, DELTA_POS_BOUND)
        target_pos = np.clip(prev_target_pos + delta_pos,
                             home_ee_pos - TARGET_POS_BOUND, home_ee_pos + TARGET_POS_BOUND)
        prev_target_pos = target_pos

        shm['target'][:3] = target_pos
        shm['target'][3:] = home_ee_quat_xyzw

        iters += 1
        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            shm['policy_hz'][0] = iters / dt_rate
            iters = 0; t_rate = time.time()

        elapsed = time.time() - t
        if elapsed < dt:
            time.sleep(dt - elapsed)

    _close_shm(shm_handles)


# ── Viz process ───────────────────────────────────────────────────────────────

def viz_process_fn(shm_names: dict, events: dict, checkpoint: str) -> None:
    """Viser server: renders scene, GUI sliders, writes gains to shared memory."""
    import viser
    import viser.transforms as vtf
    from scipy.spatial.transform import Rotation
    from viewer import ViserMujocoScene, ViserRobotView

    stop_ev  = events['stop']
    reset_ev = events['reset']

    shm_handles, shm = _attach_shm(shm_names)

    # Compile arm model through Entity so body IDs match the sim process exactly
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
    def get_spec():
        return mujoco.MjSpec.from_file(str(_TORQUE_XML))
    _arm_cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0,0,0), joint_pos=DEMO_JOINT_POS, joint_vel={".*": 0.0}),
        collisions=(), spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=("joint_.*",), command_field="effort"),),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
    model_arm = Entity(_arm_cfg).compile()
    mj_model_peg  = mujoco.MjSpec.from_file(str(_PEG_XML)).compile()
    mj_model_hole = mujoco.MjSpec.from_file(str(_HOLE_XML)).compile()
    hole_rot      = Rotation.from_quat(HOLE_QUAT_XYZW).as_matrix()

    server    = viser.ViserServer(label="Kinova Sim — NN Policy")
    scene     = ViserMujocoScene.create(server, model_arm)
    sim_view  = scene.add_robot("sim",     color=(0.75, 0.75, 0.75, 1.00))
    peg_view  = ViserRobotView.create(server, mj_model_peg,  "peg_sim", color=(0.20, 0.60, 0.80, 1.00))
    hole_view = ViserRobotView.create(server, mj_model_hole, "hole",    color=(0.80, 0.40, 0.20, 1.00))

    hole_view.update_from_pose(HOLE_POS, _xyzw_to_wxyz(HOLE_QUAT_XYZW))

    scene.create_visualization_gui(camera_distance=1.2, camera_azimuth=135.0, camera_elevation=30.0)

    # Static hole-top site frame
    hole_top_world = HOLE_POS + hole_rot @ HOLE_TOP_SITE_OFFSET
    ht_q = Rotation.from_matrix(hole_rot).as_quat()
    server.scene.add_frame("/hole_top", axes_length=0.04, axes_radius=0.003,
                           position=tuple(float(v) for v in hole_top_world),
                           wxyz=(float(ht_q[3]), float(ht_q[0]), float(ht_q[1]), float(ht_q[2])))

    # Dynamic frames (updated in loop)
    target_frame  = server.scene.add_frame("/target_sim", axes_length=0.10, axes_radius=0.005,
                                           position=(0,0,0), wxyz=(1,0,0,0))
    peg_top_frame = server.scene.add_frame("/peg_top",    axes_length=0.04, axes_radius=0.003,
                                           position=(0,0,0), wxyz=(1,0,0,0))

    # GUI
    with server.gui.add_folder("Policy"):
        server.gui.add_text("Checkpoint",   initial_value=os.path.basename(checkpoint))
        txt_sim_hz    = server.gui.add_text("Sim rate",    initial_value="— Hz")
        txt_policy_hz = server.gui.add_text("Policy rate", initial_value="— Hz")
        txt_steps     = server.gui.add_text("Steps",       initial_value="0")
        reset_btn     = server.gui.add_button("Reset Sim")

    with server.gui.add_folder("Gains"):
        sl_kp_task  = server.gui.add_slider("Kp task",  min=0.0,  max=10.0,  step=0.1,  initial_value=KP_TASK)
        sl_kp_joint = server.gui.add_slider("Kp joint", min=0.0,  max=300.0, step=1.0,  initial_value=KP_JOINT)
        sl_kd_joint = server.gui.add_slider("Kd joint", min=0.0,  max=300.0, step=1.0,  initial_value=KD_JOINT)
        sl_damping  = server.gui.add_slider("Damping",  min=1e-3, max=1.0,   step=1e-3, initial_value=DAMPING)

    with server.gui.add_folder("IK Weights"):
        sl_pos_w  = server.gui.add_slider("Pos weight",   min=0.0,  max=1.0,  step=0.01, initial_value=POS_WEIGHT)
        sl_ori_w  = server.gui.add_slider("Ori weight",   min=0.0,  max=1.0,  step=0.01, initial_value=ORI_WEIGHT)
        sl_max_dq = server.gui.add_slider("Max dq (rad)", min=0.01, max=2.0,  step=0.01, initial_value=MAX_DQ)

    with server.gui.add_folder("Posture"):
        sl_post_w = server.gui.add_slider("Posture weight",   min=0.0, max=1.0, step=0.01, initial_value=POSTURE_WEIGHT)
        sl_jlim_w = server.gui.add_slider("Jnt limit weight", min=0.0, max=1.0, step=0.01, initial_value=JNT_LIMIT_WEIGHT)
        server.gui.add_text("Posture target (deg)", initial_value=", ".join(f"{v:.0f}" for v in HOME_DEG))

    @reset_btn.on_click
    def _(_): reset_ev.set()

    # Init gains
    shm['gains'][:] = _pack_gains(KP_TASK, KP_JOINT, KD_JOINT, DAMPING,
                                   POS_WEIGHT, ORI_WEIGHT, POSTURE_WEIGHT, JNT_LIMIT_WEIGHT, MAX_DQ)

    period = 1.0 / VIZ_HZ
    print("[viz] Ready, listening on :8080")

    while not stop_ev.is_set():
        t = time.time()

        # Arm — read from shared memory
        sim_view.update_from_arrays(shm['body_xpos'].copy(), shm['body_xmat'].copy())

        # Peg
        d = shm['peg_sim'].copy()
        peg_view.update_from_pose(d[:3], d[3:])

        # Peg-top site frame
        peg_rot_mat   = vtf.SO3(d[3:]).as_matrix()
        peg_top_world = d[:3] + peg_rot_mat @ PEG_TOP_SITE_OFFSET
        peg_top_frame.position = tuple(float(v) for v in peg_top_world)
        peg_top_frame.wxyz     = tuple(float(v) for v in d[3:])

        # Target frame (target is pos[3]+quat_xyzw[4])
        tgt = shm['target'].copy()
        q   = tgt[3:]  # xyzw
        target_frame.position = tuple(float(v) for v in tgt[:3])
        target_frame.wxyz     = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

        # Gains from sliders → shared memory
        shm['gains'][:] = _pack_gains(
            sl_kp_task.value, sl_kp_joint.value, sl_kd_joint.value, sl_damping.value,
            sl_pos_w.value, sl_ori_w.value, sl_post_w.value, sl_jlim_w.value, sl_max_dq.value,
        )

        txt_sim_hz.value    = f"{shm['sim_hz'][0]:.0f} Hz"
        txt_policy_hz.value = f"{shm['policy_hz'][0]:.0f} Hz"
        txt_steps.value     = str(int(shm['steps'][0]))

        elapsed = time.time() - t
        if elapsed < period:
            time.sleep(period - elapsed)

    server.stop()
    _close_shm(shm_handles)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--device",     default="cuda:0")
    args = parser.parse_args()

    # Get nbody via Entity.compile() — must match what the sim process uses.
    # Entity.compile() calls spec_fn() + MjSpec.compile() (no warp/GPU init).
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
    def _get_spec():
        return mujoco.MjSpec.from_file(str(_TORQUE_XML))
    _tmp_cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0,0,0), joint_pos=DEMO_JOINT_POS, joint_vel={".*": 0.0}),
        collisions=(), spec_fn=_get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=("joint_.*",), command_field="effort"),),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
    _tmp_entity = Entity(_tmp_cfg)
    _tmp_model  = _tmp_entity.compile()
    nbody = _tmp_model.nbody
    del _tmp_entity, _tmp_model, _tmp_cfg

    # Create shared memory
    _shm_specs = {
        'joint_pos':  ((7,),         np.float32),
        'joint_vel':  ((7,),         np.float32),
        'body_xpos':  ((nbody, 3),   np.float32),
        'body_xmat':  ((nbody, 9),   np.float32),
        'target':     ((7,),         np.float64),
        'peg_sim':    ((7,),         np.float64),
        'gains':      ((len(GAINS_KEYS),), np.float64),
        'sim_hz':     ((1,),         np.float64),
        'policy_hz':  ((1,),         np.float64),
        'steps':      ((1,),         np.int64),
    }
    shm_handles = {}
    shm_names   = {}
    for key, (shape, dtype) in _shm_specs.items():
        arr = np.zeros(shape, dtype=dtype)
        shm = SharedMemory(create=True, size=max(arr.nbytes, 8))
        np.ndarray(shape, dtype=dtype, buffer=shm.buf)[:] = arr
        shm_handles[key] = shm
        shm_names[key]   = (shm.name, shape, dtype)

    # Seed gains so sim has valid values before viz writes its first update
    gains_buf = np.ndarray((len(GAINS_KEYS),), dtype=np.float64,
                           buffer=shm_handles['gains'].buf)
    gains_buf[:] = _pack_gains(KP_TASK, KP_JOINT, KD_JOINT, DAMPING,
                                POS_WEIGHT, ORI_WEIGHT, POSTURE_WEIGHT, JNT_LIMIT_WEIGHT, MAX_DQ)

    events = {
        'stop':         mp.Event(),
        'reset':        mp.Event(),
        'reset_policy': mp.Event(),
    }

    processes = [
        mp.Process(target=sim_process_fn,    name="sim",
                   args=(shm_names, events, args.device),                    daemon=True),
        mp.Process(target=policy_process_fn, name="policy",
                   args=(shm_names, events, args.device, args.checkpoint),   daemon=True),
        mp.Process(target=viz_process_fn,    name="viz",
                   args=(shm_names, events, args.checkpoint),                daemon=True),
    ]

    print("Starting processes…")
    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nStopping…")
        events['stop'].set()
        for p in processes:
            p.join(timeout=5)
    finally:
        for shm in shm_handles.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        print("Done.")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
