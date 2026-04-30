"""Lane B: headless async sim-only open-drawer eval (MuJoCo native).

The point of this driver is to test a trained checkpoint **outside mjlab**
under the same async timing structure a real Kinova would use:
- 50 Hz outer policy loop (inference + obs build).
- 500 Hz inner OSC loop (torque compute + ``mujoco.mj_step``).

This is a pass/fail signal that complements Lane A (the in-process mjlab
sweep): different physics engine (raw MuJoCo, not mujoco_warp) and
different control loop shape (async threads, not synchronous batched
vec-env steps). If a policy that's robust in Lane A drops to 0 here,
something is overfitting to mjlab specifics.

This script does **not** touch real hardware, viser, or keyboard. It runs
N episodes under nominal conditions plus a small set of (axis, value)
overrides that mirror Lane A's most sim2real-relevant axes (drawer mass,
drawer slide friction). Records per-episode terminal success and the
final handle-to-goal distance, writes a single JSON summary.

Usage (run from /ihub/homedirs/svs_ald/sudhir/manipulation):

    uv run python nn_policy/sim_open_drawer_osc_async.py \\
        --checkpoint logs/rsl_rl/open_drawer_osc_phase0/<run>/model_4900.pt \\
        --output docs/results/open_drawer_osc_phase0/lane_b.json \\
        --episodes-per-setting 4

References (do **not** modify):
- ``nn_policy/sim2real_open_drawer_osc.py`` — same task, full real+sim path.
- ``nn_policy/sim_osc.py`` — minimal sim-only async shape, no task logic.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
import torch
from scipy.spatial.transform import Rotation

# Self-contained: we duplicate the few constants + helpers we need from
# sim2real_open_drawer_osc.py rather than importing it. AGENT.md forbids
# modifying that file (it owns the user's hardware path), and importing it
# transitively pulls in `kortex_api`, which is not available on the
# cluster. The duplicated values must stay in sync with that script.
from kinova_tasks.assets.kinova_gen3.kinova_constants import (
    KINOVA_GEN3_GRIPPER_TORQUE_XML,
)
from kinova_tasks.assets.objects.articulated.drawer.drawer_constants import (
    DRAWER_XML,
)
from policy import PolicyAgent

# ── Duplicated from sim2real_open_drawer_osc.py — keep in sync. ──────────────
_TORQUE_XML = KINOVA_GEN3_GRIPPER_TORQUE_XML
_ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
_GRIPPER_JOINT_NAMES = [
    "right_driver_joint", "right_coupler_joint",
    "right_spring_link_joint", "right_follower_joint",
    "left_driver_joint", "left_coupler_joint",
    "left_spring_link_joint", "left_follower_joint",
]
_GRIPPER_OPEN_QPOS = np.zeros(8)
_GRIPPER_OPEN_CTRL = 0.0
_GRIPPER_DRIVER_MAX = 0.8

HOME_DEG = np.array([0.0, 30.0, 0.0, 90.0, 0.0, 60.0, -90.0])
DRAWER_POS_X = (0.7, 0.9)
DRAWER_POS_Y = (-0.05, 0.15)
DRAWER_POS_Z = 0.45
DRAWER_INIT_SLIDE_LO = -0.02
DRAWER_INIT_SLIDE_HI = 0.0
DRAWER_GOAL_SLIDE_LO = -0.25
DRAWER_GOAL_SLIDE_HI = -0.15
HANDLE_OFFSET_X = -0.044
HANDLE_OFFSET_Y = 0.0
HANDLE_OFFSET_Z = 0.0
SUCCESS_THRESH = 0.02

KP_POS = 50.0
KD_POS = 10.0
KP_ORI = 50.0
KD_ORI = 10.0
POSTURE_KP = 10.0
POSTURE_KD = 2.0
POSTURE_WEIGHT = 0.0
MAX_JOINT_TORQUE = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])
DELTA_POS_SCALE = 0.01
DELTA_ORI_SCALE = 0.02

PHYSICS_DT = 0.002
TARGET_HZ = 10
OSC_HZ = 500


def kinova_deg_to_rad(deg: np.ndarray) -> np.ndarray:
    s = deg.copy()
    s[s > 180.0] -= 360.0
    return np.deg2rad(s)


def handle_world_pos(drawer_base_pos: np.ndarray, slide_q: float) -> np.ndarray:
    return drawer_base_pos + np.array(
        [slide_q + HANDLE_OFFSET_X, HANDLE_OFFSET_Y, HANDLE_OFFSET_Z]
    )


def goal_handle_world_pos(drawer_base_pos: np.ndarray, goal_slide: float) -> np.ndarray:
    return drawer_base_pos + np.array(
        [goal_slide + HANDLE_OFFSET_X, HANDLE_OFFSET_Y, HANDLE_OFFSET_Z]
    )


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


def _pose_error_6d(target_pos, target_quat_xyzw, cur_pos, cur_rot):
    pos_err = target_pos - cur_pos
    ori_err = Rotation.from_matrix(
        Rotation.from_quat(target_quat_xyzw).as_matrix() @ cur_rot.T
    ).as_rotvec()
    return np.concatenate([pos_err, ori_err])


def compute_osc_torques(robot, target_pos, target_quat_xyzw, q, dq, *,
                        gains, posture_target):
    ee_pos, ee_rot = robot.fk(q)
    error = _pose_error_6d(target_pos, target_quat_xyzw, ee_pos, ee_rot)
    J = robot.jacobian(q)
    ee_vel = J @ dq
    ddx_des = np.empty(6)
    ddx_des[:3] = gains["kp_pos"] * error[:3] + gains["kd_pos"] * (0.0 - ee_vel[:3])
    ddx_des[3:] = gains["kp_ori"] * error[3:] + gains["kd_ori"] * (0.0 - ee_vel[3:])
    M, nle, J_dot_dq = robot.dynamics(q, dq)
    M_inv = np.linalg.inv(M)
    Lambda = np.linalg.inv(J @ M_inv @ J.T)
    J_dyn_inv = M_inv @ J.T @ Lambda
    F = Lambda @ (ddx_des - J_dot_dq)
    N = np.eye(7) - J.T @ J_dyn_inv.T
    tau_posture = (gains["posture_kp"] * (posture_target - q)
                   + gains["posture_kd"] * (0.0 - dq))
    tau = J.T @ F + nle + gains["posture_weight"] * (N @ tau_posture)
    return np.clip(tau, -MAX_JOINT_TORQUE, MAX_JOINT_TORQUE)


def build_obs(
    q: np.ndarray,
    dq: np.ndarray,
    ee_pos: np.ndarray,
    ee_rot: np.ndarray,
    gripper_driver_pos: float,
    handle_pos_world: np.ndarray,
    goal_handle_pos_world: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """Build 33-D observation matching the open_drawer_osc training checkpoint."""
    from mjlab.utils.lab_api.math import (
        quat_from_matrix as _quat_from_matrix,
        axis_angle_from_quat as _axis_angle_from_quat,
    )
    R_t = torch.tensor(ee_rot, dtype=torch.float32).unsqueeze(0)
    quat = _quat_from_matrix(R_t)
    ee_axis_angle = _axis_angle_from_quat(quat)[0].numpy()

    joint_vel_obs = dq.astype(np.float32)
    ee_pose_obs = np.concatenate([ee_pos, ee_axis_angle])
    gripper_obs = np.array([gripper_driver_pos / _GRIPPER_DRIVER_MAX], dtype=np.float32)
    ee_to_object_obs = handle_pos_world - ee_pos
    object_pos_obs = handle_pos_world.copy()
    object_to_goal_obs = goal_handle_pos_world - handle_pos_world
    goal_pos_obs = goal_handle_pos_world.copy()

    return np.concatenate([
        joint_vel_obs, ee_pose_obs, gripper_obs,
        ee_to_object_obs, object_pos_obs, object_to_goal_obs,
        goal_pos_obs, last_action,
    ]).astype(np.float32)

OSC_SUBSTEPS = OSC_HZ // TARGET_HZ
EPISODE_LEN_S = 10.0  # matches training cfg.episode_length_s


# ---------------------------------------------------------------------------
# Per-episode setting (one (axis, value) override).
# ---------------------------------------------------------------------------


@dataclass
class Setting:
    """One ``mjModel`` override applied just before an episode starts.

    The key is a slot in our compiled ``mjModel`` (look up the index/addr
    once, mutate per episode). Settings come in pairs of pre-resolved
    fields so the rollout code doesn't need to know XML names.
    """

    name: str
    drawer_mass_scale: float = 1.0          # multiplies drawer-base body mass
    drawer_slide_friction: float | None = None  # absolute frictionloss; None=keep XML
    drawer_slide_damping: float | None = None   # absolute damping; None=keep XML

    def apply(self, mj_model: mujoco.MjModel, *, drawer_base_body_id: int,
              drawer_slide_dofadr: int, baseline: dict) -> None:
        # Mass: scale the baseline (per-XML) mass; do NOT chain.
        mj_model.body_mass[drawer_base_body_id] = (
            baseline["body_mass"] * self.drawer_mass_scale
        )
        # Friction / damping: abs override, falling back to the XML baseline.
        mj_model.dof_frictionloss[drawer_slide_dofadr] = (
            self.drawer_slide_friction
            if self.drawer_slide_friction is not None
            else baseline["dof_frictionloss"]
        )
        mj_model.dof_damping[drawer_slide_dofadr] = (
            self.drawer_slide_damping
            if self.drawer_slide_damping is not None
            else baseline["dof_damping"]
        )


def default_settings() -> list[Setting]:
    """Small set of variations that overlap Lane A's most sim2real-relevant axes."""
    return [
        Setting("nominal"),
        Setting("drawer_mass_2x", drawer_mass_scale=2.0),
        Setting("drawer_mass_5x", drawer_mass_scale=5.0),
        Setting("drawer_friction_2x", drawer_slide_friction=0.02),
        Setting("drawer_friction_4x", drawer_slide_friction=0.04),
        Setting("drawer_damping_2x", drawer_slide_damping=2.0),
    ]


# ---------------------------------------------------------------------------
# Sim build + index resolution.
# ---------------------------------------------------------------------------


def _compile_drawer_scene() -> mujoco.MjModel:
    """Build the same Kinova + drawer scene that ``sim2real_open_drawer_osc.py`` uses."""
    spec = mujoco.MjSpec.from_file(str(_TORQUE_XML))
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = np.array([0.0, 0.0, 0.01])
    floor.pos = np.array([0.0, 0.0, 0.0])
    floor.rgba = np.array([0.8, 0.8, 0.8, 1.0])
    floor.contype = 1
    floor.conaffinity = 1
    drawer_spec = mujoco.MjSpec.from_file(str(DRAWER_XML))
    spec.attach(drawer_spec, prefix="drawer/", suffix="",
                frame=spec.worldbody.add_frame())
    mj_model = spec.compile()
    # Match training physics settings (open_drawer_osc.SimulationCfg).
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 20
    mj_model.opt.impratio = 10.0
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    mj_model.opt.timestep = PHYSICS_DT
    return mj_model


@dataclass
class SceneIdx:
    arm_q: np.ndarray
    arm_dq: np.ndarray
    arm_ctrl: np.ndarray
    gripper_q: np.ndarray
    gripper_dq: np.ndarray
    gripper_ctrl: int
    drawer_base_body: int
    drawer_base_mocap: int
    drawer_slide_qposadr: int
    drawer_slide_dofadr: int
    drawer_handle_site: int
    baseline: dict = field(default_factory=dict)


def _resolve_indices(mj_model: mujoco.MjModel) -> SceneIdx:
    arm_q = np.array([mj_model.joint(n).qposadr.item() for n in _ARM_JOINT_NAMES])
    arm_dq = np.array([mj_model.joint(n).dofadr.item() for n in _ARM_JOINT_NAMES])
    arm_ctrl = np.array([mj_model.actuator(n).id for n in _ARM_JOINT_NAMES])
    grp_q = np.array([mj_model.joint(n).qposadr.item() for n in _GRIPPER_JOINT_NAMES])
    grp_dq = np.array([mj_model.joint(n).dofadr.item() for n in _GRIPPER_JOINT_NAMES])
    grp_ctrl = mj_model.actuator("fingers_actuator").id
    drawer_base_body = mj_model.body("drawer/drawer_base").id
    drawer_base_mocap = mj_model.body("drawer/drawer_base").mocapid.item()
    drawer_slide_jnt = mj_model.joint("drawer/drawer_slide")
    drawer_slide_qposadr = drawer_slide_jnt.qposadr.item()
    drawer_slide_dofadr = drawer_slide_jnt.dofadr.item()
    drawer_handle_site = mj_model.site("drawer/object_site").id
    baseline = {
        "body_mass": float(mj_model.body_mass[drawer_base_body]),
        "dof_frictionloss": float(mj_model.dof_frictionloss[drawer_slide_dofadr]),
        "dof_damping": float(mj_model.dof_damping[drawer_slide_dofadr]),
    }
    return SceneIdx(
        arm_q=arm_q, arm_dq=arm_dq, arm_ctrl=arm_ctrl,
        gripper_q=grp_q, gripper_dq=grp_dq, gripper_ctrl=grp_ctrl,
        drawer_base_body=drawer_base_body,
        drawer_base_mocap=drawer_base_mocap,
        drawer_slide_qposadr=drawer_slide_qposadr,
        drawer_slide_dofadr=drawer_slide_dofadr,
        drawer_handle_site=drawer_handle_site,
        baseline=baseline,
    )


# ---------------------------------------------------------------------------
# Per-episode rollout.
# ---------------------------------------------------------------------------


@dataclass
class EpisodeOutcome:
    setting: str
    seed: int
    success: bool
    final_error_m: float
    final_slide: float
    goal_slide: float
    n_steps: int
    wallclock_s: float


def _reset_scene(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    idx: SceneIdx,
    home_qpos: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Place arm at home, gripper open, drawer at random (x, y, slide). Returns
    (drawer_base_pos, init_slide, goal_slide)."""
    mj_data.qpos[idx.arm_q] = home_qpos
    mj_data.qvel[idx.arm_dq] = 0.0
    mj_data.ctrl[idx.arm_ctrl] = 0.0
    mj_data.qpos[idx.gripper_q] = _GRIPPER_OPEN_QPOS
    mj_data.qvel[idx.gripper_dq] = 0.0
    mj_data.ctrl[idx.gripper_ctrl] = _GRIPPER_OPEN_CTRL

    dx = float(rng.uniform(*DRAWER_POS_X))
    dy = float(rng.uniform(*DRAWER_POS_Y))
    dz = DRAWER_POS_Z
    mj_data.mocap_pos[idx.drawer_base_mocap] = np.array([dx, dy, dz])
    mj_data.mocap_quat[idx.drawer_base_mocap] = np.array([1.0, 0.0, 0.0, 0.0])
    init_slide = float(rng.uniform(DRAWER_INIT_SLIDE_LO, DRAWER_INIT_SLIDE_HI))
    mj_data.qpos[idx.drawer_slide_qposadr] = init_slide
    mj_data.qvel[idx.drawer_slide_qposadr] = 0.0
    goal_slide = float(rng.uniform(DRAWER_GOAL_SLIDE_LO, DRAWER_GOAL_SLIDE_HI))
    mujoco.mj_forward(mj_model, mj_data)
    return mj_data.mocap_pos[idx.drawer_base_mocap].copy(), init_slide, goal_slide


def _osc_thread(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    idx: SceneIdx,
    pin_arm: PinocchioArm,
    posture_target: np.ndarray,
    state: dict,
    stop: threading.Event,
) -> None:
    """500 Hz OSC + mj_step. Reads target/gains from ``state``."""
    inner_dt = 1.0 / OSC_HZ
    gains = {
        "kp_pos": KP_POS, "kd_pos": KD_POS,
        "kp_ori": KP_ORI, "kd_ori": KD_ORI,
        "posture_kp": POSTURE_KP, "posture_kd": POSTURE_KD,
        "posture_weight": POSTURE_WEIGHT,
    }
    while not stop.is_set():
        t0 = time.time()
        target = state["osc_target"]
        q = mj_data.qpos[idx.arm_q].copy()
        dq = mj_data.qvel[idx.arm_dq].copy()
        tau = compute_osc_torques(
            pin_arm, target[:3], target[3:], q, dq,
            gains=gains, posture_target=posture_target,
        )
        mj_data.ctrl[idx.arm_ctrl] = tau
        # Match training: gripper ctrl is held at open. The training task's
        # GripperCtrlAction wrote to joint_effort_target, not ctrl, for the
        # tendon-based actuator — so the ctrl-driven gripper stayed open.
        mj_data.ctrl[idx.gripper_ctrl] = _GRIPPER_OPEN_CTRL
        mujoco.mj_step(mj_model, mj_data)
        elapsed = time.time() - t0
        if elapsed < inner_dt:
            time.sleep(inner_dt - elapsed)


def _run_episode(
    mj_model: mujoco.MjModel,
    idx: SceneIdx,
    pin_arm: PinocchioArm,
    policy: PolicyAgent,
    setting: Setting,
    seed: int,
    home_qpos: np.ndarray,
    posture_target: np.ndarray,
) -> EpisodeOutcome:
    rng = np.random.default_rng(seed)
    setting.apply(
        mj_model,
        drawer_base_body_id=idx.drawer_base_body,
        drawer_slide_dofadr=idx.drawer_slide_dofadr,
        baseline=idx.baseline,
    )
    mj_data = mujoco.MjData(mj_model)
    drawer_base_pos, init_slide, goal_slide = _reset_scene(
        mj_model, mj_data, idx, home_qpos, rng
    )
    goal_pos = goal_handle_world_pos(drawer_base_pos, goal_slide)

    # Initial OSC target = home EE pose (from FK; matches sim2real script setup).
    ee_pos0, ee_rot0 = pin_arm.fk(home_qpos)
    q_xyzw0 = Rotation.from_matrix(ee_rot0).as_quat()
    state = {
        "osc_target": np.concatenate([ee_pos0, q_xyzw0]).astype(np.float64),
    }

    stop = threading.Event()
    osc_t = threading.Thread(
        target=_osc_thread,
        args=(mj_model, mj_data, idx, pin_arm, posture_target, state, stop),
        daemon=True,
    )
    osc_t.start()

    period = 1.0 / TARGET_HZ
    last_action = np.zeros(7, dtype=np.float32)
    n_steps = int(EPISODE_LEN_S * TARGET_HZ)
    wall0 = time.time()

    final_err = float("inf")
    for step in range(n_steps):
        t0 = time.time()
        q = mj_data.qpos[idx.arm_q].copy()
        dq = mj_data.qvel[idx.arm_dq].copy()
        ee_pos, ee_rot = pin_arm.fk(q)
        gripper_driver = float(mj_data.qpos[idx.gripper_q[0]])
        slide_q = float(mj_data.qpos[idx.drawer_slide_qposadr])
        handle_pos = handle_world_pos(drawer_base_pos, slide_q)

        obs = build_obs(
            q, dq, ee_pos, ee_rot, gripper_driver,
            handle_pos, goal_pos, last_action,
        )
        action = policy.get_action(obs).astype(np.float32)
        last_action[:] = action

        # Map the 6D OSC action to a new target.
        tgt_pos = ee_pos + action[:3] * DELTA_POS_SCALE
        delta_rot = Rotation.from_rotvec(action[3:6] * DELTA_ORI_SCALE)
        tgt_quat_xyzw = (delta_rot * Rotation.from_matrix(ee_rot)).as_quat()
        state["osc_target"] = np.concatenate([tgt_pos, tgt_quat_xyzw])

        err = float(np.linalg.norm(handle_pos - goal_pos))
        final_err = err
        if err < SUCCESS_THRESH:
            # Hold the target; let physics settle for a bit so dwell counts.
            time.sleep(0.2)
            break

        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    stop.set()
    osc_t.join(timeout=1.0)

    # Recompute final error post-hold so we capture the settled state.
    slide_q = float(mj_data.qpos[idx.drawer_slide_qposadr])
    handle_pos = handle_world_pos(drawer_base_pos, slide_q)
    final_err = float(np.linalg.norm(handle_pos - goal_pos))

    return EpisodeOutcome(
        setting=setting.name,
        seed=seed,
        success=final_err < SUCCESS_THRESH,
        final_error_m=final_err,
        final_slide=slide_q,
        goal_slide=goal_slide,
        n_steps=step + 1,
        wallclock_s=time.time() - wall0,
    )


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to RSL-RL PPO open-drawer checkpoint (.pt).")
    parser.add_argument("--output", required=True,
                        help="Path to results JSON (will be overwritten).")
    parser.add_argument("--episodes-per-setting", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[lane_b] checkpoint={args.checkpoint}")
    print(f"[lane_b] output={out_path}")

    mj_model = _compile_drawer_scene()
    idx = _resolve_indices(mj_model)
    home_qpos = kinova_deg_to_rad(HOME_DEG)
    pin_arm = PinocchioArm(str(_TORQUE_XML), ee_frame="pinch_site")
    posture_target = home_qpos.copy()

    policy = PolicyAgent(args.checkpoint, device=args.device)
    if policy.obs_dim != 33:
        raise ValueError(f"checkpoint obs_dim={policy.obs_dim}, expected 33")
    if policy.action_dim != 7:
        raise ValueError(f"checkpoint action_dim={policy.action_dim}, expected 7")

    outcomes: list[EpisodeOutcome] = []
    for setting in default_settings():
        print(f"[lane_b] setting={setting.name}")
        for k in range(args.episodes_per_setting):
            seed = args.seed + 1000 * abs(hash(setting.name)) % (2 ** 31) + k
            o = _run_episode(
                mj_model, idx, pin_arm, policy, setting, seed,
                home_qpos, posture_target,
            )
            outcomes.append(o)
            print(f"    ep{k} seed={seed} success={o.success} "
                  f"err={o.final_error_m:.3f} slide={o.final_slide:.3f}/"
                  f"{o.goal_slide:.3f} n={o.n_steps} t={o.wallclock_s:.1f}s")

    by_setting: dict[str, dict] = {}
    for s in {o.setting for o in outcomes}:
        s_outs = [o for o in outcomes if o.setting == s]
        successes = sum(1 for o in s_outs if o.success)
        by_setting[s] = {
            "n_episodes": len(s_outs),
            "n_successes": successes,
            "success_rate": successes / len(s_outs),
            "mean_final_error_m": float(np.mean([o.final_error_m for o in s_outs])),
        }

    summary = {
        "checkpoint": args.checkpoint,
        "episodes_per_setting": args.episodes_per_setting,
        "by_setting": by_setting,
        "episodes": [o.__dict__ for o in outcomes],
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[lane_b] wrote {out_path}")
    for s, r in by_setting.items():
        print(f"  {s:24s}  success={r['success_rate']:.2f}  "
              f"({r['n_successes']}/{r['n_episodes']})  err={r['mean_final_error_m']:.3f}m")


if __name__ == "__main__":
    main()
