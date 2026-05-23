"""Operational Space Control (OSC) action space for task-space control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import (
  apply_delta_pose,
  compute_pose_error,
  quat_from_matrix,
)
from mjlab.utils.string import resolve_expr

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class OperationalSpaceActionCfg(ActionTermCfg):
  """Configuration for Operational Space Control (OSC) action space.

  OSC computes joint torques to track task-space targets using the full
  arm dynamics (mass matrix, gravity, Coriolis) via mjwarp:

    τ = Jᵀ Λ ẍ_des + bias + w_posture · Nᵀ τ_posture

  where:
    Λ = (J M⁻¹ Jᵀ)⁻¹   (task-space inertia)
    N = I − Jᵀ Λ J M⁻¹  (null-space projector)
    bias = qfrc_bias      (gravity + Coriolis from simulation)

  The action dimension is determined by the active objectives:

  - ``orientation_weight == 0`` -> 3D (position only)
  - ``orientation_weight > 0, use_relative_mode=True`` -> 6D (delta pose)
  - ``orientation_weight > 0, use_relative_mode=False`` -> 7D (absolute pose)
  """

  actuator_names: tuple[str, ...] | list[str]
  """Actuator name expressions to resolve the controlled joints."""

  frame_type: Literal["body", "site", "geom"] = "body"
  """Element type of the target frame."""

  frame_name: str
  """Name of the target frame element on the entity."""

  use_relative_mode: bool = True
  """If True, actions are deltas applied to the current frame pose.
  If False, actions are absolute targets."""

  delta_pos_scale: float = 1.0
  """Scaling factor for position components in relative mode (meters)."""

  delta_ori_scale: float = 1.0
  """Scaling factor for orientation components in relative mode (radians)."""

  position_weight: float = 1.0
  """Weight for the position objective."""

  orientation_weight: float = 1.0
  """Weight for the orientation objective (0 = position-only mode)."""

  kp_pos: float = 400.0
  """Proportional gain for position control."""

  kd_pos: float = 40.0
  """Derivative gain for position control."""

  kp_ori: float = 400.0
  """Proportional gain for orientation control."""

  kd_ori: float = 40.0
  """Derivative gain for orientation control."""

  damping: float = 1e-4
  """Damped-least-squares (Tikhonov) coefficient for the task-space inertia
  inversion: ``Λ = (J M⁻¹ Jᵀ + damping · I)⁻¹``.

  Larger values trade tracking accuracy for numerical stability near
  kinematic singularities (wrist alignment, fully-extended arm). The default
  ``1e-4`` is a mild regularizer that matches the previous hard-coded value;
  bump to ``1e-2``–``1e-1`` for explicit DLS behavior in tasks that drive the
  arm into / through singular configurations.
  """

  max_torque: float | list[float] | None = None
  """Per-joint torque clipping (Nm). None = no clipping."""

  posture_weight: float = 0.0
  """Weight for null-space posture regularization (0 = disabled)."""

  posture_target: dict[str, float] | None = None
  """Target joint positions for null-space posture control.

  Maps joint name patterns to target values (same format as
  ``EntityCfg.InitialStateCfg.joint_pos``). Joints not listed
  default to their initial (qpos0) value. Only used when
  ``posture_weight > 0``.
  """

  posture_kp: float = 10.0
  """Proportional gain for null-space posture control."""

  posture_kd: float = 2.0
  """Derivative gain for null-space posture control."""

  tau_offset_std_Nm: float = 0.0
  """Per-joint constant torque offset, drawn at episode reset.

  Each joint independently samples ``offset_j ~ Uniform(-σ, +σ)`` once per
  reset and the offset is added to ``compute_torques`` every step until the
  next reset. Models slow-varying biases like gravity-comp / friction-model
  mismatch on a real arm. ``0.0`` (default) disables the offset entirely.
  Set per-eval at sweep time; left at ``0.0`` during training.
  """

  tau_noise_std_Nm: float = 0.0
  """Per-step Gaussian torque noise σ, applied independently per joint
  every control step. Models PWM jitter / sensorless current-noise on a real
  arm. ``0.0`` (default) disables the noise. Set per-eval at sweep time.
  """

  action_noise_std: float = 0.0
  """Per-step Gaussian noise σ added to the raw action vector before any
  scaling/clip, in normalized action units (the policy's action space is
  [-1, 1] for the OSC delta channels and the gripper). Models communication
  channel noise on a real robot. ``0.0`` (default) disables the noise.
  """

  # ── Per-episode domain randomization of controller params ──────────────────
  # Each range is an absolute ``(lo, hi)`` bound sampled LOG-uniformly once per
  # episode reset, independently per environment. ``None`` (default) disables
  # randomization for that param and the scalar value above is used unchanged.
  #
  # Log-uniform: ``v = exp(Uniform(ln lo, ln hi))`` — multiplicative symmetry,
  # so e.g. halving and doubling the nominal are equally likely. Use for gains
  # and scales, which act multiplicatively on the closed-loop dynamics.
  dr_delta_pos_scale_range: tuple[float, float] | None = None
  """Log-uniform ``(lo, hi)`` for ``delta_pos_scale`` (m per unit action)."""

  dr_delta_ori_scale_range: tuple[float, float] | None = None
  """Log-uniform ``(lo, hi)`` for ``delta_ori_scale`` (rad per unit action)."""

  dr_kp_pos_range: tuple[float, float] | None = None
  """Log-uniform ``(lo, hi)`` for ``kp_pos``."""

  dr_kd_pos_range: tuple[float, float] | None = None
  """Log-uniform ``(lo, hi)`` for ``kd_pos``."""

  dr_kp_ori_range: tuple[float, float] | None = None
  """Log-uniform ``(lo, hi)`` for ``kp_ori``."""

  dr_kd_ori_range: tuple[float, float] | None = None
  """Log-uniform ``(lo, hi)`` for ``kd_ori``."""

  def build(self, env: ManagerBasedRlEnv) -> OperationalSpaceAction:
    return OperationalSpaceAction(self, env)


class OperationalSpaceAction(ActionTerm):
  """Converts task-space commands into joint torques via Operational Space Control.

  Uses mjwarp's ``crb`` to compute the mass matrix and reads ``qfrc_bias``
  (gravity + Coriolis) directly from simulation data for full dynamics
  compensation.
  """

  cfg: OperationalSpaceActionCfg
  _entity: Entity

  def __init__(self, cfg: OperationalSpaceActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    joint_ids, _ = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._num_joints = len(joint_ids)
    self._joint_dof_ids = self._entity.indexing.joint_v_adr[self._joint_ids]

    self._frame_type = cfg.frame_type
    self._resolve_frame(cfg.frame_name)

    if cfg.orientation_weight > 0:
      self._action_dim = 6 if cfg.use_relative_mode else 7
      self._task_dim = 6
    else:
      self._action_dim = 3
      self._task_dim = 3

    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._desired_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self._desired_quat = torch.zeros(self.num_envs, 4, device=self.device)
    self._desired_quat[:, 0] = 1.0

    # Torque clipping.
    if cfg.max_torque is not None:
      mt = cfg.max_torque
      if isinstance(mt, (int, float)):
        mt = [float(mt)] * self._num_joints
      self._max_torque = torch.tensor(mt, device=self.device, dtype=torch.float32)
    else:
      self._max_torque = None

    # Posture regularization target.
    q_target = self._entity.data.default_joint_pos[:, self._joint_ids].clone()
    if cfg.posture_target is not None:
      joint_names = tuple(self._entity.joint_names[i] for i in joint_ids)
      overrides = resolve_expr(cfg.posture_target, joint_names)
      for j, val in enumerate(overrides):
        if val is not None:
          q_target[:, j] = val
    self._posture_target = q_target

    nworld = self.num_envs
    nv = self._env.sim.mj_model.nv

    with wp.ScopedDevice(self._env.sim.wp_device):
      self._jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._jacr_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._point_wp = wp.zeros(nworld, dtype=wp.vec3)
      self._body_wp = wp.zeros(nworld, dtype=wp.int32)
      self._body_wp.fill_(self._body_id)

    self._jacp_torch = wp.to_torch(self._jacp_wp)
    self._jacr_torch = wp.to_torch(self._jacr_wp)
    self._point_torch = wp.to_torch(self._point_wp).view(nworld, 3)

    # Per-episode torque offset (drawn at reset, held until next reset).
    self._tau_offset = torch.zeros(
      self.num_envs, self._num_joints, device=self.device
    )

    # Per-episode controller-param tensors, shape (num_envs, 1) so they
    # broadcast over the task-space error/velocity tensors. Initialized to the
    # scalar cfg values; overwritten in reset() for params whose DR range is
    # set. The (lo, hi) tuples are cached as log-bounds for cheap sampling.
    def _col(v: float) -> torch.Tensor:
      return torch.full((self.num_envs, 1), float(v), device=self.device)

    self._delta_pos_scale = _col(cfg.delta_pos_scale)
    self._delta_ori_scale = _col(cfg.delta_ori_scale)
    self._kp_pos = _col(cfg.kp_pos)
    self._kd_pos = _col(cfg.kd_pos)
    self._kp_ori = _col(cfg.kp_ori)
    self._kd_ori = _col(cfg.kd_ori)

    self._dr_specs: tuple[tuple[torch.Tensor, tuple[float, float]], ...] = tuple(
      (tensor, rng)
      for tensor, rng in (
        (self._delta_pos_scale, cfg.dr_delta_pos_scale_range),
        (self._delta_ori_scale, cfg.dr_delta_ori_scale_range),
        (self._kp_pos, cfg.dr_kp_pos_range),
        (self._kd_pos, cfg.dr_kd_pos_range),
        (self._kp_ori, cfg.dr_kp_ori_range),
        (self._kd_ori, cfg.dr_kd_ori_range),
      )
      if rng is not None
    )

  # ── ActionTerm interface ────────────────────────────────────────────────────

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    if self.cfg.action_noise_std > 0.0:
      actions = actions + self.cfg.action_noise_std * torch.randn_like(actions)
    self._raw_actions[:] = actions

    frame_pos, frame_quat = self._get_frame_pose()
    if self._action_dim == 3:
      if self.cfg.use_relative_mode:
        self._desired_pos[:] = frame_pos + actions * self._delta_pos_scale
      else:
        self._desired_pos[:] = actions
      self._desired_quat[:] = frame_quat
    elif self._action_dim == 6:
      delta = actions.clone()
      delta[:, :3] *= self._delta_pos_scale
      delta[:, 3:] *= self._delta_ori_scale
      target_pos, target_quat = apply_delta_pose(frame_pos, frame_quat, delta)
      self._desired_pos[:] = target_pos
      self._desired_quat[:] = target_quat
    else:
      assert self._action_dim == 7
      self._desired_pos[:] = actions[:, :3]
      self._desired_quat[:] = actions[:, 3:7]

  def apply_actions(self) -> None:
    """Compute OSC torques and apply them as joint effort targets."""
    tau = self.compute_torques()
    self._entity.set_joint_effort_target(tau, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._desired_pos[env_ids] = 0.0
    self._desired_quat[env_ids] = 0.0
    self._desired_quat[env_ids, 0] = 1.0
    if self.cfg.tau_offset_std_Nm > 0.0:
      shape = self._tau_offset[env_ids].shape
      self._tau_offset[env_ids] = (
        torch.rand(shape, device=self.device) * 2.0 - 1.0
      ) * self.cfg.tau_offset_std_Nm
    else:
      self._tau_offset[env_ids] = 0.0

    # Per-episode log-uniform DR of enabled controller params.
    for tensor, (lo, hi) in self._dr_specs:
      log_lo, log_hi = math.log(lo), math.log(hi)
      shape = tensor[env_ids].shape
      u = torch.rand(shape, device=self.device) * (log_hi - log_lo) + log_lo
      tensor[env_ids] = torch.exp(u)

  # ── Core computation ────────────────────────────────────────────────────────

  def compute_torques(self) -> torch.Tensor:
    """Compute OSC joint torques for the current target.

    Returns:
      Joint torques of shape ``(num_envs, num_joints)``.
    """
    # Current frame pose and error.
    frame_pos, frame_quat = self._get_frame_pose()
    pos_error, rot_error = compute_pose_error(
      frame_pos, frame_quat, self._desired_pos, self._desired_quat
    )

    # Jacobian at current EE position.
    self._point_torch[:] = frame_pos
    self._compute_jacobian()
    jacp = self._jacp_torch[:, :, self._joint_dof_ids]  # (B, 3, n)
    jacr = self._jacr_torch[:, :, self._joint_dof_ids]  # (B, 3, n)

    # Current joint state.
    q = self._entity.data.joint_pos[:, self._joint_ids]   # (B, n)
    qd = self._entity.data.joint_vel[:, self._joint_ids]  # (B, n)

    # Task-space velocity.
    vel_pos = torch.einsum("bti,bi->bt", jacp, qd)  # (B, 3)

    # Build full Jacobian and desired acceleration.
    if self._task_dim == 6:
      vel_rot = torch.einsum("bti,bi->bt", jacr, qd)  # (B, 3)
      J = torch.cat([jacp, jacr], dim=1)               # (B, 6, n)
      w_pos = self.cfg.position_weight
      w_ori = self.cfg.orientation_weight
      ddx_des = torch.cat([
        w_pos * (self._kp_pos * pos_error - self._kd_pos * vel_pos),
        w_ori * (self._kp_ori * rot_error - self._kd_ori * vel_rot),
      ], dim=1)  # (B, 6)
    else:
      J = jacp  # (B, 3, n)
      ddx_des = self.cfg.position_weight * (
        self._kp_pos * pos_error - self._kd_pos * vel_pos
      )  # (B, 3)

    # Mass matrix for controlled joints: M (B, n, n).
    M = self._compute_mass_matrix()

    # Task-space inertia: Λ = (J M⁻¹ Jᵀ)⁻¹.
    M_inv = torch.linalg.inv(M)                                      # (B, n, n)
    J_T = J.transpose(1, 2)                                          # (B, n, task_dim)
    JMinvJT = torch.bmm(J, torch.bmm(M_inv, J_T))                   # (B, task_dim, task_dim)
    JMinvJT = JMinvJT + self.cfg.damping * torch.eye(
      self._task_dim, device=self.device
    ).unsqueeze(0)
    Lambda = torch.linalg.inv(JMinvJT)                               # (B, task_dim, task_dim)

    # Dynamically consistent pseudo-inverse: J† = M⁻¹ Jᵀ Λ.
    J_dyn_inv = torch.bmm(M_inv, torch.bmm(J_T, Lambda))            # (B, n, task_dim)

    # Task-space torques: τ_task = Jᵀ Λ ẍ_des.
    F = torch.bmm(Lambda, ddx_des.unsqueeze(-1)).squeeze(-1)         # (B, task_dim)
    tau_task = torch.bmm(J_T, F.unsqueeze(-1)).squeeze(-1)          # (B, n)

    # Bias compensation (gravity + Coriolis).
    bias_full = self._env.sim.data.qfrc_bias                         # (B, nv)
    bias = bias_full[:, self._joint_dof_ids]                         # (B, n)

    tau = tau_task + bias

    # Null-space posture control.
    if self.cfg.posture_weight > 0:
      tau_posture = (
        self.cfg.posture_kp * (self._posture_target - q)
        + self.cfg.posture_kd * (-qd)
      )  # (B, n)
      # N = I − Jᵀ Λ J M⁻¹  (dynamically consistent null-space projector transposed)
      I = torch.eye(self._num_joints, device=self.device).unsqueeze(0)
      N = I - torch.bmm(J_T, J_dyn_inv.transpose(1, 2))             # (B, n, n)
      tau_null = torch.bmm(N, tau_posture.unsqueeze(-1)).squeeze(-1) # (B, n)
      tau = tau + self.cfg.posture_weight * tau_null

    # Per-episode torque offset (zero unless tau_offset_std_Nm > 0 at reset).
    tau = tau + self._tau_offset

    # Per-step torque noise (zero unless tau_noise_std_Nm > 0 in cfg).
    if self.cfg.tau_noise_std_Nm > 0.0:
      tau = tau + self.cfg.tau_noise_std_Nm * torch.randn_like(tau)

    # Optional torque clipping.
    if self._max_torque is not None:
      tau = torch.clamp(tau, -self._max_torque, self._max_torque)

    return tau

  # ── Private ─────────────────────────────────────────────────────────────────

  def _compute_mass_matrix(self) -> torch.Tensor:
    """Call mjwarp.crb and return the joint sub-matrix M (B, n, n)."""
    with wp.ScopedDevice(self._env.sim.wp_device):
      mjwarp.crb(self._env.sim.wp_model, self._env.sim.wp_data)
    M_full = wp.to_torch(self._env.sim.wp_data.qM)  # (B, nv, nv)
    # Extract rows and columns for controlled DOFs.
    M = M_full[:, self._joint_dof_ids, :][:, :, self._joint_dof_ids]  # (B, n, n)
    # Symmetrize (crb may only fill one triangle).
    return 0.5 * (M + M.transpose(1, 2))

  def _resolve_frame(self, frame_name: str) -> None:
    if self._frame_type == "body":
      ids, _ = self._entity.find_bodies(frame_name)
      local_id = ids[0]
      self._frame_id = int(self._entity.indexing.body_ids[local_id].item())
      self._body_id = self._frame_id
    elif self._frame_type == "site":
      ids, _ = self._entity.find_sites(frame_name)
      local_id = ids[0]
      self._frame_id = int(self._entity.indexing.site_ids[local_id].item())
      self._body_id = int(self._env.sim.mj_model.site_bodyid[self._frame_id])
    elif self._frame_type == "geom":
      ids, _ = self._entity.find_geoms(frame_name)
      local_id = ids[0]
      self._frame_id = int(self._entity.indexing.geom_ids[local_id].item())
      self._body_id = int(self._env.sim.mj_model.geom_bodyid[self._frame_id])
    else:
      raise ValueError(f"Unknown frame_type: {self._frame_type}")

  def _get_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    data = self._env.sim.data
    if self._frame_type == "body":
      pos = data.xpos[:, self._frame_id]
      quat = data.xquat[:, self._frame_id]
    elif self._frame_type == "site":
      pos = data.site_xpos[:, self._frame_id]
      xmat = data.site_xmat[:, self._frame_id]
      quat = quat_from_matrix(xmat)
    else:
      assert self._frame_type == "geom"
      pos = data.geom_xpos[:, self._frame_id]
      xmat = data.geom_xmat[:, self._frame_id]
      quat = quat_from_matrix(xmat)
    return pos, quat

  def _compute_jacobian(self) -> None:
    with wp.ScopedDevice(self._env.sim.wp_device):
      mjwarp.jac(
        self._env.sim.wp_model,
        self._env.sim.wp_data,
        self._jacp_wp,
        self._jacr_wp,
        self._point_wp,
        self._body_wp,
      )
