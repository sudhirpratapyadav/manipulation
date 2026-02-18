"""Differential IK action with targets expressed relative to a home pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import DifferentialIKAction, DifferentialIKActionCfg
from mjlab.utils.lab_api.math import apply_delta_pose

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class HomeRelativeIKActionCfg(DifferentialIKActionCfg):
    """Differential IK where actions are offsets from a configured home EE pose.

    home_pos is specified relative to the robot base (env-local frame).
    It is automatically offset by each env's origin for the global IK solve.

    target_pos = (home_pos + env_origin) + action[:3] * pos_scale
    target_quat = delta_quat(action[3:6] * ori_scale) * home_quat

    Zero action = target at home pose.
    """

    home_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Home EE position (x, y, z) relative to robot base."""

    home_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Home EE orientation as quaternion (w, x, y, z)."""

    pos_scale: float = 0.1
    """Scaling for position offsets (meters)."""

    ori_scale: float = 0.1
    """Scaling for orientation offsets (radians)."""

    max_pos_delta: float = 0.0
    """Maximum position delta from current EE position (meters). 0 = no movement."""

    def build(self, env: ManagerBasedRlEnv) -> HomeRelativeIKAction:
        return HomeRelativeIKAction(self, env)


class HomeRelativeIKAction(DifferentialIKAction):
    """IK action: policy outputs 6D offsets from home pose → absolute IK target."""

    cfg: HomeRelativeIKActionCfg

    def __init__(
        self, cfg: HomeRelativeIKActionCfg, env: ManagerBasedRlEnv
    ) -> None:
        cfg.use_relative_mode = True
        cfg.orientation_weight = max(cfg.orientation_weight, 1.0)
        super().__init__(cfg, env)

        # Home pose in local (robot base) frame
        home_pos_local = torch.tensor(
            cfg.home_pos, device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        # Offset by each env's origin to get global coordinates
        env_origins = self._env.scene.env_origins  # (num_envs, 3)
        self._home_pos = home_pos_local + env_origins

        self._home_quat = torch.tensor(
            cfg.home_quat, device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        # Store previous desired position for delta-based clipping
        self._prev_desired_pos = self._home_pos.clone()

        self._printed_ee = False  # for debug print

    @property
    def action_dim(self) -> int:
        return 6

    def process_actions(self, actions: torch.Tensor) -> None:
        # DEBUG: Print step counter
        if not hasattr(self, '_step_counter'):
            self._step_counter = 0
        self._step_counter += 1
        print(f"[ACTION CLIP DEBUG] step={self._step_counter}")
        print(f"  actions[0]={actions[0, :3].tolist()}")
        print(f"  prev_desired[0]={self._prev_desired_pos[0].tolist()}")
        print(f"  home[0]={self._home_pos[0].tolist()}")

        # DEBUG: Check input actions
        if torch.isnan(actions).any():
            nan_envs = torch.where(torch.isnan(actions).any(dim=1))[0][:3]
            print(f"[ACTION CLIP DEBUG] INPUT actions has NaN in envs: {nan_envs.tolist()}")

        self._raw_actions[:] = actions

        delta = actions.clone()
        delta[:, :3] *= self.cfg.pos_scale
        delta[:, 3:] *= self.cfg.ori_scale

        # Compute target position from home-relative action
        target_pos, target_quat = apply_delta_pose(
            self._home_pos, self._home_quat, delta
        )

        # DEBUG: Check home position
        if torch.isnan(self._home_pos).any():
            nan_envs = torch.where(torch.isnan(self._home_pos).any(dim=1))[0][:3]
            print(f"[ACTION CLIP DEBUG] home_pos has NaN in envs: {nan_envs.tolist()}")

        if torch.isnan(target_pos).any():
            nan_envs = torch.where(torch.isnan(target_pos).any(dim=1))[0][:3]
            print(f"[ACTION CLIP DEBUG] target_pos has NaN in envs: {nan_envs.tolist()}")
            for env_id in nan_envs:
                print(f"  env {env_id}: target={target_pos[env_id].tolist()}, home={self._home_pos[env_id].tolist()}")

        # Compute delta from PREVIOUS action target to new target
        delta_from_prev = target_pos - self._prev_desired_pos

        # DEBUG: Check delta
        if torch.isnan(delta_from_prev).any():
            nan_envs = torch.where(torch.isnan(delta_from_prev).any(dim=1))[0][:3]
            print(f"[ACTION CLIP DEBUG] delta_from_prev has NaN in envs: {nan_envs.tolist()}")

        print(f"  target[0]={target_pos[0].tolist()}")
        print(f"  delta_from_prev[0]={delta_from_prev[0].tolist()}")
        # Print current EE position
        ee_pos, ee_quat = self._get_frame_pose()
        print(f"  current_ee[0]={ee_pos[0].tolist()}")

        # Clip delta to be within max_pos_delta box
        delta_clipped = torch.clamp(
            delta_from_prev,
            min=-self.cfg.max_pos_delta,
            max=self.cfg.max_pos_delta
        )

        print(f"  delta_clipped[0]={delta_clipped[0].tolist()}")

        # Apply clipped target: prev_desired_pos + clipped_delta
        clipped_target_pos = self._prev_desired_pos + delta_clipped

        print(f"  clipped_target[0]={clipped_target_pos[0].tolist()}")

        # DEBUG: Check final output
        if torch.isnan(clipped_target_pos).any():
            nan_envs = torch.where(torch.isnan(clipped_target_pos).any(dim=1))[0][:3]
            print(f"[ACTION CLIP DEBUG] clipped_target_pos has NaN in envs: {nan_envs.tolist()}")

        # Store for next iteration
        self._prev_desired_pos[:] = clipped_target_pos

        self._desired_pos[:] = clipped_target_pos
        self._desired_quat[:] = target_quat

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._desired_pos[env_ids] = self._home_pos[env_ids]
        self._desired_quat[env_ids] = self._home_quat[env_ids]
        self._prev_desired_pos[env_ids] = self._home_pos[env_ids]

