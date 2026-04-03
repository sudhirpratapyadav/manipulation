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
        # if not hasattr(self, '_step_counter'):
        #     self._step_counter = 0
        # self._step_counter += 1

        self._raw_actions[:] = actions

        delta = actions.clone()
        delta[:, :3] *= self.cfg.pos_scale
        delta[:, 3:] *= self.cfg.ori_scale

        # Compute target position from home-relative action
        target_pos, target_quat = apply_delta_pose(
            self._home_pos, self._home_quat, delta
        )

        # Compute delta from PREVIOUS action target to new target
        delta_from_prev = target_pos - self._prev_desired_pos

        # Clip delta to be within max_pos_delta box
        delta_clipped = torch.clamp(
            delta_from_prev,
            min=-self.cfg.max_pos_delta,
            max=self.cfg.max_pos_delta
        )

        # Apply clipped target: prev_desired_pos + clipped_delta
        clipped_target_pos = self._prev_desired_pos + delta_clipped

        # --- Debug print for env 0 ---
        # fmt = lambda t: [f"{v:.4f}" for v in t[0].tolist()]
        # print(
        #     f"[DBG step={self._step_counter:4d}]"
        #     f"  raw_action={fmt(actions)}"
        #     f"  scaled_delta={fmt(delta)}"
        #     f"  home={fmt(self._home_pos)}"
        #     f"  target={fmt(target_pos)}"
        #     f"  d_prev={fmt(delta_from_prev)}"
        #     f"  d_clip={fmt(delta_clipped)}"
        #     f"  final_target={fmt(clipped_target_pos)}"
        # )

        # Store for next iteration
        self._prev_desired_pos[:] = clipped_target_pos

        self._desired_pos[:] = clipped_target_pos
        # Always target home orientation so IK tracks orientation without drift,
        # regardless of what orientation action the policy outputs.
        self._desired_quat[:] = self._home_quat

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._desired_pos[env_ids] = self._home_pos[env_ids]
        self._desired_quat[env_ids] = self._home_quat[env_ids]
        self._prev_desired_pos[env_ids] = self._home_pos[env_ids]

