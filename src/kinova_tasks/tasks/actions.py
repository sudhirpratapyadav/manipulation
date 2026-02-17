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

        self._printed_ee = False  # for debug print

    @property
    def action_dim(self) -> int:
        return 6

    def process_actions(self, actions: torch.Tensor) -> None:
        # DEBUG: uncomment to print EE pose on first step
        # if not self._printed_ee:
        #     self._printed_ee = True
        #     ids, _ = self._entity.find_sites(self.cfg.frame_name)
        #     local_id = ids[0]
        #     ee_pos_w = self._entity.data.site_pos_w[0, local_id]
        #     ee_quat_w = self._entity.data.site_quat_w[0, local_id]
        #     env_origin = self._env.scene.env_origins[0]
        #     local_pos = ee_pos_w - env_origin
        #     print(f"[HomeRelativeIK] EE pos (local): {local_pos.tolist()}")
        #     print(f"[HomeRelativeIK] EE quat (wxyz): {ee_quat_w.tolist()}")
        self._raw_actions[:] = actions

        delta = actions.clone()
        delta[:, :3] *= self.cfg.pos_scale
        delta[:, 3:] *= self.cfg.ori_scale

        target_pos, target_quat = apply_delta_pose(
            self._home_pos, self._home_quat, delta
        )
        self._desired_pos[:] = target_pos
        self._desired_quat[:] = target_quat

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._desired_pos[env_ids] = self._home_pos[env_ids]
        self._desired_quat[env_ids] = self._home_quat[env_ids]

