"""mjlab-flavored DistillationRunner.

Mirrors the patches in `mjlab.rl.runner.MjlabOnPolicyRunner`:
  - persists env's `common_step_counter` across save/load,
  - migrates legacy checkpoint key names if needed,
  - respects `upload_model` flag for W&B artifact uploads.

Not subclassed from `MjlabOnPolicyRunner` because that class extends
`OnPolicyRunner` (PPO) — Distillation needs `DistillationRunner` as its
parent so the algo construction and `learn()` gating are correct.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.distillation_runner import DistillationRunner

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabDistillationRunner(DistillationRunner):
    """DistillationRunner with mjlab env-state save/load and ckpt migration."""

    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        # Strip None-valued optional configs so MLPModel doesn't choke.
        for key in ("student", "teacher"):
            if key in train_cfg:
                for opt in ("cnn_cfg", "distribution_cfg"):
                    if train_cfg[key].get(opt) is None:
                        train_cfg[key].pop(opt, None)
                if train_cfg[key].get("rnn_type") is None:
                    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                        train_cfg[key].pop(opt, None)
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos=None) -> None:
        env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
        infos = {**(infos or {}), "env_state": env_state}
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.cfg.get("upload_model", True):
            self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

        # Legacy `model_state_dict` -> split actor/critic state dicts.
        if "model_state_dict" in loaded_dict:
            print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
            model_state_dict = loaded_dict.pop("model_state_dict")
            actor_state_dict, critic_state_dict = {}, {}
            for key, value in model_state_dict.items():
                if key.startswith("actor."):
                    actor_state_dict[key.replace("actor.", "mlp.")] = value
                elif key.startswith("actor_obs_normalizer."):
                    actor_state_dict[
                        key.replace("actor_obs_normalizer.", "obs_normalizer.")
                    ] = value
                elif key in ["std", "log_std"]:
                    actor_state_dict[key] = value
                if key.startswith("critic."):
                    critic_state_dict[key.replace("critic.", "mlp.")] = value
                elif key.startswith("critic_obs_normalizer."):
                    critic_state_dict[
                        key.replace("critic_obs_normalizer.", "obs_normalizer.")
                    ] = value
            loaded_dict["actor_state_dict"] = actor_state_dict
            loaded_dict["critic_state_dict"] = critic_state_dict

        # rsl-rl 4.x -> 5.x distribution key rename.
        actor_sd = loaded_dict.get("actor_state_dict", {})
        if "std" in actor_sd:
            actor_sd["distribution.std_param"] = actor_sd.pop("std")
        if "log_std" in actor_sd:
            actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

        # mjlab's play.py passes `load_cfg={"actor": True}` (PPO jargon).
        # Distillation only knows "student"/"teacher", so translate.
        if load_cfg is not None and "actor" in load_cfg:
            load_cfg = dict(load_cfg)
            load_cfg["student"] = load_cfg.pop("actor")

        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]

        infos = loaded_dict.get("infos") or {}
        if "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"][
                "common_step_counter"
            ]
        return infos
