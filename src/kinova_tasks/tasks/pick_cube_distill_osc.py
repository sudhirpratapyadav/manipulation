"""Phase A: state -> vision distillation cfg for Kinova pick OSC.

Builds on `pick_cube_vision_osc.py` (env) and the trained
`pick_cube_osc` PPO actor (teacher checkpoint at
wandb run jn3l22j9, model_4999.pt).

Algorithm: RSL-RL `Distillation` (DAgger-style, MSE on action mean).
Student: vision CNN (spatial-softmax) + MLP, same arch as the vision
PPO actor.  Teacher: state MLP 33 -> 512 -> 256 -> 128 -> 7, elu,
scalar-std Gaussian, with `obs_normalizer`.

Obs groups (resolved at runtime by RSL-RL):
  student: ("actor", "camera")  — proprio + RGB
  teacher: ("critic",)          — full 33D privileged state
                                  (joint_vel + ee_pose + gripper +
                                   ee_to_object + object_pos +
                                   object_to_goal + goal_pos + actions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl.config import RslRlBaseRunnerCfg, RslRlModelCfg

from kinova_tasks.tasks.pick_cube_vision_osc import (
    _VISION_CNN_CFG,
    _VISION_MODEL_CLS,
    kinova_pick_cube_vision_osc_env_cfg,
)


# Teacher arch matches the trained checkpoint
# (wandb run jn3l22j9, model_4999.pt).
_TEACHER_HIDDEN_DIMS = (512, 256, 128)
_TEACHER_ACTIVATION = "elu"


@dataclass
class RslRlDistillationAlgorithmCfg:
    """RSL-RL Distillation algorithm config (mirrors PPO cfg style)."""

    num_learning_epochs: int = 1
    """Distillation passes over the on-policy batch per iteration."""
    gradient_length: int = 15
    """Gradient accumulation length (matters for RNN students)."""
    learning_rate: float = 1.0e-3
    max_grad_norm: float | None = 1.0
    loss_type: str = "mse"
    """`mse` or `huber`."""
    optimizer: str = "adam"
    class_name: str = "Distillation"


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
    """Runner cfg for state->vision distillation."""

    class_name: str = "DistillationRunner"
    student: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=_VISION_CNN_CFG,
            class_name=_VISION_MODEL_CLS,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        )
    )
    teacher: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=_TEACHER_HIDDEN_DIMS,
            activation=_TEACHER_ACTIVATION,
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        )
    )
    algorithm: RslRlDistillationAlgorithmCfg = field(
        default_factory=RslRlDistillationAlgorithmCfg
    )
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "student": ("actor", "camera"),
            "teacher": ("critic",),
        }
    )


def kinova_pick_cube_distill_osc_env_cfg(play: bool = False):
    """Vision env (same as `pick_cube_vision_osc`) for distillation.

    Critic group keeps the full privileged state (used as teacher input);
    student reads `actor` + `camera`.
    """
    return kinova_pick_cube_vision_osc_env_cfg(play=play)


def kinova_pick_cube_distill_osc_runner_cfg() -> RslRlDistillationRunnerCfg:
    """Distillation runner cfg.  Phase A defaults from `plan.md`."""
    return RslRlDistillationRunnerCfg(
        experiment_name="kinova_pick_cube_distill_osc",
        wandb_project="mjlab-kinova-tasks-osc-vision",
        num_steps_per_env=24,
        max_iterations=2_000,
        save_interval=100,
    )
