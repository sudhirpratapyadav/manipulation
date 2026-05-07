"""Phase A (MCR variant): state -> vision distillation with a frozen MCR
ResNet-50 encoder instead of the from-scratch spatial-softmax CNN.

Mirrors `pick_cube_distill_osc.py` exactly except:

  * Student `cnn_cfg` carries `weights_path` for the MCR checkpoint.
  * Student `class_name` = `kinova_tasks.encoders.mcr_encoder:MCRCNNModel`.
  * Student `hidden_dims` widened to (512, 256, 128) — the MCR encoder
    feature is 4096-D (2 × 2048 from wrist+d455), so the first MLP layer
    needs to be wider than the (256, 256, 128) used with the
    spatial-softmax CNN's 128-D output.

Env is the same vision env (`pick_cube_vision_osc`) — cameras still
render at 64×64; the encoder upsamples to 224×224 internally.

See `docs/sim2real/plan_v2.md`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from mjlab.rl.config import RslRlModelCfg

from kinova_tasks.tasks.pick_cube_distill_osc import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    _TEACHER_ACTIVATION,
    _TEACHER_HIDDEN_DIMS,
)
from kinova_tasks.tasks.pick_cube_vision_osc import (
    kinova_pick_cube_vision_osc_env_cfg,
)


_MCR_MODEL_CLS = "kinova_tasks.encoders.mcr_encoder:MCRCNNModel"

# Default location for the MCR checkpoint.  Can be overridden at runtime via
# the `MCR_WEIGHTS_PATH` env var (read inside the encoder).
_DEFAULT_MCR_WEIGHTS = "assets/mcr/mcr_resnet50.pth"


def _resolve_mcr_weights_path() -> str:
    """Return the MCR weights path stored in the cfg.

    Resolution order:
      1. `MCR_WEIGHTS_PATH` env var (also re-checked inside the encoder so
         env-var users don't need to set it before importing this module).
      2. Module default (`assets/mcr/mcr_resnet50.pth`).

    The path is *not* required to exist at config time — the encoder errors
    out with a download hint if the file is missing at construction.  This
    keeps the task registry import-safe on machines that haven't downloaded
    the weights yet.
    """
    return os.environ.get("MCR_WEIGHTS_PATH", _DEFAULT_MCR_WEIGHTS)


def kinova_pick_cube_distill_mcr_osc_env_cfg(play: bool = False):
    """Vision env (same as `pick_cube_vision_osc`) for MCR distillation.

    Identical to `kinova_pick_cube_distill_osc_env_cfg` — no env-side change
    is needed; only the student's encoder swaps.
    """
    return kinova_pick_cube_vision_osc_env_cfg(play=play)


def _runner_cfg(
    experiment_name: str,
    pool: str,
    hidden_dims: tuple[int, ...] = (512, 256, 128),
    unfreeze_layers: tuple[str, ...] = (),
) -> RslRlDistillationRunnerCfg:
    weights_path = _resolve_mcr_weights_path()
    cnn_cfg = {"weights_path": weights_path, "pool": pool}
    if unfreeze_layers:
        cnn_cfg["unfreeze_layers"] = list(unfreeze_layers)
    return RslRlDistillationRunnerCfg(
        experiment_name=experiment_name,
        wandb_project="mjlab-kinova-tasks-osc-vision",
        num_steps_per_env=24,
        max_iterations=5_000,
        save_interval=100,
        student=RslRlModelCfg(
            hidden_dims=hidden_dims,
            activation="elu",
            obs_normalization=True,
            cnn_cfg=cnn_cfg,
            class_name=_MCR_MODEL_CLS,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        teacher=RslRlModelCfg(
            hidden_dims=_TEACHER_HIDDEN_DIMS,
            activation=_TEACHER_ACTIVATION,
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        algorithm=RslRlDistillationAlgorithmCfg(),
        obs_groups={
            "student": ("actor", "camera"),
            "teacher": ("critic",),
        },
    )


def kinova_pick_cube_distill_mcr_osc_runner_cfg() -> RslRlDistillationRunnerCfg:
    """Distillation runner cfg using a frozen MCR ResNet-50 student encoder
    with global avg-pool (standard MCR usage)."""
    return _runner_cfg("kinova_pick_cube_distill_mcr_osc", pool="avg")


def kinova_pick_cube_distill_mcr_ss_osc_runner_cfg() -> RslRlDistillationRunnerCfg:
    """MCR variant where the layer4 feature map (2048×7×7) is reduced via
    spatial-softmax instead of global avg-pool.  Preserves spatial
    localization info that avg-pool throws away.  Output is 8192-D
    (2× wider than avg-pool variant)."""
    return _runner_cfg("kinova_pick_cube_distill_mcr_ss_osc", pool="spatial_softmax")


def kinova_pick_cube_distill_mcr_widehead_osc_runner_cfg() -> RslRlDistillationRunnerCfg:
    """MCR avg-pool variant with a wider MLP head (1024, 512, 256, 128).

    Tests whether MCR_02's slow convergence is bottlenecked by a
    too-narrow head trying to project 4096-D MCR features into 7-D
    actions.  Same encoder, same teacher, same lr.
    """
    return _runner_cfg(
        "kinova_pick_cube_distill_mcr_widehead_osc",
        pool="avg",
        hidden_dims=(1024, 512, 256, 128),
    )


def kinova_pick_cube_distill_mcr_smallhead_osc_runner_cfg() -> RslRlDistillationRunnerCfg:
    """MCR/R3M avg-pool with a *smaller* MLP head (256, 128).

    Tests whether the 4096-D pretrained feature is so directly usable
    that a much smaller head suffices.  If this matches R3M_05's
    convergence rate, head capacity was overkill.

    Designed to load either MCR or R3M weights (set MCR_WEIGHTS_PATH).
    """
    return _runner_cfg(
        "kinova_pick_cube_distill_mcr_smallhead_osc",
        pool="avg",
        hidden_dims=(256, 128),
    )


def kinova_pick_cube_distill_mcr_ll4_osc_runner_cfg() -> RslRlDistillationRunnerCfg:
    """MCR avg-pool with the last ResNet stage (`layer4`) unfrozen.

    All earlier stages stay frozen.  Adds ~15M trainable params (the
    layer4 bottleneck blocks).  Tests whether partial fine-tuning lets
    the encoder adapt to pick-cube geometry that the frozen DROID-pretrained
    features don't directly encode.

    Designed to load either MCR or R3M weights (set MCR_WEIGHTS_PATH).
    """
    return _runner_cfg(
        "kinova_pick_cube_distill_mcr_ll4_osc",
        pool="avg",
        unfreeze_layers=("layer4",),
    )
