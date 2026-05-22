# Kinova Gen3 Manipulation Tasks

This repository contains manipulation tasks for the Kinova Gen3 robot arm using mjlab. The tasks are designed for reinforcement learning research in robotic manipulation, focusing on object lifting with a parallel gripper.

## Overview

This package provides:
- **Kinova Gen3 robot model** with Robotiq 2F-85 parallel gripper
- **Lift cube task** with position control
- **Manager-based environment** configuration for easy customization
- **PPO training** setup with tested hyperparameters

## Installation

### Prerequisites
- Python 3.10-3.13
- mjlab >= 1.1.0

### Install from source

```bash
# Clone or navigate to this repository
cd manipulation

# Install the package in development mode
pip install -e .
```

Alternatively, using `uv`:
```bash
uv pip install -e .
```

### Local vs cluster setup (pyproject divergence)

The local workstation and the SLURM cluster (`svs_ald` / dgx2) currently
need **different pyproject configurations** because of a CUDA driver
mismatch:

| Where | GPU driver | Max CUDA runtime | torch       | mjlab branch        |
| ----- | ---------- | ---------------- | ----------- | ------------------- |
| Local | recent     | cu128+           | `>=2.7`     | `sudhir/main`       |
| dgx2  | 550.54.15  | **cu124 only**   | `==2.6.0`   | `probe-local`       |

**Why:** Upstream `mjlab>=1.3.0` requires `torch>=2.7`, but PyTorch's
`+cu124` wheels stop at `torch==2.6.0`. The dgx2 driver 550 does not
support cu128/cu130 wheels (needs driver ≥ 555 for cu128, ≥ 580 for
cu130), so the cluster is pinned to `cu124` and therefore to
`torch<2.7`. The cluster's local `mjlab` is parked on the
`probe-local` branch which carries one extra commit (`ae21bc7f probe`)
that downpins mjlab's own `torch>=2.7` floor to `torch>=2.6` so the
resolver succeeds.

**Cluster-local pyproject overrides** (kept as unstaged edits on
`svs_ald:~/sudhir/manipulation/pyproject.toml`, NOT committed):

```toml
# dependencies: drop kortex-api (not on dgx2), add torch pin
"torch>=2.6.0,<2.7.0",

# uv.sources: add cu124 index, no protobuf override
[tool.uv.sources]
mjlab = { path = "../mjlab", editable = true }
torch = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

**Resolution plan:** ask cluster admin to upgrade the dgx2 driver from
550.54.15 to 570.x or newer (A100 + Rocky 8.6 supports this; no kernel
update needed). Once driver ≥ 555 is in place, the cluster can install
`torch>=2.7+cu128`, the local-mjlab `probe-local` branch can be
retired, and both environments collapse onto a single shared
pyproject.

## Available Tasks

Once installed, the task will be automatically registered with mjlab:

- **Mjlab-Lift-Cube-Kinova**: Object lifting task with Kinova Gen3 + gripper

## Usage

### Training

Train the lift task using the mjlab CLI:

```bash
train Mjlab-Lift-Cube-Kinova
```

Training options:
```bash
# Specify number of parallel environments
train Mjlab-Lift-Cube-Kinova --env.scene.num-envs 8192

# Use specific GPU(s)
train Mjlab-Lift-Cube-Kinova --gpu-ids 0 1

# Modify training parameters
train Mjlab-Lift-Cube-Kinova --runner.max-iterations 10000
```

### Evaluation

Play/evaluate a trained policy:

```bash
# Using a wandb checkpoint
play Mjlab-Lift-Cube-Kinova --wandb-run-path your-org/your-project/run-id

# Using a local checkpoint
play Mjlab-Lift-Cube-Kinova --checkpoint-path /path/to/checkpoint.pt
```

### Listing Tasks

List all available mjlab tasks:
```bash
list_envs
```

## Task Details

### Kinova Gen3 Lift Cube

**Objective**: Reach, grasp, and lift a cube to a target height.

**Robot**:
- 7-DOF Kinova Gen3 arm
- Robotiq 2F-85 parallel gripper
- Position control with delta actions

**Observations** (actor/critic):
- Joint positions (7 arm + gripper)
- Joint velocities
- End-effector to cube distance
- Cube to goal distance
- Previous actions

**Actions**:
- Joint position deltas (scale: 0.04)
- 7 arm joints + 2 gripper driver joints

**Rewards**:
- Reaching the cube (staged reward)
- Lifting the cube to target height
- Action smoothness
- Joint position limit penalty
- Joint velocity penalty (curriculum)

**Terminations**:
- Time limit: 20 seconds
- End-effector ground collision

**Domain Randomization**:
- Fingertip friction (slide, spin, roll)
- Object pose randomization

**Training Parameters**:
- Default parallel environments: 4096
- Episode length: 20 seconds
- Control frequency: 50 Hz (decimation=4)
- Network: (512, 256, 128) with ELU activation
- Max iterations: 5,000
- Expected training time: ~1-2 hours on modern GPU

## Repository Structure

```
manipulation/
├── pyproject.toml              # Package configuration
├── README.md                   # This file
└── src/
    └── kinova_lift/
        ├── __init__.py         # Task registration
        ├── env_cfgs.py         # Environment configuration
        ├── rl_cfg.py           # RL training parameters
        └── kinova_gen3/
            ├── kinova_constants.py  # Robot definition
            └── xmls/
                ├── gen3_gripper.xml # MuJoCo model
                └── assets/          # STL meshes
```

## Customization

### Modifying the Environment

Edit `src/kinova_lift/env_cfgs.py` to customize:
- Observation space
- Reward weights
- Object properties (mass, size)
- Domain randomization parameters
- Number of parallel environments

### Modifying Training

Edit `src/kinova_lift/rl_cfg.py` to adjust:
- Network architecture
- Learning rate and schedule
- PPO hyperparameters
- Training iterations

### Adding New Tasks

Follow the pattern in this repository:
1. Create environment configuration (inherit from base mjlab tasks)
2. Create RL configuration
3. Register task in `__init__.py`

## Development Roadmap

Future tasks to add:
- [ ] Push button task
- [ ] Open door task
- [ ] Peg-in-hole task
- [ ] Pick and place task
- [ ] Vision-based variants (RGB/Depth)
- [ ] Torque control variants

## References

- [mjlab](https://github.com/braxlab/mjlab) - MuJoCo + Warp robotics framework
- [Kinova Gen3](https://www.kinovarobotics.com/product/gen3-robots) - Robot specifications
- [Robotiq 2F-85](https://robotiq.com/products/2f85-140-adaptive-robot-gripper) - Gripper specifications

## License

This project follows the same license as mjlab (Apache 2.0).

## Citation

If you use this work in your research, please cite:

```bibtex
@software{kinova_lift_mjlab,
  title={Kinova Gen3 Manipulation Tasks for mjlab},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/manipulation}
}
```

uv run play Mjlab-Peg-In-Hole-Kinova --viewer viser --num-envs 4 --agent random

CUDA_VISIBLE_DEVICES=1 uv run play Mjlab-Peg-In-Hole-Kinova --viewer viser --num-envs 4 --checkpoint-file /media/cvlab/EXTDRIVE/sudhir/continual_learning/manipulation/logs/rsl_rl/peg_in_hole/2026-02-27_22-44-00/model_100.pt

nohup env WANDB_API_KEY=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 WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur CUDA_VISIBLE_DEVICES=1 uv run train Mjlab-Peg-In-Hole-Kinova --env.scene.num-envs 4096 --agent.max-iterations 1_000   --agent.wandb-project mjlab-kinova-tasks --agent.experiment-name peg_in_hole --enable-nan-guard True > mjlab_train_peg_in_hole.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 uv run play Mjlab-Peg-In-Hole-Kinova --viewer viser --agent random



nohup env CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=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 WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur CUDA_VISIBLE_DEVICES=1 uv run train Mjlab-Reach-Osc-Kinova --env.scene.num-envs 4096 --agent.max-iterations 1_000   --agent.wandb-project mjlab-kinova-tasks-torque --agent.experiment-name reach_osc > reach_osc.log 2>&1 &


nohup env WANDB_API_KEY=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 WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur CUDA_VISIBLE_DEVICES=1 uv run train Mjlab-Reach-Osc-Kinova --env.scene.num-envs 4096 --agent.max-iterations 1_000   --agent.wandb-project mjlab-kinova-tasks-torque --agent.experiment-name reach_osc > reach_osc.log 2>&1 &


uv run train Mjlab-Reach-Osc-Kinova --env.scene.num-envs 1024 --agent.max-iterations 1_0 --agent.experiment-name reach_osc




nohup env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 WANDB_API_KEY=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 WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur CUDA_VISIBLE_DEVICES=1 uv run train Mjlab-Reach-Osc-Kinova --env.scene.num-envs 1024 --agent.max-iterations 5_000   --agent.wandb-project mjlab-kinova-tasks-torque --agent.experiment-name reach_osc --video True --video-length 100 --video-interval 100 > reach_osc.log 2>&1 &



CUDA_VISIBLE_DEVICES=1 uv run play Mjlab-Reach-Osc-Kinova --viewer viser 
--checkpoint-file /media/cvlab/EXTDRIVE/sudhir/continual_learning/manipulation/logs/rsl_rl/reach_osc/2026-04-05_13-42-17/model_300.pt




env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 WANDB_API_KEY=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 WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur CUDA_VISIBLE_DEVICES=1 uv run train Mjlab-Peg-In-Hole-Osc-Kinova --env.scene.num-envs 1024 --agent.max-iterations 5_00   --agent.wandb-project mjlab-kinova-tasks-torque --agent.experiment-name peg_in_hole_osc --video True --video-length 100 --video-interval 100


uv run play Mjlab-Peg-In-Hole-Osc-Kinova --num-envs 1 --viewer viser 
--checkpoint-file 



CUDA_VISIBLE_DEVICES=1 uv run play Mjlab-Pick-Cube-Osc-Kinova --viewer viser --num-envs 4 --agent random

env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 WANDB_API_KEY=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 WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur CUDA_VISIBLE_DEVICES=1 uv run train Mjlab-Pick-Cube-Osc-Kinova --env.scene.num-envs 1024 --agent.max-iterations 5_00   --agent.wandb-project mjlab-kinova-tasks-osc --agent.experiment-name pick_cube_osc --video True --video-length 100 --video-interval 100 --agent.resume True --agent.load-run "2026-04-10_18-30-12" --agent.load-checkpoint "model_499.pt"



CUDA_VISIBLE_DEVICES=1 uv run play Mjlab-Open-Door-Osc-Kinova --viewer viser --num-envs 4 --agent random


env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 WANDB_API_KEY=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 WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur CUDA_VISIBLE_DEVICES=1 uv run train Mjlab-Open-Door-Osc-Kinova --env.scene.num-envs 1024 --agent.max-iterations 5_000   --agent.wandb-project mjlab-kinova-tasks-osc --agent.experiment-name open_door_osc --video True --video-length 100 --video-interval 100


CUDA_VISIBLE_DEVICES=1 uv run play Mjlab-Open-Drawer-Osc-Kinova --viewer viser --num-envs 4 --agent random

# load from wandb checkpoint
CUDA_VISIBLE_DEVICES=1 uv run play Mjlab-Open-Drawer-Osc-Kinova --viewer viser --num-envs 4 --wandb-run-path <entity>/<project>/<run-id>
# load specific checkpoint
CUDA_VISIBLE_DEVICES=1 uv run play Mjlab-Open-Drawer-Osc-Kinova --viewer viser --num-envs 4 --wandb-run-path <entity>/<project>/<run-id> --wandb-checkpoint-name model_4000.pt

# example (open_door_osc run)
# note: set WANDB_API_KEY explicitly to avoid .netrc credentials overriding
WANDB_API_KEY=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 CUDA_VISIBLE_DEVICES=1 uv run play Mjlab-Open-Door-Osc-Kinova --viewer viser --num-envs 4 --wandb-run-path sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/5pro663s


WANDB_API_KEY=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 CUDA_VISIBLE_DEVICES=1 uv run play Mjlab-Open-Drawer-Osc-Kinova --viewer viser --num-envs 4 --wandb-run-path sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/5pro663s



WANDB_API_KEY=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 CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 uv run play Mjlab-Pick-Cube-Distill-Osc-Kinova --viewer viser  --num-envs 4  --checkpoint-file logs/rsl_rl/kinova_pick_cube_distill_osc/2026-05-03_13-25-13_A_02_full/model_700.pt


# === Phase A distillation: state -> vision (DAgger, MSE on actions) ===
# teacher = trained pick_cube_osc PPO actor (wandb run jn3l22j9)
# launch (foreground)
env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 WANDB_API_KEY=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 WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur CUDA_VISIBLE_DEVICES=1 uv run train-distill --teacher-ckpt wandb/run-20260430_220905-jn3l22j9/files/model_4999.pt --env.scene.num-envs 1024 --agent.max-iterations 2000 --agent.save-interval 100 --agent.wandb-project mjlab-kinova-tasks-osc-vision --agent.experiment-name kinova_pick_cube_distill_osc --agent.run-name A_03_cams64 --agent.wandb-tags '("phase_a","dagger_mse","teacher_jn3l22j9","cams64")' --video True --video-length 100 --video-interval 200

# launch (background, log to file)
nohup env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 WANDB_API_KEY=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 WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur CUDA_VISIBLE_DEVICES=1 uv run train-distill --teacher-ckpt wandb/run-20260430_220905-jn3l22j9/files/model_4999.pt --env.scene.num-envs 1024 --agent.max-iterations 2000 --agent.save-interval 100 --agent.wandb-project mjlab-kinova-tasks-osc-vision --agent.experiment-name kinova_pick_cube_distill_osc --agent.run-name A_03_cams64 --agent.wandb-tags '("phase_a","dagger_mse","teacher_jn3l22j9","cams64")' --video True --video-length 100 --video-interval 200 > logs/distill_A_03_cams64.log 2>&1 &

# play a saved distillation checkpoint
WANDB_API_KEY=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 CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 uv run play Mjlab-Pick-Cube-Distill-Osc-Kinova --viewer viser --num-envs 4 --checkpoint-file logs/rsl_rl/kinova_pick_cube_distill_osc/<TS>_<run-name>/model_1999.pt


# === Phase A.MCR — frozen pretrained ResNet-50 distillation ===
# Plan: docs/sim2real/plan_v2.md.  Encoder: src/kinova_tasks/encoders/mcr_encoder.py.
# Results: docs/sim2real/exp_tracker.md (Phase A.MCR section).
#
# Verdict (after MCR_01..R3M_09): R3M-DROID weights >> MCR weights for this task,
# despite Vakil 2025's ranking saying the opposite. Frozen R3M-DROID with the
# default (512,256,128) head and lr=3e-4 is the production setup.  Layer4
# unfreezing HURTS at fixed env-step budget; head width does NOT matter.
#
# IMPORTANT: launching with `uv run` may fail with "kortex_api timeout" if uv
# tries to refresh deps.  Use the direct binary `.venv/bin/train-distill` instead.

# One-time setup: download both pretrained ResNet-50 checkpoints (~90 MB each).
mkdir -p assets/mcr
curl -L -o assets/mcr/mcr_resnet50.pth \
  "https://huggingface.co/GqJiang/robots-pretrain-robots/resolve/main/mcr_resnet50.pth"
curl -L -o assets/mcr/r3mdroid_resnet50.pth \
  "https://huggingface.co/GqJiang/robots-pretrain-robots/resolve/main/r3mdroid_resnet50.pth"

# === Available task variants (all use same env, same teacher, vary the student) ===
#   Mjlab-Pick-Cube-Distill-Mcr-Osc-Kinova            avg-pool, hd=(512,256,128) — production base
#   Mjlab-Pick-Cube-Distill-Mcr-Ss-Osc-Kinova         spatial-softmax over layer4
#   Mjlab-Pick-Cube-Distill-Mcr-Widehead-Osc-Kinova   hd=(1024,512,256,128)
#   Mjlab-Pick-Cube-Distill-Mcr-Smallhead-Osc-Kinova  hd=(256,128)
#   Mjlab-Pick-Cube-Distill-Mcr-Ll4-Osc-Kinova        layer4 unfrozen — needs --env.scene.num-envs 512 (OOM at 1024)
# To switch encoder weights between MCR and R3M-DROID, set MCR_WEIGHTS_PATH:
#   MCR_WEIGHTS_PATH=assets/mcr/mcr_resnet50.pth         (default if unset)
#   MCR_WEIGHTS_PATH=assets/mcr/r3mdroid_resnet50.pth    (recommended)

# === The recommended run: R3M-DROID, frozen, default head, lr=3e-4, 5000 iters ===
# (background, with wandb)
nohup env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  MCR_WEIGHTS_PATH=assets/mcr/r3mdroid_resnet50.pth \
  WANDB_API_KEY=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 \
  WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur \
  CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/train-distill \
    --task-id Mjlab-Pick-Cube-Distill-Mcr-Osc-Kinova \
    --teacher-ckpt wandb/run-20260430_220905-jn3l22j9/files/model_4999.pt \
    --env.scene.num-envs 1024 \
    --agent.max-iterations 5000 \
    --agent.save-interval 100 \
    --agent.wandb-project mjlab-kinova-tasks-osc-vision \
    --agent.experiment-name kinova_pick_cube_distill_mcr_osc \
    --agent.run-name R3M_REPRO \
    --agent.wandb-tags '("phase_a","r3m","frozen_resnet50","avg_pool","layernorm","r3mdroid","teacher_jn3l22j9","lr3e-4")' \
    --agent.algorithm.learning-rate 3e-4 \
    --video True --video-length 100 --video-interval 200 \
  > logs/distill_R3M_REPRO.log 2>&1 &

# === Smoke (32 envs × 30 iters, no wandb, ~5 minutes wall clock) ===
env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  MCR_WEIGHTS_PATH=assets/mcr/r3mdroid_resnet50.pth \
  CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/train-distill \
    --task-id Mjlab-Pick-Cube-Distill-Mcr-Osc-Kinova \
    --teacher-ckpt wandb/run-20260430_220905-jn3l22j9/files/model_4999.pt \
    --env.scene.num-envs 32 \
    --agent.max-iterations 30 \
    --agent.save-interval 30 \
    --agent.experiment-name kinova_pick_cube_distill_mcr_osc_smoke \
    --agent.run-name R3M_smoke \
    --agent.logger tensorboard

# === Play a saved MCR/R3M distillation checkpoint ===
WANDB_API_KEY=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 CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 MCR_WEIGHTS_PATH=assets/mcr/r3mdroid_resnet50.pth \
  uv run play Mjlab-Pick-Cube-Distill-Mcr-Osc-Kinova --viewer viser --num-envs 4 \
  --checkpoint-file logs/rsl_rl/kinova_pick_cube_distill_mcr_osc/<TS>_<run-name>/model_<N>.pt