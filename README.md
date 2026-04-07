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




nohup env CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=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 WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur CUDA_VISIBLE_DEVICES=1 uv run train Mjlab-Reach-Osc-Kinova --env.scene.num-envs 4096 --agent.max-iterations 1_000   --agent.wandb-project mjlab-kinova-tasks-torque --agent.experiment-name reach_osc > reach_osc.log 2>&1 &


nohup env WANDB_API_KEY=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 WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur CUDA_VISIBLE_DEVICES=1 uv run train Mjlab-Reach-Osc-Kinova --env.scene.num-envs 4096 --agent.max-iterations 1_000   --agent.wandb-project mjlab-kinova-tasks-torque --agent.experiment-name reach_osc > reach_osc.log 2>&1 &


uv run train Mjlab-Reach-Osc-Kinova --env.scene.num-envs 1024 --agent.max-iterations 1_0 --agent.experiment-name reach_osc




nohup env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 WANDB_API_KEY=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 WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur CUDA_VISIBLE_DEVICES=1 uv run train Mjlab-Reach-Osc-Kinova --env.scene.num-envs 1024 --agent.max-iterations 5_000   --agent.wandb-project mjlab-kinova-tasks-torque --agent.experiment-name reach_osc --video True --video-length 100 --video-interval 100 > reach_osc.log 2>&1 &



CUDA_VISIBLE_DEVICES=1 uv run play Mjlab-Reach-Osc-Kinova --viewer viser 
--checkpoint-file /media/cvlab/EXTDRIVE/sudhir/continual_learning/manipulation/logs/rsl_rl/reach_osc/2026-04-05_13-42-17/model_300.pt