#!/bin/bash
#SBATCH --job-name=open_drawer_osc_phase1
#SBATCH --partition=1gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/svs_ald/sudhir/manipulation/logs/open_drawer_osc_phase1.log
#SBATCH --qos=1gpu
#SBATCH --time=24:00:00

# Phase 1: wider initial-pose randomization on top of the Phase 0 baseline.
# Change vs. Phase 0:
#   - reset_robot_joints: joint_delta_deg 5° → 15°
#   - reset_base.pose_range: {} → ±2cm in x/y, ±2° yaw
# Same task code path, registered under task_id ...-Phase1. See plan §3.
#
# Run as a job-step inside the 8-GPU (or 3-GPU fallback) holder:
#   srun --jobid=<holder> --gres=gpu:1 -n1 --exclusive bash slurm/open_drawer_osc_phase1.sh

echo "Running on: $(hostname)"
echo "Date: $(date)"
echo ""

cd /ihub/homedirs/svs_ald/sudhir/manipulation

env MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    WANDB_API_KEY=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 \
    WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur \
    uv run train Mjlab-Open-Drawer-Osc-Kinova-Phase1 \
    --env.scene.num-envs 1024 \
    --agent.max-iterations 5_000 \
    --agent.wandb-project mjlab-kinova-tasks-osc \
    --agent.experiment-name open_drawer_osc_phase1 \
    --agent.run-name init_pose_dr \
    --agent.wandb-tags '("phase=1","init_pose_dr")' \
    --video True \
    --video-length 100 \
    --video-interval 100
