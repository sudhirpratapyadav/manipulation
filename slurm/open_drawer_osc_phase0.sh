#!/bin/bash
#SBATCH --job-name=open_drawer_osc_phase0
#SBATCH --partition=1gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/svs_ald/sudhir/manipulation/logs/open_drawer_osc_phase0.log
#SBATCH --qos=1gpu
#SBATCH --time=24:00:00

# Phase 0 baseline run for the open-drawer OSC task.
# No model changes vs. the existing config — only difference is the new
# success_rate metric registered in the env cfg. Acts as the reference for
# all subsequent phases (Phase 1 init-pose DR, Phase 2 targeted DR, etc.).
# See docs/open_drawer_improvement_plan.md and docs/experiments_log.md.

echo "Running on: $(hostname)"
echo "Date: $(date)"
echo ""

cd /ihub/homedirs/svs_ald/sudhir/manipulation

env MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    WANDB_API_KEY=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 \
    WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur \
    CUDA_VISIBLE_DEVICES=0 \
    uv run train Mjlab-Open-Drawer-Osc-Kinova \
    --env.scene.num-envs 1024 \
    --agent.max-iterations 5_000 \
    --agent.wandb-project mjlab-kinova-tasks-osc \
    --agent.experiment-name open_drawer_osc_phase0 \
    --agent.run-name baseline \
    --agent.wandb-tags '("phase=0","baseline")' \
    --video True \
    --video-length 100 \
    --video-interval 100
