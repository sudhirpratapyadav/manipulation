#!/bin/bash
#SBATCH --job-name=open_drawer_osc_phase2
#SBATCH --partition=1gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/svs_ald/sudhir/manipulation/logs/open_drawer_osc_phase2.log
#SBATCH --qos=1gpu
#SBATCH --time=24:00:00

# Phase 2: targeted drawer + arm DR on top of Phase 1 init-pose randomization.
# Knobs (defined in tasks/open_drawer_osc.py:_phase2_knobs):
#   - drawer slide frictionloss: U(0.005, 0.02)
#   - drawer slide damping:      U(0.5, 2.0)
#   - drawer base mass:          ±34.7% (log-uniform alpha)
#   - arm link mass:             ±4.77% (log-uniform alpha)
# Run as a job-step inside the holder allocation:
#   srun --jobid=<holder> --gres=gpu:1 -n1 --exclusive bash slurm/open_drawer_osc_phase2.sh

echo "Running on: $(hostname)"
echo "Date: $(date)"
echo ""

cd /ihub/homedirs/svs_ald/sudhir/manipulation

env MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    WANDB_API_KEY=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 \
    WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur \
    uv run train Mjlab-Open-Drawer-Osc-Kinova-Phase2 \
    --env.scene.num-envs 1024 \
    --agent.max-iterations 4_000 \
    --agent.wandb-project mjlab-kinova-tasks-osc \
    --agent.experiment-name open_drawer_osc_phase2 \
    --agent.run-name drawer_dr \
    --agent.wandb-tags '("phase=2","drawer_dr")' \
    --video True \
    --video-length 100 \
    --video-interval 100
