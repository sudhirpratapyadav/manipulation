#!/bin/bash
#SBATCH --job-name=baseline_dr_envs8192
#SBATCH --partition=1gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/svs_ald/sudhir/manipulation/logs/baseline_dr_envs8192.log
#SBATCH --qos=1gpu
#SBATCH --time=48:00:00

# Hyperparam variant A: baseline_dr with num_envs=8192 (vs reference 1024).
# 8x more parallel envs ~= 8x more samples per gradient update at the cost of
# ~8x wall-clock per iter. See docs/hyperparam_experiments.md.
#
# Run as a job-step inside the holder:
#   srun --jobid=<holder> --gres=gpu:1 -n1 --exclusive bash \
#        slurm/baseline_dr_envs8192.sh

echo "Running on: $(hostname)"
echo "Date: $(date)"
echo ""

cd /ihub/homedirs/svs_ald/sudhir/manipulation

env MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    WANDB_API_KEY=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 \
    WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur \
    uv run train Mjlab-Open-Drawer-Osc-Kinova-BaselineDr \
    --env.scene.num-envs 8192 \
    --agent.max-iterations 5_000 \
    --agent.wandb-project mjlab-kinova-tasks-osc \
    --agent.experiment-name open_drawer_osc_baseline_dr \
    --agent.run-name baseline_dr_envs8192 \
    --agent.wandb-tags '("baseline_dr","hp_variant","num_envs=8192")' \
    --video True \
    --video-length 100 \
    --video-interval 100
