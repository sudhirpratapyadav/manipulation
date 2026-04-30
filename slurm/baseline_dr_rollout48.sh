#!/bin/bash
#SBATCH --job-name=baseline_dr_rollout48
#SBATCH --partition=1gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/svs_ald/sudhir/manipulation/logs/baseline_dr_rollout48.log
#SBATCH --qos=1gpu
#SBATCH --time=24:00:00

# Hyperparam variant B: baseline_dr with num_steps_per_env=48 (vs ref 24).
# 2x longer rollouts -> stronger terminal-reward signal per gradient update.
# See docs/hyperparam_experiments.md.
#
# Run as a job-step inside the holder:
#   srun --jobid=<holder> --gres=gpu:1 -n1 --exclusive bash \
#        slurm/baseline_dr_rollout48.sh

echo "Running on: $(hostname)"
echo "Date: $(date)"
echo ""

cd /ihub/homedirs/svs_ald/sudhir/manipulation

env MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    WANDB_API_KEY=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 \
    WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur \
    uv run train Mjlab-Open-Drawer-Osc-Kinova-BaselineDr \
    --env.scene.num-envs 1024 \
    --agent.num-steps-per-env 48 \
    --agent.max-iterations 5_000 \
    --agent.wandb-project mjlab-kinova-tasks-osc \
    --agent.experiment-name open_drawer_osc_baseline_dr \
    --agent.run-name baseline_dr_rollout48 \
    --agent.wandb-tags '("baseline_dr","hp_variant","num_steps_per_env=48")' \
    --video True \
    --video-length 100 \
    --video-interval 100
