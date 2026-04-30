#!/bin/bash
#SBATCH --job-name=baseline_dr_v2_envs8192
#SBATCH --partition=1gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/svs_ald/sudhir/manipulation/logs/baseline_dr_v2_envs8192.log
#SBATCH --qos=1gpu
#SBATCH --time=48:00:00

# baseline_dr_v2 with num_envs=8192 (vs default 1024). 8x more parallel envs
# per gradient update, ~2.5x wall-clock per iter on A100. See
# docs/hyperparam_experiments.md for the rationale on this knob.
#
# Run as a job-step inside the holder:
#   srun --jobid=<holder> --gres=gpu:1 -n1 --exclusive bash \
#        slurm/baseline_dr_v2_envs8192.sh

echo "Running on: $(hostname)"
echo "Date: $(date)"
echo ""

cd /ihub/homedirs/svs_ald/sudhir/manipulation

env MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    WANDB_API_KEY=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 \
    WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur \
    uv run train Mjlab-Open-Drawer-Osc-Kinova-BaselineDrV2 \
    --env.scene.num-envs 8192 \
    --agent.max-iterations 5_000 \
    --agent.wandb-project mjlab-kinova-tasks-osc \
    --agent.experiment-name open_drawer_osc_baseline_dr_v2 \
    --agent.run-name baseline_dr_v2_envs8192 \
    --agent.wandb-tags '("baseline_dr_v2","longer_drawer","impulse_curriculum","hp_envs8192")' \
    --video True \
    --video-length 100 \
    --video-interval 100
