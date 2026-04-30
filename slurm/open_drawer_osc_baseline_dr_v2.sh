#!/bin/bash
#SBATCH --job-name=open_drawer_osc_baseline_dr_v2
#SBATCH --partition=1gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/svs_ald/sudhir/manipulation/logs/open_drawer_osc_baseline_dr_v2.log
#SBATCH --qos=1gpu
#SBATCH --time=24:00:00

# baseline_dr_v2: baseline_dr (P1 init-pose floor + drawer DR + curriculum
# widening of init ranges) + longer drawer (joint range -0.40 m, goal in
# [-0.35, -0.10]) + stepped impulse curriculum on drawer base body.
#
# Impulse curriculum (PPO iter, common_step_counter // num_steps_per_env):
#   iter   0..299  : magnitude 0 (off)
#   iter 300..399  : 3 N
#   iter 400..499  : 6 N
#   iter 500..599  : 9 N
#   iter 600..699  : 12 N
#   iter 700+      : 15 N (peak, frozen)
#
# Run as a job-step inside the holder:
#   srun --jobid=<holder> --gres=gpu:1 -n1 --exclusive bash \
#        slurm/open_drawer_osc_baseline_dr_v2.sh

echo "Running on: $(hostname)"
echo "Date: $(date)"
echo ""

cd /ihub/homedirs/svs_ald/sudhir/manipulation

env MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    WANDB_API_KEY=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 \
    WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur \
    uv run train Mjlab-Open-Drawer-Osc-Kinova-BaselineDrV2 \
    --env.scene.num-envs 1024 \
    --agent.max-iterations 5_000 \
    --agent.wandb-project mjlab-kinova-tasks-osc \
    --agent.experiment-name open_drawer_osc_baseline_dr_v2 \
    --agent.run-name baseline_dr_v2 \
    --agent.wandb-tags '("baseline_dr_v2","longer_drawer","impulse_curriculum")' \
    --video True \
    --video-length 100 \
    --video-interval 100
