#!/bin/bash
#SBATCH --job-name=open_drawer_osc_baseline_dr
#SBATCH --partition=1gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/svs_ald/sudhir/manipulation/logs/open_drawer_osc_baseline_dr.log
#SBATCH --qos=1gpu
#SBATCH --time=24:00:00

# baseline_dr: Phase 1 init-pose floor + Phase 2 drawer DR (no arm-link mass)
# + step-stepped curriculum widening of drawer cube and init-pose ranges.
#
# Curriculum (env steps via env.common_step_counter):
#   iter   0..500 : frozen at start values (warm-up at Phase 1 floor)
#   iter 500..3000: linear ramp to end values
#   iter 3000+    : frozen at end values
#
# Curriculum knobs:
#   drawer cube half-extent : 0.10 -> 0.20 m  (centered at (0.8, 0, 0.4))
#   joint_delta_deg         : 15.0 -> 30.0
#   base xy half-extent     : 0.02 -> 0.05 m
#   base yaw half-extent    : 2 deg -> 10 deg
#
# Run as a job-step inside the holder allocation:
#   srun --jobid=<holder> --gres=gpu:1 -n1 --exclusive bash \
#        slurm/open_drawer_osc_baseline_dr.sh

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
    --agent.max-iterations 5_000 \
    --agent.wandb-project mjlab-kinova-tasks-osc \
    --agent.experiment-name open_drawer_osc_baseline_dr \
    --agent.run-name baseline_dr \
    --agent.wandb-tags '("baseline_dr","curriculum")' \
    --video True \
    --video-length 100 \
    --video-interval 100
