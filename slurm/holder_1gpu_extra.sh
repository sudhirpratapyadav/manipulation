#!/bin/bash
#SBATCH --job-name=holder_1gpu_extra
#SBATCH --partition=1gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/svs_ald/sudhir/manipulation/logs/holder_1gpu_extra.log
#SBATCH --qos=1gpu
#SBATCH --time=24:00:00

# 1-GPU holder so we can launch baseline_dr_rollout48 on the free GPU that's
# outside our 3-GPU holder 18278's allocation.
echo "Holder up on $(hostname). GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
sleep 86400
