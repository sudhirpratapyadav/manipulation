#!/bin/bash
#SBATCH --job-name=holder_3gpu
#SBATCH --partition=1gpu
#SBATCH --nodelist=dgx2
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --output=/ihub/homedirs/svs_ald/sudhir/manipulation/logs/holder_3gpu.log
#SBATCH --qos=3gpu
#SBATCH --time=20-00:00:00

# 3-GPU holder fallback when dgx2 doesn't have 8 free GPUs.
# Use only when the 8gpu holder is PD waiting on Resources for too long.
# Same pattern as holder_8gpu.sh, just narrower.

echo "Holder allocation up on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,uuid --format=csv 2>&1 | head -10
echo "Sleeping for the duration of the allocation."
sleep 1728000
