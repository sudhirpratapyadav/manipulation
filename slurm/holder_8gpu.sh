#!/bin/bash
#SBATCH --job-name=holder_8gpu
#SBATCH --partition=1gpu
#SBATCH --nodelist=dgx2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=/ihub/homedirs/svs_ald/sudhir/manipulation/logs/holder_8gpu.log
#SBATCH --qos=8gpu
#SBATCH --time=20-00:00:00

# Long-lived 8-GPU allocation. Does nothing on its own; everything else
# runs as `srun --jobid=<this>` job-steps. See docs/cluster_workflow.md.
#
# Submit:
#   sbatch slurm/holder_8gpu.sh
# Find the job id:
#   squeue -u $USER -h -o "%i" -n holder_8gpu
# Launch a single-GPU step inside it:
#   srun --jobid=<id> --gres=gpu:1 -n1 --exclusive bash slurm/open_drawer_osc_phase0.sh
# Cancel the holder when done:
#   scancel <id>

echo "Holder allocation up on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,uuid --format=csv 2>&1 | head -10
echo "Sleeping for the duration of the allocation."
# Sleep for 20 days; SLURM will tear it down on time-out.
sleep 1728000
