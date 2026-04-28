#!/bin/bash
#SBATCH --job-name=eval_open_drawer_osc
#SBATCH --partition=1gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dgx1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/svs_ald/sudhir/manipulation/logs/eval_open_drawer_osc.log
#SBATCH --qos=1gpu
#SBATCH --time=02:00:00

# Phase 0 evaluation skeleton.
# Runs the OOD eval task variant (Mjlab-Open-Drawer-Osc-Kinova-Eval) against
# a trained checkpoint. Reports success_rate and object_to_goal_error to W&B.
#
# Required: pass either CHECKPOINT_FILE or WANDB_RUN_PATH (e.g. via
# `sbatch --export=WANDB_RUN_PATH=entity/project/run_id slurm/eval_open_drawer_osc.sh`).
# Optional: WANDB_CHECKPOINT_NAME (e.g. model_4900.pt), NUM_ENVS (default 64).

echo "Running on: $(hostname)"
echo "Date: $(date)"
echo ""

cd /ihub/homedirs/svs_ald/sudhir/manipulation

NUM_ENVS=${NUM_ENVS:-64}

PLAY_ARGS=(
    --num-envs "$NUM_ENVS"
    --video True
    --video-length 200
)

if [[ -n "$CHECKPOINT_FILE" ]]; then
    PLAY_ARGS+=(--checkpoint-file "$CHECKPOINT_FILE")
elif [[ -n "$WANDB_RUN_PATH" ]]; then
    PLAY_ARGS+=(--wandb-run-path "$WANDB_RUN_PATH")
    if [[ -n "$WANDB_CHECKPOINT_NAME" ]]; then
        PLAY_ARGS+=(--wandb-checkpoint-name "$WANDB_CHECKPOINT_NAME")
    fi
else
    echo "ERROR: must set CHECKPOINT_FILE or WANDB_RUN_PATH" >&2
    exit 1
fi

env MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    WANDB_API_KEY=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 \
    WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur \
    CUDA_VISIBLE_DEVICES=0 \
    uv run play Mjlab-Open-Drawer-Osc-Kinova-Eval \
    "${PLAY_ARGS[@]}"
