#!/bin/bash
#SBATCH --job-name=lqh-mlp-pruning-pilot
#SBATCH --output=logs/mlp_pruning/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=350G
#SBATCH --time=02:00:00
#SBATCH --qos=promoted
#SBATCH --wckey=v1/model_development/ablation_and_research/general
#SBATCH --comment=billable=n

set -euo pipefail

REPO_ROOT=/home/asher.labovich@liquid.ai/lqh-projects/lqh/lqh
source /home/asher.labovich@liquid.ai/liquid_lfm/.venv/bin/activate
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/mlp_pruning experiments/mlp_pruning/pilots
srun --kill-on-bad-exit=1 --cpu-bind=none python scripts/mlp_pruning_pilot.py \
  --steps "${STEPS:-200}" --prune-ratio "${PRUNE_RATIO:-0.25}" --dataset-percent "${DATASET_PERCENT:-5}" \
  --trainable "${TRAINABLE:-predictor_only}" --learning-rate "${LEARNING_RATE:-1e-4}" \
  --ffn-learning-rate "${FFN_LEARNING_RATE:-1e-5}" \
  --output "experiments/mlp_pruning/pilots/pilot_${SLURM_JOB_ID}"
