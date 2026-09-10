#!/bin/bash
#SBATCH --job-name=lqh-router-pilot
#SBATCH --output=logs/router_locality/%x_%j.out
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
mkdir -p logs/router_locality experiments/router_locality/pilots
srun --kill-on-bad-exit=1 --cpu-bind=none python scripts/router_locality_pilot.py \
  --steps "${STEPS:-200}" --union-weight "${UNION_WEIGHT:-0.1}" --churn-weight "${CHURN_WEIGHT:-0.05}" --balance-weight "${BALANCE_WEIGHT:-0.1}" --dataset-percent "${DATASET_PERCENT:-1}" --trainable "${TRAINABLE:-router}" --output "experiments/router_locality/pilots/pilot_${SLURM_JOB_ID}"
