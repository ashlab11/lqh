#!/bin/bash
#SBATCH --job-name=lqh-shifted-router-trace
#SBATCH --output=logs/shifted_router/%x_%j.out
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
mkdir -p logs/shifted_router experiments/shifted_router/results
srun --kill-on-bad-exit=1 --cpu-bind=none python scripts/shifted_router_trace.py \
  --output "experiments/shifted_router/results/trace_${SLURM_JOB_ID}.json"
