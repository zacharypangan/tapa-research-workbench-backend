#!/usr/bin/env bash
# Mirrors the existing personal gpu-python allocation; no permanent service.
#SBATCH --job-name=stling-prod-text
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
if (( $# != 3 )); then
    echo 'Usage: sbatch pilot.sbatch SCRIPT_DIRECTORY PROJECT_DIRECTORY PRIVATE_CONFIG' >&2
    exit 2
fi
exec /home/zachary/.conda/envs/zac-research/bin/python \
    "$1/run_pilot.py" --project "$2" --worker-config "$3"
