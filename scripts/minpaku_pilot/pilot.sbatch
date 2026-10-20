#!/usr/bin/env bash
# Mirrors the existing personal gpu-python allocation; no permanent service.
#SBATCH --job-name=stling-model-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
if (( $# != 3 && $# != 4 )); then
    echo 'Usage: sbatch pilot.sbatch SCRIPT_DIRECTORY PROJECT_DIRECTORY provision|check|serve|worker [PRIVATE_CONFIG]' >&2
    exit 2
fi
extra=()
if [[ "$3" == worker && $# == 4 ]]; then
    extra=(--worker-config "$4")
elif [[ "$3" == worker || $# == 4 ]]; then
    echo 'Only worker mode requires the fourth private configuration path' >&2
    exit 2
fi
exec /home/zachary/.conda/envs/zac-research/bin/python \
    "$1/run_pilot.py" --project "$2" --mode "$3" "${extra[@]}"
