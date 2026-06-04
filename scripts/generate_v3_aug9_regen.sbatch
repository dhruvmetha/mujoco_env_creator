#!/bin/bash
#SBATCH --job-name=gen_v3_aug9_regen
#SBATCH --partition=main-redhat
#SBATCH --output=/scratch/dm1487/logs/gen_v3_aug9_regen_%A_%a.out
#SBATCH --error=/scratch/dm1487/logs/gen_v3_aug9_regen_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --array=0-5
# Regenerate ONLY the 6 dead aug9_car_v3 templates with corrected per-template
# rescale factors (set1/b1,b2 + set2/b1,b2 @0.30; set2/b3,b5 @0.40).
# 6 shards x (10 commands each via xargs) ; joblist has 60 lines.

set -euo pipefail

REPO=/cache/home/dm1487/projects/namo/mujoco_env_creator
JOBLIST=$REPO/scripts/v3_aug9_regen_joblist.txt
NUM_TASKS=6
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PARALLEL=${SLURM_CPUS_PER_TASK:-16}

cd "$REPO"

export PYTHONPATH=/cache/home/dm1487/projects/namo/namo_cpp/python:/cache/home/dm1487/projects/namo/namo_cpp/build_python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/scratch/dm1487/envs/namo/lib:${LD_LIBRARY_PATH:-}

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p /scratch/dm1487/logs
mkdir -p /scratch/dm1487/datasets/car_envs/v3/aug9_car

TOTAL=$(wc -l < "$JOBLIST")
MY_LINES=$(awk -v t="$TASK_ID" -v n="$NUM_TASKS" 'NR % n == t' "$JOBLIST" | wc -l)

echo "[task $TASK_ID/$NUM_TASKS] joblist=$JOBLIST total=$TOTAL my_lines=$MY_LINES parallel=$PARALLEL"
echo "[task $TASK_ID] start: $(date)"

awk -v t="$TASK_ID" -v n="$NUM_TASKS" 'NR % n == t' "$JOBLIST" \
  | xargs -d '\n' -P "$PARALLEL" -I {} bash -c '{}'

echo "[task $TASK_ID] end: $(date)"
