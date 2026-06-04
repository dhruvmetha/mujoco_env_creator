#!/bin/bash
#SBATCH --job-name=gen_v3_feb_test
#SBATCH --partition=main-redhat
#SBATCH --output=/scratch/dm1487/logs/gen_v3_feb_test_%A_%a.out
#SBATCH --error=/scratch/dm1487/logs/gen_v3_feb_test_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-19
# 20 shards x 32 cpus. test joblist has 422 commands (one per feb_car template, ~12 envs each).
# Each shard gets ~21 commands, xargs -P 32 keeps them in flight.

set -euo pipefail

REPO=/cache/home/dm1487/projects/namo/mujoco_env_creator
JOBLIST=$REPO/scripts/v3_feb_test_joblist.txt
NUM_TASKS=20
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PARALLEL=${SLURM_CPUS_PER_TASK:-32}

cd "$REPO"

export PYTHONPATH=/cache/home/dm1487/projects/namo/namo_cpp/python:/cache/home/dm1487/projects/namo/namo_cpp/build_python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/scratch/dm1487/envs/namo/lib:${LD_LIBRARY_PATH:-}

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p /scratch/dm1487/logs
mkdir -p /scratch/dm1487/datasets/car_envs/v3/test/feb_car

TOTAL=$(wc -l < "$JOBLIST")
MY_LINES=$(awk -v t="$TASK_ID" -v n="$NUM_TASKS" 'NR % n == t' "$JOBLIST" | wc -l)

echo "[task $TASK_ID/$NUM_TASKS] joblist=$JOBLIST total=$TOTAL my_lines=$MY_LINES parallel=$PARALLEL"
echo "[task $TASK_ID] start: $(date)"

awk -v t="$TASK_ID" -v n="$NUM_TASKS" 'NR % n == t' "$JOBLIST" \
  | xargs -d '\n' -P "$PARALLEL" -I {} bash -c '{}'

echo "[task $TASK_ID] end: $(date)"
