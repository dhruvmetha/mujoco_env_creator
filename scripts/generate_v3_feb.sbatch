#!/bin/bash
#SBATCH --job-name=gen_v3_feb
#SBATCH --partition=main-redhat
#SBATCH --output=/scratch/dm1487/logs/gen_v3_feb_%A_%a.out
#SBATCH --error=/scratch/dm1487/logs/gen_v3_feb_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-59
# 60 shards x 32 cpus = 1920 cpu-cores of parallelism
# v3 joblist has 4220 chunked commands (422 templates × 10 chunks each).
# Each shard gets ~70 commands, xargs -P 32 keeps 32 in flight per shard.
# → 60 × 32 = 1920 commands running concurrently at peak.

set -euo pipefail

REPO=/cache/home/dm1487/projects/namo/mujoco_env_creator
JOBLIST=$REPO/scripts/v3_feb_joblist.txt
NUM_TASKS=60
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PARALLEL=${SLURM_CPUS_PER_TASK:-32}

cd "$REPO"

# Python env + namo_rl bindings — let generate_envs.py find namo_rl optionally
export PYTHONPATH=/cache/home/dm1487/projects/namo/namo_cpp/python:/cache/home/dm1487/projects/namo/namo_cpp/build_python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/scratch/dm1487/envs/namo/lib:${LD_LIBRARY_PATH:-}

# Single-thread BLAS so $PARALLEL processes don't oversubscribe
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p /scratch/dm1487/logs
mkdir -p /scratch/dm1487/datasets/car_envs/v3/feb_car

TOTAL=$(wc -l < "$JOBLIST")
MY_LINES=$(awk -v t="$TASK_ID" -v n="$NUM_TASKS" 'NR % n == t' "$JOBLIST" | wc -l)

echo "[task $TASK_ID/$NUM_TASKS] joblist=$JOBLIST total=$TOTAL my_lines=$MY_LINES parallel=$PARALLEL"
echo "[task $TASK_ID] start: $(date)"

# xargs runs each assigned command in parallel; cap parallelism at $PARALLEL
awk -v t="$TASK_ID" -v n="$NUM_TASKS" 'NR % n == t' "$JOBLIST" \
  | xargs -d '\n' -P "$PARALLEL" -I {} bash -c '{}'

echo "[task $TASK_ID] end: $(date)"
