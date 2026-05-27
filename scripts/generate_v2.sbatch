#!/bin/bash
#SBATCH --job-name=gen_v2
#SBATCH --output=/common/users/dm1487/scratch_namo/slurm_logs/gen_v2_%A_%a.out
#SBATCH --error=/common/users/dm1487/scratch_namo/slurm_logs/gen_v2_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-31

# Total: 32 array tasks x 32 cpus = 1024 cpus of parallelism.
# Joblist lines are assigned round-robin (NR % 32 == task_id) so aug9 sub-chunks
# (heavier) and feb chunks (lighter) are spread evenly across all 32 tasks.
#
# Restart a failed shard:    sbatch --array=2 scripts/generate_v2.sbatch
# Track progress:            tail -f .../slurm_logs/gen_v2_<jobid>_<taskid>.out

set -euo pipefail

REPO=/common/home/dm1487/robotics_research/ktamp/mujoco_env_creator
JOBLIST=$REPO/scripts/v2_joblist.txt
NUM_TASKS=32
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PARALLEL=${SLURM_CPUS_PER_TASK:-32}

cd "$REPO"

# Python env + namo_rl bindings.
# LD_LIBRARY_PATH must point to the mjxrl env's lib dir so namo_rl.so picks
# up the env's libstdcxx-ng>=13.2 (otherwise the system libstdc++ wins and
# the GLIBCXX_3.4.32 symbol is missing). Interactive shells get this via
# 'conda activate'; sbatch starts fresh so we set it explicitly.
export PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/python:/common/home/dm1487/robotics_research/ktamp/namo/build_python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/common/users/dm1487/envs/mjxrl/lib:${LD_LIBRARY_PATH:-}

# Limit per-process threads so $PARALLEL processes don't oversubscribe cpus
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p /common/users/dm1487/scratch_namo/slurm_logs
mkdir -p /common/users/dm1487/corl2026/namo/envs/v2/aug9_car
mkdir -p /common/users/dm1487/corl2026/namo/envs/v2/feb_car

TOTAL=$(wc -l < "$JOBLIST")
MY_LINES=$(awk -v t="$TASK_ID" -v n="$NUM_TASKS" 'NR % n == t' "$JOBLIST" | wc -l)

echo "[task $TASK_ID/$NUM_TASKS] joblist=$JOBLIST total=$TOTAL my_lines=$MY_LINES parallel=$PARALLEL"
echo "[task $TASK_ID] start: $(date)"

awk -v t="$TASK_ID" -v n="$NUM_TASKS" 'NR % n == t' "$JOBLIST" \
  | xargs -d '\n' -P "$PARALLEL" -I {} bash -c '{}'

echo "[task $TASK_ID] end: $(date)"
