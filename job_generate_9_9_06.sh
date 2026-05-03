#!/bin/bash
#SBATCH --job-name=gen_9_9_06
#SBATCH --output=/common/users/tdn39/Robotics/Mujoco/mujoco_env_creator/slurm_logs/gen_9_9_06_%j.out
#SBATCH --error=/common/users/tdn39/Robotics/Mujoco/mujoco_env_creator/slurm_logs/gen_9_9_06_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=96G

set -e

# Change to the directory containing the script
cd /common/users/tdn39/Robotics/Mujoco/namo_cpp

# Set up environment
source /common/home/tdn39/.virtualenvs/mujoco/bin/activate
export PYTHONPATH=$PYTHONPATH:/common/users/tdn39/Robotics/Mujoco/namo_cpp/build_python

# Limit threads per process to avoid hitting RLIMIT_NPROC
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Configuration
INPUT_DIR="/common/users/tdn39/Robotics/Mujoco/templates/Nov25/9_9_06deletion"
OUTPUT_DIR="/common/users/shared/robot_learning/dm1487/namo/mj_env_configs/nov28/9_9_06deletion"
NAMO_CONFIG="/common/users/tdn39/Robotics/Mujoco/namo_cpp/config/namo_config.yaml"
NUM_ENVS=10
# We use 1 worker per python process because we parallelize via xargs
WORKER_THREADS=1 
PARALLEL_JOBS=40

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Starting job for $INPUT_DIR"
echo "Outputting to $OUTPUT_DIR"

# Use find and xargs to parallelize processing of templates
# -P 40 runs 40 processes in parallel
find "$INPUT_DIR" -name "*.xml" -print0 | xargs -0 -P "$PARALLEL_JOBS" -I {} \
    python3 /common/users/tdn39/Robotics/Mujoco/mujoco_env_creator/generate_envs.py \
    "{}" \
    --namo-config "$NAMO_CONFIG" \
    --num-envs "$NUM_ENVS" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers "$WORKER_THREADS" \
    --num-objects 25

echo "Job complete."
