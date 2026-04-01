#!/bin/bash
#SBATCH -J coral_bleaching_lstm
#SBATCH --account=paceship-aquatic
#SBATCH --partition=gpu-a100      # or gpu-v100 / gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wgao305@gatech.edu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Defaults to the directory where you submit sbatch from.
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
DATA_PATH="${DATA_PATH:-$PROJECT_DIR/datasets/sequences.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/$SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

module purge
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate coral

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Project Dir: $PROJECT_DIR"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "=========================================="

python bleaching_lstm.py \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-workers "$((SLURM_CPUS_PER_TASK - 1))"

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
