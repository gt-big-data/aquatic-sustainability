#!/bin/bash
#SBATCH -J coral_bleaching_lstm
#SBATCH --account=paceship-aquatic
#SBATCH --partition=cpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wgao305@gatech.edu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Optional CLI args passed after the sbatch script path:
#   sbatch run_bleaching_lstm.sbatch --data-path /path/to/file.npz --output-dir /path/to/out --loss ce
CLI_DATA_PATH=""
CLI_OUTPUT_DIR=""
CLI_LOSS=""
CLI_FOCAL_GAMMA=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path)
      CLI_DATA_PATH="$2"
      shift 2
      ;;
    --output-dir)
      CLI_OUTPUT_DIR="$2"
      shift 2
      ;;
    --loss)
      CLI_LOSS="$2"
      shift 2
      ;;
    --focal-gamma)
      CLI_FOCAL_GAMMA="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: sbatch run_bleaching_lstm.sbatch [--data-path PATH] [--output-dir DIR] [--loss ce|focal] [--focal-gamma FLOAT]" >&2
      exit 1
      ;;
  esac
done

# Defaults to the directory where you submit sbatch from.
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
DATA_PATH="${CLI_DATA_PATH:-${DATA_PATH:-$PROJECT_DIR/datasets/sequences_reduced_16.npz}}"
OUTPUT_DIR="${CLI_OUTPUT_DIR:-${OUTPUT_DIR:-$PROJECT_DIR/outputs/$SLURM_JOB_ID}}"
LOSS="${CLI_LOSS:-${LOSS:-focal}}"
FOCAL_GAMMA="${CLI_FOCAL_GAMMA:-${FOCAL_GAMMA:-2.0}}"

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
echo "Loss: $LOSS"
echo "Focal Gamma: $FOCAL_GAMMA"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "=========================================="

python bleaching_lstm.py \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --loss "$LOSS" \
  --focal-gamma "$FOCAL_GAMMA" \
  --num-workers "$((SLURM_CPUS_PER_TASK - 1))"

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
