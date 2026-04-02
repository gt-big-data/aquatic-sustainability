#!/bin/bash
#SBATCH -J coral_extract_sequence
#SBATCH --account=paceship-aquatic
#SBATCH --partition=cpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wgao305@gatech.edu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Defaults to the directory where you submit sbatch from.
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_DIR/extract_sequence.py}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/outputs/$SLURM_JOB_ID}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

module purge
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate coral

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Project Dir: $PROJECT_DIR"
echo "Script Path: $SCRIPT_PATH"
echo "Log Dir: $LOG_DIR"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "=========================================="

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: Script not found: $SCRIPT_PATH" >&2
  exit 1
fi

python "$SCRIPT_PATH" 2>&1 | tee "$LOG_DIR/extract_sequence.log"

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
