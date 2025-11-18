#!/bin/bash
#SBATCH -J s1_oil_inference
#SBATCH --account=gts-wgao305
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH -t 4:00:00
#SBATCH -q embers
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wgao305@gatech.edu
#SBATCH --array=0-99%50
#SBATCH --output=/storage/project/ps-aquatic-0/wgao305/logs/inference_%A_%a.out
#SBATCH --error=/storage/project/ps-aquatic-0/wgao305/logs/inference_%A_%a.err

# Exit on error
set -e

# Base project directory
PROJECT_DIR="/storage/project/ps-aquatic-0/wgao305"

# Create logs directory
mkdir -p "${PROJECT_DIR}/logs"

# Load required modules
module load anaconda3
module load gdal

# Activate conda environment
conda activate oilspill

# Define paths
SCENE_CSV="${PROJECT_DIR}/scene_tasks.csv"
SAVED_MODEL="${PROJECT_DIR}/model.keras"
RESULTS_DIR="${PROJECT_DIR}/results"

# Get row index
ROW_INDEX=${SLURM_ARRAY_TASK_ID}

# Print job info
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${HOSTNAME}"
echo "Working Dir: $(pwd)"
echo "Started: $(date)"
echo "=========================================="

# Run inference
python worker_inference.py \
    --scene-csv "$SCENE_CSV" \
    --row-index "$ROW_INDEX" \
    --saved-model "$SAVED_MODEL" \
    --results-dir "$RESULTS_DIR" \
    --chip-px 400 \
    --stride 200 \
    --batch-size 16

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="