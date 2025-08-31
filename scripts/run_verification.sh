#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpupr.24h
#SBATCH --gres=gpumem:38g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g

# Get script directory and project root
PROJECT_ROOT="/cluster/home/lbarinka/trace"
SCRIPT_DIR="$PROJECT_ROOT/scripts"
EVALUATIONS_DIR="$PROJECT_ROOT/evaluations"

# Use absolute paths for all files
INPUT_FILE="$PROJECT_ROOT/predictions_reftcl/results-0-0-C-STANCE.json"
OUTPUT_FILE="$PROJECT_ROOT/evaluations/llama_coherence_analysis_results.json"
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"

source /cluster/home/${USER}/miniconda3_new/etc/profile.d/conda.sh
module load eth_proxy
module load stack/2024-06 cuda/12.8.0
conda activate reftcl

# Set PYTHONPATH to include evaluations directory
export PYTHONPATH="$EVALUATIONS_DIR:$PYTHONPATH"

# Run the Python verification script
python3 "$SCRIPT_DIR/run_coherence_verification.py" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --model "$MODEL_NAME"
