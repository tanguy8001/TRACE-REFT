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
OUTPUT_DIR="$PROJECT_ROOT/evaluations"

source /cluster/home/${USER}/miniconda3_new/etc/profile.d/conda.sh
module load eth_proxy
module load stack/2024-06 cuda/12.8.0
conda activate reftcl

# Set PYTHONPATH to include evaluations directory
export PYTHONPATH="$EVALUATIONS_DIR:$PYTHONPATH"

# Run the modular Python evaluation script with all datasets
python3 "$SCRIPT_DIR/run_modular_evaluation.py" \
    --datasets \
        "$PROJECT_ROOT/predictions_o-lora/results-7-0-C-STANCE.json" \
        "$PROJECT_ROOT/predictions_o-lora/results-7-1-FOMC.json" \
        "$PROJECT_ROOT/predictions_o-lora/results-7-2-MeetingBank.json" \
        "$PROJECT_ROOT/predictions_o-lora/results-7-3-Py150.json" \
        "$PROJECT_ROOT/predictions_o-lora/results-7-4-ScienceQA.json" \
        "$PROJECT_ROOT/predictions_o-lora/results-7-5-NumGLUE-cm.json" \
        "$PROJECT_ROOT/predictions_o-lora/results-7-6-NumGLUE-ds.json" \
        "$PROJECT_ROOT/predictions_o-lora/results-7-7-20Minuten.json" \
    --output-dir "$OUTPUT_DIR" \
    --judge-llama \
    --llama-model "meta-llama/Llama-3.2-3B-Instruct"