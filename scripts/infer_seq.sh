#!/bin/bash
#SBATCH --output=/cluster/home/tdieudonne/clmm/TRACE/logs/infer_seq_%j.out
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpupr.24h
#SBATCH --gres=gpumem:38g
#SBATCH --mem-per-cpu=16g

source /cluster/home/tdieudonne/miniconda3/etc/profile.d/conda.sh
module load eth_proxy
module load stack/2024-06 cuda/12.8.0
conda activate trace

mkdir -p /cluster/home/tdieudonne/clmm/TRACE/logs

cd /cluster/home/tdieudonne/clmm/TRACE

cl_method="O-LoRA"
port=$(shuf -i25000-30000 -n1)

# Paths customized for this environment
DATA_PATH="/cluster/scratch/tdieudonne/TRACE_data/TRACE-Benchmark/LLM-CL-Benchmark_500"
MODEL_PATH="/cluster/scratch/tdieudonne/initial_model/llama-7b-hf"
INFERENCE_MODEL_PATH="/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/${cl_method}"
INFER_OUTPUT_PATH="${INFERENCE_MODEL_PATH}/predictions"

mkdir -p "$INFER_OUTPUT_PATH"

echo "Starting inference with the following parameters:"
echo "Data path: $DATA_PATH"
echo "Base model path: $MODEL_PATH"
echo "Inference model path: $INFERENCE_MODEL_PATH"
echo "Inference output path: $INFER_OUTPUT_PATH"
echo "CL method: $cl_method"
echo "Port: $port"

deepspeed --include=localhost:0 --master_port $port inference/infer_single.py \
    --data_path "$DATA_PATH" \
    --data_output_path "/cluster/scratch/tdieudonne/TRACE_cache" \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path "$MODEL_PATH" \
    --inference_model_path "$INFERENCE_MODEL_PATH" \
    --inference_batch 4 \
    --max_prompt_len 512 \
    --max_ans_len 256 \
    --seed 1234 \
    --deepspeed \
    --CL_method "$cl_method" \
    --inference_output_path "$INFER_OUTPUT_PATH" 2>&1 | tee -a "$INFERENCE_MODEL_PATH"/infer.log