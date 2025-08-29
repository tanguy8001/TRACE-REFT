#!/bin/bash
#SBATCH --output=/cluster/home/tdieudonne/clmm/TRACE/logs/infer_reft_cl_%j.out
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=2
#SBATCH --partition=gpupr.24h
#SBATCH --gres=gpumem:38g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g

USERNAME="${USERNAME:-tdieudonne}"
MODEL_NAME="${MODEL_NAME:-llama-2-7b-chat}"
BENCHMARK_SIZE="${BENCHMARK_SIZE:-500}"
cl_method="REFT-CL"
port=$(shuf -i25000-30000 -n1)

DATA_PATH="/cluster/scratch/${USERNAME}/TRACE_data/TRACE-Benchmark/LLM-CL-Benchmark_${BENCHMARK_SIZE}"
MODEL_PATH="/cluster/scratch/${USERNAME}/initial_model/${MODEL_NAME}"
INFERENCE_MODEL_PATH="/cluster/scratch/${USERNAME}/outputs_LLM-CL/cl/${cl_method}"
INFER_OUTPUT_PATH="${INFERENCE_MODEL_PATH}/predictions"
CACHE_PATH="/cluster/scratch/${USERNAME}/TRACE_cache"

mkdir -p "$INFER_OUTPUT_PATH"
mkdir -p "/cluster/home/${USERNAME}/clmm/TRACE/logs"

source /cluster/home/${USERNAME}/miniconda3/etc/profile.d/conda.sh
module load eth_proxy
module load stack/2024-06 cuda/12.8.0
conda activate reftcl

echo "Starting REFT-CL inference with the following parameters:"
echo "Data path: $DATA_PATH"
echo "Base model path: $MODEL_PATH"
echo "Inference model path: $INFERENCE_MODEL_PATH"
echo "Inference output path: $INFER_OUTPUT_PATH"
echo "CL method: $cl_method"
echo "Port: $port"
#C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten
deepspeed --include=localhost:0 --master_port $port clmm/TRACE/inference/infer_single.py \
  --data_path "$DATA_PATH" \
  --data_output_path "$CACHE_PATH" \
  --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
  --model_name_or_path "$MODEL_PATH" \
  --inference_model_path "$INFERENCE_MODEL_PATH" \
  --inference_batch 4 \
  --max_prompt_len 1024 \
  --max_ans_len 512 \
  --temperature 0.1 \
  --seed 1234 \
  --deepspeed \
  --CL_method "$cl_method" \
  --reft_layers "3;9;18;24" \
  --reft_rank 4 \
  --reft_eps 1e-6 \
  --inference_output_path "$INFER_OUTPUT_PATH" 2>&1 | tee -a "$INFERENCE_MODEL_PATH"/infer.log




