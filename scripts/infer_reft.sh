#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpupr.24h
#SBATCH --gres=gpumem:38g
#SBATCH --mem-per-cpu=16g

source /cluster/home/${USERNAME}/miniconda3_new/etc/profile.d/conda.sh
module load eth_proxy
module load stack/2024-06 cuda/12.8.0
conda activate reftcl

USERNAME="${USERNAME:-lbarinka}"
MODEL_NAME="${MODEL_NAME:-llama-2-7b-chat}"
BENCHMARK_SIZE="${BENCHMARK_SIZE:-500}"

cd /cluster/home/${USERNAME}/trace

cl_method="REFT-CL"
port=$(shuf -i25000-30000 -n1)

DATA_PATH="/cluster/scratch/${USERNAME}/TRACE_data/TRACE-Benchmark/LLM-CL-Benchmark_${BENCHMARK_SIZE}"
MODEL_PATH="/cluster/scratch/${USERNAME}/initial_model/${MODEL_NAME}"
INFERENCE_MODEL_PATH="/cluster/scratch/${USERNAME}/outputs_LLM-CL/cl/REFT-CL_optimized_alpha"
DATA_CACHE="/cluster/scratch/${USERNAME}/TRACE_cache"
INFER_OUTPUT_PATH="${INFERENCE_MODEL_PATH}/predictions"

mkdir -p "$INFER_OUTPUT_PATH"

echo "Starting REFT-CL inference with:" \
     "\n  DATA_PATH=$DATA_PATH" \
     "\n  MODEL_PATH=$MODEL_PATH" \
     "\n  INFERENCE_MODEL_PATH=$INFERENCE_MODEL_PATH" \
     "\n  INFER_OUTPUT_PATH=$INFER_OUTPUT_PATH" \
     "\n  PORT=$port"

deepspeed --include=localhost:0 --master_port $port inference/infer_single.py \
  --data_path "$DATA_PATH" \
  --data_output_path "$DATA_CACHE" \
  --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
  --model_name_or_path "$MODEL_PATH" \
  --inference_model_path "$INFERENCE_MODEL_PATH" \
  --inference_output_path "$INFER_OUTPUT_PATH" \
  --inference_batch 4 \
  --max_prompt_len 1024 \
  --max_ans_len 512 \
  --seed 1234 \
  --CL_method "$cl_method" \
  --reft_layer1 "0;8;16;24" \
  --reft_layer2 "1;9;17;25" \
  --reft_layer3 "2;10;18;26" \
  --reft_layer4 "3;11;19;27" \
  --reft_layer5 "4;12;20;28" \
  --reft_layer6 "5;13;21;29" \
  --reft_layer7 "6;14;22;30" \
  --reft_layer8 "7;15;23;31" \
  --reft_rank 4 \
  --reft_eps 1e-6 \
  --deepspeed 2>&1 | tee -a "$INFERENCE_MODEL_PATH"/infer.log


