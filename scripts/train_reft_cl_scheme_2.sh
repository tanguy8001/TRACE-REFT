#!/bin/bash
#SBATCH --output=/cluster/home/lbarinka/trace/logs/train_seq_cl_%j.out
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpupr.24h
#SBATCH --gres=gpumem:38g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g
USERNAME="${USERNAME:-lbarinka}"
MODEL_NAME="${MODEL_NAME:-llama-2-7b-chat}"
BENCHMARK_SIZE="${BENCHMARK_SIZE:-500}"
cl_method="REFT-CL"
port=$(shuf -i25000-30000 -n1)

DATA_PATH="/cluster/scratch/${USERNAME}/TRACE_data/TRACE-Benchmark/LLM-CL-Benchmark_${BENCHMARK_SIZE}"
MODEL_PATH="/cluster/scratch/${USERNAME}/initial_model/${MODEL_NAME}"
OUTPUT_DIR="/cluster/scratch/${USERNAME}/outputs_LLM-CL/cl/${cl_method}_rank8_9layers_2"
DATA_CACHE="/cluster/scratch/${USERNAME}/reft_cl_outputs"
CONDA_NAME="miniconda3_new"
mkdir -p "$OUTPUT_DIR"j
mkdir -p "$DATA_CACHE"
mkdir -p "$DATA_PATH"

source /cluster/home/${USERNAME}/${CONDA_NAME}/etc/profile.d/conda.sh
module load eth_proxy
module load stack/2024-06 cuda/12.8.0
conda activate reftcl

echo "Starting training with the following parameters:"
echo "Data path: $DATA_PATH"
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Data cache: $DATA_CACHE"
echo "CL method: $cl_method"
echo "Port: $port"

deepspeed  --include=localhost:0 --master_port $port trace/training/main.py \
  --data_path "${DATA_PATH}" \
  --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
  --data_output_path "${DATA_CACHE}" \
  --model_name_or_path "${MODEL_PATH}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --max_prompt_len 1024 \
  --max_ans_len 512 \
  --learning_rate 9e-3 \
  --num_train_epochs 5,3,7,5,3,5,5,7 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 0 \
  --output_dir "${OUTPUT_DIR}" \
  --seed 1234 \
  --CL_method "$cl_method" \
  --reft_layer_task_1 "0;8;16;24" \
  --reft_layer_task_2 "1;9;17;25" \
  --reft_layer_task_3 "2;10;18;26" \
  --reft_layer_task_4 "3;11;19;27" \
  --reft_layer_task_5 "4;12;20;28" \
  --reft_layer_task_6 "5;13;21;29" \
  --reft_layer_task_7 "6;14;22;30" \
  --reft_layer_task_8 "7;15;23;31" \
  --reft_rank 8 \
  --reft_eps 1e-8 \
  --gradient_checkpointing \
  --disable_dropout \
  --print_loss \
  --deepspeed \
  --zero_stage 2 \
  --precision bf16 \
  2>&1 | tee -a "$OUTPUT_DIR"/train.log



