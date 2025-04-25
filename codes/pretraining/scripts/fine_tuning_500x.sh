#!/usr/bin/env bash
set -e
cd ../

# === Model Configuration ===
MODEL_NAME="meta-llama/Llama-3.2-1B"
DATASET_NAME="emozilla/pg19"

# === Training Configuration ===
MAX_STEPS=40000
BATCH_SIZE=4
EVAL_BATCH_SIZE=4 
GRAD_ACCUM_STEPS=1
LR=3e-4
WARMUP_STEPS=300
LR_SCHEDULER="constant_with_warmup"
SAVE_STEPS=1000
EVAL_STEPS=500
LOGGING_STEPS=20

# === Memory Configuration ===
NUM_MEM=4
MAX_LENGTH=256

# === LoRA Configuration ===
USE_PEFT=true
LORA_R=64
LORA_ALPHA=32
LORA_DROPOUT=0.05

# === Output & Logging Configuration ===
SAVE_STRATEGY="steps"
EVAL_STRATEGY="steps"
OUTPUT_DIR="/root/500xCompressor/results/500x_lora2"
LOGGING_DIR="/root/500xCompressor/results/500x_lora2"
DEEPSPEED_CONFIG="/root/500xCompressor/codes/deepspeed_configurations.json"

# === Launch Training ===
python train_500xCompressor.py \
    --model_name $MODEL_NAME \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LR \
    --warmup_steps $WARMUP_STEPS \
    --lr_scheduler_type $LR_SCHEDULER \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --num_mem $NUM_MEM \
    --max_length $MAX_LENGTH \
    --dataset_name $DATASET_NAME \
    --save_strategy $SAVE_STRATEGY \
    --eval_strategy $EVAL_STRATEGY \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --deepspeed_config $DEEPSPEED_CONFIG \
    $( [ "$USE_PEFT" = true ] && echo "--use_peft --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT" )

echo "✅ Training completed successfully."
