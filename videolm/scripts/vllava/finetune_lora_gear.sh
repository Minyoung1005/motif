#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16996
ARG_RANK=0 #${3:-0}

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=16 #4
GRADIENT_ACCUMULATION_STEPS=1 #$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

# Log Arguments
# export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=videollama2_vllava #videollama2qwen2_vllava
RUN_NAME=clip_7b_8f_train_ep30_lora_gear_1125
DATA_DIR=datasets
OUTP_DIR=data


deepspeed --master_port $MASTER_PORT videollama2/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero3.json \
    --model_type videollama2_mistral \
    --model_path VideoLLaMA2-7B \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type stc_connector \
    --data_path   /home/mhwang2/Projects/personalized_robots/data/LLAVA/gear/geardata_1125_train_2442_neg10.json \
    --data_folder /home/mhwang2/Projects/personalized_robots/LLaVA/scripts/v1_5/ \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 8 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/finetune_${RUN_NAME} \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "epoch" \
    --num_train_epochs 30 \
    --save_total_limit 99 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name $RUN_NAME \


    # --model_type videollama2_qwen2 \
    # --model_path Qwen/Qwen2-7B-Instruct \
    # --vision_tower google/siglip-so400m-patch14-384 \
    # --mm_projector_type stc_connector_v35 \
    # --pretrain_mm_mlp_adapter ${OUTP_DIR}/${WANDB_PROJECT}/pretrain_${RUN_NAME}/mm_projector.bin \
    # --save_steps 500 \


