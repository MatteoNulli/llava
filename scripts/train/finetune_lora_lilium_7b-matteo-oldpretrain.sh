#!/bin/bash

export HF_DATASETS_OFFLINE=1

MODEL_NAME="lilium-2-7b-chat"
DATA_DIR=/mnt/nushare2/data/mnulli/verified_conversations/finetuningdata
# DATA_DIR=/mnt/nushare2/data/vorshulevich/vlm_data/
IMG_DIR='None'

SAVE_DIR=/mnt/nushare2/data/mnulli/finetuning/llava-${MODEL_NAME}-lora-15m-fash-short-us_set-llava-mix-todel

MODEL_DIR=/mnt/nushare2/data/baliao/multimodal/model_zoos
PROJECTOR=/mnt/nushare2/data/baliao/multimodal/02_llava/${MODEL_NAME}-pretrain/mm_projector.bin

TOOL_DIR=/data/chatgpt/notebooks/mnulli/llava

mkdir -p $SAVE_DIR

deepspeed $TOOL_DIR/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-4\
    --deepspeed $TOOL_DIR/scripts/zero2.json \
    --model_name_or_path $MODEL_DIR/${MODEL_NAME} \
    --version 'llava_lilium_2' \
    --data_path $DATA_DIR/us_set_jsons_1.5Mfash-short_llava-mix665k.json \
    --image_folder $IMG_DIR \
    --vision_tower $MODEL_DIR/openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $PROJECTOR \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $SAVE_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.06 \
    --max_grad_norm 0.3 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --overwrite_output_dir \
    2>&1 | tee $SAVE_DIR/out