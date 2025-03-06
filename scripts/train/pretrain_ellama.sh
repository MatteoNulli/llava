#!/bin/bash

export HF_DATASETS_OFFLINE=1


MODEL_NAME="e-Llama-3_1-8B"
DATA_DIR=/mnt/nushare2/data/mnulli/pretrainingdata
SAVE_DIR=/mnt/nushare2/data/mnulli/pretraining/${MODEL_NAME}-bliplaion
MODEL_DIR=/data/chatgpt/data/mnulli/.cache/ellement/core-ai-nlp
VIS_TOWER_DIR=/mnt/nushare2/data/baliao/multimodal/model_zoos

TOOL_DIR=/data/chatgpt/notebooks/mnulli/llava

mkdir -p $SAVE_DIR
deepspeed $TOOL_DIR/train_mem.py \
    --deepspeed $TOOL_DIR/scripts/zero2.json \
    --model_name_or_path $MODEL_DIR/${MODEL_NAME}/final \
    --version ellama \
    --data_path $DATA_DIR/blip_laion_cc_sbu_558k.json \
    --image_folder $DATA_DIR/images \
    --vision_tower $VIS_TOWER_DIR/openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $SAVE_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --overwrite_output_dir 2>&1 | tee $SAVE_DIR/out