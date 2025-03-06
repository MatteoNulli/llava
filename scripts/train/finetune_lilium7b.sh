#!/bin/bash
  
export HF_DATASETS_OFFLINE=1
DATA_DIR=/mnt/nushare2/data/baliao/multimodal/data/LLaVA-Instruct-665k
# DATA_DIR=/mnt/nushare2/data/mnulli/verified_conversations/finetuningdata
IMG_DIR='None' 

##changed the save directory to specific model name
VERSION='llava_lilium_2'
SAVE_DIR=/data/chatgpt/notebooks/mnulli/llava/exp_results/finetuned/lilium_try

MODEL_DIR=/mnt/nushare2/data/baliao/multimodal/model_zoos
PRETRAIN_CKPT=/mnt/nushare2/data/baliao/multimodal/00_reproduce
PROJECTOR=$PRETRAIN_CKPT/lilium-2-7b-chat-pretrain/mm_projector.bin

mkdir -p $SAVE_DIR
deepspeed /data/chatgpt/notebooks/mnulli/llava/train_mem.py \
    --deepspeed /data/chatgpt/notebooks/mnulli/llava/scripts/zero2.json \
    --model_name_or_path $MODEL_DIR/lilium-2-7b-chat \
    --version $VERSION \
    --data_path $DATA_DIR/llava_v1_5_mix665k.json \
    --image_folder $DATA_DIR/images \
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
    --learning_rate 2e-4 \
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