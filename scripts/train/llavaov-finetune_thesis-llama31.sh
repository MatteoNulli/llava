#!/bin/bash

export HTTP_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export HTTPS_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export http_proxy=http://httpproxy-tcop.vip.ebay.com:80
export https_proxy=http://httpproxy-tcop.vip.ebay.com:80
export no_proxy=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com
export NO_PROXY=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com

export HUGGINGFACE_HUB_CACHE=/mnt/nushare2/data/vorshulevich/.cache/huggingface/hub
export HF_HOME=/mnt/nushare2/data/vorshulevich/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/nushare2/data/vorshulevich/.cache/huggingface/hub
export HF_DATASETS_CACHE=/mnt/nushare2/data/vorshulevich/.cache/huggingface/datasets
export TORCH_EXTENSIONS_DIR=/mnt/nushare2/data/vorshulevich/.llm_cache/torch_cache
export TRITON_CACHE_DIR=/mnt/nushare2/data/vorshulevich/.triton/autotune

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export TORCH_NCCL_ENABLE_MONITORING=0
export HF_DATASETS_OFFLINE=1

RANK=${RANK:-0}
ADDR=${ADDR:-"127.0.0.1"}
PORT=${PORT:-"29501"}
NNODES=${NNODES:-1}
NUM_GPUS=${NUM_GPUS:-8}


# First job
# echo "Starting pretraining job..."


# MODEL_NAME="Meta-Llama-3_1-8B-Instruct"
BASE_RUN_NAME="standard_llava15-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion"
# DATA_DIR=/mnt/nushare2/data/mnulli/pretrainingdata
BASE_SAVE_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/${BASE_RUN_NAME}
# MODEL_DIR=/mnt/mtrepo/data/wwalentynowicz/models/$MODEL_NAME
# VIS_TOWER_DIR=/mnt/nushare2/data/baliao/multimodal/model_zoos

# TOOL_DIR=/data/chatgpt/notebooks/mnulli/llava

# mkdir -p $BASE_SAVE_DIR

cd /opt/krylov-workflow/src/run_fn_0/


# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
#     llava/train/train_mem.py \
#     --deepspeed scripts/zero3.json \
#     --model_name_or_path $MODEL_DIR \
#     --version llama3 \
#     --data_path $DATA_DIR/blip_laion_cc_sbu_558k.json \
#     --image_folder $DATA_DIR/images \
#     --vision_tower $VIS_TOWER_DIR/openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir $BASE_SAVE_DIR \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to none \
#     --sam2_masking_token True \
#     --overwrite_output_dir 2>&1 | tee $BASE_SAVE_DIR/out


# Second job
echo "Starting finetuning job..."

MODEL_NAME="Meta-Llama-3_1-8B-Instruct"
# DATA_DIR=/mnt/nushare2/data/mnulli/verified_conversations/finetuningdata/format_adjusted_llava-mix665k.json
DATA_DIR=/mnt/nushare2/data/mnulli/verified_conversations/finetuningdata/onevision.yaml
IMG_DIR='None' 
SFT_RUN_NAME="old_llavacodebase-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-ovdata-lora"

MODEL_DIR=/mnt/mtrepo/data/wwalentynowicz/models/${MODEL_NAME}
VIS_TOWER_DIR=/mnt/nushare2/data/baliao/multimodal/model_zoos
PROJECTOR=${BASE_SAVE_DIR}/mm_projector.bin
# MASK_TOKEN=${BASE_SAVE_DIR}/mm_bom_mask_token.bin

SAVE_DIR=/mnt/nushare2/data/mnulli/llava_ov/finetuning/sft/${SFT_RUN_NAME}

TOOL_DIR=/data/chatgpt/notebooks/mnulli/llava

mkdir -p $SAVE_DIR

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-4 \
    --model_name_or_path $MODEL_DIR \
    --version llama3 \
    --data_path $DATA_DIR \
    --image_folder $IMG_DIR \
    --vision_tower $VIS_TOWER_DIR/openai/clip-vit-large-patch14-336 \
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
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
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