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

pip install --proxy http://httpproxy-tcop.vip.ebay.com:80 pycocotools

cd /opt/krylov-workflow/src/run_fn_0/



# First job
echo "Starting pretraining job..."
CAP_EPOCHS=1

MODEL_NAME="Meta-Llama-3_1-8B-Instruct"
MODEL_DIR=/mnt/mtrepo/data/wwalentynowicz/models/${MODEL_NAME}
# MODEL_NAME="meta-llama--Llama-3.2-1B-Instruct"
# MODEL_DIR=/mnt/nushare2/data/mnulli/model_zoos/language_models/${MODEL_NAME}

#standard llava pretraining data
DATA_PATH=/mnt/nushare2/data/mnulli/pretrainingdata/blip_laion_cc_sbu_558k.json
# ## share gpt4v pretraining data
# DATA_PATH=/mnt/nushare2/data/mnulli/thesis/data/training_data/sharegpt4v_captioning_data/share_gpt4v_format_adjusted_share-captioner_coco_lcs_sam_1246k_1107.json
IMG_DIR='None' 
FILE_NAME_CAP=$(echo "${DATA_PATH##*/}" | cut -d'_' -f1,2)

echo DATAFILE_NAME=$FILE_NAME_CAP

VIS_TOWER=/mnt/nushare2/data/baliao/multimodal/model_zoos/openai/clip-vit-large-patch14-336
# VIS_TOWER=/mnt/nushare2/data/vorshulevich/models/siglip2-so400m-patch14-384
VIS_TOWER_NAME=$(echo "$VIS_TOWER" | awk -F'/' '{print $(NF-1)"-"$NF}')

echo VIS_TOWER_NAME=$VIS_TOWER_NAME


BASE_RUN_NAME="standard_llava15-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion"
# BASE_RUN_NAME="standard_llava-$MODEL_NAME-$VIS_TOWER_NAME-$FILE_NAME_CAP-$CAP_EPOCHS-EPOCHS"

BASE_SAVE_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/captioning_standard_llava/${BASE_RUN_NAME}

TOOL_DIR=/data/chatgpt/notebooks/mnulli/llava

# mkdir -p $BASE_SAVE_DIR

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
#     llava/train/train_mem.py \
#     --deepspeed scripts/zero3.json \
#     --model_name_or_path $MODEL_DIR \
#     --version llama3 \
#     --data_path $DATA_PATH \
#     --image_folder $IMG_DIR \
#     --vision_tower $VIS_TOWER \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir $BASE_SAVE_DIR \
#     --num_train_epochs $CAP_EPOCHS \
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
#     --overwrite_output_dir 2>&1 | tee $BASE_SAVE_DIR/out



# Second job
echo "Starting finetuning job..."
SFT_EPOCHS=1

MODEL_NAME="Meta-Llama-3_1-8B-Instruct"
MODEL_DIR=/mnt/mtrepo/data/wwalentynowicz/models/${MODEL_NAME}
# MODEL_NAME="meta-llama--Llama-3.2-1B-Instruct"
# MODEL_DIR=/mnt/nushare2/data/mnulli/model_zoos/language_models/${MODEL_NAME}

# DATA_PATH_SFT=/mnt/nushare2/data/mnulli/verified_conversations/finetuningdata/llava_mix665k_format_adjusted.json
DATA_PATH_SFT=/mnt/nushare2/data/mnulli/thesis/data/training_data/nyu-visionx--Cambrian-10M--extracted/Cambrian7M_withsystemprompt.json
IMG_DIR='None' 

FILE_NAME_SFT=$(echo "${DATA_PATH_SFT##*/}" | cut -d'_' -f1,2)

echo DATAFILE_NAME=$FILE_NAME_SFT

VIS_TOWER=/mnt/nushare2/data/baliao/multimodal/model_zoos/openai/clip-vit-large-patch14-336
# VIS_TOWER=/mnt/nushare2/data/vorshulevich/models/siglip2-so400m-patch14-384
VIS_TOWER_NAME=$(echo "$VIS_TOWER" | awk -F'/' '{print $(NF-1)"-"$NF}')
echo VIS_TOWER_NAME=$VIS_TOWER_NAME

SFT_RUN_NAME="standard_llava15-$MODEL_NAME-$VIS_TOWER_NAME-$FILE_NAME_CAP-$FILE_NAME_SFT-lora-$SFT_EPOCHS-EPOCHS"

PROJECTOR=${BASE_SAVE_DIR}/mm_projector.bin

SAVE_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_standard_llava/${SFT_RUN_NAME}

# TOOL_DIR=/data/chatgpt/notebooks/mnulli/llava

mkdir -p $SAVE_DIR

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-4 \
    --model_name_or_path $MODEL_DIR \
    --version llama3 \
    --data_path $DATA_PATH_SFT \
    --image_folder $IMG_DIR \
    --vision_tower $VIS_TOWER \
    --pretrain_mm_mlp_adapter $PROJECTOR \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $SAVE_DIR \
    --num_train_epochs $SFT_EPOCHS \
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