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

RANK=${RANK:-0}
ADDR=${ADDR:-"127.0.0.1"}
PORT=${PORT:-"29501"}
NNODES=${NNODES:-1}
NUM_GPUS=${NUM_GPUS:-8}

export HF_DATASETS_OFFLINE=1


MODEL_NAME="Meta-Llama-3_1-8B-Instruct"
BASE_RUN_NAME="8bs_globalviewmasking_oldllavacodebase-Meta-Llama-3_1-8B-Instruct-siglip2-so400m-patch16-512-bliplaion"
DATA_DIR=/mnt/nushare2/data/mnulli/pretrainingdata
BASE_SAVE_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/captioning/${BASE_RUN_NAME}
MODEL_DIR=/mnt/mtrepo/data/wwalentynowicz/models/$MODEL_NAME
VIS_TOWER_DIR=/mnt/nushare2/data/mnulli/model_zoos/siglip/models--google--siglip2-so400m-patch16-512/snapshots/ceea1cba8130d8271436da4828633198c176a775
# VIS_TOWER_DIR=/mnt/nushare2/data/baliao/multimodal/model_zoos/openai/clip-vit-large-patch14-336

# TOOL_DIR=/data/chatgpt/notebooks/mnulli/llava

# mkdir -p $BASE_SAVE_DIR

cd /opt/krylov-workflow/src/run_fn_0/


ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $MODEL_DIR \
    --version llama3 \
    --data_path $DATA_DIR/blip_laion_cc_sbu_558k.json \
    --image_folder $DATA_DIR/images \
    --vision_tower $VIS_TOWER_DIR \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $BASE_SAVE_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
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
    --sam2_masking_token True \
    --overwrite_output_dir 2>&1 | tee $BASE_SAVE_DIR/out