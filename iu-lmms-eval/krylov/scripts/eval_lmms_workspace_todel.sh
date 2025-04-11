#!/bin/bash


export http_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export https_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export no_proxy=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com
export HTTP_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export HTTPS_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export NO_PROXY=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com

PORT=${PORT:-"29501"}
NUM_MACHINES=${NUM_MACHINES:-1}
NUM_GPUS=${NUM_GPUS:-2}

cd iu-lmms-eval/

# TASK=textvqa_test,ai2d,mme,mmbench_en_dev,mmstar,hallusion_bench_image,cvbench,mmmu_val
TASK=cvbench

if [[ "$TASK" =~ mmbench ]]; then
    pip install --proxy http://httpproxy-tcop.vip.ebay.com:80 openpyxl
fi

# CKPT_PATH=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_global_view_llava-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
# CKPT_PATH=/mnt/nushare2/data/mnulli/thesis/testruns/sft/4b_global_view_llava-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
# CKPT_PATH=/mnt/nushare2/data/mnulli/thesis/testruns/sft/no_global_view_llava-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
# CKPT_PATH=/mnt/nushare2/data/mnulli/thesis/testruns/sft_standard_llava/standard_llava15-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
CKPT_PATH=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_no_global_view_llava-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
# CKPT_PATH=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_no_global_view_oldllavacodebase-meta-llama--Llama-3.2-1B-Instruct-openclip-bliplaion_llava-lora-3-EPOCHS
# CKPT_PATH=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_no_global_view_oldllavacodebase-meta-llama--Llama-3.2-1B-Instruct-openclip-bliplaion_llava-lora-1-EPOCHS
# CKPT_PATH=/mnt/nushare2/data/mnulli/thesis/testruns/sft_standard_llava/standard_llava15-meta-llama--Llama-3.2-1B-Instruct-openclip-bliplaion-lora-1-EPOCHS

MODEL_BASE=/mnt/mtrepo/data/wwalentynowicz/models/Meta-Llama-3_1-8B-Instruct
# MODEL_BASE=/mnt/nushare2/data/mnulli/model_zoos/language_models/meta-llama--Llama-3.2-1B-Instruct
# builder in LLava expect a particular model_name for parsing
MODEL_NAME=llava
CONV_MODE=llama3

echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

OUT_NAME=$(basename "$CKPT_PATH")

OUTPUT_PATH=/mnt/nushare2/data/mnulli/llava_ov/playground/lmms_eval_results/$TASK_SUFFIX/$OUT_NAME

echo "OUTPUT_PATH: $OUTPUT_PATH"

accelerate launch --num_machines $NUM_MACHINES --num_processes $NUM_GPUS --main_process_port $PORT --mixed_precision no --dynamo_backend no \
    lmms_eval/__main__.py \
    --model $MODEL_NAME \
    --model_args pretrained=$CKPT_PATH,model_base=$MODEL_BASE,conv_template=$CONV_MODE \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --verbosity='DEBUG' \
    --output_path $OUTPUT_PATH