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
TASK=mmbench_en_dev

if [[ "$TASK" =~ mmbench ]]; then
    pip install --proxy http://httpproxy-tcop.vip.ebay.com:80 openpyxl
fi

CKPT_PATH=/mnt/nushare2/data/mnulli/finetuning/llava-lilium-2-7b-chat-lora-15Mfash-short_llavamix-lr1e-4
# CKPT_PATH=/mnt/nushare2/data/mnulli/finetuning/llava-vicuna-7b-chat-lora-1halfM-short_fash-llava_mix_llavaparams
MODEL_BASE=/mnt/nushare2/data/baliao/multimodal/model_zoos/lilium-2-7b-chat

# builder in LLava expect a particular model_name for parsing
MODEL_NAME=llava
# CONV_MODE=llama3
CONV_MODE=llava_lilium_2
# CONV_MODE=v1

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