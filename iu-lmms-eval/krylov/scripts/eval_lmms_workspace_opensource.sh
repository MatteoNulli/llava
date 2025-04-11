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

TASK=cvbench

if [[ "$TASK" =~ mmbench ]]; then
    pip install --proxy http://httpproxy-tcop.vip.ebay.com:80 openpyxl
fi


CKPT_PATH=/mnt/nushare2/data/mnulli/model_zoos/opensource-vlms/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369
# builder in LLava expect a particular model_name for parsing
MODEL_NAME=llava_hf

echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

OUTPUT_PATH=/mnt/nushare2/data/mnulli/llava_ov/playground/lmms_eval_results/$TASK_SUFFIX/$MODEL_NAME

accelerate launch --num_machines $NUM_MACHINES --num_processes $NUM_GPUS --main_process_port $PORT --mixed_precision no --dynamo_backend no \
    lmms_eval/__main__.py \
    --model $MODEL_NAME \
    --model_args pretrained=$CKPT_PATH \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --verbosity='DEBUG' \
    --output_path $OUTPUT_PATH