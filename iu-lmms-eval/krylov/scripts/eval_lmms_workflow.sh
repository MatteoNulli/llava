#!/bin/bash


export http_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export https_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export no_proxy=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com
export HTTP_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export HTTPS_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export NO_PROXY=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com

NUM_MACHINES=${NUM_MACHINES:-1}
NUM_GPUS=${NUM_GPUS:-8}

cd /opt/krylov-workflow/src/run_fn_0/iu-lmms-eval/

TASK=ebench_sm,ebench_sm_gen,textvqa_test,ai2d,mme,mmbench_en_dev,mmstar,hallusion_bench_image,cvbench,mmmu_val

if [[ "$TASK" =~ mmbench ]]; then
    pip install --proxy http://httpproxy-tcop.vip.ebay.com:80 openpyxl
fi

CKPT_PATH=/mnt/nushare2/data/mnulli/finetuning/from-blip-pretrain/e-Llama-3_1-8B-Instruct-DPO-epoch-1-lora-1_5M_fashion-4-4
MODEL_BASE=/mnt/nushare2/data/vorshulevich/models/e-Llama-3_1-8B-Instruct-DPO-epoch-1
# builder in LLava expect a particular model_name for parsing
MODEL_NAME=llava_onevision
CONV_MODE=llava_llama_3

echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

OUTPUT_PATH=/mnt/nushare2/data/mnulli/llava_ov/playground/lmms_eval_results/$TASK_SUFFIX/$MODEL_NAME

accelerate launch --num_machines $NUM_MACHINES --num_processes $NUM_GPUS --main_process_port 12380 --mixed_precision no --dynamo_backend no \
    lmms_eval/__main__.py \
    --model $MODEL_NAME \
    --model_args pretrained=$CKPT_PATH,model_base=$MODEL_BASE,conv_template=$CONV_MODE \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --verbosity='DEBUG' \
    --output_path $OUTPUT_PATH