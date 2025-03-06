#!/bin/bash


export http_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export https_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export no_proxy=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com
export HTTP_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export HTTPS_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export NO_PROXY=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com

MODEL_DIR=/mnt/nushare2/data/mnulli/finetuning/from-blip-pretrain/e-Llama-3_1-8B-Instruct-lora-1_5M_fashion-4-4
MODEL_NAME='e-Llama-3_1-8B-Instruct-lora-1_5M_fashion-4-4'
CONV_MODE='llava_llama_3'
MODEL_BASE=/mnt/nushare2/data/vorshulevich/models/e-Llama-3_1-8B-Instruct-DPO-epoch-1/

DATA_DIR=/mnt/nushare2/data/mnulli/llava_ov/playground

cd /opt/krylov-workflow/src/run_fn_0/


#eBench-Fashion-sm
SPLIT="ebench-sm-tc"

python -m llava.eval.model_vqa_ebench-sm \
    --model-path $MODEL_DIR \
    --model-base $MODEL_BASE \
    --question-file $DATA_DIR/ebench-sm/$SPLIT.tsv \
    --answers-file $DATA_DIR/ebench-sm/answers/$SPLIT/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE


python scripts/eval/eval-ecommerce/convert_ebench-sm_for_submission.py \
    --annotation-file $DATA_DIR/ebench-sm/$SPLIT.tsv \
    --result-dir $DATA_DIR/ebench-sm/answers/$SPLIT \
    --upload-dir $DATA_DIR/ebench-sm/answers_upload/$SPLIT \
    --experiment $MODEL_NAME


#eBench-General-sm
SPLIT="ebench-sm-gen"


python -m llava.eval.model_vqa_ebench-sm \
    --model-path $MODEL_DIR \
    --model-base $MODEL_BASE \
    --question-file $DATA_DIR/ebench-sm/$SPLIT.tsv \
    --answers-file $DATA_DIR/ebench-sm/answers/$SPLIT/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE


python scripts/eval/eval-ecommerce/convert_ebench-sm_for_submission.py \
    --annotation-file $DATA_DIR/ebench-sm/$SPLIT.tsv \
    --result-dir $DATA_DIR/ebench-sm/answers/$SPLIT \
    --upload-dir $DATA_DIR/ebench-sm/answers_upload/$SPLIT \
    --experiment $MODEL_NAME