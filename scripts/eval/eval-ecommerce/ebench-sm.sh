#!/bin/bash

SPLIT="ebench-sm-tc"

# Change Paths to your {Model, Data}
MODEL_NAME=superpod_lr5_ga4_llava-onevision-siglip-e-Llama-3_1-8B-si_stage_am9-lora

MODEL_DIR=/mnt/nushare2/data/vorshulevich/models/vlm/llava_ov/finetune/$MODEL_NAME
CONV_MODE='llava_llama_3'
MODEL_BASE=/mnt/nushare2/data/vorshulevich/models/e-Llama-3_1-8B-Instruct-DPO-epoch-1
PATH_TO_PLAYGROUND=/mnt/nushare2/data/vorshulevich/data/eval_data/playground

python -m llava.eval.model_vqa_ebench-sm \
    --model-path $MODEL_DIR \
    --model-base $MODEL_BASE \
    --question-file $PATH_TO_PLAYGROUND/ebench-sm/$SPLIT.tsv \
    --answers-file $PATH_TO_PLAYGROUND/ebench-sm/answers/$SPLIT/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE

python scripts/eval/eval-ecommerce/convert_ebench-sm_for_submission.py \
    --annotation-file $PATH_TO_PLAYGROUND/ebench-sm/$SPLIT.tsv \
    --result-dir $PATH_TO_PLAYGROUND/ebench-sm/answers/$SPLIT \
    --upload-dir $PATH_TO_PLAYGROUND/ebench-sm/answers_upload/$SPLIT \
    --experiment $MODEL_NAME