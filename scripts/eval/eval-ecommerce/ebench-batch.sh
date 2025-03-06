#!/bin/bash

SPLIT="ebench-sm-title-cat_uk-gpt4o"

MODEL_DIR=/mnt/nushare2/data/vorshulevich/models/vlm/llava_ov/finetune/llava-onevision-siglip-so400m-patch14-384-e-Llama-3_1-8B-Instruct-DPO-epoch-1-si_stage_am9-lora
MODEL_NAME='llava-onevision-siglip-so400m-patch14-384-e-Llama-3_1-8B-Instruct-DPO-epoch-1-si_stage_am9-lora-v2'
CONV_MODE='llava_llama_3'
MODEL_BASE=/mnt/nushare2/data/vorshulevich/models/e-Llama-3_1-8B-Instruct-DPO-epoch-1

python -m llava.eval.model_vqa_ebench-sm-batch \
    --model-path $MODEL_DIR \
    --model-base $MODEL_BASE \
    --question-file playground/data/eval/ebench-sm/$SPLIT.tsv \
    --answers-file playground/data/eval/ebench-sm/answers/$SPLIT/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE

# mkdir -p playground/data/eval/ebench/answers_upload/$SPLIT

python scripts/eval-ecommerce/convert_ebench-sm_for_submission.py \
    --annotation-file playground/data/eval/ebench-sm/$SPLIT.tsv \
    --result-dir playground/data/eval/ebench-sm/answers/$SPLIT \
    --upload-dir playground/data/eval/ebench-sm/answers_upload/$SPLIT \
    --experiment $MODEL_NAME