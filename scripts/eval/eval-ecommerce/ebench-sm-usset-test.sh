#!/bin/bash


##us+uk
# SPLIT="ebench-sm"
##us
# SPLIT="ebench-sm_us"
##us2
SPLIT="ebench-sm-us-gpt4o"


# MODEL_DIR=/mnt/nushare2/data/vorshulevich/models/vlm/llava_ov/finetune/llava-onevision-openai-clip-vitl-patch14-336-e-Llama-3_1-8B-Instruct-DPO-epoch-1-si_stage_old-llava-params-lora
# MODEL_NAME='llava-onevision-openclip-vitl-patch14-336-e-Llama-3_1-8B-Instruct-DPO-epoch-1-si_stage_old-llava-params-lora'
# CONV_MODE='llava_llama_3'
# MODEL_BASE=/mnt/nushare2/data/vorshulevich/models/e-Llama-3_1-8B-Instruct-DPO-epoch-1

MODEL_DIR=/mnt/nushare2/data/mnulli/finetuning/from-blip-pretrain/e-Llama-3_1-8B-Instruct-DPO-epoch-1-lora-1_5M_fashion-4-4
MODEL_NAME='e-Llama-3_1-8B-Instruct-DPO-epoch-1-lora-1_5M_fashion-4-4-v2'
CONV_MODE='llava_llama_3'
MODEL_BASE=/mnt/nushare2/data/vorshulevich/models/e-Llama-3_1-8B-Instruct-DPO-epoch-1

python -m llava.eval.model_vqa_ebench-sm \
    --model-path $MODEL_DIR \
    --model-base $MODEL_BASE \
    --question-file playground/data/eval/ebench-sm/$SPLIT.tsv \
    --answers-file playground/data/eval/ebench-sm/answers/$SPLIT/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE


python scripts/eval/eval-ecommerce/convert_ebench-sm_for_submission.py \
    --annotation-file playground/data/eval/ebench-sm/$SPLIT.tsv \
    --result-dir playground/data/eval/ebench-sm/answers/$SPLIT \
    --upload-dir playground/data/eval/ebench-sm/answers_upload/$SPLIT \
    --experiment $MODEL_NAME