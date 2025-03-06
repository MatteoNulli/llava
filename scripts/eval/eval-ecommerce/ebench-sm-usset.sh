#!/bin/bash

#us
SPLIT="ebench-sm-gen"

MODEL_DIR=/mnt/nushare2/data/vorshulevich/models/vlm/llava_ov/finetune/superpod_llava-onevision-siglip-e-Llama-3_1-8B-mid_stage-si_ov_stage_am9-lora
MODEL_NAME='superpod_llava-onevision-siglip-e-Llama-3_1-8B-mid_stage-si_ov_stage_am9-lora-1'
CONV_MODE='llava_llama_3'
MODEL_BASE=/mnt/nushare2/data/vorshulevich/models/e-Llama-3_1-8B-Instruct-DPO-epoch-1
PLAYGROUND_DIR=/mnt/nushare2/data/mnulli/llava_ov/playground

python -m llava.eval.model_vqa_ebench-sm \
    --model-path $MODEL_DIR \
    --model-base $MODEL_BASE \
    --question-file $PLAYGROUND_DIR/ebench-sm/$SPLIT.tsv \
    --answers-file $PLAYGROUND_DIR/ebench-sm/answers/ebench-sm-gen-1/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE


python scripts/eval/eval-ecommerce/convert_ebench-sm_for_submission.py \
    --annotation-file $PLAYGROUND_DIR/ebench-sm/$SPLIT.tsv \
    --result-dir $PLAYGROUND_DIR/ebench-sm/answers/ebench-sm-gen-1 \
    --upload-dir $PLAYGROUND_DIR/ebench-sm/answers_upload/ebench-sm-gen-1 \
    --experiment $MODEL_NAME