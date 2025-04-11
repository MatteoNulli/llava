#!/bin/bash


# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_global_view_llava-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora 
MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/no_global_view_llava-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_standard_llava/standard_llava15-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora

MODEL_NAME=$(basename "$MODEL_DIR")
echo evaluating vismin-bench on $MODEL_NAME

MODEL_BASE=/mnt/mtrepo/data/wwalentynowicz/models/Meta-Llama-3_1-8B-Instruct
CONV_MODE='llama3'

DATA_DIR=/mnt/nushare2/data/mnulli/llava_ov/playground/mair-lab___vismin-bench/default/0.0.0/496293372f53df900b502f12f133fc5cad5d499a

cd llava/eval/vismin-bench

python -m eval_vismin \
    --model-path $MODEL_DIR \
    --model-base $MODEL_BASE \
    --question-file $DATA_DIR/combined.csv \
    --answers-file $DATA_DIR/answers/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE

# mkdir -p playground/data/eval/ebench/answers_upload/$SPLIT


# python scripts/convert_ebench-sm_for_submission.py \
#     --annotation-file playground/data/eval/ebench-sm/$SPLIT.tsv \
#     --result-dir playground/data/eval/ebench-sm/answers/$SPLIT \
#     --upload-dir playground/data/eval/ebench-sm/answers_upload/$SPLIT \
#     --experiment $MODEL_NAME