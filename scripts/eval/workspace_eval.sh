#!/bin/bash


export http_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export https_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export no_proxy=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com
export HTTP_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export HTTPS_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export NO_PROXY=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com

MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_global_view_llava-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
MODEL_NAME='sam2-trying-sft'
CONV_MODE='llama3'
MODEL_BASE=/mnt/mtrepo/data/wwalentynowicz/models/Meta-Llama-3_1-8B-Instruct

PLAYGROUND_DIR=/mnt/nushare2/data/mnulli/llava_ov/playground


# #CVBench
echo evaluating CVBench
SPLIT="test"
python -m llava.eval.model_vqa_nyu-cvbench \
    --model-path $MODEL_DIR \
    --model-base $MODEL_BASE \
    --question-file $PLAYGROUND_DIR/nyu-cvbench/data/$SPLIT.parquet \
    --answers-file $PLAYGROUND_DIR/nyu-cvbench/answers/$SPLIT/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE


python -m llava.eval.convert_nyu-cvbench_for_submission \
    --annotation-file $PLAYGROUND_DIR/nyu-cvbench/data/$SPLIT.parquet \
    --result-dir $PLAYGROUND_DIR/nyu-cvbench/answers/$SPLIT \
    --upload-dir $PLAYGROUND_DIR/nyu-cvbench/answers_upload/$SPLIT \
    --experiment $MODEL_NAME

# #MMbench
# echo evaluating MMbench
# SPLIT="mmbench_dev_20230712"

# python -m llava.eval.model_vqa_mmbench \
#     --model-path $MODEL_DIR \
#     --model-base $MODEL_BASE \
#     --question-file $PLAYGROUND_DIR/mmbench/mmbench_dev_20230712.tsv \
#     --answers-file $PLAYGROUND_DIR/mmbench/answers/mmbench_dev_20230712/$MODEL_NAME.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode $CONV_MODE \
#     --annotation-file $PLAYGROUND_DIR/mmbench/$SPLIT.tsv \
#     --result-dir $PLAYGROUND_DIR/mmbench/answers/$SPLIT \
#     --experiment $MODEL_NAME


# MMMU
# echo evaluating MMMU
# SPLIT="mmmu"

# python -m llava.eval.mmmu_eval \
#     --model_path $MODEL_DIR \
#     --model_base $MODEL_BASE \
#     --answers_file $PLAYGROUND_DIR/mmmu/answers/$MODEL_NAME.jsonl \
#     --question_file $PLAYGROUND_DIR/mmmu/data_val/validation.parquet \
#     --question_extension 'Answer the question with only a single letter from the options.' \
#     --temperature 0 \
#     --conv_mode $CONV_MODE


# python llava/eval/mmmu_test.py \
#     --answers_file $PLAYGROUND_DIR/mmmu/answers/${MODEL_NAME}_0.jsonl \
#     --output_file $PLAYGROUND_DIR/mmmu/answers/incorrect_${MODEL_NAME}_0.jsonl \
#     --csv_file $PLAYGROUND_DIR/mmmu/answers/correct_${MODEL_NAME}_0.jsonl \



# # #MME
# echo evaluating MME
# python -m llava.eval.model_mme \
#     --model-path $MODEL_DIR \
#     --model-base $MODEL_BASE \
#     --question-file $PLAYGROUND_DIR/mme_old/testdata.parquet \
#     --answers-file  $PLAYGROUND_DIR/mme_old/answers/$MODEL_NAME.jsonl \
#     --answers-folder  $PLAYGROUND_DIR/mme_old/answers/$MODEL_NAME \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode $CONV_MODE


# python llava/eval/mme_calculation.py \
#     --results_dir  $PLAYGROUND_DIR/mme_old/answers/$MODEL_NAME \


# # #sqa
# echo evaluating Science Question Answering - sqa
# DATA_DIR= $PLAYGROUND_DIR/scienceqa_old
# python llava/eval/model_vqa_science.py \
#     --model-path $MODEL_DIR \
#     --model-base $MODEL_BASE \
#     --question-file $DATA_DIR/llava_test_CQM-A.json \
#     --image-folder $DATA_DIR/images/test \
#     --answers-file $DATA_DIR/answers/$MODEL_NAME.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode $CONV_MODE

# python llava/eval/eval_science_qa.py \
#     --base-dir $DATA_DIR \
#     --result-file $DATA_DIR/answers/$MODEL_NAME.jsonl \
#     --output-file $DATA_DIR/answers/${MODEL_NAME}_output.jsonl \
#     --output-result $DATA_DIR/answers/${MODEL_NAME}_result.json

