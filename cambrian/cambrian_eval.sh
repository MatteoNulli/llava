#!/bin/bash

cd /data/chatgpt/notebooks/mnulli/thesis/cambrian/

bash eval/scripts/run_benchmark.sh \
    --benchmark MMVP \
    --ckpt /mnt/nushare2/data/mnulli/model_zoos/opensource-vlms/models--nyu-visionx--cambrian-8b/snapshots/86557aee30551709fcfafd521c7ef009c32eec94 \
    --conv_mode llama_3