#!/bin/bash

export http_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export https_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export no_proxy=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com
export HTTP_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export HTTPS_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export NO_PROXY=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com


pip install --upgrade --proxy http://httpproxy-tcop.vip.ebay.com:80 pip
pip install --upgrade --proxy http://httpproxy-tcop.vip.ebay.com:80 hydra-core
pip install --upgrade --proxy http://httpproxy-tcop.vip.ebay.com:80 fire


## Cambrian SFT
# echo creating masks for Cambrian SFT Data
DATA_PATH=/mnt/nushare2/data/mnulli/thesis/data/training_data/nyu-visionx--Cambrian-10M--extracted/Cambrian7M_withsystemprompt.jsonl # DATA_PATH=PATH TO CAMBRIAN JSON FILE (.jsonl)
ARRAYS_DIR=/mnt/nushare2/data/mnulli/thesis/data/sam2/cambrian_segmentation_data/arrays # ARRAYS_DIR=PATH TO STORED NPARRAYS WITH MASKS
METADATA_DIR=/mnt/nushare2/data/mnulli/thesis/data/sam2/cambrian_segmentation_data/metadata  # METADATA_DIR=PATH TO STORED METADATA ABOUT MASKS
CAPTIONING=False ##FLAG FOR PROCESSING
CAMBRIAN=True ##FLAG FOR PROCESSING

cd /opt/krylov-workflow/src/run_fn_0/sam2/


pip install --upgrade --proxy http://httpproxy-tcop.vip.ebay.com:80 hydra-core
pip install --upgrade --proxy http://httpproxy-tcop.vip.ebay.com:80 fire
pip install --upgrade --proxy http://httpproxy-tcop.vip.ebay.com:80 iopath



SAM2_CHECKPOINT=/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data/checkpoints/sam2.1_hiera_large.pt # SAM2_CHECKPOINT=PATH TO SAM2 CHECKPOINT (.pt)
DEVICE=cuda

PARTITION_ID=4
TOTAL_PARTITIONS=10

MODEL_CFG=configs/sam2.1/sam2.1_hiera_l.yaml


python automatic_mask_generator_llava_cambrian.py \
    --sam2_checkpoint $SAM2_CHECKPOINT \
    --model_cfg $MODEL_CFG \
    --device $DEVICE \
    --metadata_directory $METADATA_DIR \
    --arrays_directory $ARRAYS_DIR \
    --data_path $DATA_PATH \
    --total-partitions $TOTAL_PARTITIONS \
    --partition-id $PARTITION_ID \
    --captioning $CAPTIONING \
    --cambrian $CAMBRIAN
