#!/bin/bash

export http_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export https_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export no_proxy=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com
export HTTP_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export HTTPS_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export NO_PROXY=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com


# pip install --upgrade --proxy http://httpproxy-tcop.vip.ebay.com:80 numpy==1.26.4
# pip install --upgrade --proxy http://httpproxy-tcop.vip.ebay.com:80 matplotlib
# pip install --upgrade --proxy http://httpproxy-tcop.vip.ebay.com:80 sam2

## LLaVa Captioning
# echo creating masks for Captioning Data
# DATA_PATH=/mnt/nushare2/data/mnulli/pretrainingdata/blip_laion_cc_sbu_558k.json  ## PATH TO LLAVA Captioning JSON FILE (.json) 
# ARRAYS_DIR=/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data_try/arrays ## PATH TO STORED NPARRAYS WITH MASKS
# METADATA_DIR=/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data_try/metadata ## PATH TO STORED METADATA ABOUT MASKS 
# CAPTIONING=True ##FLAG FOR PROCESSING

## LLaVa SFT
# echo creating masks for SFT Data
# DATA_PATH=/mnt/nushare2/data/mnulli/verified_conversations/finetuningdata/llava_mix665k_format_adjusted.json ## PATH TO LLAVA SFT JSON FILE (.json) 
# ARRAYS_DIR=/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data_try/arrays ## PATH TO STORED NPARRAYS WITH MASKS
# METADATA_DIR=/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data_try/metadata ## PATH TO STORED METADATA ABOUT MASKS 
# CAPTIONING=False ##FLAG FOR PROCESSING

## Cambrian SFT
# echo creating masks for Cambrian SFT Data
# DATA_PATH=PATH TO CAMBRIAN JSON FILE (.json) 
# ARRAYS_DIR=PATH TO STORED NPARRAYS WITH MASKS
# METADATA_DIR=PATH TO STORED METADATA ABOUT MASKS
# CAPTIONING=False ##FLAG FOR PROCESSING
# CAMBRIAN=True ##FLAG FOR PROCESSING



## BENCHMARKS
echo creating masks for Benchmarks Data "aro"
BENCHMARKS_IMAGES_DIR=/mnt/nushare2/data/mnulli/llava_ov/playground/gowitheflow___aro-visual-relation/combined
ARRAYS_DIR=/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data_benchmarks/gowitheflow___aro/arrays
METADATA_DIR=/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data_benchmarks/gowitheflow___aro/metadata
CAPTIONING=False ##FLAG FOR PROCESSING

PARTITION_ID=0
TOTAL_PARTITIONS=10

cd /opt/krylov-workflow/src/run_fn_0/


SAM2_CHECKPOINT=/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data/checkpoints/sam2.1_hiera_large.pt
DEVICE=cuda
MODEL_CFG=configs/sam2.1/sam2.1_hiera_l.yaml


python automatic_mask_generator_llava.py \
    --sam2_checkpoint $SAM2_CHECKPOINT \
    --model_cfg $MODEL_CFG \
    --device $DEVICE \
    --metadata_directory $METADATA_DIR \
    --arrays_directory $ARRAYS_DIR \
    --data_path $DATA_PATH \
    --benchmark_images_dir $BENCHMARKS_IMAGES_DIR \
    --total-partitions $TOTAL_PARTITIONS \
    --partition-id $PARTITION_ID \
    --captioning $CAPTIONING \
    --cambrian $CAMBRIAN
