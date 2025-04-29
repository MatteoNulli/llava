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



## Modifying permissions
# N=0
# chmod -R a+rw /mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data_cap/arrays/partition_$N


for N in {0..9}; do
    chmod -R a+rw "/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data_cap/arrays/partition_${N}"
done

echo "Modified Permissions of /mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data_cap/"


# cd /opt/krylov-workflow/src/run_fn_0/
# N=9
# echo unzipping /mnt/nushare2/data/mnulli/thesis/data/sam2/partitions/partition_$N.tar.gz into /mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data_cap/arrays/partition_$N
# tar -xvzf /mnt/nushare2/data/mnulli/thesis/data/sam2/partitions/partition_$N.tar.gz --strip-components=1 -C /mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data_cap/arrays/partition_$N


