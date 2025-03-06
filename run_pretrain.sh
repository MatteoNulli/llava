python submit.py \
    pretrain_thesis.sh \
    --ems_project thesis-train \
    --experiment_name thesis-oldcodebase-llama31-clip \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 8 \
    --num_nodes 1 \
    --cpu 64 \
    --memory 512 \
    --pvc