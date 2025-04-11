python submit.py \
    ./scripts/train/pretrain_thesis.sh \
    --ems_project thesis-training \
    --experiment_name thesis-llama31-siglip2-globalview \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 8 \
    --num_nodes 1 \
    --cpu 64 \
    --memory 512 \
    --pvc