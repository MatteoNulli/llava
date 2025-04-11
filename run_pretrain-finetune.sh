python submit.py \
    ./scripts/train/pretrain-finetune_thesis-llama31-8b.sh \
    --ems_project thesis-training \
    --experiment_name llama31-avg_global-view-2cap-2epochssft \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 8 \
    --num_nodes 1 \
    --cpu 60 \
    --memory 512 \
    --pvc