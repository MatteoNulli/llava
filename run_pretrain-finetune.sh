python submit.py \
    ./scripts/train/pretrain-finetune_thesis-llama32.sh \
    --ems_project thesis-training \
    --experiment_name llama32-3B-noglob-1cap-1sft \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 8 \
    --num_nodes 1 \
    --cpu 60 \
    --memory 512 \
    --pvc