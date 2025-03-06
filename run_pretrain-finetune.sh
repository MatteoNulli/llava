python submit.py \
    ./scripts/train/pretrain-finetune_thesis-llama31.sh \
    --ems_project thesis-train \
    --experiment_name pre_sft_no_global_view \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 8 \
    --num_nodes 1 \
    --cpu 60 \
    --memory 512 \
    --pvc