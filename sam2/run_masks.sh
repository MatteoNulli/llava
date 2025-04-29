python submit.py \
    changing_permissions.sh \
    --ems_project thesis-train \
    --experiment_name changing_permissions \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 1 \
    --num_nodes 1 \
    --cpu 16 \
    --memory 128 \
    --pvc