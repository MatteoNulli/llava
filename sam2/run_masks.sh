python submit.py \
    automatic_masks_cambrian.sh \
    --ems_project thesis-training \
    --experiment_name automatic_masks_cambrian_p4 \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 8 \
    --num_nodes 1 \
    --cpu 16 \
    --memory 128 \
    --pvc