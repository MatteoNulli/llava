python ./krylov/submit.py \
    ./krylov/scripts/eval_lmms_workflow.sh \
    --ems_project thesis-benchmarking \
    --experiment_name standard_allbenchs32 \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 2 \
    --num_nodes 1 \
    --cpu 16 \
    --memory 128 \
    --pvc