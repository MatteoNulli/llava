python submit.py \
    ebench-sm3.sh\
    --ems_project llava-benchmarking \
    --experiment_name eval-ebench-sm-molmo \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/gen-ai/ellement:latest \
    --gpu_per_node 8 \
    --num_nodes 1 \
    --cpu 60 \
    --memory 512 \
    --pvc