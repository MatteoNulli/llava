python krylov/submit_nlp.py \
    krylov/scripts/eval_lmms_workflow.sh \
    --ems_project llava-benchmarking \
    --experiment_name benchmarking-ebench-sm-internvl2 \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/image-understanding/0.0.7-bumped4:latest \
    --gpu_per_node 4 \
    --num_nodes 1 \
    --cpu 16 \
    --memory 128 \
    --pvc