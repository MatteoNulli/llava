python submit.py \
    finetune_freezelm.sh \
    --ems_project llava-finetuning \
    --experiment_name freezinglm_sft \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/gen-ai/ellement:latest \
    --gpu_per_node 8 \
    --num_nodes 1 \
    --cpu 60 \
    --memory 512 \
    --pvc