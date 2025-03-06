python submit.py \
    finetune_lora_lilium_7b-matteo-oldpretrain-siglip.sh \
    --ems_project llava-finetuning \
    --experiment_name lilium_siglip-llava_mix_lr5 \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/vorshulevich/open-clip:latest \
    --gpu_per_node 8 \
    --num_nodes 1 \
    --cpu 60 \
    --memory 512 \
    --pvc