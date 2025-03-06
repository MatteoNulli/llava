from .submit_pykrylov import submit_task
from fire import Fire

model_configs = {
    "llava_1_5_7b_hf": {
        "model_name": "llava_hf",
        "model_args": "pretrained=llava-hf/llava-1.5-7b-hf,device_map=auto",
    },
    "llava_1_5_13b_hf": {
        "model_name": "llava_hf",
        "model_args": "pretrained=llava-hf/llava-1.5-13b-hf,device_map=auto",
    },
    "llava_1_6_7b_hf": {
        "model_name": "llava_hf",
        "model_args": "pretrained=llava-hf/llava-v1.6-mistral-7b-hf,device_map=auto",
    },
    "llava_1_6_13b_hf": {
        "model_name": "llava_hf",
        "model_args": "pretrained=llava-hf/llava-v1.6-vicuna-13b-hf,device_map=auto",
        "gpu": "a100",
        "cluster": "tess137",
        "namespace": "ebay-slc-a100",
    },
    "idefics2_8b": {
        "model_name": "idefics2",
        "model_args": "device_map=auto",
    },
    "DocOwl1_5_Chat": {
        "model_name": "mplug_docowl",
        "model_args": "pretrained=/mnt/iu-pvc/lmms-eval/models/DocOwl/DocOwl1.5-Chat,device_map=auto",
        "image": "hub.tess.io/splaneta/lmms-eval:mplug-0.0.2",
    },
    "TinyLLaVA_Phi_2_SigLIP_3B": {
        "model_name": "tinyllava",
    },
    "XComposer2_7b": {
        "model_name": "xcomposer2",
        "model_args": "pretrained=/mnt/iu-pvc/lmms-eval/models/XComposer2/vl_7b/,device_map=auto",
        "image": "hub.tess.io/image-understanding/lmms-eval:2bd3bc3cf3eefd03792298950dea409d2672b0d4"
    },
    "XComposer2d5": {
        "model_name": "xcomposer2d5",
        "model_args": "half=false,use_json_enforcer=true",
        "gpu": "a100",
        "cluster": "tess137",
        "namespace": "ebay-slc-a100",
    },

}


def submit_multiple(task_name: str):
    common_submit_args = {
        "script": "krylov/scripts/evaluate.sh",
        "lmms_eval_task": task_name,
    }
    for model_name, model_config in model_configs.items():
        submit_task(**common_submit_args, **model_config, experiment_name=f"{task_name}_{model_name}")


if __name__ == "__main__":
    Fire(submit_multiple)
