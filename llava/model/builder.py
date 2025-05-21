#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil
import json

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
import torch
from llava.model import *
from llava.train.train import find_all_linear_names_vision
from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device not in ["cuda", "cuda:0"]:
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    if (
        "llava" in model_name.lower()
        or "lilium" in model_name.lower()
        or "e-llama" in model_name.lower()
    ):
        # print('model_name', model_name)
        # Load LLaVA model
        # if "lora" in model_name.lower() and model_base is None:
        #     warnings.warn(
        #         "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
        #     )
        # captioning_only = True
        # if captioning_only:
        #     from llava.model.language_model.llava_llama import LlavaConfig

        #     cfg_pretrained = LlavaConfig.from_pretrained(model_path)
        #     tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        #     model = LlavaLlamaForCausalLM.from_pretrained(
        #         model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
        #     )
        #     if os.path.exists(os.path.join(model_path, "mm_bom_mask_token.bin")):
        #         model.mm_bom_mask_token = torch.load(
        #             os.path.join(model_path, "mm_bom_mask_token.bin"),
        #             map_location="cpu",
        #         )

        if "openclip" in model_name.lower() and model_base is not None:
            print("\n Loading LLaVA from model base...")
            from llava.model.language_model.llava_llama import LlavaConfig

            cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
            )

            print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"),
                    map_location="cpu",
                )

                non_lora_trainables = {
                    (k[11:] if k.startswith("base_model.") else k): v
                    for k, v in non_lora_trainables.items()
                }
                if any(k.startswith("model.model.") for k in non_lora_trainables):
                    non_lora_trainables = {
                        (k[6:] if k.startswith("model.") else k): v
                        for k, v in non_lora_trainables.items()
                    }

                if "model.mm_bom_mask_token.mm_bom_mask_token" in non_lora_trainables:
                    non_lora_trainables["model.mm_bom_mask_token"] = (
                        non_lora_trainables.pop(
                            "model.mm_bom_mask_token.mm_bom_mask_token"
                        )
                    )

                model.load_state_dict(non_lora_trainables, strict=False)

                if "model.mm_bom_mask_token" in non_lora_trainables:
                    print(
                        f"Loading mm_bom_mask_token weights from {os.path.join(model_path, 'non_lora_trainables.bin')}"
                    )

                    # Get the tensor from the state dict
                    loaded_token = non_lora_trainables["model.mm_bom_mask_token"]

                    # Ensure the tensor is on the same device as the model's parameter
                    loaded_token = loaded_token.to(
                        model.mm_bom_mask_token.mm_bom_mask_token.device
                    )

                    # Update the parameter in your BOMMaskToken instance
                    # This copies the data from loaded_token into the existing parameter tensor
                    model.mm_bom_mask_token.mm_bom_mask_token.data.copy_(loaded_token)

            from peft import PeftModel, LoraConfig

            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)

        elif "lora" in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig

            print("model_path", model_path)
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print("Loading LLaVA from base model...")
            # print("lora_cfg_pretrained", lora_cfg_pretrained)
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )

            print("Loading additional LLaVA weights...")
            ##trying something, to be deleted
            if (
                model.config.mm_projector_type == "mlp2x_gelu,subobject_tokenization"
                and os.path.exists(
                    "/data/chatgpt/notebooks/mnulli/merged_non_lora_trainables.bin"
                )
            ):

                non_lora_trainables = torch.load(
                    "/data/chatgpt/notebooks/mnulli/merged_non_lora_trainables.bin",
                    map_location="cpu",
                )
            elif os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"),
                    map_location="cpu",
                )
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id, filename=filename, subfolder=subfolder
                    )
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(
                    model_path, "non_lora_trainables.bin"
                )
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v
                for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v
                    for k, v in non_lora_trainables.items()
                }

            if "model.mm_bom_mask_token.mm_bom_mask_token" in non_lora_trainables:
                non_lora_trainables["model.mm_bom_mask_token"] = (
                    non_lora_trainables.pop("model.mm_bom_mask_token.mm_bom_mask_token")
                )

            model.load_state_dict(non_lora_trainables, strict=False)

            if "model.mm_bom_mask_token" in non_lora_trainables:
                print(
                    f"Loading mm_bom_mask_token weights from {os.path.join(model_path, 'non_lora_trainables.bin')}"
                )

                # Get the tensor from the state dict
                loaded_token = non_lora_trainables["model.mm_bom_mask_token"]

                # Ensure the tensor is on the same device as the model's parameter
                loaded_token = loaded_token.to(
                    model.mm_bom_mask_token.mm_bom_mask_token.device
                )

                # Update the parameter in your BOMMaskToken instance
                # This copies the data from loaded_token into the existing parameter tensor
                model.mm_bom_mask_token.mm_bom_mask_token.data.copy_(loaded_token)

            from peft import PeftModel, LoraConfig

            print("Loading LoRA weights...")
            try:
                model = PeftModel.from_pretrained(model, model_path)
            except:
                # # Read the JSON file
                input_file = model_path + "/adapter_config.json"
                output_file = "/data/chatgpt/notebooks/mnulli" + "/adapter_config.json"
                with open(input_file, "r") as f:
                    data = json.load(f)

                # Remove null types if they exists
                data = {k: v for k, v in data.items() if v is not None}
                if "lora_bias" in data:
                    del data["lora_bias"]

                # Write the modified JSON to new file
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)

                print(
                    f"Successfully removed 'usless' configs and saved to {output_file}"
                )

                # Get the config first
                config_dict = LoraConfig.from_pretrained(
                    "/data/chatgpt/notebooks/mnulli"
                ).to_dict()

                # Create new config and load model

                # print('config_dict', config_dict)
                config = LoraConfig(**config_dict)
                model = PeftModel.from_pretrained(model, model_path, config=config)

            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")
        elif model_base is not None:
            # this may be mm projector only
            print("Loading LLaVA from base model...")
            if "mpt" in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                    shutil.copyfile(
                        os.path.join(model_base, "configuration_mpt.py"),
                        os.path.join(model_path, "configuration_mpt.py"),
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(
                    model_path, trust_remote_code=True
                )
                model = LlavaMptForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )
                print("model", model)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )

            mm_projector_weights = torch.load(
                os.path.join(model_path, "mm_projector.bin"), map_location="cpu"
            )
            mm_projector_weights = {
                k: v.to(torch.float16) for k, v in mm_projector_weights.items()
            }
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
            elif "mistral" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )

    image_processor = None

    if (
        "llava" in model_name.lower()
        or "lilium" in model_name.lower()
        or "e-llama" in model_name.lower()
    ):
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()

        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)

        if "openclip-lora" in model_name.lower():
            print("Loading Vision Encoder LoRA weights...")
            # print('vision_tower', vision_tower)
            # print('model_path', model_path)
            from peft import PeftModel, LoraConfig

            # # Read the JSON file
            input_file = model_path + "/adapter_config.json"
            output_file = "/data/chatgpt/notebooks/mnulli" + "/adapter_config.json"
            with open(input_file, "r") as f:
                data = json.load(f)

            # Remove null types if they exists
            data = {k: v for k, v in data.items() if v is not None}
            if "lora_bias" in data:
                del data["lora_bias"]

            # Write the modified JSON to new file
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

            print(f"Successfully removed 'usless' configs and saved to {output_file}")

            # Get the config first
            config_dict = LoraConfig.from_pretrained(
                "/data/chatgpt/notebooks/mnulli"
            ).to_dict()

            # Create new config and load model

            # print('config_dict', config_dict)
            config = LoraConfig(**config_dict)
            # print(find_all_linear_names_vision(vision_tower.vision_tower.vision_model))

            vision_tower = PeftModel.from_pretrained(
                vision_tower,
                model_path,
                # task_type='FEATURE_EXTRACTION',
                config=config,
                # # Specify the correct target modules based on the printout above
                target_modules=find_all_linear_names_vision(
                    vision_tower.vision_tower.vision_model
                ),
            )

            # print('vision_tower before', vision_tower)

            # Merge LoRA weights
            vision_tower = vision_tower.merge_and_unload()

            # print('vision_tower after', vision_tower)

        vision_tower.to(device=device)

        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
