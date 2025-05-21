import open_clip
import torch
import numpy as np
import requests
import random
import re
import argparse
import os
import io
import json
import pandas as pd
import shortuuid
import base64
import math
import glob

import inspect
import json
import argparse
import pandas as pd
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm
import ssl
import urllib.request
from urllib.error import URLError
from getpass import getpass


from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

from internvl.utils import *

from open_clip import (
    create_model_from_pretrained,
    get_tokenizer,
)  # works on open-clip-torch>=2.23.0, timm>=0.9.8

# from qwen_vl_utils import process_vision_info

# from vllm import LLM
# from vllm.sampling_params import SamplingParams

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    AutoProcessor,
    GenerationConfig,
    Qwen2VLForConditionalGeneration,
    MllamaForConditionalGeneration,
    LlavaForConditionalGeneration,
)

import sys

llava_pth = os.path.abspath(
    os.path.join(os.path.split(__file__)[0], "../../../")
)  # -> /data/chatgpt/notebooks/mnulli/llava/llava/eval/conme/ -> /data/chatgpt/notebooks/mnulli/llava/
sys.path.append(f"{llava_pth}")
print("llava_pth", llava_pth)

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    load_image_from_base64,
    get_model_name_from_path,
)


def string_matching_acc(df, n_errors):
    print("Processing dataframe...")

    # Apply normalization to both columns
    df["answer"] = df["answer"]
    df["prediction"] = df["prediction"]

    # Remove unwanted strings
    df = df[df["answer"].str.lower() != "<_s_k_i_p_>"]

    # # Modified matching logic: check if answer appears within prediction
    # correct_predictions = sum(row['answer'] in row['prediction'] for _, row in df.iterrows())
    # total_predictions = len(df)
    # accuracy = correct_predictions / total_predictions

    # Exact string matching
    correct_predictions = sum(
        row["answer"] == row["prediction"] for _, row in df.iterrows()
    )
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions

    print(f"Results for {args.experiment}")
    print(f"Number of errors:{n_errors}")
    print(f"Number of rows with exact string matching: {correct_predictions}")
    print(f"Total String Matching Accuracy: {accuracy:.4f}")

    return accuracy


def load_generative_model(args, device):
    print(args.model_path.lower())

    if "qwen" in args.model_path.lower():

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )

        # default processer
        processor = AutoProcessor.from_pretrained(args.model_path)
        tokenizer = processor.tokenizer

    elif "internvl2" in args.model_path.lower():
        # device_map = split_model('InternVL2-8B')
        model = (
            AutoModel.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, use_fast=False
        )
        processor = None
        model_config = model.config

    elif "molmo" in args.model_path.lower():

        processor = AutoProcessor.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda:0",
        )
        tokenizer = processor.tokenizer

        # load the model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        model.to(dtype=torch.bfloat16)

    elif "cambrian" in args.model_path.lower():
        cambrian_pth = os.path.abspath(
            os.path.join(os.path.split(__file__)[0], "../../../cambrian")
        )  # -> /data/chatgpt/notebooks/mnulli/llava/llava/eval/conme/ --> /data/chatgpt/notebooks/mnulli/llava/cambrian
        # print("cambrian_pth", cambrian_pth)
        # sys.path.append(f"{cambrian_pth}")

        # from cambrian.conversation import conv_templates
        # from cambrian.mm_utils import (
        #     get_model_name_from_path,
        #     process_images,
        #     tokenizer_image_token,
        # )
        # from cambrian.model.builder import load_pretrained_model

        pretrained = args.model_path
        device = (
            torch.device(f"cuda:{accelerator.local_process_index}")
            if accelerator.num_processes > 1
            else "cuda:0"
        )

        model_name = get_model_name_from_path(pretrained)
        tokenizer, model, processor, context_len = load_pretrained_model(
            pretrained, None, model_name, device_map=device
        )

        if "nyu-visionx--cambrian-8b" in model_name:
            model_name = "cambrian-8b"
        conv_mode = {
            "cambrian-8b": "llama_3",
            "cambrian-13b": "vicuna_v1",
            "cambrian-34b": "chatml_direct",
        }.get(model_name)

    elif "sa2va" in args.model_path.lower():
        print(f"Loading sa2va from {args.model_path}")
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="cuda:0",
            device="cuda:0",
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, use_fast=False
        )

        processor = None
    elif "llava-hf" in args.model_path.lower():
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        print(f"Loading LLaVa-hf from {args.model_path}")

        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
        model = model.to("cuda")

        processor = AutoProcessor.from_pretrained(args.model_path)
        tokenizer = None

    else:
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)

        tokenizer, model, processor, context_len = load_pretrained_model(
            model_path,
            args.model_base,
            model_name,
            device="cuda:0",
            device_map="cuda:0",
        )

        model = model.to(device="cuda:0", dtype=torch.float16)

        if args.sam2:
            model.model.sam2_masking_token = True
        else:
            model.model.sam2_masking_token = False
        if args.custom_rotary_embedding:
            model.model.custom_rotary_embedding = True
        else:
            model.model.custom_rotary_embedding = False

    model.to(device)

    return model, processor, tokenizer


# def text_correct(result):
#     return result["caption_0_image_0"] > result["caption_1_image_0"] and result["caption_1_image_1"] > result["caption_0_image_1"]

# def image_correct(result):
#     return result["caption_0_image_0"] > result["caption_0_image_1"] and result["caption_1_image_1"] > result["caption_1_image_0"]

# def group_correct(result):
#     return image_correct(result) and text_correct(result)


def read_mask_arrays(mask_files):
    """
    Read all mask arrays from a specified directory.

    Args:
    - mask_files: List of .npy mask files

    Returns:
    - List of loaded mask arrays
    """
    # Load masks
    masks = [np.load(file) for file in mask_files]
    return masks


def find_mask_files(base_dir, image_id):
    """
    Given an image_id, find its corresponding mask files in the partition directories.

    Args:
        base_dir (str): The root directory containing partition folders.
        image_id (str): The image identifier in the format '00000/000000073.jpg'.

    Returns:
        list: A list of full paths to mask_n.npy files.
    """

    partitions = [
        os.path.join(base_dir, p)
        for p in os.listdir(base_dir)
        if p.startswith("partition_")
    ]

    for partition_dir in partitions:
        image_path = os.path.join(partition_dir, image_id)
        if os.path.exists(image_path) and os.path.isdir(image_path):
            # Retrieve all mask_n.npy files
            mask_files = sorted(glob.glob(os.path.join(image_path, "mask_*.npy")))
            return mask_files  # Return the list of mask file paths

    return []  # Return empty list if image_id is not found


def eval(args):

    # for file in args.question_file:
    #     # that directory
    # questions = pd.read_parquet(args.question_file, engine='pyarrow')
    device = "cuda:0"
    model, processor, tokenizer = load_generative_model(args, device)

    for filename in os.listdir(args.question_file):
        f = os.path.join(args.question_file, filename)
        # checking if it is a file
        answers_file = os.path.expanduser(args.answers_file)
        answers_file = (
            answers_file.split(".json")[0] + filename.split(".csv")[0] + ".json"
        )
        print("answers_file", answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")
        if os.path.isfile(f):
            questions = pd.read_csv(f)

            # print('args.question_file', args.question_file)
            for index, row in tqdm(questions.iterrows(), total=len(questions)):

                # idx = row['id']

                question = row["question"]
                question += (
                    "\nAnswer with the option's letter from the given choices directly."
                )
                image = args.image_folder + row["image"]

                if args.sam2:

                    image_id = row["image"].split("images/")[-1]

                    mask_files = find_mask_files(
                        base_dir="/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data_benchmarks/conme/arrays/",
                        image_id=image_id,
                    )
                    masks = read_mask_arrays(mask_files)
                    if len(masks) > 1:
                        masks = [
                            torch.from_numpy(mask_np).to(model.device)
                            for mask_np in masks
                        ]
                        masks = torch.stack(masks).unsqueeze(0)

                    else:
                        # print(f"Failed to read mask files for image {image_id}, and image_file {image_file}")
                        masks = torch.stack(
                            [torch.zeros(3, 3), torch.zeros(3, 3)]
                        ).unsqueeze(0)

                    # if masks.shape[1] < 20:
                    #     model.sam2_masking_token = True
                    # else:
                    #     model.sam2_masking_token = False
                    # print(masks.shape)
                    # else:
                    #     model.sam2_masking_token = False

                    # # print("image", image)
                    # image_pil = Image.open(image).convert("RGB")
                    # # print("image_pil", image_pil)
                    # image_array = np.array(image_pil)
                    # # print("image_array", image_array)
                    # # print("image_array.shape", image_array.shape)

                    # import sys

                    # sam2_pth = os.path.abspath(
                    #     os.path.join(os.path.split(__file__)[0], "../sam2/")
                    # )  #  /data/chatgpt/notebooks/mnulli/llava/conme/ -> /data/chatgpt/notebooks/mnulli/llava/sam2/
                    # sys.path.append(f"{sam2_pth}")

                    # from sam2.build_sam import build_sam2
                    # from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

                    # sam2_checkpoint = "/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data/checkpoints/sam2.1_hiera_large.pt"
                    # # sam2_checkpoint = "/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data/checkpoints/sam2.1_hiera_tiny.pt"
                    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

                    # sam2 = build_sam2(
                    #     model_cfg,
                    #     sam2_checkpoint,
                    #     device=model.device,
                    #     apply_postprocessing=False,
                    # )

                    # mask_generator = SAM2AutomaticMaskGenerator(
                    #     model=sam2,
                    #     points_per_batch=6,
                    #     pred_iou_thresh=0.9,
                    # )

                    # masks_dict = mask_generator.generate(image_array)
                    # masks = []
                    # for mask in masks_dict:
                    #     print(
                    #         "torch.tensor(mask['segmentation'], device=model.device)",
                    #         torch.tensor(mask["segmentation"], device=model.device),
                    #     )
                    #     masks.append(
                    #         torch.tensor(mask["segmentation"], device=model.device)
                    #     )
                    # print("masks", masks)

                    # if len(masks) > 1:
                    #     masks = (
                    #         torch.stack(masks, dim=0)
                    #         .unsqueeze(0)
                    #         .to(device=model.device)
                    #     )

                    #     if masks.shape[1] < 20:
                    #         model.sam2_masking_token = True
                    #     else:
                    #         model.sam2_masking_token = False

                    # else:
                    #     model.sam2_masking_token = False

                answer = row["answer"]

                # print('question', question)

                if "qwen" in args.model_path.lower():

                    image = Image.open(image).convert("RGB")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": question},
                            ],
                        }
                    ]

                    # Preparation for inference
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to("cuda")

                    # Inference: Generation of the output
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    outputs = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    outputs = outputs[0].strip(".")

                elif "internvl2" in args.model_path.lower():
                    # Load the corresponding image

                    pixel_values = (
                        load_image(image, max_num=12).to(torch.bfloat16).cuda()
                    )

                    question = "<image>\n" + question
                    # print('qs', qs)
                    with torch.inference_mode():
                        generation_config = dict(
                            max_new_tokens=1024,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                        )
                        # generation_config = dict(max_new_tokens=1024, do_sample=True if args.temperature > 0 else False,)

                        # print(batch["num_patches_list"].device)
                        outputs = model.chat(
                            tokenizer,
                            pixel_values=pixel_values,
                            question=question,
                            generation_config=generation_config,
                        )
                        outputs = outputs.strip()

                elif "molmo" in args.model_path.lower():

                    question = "<image>" + question
                    with torch.inference_mode():

                        image = [Image.open(image).convert("RGB")]

                        inputs = processor.process(images=image, text=question)

                        # move inputs to the correct device and make a batch of size 1
                        inputs = {
                            k: v.to(model.device).unsqueeze(0)
                            for k, v in inputs.items()
                        }

                        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
                        with torch.autocast(
                            device_type="cuda", enabled=True, dtype=torch.bfloat16
                        ):

                            output = model.generate_from_batch(
                                inputs,
                                GenerationConfig(
                                    max_new_tokens=200, stop_strings="<|endoftext|>"
                                ),
                                tokenizer=processor.tokenizer,
                            )

                            # only get generated tokens; decode them to text
                            generated_tokens = output[0, inputs["input_ids"].size(1) :]
                            outputs = processor.tokenizer.decode(
                                generated_tokens, skip_special_tokens=True
                            )

                elif "sa2va" in args.model_path.lower():

                    image = Image.open(image).convert("RGB")

                    question = "<image>" + question

                    input_dict = {
                        "image": image,
                        "text": question,
                        "past_text": "",
                        "mask_prompts": None,
                        "tokenizer": tokenizer,
                    }

                    return_dict = model.predict_forward(**input_dict)
                    outputs = return_dict["prediction"]

                elif "llava-hf" in args.model_path.lower():
                    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
                    # Each value in "content" has to be a list of dicts with types ("text", "image")

                    image = Image.open(image).convert("RGB")

                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {"type": "image"},
                            ],
                        },
                    ]
                    prompt = processor.apply_chat_template(
                        conversation, add_generation_prompt=True
                    )
                    print("prompt", prompt)

                    inputs = (
                        processor(images=image, text=prompt, return_tensors="pt")
                        .to(torch.float16)
                        .to(device)
                    )

                    if len(inputs["pixel_values"].shape) > 4:
                        inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)

                    allowed = set(inspect.signature(model.forward).parameters)
                    gen_inputs = {k: v for k, v in inputs.items() if k in allowed}

                    for key in list(gen_inputs.keys()):
                        if str(gen_inputs[key].device) != "cuda:0":
                            gen_inputs[key] = gen_inputs[key].to(device)

                    output = model.generate(
                        **gen_inputs, max_new_tokens=200, do_sample=False
                    )

                    outputs = processor.decode(
                        output[0], skip_special_tokens=True
                    ).split("assistant")[-1]
                    # print("outputs", outputs)
                else:
                    question = "<image>" + question
                    conv = conv_templates[args.conv_mode].copy()

                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)

                    if "lilium" in args.model_base.lower():
                        prompt = conv.sep + conv.get_prompt()
                    else:
                        prompt = conv.get_prompt()

                    # print("prompt", prompt)
                    # print("image", image)

                    image = Image.open(image).convert("RGB")
                    input_ids = (
                        tokenizer_image_token(
                            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                        )
                        .unsqueeze(0)
                        .cuda()
                    )

                    image_tensor = process_images([image], processor, model.config)[0]
                    if model.model.sam2_masking_token:
                        with torch.inference_mode():
                            attention_mask = torch.ones_like(input_ids)
                            output_ids = model.generate(
                                input_ids,
                                attention_mask=attention_mask,
                                # pad_token_id=tokenizer.pad_token_id,
                                images=image_tensor.unsqueeze(0).half().cuda(),
                                masks=masks,
                                image_sizes=[image.size],
                                do_sample=True if args.temperature > 0 else False,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                num_beams=args.num_beams,
                                # no_repeat_ngram_size=3,
                                max_new_tokens=1024,
                                use_cache=True,
                            )
                    else:
                        with torch.inference_mode():
                            attention_mask = torch.ones_like(input_ids)
                            output_ids = model.generate(
                                input_ids,
                                attention_mask=attention_mask,
                                # pad_token_id=tokenizer.pad_token_id,
                                images=image_tensor.unsqueeze(0).half().cuda(),
                                image_sizes=[image.size],
                                do_sample=True if args.temperature > 0 else False,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                num_beams=args.num_beams,
                                # no_repeat_ngram_size=3,
                                max_new_tokens=1024,
                                use_cache=True,
                            )

                    outputs = tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )[0].strip()
                    # print("outputs", outputs)
                    # print("len(masks)", len(masks))
                    # print("masks[0].shape", masks[0].shape)

                ans_file.write(
                    json.dumps(
                        {
                            "partition": filename,
                            "question_id": index,
                            "prompt": question,
                            "output": outputs,
                            "answer": answer,
                            "metadata": {},
                        }
                    )
                    + "\n"
                )
                ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--sam2", type=str, default=False)
    parser.add_argument("--custom_rotary_embedding", type=str, default=False)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--vllm", type=str, default="False")
    parser.add_argument("--gpt4o", type=str, default=False)
    parser.add_argument("--llama90b-endpoint", type=str, default=False)
    parser.add_argument("--phi3", type=str, default=False)
    parser.add_argument("--molmo", type=str, default=False)
    parser.add_argument("--qwen2", type=str, default=False)
    parser.add_argument("--pixtral", type=str, default=False)
    parser.add_argument("--aria", type=str, default=False)
    parser.add_argument("--llama3-2", type=str, default=False)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--context", type=str, default=False)
    parser.add_argument("--textonly", type=str, default=False)
    parser.add_argument("--retrieval", type=str, default=False)
    parser.add_argument("--siglip", type=str, default=False)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    args = parser.parse_args()

    eval(args)

    count = 0
    accs = {}
    for filename in os.listdir(args.question_file):
        answers_file = os.path.expanduser(args.answers_file)
        answers_file = (
            answers_file.split(".json")[0] + filename.split(".csv")[0] + ".json"
        )

        print("answers_file", answers_file)
        data = pd.read_json(answers_file, lines=True)

        # data = json.load(open(answers_file))
        tot = 0
        correct = 0
        for index, row in data.iterrows():
            # print(row['response'], row['answer'])

            response = row["output"]
            if row["answer"].lower() in response.lower():
                correct += 1
            tot += 1

            # print(response, row['answer'])
        # print(f'{filename} acc', correct/tot)
        # print('')
        accs[f"{filename}"] = {"accuracy": correct / tot, "total length": tot}
        print(f'accs["{filename}"]', accs[f"{filename}"])

    final_acc = 0

    datasets_length = 0
    for k, v in accs.items():
        datasets_length += v["total length"]

    for k, v in accs.items():
        print(f"ConME accuracy for {k} ", v["accuracy"])
        final_acc += v["accuracy"] * (v["total length"] / datasets_length)

    print(f"ConME accuracy for {args.model_path} ", final_acc)

    # print(data)

    # cur_df = df.copy()
    # # cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])

    # cur_df.insert(5, 'prediction', None)
    # for line_number, pred in enumerate(open(os.path.join(args.result_dir, f"{args.experiment}.jsonl"))):

    #     try:
    #         # print(f'Processing line {line_number}')

    #         pred = json.loads(pred)
    #         pred['output'] = pred['output'].strip('.')
    #         # print('pred', pred)
    #         print('pred', pred['output'], 'ans', pred['answer'])
    #         # print('pred', )
    #         # print(pred)
    #         cur_df.loc[df['index'] == pred['question_id'], 'prediction'] = pred['output']

    #     except json.JSONDecodeError as e:
    #         count +=1
    #         print(f"JSON decode error on line {line_number}: {e}")
    #         continue
    #     except KeyError as e:
    #         count +=1
    #         print(f"Missing key error on line {line_number}: {e}")
    #         continue
    #     except Exception as e:
    #         count +=1
    #         print(f"Unexpected error on line {line_number}: {type(e).__name__}: {e}")
    #         continue

    # print(string_matching_acc(cur_df, count))

    # cur_df.to_csv(os.path.join(args.upload_dir, f"{args.experiment}.csv"), index=False)
    # print(f'File saved at {os.path.join(args.upload_dir, f"{args.experiment}.csv")}')

    # cur_df.to_excel(os.path.join(args.upload_dir, f"{args.experiment}.xlsx"), index=False, engine='openpyxl')

    ## calculate accuracy based on answer file

    # from utils import t2i_score

    # t2i_score(
    ## calculate accuracy
