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
import ast

from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

from open_clip import (
    create_model_from_pretrained,
    get_tokenizer,
)  # works on open-clip-torch>=2.23.0, timm>=0.9.8
from qwen_vl_utils import process_vision_info
from vllm import LLM
from vllm.sampling_params import SamplingParams


from internvl.utils import *

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
)  # -> /data/chatgpt/notebooks/mnulli/llava/llava/eval/winoground/ -> /data/chatgpt/notebooks/mnulli/llava/
sys.path.append(f"{llava_pth}")

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

    elif "molmo" in args.model_path.lower():

        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cuda:0",
        )
        tokenizer = processor.tokenizer

        # load the model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cuda:0",
        )
        model.to(dtype=torch.bfloat16)

    elif "sa2va" in args.model_path.lower():

        model = (
            AutoModel.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, use_fast=False
        )

        processor = None

    elif "internvl2" in args.model_path.lower():
        # device_map = split_model('InternVL2-8B')
        model = (
            AutoModel.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
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

    model.to(device)

    return model, processor, tokenizer


def eval(args):

    questions = pd.read_csv(args.question_file)
    device = "cuda:0"
    model, processor, tokenizer = load_generative_model(args, device)

    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    total = 0
    image_caption_match_results = {}

    for index, row in tqdm(questions.iterrows(), total=len(questions)):

        caption_0 = row["text_0"]
        caption_1 = row["text_1"]

        captions = {"caption_0": caption_0, "caption_1": caption_1}

        image_0 = ast.literal_eval(row["image_0"])["bytes"]
        image_1 = ast.literal_eval(row["image_1"])["bytes"]

        idx = row["id"]

        text_qs_0 = row["text_question_0"]
        text_qs_1 = row["text_question_1"]

        image_text_t = {
            "image_0": [image_0, text_qs_0],
            "image_1": [image_1, text_qs_1],
        }

        image_qs_0 = row["image_question_0"]
        image_qs_1 = row["image_question_1"]

        image_text_i = {
            "caption_0": [image_0, image_1, image_qs_0],
            "caption_1": [image_0, image_1, image_qs_1],
        }

        text_correct = False
        image_correct = False

        # for text results
        match_found = False
        for k, (image, text) in image_text_t.items():

            if "qwen" in args.model_path.lower():

                image = Image.open(io.BytesIO(image)).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text_qs},
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

            elif "molmo" in args.model_path.lower():

                image = [Image.open(io.BytesIO(image)).convert("RGB")]

                text_qs = "<image>\n" + text_qs

                inputs = processor.process(images=image, text=text_qs)

                # move inputs to the correct device and make a batch of size 1
                inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

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

                image = Image.open(io.BytesIO(image)).convert("RGB")

                qs = "<image>\n" + text_qs

                input_dict = {
                    "image": image,
                    "text": qs,
                    "past_text": "",
                    "mask_prompts": None,
                    "tokenizer": tokenizer,
                }

                return_dict = model.predict_forward(**input_dict)
                outputs = return_dict["prediction"]

            elif "internvl2" in args.model_path.lower():
                pixel_values = load_image(image, max_num=12).to(torch.float16).cuda()

                qs = "<image>\n" + text_qs
                # print('qs', qs)
                with torch.inference_mode():
                    # generation_config = dict(
                    #         max_new_tokens=1024,
                    #         do_sample=True if args.temperature > 0 else False,
                    #         temperature=args.temperature,
                    #         top_p=args.top_p,
                    #         num_beams=args.num_beams,
                    #         use_cache=True,)
                    generation_config = dict(
                        max_new_tokens=1024,
                        do_sample=True if args.temperature > 0 else False,
                    )

                    # print(batch["num_patches_list"].device)
                    outputs = model.chat(
                        tokenizer,
                        pixel_values=pixel_values,
                        question=qs,
                        generation_config=generation_config,
                    )

                # print('outputs', outputs)
            else:
                conv = conv_templates[args.conv_mode].copy()

                conv.append_message(
                    conv.roles[0],
                    "<image>" + text.strip("<image_0>").strip("<image_1>"),
                )
                conv.append_message(conv.roles[1], None)

                if "lilium" in args.model_base.lower():
                    prompt = conv.sep + conv.get_prompt()
                else:
                    prompt = conv.get_prompt()

                print("prompt", prompt)
                image = Image.open(io.BytesIO(image)).convert("RGB")
                input_ids = (
                    tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .cuda()
                )

                image_tensor = process_images([image], processor, model.config)[0]

                with torch.inference_mode():
                    attention_mask = torch.ones_like(input_ids)
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
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

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                    0
                ]

            image_caption_match_results[str(idx) + f"_{k}"] = outputs
            # print (f"Image {k} entails which caption?: ", outputs.lower())

            # print("text_qs", text_qs)
            print("text outputs.lower()", outputs.lower())

            # match = re.search(' yes ', outputs.lower()) or re.search('<yes>', outputs.lower()) or re.search('yes', outputs.lower())

            if k == "image_0":
                # Matches either "(a)" or the letter "a"
                match = re.search(r"\(a\)|\ba\b", outputs.lower())
            elif k == "image_1":
                # Matches either "(b)" or the letter "b"
                match = re.search(r"\(b\)|\bb\b", outputs.lower())
                # match = re.search('\(b\)(?!.*\(a\).*\(b\))(?=.*\(a\)|$)', outputs.lower())

            # print('match found', match_found)
            if match and k == "image_0":
                match_found = True
                print("match found", match_found)

            # print('match', match)
            if match and k == "image_1" and match_found:
                print("match found", match_found)
                text_correct_count += 1
                match_found = False
                text_correct = True
                print("text_correct", text_correct)

        # for image results
        match_found = False
        for k, (image_0, image_1, caption) in image_text_i.items():

            if "qwen" in args.model_path.lower():

                if k == "caption_0":
                    image_0 = Image.open(io.BytesIO(image_0)).convert("RGB")
                    image_1 = Image.open(io.BytesIO(image_1)).convert("RGB")
                else:
                    image_0 = image_0
                    image_1 = image_1

                image_qs = (
                    "Which image better aligns with the description "
                    + f"{caption}"
                    + "? The first or the second image? Note, you must choose one of two options."
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_0},
                            {"type": "image", "image": image_1},
                            {"type": "text", "text": image_qs},
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

            elif "molmo" in args.model_path.lower():

                images = [
                    Image.open(io.BytesIO(image_0)).convert("RGB"),
                    Image.open(io.BytesIO(image_1)).convert("RGB"),
                ]

                image_qs = (
                    "<image>\n"
                    + "<image>\n"
                    + "Which image better aligns with the description "
                    + f"{caption}"
                    + "? The first or the second image? First, describe the image information relevant to the question. Then, provide your answer. Note, you must choose one of two options."
                )

                qs = image_qs

                inputs = processor.process(images=images, text=qs)

                # move inputs to the correct device and make a batch of size 1
                inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

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

                image_0 = Image.open(io.BytesIO(image_0)).convert("RGB")
                image_1 = Image.open(io.BytesIO(image_1)).convert("RGB")

                image_qs = (
                    "<image> Which image better aligns with the description "
                    + f"{caption}"
                    + "? The first or the second image? First, describe the image information relevant to the question. Then, provide your answer. Note, you must choose one of two options."
                )

                qs = image_qs

                input_dict = {
                    "video": [image_0, image_1],
                    "text": qs,
                    "past_text": "",
                    "mask_prompts": None,
                    "tokenizer": tokenizer,
                }

                return_dict = model.predict_forward(**input_dict)
                outputs = return_dict["prediction"]
                # outputs = ''

            elif "internvl2" in args.model_path.lower():

                pixel_values1 = load_image(image_0, max_num=12).to(torch.float16).cuda()
                pixel_values2 = load_image(image_1, max_num=12).to(torch.float16).cuda()

                pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
                num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

                image_qs = (
                    "First image: <image>\nSecond image: <image>\n Which image better aligns with the description "
                    + f"{caption}"
                    + "? The first or the second image? Note, you must choose one of two options."
                )
                qs = image_qs

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
                    outputs, _ = model.chat(
                        tokenizer=tokenizer,
                        pixel_values=pixel_values,
                        question=qs,
                        generation_config=generation_config,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=True,
                    )
                    # outputs = model.chat(
                    #     tokenizer,
                    #     pixel_values=pixel_values,
                    #     question=qs,
                    #     generation_config=generation_config)

                # print('outputs', outputs)
            else:

                conv = conv_templates[args.conv_mode].copy()

                caption = caption.strip("<image_0>")

                conv.append_message(
                    conv.roles[0],
                    "First Image: <image> "
                    + "Second Image: <image>"
                    + caption.strip(" <image_1>"),
                )
                conv.append_message(conv.roles[1], None)

                if "lilium" in args.model_base.lower():
                    prompt = conv.sep + conv.get_prompt()
                else:
                    prompt = conv.get_prompt()

                print("prompt", prompt)
                image_0_pil = Image.open(io.BytesIO(image_0)).convert("RGB")
                image_1_pil = Image.open(io.BytesIO(image_1)).convert("RGB")

                input_ids = (
                    tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .cuda()
                )

                image_tensor_0 = process_images([image_0_pil], processor, model.config)[
                    0
                ]
                image_tensor_1 = process_images([image_1_pil], processor, model.config)[
                    0
                ]

                with torch.inference_mode():
                    attention_mask = torch.ones_like(input_ids)
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        images=[image_tensor_0, image_tensor_1],
                        image_sizes=[image.size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True,
                    )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                    0
                ].strip()

            image_caption_match_results[str(idx) + f"_{k}"] = outputs
            # print (f"Image {k} entails which caption?: ", outputs.lower())

            # result_text[f'{k}'] = outputs.lower()
            # print('caption', caption)
            # print("image_qs", image_qs)
            print("image outputs.lower()", outputs.lower())

            # match = re.search(' yes ', outputs.lower()) or re.search('<yes>', outputs.lower()) or re.search('yes', outputs.lower())

            if k == "caption_0":
                match = False

                first_match = re.search("first", outputs.lower())
                second_match = re.search("second", outputs.lower())
                first_index = first_match.start() if first_match else -1
                second_index = second_match.start() if second_match else -1

                if first_index != -1 and (
                    second_index == -1 or first_index < second_index
                ):
                    match = True

                if match:
                    print("match_found", match_found)
                    match_found = True
                # match = re.search('first', outputs.lower())

            elif k == "caption_1":
                match = False

                first_match = re.search("first", outputs.lower())
                second_match = re.search("second", outputs.lower())
                first_index = first_match.start() if first_match else -1
                second_index = second_match.start() if second_match else -1

                if second_index != -1 and (
                    first_index == -1 or second_index < first_index
                ):
                    match = True

                if match_found and match:
                    print("match_found", match_found)
                    image_correct_count += 1
                    match_found = False
                    image_correct = True
                    print("image_correct", image_correct)

        if text_correct and image_correct:
            group_correct_count += 1

        total += 1

        if index % 10 == 0 or index == 2083:
            print("Results for filename", args.model_path)
            print(
                "Current Text Acc: {}/{} = {}%".format(
                    text_correct_count, total, text_correct_count / total * 100
                )
            )
            print(
                "Current Image Acc: {}/{} = {}%".format(
                    image_correct_count, total, image_correct_count / total * 100
                )
            )
            print(
                "Current Group Acc: {}/{} = {}%\n".format(
                    group_correct_count, total, group_correct_count / total * 100
                )
            )

    #     ans_file.write(json.dumps({"question_id": idx,
    #                             "prompt": q,
    #                             "output": outputs,
    #                             "answer": answer,
    #                             "answer_id": ans_id,
    #                             "model_id": model_name,
    #                             "metadata": {}}) + "\n")
    #     ans_file.flush()

    # ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
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
    parser.add_argument("--llava_lilium", type=str, default=False)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--context", type=str, default=False)
    parser.add_argument("--textonly", type=str, default=False)
    parser.add_argument("--retrieval", type=str, default=False)
    parser.add_argument("--siglip", type=str, default=False)

    args = parser.parse_args()

    eval(args)
