import argparse
import os
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import pandas as pd
import io
import base64


from datasets import load_dataset, concatenate_datasets
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from vllm import LLM
from vllm.sampling_params import SamplingParams

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig, 
    BitsAndBytesConfig, 
    AutoProcessor, 
    GenerationConfig, 
    Qwen2VLForConditionalGeneration,
    MllamaForConditionalGeneration
)

from qwen_vl_utils import process_vision_info

from PIL import Image
import math



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i,i+chunk_size-1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    # get kth chunk out of n chunks cut from lst length
    chunks = split_list(lst, n)
    return chunks[k]


def process(line, args, tokenizer, image_processor, model_config):
    qs = line["question"]
    # if line["image_2"] is not None:
    #     return None, None, None, None

    if line["question_type"] == "multiple-choice":
        qs += " Options:"
        options = re.findall(r"'(.*?)'", line["options"])
        for i in range(len(options)):
            option = options[i]
            qs += f"\n{chr(ord('A')+i)}. {option}"
        qs += f"\n{args.question_extension}"
    else:
        qs += f"\nAnswer the question using a single word or phrase."

    if line["image_1"] is not None:
        if args.gpt4o or args.llama90b or args.phi3 or args.molmo or args.qwen2 or args.pixtral or args.llama3_2:
            qs = qs
        elif model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else: 
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
           
    # remove <image \d> tags
    qs = re.sub(r'<image \d+>', '', qs).strip()

    
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    placeholder = None
    if line["image_1"] is None:
        image = None
        image_size = None
        image_tensor = None
        
    else:
        # image = line["image_1"].convert('RGBA')
        # image = line["image_1"].convert('RGB')
        if args.phi3 or args.molmo:
            placeholder = ''
            for i in range(1,2):
                ##single image support
                image = [Image.open(io.BytesIO(line["image_1"]["bytes"]))]
                image_size = [image[0].size]
                placeholder += f"<|image_{i}|>\n"
        elif args.pixtral:
            # print(base64.b64encode(line["image_1"]["bytes"]).decode('utf-8'))
            image = base64.b64encode(line["image_1"]["bytes"]).decode('utf-8')
            image_size = None
        else:
            image = Image.open(io.BytesIO(line["image_1"]["bytes"])).convert('RGB')
            image_size = [image.size]
        if args.gpt4o or args.llama90b or args.phi3 or args.molmo or args.qwen2 or args.pixtral or args.llama3_2:    
            image_tensor = image
            input_ids = None
        else:
            image_tensor = process_images([image], image_processor, model_config)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt, qs, placeholder


def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model
    # disable_torch_init()  # DO NOT ENABLE THIS: KILLS PERFORMANCE
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if args.phi3:
        assert 'phi-3.5' in args.model_base.lower()
        
        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        model = AutoModelForCausalLM.from_pretrained(
          args.model_path, 
          device_map="cuda:0", 
          trust_remote_code=True, 
          torch_dtype="auto", 
          _attn_implementation='flash_attention_2'    
        )

        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        processor = AutoProcessor.from_pretrained(args.model_path, 
          trust_remote_code=True, 
          num_crops=4
        ) 

        tokenizer = processor.tokenizer


    elif args.molmo:
        assert 'molmo' in args.model_base.lower()
        
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='cuda:0'
        )

        # load the model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='cuda:0'
        )
        tokenizer = processor.tokenizer
        
    elif args.qwen2:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='cuda:0'
        )

        # default processer
        processor = AutoProcessor.from_pretrained(args.model_path)
        tokenizer = processor.tokenizer

        
    elif args.pixtral:
        
        assert 'pixtral' in args.model_path.lower()
        
        max_img_per_msg = 5

        sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)

        # Lower max_num_seqs or max_model_len on low-VRAM GPUs.
        llm = LLM(model=args.model_path, tokenizer_mode="mistral", limit_mm_per_prompt={"image": max_img_per_msg})

        tokenizer = llm.get_tokenizer
        processor = None
        model = None

    elif args.gpt4o:
        
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.prompts import PromptTemplate
        from pychomsky.chchat import EbayLLMChatWrapper

        
        model = EbayLLMChatWrapper(
            model_name='azure-chat-completions-gpt-4o-mini-2024-07-18',
            temperature=0.5,
            max_tokens=128
        )
    elif args.llama3_2:
        
        assert 'llama-3_2' in args.model_path.lower()
        
        model = MllamaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        tokenizer = processor.tokenizer
        
        model = model.to(device='cuda:0', dtype=torch.float16)
        

    else:

        tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,device_map='cuda:0', device='cuda:0')
    
    ## Load from hf
    # validation_dataset = load_dataset("lmms-lab/MMMU", split="validation")
    # dev_dataset = load_dataset("lmms-lab/MMMU", split="dev")
    ## questions = concatenate_datasets([validation_dataset, dev_dataset])
    # questions = concatenate_datasets([validation_dataset])

    ## Load from local
    questions = pd.read_parquet('/data/chatgpt/notebooks/mnulli/llava/playground/data/eval/mmmu/datasets--lmms-lab--MMMU/snapshots/364f2e2eb107b36e07ff4c5a15f5947a759cef47/data/validation-00000-of-00001.parquet', engine='pyarrow')

    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")

    basename = os.path.basename(answers_file)
    basename = os.path.splitext(basename)[0]
    answers_dir = os.path.dirname(answers_file)
    chunk_fname = f"{basename}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(answers_dir, chunk_fname)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

    ans_file = open(chunk_file, "w")

    idx = -1
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    print(valid_chunk)
    maxloop = questions.index.max()
    for _, line in tqdm(questions.iterrows(), total=len(questions)):
        idx = idx+1

        if idx<valid_chunk[0] or idx>valid_chunk[1]:
            continue

        # print('line', line)
        if args.pixtral:
            input_ids, image_tensor, image_sizes, prompt, qs, placeholder = process(line, args, tokenizer, processor, None)
        else:
            input_ids, image_tensor, image_sizes, prompt, qs, placeholder = process(line, args, tokenizer, processor, model.config)
        gt_answer = line["answer"]
        category = line["id"].split('_')[1]
            
        if args.molmo:
            
            image = image_tensor
            inputs = processor.process(
                    images=image,
                    text=qs
                )

            # move inputs to the correct device and make a batch of size 1
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

            # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):

                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )

                
                # only get generated tokens; decode them to text
                generated_tokens = output[0,inputs['input_ids'].size(1):]
                outputs = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        elif args.phi3:

            image = image_tensor
            messages = [
                {"role": "user", "content": placeholder+qs},
            ]
            # print('messages', messages)
            
            prompt = processor.tokenizer.apply_chat_template(
              messages, 
              tokenize=False, 
              add_generation_prompt=True
            )

            inputs = processor(prompt, image, return_tensors="pt").to("cuda:0") 

            generation_args = { 
                "max_new_tokens": 1000, 
                "temperature": 0.0, 
                "do_sample": False, 
            } 

            generate_ids = model.generate(**inputs, 
              eos_token_id=processor.tokenizer.eos_token_id, 
              **generation_args
            )

            # remove input tokens 
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            outputs = processor.batch_decode(generate_ids, 
              skip_special_tokens=True, 
              clean_up_tokenization_spaces=False)[0] 
        
        elif args.qwen2:


            image = image_tensor
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": qs},
                    ],
                }
            ]
            
            # print('messages', messages)

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
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            outputs = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            outputs = outputs[0].strip('.')
            
            # print('outputs', outputs)
        elif args.pixtral:
            
            # print('qs', qs)
            image = image_tensor
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": qs}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]
                },
            ]
            
            outputs = llm.chat(messages, sampling_params=sampling_params)

            outputs = outputs[0].outputs[0].text
        
        elif args.gpt4o:
            skip = False
            # print('qs', qs)
            image = image_tensor
            message = HumanMessage(
                content=[
                    {"type": "text", "text": qs},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                ],
            )
            try:
                # print('message', message)
                output = model.invoke([message])
                outputs = output.content
            except:
                print(f'Index {idx} was probably flagged as inappropriate')
                c += 1
                print(f'# of inappropriate indixes so far: {c}')
                skip = True
            
            if skip == True:
                continue
            # print('output', output)
        
        elif args.llama3_2:
            # print('Evaluating Llama 3.2-V...')
            print('qs', qs)
            image = image_tensor
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": qs}
                ]}
            ]

            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            with torch.inference_mode():
                output = model.generate(**inputs, 
                    do_sample=False,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)
                outputs = processor.decode(output[0]).split('assistant<|end_header_id|>')[1].strip('<|eot_id|>')
                print('outputs', outputs)
        else:
        
            # print('line', line)
            input_ids, image_tensor, image_sizes, prompt, qs, placeholder = process(line, args, tokenizer, processor, model.config)
            
            if input_ids is None:
                continue
            gt_answer = line["answer"]
            category = line["id"].split('_')[1]
            input_ids = input_ids.to(device='cuda:0', non_blocking=True)
            with torch.inference_mode():
                if type(image_tensor) == list:
                    image_tensor = image_tensor[0].to('cuda:0')
                else:
                    image_tensor = image_tensor.to('cuda:0')
                model.to('cuda:0')
                tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    # top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # print('prompt', prompt)
        # print('outputs', outputs)
        ans_file.write(json.dumps({
            "model_id":model_name,
            "question_id": idx,
            "prompt": prompt,
            "answer": outputs,
            "gt_answer": gt_answer,
            "category": category,
            "type": line["question_type"]
        }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="Answer with the option's letter from the given choices directly.")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gpt4o", type=str, default=False)
    parser.add_argument("--llama90b", type=str, default=False)
    parser.add_argument("--phi3", type=str, default=False)
    parser.add_argument("--molmo", type=str, default=False)
    parser.add_argument("--qwen2", type=str, default=False)
    parser.add_argument("--pixtral", type=str, default=False)
    parser.add_argument("--llama3-2", type=str, default=False)
    args = parser.parse_args()

    eval_model(args)

