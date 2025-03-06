import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
import base64
import math

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoProcessor, GenerationConfig, Qwen2VLForConditionalGeneration, MllamaForConditionalGeneration, LlavaForConditionalGeneration
from qwen_vl_utils import process_vision_info
from vllm import LLM
from vllm.sampling_params import SamplingParams

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
from llava.model import *

from PIL import Image



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False



def preprocess_fewshot_data(d):
    title = d['title']
    category = d['category_context']

    kvpairs = '\n'.join([f'{k}: {v}' for k, v in d['aspects'].items()])
    return f"""For an e-commerce website, under the category \"{category}\", the listing with the title \"{title}\" has the following aspect key-value pairs:
{kvpairs}"""


def preprocess_data_gen(d):
    title = d['title']
    category = d['category_context']

    # return f"""For an e-commerce website, under the category \"{category}\", the listing with the title \"{title}\" has the following aspect key-value pairs:\n"""
    if args.textonly:
        return f"""The following is a listing from an e-commerce website. It has this title \"{title}\, and falls under the category \"{category}\":\n"""
    else:
        return f"""The following is a listing from an e-commerce website. It has this title \"{title}\, and falls under the category \"{category}\" and image """

    
def preprocess_retrieval(d):
    context_examples = d['context examples']
    
    return f"""This listing has similar items whose corresponding aspect values are \"{context_examples}\". Given this answer the following: \n"""
    
    

def eval_model(args):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    path_file_check = 'playground/data/eval/ebench-sm/answers/ebench-sm-title-cat_uk-gpt4o/gpt4o.jsonl'
    file_check = open(path_file_check, 'r')
            
    model_name = get_model_name_from_path(model_path)

    
    if args.vllm == True:
        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=1024)
        
        from llava.model.language_model.llava_llama import LlavaConfig
        lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path) 
    
        kwargs = {}
        device='cuda'
        load_8bit=False 
        load_4bit=False
        use_flash_attn=False
        
        if device != "cuda":
            kwargs['device_map'] = {"": device}

        if load_8bit:
            kwargs['load_in_8bit'] = True
        elif load_4bit:
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        else:
            kwargs['torch_dtype'] = torch.float16

        if use_flash_attn:
            kwargs['attn_implementation'] = 'flash_attention_2'
        
        
        model = LlavaLlamaForCausalLM.from_pretrained(args.model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        llm = LLM(model=model, 
                 tokenizer=args.model_base)
        
        model = model.to(device='cuda:0', dtype=torch.float16)
        
        batches = batching(args)
        
    elif args.llava_lilium:
        print('Evaluating on internal model checkpoints')
        model_id = "/mnt/nushare2/data/vorshulevich/vlm/hf_models/lilium-2-vl-7b-chat"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
        ).to(0)

        processor = AutoProcessor.from_pretrained(model_id)

        
    elif args.phi3 or args.aria:
        assert 'phi-3.5' in args.model_base.lower() or 'aria' in args.model_base.lower()
        
        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        model = AutoModelForCausalLM.from_pretrained(
          args.model_path, 
          device_map="cuda", 
          trust_remote_code=True, 
          torch_dtype="auto", 
          _attn_implementation='flash_attention_2'    
        )

        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        if args.aria:
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        else:
            processor = AutoProcessor.from_pretrained(args.model_path, 
              trust_remote_code=True, 
              num_crops=4
            ) 

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
        model.to(dtype=torch.bfloat16)
        
    elif args.qwen2:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        # default processer
        processor = AutoProcessor.from_pretrained(args.model_path)

        
    elif args.pixtral:
        
        assert 'pixtral' in args.model_path.lower()
        
        max_img_per_msg = 5

        sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)

        # Lower max_num_seqs or max_model_len on low-VRAM GPUs.
        llm = LLM(model=args.model_path, tokenizer_mode="mistral", limit_mm_per_prompt={"image": max_img_per_msg})

        
    elif args.gpt4o:
        print('yes, this is gpt4o')
        
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.prompts import PromptTemplate
        from pychomsky.chchat import EbayLLMChatWrapper

        
        model = EbayLLMChatWrapper(
            model_name='azure-chat-completions-gpt-4o-mini-2024-07-18',
            # model_name='azure-chat-completions-gpt-4o-2024-05-13-sandbox',
            temperature=0,
            max_tokens=128
        )
        
    elif args.llama90b_endpoint:
        url = "https://genai-h100.inference.vip.ebay.com/inference/v1/domain/genai-h100/inferenceservice/chao-llm-h100/predict"
        token = 'v^1.1#i^1#f^0#r^0#p^1#I^3#t^H4sIAAAAAAAAAO1Yb2wURRTvtddiAxWJBLAp5NxqMJDbm7m9vbsuvcOjLXC2cKVXammiMrs32257t7vszHl3hWipBL8YITEUTYtUIiKY4J8YiYYILYmKRCUhgOIHEaMhmphIiGhUcO9ayrViC7RCP9gvzc68efN+v/d+M+8GdBQULti8fPOlIsuU3N4O0JFrscCpoLAgf+HdebnF+Tkgy8DS2/FAh7Uz73w5QbGoLtRhomsqwbZkLKoSITPoY+KGKmiIKERQUQwTgUpCOLCiRnCyQNANjWqSFmVswUofI0cikhPxwMu7JNkbcZqj6lWf9ZqPgXyZzEPodAIOy24PMOcJieOgSihSqY9xAqfLDqEd8PWQE1y8wPGsiytrYmwN2CCKppomLGD8mXCFzFojK9bRQ0WEYIOaThh/MLA0HAoEK6tW1pc7snz5B3kIU0TjZPhXhRbBtgYUjePRtyEZayEclyRMCOPwD+ww3KkQuBrMLYSfoVqUEXQ73R4IRCBJwDUhVC7VjBiio8eRHlEidjljKmCVKjQ1FqMmG2Irlujg10rTRbDSlv63Ko6iiqxgw8dULQmsCdTWMv707lhEKXsMGW2Y6lEkYbtk1lA8hg0lIrhlF5TdQLRLLk62uzwcsiPJK9sxxDwE0IuhSxwMYmCnwRSMiKJCUyNKmlBiW6nRJdhEhEfyxmXxZhqF1JARkGk62iE7Tza/XFM64QMZjtMWNZ1zHDNJsmU+x87O0GpKDUWMUzzkYeREhj4fg3RdiTAjJzN1OlhaSeJjWijVBYcjkdZ6IsEmOFYzmh1OAKCjcUVNWGrBMcQM2SvXFvybsV3JQJGwuYooAk3pZixJs47NANRmxu8CbrfbM8j78LD8I0f/MZCF2TFcLROlHuSRvADJvCxiN4/cZROhHv9gATvGql+Ol52cV8b2iLvMrN8yWbaLfMRthzLGAGNRlMq8/4vo5mQQxpKB6Q3r4LZo4JGaR8WGdnd1NIQc6jqltS6WqOHWeVPtTUvdze2tdU0t1UmvtCoggWbfjSrluuAroorJTL25fzYBaa3feRKWa4TiyLjghSVNx7VaVJFSkyvBkNBaZNDUuNAFdD144wf4bcE1gWfHrXFyc5faJLzQrouKpOv4v0E12r0+FrK0D2I6QbrCprPOSlrMoSGzX3FkInY8rMdFU3zjwq6YrfCkyqcJcgCtEhnoYdkMZJY8KbEGJlrcMNt3NpRu2+q1NqyaFx01tGgUGw1w3HqPxeIUiVE82YQ/AUWuoEl2C0MPB7zQ7Cr4ceGSMnfsE5PgWLJufPeOndY30ag7hj8p+HMyf7DT0g86LR/mWiygHDwIS8H9BXmrrXnTiolCMasgmSVKs2r+UjYw24ZTOlKM3HtzLuzatryiuCrUtWB9fep498c507JeNHofA3OG3jQK8+DUrAcOUHJtJh9On13kdEEIeMi5eI5vAqXXZq1wlnXmm0Vdia75BxtLuq1r9x7r/eP7Zf07QdGQkcWSn2PttORse6+vYd+51c/3LeIe339+9+IZ7//+4uGjx9/oP/HlvvZkS/LUN7tSpz8qUWZ1f3Dlt03P8Lylp5G9Mvep3LNBx6k9Fzx9U/LXbWiUZ9O9h5tigYufH9rwsgJ+gj09Bwrak+z5fdPOPff6XV+EG+VQ7Lu3g7u7jnzWXGmJLry8fu1XxckfXnkof2Ot1ut69ch++Ze6LQdKgn0/lz+t7y5ZVDXzYseaw2dO+tRLpe8UfbJnzunp1YdOdr0wL29HFw7d07P4rYtHe6pXnwx/PeNb65Yf1y7bP3vKzmfF0qZ5zn6XZr9k29qmneme89om769r5v7VeuJTGmnYPj+VKHzpYO59Z49tJ5f/7NjRn88P5PJvi9pUWGsSAAA='

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            'Authorization': f'Bearer {token}'
        }
    
    elif args.llama3_2:
        
        assert 'llama-3_2' in args.model_path.lower()
        
        model = MllamaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        
        model = model.to(device='cuda:0', dtype=torch.float16)
        
        
    elif args.textonly:
        # model_path = os.path.expanduser(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="balanced", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        model_name = get_model_name_from_path(args.model_path)
        model = model.to(device='cuda:0', dtype=torch.float16)
        
    else:
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)

        print('model_path', model_path)
        print('args.model_base', args.model_base)
        print('model_name', model_name)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device='cuda:0', device_map='cuda:0')

        print('model', model)

        
    
        model = model.to(device='cuda:0', dtype=torch.float16)
    
    
    # print('questions', questions)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    c = 0
    for index, row in tqdm(questions.iterrows(), total=len(questions)):

        # print('row', row)
        # print('index', index)
        
        

        answer = row['answer']

        idx = row['index']
        q = row['question']
        hint = row['hint']
        
        
            
        if args.gpt4o or args.llama90b_endpoint:
            existing_question_ids = {}
            with open(path_file_check, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    existing_question_ids[data['question_id']] = {}
                    existing_question_ids[data['question_id']]["prompt"] = data['prompt']
                    existing_question_ids[data['question_id']]["output"] = data['output']
                    existing_question_ids[data['question_id']]["answer"] = data['answer']
                    existing_question_ids[data['question_id']]["answer_id"] = data['answer_id']
                    existing_question_ids[data['question_id']]["model_id"] = data['model_id']
                    
            if idx in existing_question_ids.keys():
                
                print(f'Skipping question_id {idx} as it is already there')
                
                ans_file.write(json.dumps({"question_id": idx,
                                "prompt": existing_question_ids[data['question_id']]["prompt"],
                                "output": existing_question_ids[data['question_id']]["output"],
                                "answer": existing_question_ids[data['question_id']]["answer"],
                                "answer_id": existing_question_ids[data['question_id']]["answer_id"],
                                "model_id": existing_question_ids[data['question_id']]["model_id"],
                                "metadata": {}}) + "\n")
                continue
            
            found = False
            with open(answers_file, "r") as file:
                for line in file:
                    # Parse the line as JSON
                    data = json.loads(line)

                    # Check if the question_id matches
                    if data.get('question_id') == idx:
                        found = True
                        break
                        
            if found == True:
                print('Skipping this question_id as it is already there')
                continue
                
        ###EBAY SPECIFIC
        if args.textonly == False or not args.textonly and args.gpt4o==False:
            image = Image.open(row['image'])
        
        if args.gpt4o or args.llama90b_endpoint or args.pixtral:
            # print(row['image'])
            # print('this should not be printed')
            image = encode_image(row['image'])
            # print('image.size', image.size)
            # print('DEFAULT_IMAGE_TOKEN', DEFAULT_IMAGE_TOKEN)
            
        elif args.phi3 or args.molmo:
            # print('this should not be printed')
            placeholder = ''
            for i in range(1,2):
                ##single image support
                image = [Image.open(row['image'])]
                placeholder += f"<|image_{i}|>\n"
        else:
            
            image = Image.open(row['image'])
        
            
            ##for multi image simply open them and put them as different items in the list and increase the for loop.
            
        
        if args.context and args.shots == 0:
            qs = preprocess_data_gen(row)
            if args.llama90b_endpoint or args.phi3 or args.molmo or args.qwen2 or args.pixtral or args.aria or args.llama3_2:
                qs = qs + ':' + '\n ' + q
            else:
                if args.retrieval:
                    retrieval = preprocess_retrieval(row)
                    qs = qs + DEFAULT_IMAGE_TOKEN + ' ' + retrieval + '\n' + q
                else:
                    qs = qs + DEFAULT_IMAGE_TOKEN + ':' + '\n ' + q
                
        elif args.context and args.shots > 0:
            qs = preprocess_fewshot_data(row)
            qs = qs + preprocess_data_gen(row)
            if args.llama90b_endpoint or args.phi3 or args.molmo or args.qwen2 or args.pixtral or args.aria or args.llama3_2:
                qs = qs + ':' + '\n ' + q
            else:
                if args.retrieval:
                    retrieval = preprocess_retrieval(row)
                    qs = qs + DEFAULT_IMAGE_TOKEN + ' ' + retrieval + '\n ' + q
                else:
                    qs = qs + DEFAULT_IMAGE_TOKEN + ':' + '\n ' + q

        else:
            if args.llama90b_endpoint or args.phi3 or args.molmo or args.qwen2 or args.pixtral or args.aria or args.llama3_2:
                qs = q
            else:  
                qs = DEFAULT_IMAGE_TOKEN + q 
        
        # if args.retrieval:
        #     retrieval = preprocess_retrieval(row)
        #     qs = retrieval + qs
                    
        if args.textonly == False or not args.textonly:
            
            if not is_none(hint):
                qs = hint + '\n' + qs
                
            if not args.gpt4o and not args.llama90b_endpoint and not args.pixtral or args.llama3_2:
                model.config.mm_use_im_start_end = False

            if args.single_pred_prompt:
                qs = qs + "\n" + "Answer by generating the aspect characteristics. Limit your self to a couple of words at most."

            
            # print('args.conv_mode', args.conv_mode)
            conv = conv_templates[args.conv_mode].copy()
            # print('conv', conv)
    
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            
            

            if "lilium" in args.model_base.lower():
                prompt = conv.sep + conv.get_prompt()
            # elif 'e-llama' in args.model_base.lower():
            #     prompt = conv.sep + conv.get_prompt()
            else:
                prompt = conv.get_prompt()
            
                
            # print('prompt', prompt)

            # print('model.config.mm_use_im_start_end', model.config.mm_use_im_start_end)
            # if "e-llama" in args.model_base.lower():
            #     model.config.mm_use_im_start_end = True
                
            # if model.config.mm_use_im_start_end:
            #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            # else:
            #     # print('Image token is not there')
            #     qs = qs


        
        elif args.textonly:
            
            if not is_none(hint):
                qs = hint + '\n' + qs
            qs = qs + "\n" + "Answer by generating the aspect characteristics. Limit your self to a couple of words at most."


        if args.textonly:
                chat = [
                  {"role": "user", "content": qs}
                ]
                # print('chat', chat)
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                # print('prompt',prompt)
                input_ids = tokenizer(
                        prompt,
                        return_tensors="pt",
                        return_attention_mask=True,
                    ).to('cuda')
                    
                    
                with torch.inference_mode():
            
                    output_ids = model.generate(
                        **input_ids,                 
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=1024, 
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        use_cache=True)
                
                output_ids = output_ids.to('cuda')
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        elif args.llava_lilium:
            # print('prompt', prompt)
            # print('qs', qs)
            
            inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
            
            output = model.generate(
                **inputs, 
                max_new_tokens=1000,
                temperature=0,
            )
            
            outputs = processor.decode(output[0]).split('[/INST]')[1].strip('</s>')
            
            # print('outputs', outputs)
        
        
        elif args.molmo:
            

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
            
            outputs = outputs[0]
            
            # print('outputs', outputs)
        elif args.pixtral:
            
            # print('qs', qs)
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": qs}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]
                },
            ]

            outputs = llm.chat(messages, sampling_params=sampling_params)

            outputs = outputs[0].outputs[0].text
        
        elif args.aria:
            
            # print('qs', qs)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": None, "type": "image"},
                        {"text": qs, "type": "text"},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=text, images=image, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    stop_strings=["<|im_end|>"],
                    tokenizer=processor.tokenizer,
                    do_sample=True,
                    temperature=0.9,
                )
                output_ids = output[0][inputs["input_ids"].shape[1]:]
                outputs = processor.decode(output_ids, skip_special_tokens=True)

            outputs = outputs.strip('<|im_end|>')
            # print('outs', outputs)

        elif args.gpt4o:
            skip = False
            # print('qs', qs)
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
            
        elif args.llama90b_endpoint:
            
            import requests
            
            template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 05 Nov 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            qs = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language." + qs
            # print('qs', qs)
            body = {
                "input": {
                    "text_input": [template.format(user_message=qs)],
                    "multi_modal_data": [f'{{\n "image": ["{image}"]\n}}'],
                    "sampling_parameters": ["{\"max_tokens\": 100, \"temperature\": 0.0}"]
                },
                "appName": "chao-test-llama-3-2-90b-vision-instruct-0-djiv"
            }

            # print('template.format(user_message=qs)', template.format(user_message=qs))
            response = requests.post(
                url, 
                headers=headers, 
                data=json.dumps(body), 
                verify=False,
            )
            outputs = response.json()['output']['text_output'][0].split('Limit your self to a couple of words at most.<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[1]
            # print('outputs', outputs)
            
        elif args.llama3_2:
            
            # print('qs', qs)
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
            ).to('cuda:0')

            # print(inputs.device)
            output = model.generate(**inputs, max_new_tokens=30)
            outputs = processor.decode(output[0]).split('assistant<|end_header_id|>')[1].strip('<|eot_id|>')
            # print('outputs', outputs)
        else:
            
            # print('prompt', prompt)
            # prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to('cuda:0')
            # print('IMAGE_TOKEN_INDEX', IMAGE_TOKEN_INDEX)
            
            # print('input_ids', input_ids)
            # print('tokenizer.decode(input_ids)', tokenizer.decode(input_ids[0]))
            # if args.siglip:
            #     image_tensor = process_images(image, image_processor, model.config)[0]
            # else:
            image_tensor = process_images([image], image_processor, model.config)[0]

            vision_tower = model.get_vision_tower()

            weights = [
                vision_tower.vision_tower.vision_model.encoder.layers[0].mlp.fc1.weight[0],
                vision_tower.vision_tower.vision_model.encoder.layers[1].self_attn.k_proj.weight[0],
                vision_tower.vision_tower.vision_model.encoder.layers[2].self_attn.k_proj.weight[0]
                ]

            with open('/data/chatgpt/notebooks/mnulli/weights.txt', 'w') as f:
                for weight in weights:
                    f.write(str(weight) + '\n')
                    
                
        
            
            input_ids = input_ids.to('cuda:0')
            image_tensor = image_tensor.to('cuda:0')
            
            # print('input ids ', input_ids)
            # print("image_tesor", image_tensor)
            # print('images tensor unsqueeze ', image_tensor.unsqueeze(0).half())
            
            with torch.inference_mode():
                if args.conv_mode == 'llama3':
                    # print('input_ids',input_ids)
                    # attention_mask = torch.tensor([[1] * len(input_ids[0])], dtype=torch.long)
                    # # print('attention_mask',attention_mask)
                    # attention_mask = attention_mask.to('cuda:0')
                    # print('attention_mask',attention_mask)
                    
                    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
                    
                    output_ids = model.generate(
                        input_ids,
                        # attention_mask = attention_mask,
                        pad_token_id=tokenizer.pad_token_id,
                        images=image_tensor.unsqueeze(0).half().to('cuda:0'),
                        image_sizes=[image.size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True)
                else:
                    # args.temperature = 0
                    max_new_tokens = 1024
                    
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().to('cuda:0'),
                        image_sizes=[image.size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=max_new_tokens,
                        use_cache=True).to('cuda:0')
                
            # print('output_ids', output_ids)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # print('prompt', prompt)
            # print('outputs', outputs)
            
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                "prompt": q,
                                "output": outputs,
                                "answer": answer,
                                "answer_id": ans_id,
                                "model_id": model_name,
                                "metadata": {}}) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
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

    eval_model(args)
