import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math

from transformers import  AutoModelForCausalLM, AutoTokenizer


all_options = ['A', 'B', 'C', 'D']


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

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


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
        return f"""The image depicts a listing from an e-commerce website. It has this title \"{title}\, and falls under the category \"{category}\":\n"""

def eval_model(args):
    # Model
    disable_torch_init()
    if args.textonly:
        # model_path = os.path.expanduser(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="balanced", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        model_name = get_model_name_from_path(args.model_path)
        
    else:
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)

        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for index, row in tqdm(questions.iterrows(), total=len(questions)):

        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            ###EBAY SPECIFIC
            if args.textonly == False or not args.textonly:
                image = Image.open(row['image'])
            
            ###
            # image = load_image_from_base64(row['image'])
            
            if args.context and args.shots == 0:
                qs = preprocess_data_gen(row)
                
            elif args.context and args.shots > 0:
                qs = preprocess_fewshot_data(row)
                qs = qs + preprocess_data_gen(row)
                
            else:
                qs = ''

            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            
            qs = qs + question 
            
            cur_prompt = question
            
            model.config.mm_use_im_start_end = False
            if args.textonly == False or not args.textonly:
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            if args.textonly == False or not args.textonly:
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                if "lilium" in args.model_base.lower():
                    prompt = conv.sep + conv.get_prompt()
                else:
                    prompt = conv.get_prompt()


            
            if args.textonly:
                # tokenizer.chat_template = 
                
                chat = [
                  {"role": "user", "content": qs}
                ]
                # print('chat', chat)

                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                # print('prompt', prompt)
                input_ids = tokenizer(
                        prompt,
                        return_tensors="pt",
                        return_attention_mask=False d,
                    ).to('cuda')
                    
                    
                output_ids = model.generate(
                    **input_ids,                 
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024, 
                    use_cache=True)
                
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    
            else:
               
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                image_tensor = process_images([image], image_processor, model.config)[0]
            
            
                input_ids = input_ids.to('cuda')
                image_tensor = image_tensor.to('cuda')

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True)

            
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # print("outputs", outputs)
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
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
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--context", type=str, default=False)
    parser.add_argument("--textonly", type=str, default=False)

    args = parser.parse_args()

    eval_model(args)
