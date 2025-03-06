import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
from io import BytesIO


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
from vllm import LLM
from vllm.sampling_params import SamplingParams

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math


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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
    
def eval_model(args):
    # Model
    disable_torch_init()
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
        
    elif args.qwen2:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='cuda:0'
        )

        # default processer
        processor = AutoProcessor.from_pretrained(args.model_path)

        
    elif args.pixtral:
        
        assert 'pixtral' in args.model_path.lower()
        
        max_img_per_msg = 5

        sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)

        # Lower max_num_seqs or max_model_len on low-VRAM GPUs.
        llm = LLM(model=args.model_path, tokenizer_mode="mistral", limit_mm_per_prompt={"image": max_img_per_msg}, max_model_len=32768)
        
    elif args.gpt4o:
        
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.prompts import PromptTemplate
        from pychomsky.chchat import EbayLLMChatWrapper

        
        model = EbayLLMChatWrapper(
            model_name='azure-chat-completions-gpt-4o-mini-2024-07-18',
            temperature=0.5,
            max_tokens=128
        )
    elif args.llama90b:
        
        assert 'llama-3_2' in args.model_path.lower()
        
        model = MllamaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map='cuda:0'
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        
        model = model.to(device='cuda:0', dtype=torch.float16)
    
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
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda:0", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        model_name = get_model_name_from_path(args.model_path)
        model = model.to(device='cuda:0', dtype=torch.float16)
        
    else:
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)

        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device='cuda:0', device_map="cuda:0")
        
        model = model.to(device='cuda:0', dtype=torch.float16)
        # print('image_processor.device()',image_processor.device())
        # model = model.to(device='cuda:0', dtype=torch.float16)
    

    questions = pd.read_parquet(os.path.expanduser(args.question_file), engine ='pyarrow')
    # print('questions', questions)
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        # print(row)
        # options = get_options(row, all_options)
        # cur_option_char = all_options[:len(options)]


        
        idx = row['index']
        question = row['question']
        answer = row['answer']
        
        # Image preprocessing
        # if args.textonly == False or not args.textonly and args.gpt4o==False:
        #     image = load_image_from_base64(row['image'])
        #     # print('image', image)

        if args.gpt4o or args.llama90b or args.pixtral:
            image = row['image']
            
        elif args.phi3 or args.molmo:
            placeholder = ''
            for i in range(1,2):
                ##single image support
                image = [Image.open(BytesIO(row['image']))]
                placeholder += f"<|image_{i}|>\n"
        
        else:
            image = Image.open(BytesIO(row['image']))

        q = question
        
        
        ## <image> token handling preprocessing
        model.config.mm_use_im_start_end = False
        # print("model.config", model.config)
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + q
        else:
            if args.gpt4o or args.llama90b or args.phi3 or args.molmo or args.qwen2 or args.pixtral or args.llama3_2:
                qs = q
            else:  
                qs = DEFAULT_IMAGE_TOKEN + '\n' + q 

        
        if args.textonly == False or not args.textonly:

            if not args.gpt4o and not args.llama90b and not args.pixtral or args.llama3_2:
                model.config.mm_use_im_start_end = False

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            
            conv = conv_templates[args.conv_mode].copy()

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            if "lilium" in args.model_base.lower():
                prompt = conv.sep + conv.get_prompt()
            elif "llama3" in args.conv_mode:
                prompt = conv.sep + conv.get_prompt()
            else:
                prompt = conv.get_prompt()
    

        elif args.textonly:
        
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
    
        
        elif args.molmo:
            # print('Evaluting molmo...')
            print('qs', qs)
            # process the image and text
            inputs = processor.process(
                images=image,
                text=qs
            )

            # move inputs to the correct device and make a batch of size 1
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

            # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )

            # only get generated tokens; decode them to text
            generated_tokens = output[0,inputs['input_ids'].size(1):]
            outputs = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print('outputs',outputs)
            
        elif args.phi3:
            # print('Evaluting phi3...')
            # print('qs', qs)
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
            # print('outputs', outputs)

        elif args.qwen2:
            # print('Evaluting Qwen2-VL...')
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
        
        elif args.pixtral:
            # print('Evaluating PIXTRAL...')
            # print('qs', qs)
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": qs}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]
                },
            ]

            outputs = llm.chat(messages, sampling_params=sampling_params)

            outputs = outputs[0].outputs[0].text


        elif args.gpt4o:
            # print('Evaluating gpt4o...')
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
                
        elif args.llama90b:
        
            ##to adapt this to 
            import requests
            qs = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language." + qs
            # print('qs', qs)
            body = {
                "input": {
                    "text_input": [qs],
                    "multi_modal_data": [f'{{\n "image": ["{image}"]\n}}'],
                    "sampling_parameters": ["{\"max_tokens\": 25, \"top_p\": 0.9}"]
                },
                "appName": "chao-test-llama-3-2-11b-vision-instruct-0-o4vn"
            }

            response = requests.post(
                    url, 
                    headers=headers, 
                    data=json.dumps(body), 
                    verify=False,
                )
            outputs = response.json()['output']['text_output'][0].split('Limit your self to a couple of words at most.')[1]
            # print('outputs', outputs)
            
        elif args.llama3_2:
            # print('Evaluating Llama 3.2-V...')
            # print('qs', qs)
            qs = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language." + qs
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
            
            output = model.generate(**inputs, max_new_tokens=1024)
            outputs = processor.decode(output[0]).split('assistant<|end_header_id|>')[1].strip('<|eot_id|>')
            # print('outputs', outputs)
        
        else:
            # print('qs', prompt)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            try:
                image_tensor = process_images([image], image_processor, model.config)[0].cuda()
            except:
                print(f'Skipped index {index} for reasonsm to do with image encoding')
                continue

            # input_ids = input_ids.to(model.device)
            # image_tensor = image_tensor.to(model.device)

            # print('input_ids device', input_ids.device)
            # print('image_tensor.device', image_tensor.device)
            # print('model.device', model.device)
            # print('image', image)

            with torch.inference_mode():
                if args.conv_mode == 'llama3':
                    # print('here?')
                    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

                    output_ids = model.generate(
                        input_ids,
                        # attention_mask = attention_mask,
                        pad_token_id=tokenizer.pad_token_id,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True)
                else:
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

    
        ans_id = shortuuid.uuid()
        # print(type(idx))
        ans_file.write(json.dumps({"question_id": str(idx),
                                "prompt": prompt,
                                "text": outputs,
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
    parser.add_argument("--gpt4o", type=str, default=False)
    parser.add_argument("--llama90b", type=str, default=False)
    parser.add_argument("--phi3", type=str, default=False)
    parser.add_argument("--molmo", type=str, default=False)
    parser.add_argument("--qwen2", type=str, default=False)
    parser.add_argument("--pixtral", type=str, default=False)
    parser.add_argument("--llama3-2", type=str, default=False)
    parser.add_argument("--textonly", type=str, default=False)
    args = parser.parse_args()

    eval_model(args)
