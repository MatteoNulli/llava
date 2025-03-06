import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import base64

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from vllm import LLM
from vllm.sampling_params import SamplingParams


from PIL import Image
import math
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    Qwen2VLForConditionalGeneration,
)

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


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if args.llama3_2:
    
        assert 'llama-3_2' in args.model_path.lower()
        
        model = MllamaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        
        model = model.to(device='cuda:0', dtype=torch.float16)
        
    elif args.phi3:
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

    elif args.pixtral:
        
        assert 'pixtral' in args.model_path.lower()
        
        max_img_per_msg = 5

        sampling_params = SamplingParams(max_tokens=8192, temperature=0)

        # Lower max_num_seqs or max_model_len on low-VRAM GPUs.
        llm = LLM(model=args.model_path, tokenizer_mode="mistral", limit_mm_per_prompt={"image": max_img_per_msg})
       
    elif args.qwen2:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        # default processer
        processor = AutoProcessor.from_pretrained(args.model_path)

    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
        model = model.to(device='cuda:0', dtype=torch.float16)

    
    
    # print('normal path', args.question_file)
    # print('path expander', os.path.expanduser(args.question_file), "r")
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        # print('question', question)
        cur_prompt = question['value']
        qs = question['value'].replace('<image>', '').strip()


        

        if 'image' in line:
            if args.phi3 or args.molmo:
                placeholder = ''
                for i in range(1,2):
                    ##single image support
                    
                    image_file = line["image"]
                    image = [Image.open(os.path.join(args.image_folder, image_file))]
                    placeholder += f"<|image_{i}|>\n"
            elif args.pixtral:
                image_file = line['image']
                image_path = os.path.join(args.image_folder, image_file)
                image = encode_image(image_path)


            elif args.llama3_2 or args.qwen2: 
                image_file = line["image"]
                image = Image.open(os.path.join(args.image_folder, image_file))
                
            else:            
                image_file = line["image"]
                image = Image.open(os.path.join(args.image_folder, image_file))
                # print('image', image)
                image_tensor = process_images([image], image_processor, model.config)[0]
                images = image_tensor.unsqueeze(0).half().cuda()
                image_sizes = [image.size]

            if not args.pixtral:
                if getattr(model.config, 'mm_use_im_start_end', False):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                elif args.gpt4o or args.llama90b or args.phi3 or args.molmo or args.qwen2 or args.pixtral or args.llama3_2:
                    qs = '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                cur_prompt = '<image>' + '\n' + cur_prompt
            else:
                qs = '\n' + qs
        else:
            images = None
            image_sizes = None
            image = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        # if "lilium" in args.model_base.lower():
        if "lilium" in args.conv_mode: 
            # print("Entering")
            prompt = conv.sep + conv.get_prompt()
        elif 'llama3' in args.conv_mode:    
            prompt = conv.sep + conv.get_prompt()
        else:
            prompt = conv.get_prompt()


        if args.llama3_2:
            
            # print('qs', qs)
            # qs = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language." + qs
            
            if images==None:

                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": qs}
                    ]}
                ]
                input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

                # print(processor.tokenizer)
                inputs = processor.tokenizer( 
                    text = input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(model.device)
            
            else:
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
            outputs = processor.decode(output[0], skip_special_tokens=True).split('assistant')[1].strip('\n').strip('.')
            # print(outputs)
        
        elif args.molmo:

            if images==None:
                # inputs = processor.tokenizer.process(text=qs).to(model.device)
                inputs = processor.process(
                    images=None,
                    text=qs
                )
            else:
                inputs = processor.process(
                    images=image,
                    text=qs
                )

            # move inputs to the correct device and make a batch of size 1
            # print('inputs', inputs.items())
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

            # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )

            # only get generated tokens; decode them to text
            generated_tokens = output[0,inputs['input_ids'].size(1):]
            outputs = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip(' ')
            # print(outputs)

        elif args.pixtral:
            
                # print('qs', qs)
                if image == None:
                    messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": qs}]
                    },
                ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": qs}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]
                        },
                    ]

                outputs = llm.chat(messages, sampling_params=sampling_params)

                outputs = outputs[0].outputs[0].text

        elif args.phi3:
            # print('Evaluting phi3...')
            # print('qs', qs)
            if images==None:
                messages = [
                    {"role": "user", "content": qs},
                ]
                prompt = processor.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                inputs = processor.tokenizer(prompt, return_tensors="pt").to("cuda:0") 

            else:
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
            if images==None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": qs},
                        ],
                    }
                ]
                # Preparation for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor.tokenizer(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                )

            else:

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

        else:
            # print("prompt", prompt)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            with torch.inference_mode():
                if args.conv_mode == 'llama3':
                    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id 

                    output_ids = model.generate(
                    input_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    images=images,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                )
                else:
                    output_ids = model.generate(
                        input_ids,
                        images=images,
                        image_sizes=image_sizes,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=1024,
                        use_cache=True,
                    )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print("outputs", outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
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
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
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
