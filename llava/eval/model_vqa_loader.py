import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration, 
    MllamaForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, model_family=None):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.model_family = model_family

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if args.qwen2 or args.llama3_2 or args.llama90b or args.phi3 or args.molmo or args.pixtral:
            qs = qs  
        else:  
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        if "lilium" in self.model_family.lower():
            prompt = conv.sep + conv.get_prompt()
        elif "llama3" in args.conv_mode:
            # print('conv', conv)
            prompt = conv.sep + conv.get_prompt()
        else:
            prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, model_family=None):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, model_family=model_family)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    if args.qwen2:
        print('Loading Qwen2VL model.')
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='cuda:0'
        )

        # default processer
        processor = AutoProcessor.from_pretrained(args.model_path)
        # print('processor', processor)
        tokenizer = processor.tokenizer
        
    elif args.llama3_2:
        print('Loading Llama3.2-V model.')
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
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
        processor = image_processor

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
    

    # image_path = args.image_folder
    # image = Image.open(args.image_folder + line["image"]).convert('RGB')

    if args.qwen2:
        print('Evaluating Qwen2VL model.')
        for line in tqdm(questions):
            idx = line["question_id"]
            cur_prompt = line["text"]
            qs = cur_prompt

            image = Image.open(os.path.join(args.image_folder, line["image"])).convert('RGB')
            # print('qs', qs)
            # qs = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language." + qs
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
            

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")

    elif args.llama3_2:
        print('Evaluating Llama3.2-V model.')
        for line in tqdm(questions):
            idx = line["question_id"]
            cur_prompt = line["text"]
            qs = cur_prompt
            image = Image.open(os.path.join(args.image_folder, line["image"])).convert('RGB')

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
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
    else:

        data_loader = create_data_loader(questions, args.image_folder, tokenizer, processor, model.config, model_family=args.model_base)

        for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
            idx = line["question_id"]
            cur_prompt = line["text"] 
            # qs = cur_prompt
            # print('image_sizes', image_sizes)

            input_ids = input_ids.to(device='cuda', non_blocking=True)


            with torch.inference_mode():
                if args.conv_mode == 'llama3':
                    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                    
                    output_ids = model.generate(
                        input_ids,
                        pad_token_id=tokenizer.pad_token_id,
                        images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                        image_sizes=image_sizes,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True)

                else:
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                        image_sizes=image_sizes,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--gpt4o", type=str, default=False)
    parser.add_argument("--llama90b", type=str, default=False)
    parser.add_argument("--phi3", type=str, default=False)
    parser.add_argument("--molmo", type=str, default=False)
    parser.add_argument("--qwen2", type=str, default=False)
    parser.add_argument("--pixtral", type=str, default=False)
    parser.add_argument("--llama3-2", type=str, default=False)
    args = parser.parse_args()

    eval_model(args)
