import os
import math
import json
import torch
import argparse
import shortuuid
import pandas as pd
from tqdm import tqdm
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
from llava.model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

all_options = ['A', 'B', 'C', 'D']


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

    
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value == 'nan':
        return True
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


def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    """Pad a sequence to the desired max length."""
    if len(sequence) >= max_length:
        return sequence
    return torch.cat(
        [
            torch.full(
                (max_length - len(sequence),), padding_value, dtype=sequence.dtype
            ),
            sequence,
        ]
    )


class CustomDataset(Dataset):
    def __init__(
        self,
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        model_family=None,
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.model_family = model_family

    def __getitem__(self, index):
        row = self.questions.iloc(0)[index]
        answer = row["answer"]
        idx = row["index"]
        q = row["question"]
        hint = row["hint"]
        
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        # if args.all_rounds:
        #     num_rounds = len(options)
        # else:
        #     num_rounds = 1

        # image = Image.open(os.path.join(self.image_folder, row["image"])).convert("RGB")
        image = load_image_from_base64(row['image'])

        if "context examples" in row.keys():
            if not is_none(hint):
                q = hint + '\n' + q 
            for option_char, option in zip(all_options[:len(options)], options):
                q = q + '\n' + option_char + '. ' + option


            self.model_config.mm_use_im_start_end = False

            context = row["context examples"]
            context = str(context)
            if context != "nan":

                if args.single_pred_prompt:
                    context = (
                        "The following are aspects values corresponding to those of similar items to the one in question: "
                        + context
                        + "\n"
                        + "Taking these into consideration, answer the following: \n"
                    )

                    qs = (
                        DEFAULT_IMAGE_TOKEN
                        + context
                        + q
                        + "\n"
                        + "Answer by generating the aspect characteristics. Limit your self to a couple of words at most."
                    )

            else:
                if args.llama3_2:
                    qs = (
                        q
                        + "\n"
                        + "Answer by generating the aspect characteristics. Limit your self to a couple of words at most."
                    )
        
                if args.single_pred_prompt:
                    qs = (
                        DEFAULT_IMAGE_TOKEN
                        + q
                        + "\n"
                        + "Answer by generating the aspect characteristics. Limit your self to a couple of words at most."
                    )

        else:
            if not is_none(hint):
                q = hint + '\n' + q
            for option_char, option in zip(all_options[:len(options)], options):
                q = q + '\n' + option_char + '. ' + option

            self.model_config.mm_use_im_start_end = False

            if args.single_pred_prompt:
                qs = (
                    DEFAULT_IMAGE_TOKEN
                    + q
                    + "\n"
                    + "Answer by generating the aspect characteristics. Limit your self to a couple of words at most."
                )

        if self.model_config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

        if "lilium" in args.model_base.lower():
            prompt = conv.sep + conv.get_prompt()
        elif "llama3" in args.conv_mode:
            prompt = conv.sep + conv.get_prompt()
        else:
            prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        image_tensor = process_images([image], self.image_processor, self.model_config)[
            0
        ]
        image_tensor = image_tensor



        return input_ids, image_tensor, image, image.size, idx, q, qs, answer

    def __len__(self):
        return self.questions.shape[0]


def collate_fn(batch):
    input_ids, image_tensors, image, image_sizes, question_id, qs, prompt, answer = zip(*batch)
    # Determine the maximum length of input_ids in the batch
    max_len = max([len(seq) for seq in input_ids])
    # Pad each sequence in input_ids to the max_len
    padded_input_ids = [
        pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in input_ids
    ]
    input_ids = torch.stack(padded_input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return {
        "input_ids": input_ids,
        "image_tensor": image_tensors,
        "image": image, 
        "image_sizes": image_sizes,
        "question_id": question_id,
        "qs": qs,
        "prompt": prompt,
        "answer": answer,
    }



def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    batch_size=1,
    num_workers=4,
    model_family=None,
):
    dataset = CustomDataset(
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        model_family=model_family,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if args.llama3_2:
        
        assert 'llama-3_2' in args.model_path.lower()

        model_name = args.model_path.split('/')[-2]
        assert 'llama-3_2' in model_name.lower()
        # print('model_name', model_name)
        
        model = MllamaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        image_processor = AutoProcessor.from_pretrained(args.model_path, padding_side='left')
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')
        
        model = model.to(device='cuda:0', dtype=torch.float16)

        model_config = model.config

    else:
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path,
            args.model_base,
            model_name,
        )
        model_config = model.config

    # GPU setup
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])
        model.to('cuda')
    else:
        model = model.to(
            device="cuda",
            dtype=torch.float16,
        )

    # print('questions', questions)
    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model_config,
        model_family=args.model_base,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
    #     args.conv_mode = args.conv_mode + '_mmtag'
    #     print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
    # if args.llama3_2:

    #     for batch in tqdm(data_loader, total=len(data_loader)):
    #         # print('------')
    #         # print('batch prompt', batch['image'])
    #         # print('------')
    #         image_sizes = batch['image_sizes']
    #         print('image_sizes', image_sizes)
           
    #         for prompt, image, answer, q_id in zip(batch['prompt'], batch['image'], batch['answer'], batch['question_id']):


    #             messages = [
    #                 {"role": "user", "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": prompt,}
    #                 ]}
    #             ]
    #             input_text = image_processor.apply_chat_template(messages, add_generation_prompt=True)
    #             inputs = image_processor(
    #                 image,
    #                 input_text,
    #                 add_special_tokens=False,
    #                 return_tensors="pt"
    #             ).to('cuda:0')

    #             print('image_processor', image_processor)

    #             print("inputs", inputs.keys())
    #             print('inputs shapes', inputs['input_ids'].shape)
    #             print('attention_mask shapes', inputs['attention_mask'].shape)
    #             print('pixel_values shapes', inputs['pixel_values'].shape)

    #             output = model.generate(**inputs, max_new_tokens=30)

    #             outputs = image_processor.decode(output[0]).split('assistant<|end_header_id|>')[1].strip('<|eot_id|>')

    #             # print('outputs', outputs)

    #             ans_id = shortuuid.uuid()
    #             ans_file.write(json.dumps({
    #                 # "question_id": q_id,
    #                 "prompt": prompt,
    #                 "output": outputs,
    #                 "answer": answer,
    #                 "model_id": model_name,
    #                 "metadata": {}}) + "\n")
    #             ans_file.flush()

    
    for batch in tqdm(data_loader, total=len(data_loader)):
        if args.llama3_2:
            messages = [[
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]}
                ] for prompt in batch['prompt']]

            # print('messages', messages)
            # input1 = [{"role": "user", "content": "Hi, how are you?"}]
            # input2 = [{"role": "user", "content": "How are you feeling today?"}]
            
            input_texts = image_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = [image for image in batch['image']]
            print(len(images), len(input_texts))
            inputs = image_processor(
                    images,
                    input_texts,
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to('cuda:0')

            print("inputs", inputs.keys())
            print('inputs shapes', inputs['input_ids'].shape)
            print('inputs shapes', inputs['attention_mask'].shape)
            print('inputs shapes', inputs['pixel_values'].shape)
            print('inputs shapes', inputs['aspect_ratio_ids'].shape)
            
            output_ids = model.generate(**inputs, max_new_tokens=1024)
            
            outputs = image_processor.batch_decode(output_ids, skip_special_tokens=True)

            # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # for out in  outputs:
            #     print('outputs POST PROC', out.split('assistant<|end_header_id|>')[1].split('<|start_header_id|>')[1])

            # split('assistant<|end_header_id|>')[1].strip('<|eot_id|>')

        else:
            with torch.inference_mode():
                output_ids = model.generate(
                    batch["input_ids"].to("cuda"),
                    images=batch["image_tensor"].half().to("cuda"),
                    image_sizes=[batch["image_sizes"]],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,

                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for cnt, output in enumerate(outputs):
            ans_id = shortuuid.uuid()
            # print("question_id", batch["question_id"][cnt],)
            # print('\n')
            # print("prompt", batch["prompt"][cnt])
            # print('\n')
            # print("output", output.strip(),)
            # print('\n')
            # print("answer", batch["answer"][cnt],)
            # print('\n')
            # print("answer_id", ans_id,)
            # print('\n')
            # print("model_id", model_name,)
            # if args.llama3_2:
            #     # print(output)
            #     output = output.split('<|start_header_id|>')[1]

            ans_file.write(
                json.dumps(
                    {
                        # "question_id": batch["question_id"][cnt],
                        "prompt": batch["prompt"][cnt],
                        "output": output.split(),
                        "answer": batch["answer"][cnt],
                        "answer_id": ans_id,
                        "model_id": model_name,
                        "metadata": {},
                    }
                )
                + "\n"
            )
            ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/nushare2/data/mnulli/finetuning/llava-lilium-2-7b-chat-lora-15Mfash-short_llavamix-lr1e-4",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default="/mnt/nushare2/data/baliao/multimodal/model_zoos/lilium-2-7b-chat",
    )
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument(
        "--question-file",
        type=str,
        default="playground/data/eval/ebench-sm/ebench-sm_uk.tsv",
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        default="playground/data/eval/ebench-sm/answers/ebench-sm_uk/llava-lilium-2-7b-chat-lora-15Mfash-short_llavamix-lr1e-4.jsonl",
    )
    parser.add_argument("--conv-mode", type=str, default="llava_lilium_2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true", default="True")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--vllm", type=str, default="False")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--llama3-2", type=str, default=True)

    args = parser.parse_args()

    eval_model(args)