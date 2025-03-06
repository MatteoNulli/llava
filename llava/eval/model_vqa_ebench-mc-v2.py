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
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    AutoProcessor, 
    GenerationConfig, 
    Qwen2VLForConditionalGeneration, 
    MllamaForConditionalGeneration, 
    LlavaForConditionalGeneration

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
        # hint = row["hint"]
        hint = None
        # print('hint', hint)
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        # if args.all_rounds:
        #     num_rounds = len(options)
        # else:
        #     num_rounds = 1

        image = Image.open(os.path.join(self.image_folder, row["image"])).convert("RGB")

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
        if args.molmo:
            image_tensor = [image]
        else:
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
            image_tensor = image_tensor



        return input_ids, image_tensor, image.size, idx, q, answer

    def __len__(self):
        return self.questions.shape[0]


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, question_id, prompt, answer = zip(*batch)
    # Determine the maximum length of input_ids in the batch
    max_len = max([len(seq) for seq in input_ids])
    # Pad each sequence in input_ids to the max_len
    padded_input_ids = [
        pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in input_ids
    ]
    input_ids = torch.stack(padded_input_ids, dim=0)
    if args.molmo:
        image_tensors = image_tensors
    else:
        image_tensors = torch.stack(image_tensors, dim=0)
    return {
        "input_ids": input_ids,
        "image_tensor": image_tensors,
        "image_sizes": image_sizes,
        "question_id": question_id,
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

    model_name = get_model_name_from_path(model_path) 

    if args.molmo:
        # print('in molmo!!')
        assert 'molmo' in args.model_base.lower()
        
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='cuda:-'
        )

        # load the model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='cuda:0'
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer, model, processor, context_len = load_pretrained_model(
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
        processor,
        model_config,
        model_family=args.model_base,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
    #     args.conv_mode = args.conv_mode + '_mmtag'
    #     print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for batch in tqdm(data_loader, total=len(data_loader)):

        # if args.molmo:
        #     prompts = []

        #     for cnt, q in enumerate(batch["prompt"]):
        #         prompts.append(q + '\n' + "Answer by generating the aspect characteristics. Limit your self to a couple of words at most.")
            
        #     print(prompts)
        #     print(batch["image_tensor"])

        #     inputs = processor.process(
        #             images=batch["image_tensor"],
        #             text=prompts
        #     )

            

        #     # move inputs to the correct device and make a batch of size 1
        #     inputs = {k: v.to(model.device) for k, v in inputs.items()}

        #     # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        #     output = model.generate_from_batch(
        #         inputs,
        #         GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        #         tokenizer=processor.tokenizer
        #     )

        #     # only get generated tokens; decode them to text
        #     generated_tokens = output[0,inputs['input_ids'].size(1):]
        #     outputs = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        with torch.inference_mode():
            output_ids = model.generate(
                batch["input_ids"].to("cuda"),
                pad_token_id=tokenizer.pad_token_id,
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
            ans_file.write(
                json.dumps(
                    {
                        "question_id": batch["question_id"][cnt],
                        "prompt": batch["prompt"][cnt],
                        "output": output.strip(),
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
    parser.add_argument("--gpt4o", type=str, default=False)
    parser.add_argument("--llama90b-endpoint", type=str, default=False)
    parser.add_argument("--phi3", type=str, default=False)
    parser.add_argument("--molmo", type=str, default=False)
    parser.add_argument("--qwen2", type=str, default=False)
    parser.add_argument("--pixtral", type=str, default=False)
    parser.add_argument("--aria", type=str, default=False)    
    parser.add_argument("--llama3-2", type=str, default=False)
    args = parser.parse_args()

    eval_model(args)