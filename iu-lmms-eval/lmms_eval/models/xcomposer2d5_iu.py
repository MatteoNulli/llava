import torch
from transformers import AutoModel, AutoTokenizer, LogitsProcessorList
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from datetime import timedelta
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from typing import Optional, Sequence, List, Tuple, Union
import re
from tqdm import tqdm
from loguru import logger as eval_logger

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria
from lmms_eval.structure_enforcer import JSONLogitsProcessor
from lmms_eval.models.model_utils.xcomposer2d5.modeling_internlm_xcomposer2 import InternLMXComposer2ForCausalLM
from lmms_eval.models.model_utils.xcomposer2d5.tokenization_internlm2 import InternLM2Tokenizer

pattern = re.compile(r"[A-Z]")


@register_model("xcomposer2d5_iu")
class XComposer2d5IU(lmms):
    image_token = "<ImageHere>"

    def __init__(
        self,
        pretrained: str = "/mnt/iu-pvc/lmms-eval/models/internlm-xcomposer2d5-7b/",
        revision: str = "5a67bb96cc2974e1857dc76badca3e50c258cb73",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        device_map="cuda:0",
        need_bos: bool = True,
        padding: bool = False,
        half: bool = True,
        use_json_enforcer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # Whether use outlines or not
        self._use_json_enforcer = use_json_enforcer

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        if half:
            self._dtype = torch.bfloat16
        else:
            self._dtype = torch.float32

        self.pretrained = pretrained
        self.need_bos = need_bos
        self.padding = padding
        self._model = InternLMXComposer2ForCausalLM.from_pretrained(self.pretrained, revision=revision, torch_dtype=self._dtype, device_map=self.device_map, local_files_only=True, trust_remote_code=True).eval()
        self._tokenizer = InternLM2Tokenizer.from_pretrained(self.pretrained, revision=revision, torch_dtype=self._dtype, local_files_only=True, trust_remote_code=True)
        self.model.tokenizer = self.tokenizer
        self.batch_size_per_gpu = batch_size

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = [Image_transform(image[0]) for image in visuals]
            images = torch.cat([self.model.vis_processor(image).unsqueeze(0).to(self.device, dtype=self._dtype) for image in visuals], dim=0)

            # enforce JSON outputs
            if "json_schema" in gen_kwargs and self._use_json_enforcer:
                logits_processor = LogitsProcessorList([JSONLogitsProcessor(schema=gen_kwargs["json_schema"], tokenizer=self.tokenizer)])
            else:
                logits_processor = None

            if visuals:
                contexts = len(visuals) * self.image_token + contexts
            response, _ = self.model.chat(self.tokenizer, query=contexts, image=images, history=[], do_sample=False, logits_processor=logits_processor, meta_instruction="")
            response = response.split("[UNUSED_TOKEN_145]")[0].strip()
            response = response.split("<|im_end|>")[0].strip()
            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def generate_until_4khd(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if "[UNUSED_TOKEN_146]" not in contexts:
                contexts = f"[UNUSED_TOKEN_146]user\n{contexts}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if visuals:
                contexts = len(visuals) * self.image_token + contexts

            if "hd_num" not in gen_kwargs:
                if listinstr(["docvqa_test", "infovqa_test"], task.lower()):
                    self.model.hd_num = 65
                elif listinstr(["docvqa_val", "infovqa_val", "OCRBench"], task.lower()):
                    self.model.hd_num = 55
                elif listinstr(["mmmu", "mmbench", "mmvet"], task.lower()):
                    self.model.hd_num = 16
                else:
                    self.model.hd_num = 25
            else:
                self.model.hd_num = gen_kwargs.pop("hd_num")

            pt1 = 0
            embeds = []
            im_mask = []
            images_loc = [0]
            need_bos = self.need_bos
            for i, pts in enumerate(images_loc + [len(contexts)]):
                subtext = contexts[pt1:pts]
                if need_bos or len(subtext) > 0:
                    text_embeds = self.model.encode_text(subtext, add_special_tokens=need_bos).to(self.device)
                    embeds.append(text_embeds)
                    im_mask.append(torch.zeros(text_embeds.shape[:2]).to(self.device))
                    need_bos = False
                if i < len(visuals):
                    image = visuals[i]

                    image = HD_transform(image, im_num=self.model.hd_num)
                    image = self.model.vis_processor(image).unsqueeze(0).to(self.device)
                    image_embeds = self.model.encode_img(image)
                    embeds.append(image_embeds)
                    im_mask.append(torch.ones(image_embeds.shape[:2]).to(self.device))
                pt1 = pts
            embeds = torch.cat(embeds, dim=1)
            im_mask = torch.cat(im_mask, dim=1)
            im_mask = im_mask.bool()

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "repetition_penalty" not in gen_kwargs:
                gen_kwargs["repetition_penalty"] = 1.0

            # enforce JSON outputs
            if "json_schema" in gen_kwargs and self._use_json_enforcer:
                logits_processor = LogitsProcessorList([JSONLogitsProcessor(schema=gen_kwargs["json_schema"], tokenizer=self.tokenizer)])
            else:
                logits_processor = None

            outputs = self.model.generate(
                inputs_embeds=embeds,
                im_mask=im_mask,
                temperature=gen_kwargs["temperature"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                num_beams=gen_kwargs["num_beams"],
                do_sample=gen_kwargs["do_sample"],
                repetition_penalty=gen_kwargs["repetition_penalty"],
                logits_processor=logits_processor,
            )
            output_token = outputs[0]
            if output_token[0] == 0 or output_token[0] == 1:
                output_token = output_token[1:]
            output_text = self.model.tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split("[UNUSED_TOKEN_145]")[0].strip()
            output_text = output_text.split("<|im_end|>")[0].strip()
            res.append(output_text)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for XComposer2d5")

def Image_transform(img, hd_num=25):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width/ height)
    scale = 1
    while scale*np.ceil(scale/ratio) <= hd_num:
        scale += 1
    scale -= 1
    scale = min(np.ceil(width / 560), scale)
    new_w = int(scale * 560)
    new_h = int(new_w / ratio)

    img = transforms.functional.resize(img, [new_h, new_w],)
    img = padding_560(img)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img

def padding_560(b):
    width, height = b.size
    tar = int(np.ceil(height / 560) * 560)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255, 255, 255])

    return b


def HD_transform(img, im_num=16):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= im_num:
        scale += 1
    scale -= 1
    new_w = int(scale * 560)
    new_h = int(new_w / ratio)

    img = transforms.functional.resize(
        img,
        [new_h, new_w],
    )
    img = padding_560(img)
    width, height = img.size
    assert width * height <= im_num * 560 * 560
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img


def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False


def DATASET_TYPE(dataset):
    # Dealing with Custom Dataset
    dataset = dataset.lower()
    if listinstr(["mmbench", "seedbench", "ccbench", "mmmu", "scienceqa", "ai2d", "mmstar"], dataset):
        return "multi-choice"
    elif listinstr(["mme", "hallusion"], dataset):
        return "Y/N"
    elif "coco" in dataset:
        return "Caption"
    elif listinstr(["ocrvqa", "textvqa", "chartqa", "mathvista", "docvqa", "infovqa", "llavabench", "mmvet", "ocrbench"], dataset):
        return "VQA"
    else:
        return "QA"
