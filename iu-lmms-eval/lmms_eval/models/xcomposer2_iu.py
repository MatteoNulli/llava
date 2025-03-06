from multiprocessing import context
import torch
from transformers import AutoModel, AutoTokenizer, LogitsProcessorList
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from datetime import timedelta
import logging

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria
from lmms_eval.structure_enforcer import JSONLogitsProcessor

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState

from typing import Optional, Sequence, List, Tuple, Union
import re
from tqdm import tqdm

eval_logger = logging.getLogger("lmms-eval")

@register_model("xcomposer2_iu")
class XComposer2IU(lmms):
    image_token = "<ImageHere>"

    def __init__(
        self,
        pretrained: str = "/mnt/iu-pvc/lmms-eval/models/XComposer2/vl_7b/",
        batch_size: Optional[Union[int, str]] = 1,
        device_map: Optional[str] = "auto",
        device: Optional[str] = "cuda:0",
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
        self._model = AutoModel.from_pretrained(self.pretrained, torch_dtype=self._dtype, device_map=self.device_map, local_files_only=True, trust_remote_code=True).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained, torch_dtype=self._dtype, local_files_only=True, trust_remote_code=True)
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
    def dtype(self):
        return self._dtype

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
            visuals = self.flatten(visuals)
            images = torch.cat([self.model.vis_processor(image).unsqueeze(0).to(self.device, dtype=self.dtype) for image in visuals], dim=0)

            # enforce JSON outputs
            if "json_schema" in gen_kwargs and self._use_json_enforcer:
                logits_processor = LogitsProcessorList([JSONLogitsProcessor(schema=gen_kwargs["json_schema"], tokenizer=self.tokenizer)])
            else:
                logits_processor = None

            if visuals:
                contexts = len(visuals) * self.image_token + contexts
            response, _ = self.model.chat(self.tokenizer, query=contexts, image=images, history=[], do_sample=False, logits_processor=logits_processor)
            response = response.split("[UNUSED_TOKEN_145]")[0].strip()
            response = response.split("<|im_end|>")[0].strip()
            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for XComposer2")
