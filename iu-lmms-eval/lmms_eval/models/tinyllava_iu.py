import torch

from tqdm import tqdm
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import tempfile
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.structure_enforcer import JSONLogitsProcessor
from .model_utils.tiny_llava.modeling_tinyllava_phi import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, tokenizer_image_token, conv_phi_v0, process_images


@register_model("tinyllava_iu")
class TinyLlavaIU(lmms):
    """
    TinyLLaVA Model
    """
    def __init__(
        self,
        pretrained: str = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        revision="a98601f69e72442f71721aefcfbcdce26db8982a",
        use_json_enforcer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Whether use outlines or not
        self._use_json_enforcer = use_json_enforcer

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            device_map=self._device,
            revision=revision,
        )
        self._config = self._model.config
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            local_files_only=True,
            trust_remote_code=True,
            revision=revision,
        )
        self.model.eval()
        self.batch_size_per_gpu = int(batch_size)
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
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

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
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for TinyLLaVA")

    def chat(
        self,
        prompt: str,
        image: Image = None,
        max_new_tokens: int = 512,
        num_beams = 1,
        top_p=None,
        temperature=0,
        **kwargs
    ):
        image_processor = self.model.vision_tower._image_processor

        if image is not None:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv = conv_phi_v0.copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if image is not None:
            image_tensor = process_images(image, image_processor, self.model.config).to(self.device)
        else:
            image_tensor = None

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0).to(self.device)
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                **kwargs
            )

        return outputs

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            # tinyllava only supports one visual input
            assert len(visuals) == 1, "TinyLLaVA only supports one visual input"

            # enforce JSON outputs
            if "json_schema" in gen_kwargs and self._use_json_enforcer:
                logits_processor = LogitsProcessorList([JSONLogitsProcessor(
                    schema=gen_kwargs["json_schema"],
                    tokenizer=self.tokenizer,
                    )])
            else:
                logits_processor = None

            outputs = self.chat(prompt=contexts, image=visuals[0],
                                logits_processor=logits_processor,
                                return_dict_in_generate=True,
                                output_scores=True,
                )

            text_outputs = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            beam_indices = outputs.beam_indices if gen_kwargs["num_beams"] > 1 else None
            scores = [x[:1] for x in outputs.scores]
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, scores, beam_indices=beam_indices, normalize_logits=True
            )
            res.append({"output": text_outputs, "confidence": transition_scores, "tokens": outputs.sequences[..., -transition_scores.shape[-1]:]})
            pbar.update(1)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for TinyLLaVA")