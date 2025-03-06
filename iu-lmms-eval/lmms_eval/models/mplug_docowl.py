import logging
import warnings
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from tqdm import tqdm
from transformers import TextStreamer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.mplug_docowl.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from lmms_eval.models.model_utils.mplug_docowl.conversation import conv_templates
from lmms_eval.models.model_utils.mplug_docowl.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from lmms_eval.models.model_utils.mplug_docowl.model.builder import (
    load_pretrained_model,
)
from lmms_eval.models.model_utils.mplug_docowl.processor import DocProcessor

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")


@register_model("mplug_docowl")
class mPLUGDocOwl(lmms):
    """
    mPLUG-DocOwl 1.5 model
    https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl1.5
    """

    def __init__(
        self,
        pretrained: str = "mPLUG/DocOwl1.5-Omni",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        anchors="grid_9",
        add_global_img=True,
        load_8bit=False,
        load_4bit=False,
        device_map: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        model_name = get_model_name_from_path(pretrained)
        self._tokenizer, self._model, _, _ = load_pretrained_model(
            pretrained,
            None,
            model_name,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device="cuda",
        )
        self.doc_image_processor = DocProcessor(
            image_size=448,
            anchors=anchors,
            add_global_img=add_global_img,
            add_textual_crop_indicator=True,
        )
        self.streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self._config = self._model.config

        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
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

    # should be deleted since max_new_tokens is decided by gen_kwargs not a model property
    # @property
    # def max_new_tokens(self) -> int:
    #     return 256

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "mPLUG DocOwl does not support loglikelihood evaluation yet"
        )

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            visuals = self.flatten(visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}"
                    )
            assert (
                self.batch_size_per_gpu == 1
            ), "Do not support batch_size_per_gpu > 1 for now"

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            question_input = []
            for visual, context in zip(visuals, contexts):
                if DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    image_tokens = (
                        [DEFAULT_IMAGE_TOKEN] * len(visual)
                        if isinstance(visual, list)
                        else [DEFAULT_IMAGE_TOKEN]
                    )
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                question_input.append(question)

            # When there are no visuals, e.g. pure text, the loop above won't execute
            if len(visuals) == 0:
                for context in contexts:
                    question_input.append(context)

            query = "\n".join(question_input)
            text_outputs = self.inference(
                images=visuals, query=query, gen_kwargs=gen_kwargs
            )
            res.append(text_outputs)
        return res

    def inference(self, images, query, gen_kwargs):
        image_tensor, patch_positions, text = self.doc_image_processor(
            images=images, query=query
        )
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        patch_positions = patch_positions.to(self.model.device)

        conv = conv_templates["mplug_owl2"].copy()

        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # preconfigure gen_kwargs with defaults
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 512
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 1.0

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                patch_positions=patch_positions,
                do_sample=False,
                temperature=gen_kwargs["temperature"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                streamer=self.streamer,
                use_cache=self.use_cache,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()

        return outputs.replace("</s>", "")
