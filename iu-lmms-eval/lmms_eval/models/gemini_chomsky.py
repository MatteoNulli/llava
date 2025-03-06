from io import BytesIO
import numpy as np
import base64
from typing import List, Tuple
from tqdm import tqdm
import time

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.chomsky import export_proxies

from accelerate import Accelerator, DistributedType
from pychomsky.chchat import GoogleGenAIWrapper
from langchain.schema import HumanMessage

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from PIL import Image

NUM_SECONDS_TO_SLEEP = 30
from loguru import logger as eval_logger


@register_model("gemini_chomsky")
class Gemini(lmms):
    def __init__(
        self,
        model_version: str = "gcp-chat-completions-chat-gemini-1.5-flash-002-sandbox",
        modality: str = "image",
        max_frames_for_video: int = 10,
        **kwargs,
    ) -> None:
        """
        List of available gemini models in chomsky can be found here:
        https://wiki.corp.ebay.com/display/COREAI/Chomsky+Model+Enumeration
        """

        super().__init__()
        self.model_version = model_version
        self.modality = modality
        self.max_frames_for_video = max_frames_for_video
        self.image_token = "<image>"

        export_proxies()
        self.client = GoogleGenAIWrapper(model_name=self.model_version)

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

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
            # encode, pad, and truncate contexts for this batch
            # visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = [doc_to_visual(self.task_dict[task][split][0])]
            visuals = self.flatten(visuals)
            imgs = []  # multiple images or frames for video
            for visual in visuals:
                if self.modality == "image":
                    img = self.encode_image(visual)
                    imgs.append(img)
                elif self.modality == "video":
                    frames = self.encode_video(visual, self.max_frames_for_video)
                    imgs.extend(frames)

            messages = []
            # When there is no image token in the context, append the image to the text
            if self.image_token not in contexts:
                content = []
                content.append({"type": "text", "text": contexts})
                for img in imgs:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
                messages.append(HumanMessage(content=content))
            else:
                contexts = contexts.split(self.image_token)
                for idx, img in enumerate(imgs):
                    content = []
                    content.append({"type": "text", "text": contexts[idx]})
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
                    messages.append(HumanMessage(content=content))

                # If n image tokens are in the contexts
                # contexts will be splitted into n+1 chunks
                # Manually add it into the payload
                content = [({"type": "text", "text": contexts[-1]})]
                messages.append(HumanMessage(content=content))

            if "temperature" in gen_kwargs:
                self.client.temperature = gen_kwargs["temperature"]
            if "top_p" in gen_kwargs:
                self.client.top_p = gen_kwargs["top_p"]
            if "max_new_tokens" in gen_kwargs:
                self.client.max_output_tokens = gen_kwargs["max_new_tokens"]
            if "top_k" in gen_kwargs:
                self.client.top_k = gen_kwargs["top_k"]

            for attempt in range(5):
                try:
                    response = self.client.invoke(messages)
                    response = response.content
                except Exception as e:
                    try:
                        error_msg = response.json()
                    except:
                        error_msg = ""

                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.\nReponse: {error_msg}")
                    if attempt <= 5:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty string
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.\nResponse: {response.json()}")
                        response = ""
            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Gemini")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Gemini not support"
