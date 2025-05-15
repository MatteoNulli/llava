#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Callable, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn


from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
)
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput


from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, BOMMaskToken


class MyLlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        # group_ranges: list of length batch; each element is a list of
        #               (start, end) tuples indicating intervals within
        #               which all tokens should share the same pos‐id.
        #               If None, falls back to original per‐token positions.
        self.group_ranges: Optional[List[List[Tuple[int, int]]]] = None

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:            [batch, seq_len, dim]  (only used for device/dtype)
        position_ids: [1, seq_len]       absolute token positions
        """
        batch_size, seq_len, _ = x.shape

        # 1) Expand the shared position_ids to full batch
        updated_pos_ids = position_ids
        if updated_pos_ids.shape[0] == 1 and batch_size > 1:
            # expand then clone so we can mutate
            updated_pos_ids = updated_pos_ids.expand(batch_size, seq_len).clone()
        else:
            updated_pos_ids = updated_pos_ids.clone()

        # 2) If grouping requested, mask-and-replace so that
        #    all tokens in [st,ed] get the same rep ID = st
        if self.group_ranges is not None:
            for i, ranges in enumerate(self.group_ranges):
                for st, ed in ranges:
                    mask = (updated_pos_ids[i] >= st) & (updated_pos_ids[i] <= ed)
                    updated_pos_ids[i, mask] = st

        # 3) Now `updated_pos_ids` is [batch, seq_len] with your grouped IDs
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(batch_size, -1, 1).to(x.device)
        )
        updated_pos_ids_expanded = updated_pos_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ updated_pos_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.config = config
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.sam2_masking_token = False
        # projector_input_features = self.model.
        if hasattr(self.model, "mm_projector"):
            projector_input_features = self.model.mm_projector[0].in_features
            self.mm_bom_mask_token = BOMMaskToken(
                projector_input_features, self.dtype, device="cuda"
            )
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        masks: Optional[torch.LongTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # # cache_position=None
        # print('inside LM')
        # print('input ids', input_ids)
        # print('before requires_grad: ', input_ids.requires_grad)  # Output: False
        # input_ids.requires_grad = True
        # print('after requires_grad: ', input_ids.requires_grad)  # Output: True
        if self.model.custom_rotary_embedding:
            self.model.rotary_emb = MyLlamaRotaryEmbedding(self.config)
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                masks,
                image_sizes,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        masks: Optional[torch.LongTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:

            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    masks,
                    image_sizes=image_sizes,
                )
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
