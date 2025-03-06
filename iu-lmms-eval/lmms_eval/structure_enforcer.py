"""
 _______________________________
/ Don't want to self-host?       \
\\ Try .json at http://dottxt.co /
 -------------------------------
       \\   ^__^
        \\  (oo)\\_______
            (__)\\       )\\/\
                ||----w |
                ||     ||
Copyright 2024- the Outlines developers
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List, Optional, Type, Union

import torch
from pydantic import BaseModel

from transformers import LogitsProcessor
from outlines.fsm.guide import Guide, RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import convert_json_schema_to_str
from outlines.models.tokenizer import Tokenizer


class FSMLogitsProcessor(LogitsProcessor):
    """Bias generation using a finite state machine.
    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, tokenizer: Tokenizer, fsm: Guide, device: Optional[str] = "cuda"):
        """A FSM-based logits processor.
        Parameters
        ----------
        tokenizer
            The tokenizer used to convert tokens to ids.
        fsm
            The finite state machine which is used to bias the logits.
        device
            The device where the model is running.
        """
        self.tokenizer = tokenizer
        self.fsm: Guide = fsm
        self.device = device
        self._fsm_state = 0
        self._first_call = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Use the FSM to bias the logits before sampling the next token.
        Parameters
        ----------
        input_ids
            The input token ids.
        scores
            The logits.
        Returns
        -------
        torch.Tensor
            The biased logits.
        """
        if self._first_call:
            self._first_call = False
        elif input_ids.numel() != 0:
            last_token = input_ids[..., -1].item()
            self._fsm_state = self.fsm.get_next_state(self._fsm_state, last_token)

        allowed_tokens = self.fsm.get_next_instruction(self._fsm_state).tokens
        mask = torch.full((scores.shape[-1],), torch.finfo(scores.dtype).min, device=self.device)
        mask[allowed_tokens] = 0.
        biased_scores = scores + mask

        return biased_scores

    @staticmethod
    def _adapt_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
        """Adapt tokenizer to work with the FSM.

        The API of Outlines tokenizers is slightly different to that of
        `transformers`. In addition we need to handle the missing spaces to
        Llama's tokenizer to be able to compile FSMs for this model.

        """
        tokenizer.vocabulary = tokenizer.get_vocab()
        tokenizer.special_tokens = set(tokenizer.all_special_tokens)

        def convert_token_to_string(token: str) -> str:
            from transformers.file_utils import SPIECE_UNDERLINE

            string = tokenizer.convert_tokens_to_string([token])

            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

            return string

        tokenizer.convert_token_to_string = convert_token_to_string
        return tokenizer

    def copy(self) -> "FSMLogitsProcessor":
        """Return a copy of the logits processor."""
        return FSMLogitsProcessor(tokenizer=self.tokenizer, fsm=self.fsm.copy())

    def reset(self) -> None:
        """Reset the state of the FSM."""
        self._fsm_state = 0
        self._first_call = True


class RegexLogitsProcessor(FSMLogitsProcessor):
    """Bias generation based on a regular expression.
    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, regex_string: str, tokenizer: Tokenizer, **kwargs):
        """Compile the FSM that drives the regex-guided generation.
        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        tokenizer
            An Outlines tokenizer
        """
        tokenizer = self._adapt_tokenizer(tokenizer)
        fsm = RegexGuide(regex_string, tokenizer)
        super().__init__(tokenizer=tokenizer, fsm=fsm, **kwargs)


class JSONLogitsProcessor(RegexLogitsProcessor):
    """Bias generation based on a JSON schema.
    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(
        self,
        schema: Union[dict, Type[BaseModel], str],
        tokenizer: Tokenizer,
        whitespace_pattern: Optional[str] = r"\s?",
        allow_non_json_prefix: bool = False,
        **kwargs
    ):
        """Compile the FSM that drives the JSON-guided generation.
        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to generate.
        tokenizer
            The tokenizer used to convert tokens to ids.
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string
            literals). For example, to allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`
        allow_non_json_prefix
            Whether to allow non-JSON prefixes in the input. If set to False, the
            FSM will only start from the first JSON object in the input.
            Works by prepending regex: [^\{]*
        """
        schema_str = convert_json_schema_to_str(json_schema=schema)
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        if allow_non_json_prefix:
            regex_string = r"[^\{]*" + regex_string
        super().__init__(regex_string=regex_string, tokenizer=tokenizer, **kwargs)
