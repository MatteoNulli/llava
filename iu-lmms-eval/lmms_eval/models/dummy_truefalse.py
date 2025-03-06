import json
import re
import warnings
from typing import List, Tuple

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


@register_model("dummy_truefalse")
class Dummy_TrueFalse(lmms):
    """
    Dummy_TrueFalse Model
    Returns either True or False for each input

    How to run:
    $ python -m krylov.submit_pykrylov krylov/scripts/evaluate.sh --lmms_eval_task 'regulatory_doc'
      --experiment_name 'dummy_truefalse' --model_name 'dummy_truefalse' --model_args "const_result=true"
      --local
    """

    def __init__(
        self,
        const_result: bool = True,  # str is problematic due to bash interpretation of quotes
        **kwargs,
    ) -> None:
        super().__init__()
        self.const_result = const_result
        self.schema_re = r" JSON format.*:\s({.*})"

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not supported"

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        pbar = tqdm(total=len(requests), desc="Model Responding")

        for i, req in enumerate(requests):
            req_txt = req.args[0]
            m = re.search(self.schema_re, req_txt)
            if m:
                schema_found = m.group(1)
            else:
                eval_logger.error(f"Invalid input: {req_txt}")
                res.append("False")
                pbar.update(1)
                continue

            txt_out = schema_found.replace(
                "true/false", "true" if self.const_result else "false"
            )

            res.append(txt_out)
            pbar.update(1)

        pbar.close()
        return res
