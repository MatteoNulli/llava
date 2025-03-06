import datetime
import json
import os
import io
import ast
from PIL import Image


from collections import defaultdict

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))


replace_prompt = " Please answer yes or no."


def mmvp_doc_to_visual(doc):
    return [Image.open(doc["image"]).convert("RGB")]


def mmvp_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    options = doc["Options"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return question + " " + options + post_prompt


def mmvp_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]
    gt = doc["answer"]

    return {
        "match_score": {"prediction": pred, "answer": gt},
    }


def mmvp_acc_results(results):

    correct_predictions = 0
    total_predictions = 0
    for result in results:
        total_predictions += 1
        if result["answer"] in result["prediction"]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions

    eval_logger.info(f"Total match score accuracy: {accuracy:.4f}")

    return accuracy
