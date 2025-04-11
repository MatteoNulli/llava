import datetime
import json
import os
import io
import ast
import re
from PIL import Image


from collections import defaultdict

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))


replace_prompt = " Please answer yes or no."


def mmvp_doc_to_visual(doc):
    # print("image", doc["images"])
    # print("image opened", Image.open(doc["images"]).convert("RGB"))
    image_id = doc["images"].split("/")[-1]
    return [Image.open(doc["images"]).convert("RGB"), image_id]


def mmvp_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["Question"]
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
    # print("doc", doc)
    gt = doc["Correct Answer"]

    return {
        "match_score": {"prediction": pred, "answer": gt},
    }


def normalize_option(text):
    """
    Extracts the first alphabetical character from the string,
    ignoring any leading parentheses, spaces, or trailing punctuation.
    Returns the letter in lowercase.
    If there's content within parentheses, return that letter.
    Otherwise, return the first alphabetical character.
    """
    # First, check if there's an alphabetical character within parentheses
    match_paren = re.search(r"\(\s*([a-zA-Z])\s*\)", text)
    if match_paren:
        return match_paren.group(1).lower()

    # If no valid letter in parentheses, look for the first alphabetical character elsewhere
    match_letter = re.search(r"[a-zA-Z]", text)
    if match_letter:
        return match_letter.group(0).lower()

    return "None"


def mmvp_acc_results(results):

    correct_predictions = 0
    total_predictions = 0
    for result in results:
        total_predictions += 1
        pred = normalize_option(result["prediction"])
        ans = normalize_option(result["answer"])
        print("")
        print("prediction:", result["prediction"], "normalized pred:", pred)
        print("answer:", result["answer"], "normalized answer:", ans)
        if ans in pred:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions

    eval_logger.info(f"Total match score accuracy: {accuracy:.4f}")

    return accuracy
