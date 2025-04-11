import os
import json
import argparse
import pandas as pd
import string
import re
import nltk
import io
import ssl
import urllib.request

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm
from urllib.error import URLError
from getpass import getpass
from PIL import Image

from loguru import logger as eval_logger


def cvbench_doc_to_visual(doc):
    # try:
    #     return {f"{doc['image']}": [doc["image"].convert("RGB")]}
    # except:
    # print('\n Opening the image in a different way... \n Image is probably in bytes, string or different format. \n')
    # byts = ast.literal_eval(doc['image'])['bytes']
    return [doc["image"], doc["filename"]]


def cvbench_doc_to_text_mc(doc, lmms_eval_specific_kwargs=None):
    question = doc["prompt"]
    post = lmms_eval_specific_kwargs["post_prompt"]
    return question + " " + post


def cvbench_process_results(doc, results):
    """
    Processes string matching results and stores necessary information for metric computation.
    """
    pred = results[0]
    gt = doc["answer"]
    source = doc["source"]
    return {
        "2D_matching_accuracy": {"prediction": pred, "answer": gt, "source": source},
        "3D_matching_accuracy": {"prediction": pred, "answer": gt, "source": source},
        "overall_matching_accuracy": {"prediction": pred, "answer": gt, "source": source},
    }


def retrieve_special_characters(text):
    # First, look for a single letter between brackets,
    # allowing for optional spaces inside the parentheses.
    bracket_match = re.search(r"\(\s*([A-Ea-e])\s*\)", text)
    if bracket_match:
        # Return just the letter (without the brackets)
        return bracket_match.group(1)

    # If no bracket match, look for a standalone letter A–E.
    # This regex uses lookarounds to ensure the letter is not part of a larger word.
    standalone_match = re.search(r"(?<!\S)([A-Ea-e])(?!\S)", text)
    if standalone_match:
        return standalone_match.group(1)

    return None  # Return None if no match found.


def calculate_accuracy(results, source):
    # removing noise from outputs
    results_sourced = [result for result in results if result["source"] == source]

    all_elements = len(results_sourced)
    pos = 0
    for result in results_sourced:
        pred = retrieve_special_characters(result["prediction"])
        ans = retrieve_special_characters(result["answer"])
        # print("")
        print("pred", result["prediction"], "pred after processing", pred, "ans", result["answer"], "ans after processing", ans)
        if ans == pred:
            print("Yes")
            pos += 1

    accuracy = pos / all_elements

    return accuracy


def accuracies_2D(results):
    # Calculate accuracy for each source
    accuracy_2d_ade = calculate_accuracy(results, "ADE20K")
    accuracy_2d_coco = calculate_accuracy(results, "COCO")
    accuracy_3d_omni = calculate_accuracy(results, "Omni3D")

    # Calculate the accuracy for each type
    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2

    eval_logger.info(f"nyu-cvbench 2D accuracy:", accuracy_2d)

    return accuracy_2d


def accuracies_3D(results):
    # Calculate accuracy for each source
    accuracy_2d_ade = calculate_accuracy(results, "ADE20K")
    accuracy_2d_coco = calculate_accuracy(results, "COCO")
    accuracy_3d_omni = calculate_accuracy(results, "Omni3D")

    # Calculate the accuracy for each type
    accuracy_3d = accuracy_3d_omni

    eval_logger.info(f"nyu-cvbench 3D accuracy:", accuracy_3d)

    return accuracy_3d


def combine_accuracies(results):
    # Calculate accuracy for each source
    accuracy_2d_ade = calculate_accuracy(results, "ADE20K")
    accuracy_2d_coco = calculate_accuracy(results, "COCO")
    accuracy_3d_omni = calculate_accuracy(results, "Omni3D")

    # Calculate the accuracy for each type
    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni

    # Compute the combined accuracy as specified
    combined_accuracy = (accuracy_2d + accuracy_3d) / 2

    eval_logger.info(f"nyu-cvbench accuracy:", combined_accuracy)

    return combined_accuracy
