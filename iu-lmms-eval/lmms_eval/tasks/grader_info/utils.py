import os
import json
from PIL import Image
import io
from enum import Enum
from collections import Counter
import re
from jsonschema import validate, exceptions
from loguru import logger as eval_logger


dir_name = os.path.dirname(os.path.abspath(__file__))


regex_schema = (
        r"(?:"  # Start of the string
        r"(True),"  # is_graded: true
        r"([\d#-]{3,14}),"  # certification_number: 3-14 numbers, # or -
        r"([A-Z]{3,4}),"  # professional_grader: 3 or 4 big letters
        r"(?:10|(?:[0-9](?:\.5)?)|authentic|Authentic)"  # grade: 0 - 10, authentic or Authentic
        r"|(False))"  # is_graded: false
    )

aspects = ("is_graded", "certification_number", "professional_grader", "grade")


class ClassResult(str, Enum):
    TP = 'true_positive'
    FN = 'false_negative'
    FP = 'false_positive'
    TN = 'true_negative'


def grader_info_to_visual(row):
    image = Image.open(io.BytesIO(row["image"]))
    return [image.convert("RGB")]


def grader_info_to_text(row, lmms_eval_specific_kwargs=None):
    return lmms_eval_specific_kwargs['post_prompt']


def grader_info_to_target(row):
    target = {aspect: row[aspect] for aspect in aspects if row[aspect] is not None}
    return target


def grader_info_process_results(row, results):
    response = results[0]
    row = grader_info_to_target(row)
    eval_logger.debug(f"Results: {results}")
    eval_logger.debug(f"Target: {row}")
    data_dict = dict()

    data_dict['regex_validity_rate'] = int(bool(re.fullmatch(regex_schema, response)))

    prediction = dict(zip(aspects, response.split(",")))
    prediction["is_graded"] = prediction["is_graded"] == "True"

    eval_logger.debug(f"Prediction: {prediction}")
    for aspect in aspects:
        if aspect == "is_graded":
            data_dict[f"{aspect}_precision"] = classify_result(row[aspect], prediction[aspect])
            data_dict[f"{aspect}_recall"] = classify_result(row[aspect], prediction[aspect])
            data_dict[f"{aspect}_f1"] = classify_result(row[aspect], prediction[aspect])
        if aspect in row:
            data_dict[f"{aspect}_accuracy"] = row[aspect] == prediction.get(aspect)
    eval_logger.debug(f"Data dict: {data_dict}")
    return data_dict


def extract_json_string(s):
    match = re.search(r"{([^}]*)}", s)
    if match:
        return match.group(0)
    else:
        return s


def classify_result(target, pred):
    if target and pred:
        return ClassResult.TP
    elif target and not pred:
        return ClassResult.FN
    elif not target and pred:
        return ClassResult.FP
    elif not target and not pred:
        return ClassResult.TN


def calculate_accuracy(results):
    eval_logger.debug(f"Results: {results}")
    return sum(results) / len(results)


def calculate_precision(results):
    eval_logger.debug(f"Results: {results}")
    counts = count_classifications(results)
    try:
        precision = (
            counts[ClassResult.TP] /
            (counts[ClassResult.TP] + counts[ClassResult.FP])
        )
    except ZeroDivisionError:
        precision = 0
    return precision


def calculate_recall(results):
    eval_logger.debug(f"Results: {results}")
    counts = count_classifications(results)
    try:
        recall = (
            counts[ClassResult.TP] /
            (counts[ClassResult.TP] + counts[ClassResult.FN])
        )
    except ZeroDivisionError:
        recall = 0
    return recall


def calculate_f1(results):
    eval_logger.debug(f"Results: {results}")
    precision = calculate_precision(results)
    recall = calculate_recall(results)
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    return f1


def count_classifications(results):
    counts = Counter(results)
    for key in ClassResult:
        if key not in counts:
            counts[key] = 0
    return counts


def regex_validity_rate(results):
    eval_logger.debug(f"Results: {results}")
    return sum(results) / len(results)
