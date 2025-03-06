import os
import json
from PIL import Image
import io
from enum import Enum
from collections import Counter
import re
from jsonschema import validate, exceptions

import logging

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))


response_schema = {
    "type": "object",
    "properties": {
        "has_ce_marking": {"type": "boolean"},
        "has_ukca_marking": {"type": "boolean"},
    },
    "required": ["has_ce_marking", "has_ukca_marking"],
    "additionalProperties": False
}


class ClassResult(str, Enum):
    TP = 'true_positive'
    FN = 'false_negative'
    FP = 'false_positive'
    TN = 'true_negative'


def regulatory_doc_to_visual(doc):
    image = Image.open(io.BytesIO(doc["image"]))
    return [image.convert("RGB")]


def regulatory_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return lmms_eval_specific_kwargs['post_prompt']


def regulatory_doc_to_target(doc):
    return {
        "has_ce_marking": doc["has_ce_marking"],
        "has_ukca_marking": doc["has_ukca_marking"],
    }


def regulatory_process_results(doc, results):
    classification_results = dict()
    
    try:
        json_string = extract_json_string(results[0])
        preds = json.loads(json_string)
        validate(instance=preds, schema=response_schema)
        classification_results['json_validity_rate'] = 1
    except (json.decoder.JSONDecodeError, exceptions.ValidationError):
        classification_results['json_validity_rate'] = 0
        # Punish the model for invalid json
        preds = {
            "has_ce_marking": not doc["has_ce_marking"],
            "has_ukca_marking": not doc["has_ukca_marking"],
        }
    
    for label in preds:
        item_name = "_".join(label.split("_")[1:])
        class_result = classify_result(doc[label], preds[label])
        classification_results[f'{item_name}_precision'] = class_result
        classification_results[f'{item_name}_recall'] = class_result
        classification_results[f'{item_name}_f1'] = class_result
    return classification_results


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


def calculate_precision(results):
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


def json_validity_rate(results):
    return sum(results) / len(results)
