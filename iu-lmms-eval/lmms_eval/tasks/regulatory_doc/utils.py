import os
import json
from PIL import Image
import io
from enum import Enum
from collections import Counter
import re
from collections import defaultdict
from jsonschema import validate, exceptions

import logging

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))

response_schema = {
    "type": "object",
    "properties": {
        "has_declaration": {"type": "boolean"},
        "has_address": {"type": "boolean"},
        "has_signature": {"type": "boolean"}
    },
    "required": ["has_declaration", "has_address", "has_signature"]
}

METRIC_NAME_TEMPLATES = [
    "{entity}_page_precision",
    "{entity}_page_recall",
    "{entity}_page_f1",
    "{entity}_doc_precision",
    "{entity}_doc_recall",
    "{entity}_doc_f1"
]


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
        "has_declaration": doc["has_declaration"],
        "has_address": doc["has_address"],
        "has_signature": doc["has_signature"],
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
            "has_declaration": not doc["has_declaration"],
            "has_address": not doc["has_address"],
            "has_signature": not doc["has_signature"],
        }

    for label in preds:
        item_name = label.split("_")[1]
        class_result = classify_result(doc[label], preds[label])
        for metric_name_template in METRIC_NAME_TEMPLATES:
            if 'doc' in metric_name_template:
                classification_results[metric_name_template.format(entity=item_name)] = {
                    'filename': doc['filename'],
                    'classification_result': class_result,
                    }
            else:
                classification_results[metric_name_template.format(entity=item_name)] = class_result

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
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0
    return f1_score


def calculate_aggregated_precision(results):
    grouped_results = group_results_by_doc(results)
    doc_results = [aggregate_classification_results(results) for results in grouped_results.values()]
    return calculate_precision(doc_results)


def calculate_aggregated_recall(results):
    grouped_results = group_results_by_doc(results)
    doc_results = [aggregate_classification_results(results) for results in grouped_results.values()]
    return calculate_recall(doc_results)


def calculate_aggregated_f1(results):
    grouped_results = group_results_by_doc(results)
    doc_results = [aggregate_classification_results(results) for results in grouped_results.values()]
    return calculate_f1(doc_results)


def aggregate_classification_results(results):
    if ClassResult.TP in results:
        return ClassResult.TP
    elif ClassResult.FN in results:
        return ClassResult.FN
    elif ClassResult.FP in results:
        return ClassResult.FP
    else:
        return ClassResult.TN


def group_results_by_doc(results):
    grouped_results = defaultdict(list)
    for result in results:
        filename = result['filename']
        grouped_results[filename].append(result['classification_result'])
    return grouped_results


def count_classifications(results):
    counts = Counter(results)
    for key in ClassResult:
        if key not in counts:
            counts[key] = 0
    return counts


def json_validity_rate(results):
    return sum(results) / len(results)
