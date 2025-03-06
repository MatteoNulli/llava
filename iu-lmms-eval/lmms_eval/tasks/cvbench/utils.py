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
    try:
        return [doc["image"].convert("RGB")]
    except:
        # print('\n Opening the image in a different way... \n Image is probably in bytes, string or different format. \n')
        # byts = ast.literal_eval(doc['image'])['bytes']
        return [Image.open(io.BytesIO(doc['image']['bytes'])).convert('RGB')]



def cvbench_doc_to_text_mc(doc, lmms_eval_specific_kwargs=None):
    question = doc['prompt']
    post = lmms_eval_specific_kwargs['post_prompt']
    return question + ' ' + post + " \nAnswer: "


def cvbench_process_results(doc, results):
    """
    Processes string matching results and stores necessary information for metric computation.
    """
    pred = results[0]
    gt = doc["answer"]
    source = doc["source"]
    return {
        "string_matching_accuracy": {"prediction": pred, "answer": gt, "source": source}
        }




def retrieve_special_characters(text):
    # First, look for a single character between brackets
    bracket_match = re.search(r'\((.*?)\)', text)
    if bracket_match:
        return bracket_match.group(0)  # or group(1) if you don't want the brackets
    
    # If no brackets found, look for standalone A,B,C,D,E
    standalone_match = re.search(r'\s[A-E]\s', text)
    if standalone_match:
        return standalone_match.group().strip()  # strip to remove the spaces
    
    return None  # Return None if no match found


def calculate_accuracy(results, source):
    #removing noise from outputs
    results_sourced = [result for result in results if result['source'] == source]
    
    all_elements = len(results_sourced)
    pos = 0
    for result in results_sourced:
        pred = retrieve_special_characters(result['prediction'])

        if pred == result['answer']:    
            pos += 1

    accuracy = pos/all_elements
    
    return accuracy

def combine_accuracies(results):
    # Calculate accuracy for each source
    accuracy_2d_ade = calculate_accuracy(results, 'ADE20K')
    accuracy_2d_coco = calculate_accuracy(results, 'COCO')
    accuracy_3d_omni = calculate_accuracy(results, 'Omni3D')

    # Calculate the accuracy for each type
    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni

    # Compute the combined accuracy as specified
    combined_accuracy = (accuracy_2d + accuracy_3d) / 2


    eval_logger.info(f"nyu-cvbench accuracy:", combined_accuracy)
    
    return combined_accuracy 
