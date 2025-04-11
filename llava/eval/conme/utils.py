
import torch
from PIL import Image
import re


def t2i_score(
        sample,
        multiple_choices,
        correct_choice,
        t2i_model
):
    scores = dict()
    for choice in multiple_choices:
        score = t2i_model(images=[sample['image_path']], texts=[choice])
        scores[choice] = score.item()

    choice = max(scores, key=lambda key: scores[key])  # compare the scores of the two captions
    hit = int(choice == correct_choice)

    return hit, scores