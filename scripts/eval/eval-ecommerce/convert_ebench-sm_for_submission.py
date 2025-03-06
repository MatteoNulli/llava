import os
import json
import argparse
import pandas as pd
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm
import os
import ssl
import urllib.request
from urllib.error import URLError
from getpass import getpass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    return parser.parse_args()


def simple_stem(word):
    """
    A basic stemming function.
    This is a simplified version of the Porter stemming algorithm.
    """
    # Step 1
    word = re.sub(r"(ies|ied)$", "i", word)
    word = re.sub(r"(us|ss)$", r"\1", word)
    word = re.sub(r"(s)$", "", word)

    # Step 2
    word = re.sub(r"(eed|eedly)$", "ee", word)
    word = re.sub(r"(ed|edly|ing|ingly)$", "", word)

    # Step 3
    word = re.sub(r"(ational)$", "ate", word)
    word = re.sub(r"(tional)$", "tion", word)
    word = re.sub(r"(alize)$", "al", word)

    # Step 4
    word = re.sub(r"(icate|iciti|ical)$", "ic", word)

    # Step 5
    word = re.sub(r"(ful|ness)$", "", word)
    word = re.sub(r"(ative|ize|ise)$", "", word)

    return word


def simple_lemmatize(word):
    """
    A very basic lemmatization function.
    This covers some common cases not handled by the stemmer.
    """
    lemma_rules = {"better": "good", "best": "good", "worse": "bad", "worst": "bad", "am": "be", "is": "be", "are": "be", "was": "be", "were": "be"}

    return lemma_rules.get(word, word)


def normalize_word(word):
    """Apply both stemming and lemmatization to a word."""
    stemmed = simple_stem(word)
    return simple_lemmatize(stemmed)


def normalize_string(s):
    """Normalize the string by converting to lowercase, removing punctuation, and applying normalization."""
    # Convert to lowercase and remove punctuation
    normalized = re.sub(r"[^\w\s]", "", str(s).lower())
    # Apply normalization to each word
    return " ".join(normalize_word(word) for word in normalized.split())


def string_matching_acc(df, n_errors):
    print("Processing dataframe...")

    # Apply normalization to both columns
    df["answer"] = df["answer"].apply(normalize_string)
    df["prediction"] = df["prediction"].apply(normalize_string)

    # Remove unwanted strings
    df = df[df["answer"].str.lower() != "<_s_k_i_p_>"]

    # Modified matching logic: check if answer appears within prediction
    # for _, row in df.iterrows():
    #     if row['answer'] in row['prediction']:
    #         print(f"Answer: {row['answer']}, Prediction: {row['prediction']}")
    #     else:
    #         print(f"Answer: {row['answer']}, Incorrect Prediction: {row['prediction']}")

    correct_predictions = sum(row["answer"] in row["prediction"] for _, row in df.iterrows())
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions

    print(f"Results for {args.experiment}")
    print(f"Number of errors:{n_errors}")
    print(f"Number of rows with exact string matching: {correct_predictions}")
    print(f"Total String Matching Accuracy: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":
    args = get_args()

    count = 0

    df = pd.read_table(args.annotation_file)

    cur_df = df.copy()
    cur_df = cur_df.drop(columns=["hint", "category", "source", "image", "comment", "l2-category"])
    cur_df.insert(5, "prediction", None)
    for line_number, pred in enumerate(open(os.path.join(args.result_dir, f"{args.experiment}.jsonl"))):

        try:
            # print(f'Processing line {line_number}')

            pred = json.loads(pred)
            pred["output"] = pred["output"].strip(".")
            # print('pred', pred)
            # print('pred', pred['output'], 'ans', pred['answer'])
            # print('pred', )

            # print(pred)
            cur_df.loc[df["index"] == pred["question_id"], "prediction"] = pred["output"]

        except json.JSONDecodeError as e:
            count += 1
            print(f"JSON decode error on line {line_number}: {e}")
            continue
        except KeyError as e:
            count += 1
            print(f"Missing key error on line {line_number}: {e}")
            continue
        except Exception as e:
            count += 1
            print(f"Unexpected error on line {line_number}: {type(e).__name__}: {e}")
            continue

    print(string_matching_acc(cur_df, count))

    cur_df.to_csv(os.path.join(args.upload_dir, f"{args.experiment}.csv"), index=False)
    print(f'File saved at {os.path.join(args.upload_dir, f"{args.experiment}.csv")}')

    # cur_df.to_excel(os.path.join(args.upload_dir, f"{args.experiment}.xlsx"), index=False, engine='openpyxl')
