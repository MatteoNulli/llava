import os
import json
import argparse
import pandas as pd
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    return parser.parse_args()


def post_processing(df):
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

        
        # return None

    # Apply the function to the 'prediction' column
    for i in range(len(df)):
        
        if df.loc[i, 'prediction'] is not None:
            
            # if i < 20:
            #     print('i', i)
            #     print('before', df.loc[i, 'prediction'])
            df.loc[i, 'prediction'] = retrieve_special_characters(df.loc[i, 'prediction'])
            # if i < 20:
            #     print('after', df.loc[i, 'prediction'])

    # df['prediction'] = df['prediction'].apply(retrieve_special_characters)
    
    return df
    # # print('before post process', df['prediction'])
    # # df['prediction'] = df['prediction'].str.extract(r'(\(.*?\))', expand=False)
    # df['prediction'] = df['prediction'].str.extract(r'(\(.*?\)|\b[a-c]\b)', expand=False)

    # # print('after post process', df['prediction'])
    # return df

# def calculate_acc(df):
#     ##removing noise from outputs
#     df = post_processing(df)
#     pos = 0
#     all_elements = len(df.prediction)
#     for pred, ground in zip(df.prediction, df.answer):
#         if pred == ground:
#             pos += 1

#     acc = pos/all_elements
#     print(f"nyu-cvbench {args.experiment} accuracy:", acc)
#     return acc


# Define a function to calculate accuracy for a given source

def calculate_accuracy(df, source):
    #removing noise from outputs
    df = post_processing(df)

    source_df = df[df['source'] == source]

    pos = 0
    # accuracy = source_df['result'].mean()  # Assuming 'result' is 1 for correct and 0 for incorrect
    all_elements = len(source_df.prediction)
    for i, (pred, ground) in enumerate(zip(source_df.prediction, source_df.answer)):
        # if i < 20:
        #     print('i', i, 'pred', pred, 'ground', ground)
        if pred == ground:
            # print('yes')
            pos += 1

    accuracy = pos/all_elements
    
    return accuracy

def combine_accuracies(df):
    # Calculate accuracy for each source
    accuracy_2d_ade = calculate_accuracy(df, 'ADE20K')
    accuracy_2d_coco = calculate_accuracy(df, 'COCO')
    accuracy_3d_omni = calculate_accuracy(df, 'Omni3D')

    # Calculate the accuracy for each type
    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni

    # Compute the combined accuracy as specified
    combined_accuracy = (accuracy_2d + accuracy_3d) / 2
    
    print(f"nyu-cvbench {args.experiment} accuracy:", combined_accuracy)
    return combined_accuracy 



if __name__ == "__main__":
    args = get_args()


    df = pd.read_parquet(args.annotation_file, engine='pyarrow')
    # print('df', df)

    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['image'])
    cur_df.insert(6, 'prediction', None)
    i = 0
    for pred in open(os.path.join(args.result_dir, f"{args.experiment}.jsonl")):
        pred = json.loads(pred)
        
        # print(cur_df)
        
        # print("pred['text']", pred['text'])
        # print(cur_df['answer'][i])
        # i+=1

        cur_df.loc[df['idx'] == pred['question_id'], 'prediction'] = pred['text']

    print(combine_accuracies(cur_df))
    
    cur_df.to_csv(os.path.join(args.upload_dir, f"{args.experiment}.csv"), index=False) 
    print(f'File saved at {os.path.join(args.upload_dir, f"{args.experiment}.csv")}')
    # cur_df.to_excel(os.path.join(args.upload_dir, f"{args.experiment}.xlsx"), index=False, engine='openpyxl')
