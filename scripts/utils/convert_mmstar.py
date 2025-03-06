import os
import json
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    return parser.parse_args()


def calculate_acc(df):
    pos = 0
    all_elements = len(df.prediction)
    df['prediction'] = df['prediction'].str.strip('\n\n')
    df['prediction'] = df['prediction'].str.strip('.')
    df['prediction'] = df['prediction'].str.strip(' ')
    df['prediction'] = df['prediction'].str.strip('**')
    df['prediction'] = df['prediction'].str.split('Answer: ').str[-1]
    # df['prediction'] = df['prediction'].str.split('**Answer:**')[-1]
    # df['prediction'] = df['prediction'].str.split('*Answer:*').str[-1]
    # df['prediction'] = df['prediction'].str.split('**Correct Option:').str[-1]
    for pred, ground in zip(df.prediction, df.answer):
        # print('pred', pred, '\n', 'ground', ground)
        if pred == ground:
            # print('yes')
            pos += 1

    ebench_acc = pos/all_elements
    print(f"mmstar {args.experiment} accuracy:", ebench_acc)
    return ebench_acc


if __name__ == "__main__":
    args = get_args()

    df = pd.read_parquet(args.annotation_file)

    cur_df = df.copy()
    # cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    cur_df.insert(6, 'prediction', None)
    # print('pred', pred)

    for pred in open(os.path.join(args.result_dir, f"{args.experiment}.jsonl")):
        pred = json.loads(pred)
    
        # print('pred', pred['text'])
        # print(df['index'], pred['question_id'])
        # print('pred after split', pred.split("text")[1].split("options")[0])
        # if pred['model_id'] == '0':
        #     # print('hi')
        #     question_id = pred.split("question_id")[1].split("round_id")[0].strip('":, ')
        #     pred = pred.split("text")[1].split("options")[0]
        #     question_id = int(question_id)
        # else:
        question_id = int(pred['question_id'])
        pred = pred['text']
        
        cur_df.loc[df['index'] == question_id, 'prediction'] = pred


    # print(cur_df)
    print(calculate_acc(cur_df))
    cur_df.to_csv(os.path.join(args.upload_dir, f"{args.experiment}.csv"), index=False) 
    print(f'File saved at {os.path.join(args.upload_dir, f"{args.experiment}.csv")}')
    # cur_df.to_excel(os.path.join(args.upload_dir, f"{args.experiment}.xlsx"), index=False, engine='openpyxl')
