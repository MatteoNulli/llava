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
    for pred, ground in zip(df.prediction, df.answer):
        if pred == ground:
            pos += 1

    ebench_acc = pos/all_elements
    print(f"mmbench {args.experiment} accuracy:", ebench_acc)
    return ebench_acc


if __name__ == "__main__":
    args = get_args()

    df = pd.read_table(args.annotation_file)

    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    cur_df.insert(6, 'prediction', None)
    for pred in open(os.path.join(args.result_dir, f"{args.experiment}.jsonl")):
        # print('pred', pred)
        pred = json.loads(pred)
        # print('pred after split', pred.split("text")[1].split("options")[0])
        # if args.llama3_2:
        #     question_id = pred.split("question_id")[1].split("round_id")[0].strip('":, ')
        #     pred = pred.split("text")[1].split("options")[0]
        #     question_id = int(question_id)
        # else:
        # try:
        #     
        # except:
        #     print('pred load faild', pred)
        #     continue
        # print('prediction', pred)
        # print(cur_df)
        # print(df['index'])
        # print(int(question_id))
        # if question_id in df['index']:
        #     print('yes')
        question_id = int(pred['question_id'])
        pred = pred['text']
        
        cur_df.loc[df['index'] == question_id, 'prediction'] = pred
        # print(cur_df)

    print(calculate_acc(cur_df))
    cur_df.to_csv(os.path.join(args.upload_dir, f"{args.experiment}.csv"), index=False) 
    print(f'File saved at {os.path.join(args.upload_dir, f"{args.experiment}.csv")}')
    # cur_df.to_excel(os.path.join(args.upload_dir, f"{args.experiment}.xlsx"), index=False, engine='openpyxl')
