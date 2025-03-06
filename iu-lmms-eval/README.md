

## Evaluation

### Supported Benchmarks

Given the user has access to `/mnt/nushare2/`, we currently support the following benchmarks:

- Open Source benchmarks
```
textvqa_test,ai2d,mme,mmbench_en_dev,mmstar,hallusion_bench_image,cvbench,mmmu_val,scienceqa
```

### Krylov Workspace
To run the evaluation in workspace use [eval_lmms_workspace.sh](krylov/scripts/eval_lmms_workspace.sh).

- Based on your task accordingly modify the `TASK` (you can list multiple together like `mmmu_val,mmbench_en_dev,mme`) `CKPT_PATH`, `OUTPUT_PATH` and `MODEL_NAME`.
    - The `MODEL_NAME` needs to be one of the supported models from [here](lmms_eval/models).
- Some models require `conversation_templates` or other details. Check [here](README_lmms.md) for a more comprehensive explanation. 

### Krylov Workflow
To run the evaluation in workflow use [eval_lmms_workflow.sh](krylov/scripts/eval_lmms_workflow.sh), which can be used directly by bashing [run_eval.sh](run_eval.sh) and submitting a job.


#### Usage example
- To evaluate `InternVL2.5-8B` on `mme`, modify either `krylov/scripts/eval_lmms_workspace.sh` or `krylov/scripts/eval_lmms_workflow.sh` (depending on the evaluation location) with:
    ```
    TASK=mme
    CKPT_PATH=/mnt/nushare2/data/mnulli/model_zoos/opensource-vlms/models--OpenGVLab--InternVL2_5-8B/snapshots/d64b85a1392275381ddbb7525db05e587303d59e
    MODEL_NAME=internvl2

    OUTPUT_PATH=/mnt/nushare2/data/mnulli/llava_ov/playground/lmms_eval_results/$TASK_SUFFIX/$MODEL_NAME

    accelerate launch --num_machines $NUM_MACHINES --num_processes $NUM_GPUS --main_process_port $PORT --mixed_precision no --dynamo_backend no \
        lmms_eval/__main__.py \
        --model $MODEL_NAME \
        --model_args pretrained=$CKPT_PATH \
        --tasks $TASK \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $TASK_SUFFIX \
        --verbosity='DEBUG' \
        --output_path $OUTPUT_PATH
    ```

## Adding a new task:
A quick set-up guide is provided [here](docs/task_guide.md), below we highlight the main components of the process and some of the standard problems faced.
1. First create a new folder [here](lmms_eval/tasks), with the name `<task_name>`.
2. Create a yaml file and a utils.py (look [here](lmms_eval/tasks/ebench_sm) for an example). 
    If your file is has a .parquet extension and it is named 'test.parquet', then here are first lines for your config:
    ```
    dataset_path: "parquet"
    dataset_kwargs:
        data_dir: <suffix of the path, without mount pvc part>
    test_split: "test"
    task: <task_name>
    doc_to_visual: !function utils.ebench_sm_doc_to_visual
    doc_to_text: !function utils.ebench_sm_doc_to_text_mc
    doc_to_target: "answer"
    ...
    ...
    process_results: !function utils.ebench_sm_process_results
    metric_list:
    - metric: string_matching_accuracy
        aggregation: !function utils.ebench_sm_string_matching_acc
        higher_is_better: true
    ```
    
3. Important:
    - `<task_name>` should be the same as the one inserted in yaml
    - the `!function` is used to call user defined functions in the utils.py file from the same directory.
    - Metrics can be user defined. Meaning that the name `string_matching_accuracy` needs to match the metric name defined within the utils.py file. If this does not happen you will not able to see the results of your evaluation, as, effectively the function `utils.ebench_sm_string_matching_acc` will never be called. 

    (excursus) If u are curious, as me, why `dataset_path` is actually `parquet` here pls look into  [HF documentation](https://github.com/huggingface/datasets/blob/fb91fd3c9ea91a818681a777faf8d0c46f14c680/src/datasets/load.py#L1973).

## Adding a new model:
Refer to this guide [here](docs/model_guide.md).


## Issues
For any inquiries and issues feel free to open a pull request
