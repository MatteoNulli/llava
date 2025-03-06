#!/bin/bash
set -euo pipefail

model="$1"
model_args="$2"
task_name="$3"

if [ -n "${KRYLOV_WF_HOME:-}" ]; then
    cd "$KRYLOV_WF_HOME/src/$KRYLOV_WF_TASK_NAME"
elif [ -n "${KRYLOV_WS_NAME:-}" ]; then
    mkdir -p ./logs
    # Exclude user site packages when running in the workspace to match Docker image environment
    export PYTHONNOUSERSITE=1
fi

accelerate launch \
    --num_processes=1 \
    -m lmms_eval \
    --model "$model" \
    --model_args "$model_args" \
    --tasks "$task_name" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${model}_${task_name}" \
    --output_path ./logs/
