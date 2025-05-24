#!/bin/bash

# Clean up previous trainer outputs
if [ -d "trainer_output" ]; then
    echo "Removing existing trainer_output directory..."
    rm -rf trainer_output
fi

TASKS=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb" "wnli")

MODELS_DIR="Logs_bert"
MODELS=($(ls -d $MODELS_DIR/*/))



for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        model_base=$(basename "$model")
    
        echo "Running evaluation for model: $model_base on task: $task"
        
        python evals/bert_glue/run_glue.py \
            --model_name_or_path "$model" \
            --task_name "$task" \
            --do_train \
            --do_eval \
            --max_seq_length 512 \
            --per_device_train_batch_size 64 \
            --gradient_accumulation_steps 4 \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --max_eval_samples 500 \
            --torch_compile_backend="inductor" \
            --torch_compile_mode="reduce-overhead" \
            --output_dir "evals/bert_glue/results/$model_base/$task" \
            --overwrite_output_dir True
            
        echo "Completed evaluation for model: $model_base on task: $task"
        echo "----------------------------------------------------"
    done
done
