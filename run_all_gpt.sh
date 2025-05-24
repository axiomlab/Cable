#!/bin/bash

LOG_FILE="run_all_logs_$(date +%Y%m%d_%H%M%S).txt"
echo "Experiment started at $(date)" | tee -a "$LOG_FILE"

# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# NPROC_PER_NODE=8
PYTHON_SCRIPT="src/train_gpt.py"
RESULT_PATH="Logs/"

DATASET_DIRS=("data/wikitext-103" "fineweb-edu-10B")
MODELS=("tiny" "small" "medium")
POS_METHODS=("cable" "cable5" "cable6" "cable7" "kcable" "kcable5" "kcable6" "rotali" "fire" "kerple" "alibi" "alibi2" "rope" "t5bias" "sinusoidal" "learnable")

USE_DAPE=(false)
SEQ_LENGTHS=(1024)



run_experiment() {
    local model=$1  
    local pos_method=$2
    local use_dape=$3
    local dataset=$4
    local seq_len=$5
    

    echo "====================================================================" | tee -a "$LOG_FILE"
    echo "Starting experiment with:" | tee -a "$LOG_FILE"
    echo "Model: $model" | tee -a "$LOG_FILE"
    echo "Position method: $pos_method" | tee -a "$LOG_FILE"
    echo "Use DAPE: $use_dape" | tee -a "$LOG_FILE"
    echo "Dataset: $dataset" | tee -a "$LOG_FILE"
    echo "Sequence length: $seq_len" | tee -a "$LOG_FILE"
    echo "CUDA devices: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
    echo "Processes per node: $NPROC_PER_NODE" | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    

    if ! torchrun --nproc_per_node=$NPROC_PER_NODE \
        "$PYTHON_SCRIPT" \
        --model "$model" \
        --pos-method "$pos_method" \
        --use-dape "$use_dape" \
        --dataset-dir "$dataset" \
        --sequence-length "$seq_len" 2>&1 | tee -a "$LOG_FILE"; then
        
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOG_FILE"
        echo "Experiment failed:" | tee -a "$LOG_FILE"
        echo "Model: $model, Position method: $pos_method" | tee -a "$LOG_FILE"
        echo "Use DAPE: $use_dape, Dataset: $dataset" | tee -a "$LOG_FILE"
        echo "Sequence length: $seq_len" | tee -a "$LOG_FILE"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOG_FILE"
        return 1
    fi
    
    return 0
}


for dataset in "${DATASET_DIRS[@]}"; do

    if [[ "$dataset" == *"wikitext"* ]]; then
        dataset_name="wikitext-103"
        export CUDA_VISIBLE_DEVICES="0,1"  
        NPROC_PER_NODE=2
    else
        dataset_name="fineweb10B"
        export CUDA_VISIBLE_DEVICES="0,1,2,3"  
        NPROC_PER_NODE=4
    fi

    for model in "${MODELS[@]}"; do
        for pos_method in "${POS_METHODS[@]}"; do
            for use_dape in "${USE_DAPE[@]}"; do
                for seq_len in "${SEQ_LENGTHS[@]}"; do
                    if find Logs -type d -name "*${model}_${pos_method}_${dataset_name}*" | grep -q .; then
                        echo "Skipping: dataset=$dataset_name, model=$model, pos_method=$pos_method", use_dape=$use_dape, seq_len=$seq_len, Already trained | tee -a "$LOG_FILE"
                        continue
                    fi
                    run_experiment "$model" "$pos_method" "$use_dape" "$dataset" "$seq_len"
                done
            done
        done
    done
done

echo "All experiments completed at $(date)" | tee -a "$LOG_FILE"
