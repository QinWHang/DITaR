#!/bin/bash

# Config
MODEL_NAME="gru4rec"
DATASET="ml-1m" 
GPU_ID=0
SEED=43

# DATA_TYPE: 'clean', 'poisoned', 'retrain'
DATA_TYPE="retrain"

ATTACK_SUFFIX="ur0.3_rep_i0.3_l2_c_r_sem_i0.3_swp_i0.3_w11" 

POISON_DIR_TYPE="triple" 

EPOCHS=200
BATCH_SIZE=256
HIDDEN_SIZE=64
MAX_LEN=200
DROPOUT=0.3  
PATIENCE=10

# Eval
EVAL_STEPS=5
WATCH_METRIC="NDCG@10"
EVAL_KS="10,20"

# GRU4Rec
NUM_LAYERS=2

# --- 3. SCRIPT LOGIC (DO NOT MODIFY BELOW) ---
echo "================================================="
echo "Starting experiment:"
echo "  Model:    $MODEL_NAME"
echo "  Dataset:  $DATASET"
echo "  Data Type: $DATA_TYPE"
[ "$DATA_TYPE" != "clean" ] && echo "  Attack:   $ATTACK_SUFFIX"
echo "================================================="

TRAIN_DATA_PATH=""
OUTPUT_DIR=""

if [ "$DATA_TYPE" == "clean" ]; then
    TRAIN_DATA_PATH="./data/processed/${DATASET}/${DATASET}.inter.txt"
    OUTPUT_DIR="./Outputs/train_clean/${DATASET}/${MODEL_NAME}"
elif [ "$DATA_TYPE" == "poisoned" ]; then
    if [ -z "$ATTACK_SUFFIX" ]; then
        echo "Error: ATTACK_SUFFIX must be set for DATA_TYPE 'poisoned'."
        exit 1
    fi
    TRAIN_FILE="${DATASET}_poisoned_${ATTACK_SUFFIX}.inter.txt"
    TRAIN_DATA_PATH="./data/poisoned/${POISON_DIR_TYPE}/${DATASET}/${TRAIN_FILE}"
    OUTPUT_DIR="./Outputs/train_poison/${DATASET}/${MODEL_NAME}/${ATTACK_SUFFIX}/"
elif [ "$DATA_TYPE" == "retrain" ]; then
    if [ -z "$ATTACK_SUFFIX" ]; then
        echo "Error: ATTACK_SUFFIX must be set for DATA_TYPE 'retrain'."
        exit 1
    fi
    TRAIN_FILE="${DATASET}_poisoned_${ATTACK_SUFFIX}_retrain.inter.txt"
    TRAIN_DATA_PATH="./data/unlearning/retrain/${DATASET}/${TRAIN_FILE}"
    OUTPUT_DIR="./Outputs_rebuttal/retrain/${DATASET}/${MODEL_NAME}/${ATTACK_SUFFIX}/"
else
    echo "Error: Unknown DATA_TYPE '$DATA_TYPE'."
    exit 1
fi

# check
if [ ! -f "$TRAIN_DATA_PATH" ]; then
    echo "Error: Training data file not found at '$TRAIN_DATA_PATH'"
    exit 1
fi

CMD="python train_models/main.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --gpu_id $GPU_ID \
    --seed $SEED \
    --train_data_path \"$TRAIN_DATA_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --num_train_epochs $EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --max_len $MAX_LEN \
    --dropout_rate $DROPOUT \
    --num_layers $NUM_LAYERS \
    --patience $PATIENCE \
    --watch_metric $WATCH_METRIC \
    --eval_steps $EVAL_STEPS \
    --eval_ks \"$EVAL_KS\" \
    --log"

echo "Executing command:"
echo "$CMD"
eval "$CMD"

echo "Experiment finished."
