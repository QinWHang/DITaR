#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

DATASET="amazon-beauty"
ATTACK_SUFFIX="ur0.3_rep_i0.3_l2_c_r_sem_i0.3_swp_i0.3_w11"
DUAL_MODEL_ALPHA=0
MODEL_NAME="sasrec"  
TUNE_METRIC="f1" # Can be f1, precision, or recall
 
python DualviewIdnetification/identification/run_detection.py \
  --checkpoint_path ./Outputs/dual_view_model_outputs/${DATASET}_${MODEL_NAME}_dual_view_sw${DUAL_MODEL_ALPHA}/checkpoint.pt \
  --dataset ${DATASET} \
  --clean_data_path ./data/processed/${DATASET}/${DATASET}.inter.txt \
  --tagged_data_path ./data_/poisoned_last_final/triple/${DATASET}/${DATASET}_poisoned_${ATTACK_SUFFIX}_tagged.txt \
  --output_dir ./Outputs/detection_results/${DATASET}_${MODEL_NAME}_${ATTACK_SUFFIX}_${TUNE_METRIC} \
  --tagged_output_path ./Outputs/Suspicion_Poisoned/${DATASET}/${DATASET}_${MODEL_NAME}_${ATTACK_SUFFIX}_detect_suspect_poison.inter.txt \
  --batch_size 16 \
  --sliding_window_size 11 \
  --detection_threshold 0.475 \
  --tune_metric ${TUNE_METRIC} \
  --auto_tune \



