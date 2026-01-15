import argparse
import os
import torch
import numpy as np
import json
import pandas as pd
from tqdm import tqdm 
from collections import defaultdict

from DualviewIdentification.Identification.poison_detector import SimplifiedPoisonDetector, SimplifiedDetectionConfig
from DualviewIdentification.Identification.detection_evaluator import SimplifiedDetectionEvaluator
from DualviewIdentification.Identification.data_loader import (
    load_sequences_and_ground_truth,
    load_sequences_from_inter_file
)
from DualviewIdentification.DualviewConstruct.dual_view_models import DualViewSASRec, DualViewGRU4Rec, DualViewBERT4Rec

def load_model_from_checkpoint(checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'args' in checkpoint:
        config = checkpoint['args']
        model_state_dict = checkpoint.get('state_dict', checkpoint) 
    elif 'config' in checkpoint:
        config = checkpoint['config']
        model_state_dict = checkpoint.get('model_state_dict')
    
    class Args: pass
    args = Args()
    for key, value in config.items():
        setattr(args, key, value)
    
    model_map = {'sasrec': DualViewSASRec, 'gru4rec': DualViewGRU4Rec, 'bert4rec': DualViewBERT4Rec}
    if args.model_name not in model_map:
        raise ValueError(f"Unknown model name '{args.model_name}' in checkpoint config.")
    
    model = model_map[args.model_name](args.user_num, args.item_num, device, args)
    
    model.load_state_dict(model_state_dict)
    model.to(device).eval()
    
    print(f"Successfully loaded model '{args.model_name}' with saved configuration.")
    return model, args

def compute_item_popularity(sequences):
    item_counts = defaultdict(int)
    total_interactions = 0
    for seq in sequences:
        for item in seq:
            item_counts[item] += 1
        total_interactions += len(seq)
    
    return {item: count / total_interactions for item, count in item_counts.items()} if total_interactions > 0 else {}

def main():
    parser = argparse.ArgumentParser(description="Run Dual-View Poison Detection and Tagging")
    
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint.pt file.')
    
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., ml-1m). This should match the dataset used for training the model.')
    parser.add_argument('--clean_data_path', type=str, required=True, help='Path to clean data for computing statistics.')
    parser.add_argument('--tagged_data_path', type=str, required=True, help='Path to the _tagged.txt file, serving as both test data and ground truth.')
    parser.add_argument('--val_data_path', type=str, default=None, help='(Optional) Path to a separate validation _tagged.txt file for tuning.')
    
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for detection.')
    parser.add_argument('--sliding_window_size', type=int, default=11, help='Sliding window size for score smoothing.')
    parser.add_argument('--detection_threshold', type=float, default=0.5, help='Initial threshold for detection.')
    parser.add_argument('--auto_tune', action='store_true', help='Automatically tune threshold on a validation set.')
    parser.add_argument('--tune_metric', type=str, default='f1', choices=['f1', 'precision', 'recall'], help='Metric to optimize during auto-tuning.')

    parser.add_argument('--output_dir', type=str, default='./detection_results', help='Base directory for detection results.')
    parser.add_argument('--tagged_output_path', type=str, default=None, help='Path to save the output file with suspicious tags.')
    
    cli_args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, model_config_obj = load_model_from_checkpoint(cli_args.checkpoint_path, device)
    
    final_config = vars(model_config_obj)
    
    for key, value in vars(cli_args).items():
        final_config[key] = value

    class FinalArgs:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    final_args = FinalArgs(final_config)

    print("\n--- Final Running Configuration ---")
    print(f"Model: {final_args.model_name}, Hidden Size: {getattr(final_args, 'hidden_size', 'N/A')}, Trm Nums: {getattr(final_args, 'trm_num', 'N/A')}")
    print(f"Dataset: {final_args.dataset}")
    print(f"Detection Threshold (Initial): {final_args.detection_threshold}, Auto-tune: {final_args.auto_tune}")
    if final_args.auto_tune:
        print(f"Auto-tune metric: {final_args.tune_metric.upper()}")
    print("-------------------------------------\n")
    
    
    output_dir = os.path.join(final_args.output_dir, final_args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading test data and ground truth...")
    test_sequences, test_user_ids, ground_truth = load_sequences_and_ground_truth(final_args.tagged_data_path)
    print(f"Loaded {len(test_sequences)} test sequences and ground truth for {len(ground_truth)} users.")
    
    print("\nLoading clean data for popularity stats...")
    clean_sequences, _ = load_sequences_from_inter_file(final_args.clean_data_path)
    item_popularity = compute_item_popularity(clean_sequences)
    
    config = SimplifiedDetectionConfig(
        sliding_window_size=final_args.sliding_window_size,
        detection_threshold=final_args.detection_threshold
    )
    
    detector = SimplifiedPoisonDetector(
        clean_model=model,
        item_popularity=item_popularity,
        config=config,
        device=device
    )
    
    if final_args.auto_tune:
        print("\nPreparing validation set for auto-tuning...")
        if final_args.val_data_path:
            val_sequences, val_user_ids, val_ground_truth = load_sequences_and_ground_truth(final_args.val_data_path)
            print(f"Using separate validation set: {final_args.val_data_path}")
        else:
            val_size = min(2000, len(test_sequences) // 5)
            val_sequences, val_user_ids = test_sequences[:val_size], test_user_ids[:val_size]
            val_ground_truth = {uid: ground_truth.get(uid, {}) for uid in val_user_ids}
            print(f"Using first {val_size} samples from test set for validation.")
        
        val_labels = [val_ground_truth.get(uid, {}).get('is_poisoned', False) for uid in val_user_ids]
        detector.auto_tune_threshold(val_sequences, val_labels, val_user_ids, final_args.tune_metric)
    
    print(f"\nStarting detection with final threshold: {detector.config.detection_threshold:.3f}")
    detection_results = detector.detect_sequences(test_sequences, test_user_ids, final_args.batch_size)
    
    results_path = os.path.join(output_dir, 'detection_results.json')
    results_to_save = [{'user_id': int(r.user_id), 'is_poisoned': bool(r.is_poisoned), 
                        'confidence': float(r.confidence), 'poisoned_positions': r.poisoned_positions}
                       for r in detection_results]
    with open(results_path, 'w') as f: json.dump(results_to_save, f, indent=2)
    print(f"\nDetection results saved to {results_path}")
    
    print("\nEvaluating detection performance...")
    evaluator = SimplifiedDetectionEvaluator()
    metrics = evaluator.evaluate_detection(detection_results, ground_truth)
    evaluator.print_evaluation_report(metrics)
    
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f: json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to {metrics_path}")

    if final_args.tagged_output_path:
        print(f"\nGenerating tagged output file at: {final_args.tagged_output_path}")
        suspicious_positions_map = defaultdict(set)
        for result in detection_results:
            if result.is_poisoned:
                suspicious_positions_map[result.user_id] = set(result.poisoned_positions)

        try:
            test_df = pd.read_csv(final_args.tagged_data_path, sep='\t')
        except Exception as e:
            print(f"Error re-reading tagged data file: {e}")
            return
            
        tags = []
        user_interactions = test_df.groupby('user_id', sort=False)
        for user_id, group in tqdm(user_interactions, desc="Tagging suspicious items"):
            suspicious_pos = suspicious_positions_map.get(user_id, set())
            for i in range(len(group)):
                tags.append(1 if i in suspicious_pos else 0)
        
        if len(tags) != len(test_df):
             raise ValueError(f"Mismatch in length between tags ({len(tags)}) and DataFrame rows ({len(test_df)}).")
        
        test_df['suspicious_tag:token'] = tags
        
        output_dir_for_tag = os.path.dirname(final_args.tagged_output_path)
        if output_dir_for_tag: os.makedirs(output_dir_for_tag, exist_ok=True)
        test_df.to_csv(final_args.tagged_output_path, sep='\t', index=False)
        print("Tagged output file successfully generated.")

    print("\nDetection and Tagging process completed!")


if __name__ == '__main__':
    main()
