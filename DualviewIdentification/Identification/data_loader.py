import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

def load_sequences_and_ground_truth(tagged_file_path: str) -> Tuple[List[List[int]], List[int], Dict[int, Dict]]:
    print(f"Loading sequences and ground truth from: {tagged_file_path}")
    try:
        df = pd.read_csv(tagged_file_path, sep='\t', dtype={'user_id': int, 'item_id': int})
    except Exception as e:
        print(f"Error reading tagged file: {e}")
        return [], [], {}

    required_cols = ['user_id', 'item_id', 'is_poisoned', 'attack_type', 'position']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Tagged file must contain columns: {required_cols}")

    sequences = []
    user_ids = []
    ground_truth = defaultdict(lambda: {'is_poisoned': False, 'positions': [], 'types': {}})

    grouped = df.groupby('user_id', sort=False)
    
    for user_id, group in tqdm(grouped, desc="Processing user sequences from tagged file"):
        user_ids.append(user_id)
        sequences.append(group['item_id'].tolist())
        
        poisoned_interactions = group[group['is_poisoned'] == 1]
        
        if not poisoned_interactions.empty:
            ground_truth[user_id]['is_poisoned'] = True
            
            positions = poisoned_interactions['position'].tolist()
            types = poisoned_interactions['attack_type'].tolist()
            
            ground_truth[user_id]['positions'] = positions
            ground_truth[user_id]['types'] = {pos: typ for pos, typ in zip(positions, types)}
    
    return sequences, user_ids, dict(ground_truth)


def load_sequences_from_inter_file(inter_file_path: str) -> Tuple[List[List[int]], List[int]]:
    print(f"Loading sequences from {inter_file_path} for statistical analysis...")
    try:
        df = pd.read_csv(inter_file_path, sep='\t', dtype={'user_id:token': int, 'item_id:token': int})
        df.rename(columns={'user_id:token': 'user_id', 'item_id:token': 'item_id'}, inplace=True)
    except Exception as e:
        print(f"Error reading inter file: {e}")
        return [], []

    sequences = []
    user_ids = []

    grouped = df.groupby('user_id', sort=False)

    for user_id, group in grouped:
        user_ids.append(user_id)
        sequences.append(group['item_id'].tolist())
    
    return sequences, user_ids
