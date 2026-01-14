import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def set_seed(seed):
    '''Fix all of random seed for reproducible training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # only add when conv in your model


def get_n_params(model):
    '''Get the number of parameters of model'''
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def random_neq(l, r, s=[]):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def metric_report(data_rank, ks=[10, 20]):
    """
    Calculate NDCG and HR (Hit Rate) for multiple k values.
    In leave-one-out evaluation, Recall@k is equivalent to HR@k.

    Args:
        data_rank: List of predicted item ranks (0-indexed).
        ks: List of k values, defaults to [10, 20].

    Returns:
        Dictionary containing all evaluation metrics.
    """
    result = {}
    
    for k in ks:
        ndcg, hit_rate = 0.0, 0.0
        
        for rank in data_rank:
            if rank < k:
                hit_rate += 1
                ndcg += 1.0 / np.log2(rank + 2)
        
        num_samples = len(data_rank)
        if num_samples == 0:
            result[f'NDCG@{k}'] = 0.0
            result[f'HR@{k}'] = 0.0
        else:
            result[f'NDCG@{k}'] = ndcg / num_samples
            result[f'HR@{k}'] = hit_rate / num_samples
    
    return result

def format_metrics(metric_dict, ks=[10, 20]):
    """
    Format metrics into table form to match HR@k metrics.

    Args:
        metric_dict: Dictionary containing evaluation metrics (e.g., {'NDCG@10': ..., 'HR@10': ...})
        ks: List of k values, defaults to [10, 20]

    Returns:
        Formatted string table.
    """
    header = f"{'':<12}{'NDCG':<12}{'HR':<12}"
    rows = []

    for k in ks:
        ndcg = metric_dict.get(f'NDCG@{k}', 0.0)
        hr = metric_dict.get(f'HR@{k}', 0.0)
        
        row = f"@{k:<11}{ndcg:<12.6f}{hr:<12.6f}"
        rows.append(row)
    
    return header + "\n" + "\n".join(rows)

def record_csv(args, res_dict):
    """
    Record experiment results to CSV file.
    This function is robust and can handle any metric dictionary passed in.
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    record_file = f"metrics_summary.csv"
    csv_path = os.path.join(output_dir, record_file)

    try:
        parts = args.train_data_path.split('/')
        relevant_part = '_'.join(parts[-3:])
        model_identifier = f"{args.model_name}_{relevant_part}"
    except Exception:
        model_identifier = f"{args.model_name}_unknown_{getattr(args, 'now_str', '')}"

    record_data = {"model_identifier": model_identifier}
    record_data.update(res_dict)

    columns = ["model_identifier"] + sorted([k for k in res_dict.keys()])

    new_df = pd.DataFrame([record_data])

    if not os.path.exists(csv_path):
        new_df = new_df.reindex(columns=columns)
        new_df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
        combined_df = pd.concat([df, new_df], ignore_index=True)
        final_columns = df.columns.union(new_df.columns)
        combined_df = combined_df.reindex(columns=final_columns)
        combined_df.to_csv(csv_path, index=False)

def unzip_data(data, aug=True, aug_num=0):

    res = []
    
    if aug:
        for user in tqdm(data):

            user_seq = data[user]
            seq_len = len(user_seq)

            for i in range(aug_num+2, seq_len+1):
                
                res.append(user_seq[:i])
    else:
        for user in tqdm(data):

            user_seq = data[user]
            res.append(user_seq)

    return res

def concat_data(data_list):
    """
    Safely concatenate train, valid, test datasets.
    Handles cases where users may only exist in some datasets.
    """
    res = []

    if len(data_list) not in [2, 3]:
        raise ValueError("concat_data expects a list of 2 or 3 data dictionaries.")

    train = data_list[0]
    valid = data_list[1]
    test = data_list[2] if len(data_list) == 3 else {}

    for user in train:
        train_seq = train.get(user, [])
        valid_seq = valid.get(user, [])
        test_seq = test.get(user, [])

        full_seq = train_seq + valid_seq + test_seq

        if full_seq:
            res.append(full_seq)

    return res