import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
import random
import pickle
import faiss

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class SequentialDataPoisoner:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.input_dir = os.path.join(args.input_dir, args.dataset)
        self.output_dir = os.path.join(args.output_dir, args.dataset)
        self.embedding_dir = args.embedding_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.user_ratio = args.user_ratio
        self.poison_types = args.poison_types
        self.rng = np.random.default_rng(args.seed)

        self.user_sequences = {}
        self.item_popularity = Counter()
        self.id2item = {}
        self.dissimilarity_index = None
        self.poison_records = []

    def load_data(self):
        print(f"Loading data for dataset {self.dataset_name}...")
        inter_file = os.path.join(self.input_dir, f"{self.dataset_name}.inter.txt")
        df = pd.read_csv(inter_file, sep='\t', dtype={'user_id:token': int, 'item_id:token': int, 'timestamp:float': float})
        df.rename(columns={'user_id:token': 'user_id', 'item_id:token': 'item_id', 'timestamp:float': 'timestamp'}, inplace=True)

        for user_id, group in tqdm(df.groupby('user_id'), desc="Organizing user sequences"):
            sorted_group = group.sort_values('timestamp', ascending=True)
            self.user_sequences[user_id] = list(zip(sorted_group['item_id'], sorted_group['timestamp']))

        self.item_popularity.update(df['item_id'])
        self.total_interactions = len(df)
        self.all_item_ids = sorted(self.item_popularity.keys())
        self.n_items = len(self.all_item_ids)

        map_file = os.path.join(self.input_dir, f"{self.dataset_name}.id_map.json")
        with open(map_file, 'r') as f:
            id_maps = json.load(f)
            self.id2item = {int(k): v for k, v in id_maps['id2item'].items()}

        print(f"Loading complete: {len(self.user_sequences)} users, {self.n_items} items, {self.total_interactions} interactions.")

    def _get_filename_suffix(self):
        parts = [f"ur{self.user_ratio}"]
        if 'repeat' in self.poison_types:
            parts.append(f"rep_i{self.args.repeat_intensity}_l{self.args.repeat_length}_{self.args.repeat_item_type[0]}_{self.args.repeat_mode[0]}")
        if 'semantic' in self.poison_types:
            parts.append(f"sem_i{self.args.semantic_intensity}")
        if 'swap' in self.poison_types:
            parts.append(f"swp_i{self.args.swap_intensity}_w{self.args.swap_window_size}")
        return "_".join(parts)
    
    def apply_repeat_attack(self, user_id, user_seq):
        seq_len = len(user_seq)
        if seq_len < 2 or self.args.repeat_length <= 0:
            return user_seq, []

        n_to_poison = max(1, int(seq_len * self.args.repeat_intensity))

        user_item_pops = {item_id: self.item_popularity[item_id] for item_id, _ in user_seq}
        sorted_user_items = sorted(user_item_pops.keys(), key=lambda i: user_item_pops[i])

        if self.args.repeat_item_type == 'hot':
            candidate_items = sorted_user_items[::-1]
        elif self.args.repeat_item_type == 'cold':
            candidate_items = sorted_user_items
        else:
            candidate_items = list(user_item_pops.keys())
            self.rng.shuffle(candidate_items)

        if not candidate_items:
            return user_seq, []

        poisoned_seq = list(user_seq)
        records = []
        attacked_indices = set()

        num_attacks_done = 0
        for item_to_repeat in candidate_items:
            if num_attacks_done >= n_to_poison:
                break

            possible_start_positions = [i for i, (item, _) in enumerate(poisoned_seq) if item == item_to_repeat]
            self.rng.shuffle(possible_start_positions)

            for start_pos in possible_start_positions:
                if self.args.repeat_mode == 'replace' and start_pos + self.args.repeat_length >= seq_len:
                    continue
                if any(i in attacked_indices for i in range(start_pos, start_pos + self.args.repeat_length + 1)):
                    continue

                new_records = self._execute_repeat_operation(
                    poisoned_seq, start_pos, item_to_repeat, seq_len, user_id, attacked_indices
                )
                
                if new_records:
                    records.extend(new_records)
                    num_attacks_done += 1
                    attacked_indices.add(start_pos)
                    break

        return poisoned_seq, records

    def _build_or_load_dissimilarity_index(self):
        index_path = os.path.join(self.output_dir, f"{self.dataset_name}_dissimilarity_index_{self.n_items}.pkl")

        if os.path.exists(index_path):
            print(f"Loading dissimilarity index from cache: {index_path}")
            with open(index_path, 'rb') as f:
                self.dissimilarity_index = pickle.load(f)
            return

        print("Building dissimilarity index...")
        emb_path = os.path.join(self.embedding_dir, self.dataset_name, f"{self.dataset_name}_llama_embeddings.npy")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Semantic embedding file not found: {emb_path}")

        embeddings = np.load(emb_path).astype('float32')
        if embeddings.shape[0] != self.n_items:
             print(f"Warning: embedding count ({embeddings.shape[0]}) != item count ({self.n_items}).")

        self.dissimilarity_index = self._compute_dissimilarity_from_embeddings(embeddings)

        with open(index_path, 'wb') as f:
            pickle.dump(self.dissimilarity_index, f)
        print(f"Dissimilarity index saved to: {index_path}")

    def apply_semantic_attack(self, user_id, user_seq):
        if self.dissimilarity_index is None:
            self._build_or_load_dissimilarity_index()

        seq_len = len(user_seq)
        if seq_len < 1:
            return user_seq, []

        n_to_poison = max(1, int(seq_len * self.args.semantic_intensity))
        poison_indices = self.rng.choice(seq_len, size=n_to_poison, replace=False)

        poisoned_seq = list(user_seq)
        records = []

        for pos in poison_indices:
            original_item, ts = poisoned_seq[pos]

            replacement_item = self._find_dissimilar_replacement(original_item)

            if replacement_item and replacement_item != original_item:
                poisoned_seq[pos] = (replacement_item, ts)
                records.append({
                    'user_id': user_id, 
                    'position': pos, 
                    'original_item': original_item,
                    'poisoned_item': replacement_item, 
                    'attack_type': 'semantic', 
                    'timestamp': ts
                })

        return poisoned_seq, records

    def apply_swap_attack(self, user_id, user_seq):
        seq_len = len(user_seq)
        w_size = self.args.swap_window_size
        if seq_len < 3 or w_size <= 2:
            return user_seq, []

        n_to_poison = max(1, int(seq_len * self.args.swap_intensity))
        possible_indices = list(range(seq_len))
        self.rng.shuffle(possible_indices)

        poisoned_seq = list(user_seq)
        records = []
        attacked_indices = set()

        for pos1 in possible_indices:
            if len(records) >= n_to_poison * 2:
                break
            if pos1 in attacked_indices:
                continue

            start = max(0, pos1 - w_size // 2)
            end = min(seq_len, pos1 + w_size // 2 + 1)

            candidates = []
            for pos2 in range(start, end):
                if abs(pos1 - pos2) > 1 and pos2 not in attacked_indices:
                    candidates.append(pos2)

            if candidates:
                pos2 = self.rng.choice(candidates)
                
                swap_records = self._execute_swap_operation(poisoned_seq, pos1, pos2, user_id, attacked_indices)
                
                if swap_records:
                    records.extend(swap_records)

        return poisoned_seq, records

    def run(self):
        self.load_data()

        all_users = list(self.user_sequences.keys())
        self.rng.shuffle(all_users)
        n_poison_users = int(len(all_users) * self.user_ratio)
        poisoned_user_ids = all_users[:n_poison_users]

        user_attack_map = {}
        if self.poison_types:
            for i, user_id in enumerate(poisoned_user_ids):
                user_attack_map[user_id] = self.poison_types[i % len(self.poison_types)]

        final_sequences = {}
        all_poison_records = []

        print("Starting poisoning...")
        for user_id, seq in tqdm(self.user_sequences.items(), desc="Applying attacks"):
            attack_type = user_attack_map.get(user_id)
            new_seq, records = seq, []

            if attack_type == 'repeat':
                new_seq, records = self.apply_repeat_attack(user_id, seq)
            elif attack_type == 'semantic':
                new_seq, records = self.apply_semantic_attack(user_id, seq)
            elif attack_type == 'swap':
                new_seq, records = self.apply_swap_attack(user_id, seq)

            final_sequences[user_id] = new_seq
            all_poison_records.extend(records)

        self.poison_records = all_poison_records
        self.save_poisoned_data(final_sequences)

    def save_poisoned_data(self, final_sequences):
        print("\nSaving poisoned data...")
        suffix = self._get_filename_suffix()

        poison_lookup = {(rec['user_id'], rec['position']): rec for rec in self.poison_records}

        inter_rows = []
        tagged_rows = []

        for user_id, seq in tqdm(final_sequences.items(), desc="Generating output files"):
            for pos, (item_id, ts) in enumerate(seq):
                inter_rows.append(f"{user_id}\t{item_id}\t{ts}\n")

                key = (user_id, pos)
                if key in poison_lookup:
                    rec = poison_lookup[key]
                    original_item = rec['original_item'] if rec['original_item'] is not None else ""
                    tagged_rows.append(f"{user_id}\t{item_id}\t{ts}\t1\t{rec['attack_type']}\t{original_item}\t{pos}\n")
                else:
                    tagged_rows.append(f"{user_id}\t{item_id}\t{ts}\t0\t\t\t{pos}\n")

        inter_filename = os.path.join(self.output_dir, f"{self.dataset_name}_poisoned_{suffix}.inter.txt")
        with open(inter_filename, 'w') as f:
            f.write("user_id:token\titem_id:token\ttimestamp:float\n")
            f.writelines(inter_rows)
        print(f"Poisoned interaction file saved: {inter_filename}")

        tagged_filename = os.path.join(self.output_dir, f"{self.dataset_name}_poisoned_{suffix}_tagged.txt")
        with open(tagged_filename, 'w') as f:
            f.write("user_id\titem_id\ttimestamp\tis_poisoned\tattack_type\toriginal_item\tposition\n")
            f.writelines(tagged_rows)
        print(f"Tagged poisoned file saved: {tagged_filename}")

        records_filename = os.path.join(self.output_dir, f"{self.dataset_name}_poisoned_{suffix}_records.json")
        with open(records_filename, 'w') as f:
            json.dump(self.poison_records, f, indent=2, cls=NumpyEncoder)
        print(f"Poison record details saved: {records_filename}")

        print("\n--- Poisoning Statistics ---")
        poisoned_user_count = len(set(rec['user_id'] for rec in self.poison_records))
        total_user_count = len(self.user_sequences)
        poisoned_inter_count = len(self.poison_records)
        attack_counts = Counter(rec['attack_type'] for rec in self.poison_records)

        print(f"Total users: {total_user_count}")
        print(f"Affected users: {poisoned_user_count} ({poisoned_user_count/total_user_count:.2%})")
        print(f"Total interactions: {self.total_interactions}")
        final_total_inter = sum(len(s) for s in final_sequences.values())
        print(f"Total interactions after poisoning: {final_total_inter}")
        print(f"Modified/added interactions: {poisoned_inter_count} ({poisoned_inter_count/final_total_inter:.2%})")
        print(f"Modified records by attack type: {dict(attack_counts)}")
        print("------------------\n")

def main():
    parser = argparse.ArgumentParser(description='Poisoning attack on sequential recommendation data')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--input_dir', type=str, default='./data/processed')
    parser.add_argument('--output_dir', type=str, default='./data/poisoned')
    parser.add_argument('--embedding_dir', type=str, default='./data/semantic_embeddings')
    parser.add_argument('--poison_types', nargs='+', required=True)
    parser.add_argument('--user_ratio', type=float, default=0.3)
    parser.add_argument('--repeat_intensity', type=float, default=0.3)
    parser.add_argument('--repeat_length', type=int, default=2)
    parser.add_argument('--repeat_item_type', type=str, default='cold', choices=['hot', 'cold', 'random'])
    parser.add_argument('--repeat_mode', type=str, default='replace', choices=['replace', 'append'])
    parser.add_argument('--semantic_intensity', type=float, default=0.3)
    parser.add_argument('--swap_intensity', type=float, default=0.3)
    parser.add_argument('--swap_window_size', type=int, default=11)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    
    poisoner = SequentialDataPoisoner(args)
    poisoner.run()

if __name__ == "__main__":
    main()
