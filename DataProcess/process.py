import os
import json
import gzip
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
from tqdm import tqdm
import ast

USER_CORE = 5
ITEM_CORE = 5
RATING_THRESHOLD = 0.0
RAW_DATA_PATH = '../SequentialRecData/'
PROCESSED_DATA_PATH = './data/processed/'
DEFAULT_ATTRIBUTE = ["attr:unknown"]

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def parse_gz_json(path):
    if not os.path.exists(path):
        print(f"Error: file {path} does not exist.")
        return []
    g = gzip.open(path, 'rb')
    data_list = []
    skipped_lines = 0
    file_basename = os.path.basename(path)
    for i, l in enumerate(tqdm(g, desc=f"Reading {file_basename}")):
        try:
            decoded_line = l.decode('utf-8').strip()
            if not decoded_line: skipped_lines +=1; continue
            data_list.append(json.loads(decoded_line))
        except json.JSONDecodeError:
            try: data_list.append(ast.literal_eval(decoded_line))
            except (ValueError, SyntaxError, TypeError): skipped_lines += 1; continue
        except UnicodeDecodeError: skipped_lines += 1; continue
        except Exception: skipped_lines +=1; continue
    g.close()
    if skipped_lines > 0: print(f"File {file_basename} skipped {skipped_lines} unparseable lines.")
    return data_list

def load_ml1m(data_dir):
    print("Loading MovieLens-1M dataset...")
    ratings_file = os.path.join(data_dir, 'ratings.dat')
    movies_file = os.path.join(data_dir, 'movies.dat')
    data = []
    try:
        with open(ratings_file, 'r', encoding='iso-8859-1') as f:
            for line in tqdm(f, desc="Loading ml-1m ratings"):
                user_id, item_id, rating, timestamp = line.strip().split('::')
                if float(rating) >= RATING_THRESHOLD:
                    data.append((user_id, item_id, int(timestamp)))
    except FileNotFoundError: print(f"Error: file not found {ratings_file}"); return [], {}

    original_attributes = {}
    try:
        with open(movies_file, 'r', encoding='iso-8859-1') as f:
            for line in tqdm(f, desc="Loading ml-1m movie metadata"):
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    item_id, _, genres_str = parts
                    if genres_str and genres_str != '(no genres listed)':
                        original_attributes[item_id] = [f"genre:{g.strip().replace(' ', '_').lower()}" for g in genres_str.split('|') if g.strip()]
    except FileNotFoundError: print(f"Warning: movie metadata file {movies_file} not found.")

    print(f"Loaded {len(data)} interactions and {len(original_attributes)} items with attributes from metadata.")
    return data, original_attributes

def load_amazon_beauty(data_dir):
    print("Loading Amazon Beauty dataset...")
    reviews_file = os.path.join(data_dir, 'reviews_Beauty.json.gz')
    meta_file = os.path.join(data_dir, 'meta_Beauty.json.gz')

    data = []
    if os.path.exists(reviews_file):
        reviews = parse_gz_json(reviews_file)
        for review in tqdm(reviews, desc="Processing Beauty reviews"):
            if 'reviewerID' in review and 'asin' in review and 'unixReviewTime' in review and 'overall' in review:
                if float(review['overall']) >= RATING_THRESHOLD:
                    data.append((review['reviewerID'], review['asin'], int(review['unixReviewTime'])))
    else: print(f"Error: reviews file not found: {reviews_file}"); return [], {}

    original_attributes = {}
    if os.path.exists(meta_file):
        meta_data = parse_gz_json(meta_file)
        for item in tqdm(meta_data, desc="Processing Beauty metadata"):
            if 'asin' in item:
                current_item_attrs = []
                if 'brand' in item and item['brand'] and isinstance(item['brand'], str) and item['brand'].strip():
                    current_item_attrs.append(f"brand:{item['brand'].strip().replace(' ', '_').lower()}")
                cats_to_process = []
                if 'category' in item and item['category']:
                    if isinstance(item['category'], list): cats_to_process.extend(c for c in item['category'] if isinstance(c, str) and c.strip())
                    elif isinstance(item['category'], str) and item['category'].strip(): cats_to_process.append(item['category'])
                if 'categories' in item and item['categories']:
                    if isinstance(item['categories'], list):
                        for cat_list in item['categories']:
                            if isinstance(cat_list, list): cats_to_process.extend(c for c in cat_list if isinstance(c, str) and c.strip())
                            elif isinstance(cat_list, str) and cat_list.strip(): cats_to_process.append(cat_list)
                if cats_to_process:
                    processed_cats = [f"cat:{c.strip().replace(' ', '_').lower()}" for c in list(set(cats_to_process)) if c.strip()]
                    current_item_attrs.extend(processed_cats)

                if current_item_attrs:
                    original_attributes[item['asin']] = list(set(current_item_attrs))
    else: print(f"Warning: metadata file {meta_file} not found.")

    print(f"Loaded {len(data)} interactions and {len(original_attributes)} items with attributes from metadata.")
    return data, original_attributes

def load_yelp2018(data_dir):
    print("Loading Yelp2018 dataset...")
    reviews_file = os.path.join(data_dir, 'yelp_academic_dataset_review.json')
    business_file = os.path.join(data_dir, 'yelp_academic_dataset_business.json')
    data = []
    try:
        with open(reviews_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Yelp reviews"):
                try:
                    review = json.loads(line)
                    if 'user_id' in review and 'business_id' in review and 'stars' in review and 'date' in review:
                        if float(review['stars']) >= RATING_THRESHOLD:
                            dt_object = datetime.strptime(review['date'].split(" ")[0], '%Y-%m-%d')
                            data.append((review['user_id'], review['business_id'], int(dt_object.timestamp())))
                except json.JSONDecodeError: continue
    except FileNotFoundError: print(f"Error: file not found {reviews_file}"); return [], {}

    original_attributes = {}
    try:
        with open(business_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Yelp business metadata"):
                try:
                    business = json.loads(line)
                    current_item_attrs = []
                    if 'business_id' in business:
                        if 'categories' in business and business['categories'] and isinstance(business['categories'], str):
                            categories_str = business['categories']
                            cats = [f"cat:{cat.strip().replace(' ', '_').lower()}" for cat in categories_str.split(',') if cat.strip()]
                            if cats: current_item_attrs.extend(list(set(cats)))
                        if 'city' in business and business['city'] and isinstance(business['city'], str) and business['city'].strip():
                            current_item_attrs.append(f"city:{business['city'].strip().replace(' ', '_').lower()}")
                        if 'stars' in business and isinstance(business['stars'], (int, float)):
                            current_item_attrs.append(f"stars:{business['stars']}")

                        if current_item_attrs:
                             original_attributes[business['business_id']] = list(set(current_item_attrs))
                except json.JSONDecodeError: continue
    except FileNotFoundError: print(f"Warning: business metadata file {business_file} not found.")

    print(f"Loaded {len(data)} interactions and {len(original_attributes)} items with attributes from metadata.")
    return data, original_attributes

def filter_k_core(interactions, user_core=USER_CORE, item_core=ITEM_CORE):
    print(f"Starting K-core filtering (user>={user_core}, item>={item_core})...")
    iteration = 0
    while True:
        iteration += 1
        user_count = defaultdict(int); item_count = defaultdict(int)
        for user, item, _ in interactions: user_count[user] += 1; item_count[item] += 1
        initial_count = len(interactions)
        if initial_count == 0: print("Warning: interaction data is empty at K-core filtering start!"); return [], set()
        interactions = [(u, i, t) for u, i, t in interactions if user_count[u] >= user_core and item_count[i] >= item_core]
        if len(interactions) == initial_count:
            print(f"K-core filtering complete (after {iteration} iterations), retained {len(interactions)} interactions")
            retained_items = {item for _, item, _ in interactions} if interactions else set()
            print(f"Retained {len(retained_items)} items after K-core filtering (original IDs)")
            break
        if not interactions: print(f"Warning: K-core filtering removed all data at iteration {iteration}!"); return [], set()
    return interactions, retained_items

def get_user_sequences(interactions):
    user_seq = defaultdict(list)
    for user, item, timestamp in interactions: user_seq[user].append((item, timestamp))
    for user in user_seq: user_seq[user].sort(key=lambda x: x[1])
    return dict(user_seq)

def remap_ids(user_sequences):
    print("Remapping IDs...")
    user2id, item2id = {}, {}
    all_users = sorted(list(user_sequences.keys()))
    all_items_set = set(item for interactions_list in user_sequences.values() for item, _ in interactions_list)
    all_items = sorted(list(all_items_set))

    for idx, u in enumerate(all_users): user2id[u] = idx + 1
    for idx, i in enumerate(all_items): item2id[i] = idx + 1

    id2user = {v: k for k, v in user2id.items()}
    id2item = {v: k for k, v in item2id.items()}

    remapped_sequences = {}
    for user_orig, interactions in user_sequences.items():
        new_user_id = user2id[user_orig]
        remapped_interactions = [(item2id[item_orig], timestamp) for item_orig, timestamp in interactions if item_orig in item2id]
        if remapped_interactions: remapped_sequences[new_user_id] = remapped_interactions

    data_maps = {'user2id': user2id, 'item2id': item2id, 'id2user': id2user, 'id2item': id2item}
    n_users, n_items = len(id2user), len(id2item)
    n_interactions = sum(len(seq) for seq in remapped_sequences.values())
    print(f"Remapping complete: {n_users} users, {n_items} items, {n_interactions} interactions")
    return remapped_sequences, data_maps, (n_users, n_items, n_interactions)

def process_item_attributes(original_attributes: dict,
                            retained_items_orig_ids: set,
                            item2id_map: dict):
    print("Processing item attributes (ensuring all retained items have entries)...")

    remapped_attributes = {}
    attribute_counts = Counter()
    num_items_with_found_attrs = 0
    num_items_with_default_attrs = 0
    attribute_lens = []

    for original_id, new_numeric_id in item2id_map.items():
        if original_id not in retained_items_orig_ids:
            continue

        current_item_final_attrs = []
        attrs_found = original_attributes.get(original_id)

        if attrs_found and isinstance(attrs_found, list) and attrs_found:
            current_item_final_attrs = list(set(attrs_found))
            num_items_with_found_attrs += 1
        else:
            current_item_final_attrs = DEFAULT_ATTRIBUTE.copy()
            num_items_with_default_attrs += 1

        remapped_attributes[str(new_numeric_id)] = current_item_final_attrs
        attribute_lens.append(len(current_item_final_attrs))
        for attr in current_item_final_attrs:
            attribute_counts[attr] += 1

    print(f"Processed attributes for {len(remapped_attributes)} items in total.")
    print(f"  {num_items_with_found_attrs} items found with original attributes.")
    print(f"  {num_items_with_default_attrs} items assigned default attributes ('{DEFAULT_ATTRIBUTE[0]}').")
    print(f"Found {len(attribute_counts)} distinct attribute values (including default).")

    if attribute_lens:
        print(f"Item attribute statistics after processing:")
        print(f"  Attribute count per item: min={min(attribute_lens)}, max={max(attribute_lens)}, avg={np.mean(attribute_lens):.2f}")
    else:
        print("No valid item attributes were processed.")

    return remapped_attributes

def save_data(dataset_name, user_sequences, data_maps, item_attributes):
    """Save data with .inter.txt file globally sorted by user_id and timestamp"""
    output_dir = os.path.join(PROCESSED_DATA_PATH, dataset_name)
    ensure_dir(output_dir)

    all_interactions = []
    for user_id, interactions in user_sequences.items():
        for item_id, timestamp in interactions:
            all_interactions.append((user_id, item_id, timestamp))

    interactions_df = pd.DataFrame(all_interactions, columns=['user_id', 'item_id', 'timestamp'])

    print("Globally sorting all interactions (by user_id, timestamp)...")
    interactions_df.sort_values(by=['user_id', 'timestamp'], inplace=True)

    inter_file_name = f'{dataset_name}.inter.txt'
    inter_file_path = os.path.join(output_dir, inter_file_name)

    interactions_df.rename(columns={
        'user_id': 'user_id:token',
        'item_id': 'item_id:token',
        'timestamp': 'timestamp:float'
    }, inplace=True)

    interactions_df.to_csv(inter_file_path, sep='\t', index=False)
    print(f"Sorted interaction file saved: {inter_file_path}")

    seq_file_name = f'{dataset_name}.seq.txt'
    seq_file_path = os.path.join(output_dir, seq_file_name)
    with open(seq_file_path, 'w') as f:
        f.write("user_id:token\titem_id_sequence:token_seq\n")
        for user_id in sorted(user_sequences.keys()):
            interactions = user_sequences[user_id]
            items_only = [str(item_id) for item_id, _ in interactions]
            f.write(f"{user_id}\t{' '.join(items_only)}\n")
    print(f"Sorted sequence file saved: {seq_file_path}")

    map_file_name = f'{dataset_name}.id_map.json'
    map_file_path = os.path.join(output_dir, map_file_name)
    with open(map_file_path, 'w') as f:
        json.dump(data_maps, f, indent=4)

    item2attributes_file_name = f'{dataset_name}.item2attributes.json'
    item2attributes_file_path = os.path.join(output_dir, item2attributes_file_name)
    with open(item2attributes_file_path, 'w') as f:
        json.dump(item_attributes, f, indent=4)

    print(f"ID mapping and attribute files saved.")
    print(f"All data saved to {output_dir}")

def calculate_stats(user_sequences, n_total_items_after_remap):
    if not user_sequences: print("\n--- Dataset Statistics: User sequences empty ---"); return
    user_lens = [len(seq) for seq in user_sequences.values()]
    item_counts = Counter(item_id for seq in user_sequences.values() for item_id, _ in seq)
    item_lens_interacted = list(item_counts.values()) if item_counts else [0]
    n_users, n_items_interacted = len(user_sequences), len(item_counts)
    n_interactions = sum(user_lens)
    sparsity = 1.0 - (n_interactions / (n_users * n_total_items_after_remap)) if n_users * n_total_items_after_remap > 0 else 1.0
    print("\n--- Dataset Statistics ---")
    print(f"Number of users: {n_users}, Number of items (total after K-core): {n_total_items_after_remap}")
    print(f"Number of items (actually appearing in final sequences): {n_items_interacted}, Number of interactions: {n_interactions}")
    print(f"Average user interaction sequence length: {np.mean(user_lens):.2f}" if user_lens else "N/A")
    print(f"Average item interaction count (interacted items): {np.mean(item_lens_interacted):.2f}" if item_lens_interacted else "N/A")
    print(f"Sparsity: {sparsity:.6f}\n----------------------")

def process_dataset(dataset_name, load_function, data_dir_param, user_k=USER_CORE, item_k=ITEM_CORE):
    print(f"\n===== Processing {dataset_name} dataset =====")
    interactions, original_attributes = load_function(data_dir_param)
    if not interactions: print(f"{dataset_name} has no interaction data, skipping"); return

    filtered_interactions, retained_items_orig_ids = filter_k_core(interactions, user_k, item_k)
    if not filtered_interactions: print(f"{dataset_name} is empty after K-core filtering, skipping"); return

    user_sequences = get_user_sequences(filtered_interactions)
    if not user_sequences: print(f"{dataset_name} is empty after generating user sequences, skipping"); return

    remapped_sequences, data_maps, stats = remap_ids(user_sequences)
    if not remapped_sequences: print(f"{dataset_name} is empty after remapping IDs, skipping"); return
    n_users_final, n_items_final, _ = stats

    processed_attrs = process_item_attributes(original_attributes, retained_items_orig_ids, data_maps['item2id'])

    calculate_stats(remapped_sequences, n_items_final)
    save_data(dataset_name, remapped_sequences, data_maps, processed_attrs)
    print(f"===== {dataset_name} dataset processing complete =====\n")

def main():
    ensure_dir(PROCESSED_DATA_PATH)
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: raw data path {RAW_DATA_PATH} does not exist."); return

    ml1m_dir = os.path.join(RAW_DATA_PATH, 'ml-1m')
    if os.path.isdir(ml1m_dir):
        process_dataset('ml-1m', load_ml1m, ml1m_dir)
    else: print(f"MovieLens-1M directory not found: {ml1m_dir}, skipping.")

    amazon_dir = os.path.join(RAW_DATA_PATH, 'Amazon_Beauty')
    if os.path.isdir(amazon_dir):
        process_dataset('amazon-beauty', load_amazon_beauty, amazon_dir)
    else: print(f"Amazon Beauty directory not found: {amazon_dir}, skipping.")

    yelp_dir = os.path.join(RAW_DATA_PATH, 'yelp2018')
    if os.path.isdir(yelp_dir):
        process_dataset('yelp2018', load_yelp2018, yelp_dir)
    else: print(f"Yelp2018 directory not found: {yelp_dir}, skipping.")

    print("All dataset processing complete")

if __name__ == "__main__":
    main()
