import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

# --- Configuration ---
LLAMA_MODEL_PATH = "../llama_model"
PROCESSED_DATA_PATH_PREV = './data/processed/' 
EMBEDDING_OUTPUT_PATH = './data/semantic_embeddings/'
BATCH_SIZE = 128
MAX_TEXT_LENGTH = 512

os.makedirs(EMBEDDING_OUTPUT_PATH, exist_ok=True)

def load_llama_model(model_path):
    print(f"Loading Llama model from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    except Exception as e:
        print(f"Error loading Llama model: {e}")
        print("Please ensure the model path is correct and all necessary files are present.")
        raise

    if tokenizer.pad_token is None:
        print("Tokenizer does not have a PAD token. Adding one.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        print("Added PAD token and resized model embeddings.")

    model.eval()
    print("Llama model loaded successfully.")
    return tokenizer, model

def get_llama_embeddings_batch(texts, tokenizer, model, device):
    if not texts:
        return []
    
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=MAX_TEXT_LENGTH, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        attention_mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        embeddings = sum_embeddings / sum_mask
    return embeddings.cpu().numpy()

def parse_attributes_from_list(attribute_list):
    parsed = {}
    if not isinstance(attribute_list, list):
        return parsed
    for attr_str in attribute_list:
        if not isinstance(attr_str, str) or ':' not in attr_str:
            continue
        try:
            key, value = attr_str.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key not in parsed:
                parsed[key] = []
            parsed[key].append(value)
        except ValueError:
            continue
    return parsed

def generate_text_for_beauty(item_id, attributes_list):
    parsed_attrs = parse_attributes_from_list(attributes_list)
    description_parts = []
    if parsed_attrs.get("brand"):
        description_parts.append(f"brand is {', '.join(parsed_attrs['brand'])}")
    if parsed_attrs.get("cat"):
        description_parts.append(f"categories are {', '.join(parsed_attrs['cat'])}")
    other_attrs = [f"{key} is {', '.join(values)}" for key, values in parsed_attrs.items() if key not in ["brand", "cat"]]
    if other_attrs:
        description_parts.append("other features include: " + "; ".join(other_attrs))
    if not description_parts: return f"Beauty item {item_id} with no specific attributes provided."
    return f"The beauty item {item_id} has attributes: {'; '.join(description_parts)}."

def generate_text_for_yelp(item_id, attributes_list):
    parsed_attrs = parse_attributes_from_list(attributes_list)
    description_parts = []
    if parsed_attrs.get("name"): description_parts.append(f"name is {', '.join(parsed_attrs['name'])}")
    if parsed_attrs.get("cat"): description_parts.append(f"categories are {', '.join(parsed_attrs['cat'])}")
    if parsed_attrs.get("city"): description_parts.append(f"city is {', '.join(parsed_attrs['city'])}")
    if parsed_attrs.get("stars"): description_parts.append(f"rating is {', '.join(parsed_attrs['stars'])} stars")
    other_attrs = [f"{key} is {', '.join(values)}" for key, values in parsed_attrs.items() if key not in ["name", "cat", "city", "stars"]]
    if other_attrs: description_parts.append("details: " + "; ".join(other_attrs))
    if not description_parts: return f"Yelp business {item_id} with no attributes."
    return f"The Yelp business {item_id} has characteristics: {'; '.join(description_parts)}."

def generate_text_for_ml1m(item_id, attributes_list):
    """
    Generates a textual description for a MovieLens-1M item.
    'attributes_list' is a list of strings like ["genre:action", "genre:adventure"].
    """
    parsed_attrs = parse_attributes_from_list(attributes_list)
    description_parts = []

    # For ML-1M, the main attribute is 'genre' (from `genre:Animation` in item2attributes)
    if parsed_attrs.get("genre"):
        description_parts.append(f"genres are {', '.join(parsed_attrs['genre'])}")
    
    # Add any other attributes if they exist 
    other_attrs = []
    for key, values in parsed_attrs.items():
        if key not in ["genre"]:
            other_attrs.append(f"{key} is {', '.join(values)}")
    if other_attrs:
        description_parts.append("other details: " + "; ".join(other_attrs))

    if not description_parts:
        return f"Movie {item_id} with no specific genre information." # item_id here is the original ID from id_map
        
    return f"The movie {item_id} has the following attributes: {'; '.join(description_parts)}."

# --- Main Processing Function ---
def process_dataset_embeddings(dataset_name, text_generation_func, tokenizer, model, device):
    print(f"\n===== Generating embeddings for {dataset_name} =====")

    dataset_processed_path = os.path.join(PROCESSED_DATA_PATH_PREV, dataset_name)
    item2attr_file = os.path.join(dataset_processed_path, f'{dataset_name}.item2attributes.json')
    id_map_file = os.path.join(dataset_processed_path, f'{dataset_name}.id_map.json')
    
    dataset_embedding_dir = os.path.join(EMBEDDING_OUTPUT_PATH, dataset_name)
    os.makedirs(dataset_embedding_dir, exist_ok=True)
    
    item_str_cache_file = os.path.join(dataset_embedding_dir, f'{dataset_name}_item_text_descriptions.json')
    # This pkl stores embeddings keyed by ORIGINAL item IDs
    item_emb_orig_id_cache_file = os.path.join(dataset_embedding_dir, f'{dataset_name}_item_emb_orig_id.pkl')
    final_emb_file = os.path.join(dataset_embedding_dir, f'{dataset_name}_llama_embeddings.npy') # final ordered by NEW ID

    if not os.path.exists(item2attr_file):
        print(f"ERROR: item2attributes file not found for {dataset_name} at {item2attr_file}. Skipping.")
        return
    if not os.path.exists(id_map_file):
        print(f"ERROR: id_map file not found for {dataset_name} at {id_map_file}. Skipping.")
        return

    print(f"Loading item attributes from {item2attr_file}")
    with open(item2attr_file, 'r') as f:
        item_attributes_remapped = json.load(f) # Keys are NEW remapped IDs (strings)
    
    print(f"Loading ID map from {id_map_file}")
    with open(id_map_file, 'r') as f:
        id_map = json.load(f) # Contains 'id2item': {"new_id_str": "original_id_str"}

    item_texts_orig_id = {}
    missing_attributes_for_items_in_map = [] # For debugging

    if os.path.exists(item_str_cache_file):
        print(f"Loading cached item text descriptions from {item_str_cache_file}")
        with open(item_str_cache_file, 'r') as f:
            item_texts_orig_id = json.load(f)
    else:
        print("Generating textual descriptions for items...")
        for new_id_str, orig_id_str in tqdm(id_map['id2item'].items(), desc="Generating item texts"):
            if new_id_str in item_attributes_remapped:
                attributes_for_item = item_attributes_remapped[new_id_str]
                generated_text = text_generation_func(orig_id_str, attributes_for_item)
                if generated_text and generated_text.strip(): # Ensure text is not empty
                    item_texts_orig_id[orig_id_str] = generated_text
                # else:
                #     print(f"Warning: Empty text generated for orig_id {orig_id_str} (new_id {new_id_str}).")
            else:
                # This item is in id_map (meaning it was in interactions) but has no attributes.
                missing_attributes_for_items_in_map.append(orig_id_str)
        
        if missing_attributes_for_items_in_map:
            print(f"Warning: {len(missing_attributes_for_items_in_map)} items were in id_map but "
                  f"had no entry in item2attributes.json. First 5: {missing_attributes_for_items_in_map[:5]}")

        with open(item_str_cache_file, 'w') as f:
            json.dump(item_texts_orig_id, f, indent=2)
        print(f"Saved item text descriptions to {item_str_cache_file}")

    if not item_texts_orig_id:
        print(f"No item texts generated or loaded for {dataset_name}. Cannot proceed.")
        return
        
    item_embeddings_orig_id = {}
    if os.path.exists(item_emb_orig_id_cache_file):
        print(f"Loading cached item embeddings (orig_id keys) from {item_emb_orig_id_cache_file}")
        with open(item_emb_orig_id_cache_file, 'rb') as f:
            item_embeddings_orig_id = pickle.load(f)
    
    items_to_embed_orig_ids = []
    texts_to_embed = []
    for orig_id, text in item_texts_orig_id.items(): 
        if orig_id not in item_embeddings_orig_id:
            items_to_embed_orig_ids.append(orig_id)
            texts_to_embed.append(text)

    if texts_to_embed:
        print(f"Generating embeddings for {len(texts_to_embed)} items for {dataset_name}...")
        for i in tqdm(range(0, len(texts_to_embed), BATCH_SIZE), desc="Embedding batches"):
            batch_texts = texts_to_embed[i:i+BATCH_SIZE]
            batch_orig_ids = items_to_embed_orig_ids[i:i+BATCH_SIZE]
            batch_embeddings_np = get_llama_embeddings_batch(batch_texts, tokenizer, model, device)
            for idx, orig_id in enumerate(batch_orig_ids):
                item_embeddings_orig_id[orig_id] = batch_embeddings_np[idx]
        
        print(f"Saving item embeddings (orig_id keys) to {item_emb_orig_id_cache_file}")
        with open(item_emb_orig_id_cache_file, 'wb') as f:
            pickle.dump(item_embeddings_orig_id, f)
    else:
        print("All item embeddings loaded from cache or no new items to embed (based on available texts).")

    if not item_embeddings_orig_id:
        print(f"No embeddings generated or loaded (keyed by original_id) for {dataset_name}. Cannot proceed.")
        return

    max_new_id = 0
    if not id_map.get('id2item'):
        print("Error: 'id2item' not found in id_map.json. Cannot determine embedding array size.")
        return
        
    for new_id_str in id_map['id2item'].keys():
        try:
            max_new_id = max(max_new_id, int(new_id_str))
        except ValueError:

            continue
            
    if max_new_id == 0:
        print("Error: Could not determine the number of items (max_new_id is 0) from id_map. Skipping final embedding array.")
        return

    # Get embedding dimension
    if not item_embeddings_orig_id:
        print("Error: item_embeddings_orig_id is empty after processing. Cannot determine embedding dimension.")
        return
    first_valid_orig_id = next(iter(item_embeddings_orig_id)) # Get first key from dict
    embedding_dim = item_embeddings_orig_id[first_valid_orig_id].shape[0]
    
    # Final array is 0-indexed, corresponding to new_id 1 to max_new_id
    final_embeddings_np = np.zeros((max_new_id, embedding_dim), dtype=np.float32)
    
    print(f"Reordering embeddings. Final array size: ({max_new_id}, {embedding_dim})")
    num_placed_embeddings = 0
    items_in_map_but_no_embedding_cache = []

    for new_id_numeric in range(1, max_new_id + 1):
        new_id_str = str(new_id_numeric)
        if new_id_str in id_map['id2item']:
            original_item_id = id_map['id2item'][new_id_str]
            if original_item_id in item_embeddings_orig_id:
                final_embeddings_np[new_id_numeric - 1] = item_embeddings_orig_id[original_item_id]
                num_placed_embeddings +=1
            else:
                items_in_map_but_no_embedding_cache.append(original_item_id)
        # else:


    print(f"Successfully placed {num_placed_embeddings} embeddings into the final ordered array.")
    
    total_missing_in_final_array = max_new_id - num_placed_embeddings
    if total_missing_in_final_array > 0:
        print(f"Warning: {total_missing_in_final_array} positions in the final array are zeros.")
        if items_in_map_but_no_embedding_cache:
            print(f"  ({len(items_in_map_but_no_embedding_cache)} of these had an entry in id_map but "
                  f"no text/embedding was generated/cached. First 5: {items_in_map_but_no_embedding_cache[:5]})")


    np.save(final_emb_file, final_embeddings_np)
    print(f"Saved final ordered embeddings to {final_emb_file}")
    print(f"Shape of saved embeddings: {final_embeddings_np.shape}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(LLAMA_MODEL_PATH):
        print(f"ERROR: Llama model path not found: {LLAMA_MODEL_PATH}")
        exit()
    if not os.path.isdir(PROCESSED_DATA_PATH_PREV):
        print(f"ERROR: Processed data path not found: {PROCESSED_DATA_PATH_PREV}")
        exit()


    tokenizer, model = load_llama_model(LLAMA_MODEL_PATH)
    
    process_dataset_embeddings(
        dataset_name='ml-1m', 
        text_generation_func=generate_text_for_ml1m,
        tokenizer=tokenizer,
        model=model,
        device=device
    )
    
    process_dataset_embeddings(
        dataset_name='amazon-beauty',
        text_generation_func=generate_text_for_beauty,
        tokenizer=tokenizer,
        model=model,
        device=device
    )

    process_dataset_embeddings(
        dataset_name='yelp2018',
        text_generation_func=generate_text_for_yelp,
        tokenizer=tokenizer,
        model=model,
        device=device
    )

    print("\nAll specified datasets processed for semantic embeddings.")
