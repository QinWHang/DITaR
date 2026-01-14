# generator.py
import os
import time
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from train_models.generators.data import SeqDataset, BertRecTrainDataset, GRU4RecTrainDataset
from train_models.utils.utils import unzip_data, concat_data

class Generator(object):
    """
    Base data generator.
    - Validation and test sets are loaded from fixed location './data/test&eval/{dataset}/'.
    - Training set is dynamically created by reading the full interaction file specified by --train_data_path
      and removing the fixed validation/test items.
    """

    def __init__(self, args, logger, device):
        self.args = args
        self.dataset = args.dataset
        self.num_workers = args.num_workers
        self.bs = args.train_batch_size
        self.logger = logger
        self.device = device
        
        self.logger.info("Loading dataset with dynamic train set creation...")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Dataset is loaded: consume %.3f s" % (end - start))

    def _load_user_seq_from_file(self, file_path):
        """Load user sequence dictionary from inter.txt format file at specified path"""
        user_sequences = defaultdict(list)
        if not os.path.exists(file_path):
            self.logger.error(f"Data file not found: {file_path}")
            return dict(user_sequences)

        df = pd.read_csv(file_path, sep='\t')
        df.columns = [c.split(':')[0] for c in df.columns]
        df = df.sort_values(by=['user_id', 'timestamp'])
        
        for user_id, group in df.groupby('user_id'):
            user_sequences[user_id] = list(group['item_id'])
            
        return dict(user_sequences)

    def _load_dataset(self):
        """
        Final loading logic:
        1. Load validation and test sets from fixed clean files.
        2. Load full interactions from file specified by --train_data_path.
        3. Remove valid/test items from full interactions and validate length to generate final training set.
        """
        clean_base_path = f'./data/test&eval/{self.args.dataset}/'

        valid_path = os.path.join(clean_base_path, 'valid.txt')
        test_path = os.path.join(clean_base_path, 'test.txt')

        self.valid = self._load_user_seq_from_file(valid_path)
        self.test = self._load_user_seq_from_file(test_path)
        self.logger.info(f"Loaded fixed validation set for {len(self.valid)} users.")
        self.logger.info(f"Loaded fixed test set for {len(self.test)} users.")

        full_data_path = self.args.train_data_path
        self.logger.info(f"Loading full sequences from: {full_data_path}")
        full_user_sequences = self._load_user_seq_from_file(full_data_path)

        user_train = {}
        min_seq_len_for_train = 2

        for user_id, full_seq in full_user_sequences.items():
            valid_items = self.valid.get(user_id, [])
            test_items = self.test.get(user_id, [])
            num_to_remove = len(valid_items) + len(test_items)
            
            train_part = full_seq
            if len(full_seq) > num_to_remove:
                train_part = full_seq[:-num_to_remove]

            if len(train_part) >= min_seq_len_for_train:
                user_train[user_id] = train_part
        
        self.train = user_train
        self.logger.info(f"Dynamically created and filtered training set for {len(self.train)} users.")
        map_dir = f'./data/processed/{self.dataset}/'
        map_file_path = os.path.join(map_dir, f"{self.args.dataset}.id_map.json")
        try:
            with open(map_file_path, 'r') as f:
                id_maps = json.load(f)
                base_user_num = len(id_maps.get('user2id', {}))
                base_item_num = len(id_maps.get('item2id', {}))
        except FileNotFoundError:
            self.logger.error(f"FATAL: ID map file not found at {map_file_path}.")
            raise

        all_users = set(self.train.keys()) | set(self.valid.keys()) | set(self.test.keys())
        all_items = {item for seq in self.train.values() for item in seq} | \
                    {item for seq in self.valid.values() for item in seq} | \
                    {item for seq in self.test.values() for item in seq}
        
        max_user_in_data = max(all_users) if all_users else 0
        max_item_in_data = max(all_items) if all_items else 0

        self.user_num = max(base_user_num, max_user_in_data)
        self.item_num = max(base_item_num, max_item_in_data)

        self.logger.info(f"Final user_num used for model: {self.user_num}")
        self.logger.info(f"Final item_num used for model: {self.item_num} (Base from map: {base_item_num}, Max in data: {max_item_in_data})")

        if self.item_num > base_item_num:
            self.logger.warning(
                f"Max item ID in data ({self.item_num}) is larger than in id_map.json ({base_item_num}). "
                f"This is expected for some poisoning attacks (e.g., semantic) and has been handled by resizing the embedding layer."
            )

    def make_trainloader(self):
        """Create training data loader (SASRec format)"""
        train_dataset = unzip_data(self.train, aug=False)
        self.train_dataset = SeqDataset(train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                     sampler=RandomSampler(self.train_dataset),
                                     batch_size=self.bs,
                                     num_workers=self.num_workers)
        
        return train_dataloader

    def make_evalloader(self, test=False):
        """Create evaluation data loader"""
        if test:
            eval_dataset = concat_data([self.train, self.valid, self.test])
        else:
            eval_dataset = concat_data([self.train, self.valid])

        self.eval_dataset = SeqDataset(eval_dataset, self.item_num, self.args.max_len, self.args.test_neg)
        eval_dataloader = DataLoader(self.eval_dataset,
                                    sampler=SequentialSampler(self.eval_dataset),
                                    batch_size=100,
                                    num_workers=self.num_workers)
        
        return eval_dataloader
    
    def get_user_item_num(self):
        """Return user and item counts"""
        return self.user_num, self.item_num


class BERT4RecGenerator(Generator):
    """BERT4Rec data generator"""

    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)

    def make_trainloader(self):
        """Create BERT4Rec training data loader"""
        train_dataset = unzip_data(self.train, aug=False)
        self.train_dataset = BertRecTrainDataset(self.args, train_dataset, self.item_num, self.args.max_len)

        train_dataloader = DataLoader(self.train_dataset,
                                     sampler=RandomSampler(self.train_dataset),
                                     batch_size=self.bs,
                                     num_workers=self.num_workers)
        
        return train_dataloader


class GRU4RecGenerator(Generator):
    """GRU4Rec data generator"""

    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)

    def make_trainloader(self):
        """Create GRU4Rec training data loader"""
        train_dataset = unzip_data(self.train, aug=False)
        self.train_dataset = GRU4RecTrainDataset(train_dataset, self.item_num, self.args.max_len)

        train_dataloader = DataLoader(self.train_dataset,
                                     sampler=RandomSampler(self.train_dataset),
                                     batch_size=self.bs,
                                     num_workers=self.num_workers)
        
        return train_dataloader

