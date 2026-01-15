import torch
import torch.nn as nn
import numpy as np
import os
from train_models.models.BaseModel import BaseSeqModel

class DualViewBaseModel(BaseSeqModel):
    """
    Base class for Dual-View Sequential Modeling.
    Integrates pre-trained semantic and collaborative item embeddings.
    """
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        
        self.args = args
        self.hidden_size = args.hidden_size
        self.device = device
        
        self._load_pretrained_embeddings()
        self._init_semantic_view()
        self._init_collaborative_view()
        self._init_prediction_heads()
        
        # Standard Recommendation Loss
        self.bce_loss_func = nn.BCEWithLogitsLoss()
        
        # Initialize weights via base class or internal logic
        self._init_weights()

    def _load_pretrained_embeddings(self):
        """Load and adapt multi-view item representations."""
        # Load Semantic Embeddings
        sem_path = os.path.join(self.args.embedding_dir, 'semantic', self.args.dataset, f'{self.args.dataset}_llama_embeddings.npy')
        sem_raw = np.load(sem_path)
        
        if sem_raw.shape[0] != self.item_num:
             sem_raw = self._adjust_embedding_size(sem_raw, self.item_num)

        # Extend for [PAD] (0) and [MASK] (item_num + 1)
        sem_table = np.zeros((self.item_num + 2, sem_raw.shape[1]), dtype=np.float32)
        sem_table[1 : self.item_num + 1] = sem_raw
        self.semantic_item_emb = nn.Embedding.from_pretrained(torch.from_numpy(sem_table), freeze=self.args.freeze_semantic_emb)
        
        # Load Collaborative Embeddings
        pca_path = os.path.join(self.args.embedding_dir, 'collaborative', self.args.dataset, f'{self.args.dataset}_pca{self.args.pca_dim}_embeddings.npy')
        pca_raw = np.load(pca_path)
        
        if pca_raw.shape[0] != self.item_num:
             pca_raw = self._adjust_embedding_size(pca_raw, self.item_num)

        pca_table = np.zeros((self.item_num + 2, pca_raw.shape[1]), dtype=np.float32)
        pca_table[1 : self.item_num + 1] =
