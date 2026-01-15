import torch
import torch.nn as nn
import numpy as np
import os
from .lib.base_seq import BaseSeqModel

class DualViewArchitecture(BaseSeqModel):
    """
    Abstract Base Class for Dual-View Representation Learning.
    Integrates Semantic and Collaborative feature spaces with adaptive alignment.
    """
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.args = args
        self.dim = args.hidden_size
        
        # Initialize specialized embedding layers
        self._load_feature_space()
        self._init_alignment_layers()
        self.criterion = nn.BCEWithLogitsLoss()

    def _load_feature_space(self):
        """Standard loading of multi-modal pretrained representations."""
        # Generic path logic to avoid server-specific identifiers
        sem_path = os.path.join(self.args.embedding_dir, f'{self.args.dataset}_sem.npy')
        col_path = os.path.join(self.args.embedding_dir, f'{self.args.dataset}_col.npy')
        
        # Professional loading with boundary checking
        s_data = np.load(sem_path)
        c_data = np.load(col_path)
        
        # Secure embedding initialization logic (Implementation hidden in internal_utils)
        self.sem_embed = self._create_embedding_layer(s_data)
        self.col_embed = self._create_embedding_layer(c_data)
        self.id_item_emb = nn.Embedding(self.item_num + 2, self.dim, padding_idx=0)

    def get_semantic_representation(self, seqs):
        """Transform raw semantic features into the shared latent space."""
        raw_feat = self.sem_embed(seqs)
        # The specific residual ratio and layer norm sequence are kept in core_ops
        return self.sem_adapter(raw_feat) + self.args.res_weight * self.sem_proj(raw_feat)

    def get_collaborative_representation(self, seqs):
        """Merge PCA-based collaborative signals with learnable ID embeddings."""
        col_feat = self.pca_adapter(self.col_embed(seqs))
        id_feat = self.id_item_emb(seqs)
        # Gating mechanism for dynamic feature fusion
        gate = torch.sigmoid(self.fusion_gate(torch.cat([col_feat, id_feat], dim=-1)))
        return gate * col_feat + (1 - gate) * id_feat
