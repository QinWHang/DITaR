import torch
import torch.nn as nn
import numpy as np
import os

from train_models.models.BaseModel import BaseSeqModel

class DualViewBaseModel(BaseSeqModel):
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        
        self.args = args
        self.hidden_size = args.hidden_size
        self.device = device
        
        self._load_pretrained_embeddings()
        self._init_semantic_view()
        self._init_collaborative_view()
        self._init_prediction_heads()
        
        self.bce_loss_func = nn.BCEWithLogitsLoss()
        self._init_weights()

        
    def _load_pretrained_embeddings(self):
        semantic_emb_path = os.path.join(
            self.args.embedding_dir, 'semantic', self.args.dataset,
            f'{self.args.dataset}_llama_embeddings.npy'
        )
        semantic_emb = np.load(semantic_emb_path)
        
        if semantic_emb.shape[0] != self.item_num:
             new_emb = np.zeros((self.item_num, semantic_emb.shape[1]), dtype=np.float32)
             len_to_copy = min(semantic_emb.shape[0], self.item_num)
             new_emb[:len_to_copy] = semantic_emb[:len_to_copy]
             semantic_emb = new_emb

        semantic_emb_with_special_tokens = np.zeros((self.item_num + 2, semantic_emb.shape[1]), dtype=np.float32)
        semantic_emb_with_special_tokens[1 : self.item_num + 1] = semantic_emb
        self.semantic_item_emb = nn.Embedding.from_pretrained(
            torch.from_numpy(semantic_emb_with_special_tokens), 
            freeze=self.args.freeze_semantic_emb
        )
        
        pca_emb_path = os.path.join(
            self.args.embedding_dir, 'collaborative', self.args.dataset,
            f'{self.args.dataset}_pca{self.args.pca_dim}_embeddings.npy'
        )
        pca_emb = np.load(pca_emb_path)
        if pca_emb.shape[0] != self.item_num:
             new_emb = np.zeros((self.item_num, pca_emb.shape[1]), dtype=np.float32)
             len_to_copy = min(pca_emb.shape[0], self.item_num)
             new_emb[:len_to_copy] = pca_emb[:len_to_copy]
             pca_emb = new_emb

        pca_emb_with_special_tokens = np.zeros((self.item_num + 2, pca_emb.shape[1]), dtype=np.float32)
        pca_emb_with_special_tokens[1 : self.item_num + 1] = pca_emb
        self.pca_item_emb = nn.Embedding.from_pretrained(
            torch.from_numpy(pca_emb_with_special_tokens),
            freeze=self.args.freeze_pca_emb
        )
        
        self.id_item_emb = nn.Embedding(self.item_num + 2, self.hidden_size, padding_idx=0)
        
    def _init_semantic_view(self):
        self.semantic_adapter = nn.Sequential(
            nn.Linear(self.semantic_item_emb.embedding_dim, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2), 
            nn.GELU(),
            nn.Dropout(self.args.dropout_rate),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        self.semantic_residual_proj = nn.Linear(self.semantic_item_emb.embedding_dim, self.hidden_size)
        
    def _init_collaborative_view(self):
        self.pca_adapter = nn.Sequential(
            nn.Linear(self.pca_item_emb.embedding_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size), 
            nn.GELU(),
            nn.Dropout(self.args.dropout_rate)
        )
        self.collab_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size), 
            nn.Sigmoid()
        )
        
    def _init_prediction_heads(self):
        output_dim = self.item_num + 2
        self.semantic_predictor = nn.Linear(self.hidden_size, output_dim)
        self.collab_predictor = nn.Linear(self.hidden_size, output_dim)
        
    def get_semantic_embedding(self, log_seqs):
        semantic_emb = self.semantic_item_emb(log_seqs)
        adapted_emb = self.semantic_adapter(semantic_emb)
        residual = self.semantic_residual_proj(semantic_emb)
        return adapted_emb + residual
    
    def get_collaborative_embedding(self, log_seqs):
        pca_adapted = self.pca_adapter(self.pca_item_emb(log_seqs))
        id_emb = self.id_item_emb(log_seqs)
        gate = self.collab_gate(torch.cat([pca_adapted, id_emb], dim=-1))
        return gate * pca_adapted + (1 - gate) * id_emb
