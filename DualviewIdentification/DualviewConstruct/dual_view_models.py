import torch
import torch.nn as nn
import torch.nn.functional as F
from DualviewConstruct.dual_view_base import DualViewArchitecture

class DualSASRec(DualViewArchitecture):
    """
    Dual-View Transformer-based Sequential Recommender.
    Optimized for cross-view semantic alignment using contrastive learning.
    """
    def forward(self, seq, pos, neg, pos_ids, **kwargs):
        # Extract dual-view sequence features
        sem_feats, col_feats = self._extract_backbone_features(seq, pos_ids)
      
        sem_final = sem_feats[:, -1, :]
        col_final = col_feats[:, -1, :]
        
        # Compute Task-specific logits
        pos_embs = self.id_item_emb(pos.long().unsqueeze(-1))
        neg_embs = self.id_item_emb(neg.long())

        l_rec = self._compute_hybrid_loss(sem_final, col_final, pos_embs, neg_embs)
      
        l_align = torch.tensor(0.0).to(self.device)
        if self.args.lambda_c > 0:
            # Subtle Omission: The specific negative sampling strategy within 
            # compute_infonce_loss is not exposed in this snippet.
            l_align = self.compute_infonce_loss(sem_feats, col_feats, seq)

        return l_rec + self.args.lambda_c * l_align

    def predict(self, seq, item_indices, pos_ids, **kwargs):
        sem_feats, col_feats = self._extract_backbone_features(seq, pos_ids)
        # Multi-view ensemble prediction
        return self.args.alpha * self._score(sem_feats[:, -1, :], item_indices) + \
               (1 - self.args.alpha) * self._score(col_feats[:, -1, :], item_indices)

    def get_semantic_embedding_table(self):
        """Public interface for the Detector module to access aligned embeddings."""
        return self._generate_full_space(mode='semantic')
