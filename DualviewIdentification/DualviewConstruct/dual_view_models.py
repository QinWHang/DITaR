import torch
import torch.nn as nn
import torch.nn.functional as F

from DualviewIdentification.DualviewConstruct.dual_view_base import DualViewBaseModel
from train_models.models.SASRec import SASRecBackbone
from train_models.models.GRU4Rec import GRU4RecBackbone
from train_models.models.Bert4Rec import BertBackbone

class DualViewSASRec(DualViewBaseModel):
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.pos_emb = nn.Embedding(args.max_len + 100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.semantic_backbone = SASRecBackbone(device, args)
        self.collab_backbone = SASRecBackbone(device, args)
        self._init_weights()
        self.to(device)
        
    def log2feats(self, log_seqs, positions):
        semantic_seqs = self.get_semantic_embedding(log_seqs) * (self.hidden_size ** 0.5)
        semantic_seqs += self.pos_emb(positions.long())
        semantic_seqs = self.emb_dropout(semantic_seqs)
        
        collab_seqs = self.get_collaborative_embedding(log_seqs) * (self.hidden_size ** 0.5)
        collab_seqs += self.pos_emb(positions.long())
        collab_seqs = self.emb_dropout(collab_seqs)
        
        semantic_feats = self.semantic_backbone(semantic_seqs, log_seqs)
        collab_feats = self.collab_backbone(collab_seqs, log_seqs)
        
        return semantic_feats, collab_feats
    
    def calculate_infonce_loss(self, h_sem, h_col, temperature=0.07):
        h_sem = F.normalize(h_sem, p=2, dim=1)
        h_col = F.normalize(h_col, p=2, dim=1)
        sim_matrix = torch.matmul(h_sem, h_col.T) / temperature
        labels = torch.arange(sim_matrix.shape[0], device=h_sem.device)
        loss_sem2col = F.cross_entropy(sim_matrix, labels)
        loss_col2sem = F.cross_entropy(sim_matrix.T, labels)
        return (loss_sem2col + loss_col2sem) / 2.0
    
    def forward(self, seq, pos, neg, positions, **kwargs):
        semantic_feats, collab_feats = self.log2feats(seq, positions)
        
        semantic_final = semantic_feats[:, -1, :]
        collab_final = collab_feats[:, -1, :]
        pos_embs = self.id_item_emb(pos)
        neg_embs = self.id_item_emb(neg)

        semantic_pos_logits = (semantic_final.unsqueeze(1) * pos_embs).sum(dim=-1)
        semantic_neg_logits = (semantic_final.unsqueeze(1) * neg_embs).sum(dim=-1)
        collab_pos_logits = (collab_final.unsqueeze(1) * pos_embs).sum(dim=-1)
        collab_neg_logits = (collab_final.unsqueeze(1) * neg_embs).sum(dim=-1)

        pos_labels = torch.ones_like(semantic_pos_logits, device=self.device)
        neg_labels = torch.zeros_like(semantic_neg_logits, device=self.device)
        
        indices = (pos > 0).squeeze(-1)
        semantic_loss = self.bce_loss_func(semantic_pos_logits[indices], pos_labels[indices]) + \
                       self.bce_loss_func(semantic_neg_logits[indices], neg_labels[indices])
        collab_loss = self.bce_loss_func(collab_pos_logits[indices], pos_labels[indices]) + \
                     self.bce_loss_func(collab_neg_logits[indices], neg_labels[indices])
        recommendation_loss = self.args.alpha * semantic_loss + (1 - self.args.alpha) * collab_loss
        
        non_pad_mask = (seq != 0)
        h_sem_filtered = semantic_feats[non_pad_mask]
        h_col_filtered = collab_feats[non_pad_mask]
        contrastive_loss = self.calculate_infonce_loss(h_sem_filtered, h_col_filtered)

        total_loss = recommendation_loss + self.args.lambda_c * contrastive_loss
        return total_loss

    def predict(self, seq, item_indices, positions, **kwargs):
        semantic_feats, collab_feats = self.log2feats(seq, positions)
        semantic_final = semantic_feats[:, -1, :]
        collab_final = collab_feats[:, -1, :]
        target_item_embs = self.id_item_emb(item_indices)
        semantic_logits = (semantic_final.unsqueeze(1) * target_item_embs).sum(dim=-1)
        collab_logits = (collab_final.unsqueeze(1) * target_item_embs).sum(dim=-1)
        return self.args.alpha * semantic_logits + (1 - self.args.alpha) * collab_logits
    
    def get_dual_representations(self, seq, positions):
        return self.log2feats(seq, positions)
    
    def get_dual_predictions(self, seq, positions):
        semantic_feats, collab_feats = self.log2feats(seq, positions)
        semantic_preds = torch.softmax(self.semantic_predictor(semantic_feats), dim=-1)
        collab_preds = torch.softmax(self.collab_predictor(collab_feats), dim=-1)
        return semantic_preds, collab_preds

    def get_semantic_embedding_table(self):
        all_item_ids = torch.arange(1, self.item_num + 1, device=self.device).long()
        with torch.no_grad():
            full_table = self.get_semantic_embedding(all_item_ids)
        final_table = torch.zeros(self.item_num + 2, self.hidden_size, device=self.device)
        final_table[1 : self.item_num + 1] = full_table
        return final_table

    def get_collab_embedding_table(self):
        all_item_ids = torch.arange(1, self.item_num + 1, device=self.device).long()
        with torch.no_grad():
            full_table = self.get_collaborative_embedding(all_item_ids)
        final_table = torch.zeros(self.item_num + 2, self.hidden_size, device=self.device)
        final_table[1 : self.item_num + 1] = full_table
        return final_table


