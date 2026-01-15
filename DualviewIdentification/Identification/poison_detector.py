import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

@dataclass
class SimplifiedDetectionConfig:
    sliding_window_size: int = 10
    detection_threshold: float = 0.5
    representation_weight: float = 0.3
    prediction_weight: float = 0.3
    popularity_weight: float = 0.2
    pattern_weight: float = 0.2
    epsilon: float = 1e-8
    
@dataclass
class SimplifiedDetectionResult:
    user_id: int
    sequence: List[int]
    is_poisoned: bool
    confidence: float
    poisoned_positions: List[int]
    anomaly_scores: np.ndarray

class SimplifiedPoisonDetector:
    def __init__(self, 
                 clean_model: nn.Module,
                 item_popularity: Dict[int, float],
                 config: SimplifiedDetectionConfig,
                 device: torch.device):
        self.model = clean_model
        self.model.eval().to(device)
        self.item_popularity = item_popularity
        self.config = config
        self.device = device
        self._compute_popularity_stats()
        
    def _compute_popularity_stats(self):
        pop_values = list(self.item_popularity.values())
        if not pop_values:
            self.pop_mean = 0
            self.pop_std = 1
            self.pop_percentiles = [0, 0, 0, 0]
            return
            
        self.pop_mean = np.mean(pop_values)
        self.pop_std = np.std(pop_values)
        self.pop_percentiles = np.percentile(pop_values, [10, 25, 75, 90])
        
    def detect_sequences(self, sequences: List[List[int]], 
                        user_ids: Optional[List[int]] = None,
                        batch_size: int = 32) -> List[SimplifiedDetectionResult]:
        results = []
        user_ids = user_ids or list(range(len(sequences)))
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Detecting sequences"):
            batch_seqs = sequences[i:i+batch_size]
            batch_users = user_ids[i:i+batch_size]
            batch_results = self._detect_batch(batch_seqs, batch_users)
            results.extend(batch_results)
        return results
    
    def _detect_batch(self, sequences: List[List[int]], user_ids: List[int]) -> List[SimplifiedDetectionResult]:
        if not sequences:
            return []
        
        max_len = max(len(seq) for seq in sequences)
        if max_len == 0:
            return []
        
        padded_seqs = torch.zeros(len(sequences), max_len, dtype=torch.long, device=self.device)
        positions = torch.zeros(len(sequences), max_len, dtype=torch.long, device=self.device)
        
        for i, seq in enumerate(sequences):
            if seq:
                seq_tensor = torch.tensor(seq, device=self.device)
                padded_seqs[i, :len(seq)] = seq_tensor
                positions[i, :len(seq)] = torch.arange(1, len(seq) + 1, device=self.device)
        
        with torch.no_grad():
            semantic_feats, collab_feats = self.model.get_dual_representations(padded_seqs, positions)
            semantic_probs, collab_probs = self.model.get_dual_predictions(padded_seqs, positions)
        
        results = []
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            seq_len = len(seq)
            result = self._detect_single_sequence(
                seq, user_ids[i],
                semantic_feats[i, :seq_len].cpu().numpy(),
                collab_feats[i, :seq_len].cpu().numpy(),
                semantic_probs[i, :seq_len].cpu().numpy(),
                collab_probs[i, :seq_len].cpu().numpy()
            )
            results.append(result)
        return results
    
    def _detect_single_sequence(self, sequence: List[int], user_id: int,
                                semantic_feats: np.ndarray, collab_feats: np.ndarray,
                                semantic_probs: np.ndarray, collab_probs: np.ndarray) -> SimplifiedDetectionResult:
        anomaly_scores = self._compute_unified_anomaly_scores(sequence, semantic_feats, collab_feats, semantic_probs, collab_probs)
        smoothed_scores = self._smooth_scores(anomaly_scores)
        poisoned_positions = np.where(smoothed_scores > self.config.detection_threshold)[0].tolist()

        if poisoned_positions:
            confidence = np.mean(smoothed_scores[poisoned_positions]) if poisoned_positions else 0.0
            is_poisoned = True
        else:
            confidence = np.max(smoothed_scores) if len(smoothed_scores) > 0 else 0.0
            is_poisoned = False

        return SimplifiedDetectionResult(user_id, sequence, is_poisoned, float(confidence), poisoned_positions, smoothed_scores)

    def _smooth_scores(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) < 2:
            return scores
        window_size = min(self.config.sliding_window_size, len(scores))
        if window_size < 2:
            return scores
        smoothed = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        pad_len_before = (len(scores) - len(smoothed)) // 2
        pad_len_after = len(scores) - len(smoothed) - pad_len_before
        return np.pad(smoothed, (pad_len_before, pad_len_after), 'edge')

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        s_max, s_min = scores.max(), scores.min()
        if s_max == s_min:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min + self.config.epsilon)

   