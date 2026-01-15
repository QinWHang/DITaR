import numpy as np
from typing import List, Dict
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from DualviewIdentification.idnetification.poison_detector import SimplifiedDetectionResult

class SimplifiedDetectionEvaluator:
    def evaluate_detection(self,
                         detection_results: List[SimplifiedDetectionResult],
                         ground_truth: Dict[int, Dict]) -> Dict[str, float]:
        y_true, y_pred, y_scores = [], [], []
        for result in detection_results:
            true_label = ground_truth.get(result.user_id, {}).get('is_poisoned', False)
            y_true.append(int(true_label))
            y_pred.append(int(result.is_poisoned))
            y_scores.append(result.confidence)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
             tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        else:
            tp = sum(p and l for p, l in zip(y_pred, y_true))
            tn = sum(not p and not l for p, l in zip(y_pred, y_true))
            fp = sum(p and not l for p, l in zip(y_pred, y_true))
            fn = sum(not p and l for p, l in zip(y_pred, y_true))
        
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        pos_metrics = self._evaluate_position_level(detection_results, ground_truth)
        
        true_poison_confs = [r.confidence for r, t in zip(detection_results, y_true) if t]
        avg_conf_on_true_poison = np.mean(true_poison_confs) if true_poison_confs else 0.0

        metrics = {
            'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
            'true_positive': int(tp), 'true_negative': int(tn),
            'false_positive': int(fp), 'false_negative': int(fn),
            'position_precision': pos_metrics['precision'], 
            'position_recall': pos_metrics['recall'], 
            'position_f1': pos_metrics['f1'],
            'total_sequences': len(detection_results), 'detected_poisoned': sum(y_pred),
            'actual_poisoned': sum(y_true),
            'avg_confidence_detected': np.mean([r.confidence for r in detection_results if r.is_poisoned]) if sum(y_pred) > 0 else 0.0,
            'avg_confidence_on_true_poison': avg_conf_on_true_poison,
            'recall_repeat': pos_metrics['recall_by_type'].get('repeat', 0.0),
            'recall_semantic': pos_metrics['recall_by_type'].get('semantic', 0.0),
            'recall_swap': pos_metrics['recall_by_type'].get('swap', 0.0),
        }
        return metrics

    def _evaluate_position_level(self,
                             detection_results: List[SimplifiedDetectionResult],
                             ground_truth: Dict[int, Dict]) -> Dict[str, float]:
        total_detected, correct_detected, total_actual = 0, 0, 0
        
        correct_by_type = defaultdict(int)
        total_by_type = defaultdict(int)
        
        all_gt_attack_types = set()
        for gt_data in ground_truth.values():
            if gt_data.get('is_poisoned'):
                for attack_type in gt_data.get('types', {}).values():
                    all_gt_attack_types.add(attack_type)
        for attack_type in all_gt_attack_types:
            if attack_type:
                total_by_type[attack_type] = 0

        for gt_data in ground_truth.values():
            if gt_data.get('is_poisoned'):
                for attack_type in gt_data.get('types', {}).values():
                    if attack_type in total_by_type:
                        total_by_type[attack_type] += 1

        for result in detection_results:
            gt = ground_truth.get(result.user_id, {})
            if not gt.get('is_poisoned'):
                continue

            true_pos_set = set(gt.get('positions', []))
            pred_pos_set = set(result.poisoned_positions)
            
            total_detected += len(pred_pos_set)
            total_actual += len(true_pos_set)
            
            correctly_detected_pos = pred_pos_set & true_pos_set
            correct_detected += len(correctly_detected_pos)

            if correctly_detected_pos:
                gt_pos_types = gt.get('types', {})
                for pos in correctly_detected_pos:
                    attack_type = gt_pos_types.get(pos)
                    if attack_type is None:
                        attack_type = gt_pos_types.get(str(pos))
                    
                    if attack_type and attack_type in total_by_type:
                        correct_by_type[attack_type] += 1

        precision = correct_detected / total_detected if total_detected > 0 else 0
        recall = correct_detected / total_actual if total_actual > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        recall_by_type = {}
        for attack_type in sorted(list(total_by_type.keys())):
            total_count = total_by_type[attack_type]
            correct_count = correct_by_type.get(attack_type, 0)
            recall_by_type[attack_type] = correct_count / total_count if total_count > 0 else 0.0
            
        return {'precision': precision, 'recall': recall, 'f1': f1, 'recall_by_type': recall_by_type}

    def print_evaluation_report(self, metrics: Dict[str, float]):
        print("\n" + "="*60)
        print("DETECTION EVALUATION REPORT")
        print("="*60)
        
        print("\n[Sequence-level Detection]")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        
        print("\n[Confusion Matrix]")
        print(f"              Predicted")
        print(f"              Clean  Poisoned")
        print(f"Actual Clean    {metrics['true_negative']:<5d}  {metrics['false_positive']:<5d}")
        print(f"      Poisoned  {metrics['false_negative']:<5d}  {metrics['true_positive']:<5d}")
        
        print("\n[Position-level Detection]")
        print(f"Precision: {metrics['position_precision']:.4f}")
        print(f"Recall:    {metrics['position_recall']:.4f}")
        print(f"F1 Score:  {metrics['position_f1']:.4f}")

        print("\n[Recall by Attack Type]")
        print(f"Repeat:    {metrics['recall_repeat']:.4f}")
        print(f"Semantic:  {metrics['recall_semantic']:.4f}")
        print(f"Swap:      {metrics['recall_swap']:.4f}")
        
        print("\n[Summary & Confidence]")
        print(f"Total sequences:    {metrics['total_sequences']}")
        print(f"Detected poisoned:  {metrics['detected_poisoned']}")
        print(f"Actually poisoned:  {metrics['actual_poisoned']}")
        print(f"Avg. confidence (detected): {metrics['avg_confidence_detected']:.4f}")
        print(f"Avg. confidence (actual):   {metrics['avg_confidence_on_true_poison']:.4f}")
        print("="*60 + "\n")
