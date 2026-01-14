# earlystop.py

import numpy as np
import torch
import os

class EarlyStopping():
    """Early stops the training if validation performance doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='./checkpoint/', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation performance improved.
            verbose (bool): If True, prints a message for each validation loss improvement. 
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        os.makedirs(self.path, exist_ok=True)

    def __call__(self, indicator, epoch, model, args=None, extra_config: dict = None):
        """
        Checks if training should be stopped early and saves the model.
        
        Args:
            indicator (float): The metric value to monitor.
            epoch (int): The current epoch.
            model (torch.nn.Module): The model to save.
            args (argparse.Namespace, optional): Arguments to save with the checkpoint.
            extra_config (dict, optional): A dictionary with extra info to save.
        """
        score = indicator

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model, args, extra_config)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model, args, extra_config)
            self.counter = 0

    def save_checkpoint(self, score, model, args, extra_config: dict = None, is_last=False):
        """
        Saves model when validation performance improves or on the last epoch for no_eval mode.
        Args:
            score (float): The validation score. Ignored if is_last is True.
            model (torch.nn.Module): The model to save.
            args (argparse.Namespace): Arguments to save with the checkpoint.
            extra_config (dict, optional): A dictionary with extra info to save.
            is_last (bool): Flag to indicate if this is the final save in a no_eval run.
        """
        if self.verbose and not is_last:
            self.trace_func(f'Validation score improved ({self.best_score:.6f} --> {score:.6f}). Saving model...')
        elif is_last:
            self.trace_func(f'Saving final model state at the end of no_eval training.')

        config_to_save = vars(args).copy() if args else {}

        if extra_config:
            config_to_save.update(extra_config)

        checkpoint = {
            'state_dict': model.state_dict(),
            'best_epoch': self.best_epoch if not is_last else -1,
            'best_score': self.best_score if not is_last else -1,
            'args': config_to_save
        }

        model_path = os.path.join(self.path, 'pytorch_model.pt')
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
