# BaseModel.py
import torch.nn as nn


class BaseSeqModel(nn.Module):
    """Base model class for sequential recommendation"""

    def __init__(self, user_num, item_num, device, args) -> None:
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = device
        self.freeze_modules = []
        self.filter_init_modules = []    # all modules should be initialized
    
    def _freeze(self):
        """Freeze parameters of specified modules"""
        for name, param in self.named_parameters():
            try:
                flag = False
                for fm in self.freeze_modules:
                    if fm in name:
                        flag = True
                if flag:
                    param.requires_grad = False
            except:
                pass

    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            try:
                flag = True
                for fm in self.filter_init_modules:
                    if fm in name:
                        flag = False
                if flag:
                    nn.init.xavier_normal_(param.data)
            except:
                pass

    def _get_embedding(self, log_seqs):
        """Get sequence embedding representation"""
        raise NotImplementedError("The function for sequence embedding is missed")