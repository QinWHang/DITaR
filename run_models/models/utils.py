# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class PointWiseFeedForward(torch.nn.Module):
    """Position-wise Feed Forward Network"""
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
    
class CalculateAttention(nn.Module):
    """Calculate attention scores"""
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, num_heads, seq_length, h_size]
            K: [batch_size, num_heads, seq_length, h_size]
            V: [batch_size, num_heads, seq_length, h_size]
            mask: [batch_size, 1, seq_length, seq_length]

        Returns:
            attention: [batch_size, num_heads, seq_length, h_size]
        """
        attention_scores = torch.matmul(Q, K.transpose(-1, -2))

        attention_scores = attention_scores / sqrt(Q.size(-1))

        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask, -1e9)

        attention_probs = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_probs, V)

        return attention_output

class Multi_CrossAttention(nn.Module):
    """
    In forward, first parameter is used to compute query, second parameter for key and value
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size
        self.all_head_size  = all_head_size
        self.num_heads      = head_num
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)


    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    

    def forward(self,x,y,log_seqs):
        """
        Cross-attention: x and y are hidden layers from two models, x as input for q, y for k and v
        """

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # attention_mask = attention_mask.eq(0)
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output


