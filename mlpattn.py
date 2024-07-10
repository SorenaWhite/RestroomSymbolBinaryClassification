from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def create_src_lengths_mask(
        batch_size: int, src_lengths: Tensor, max_src_len: Optional[int] = None
):
    """
    Generate boolean mask to prevent attention beyond the end of source
    Inputs:
      batch_size : int
      src_lengths : [batch_size] of sentence lengths
      max_src_len: Optionally override max_src_len for the mask
    Outputs:
      [batch_size, max_src_len]
    """
    if max_src_len is None:
        max_src_len = int(src_lengths.max())
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()


def masked_softmax(scores, src_lengths, src_length_masking=True):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""
    if src_length_masking:
        bsz, max_src_len = scores.size()
        # compute masks
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        # Fill pad positions with -inf
        scores = scores.masked_fill(src_mask == 0, -np.inf)

    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)


# s(x, q) = v.T * tanh (W * x + b)
class MLPAttentionNetwork(nn.Module):

    def __init__(self, hidden_dim, attention_dim, src_length_masking=True):
        super(MLPAttentionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.src_length_masking = src_length_masking

        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        # v.T
        self.proj_v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, x, x_lengths):
        """
        :param x: seq_len * batch_size * hidden_dim
        :param x_lengths: batch_size
        :return: batch_size * seq_len, batch_size * hidden_dim
        """
        seq_len, batch_size, _ = x.size()
        # (seq_len * batch_size, hidden_dim)
        flat_inputs = x.view(-1, self.hidden_dim)
        # (seq_len * batch_size, attention_dim)
        mlp_x = self.proj_w(flat_inputs)
        # (batch_size, seq_len)
        att_scores = self.proj_v(mlp_x).view(seq_len, batch_size).t()
        # (seq_len, batch_size)
        normalized_masked_att_scores = masked_softmax(
            att_scores, x_lengths, self.src_length_masking
        ).t()
        # (batch_size, hidden_dim)
        attn_x = (x * normalized_masked_att_scores.unsqueeze(2)).sum(0)

        return normalized_masked_att_scores.t(), attn_x
