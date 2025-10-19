import math

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_new = x - max_val

    exp = torch.exp(x_new)
    sum_exp = torch.sum(exp, dim=dim, keepdim=True)
    return exp / sum_exp


def scaled_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor, mask: Tensor = None) -> torch.Tensor:
    """
    Here we have Tq=Tk=seq_len, they can be different
    q: (B, ..., Tq, d_k)
    k: (B, ..., Tk, d_k)
    v: (B, ..., Tk, d_v)
    mask (optional): boolean or byte tensor broadcastable to (B, ..., Tq, Tk)
                     True/1 means "keep" or "valid"; False/0 means "mask out"
    returns:
      out: (B, ..., Tq, d_v)
    """
    d_k = q.shape[-1]
    scale = math.sqrt(d_k)

    scaled_scores = q @ k.transpose(-2, -1) / scale

    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == False, -torch.inf)

    attention = torch.softmax(scaled_scores, dim=-1)
    out = attention @ v
    return out
