import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_new = x - max_val

    exp = torch.exp(x_new)
    sum_exp = torch.sum(exp, dim=dim, keepdim=True)
    return exp / sum_exp
