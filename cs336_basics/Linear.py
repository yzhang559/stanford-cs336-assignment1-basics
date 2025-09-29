import math

import torch
import torch.nn as nn


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self._reset_parameters()

    def _reset_parameters(self):
        din = self.in_features
        dout = self.out_features
        std = math.sqrt(2.0 / (din + dout))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.T
        return y
