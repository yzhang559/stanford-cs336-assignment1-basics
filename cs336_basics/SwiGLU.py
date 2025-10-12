import torch
from torch import nn

from cs336_basics.Linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int = None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            d_ff = ((d_model * 8 / 3) // 64) * 64


        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
    # FFX(x) = W2(SiLU(W1x).W3x)
        res1 = self.w1(x)
        silu = res1 * torch.sigmoid(res1)

        res2 = self.w3(x)
        element_wise = silu * res2
        return self.w2(element_wise)
