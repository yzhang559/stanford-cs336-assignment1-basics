import torch
import torch.nn as nn
from typing import Optional


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE), applied to the last dimension of the input.

    - theta: base used to construct frequencies (commonly 10_000.0)
    - d_k:   per-head dimensionality for Q/K (must be even; pairs are rotated)
    - max_seq_len: maximum sequence length to support (precompute cos/sin up to this)
    - device: optional device for buffers
    """

    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even (got {d_k}).")

        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)

        half = self.d_k // 2
        i = torch.arange(half, dtype=torch.float32, device=device)
        position = torch.arange(max_seq_len, device=device)
        freq = self.theta ** (-2 * i / self.d_k)

        angles = torch.outer(position, freq)

        self.register_buffer("sin_mem", torch.sin(angles), persistent=False)
        self.register_buffer("cos_mem", torch.cos(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to x.

        Args:
            x:               (..., seq_len, d_k)
            token_positions: (..., seq_len) integer positions for each token

        Returns:
            Tensor of same shape as x with RoPE applied on the last dimension.
        """
        sin = self.sin_mem[token_positions]
        cos = self.cos_mem[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # R * x
        x_even_new = x_even * cos - x_odd * sin
        x_odd_new = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = x_even_new
        out[..., 1::2] = x_odd_new
        return out
