import torch
from torch import nn

from cs336_basics import attention_utils
from cs336_basics.Linear import Linear
from cs336_basics.RoPE import RoPE


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, seq_len: int, d_model: int, num_heads: int, theta: float = 10000.0, is_rope: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.seq_len = seq_len

        self.is_rope = is_rope
        if self.is_rope:
            self.rope = RoPE(theta, self.d_k, seq_len, device=device)

        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, position: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape

        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, which_head, d_model // num_heads)
        '''
        Before reshape:  (B, T, d_model)
         token_1: [........] 8 dims
         token_2: [........]
             ↓ split into 2 heads
        After view: (B, T, H, d_k)
         token_1: [[head0 4dims], [head1 4dims]]
             ↓ group by head instead of by token
        After transpose: (B, H, T, d_k)
         head0: [[token1 4d], [token2 4d], ...]
         head1: [[token1 4d], [token2 4d], ...]
        '''

        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if self.is_rope:
            q = self.rope.forward(q, position)
            k = self.rope.forward(k, position)

        # Causal mask
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1) == False

        # (batch_size, num_heads, seq_len, d_v)
        attention = attention_utils.scaled_dot_product_attention(q, k, v, mask)
        # contiguous() ensures the tensor’s data is laid out contiguously in memory (row-major order)
        # so operations like .view() that rely on simple stride patterns can work.
        concat_attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_O(concat_attention)
