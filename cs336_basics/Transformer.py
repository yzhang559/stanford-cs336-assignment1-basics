import torch
from torch import nn

from cs336_basics.Embedding import Embedding
from cs336_basics.Linear import Linear
from cs336_basics.MultiHeadSelfAttention import MultiHeadSelfAttention
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.SwiGLU import SwiGLU


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads: int, max_seq_len: int, d_ff, theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

        self.mha = MultiHeadSelfAttention(seq_len=max_seq_len, d_model=d_model, num_heads=num_heads, theta=theta,
                                          device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        attn_out = self.mha(self.norm1(x), position=position)
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        self.norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)


    def forward(self, token_ids):
        batch_size, seq_len = token_ids.size()
        position = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)

        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, position)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits

