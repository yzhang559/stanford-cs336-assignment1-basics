import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self._reset_parameters()

    def _reset_parameters(self):
        std = 1.0
        nn.init.trunc_normal_(self.embedding, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]


if __name__ == '__main__':
    model = Embedding(256, 8, device='cpu')
    x = torch.randint(0, 255, (2, 4))
    print(model(x).shape)
