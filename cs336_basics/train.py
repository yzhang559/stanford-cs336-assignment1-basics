import torch
import numpy as np


def data_loading(data: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[
    torch.Tensor, torch.Tensor]:
    max_start_idx = len(data) - context_length - 1

    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    inputs = np.stack([data[i: i + context_length] for i in start_indices])
    targets = np.stack([data[i + 1: i + context_length + 1] for i in start_indices])

    return torch.from_numpy(inputs).to(dtype=torch.long, device=device), \
        torch.from_numpy(targets).to(dtype=torch.long, device=device)
