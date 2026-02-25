import os
from typing import Union, BinaryIO

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


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int,
                    out: Union[str, os.PathLike, BinaryIO]):
    torch.save({
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, out)

    if isinstance(out, (str, os.PathLike)):
        print(f"Saved checkpoint to {out}")


def load_checkpoint(src: Union[str, os.PathLike, BinaryIO], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    state = torch.load(src)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    if isinstance(src, (str, os.PathLike)):
        print(f"Loaded checkpoint from {src} at iteration {state['iteration']}")

    return state["iteration"]
