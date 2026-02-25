import argparse
import os
import time
from typing import Union, BinaryIO

import torch
import numpy as np

from cs336_basics.AdamW import AdamW, gradient_clipping, learning_rate_schedule
from cs336_basics.Transformer import TransformerLM
from cs336_basics.cross_entropy import cross_entropy


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train TransformerLM")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Data paths
    parser.add_argument("--train_data", type=str, default="data/TinyStoriesV2-GPT4-train.npy")
    parser.add_argument("--valid_data", type=str, default="data/TinyStoriesV2-GPT4-valid.npy")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")

    # Logging/eval intervals
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/mps/cpu). Auto-detected if not set.")

    return parser.parse_args()


def run_train(args):
    # ============ Configuration ============
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"using device: {device}")

    # ============ Data Loading ============
    train_data = np.load(args.train_data, mmap_mode='r')
    valid_data = np.load(args.valid_data, mmap_mode='r')

    # ============ Model Setup ============
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=device,
        dtype=torch.float32
    )
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    # ============ Optimizer Setup ============
    optimizer = AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    start_iter = 0
    if os.path.exists(args.checkpoint_path):
        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
    else:
        print("Starting from scratch")

    model.train()
    start_time = time.time()

    # ============ Training Loop ============
    for iteration in range(start_iter, args.max_iters):
        current_lr = learning_rate_schedule(iteration, args.lr_max, args.lr_min, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        data, target = data_loading(train_data, args.batch_size, args.context_length, device)
        logits = model.forward(data)
        loss = cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()

        # ============ Logging ============
        if (iteration + 1) % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            print(
                f"iter {iteration + 1}/{args.max_iters} | loss {loss.item():.4f} | lr {current_lr:.2e} | time {elapsed_time:.1f}s")

        # ============ Evaluation ============
        if (iteration + 1) % args.eval_interval == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    val_data, val_target = data_loading(valid_data, args.batch_size, args.context_length, device)
                    val_logits = model.forward(val_data)
                    val_loss = cross_entropy(val_logits, val_target)
                    val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"iter {iteration + 1}/{args.max_iters} | val_loss {avg_val_loss:.4f}")
            model.train()

        # ============ Checkpointing ============
        if (iteration + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, iteration + 1, args.checkpoint_path)


if __name__ == '__main__':
    args = parse_args()
    run_train(args)
