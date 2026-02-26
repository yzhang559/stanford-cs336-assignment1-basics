from cs336_basics.Transformer import TransformerLM
from cs336_basics.tokenizer import Tokenizer

import torch


def temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / temperature


def top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Filter and renormalize probabilities to keep only top-p nucleus."""
    if top_p >= 1.0:
        return probs
    
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    # Keep tokens where cumsum hasn't exceeded top_p yet, plus the first one that crosses
    mask = cumsum - sorted_probs <= top_p
    
    # Zero out tokens not in nucleus
    sorted_probs = sorted_probs * mask
    
    # Scatter back to original order
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
    
    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    return filtered_probs


def decoding(model: TransformerLM, tokenizer: Tokenizer, prompt: str, max_num_tokens: int,
             temperature: float = 0.7, top_p: float = 0.95, eos_token: str = "<docline>") -> str:
    model.eval()
    tokens = list(tokenizer.encode(prompt))
    eos_token_id = tokenizer.encode(eos_token)[0]

    with torch.no_grad():
        while len(tokens) < max_num_tokens:
            # Convert to tensor: shape (1, seq_len)
            idx = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)

            # Forward pass: get logits for all positions
            logits = model(idx)  # shape: (1, seq_len, vocab_size)

            # Take logits at last position only
            last_logits = logits[0, -1, :]  # shape: (vocab_size,)

            # Temperature scaling
            scaled_logits = temperature_scaling(last_logits, temperature)

            # Convert to probabilities
            probs = torch.softmax(scaled_logits, dim=-1)

            # Top-p filtering
            probs = top_p_filtering(probs, top_p)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Check for EOS
            if next_token == eos_token_id:
                break

            tokens.append(next_token)

    return tokenizer.decode(tokens)
