import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: [batch_size, seq_len, vocab_size]
    targets: [batch_size, seq_len]
    ℓi =−log softmax(oi)[xi+1]
    """
    max_logits, _ = logits.max(dim=-1, keepdim=True)
    shifted_logits = logits - max_logits

    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    # For each position (b, t), select the logit corresponding to the correct class
    correct = shifted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss = log_sum_exp - correct

    return loss.mean()
