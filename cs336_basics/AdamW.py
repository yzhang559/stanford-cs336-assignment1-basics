import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad  # gradient

                # Get or initialize state for this parameter
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p)  # first moment
                    state['v'] = torch.zeros_like(p)  # second moment

                state['t'] += 1
                t = state['t']
                m = state['m']
                v = state['v']

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(g, alpha=1 - beta1)       # m = β1*m + (1-β1)*g
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)  # v = β2*v + (1-β2)*g²

                # Compute bias-corrected learning rate
                alpha_t = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)

                # Update parameters: θ = θ - αt * m / (√v + ε)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # Apply weight decay: θ = θ - α*λ*θ
                p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss