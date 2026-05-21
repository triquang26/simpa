import torch
import torch.nn as nn


class SparsifyModule(nn.Module):
    def __init__(self, target_sparsity: float = 0.7, bypass: bool = False):
        super().__init__()
        self.target_sparsity = target_sparsity
        self.k_keep = None
        self.bypass = bypass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bypass:
            return x
        if self.target_sparsity <= 0:
            return x
        with torch.no_grad():
            if self.k_keep is None:
                # Calculate exactly once, never again
                self.k_keep = max(1, int(x.shape[-1] * (1 - self.target_sparsity)))

            # H200 Tensor Cores handle this instantly
            threshold = torch.topk(x.abs(), self.k_keep, dim=-1)[0][..., -1].unsqueeze(-1)
            mask = x.abs().ge(threshold).to(x.dtype)
        return x * mask
