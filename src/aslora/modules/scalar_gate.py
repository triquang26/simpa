import torch
import torch.nn as nn


class InputDependentScalarGate(nn.Module):
    """g = sigmoid(linear(x)) with out_dim=1, broadcast over feature dim."""
    def __init__(self, hidden_size: int, dtype: torch.dtype | None = None):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)  # g starts at 0.5 -> equal mix
        if dtype is not None:
            self.linear.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))
