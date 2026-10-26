import torch
import torch.nn as nn


class SparsifyModule(nn.Module):
    """Top-k magnitude (or random) masking of the last dimension of the input.

    mask_type:
      - "magnitude": keep the highest-magnitude entries (coordinate-wise or,
        if block_aligned, ranked by per-block squared L2 norm).
      - "random": keep a uniformly random subset at the same density,
        resampled every forward call (per token); a control for the magnitude mask.
      - "none": bypass (identity), regardless of target_sparsity.

    block_aligned: if True, selection operates on contiguous blocks of
    `block_size` coordinates (block score = squared L2 norm of the block)
    instead of individual coordinates, matching the paper's block-aligned
    magnitude mask.
    """

    def __init__(
        self,
        target_sparsity: float = 0.7,
        bypass: bool = False,
        mask_type: str = "magnitude",
        block_aligned: bool = False,
        block_size: int = 32,
    ):
        super().__init__()
        self.target_sparsity = target_sparsity
        self.bypass = bypass
        self.mask_type = mask_type
        self.block_aligned = block_aligned
        self.block_size = int(block_size)

        self.k_keep = None  # coordinate-wise cache
        self.n_blocks = None  # block-aligned cache
        self.k_blocks = None  # block-aligned cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bypass or self.target_sparsity <= 0 or self.mask_type == "none":
            return x

        with torch.no_grad():
            if self.block_aligned:
                mask = self._block_mask(x)
            else:
                mask = self._coord_mask(x)
        return x * mask

    def _coord_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.k_keep is None:
            # Calculate exactly once, never again
            self.k_keep = max(1, int(x.shape[-1] * (1 - self.target_sparsity)))

        if self.mask_type == "random":
            scores = torch.rand_like(x.float())
        else:
            scores = x.abs()

        threshold = torch.topk(scores, self.k_keep, dim=-1)[0][..., -1].unsqueeze(-1)
        mask = scores.ge(threshold).to(x.dtype)
        return mask

    def _block_mask(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        if self.n_blocks is None:
            assert d % self.block_size == 0, (
                f"block_aligned mask requires hidden dim ({d}) divisible by "
                f"block_size ({self.block_size})"
            )
            self.n_blocks = d // self.block_size
            self.k_blocks = max(1, round((1 - self.target_sparsity) * self.n_blocks))

        lead_shape = x.shape[:-1]
        x_blocks = x.reshape(*lead_shape, self.n_blocks, self.block_size)

        if self.mask_type == "random":
            block_scores = torch.rand(*lead_shape, self.n_blocks, device=x.device)
        else:
            block_scores = x_blocks.float().pow(2).sum(dim=-1)

        threshold = torch.topk(block_scores, self.k_blocks, dim=-1)[0][..., -1].unsqueeze(-1)
        block_mask = block_scores.ge(threshold).to(x.dtype)  # [..., n_blocks]
        mask = block_mask.unsqueeze(-1).expand(*lead_shape, self.n_blocks, self.block_size)
        mask = mask.reshape(*lead_shape, self.n_blocks * self.block_size)
        return mask
