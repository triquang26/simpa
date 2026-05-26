"""Streaming block-norm squared accumulator with reservoir sampling cap."""
from __future__ import annotations

import numpy as np
import torch


class BlockNormAccumulator:
    """Per-location running collection of block-norm squared values as fp32 numpy arrays.

    Use:
      acc = BlockNormAccumulator(block_size=32, max_samples_per_loc=2_000_000, seed=0)
      acc.set_attention_mask(attention_mask)  # before each batch
      ...hooks call acc.add(loc_key, raw_input_tensor)...
      values = acc.get(loc_key)  # 1D np.ndarray float32 of block-norm squared values
    """

    def __init__(
        self,
        block_size: int = 32,
        max_samples_per_loc: int = 2_000_000,
        seed: int = 0,
    ) -> None:
        self.block_size = int(block_size)
        self.max_samples_per_loc = int(max_samples_per_loc)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._storage: dict[str, list[np.ndarray]] = {}
        self._counts: dict[str, int] = {}
        self._attention_mask_flat: torch.Tensor | None = None

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        """Set the current batch's attention mask (1=keep, 0=pad).
        Shape: [B, T] or [B*T]. None means accept all tokens."""
        if attention_mask is None:
            self._attention_mask_flat = None
            return
        m = attention_mask
        if m.dim() > 1:
            m = m.reshape(-1)
        self._attention_mask_flat = m.bool()

    def add(self, loc_key: str, x: torch.Tensor) -> None:
        """x: shape [B, T, d] or [N, d] (bf16/fp32, any device).

        Flattens to [N, d], applies stored attention mask, splits the last dim
        into blocks of size ``block_size`` (drops a trailing partial block),
        computes per-block squared L2 norm in fp32, drops near-zero values,
        and appends to the per-loc storage. Reservoir-samples down to
        ``max_samples_per_loc`` when the buffer grows past 2x cap.
        """
        if x.dim() == 0:
            return
        if x.dim() >= 3:
            x_flat = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            x_flat = x
        else:
            x_flat = x.unsqueeze(0)

        N, d = x_flat.shape
        if N == 0 or d < self.block_size:
            return

        if self._attention_mask_flat is not None:
            mask = self._attention_mask_flat
            if mask.numel() == N:
                mask = mask.to(x_flat.device)
                x_flat = x_flat[mask]
            # If lengths mismatch (e.g. hook fired on intermediate shape), skip masking.

        N_kept = x_flat.shape[0]
        if N_kept == 0:
            return

        n_blocks = d // self.block_size
        if n_blocks <= 0:
            return
        usable = n_blocks * self.block_size
        x_blocks = x_flat[:, :usable].reshape(N_kept, n_blocks, self.block_size)
        # fp32 squared L2 norm per block
        norms_sq = x_blocks.to(torch.float32).pow(2).sum(dim=-1)
        flat = norms_sq.reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)
        flat = flat[flat > 1e-12]
        if flat.size == 0:
            return

        bucket = self._storage.setdefault(loc_key, [])
        bucket.append(flat)
        self._counts[loc_key] = self._counts.get(loc_key, 0) + int(flat.size)

        if self._counts[loc_key] > 2 * self.max_samples_per_loc:
            combined = np.concatenate(bucket)
            idx = self._rng.choice(combined.size, size=self.max_samples_per_loc, replace=False)
            sub = combined[idx]
            self._storage[loc_key] = [sub]
            self._counts[loc_key] = int(sub.size)

    def get(self, loc_key: str) -> np.ndarray:
        """Return concatenated fp32 numpy array for one location. Empty array if missing."""
        bucket = self._storage.get(loc_key)
        if not bucket:
            return np.empty(0, dtype=np.float32)
        if len(bucket) == 1:
            return bucket[0]
        return np.concatenate(bucket)

    def keys(self) -> list[str]:
        """All loc_keys seen so far."""
        return list(self._storage.keys())

    def total_samples(self) -> int:
        """Sum of samples across all loc_keys."""
        return int(sum(self._counts.values()))
