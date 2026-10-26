"""CPU-only checks that the Triton benchmark's (values, indices) mask matches the
training-time SparsifyModule, and that the sparse-indexed product it feeds equals
x_masked @ W.

Run: uv run python tests/test_kernel_mask.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import torch

from aslora.modules.sparsify import SparsifyModule
from triton_sparse_moe_benchmark import BlockColumnSparsifyModule


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _scatter(values: torch.Tensor, indices: torch.Tensor, d: int) -> torch.Tensor:
    return torch.zeros(values.shape[0], d, dtype=values.dtype).scatter(-1, indices, values)


def test_block_mask_matches_training_module():
    torch.manual_seed(0)
    x = torch.randn(16, 4096)
    for p in (0.3, 0.7):
        values, indices = BlockColumnSparsifyModule(target_sparsity=p, block_size=32)(x)
        ref = SparsifyModule(target_sparsity=p, mask_type="magnitude", block_aligned=True, block_size=32)(x)
        _assert(torch.equal(_scatter(values, indices, x.shape[-1]), ref), f"block mask mismatch at p={p}")


def test_block_indices_are_contiguous_runs():
    torch.manual_seed(1)
    x = torch.randn(4, 4096)
    _, indices = BlockColumnSparsifyModule(target_sparsity=0.7, block_size=32)(x)
    blocks = indices.reshape(4, -1, 32)
    _assert(bool((blocks[..., 0] % 32 == 0).all()), "each run must start on a block boundary")
    _assert(bool((blocks.diff(dim=-1) == 1).all()), "each run must be 32 consecutive rows")
    _assert(bool((blocks[..., 0].diff(dim=-1) > 0).all()), "kept blocks must be sorted by position")


def test_coordinate_mode_matches_training_module():
    torch.manual_seed(2)
    x = torch.randn(8, 4096)
    values, indices = BlockColumnSparsifyModule(target_sparsity=0.7, block_size=1)(x)
    ref = SparsifyModule(target_sparsity=0.7, mask_type="magnitude", block_aligned=False)(x)
    _assert(torch.equal(_scatter(values, indices, x.shape[-1]), ref), "coordinate mask mismatch")


def test_indexed_product_equals_masked_matmul():
    # Reference for what simpa_sparse_indexed_gemm_kernel computes per token.
    torch.manual_seed(3)
    x = torch.randn(8, 512, dtype=torch.float64)
    w = torch.randn(512, 96, dtype=torch.float64)
    values, indices = BlockColumnSparsifyModule(target_sparsity=0.7, block_size=32)(x)
    y = torch.einsum("tk,tko->to", values, w[indices])
    ref = _scatter(values, indices, 512) @ w
    _assert(torch.allclose(y, ref), "sum_j v[t,j] * W[idx[t,j], :] must equal x_masked @ W")


if __name__ == "__main__":
    tests = [
        test_block_mask_matches_training_module,
        test_block_indices_are_contiguous_runs,
        test_coordinate_mode_matches_training_module,
        test_indexed_product_equals_masked_matmul,
    ]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
