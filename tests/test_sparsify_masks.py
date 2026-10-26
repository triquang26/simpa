"""CPU-only unit tests for SparsifyModule mask_type / block_aligned extensions.

Run: uv run python tests/test_sparsify_masks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from aslora.modules.sparsify import SparsifyModule


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_none_and_bypass_are_identity():
    x = torch.randn(4, 16, 4096)
    m_none = SparsifyModule(target_sparsity=0.7, mask_type="none")
    _assert(torch.equal(m_none(x), x), "mask_type=none must be identity")

    m_bypass = SparsifyModule(target_sparsity=0.7, bypass=True, mask_type="magnitude")
    _assert(torch.equal(m_bypass(x), x), "bypass=True must be identity")

    m_zero_p = SparsifyModule(target_sparsity=0.0, mask_type="magnitude")
    _assert(torch.equal(m_zero_p(x), x), "target_sparsity<=0 must be identity")


def test_coord_magnitude_bit_identical_to_legacy():
    torch.manual_seed(0)
    x = torch.randn(8, 4096)
    p = 0.3

    # Legacy implementation (pre-refactor logic), for bit-identity comparison.
    k_keep = max(1, int(x.shape[-1] * (1 - p)))
    threshold = torch.topk(x.abs(), k_keep, dim=-1)[0][..., -1].unsqueeze(-1)
    legacy_mask = x.abs().ge(threshold).to(x.dtype)
    legacy_out = x * legacy_mask

    m = SparsifyModule(target_sparsity=p, mask_type="magnitude", block_aligned=False)
    out = m(x)
    _assert(torch.equal(out, legacy_out), "coord-wise magnitude mask must be bit-identical to legacy")


def test_coord_density_exact():
    torch.manual_seed(1)
    x = torch.randn(4, 4096)
    p = 0.7
    m = SparsifyModule(target_sparsity=p, mask_type="magnitude", block_aligned=False)
    out = m(x)
    k_keep = max(1, int(4096 * (1 - p)))
    nnz_per_row = (out != 0).sum(dim=-1)
    _assert(bool((nnz_per_row == k_keep).all()), f"expected {k_keep} nonzeros/row, got {nnz_per_row}")


def test_block_aligned_magnitude_constant_within_block():
    torch.manual_seed(2)
    x = torch.randn(4, 4096)
    block_size = 32
    p = 0.7
    m = SparsifyModule(target_sparsity=p, mask_type="magnitude", block_aligned=True, block_size=block_size)
    out = m(x)
    mask = (out != 0).float()
    mask_blocks = mask.view(4, -1, block_size)
    # Within a block, the mask must be constant (all-kept or all-dropped).
    block_sums = mask_blocks.sum(dim=-1)
    ok = ((block_sums == 0) | (block_sums == block_size)).all()
    _assert(bool(ok), "block-aligned mask must be constant within each block")

    n_blocks = 4096 // block_size
    k_blocks = max(1, round((1 - p) * n_blocks))
    kept_blocks_per_row = (block_sums == block_size).sum(dim=-1)
    _assert(bool((kept_blocks_per_row == k_blocks).all()), "wrong number of kept blocks")


def test_block_aligned_random_density_and_resampling():
    torch.manual_seed(3)
    x = torch.randn(4, 4096)
    block_size = 32
    p = 0.7
    m = SparsifyModule(target_sparsity=p, mask_type="random", block_aligned=True, block_size=block_size)
    out1 = m(x)
    out2 = m(x)
    mask1 = (out1 != 0)
    mask2 = (out2 != 0)

    n_blocks = 4096 // block_size
    k_blocks = max(1, round((1 - p) * n_blocks))
    density1 = mask1.view(4, -1, block_size).sum(dim=-1).ne(0).sum(dim=-1)
    _assert(bool((density1 == k_blocks).all()), "random block mask must match target density")

    _assert(not torch.equal(mask1, mask2), "random mask must be resampled every forward call")


def test_coord_random_density_and_resampling():
    torch.manual_seed(4)
    x = torch.randn(4, 4096)
    p = 0.3
    m = SparsifyModule(target_sparsity=p, mask_type="random", block_aligned=False)
    out1 = m(x)
    out2 = m(x)
    k_keep = max(1, int(4096 * (1 - p)))
    nnz1 = (out1 != 0).sum(dim=-1)
    _assert(bool((nnz1 == k_keep).all()), "coord random mask must match target density")
    _assert(not torch.equal(out1 != 0, out2 != 0), "coord random mask must be resampled every forward call")


def test_block_size_must_divide_hidden_dim():
    x = torch.randn(2, 100)  # 100 not divisible by 32
    m = SparsifyModule(target_sparsity=0.5, mask_type="magnitude", block_aligned=True, block_size=32)
    try:
        m(x)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for non-divisible hidden dim")


if __name__ == "__main__":
    tests = [
        test_none_and_bypass_are_identity,
        test_coord_magnitude_bit_identical_to_legacy,
        test_coord_density_exact,
        test_block_aligned_magnitude_constant_within_block,
        test_block_aligned_random_density_and_resampling,
        test_coord_random_density_and_resampling,
        test_block_size_must_divide_hidden_dim,
    ]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
