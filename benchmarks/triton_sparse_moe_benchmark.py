"""Standalone SIMPA sparse MoE Triton benchmark.

Run on a CUDA machine with a PyTorch CUDA build and Triton installed:

    uv run python benchmarks/triton_sparse_moe_benchmark.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only on incomplete envs.
    raise SystemExit("PyTorch is required to run this benchmark.") from exc

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - handled by runtime availability check.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def simpa_sparse_indexed_gemm_kernel(
        values_ptr,
        indices_ptr,
        weight_ptr,
        output_ptr,
        n_tokens: tl.constexpr,
        in_features: tl.constexpr,
        out_features: tl.constexpr,
        k_keep: tl.constexpr,
        values_stride_t: tl.constexpr,
        indices_stride_t: tl.constexpr,
        weight_stride_in: tl.constexpr,
        weight_stride_out: tl.constexpr,
        output_stride_t: tl.constexpr,
        output_stride_out: tl.constexpr,
        block_k: tl.constexpr,
        block_out: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        out_block = tl.program_id(1)

        offs_k = tl.arange(0, block_k)
        offs_out = out_block * block_out + tl.arange(0, block_out)
        k_mask = offs_k < k_keep
        out_mask = offs_out < out_features

        sparse_values = tl.load(
            values_ptr + token_id * values_stride_t + offs_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        sparse_indices = tl.load(
            indices_ptr + token_id * indices_stride_t + offs_k,
            mask=k_mask,
            other=0,
        )
        sparse_indices = tl.maximum(0, tl.minimum(sparse_indices, in_features - 1))

        weights = tl.load(
            weight_ptr
            + sparse_indices[:, None] * weight_stride_in
            + offs_out[None, :] * weight_stride_out,
            mask=k_mask[:, None] & out_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(sparse_values[:, None] * weights, axis=0)

        tl.store(
            output_ptr + token_id * output_stride_t + offs_out * output_stride_out,
            acc,
            mask=out_mask,
        )

else:

    def simpa_sparse_indexed_gemm_kernel(*args, **kwargs):
        raise RuntimeError("Triton is not installed.")


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _require_cuda_stack() -> None:
    if triton is None:
        raise RuntimeError(
            "Triton is not installed. Install Triton and use a CUDA-enabled PyTorch build "
            "before running this benchmark."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This benchmark requires a CUDA GPU and a PyTorch CUDA build."
        )
    try:
        device_name = torch.cuda.get_device_name()
    except Exception as exc:  # pragma: no cover - defensive CUDA init path.
        raise RuntimeError(f"CUDA failed to initialize: {exc}") from exc
    if not device_name:
        raise RuntimeError("CUDA initialized but no device name was reported.")


def simpa_linear(
    sparse_values: torch.Tensor,
    sparse_indices: torch.Tensor,
    weight: torch.Tensor,
    *,
    block_out: int = 32,
) -> torch.Tensor:
    """Compute y[t, o] = sum_j sparse_values[t, j] * weight[sparse_indices[t, j], o]."""
    if sparse_values.ndim != 2 or sparse_indices.ndim != 2 or weight.ndim != 2:
        raise ValueError("Expected sparse_values [T, K], sparse_indices [T, K], weight [D, H].")
    if sparse_values.shape != sparse_indices.shape:
        raise ValueError("sparse_values and sparse_indices must have the same shape.")
    if not sparse_values.is_cuda or not sparse_indices.is_cuda or not weight.is_cuda:
        raise ValueError("simpa_linear expects CUDA tensors.")

    n_tokens, k_keep = sparse_values.shape
    in_features, out_features = weight.shape
    output = torch.empty((n_tokens, out_features), device=sparse_values.device, dtype=torch.float32)
    block_k = _next_power_of_2(k_keep)
    block_k = max(16, block_k)
    grid = (n_tokens, triton.cdiv(out_features, block_out))

    simpa_sparse_indexed_gemm_kernel[grid](
        sparse_values.contiguous(),
        sparse_indices.contiguous(),
        weight.contiguous(),
        output,
        n_tokens,
        in_features,
        out_features,
        k_keep,
        sparse_values.stride(0),
        sparse_indices.stride(0),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        block_k,
        block_out,
        num_warps=4,
    )
    return output


class BlockColumnSparsifyModule(nn.Module):
    """Keep the largest activations by absolute value for each token."""

    def __init__(self, target_sparsity: float = 0.7):
        super().__init__()
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in [0, 1).")
        self.target_sparsity = target_sparsity

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k_keep = max(1, int(x.shape[-1] * (1.0 - self.target_sparsity)))
        _, indices = torch.topk(x.abs(), k_keep, dim=-1, sorted=False)
        values = torch.gather(x, dim=-1, index=indices)
        return values.contiguous(), indices.to(torch.int64).contiguous()


class DenseMoeBaseline(nn.Module):
    """PyTorch dense selected-expert baseline."""

    def __init__(self, num_experts: int, in_features: int, out_features: int, dtype: torch.dtype):
        super().__init__()
        weights = torch.empty(num_experts, in_features, out_features, dtype=dtype)
        nn.init.normal_(weights, mean=0.0, std=1.0 / math.sqrt(in_features))
        self.weights = nn.Parameter(weights)

    def forward(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.zeros((x.shape[0], self.weights.shape[-1]), device=x.device, dtype=torch.float32)
        for expert_id in range(self.weights.shape[0]):
            token_ids, topk_ids = torch.where(selected_experts == expert_id)
            if token_ids.numel() == 0:
                continue
            expert_out = x[token_ids].float() @ self.weights[expert_id].float()
            expert_out = expert_out * routing_weights[token_ids, topk_ids].float().unsqueeze(-1)
            out.index_add_(0, token_ids, expert_out)
        return out


class SIMPASparseMoe(nn.Module):
    """SIMPA-style sparse selected-expert benchmark module."""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        target_sparsity: float,
        dtype: torch.dtype,
    ):
        super().__init__()
        weights = torch.empty(num_experts, in_features, out_features, dtype=dtype)
        nn.init.normal_(weights, mean=0.0, std=1.0 / math.sqrt(in_features))
        self.weights = nn.Parameter(weights)
        self.sparsify = BlockColumnSparsifyModule(target_sparsity=target_sparsity)

    def forward(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        sparse_values, sparse_indices = self.sparsify(x)
        out = torch.zeros((x.shape[0], self.weights.shape[-1]), device=x.device, dtype=torch.float32)
        for expert_id in range(self.weights.shape[0]):
            token_ids, topk_ids = torch.where(selected_experts == expert_id)
            if token_ids.numel() == 0:
                continue
            expert_out = simpa_linear(
                sparse_values[token_ids],
                sparse_indices[token_ids],
                self.weights[expert_id],
            )
            expert_out = expert_out * routing_weights[token_ids, topk_ids].float().unsqueeze(-1)
            out.index_add_(0, token_ids, expert_out)
        return out


@dataclass
class BenchmarkConfig:
    tokens: int = 4096
    in_features: int = 4096
    out_features: int = 11008
    num_experts: int = 8
    top_k: int = 2
    target_sparsity: float = 0.7
    warmup: int = 10
    iters: int = 50
    dtype: str = "float16"


def _parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be one of: float16, bfloat16, float32.")


def _make_routing(
    x: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    router = torch.randn(x.shape[-1], num_experts, device=x.device, dtype=x.dtype)
    logits = x @ router
    probs = F.softmax(logits.float(), dim=-1)
    weights, selected = torch.topk(probs, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return selected.to(torch.int64), weights


def _time_module(module: nn.Module, *args: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        module(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        module(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_benchmark() -> None:
    parser = argparse.ArgumentParser(description="Standalone SIMPA Triton sparse MoE benchmark")
    parser.add_argument("--tokens", type=int, default=BenchmarkConfig.tokens)
    parser.add_argument("--in-features", type=int, default=BenchmarkConfig.in_features)
    parser.add_argument("--out-features", type=int, default=BenchmarkConfig.out_features)
    parser.add_argument("--num-experts", type=int, default=BenchmarkConfig.num_experts)
    parser.add_argument("--top-k", type=int, default=BenchmarkConfig.top_k)
    parser.add_argument("--target-sparsity", type=float, default=BenchmarkConfig.target_sparsity)
    parser.add_argument("--warmup", type=int, default=BenchmarkConfig.warmup)
    parser.add_argument("--iters", type=int, default=BenchmarkConfig.iters)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=BenchmarkConfig.dtype)
    args = parser.parse_args()

    try:
        _require_cuda_stack()
    except RuntimeError as exc:
        raise SystemExit(f"Cannot run benchmark: {exc}") from exc

    if args.top_k > args.num_experts:
        raise SystemExit("--top-k must be <= --num-experts.")

    device = torch.device("cuda")
    dtype = _parse_dtype(args.dtype)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    x = torch.randn(args.tokens, args.in_features, device=device, dtype=dtype)
    selected_experts, routing_weights = _make_routing(x, args.num_experts, args.top_k)

    dense = DenseMoeBaseline(
        num_experts=args.num_experts,
        in_features=args.in_features,
        out_features=args.out_features,
        dtype=dtype,
    ).to(device)
    sparse = SIMPASparseMoe(
        num_experts=args.num_experts,
        in_features=args.in_features,
        out_features=args.out_features,
        target_sparsity=args.target_sparsity,
        dtype=dtype,
    ).to(device)
    sparse.weights.data.copy_(dense.weights.data)

    print(f"Device: {torch.cuda.get_device_name()}")
    print(
        "Shape: "
        f"tokens={args.tokens}, in={args.in_features}, out={args.out_features}, "
        f"experts={args.num_experts}, top_k={args.top_k}, sparsity={args.target_sparsity}"
    )
    print(f"dtype={args.dtype}, warmup={args.warmup}, iters={args.iters}")

    dense_ms = _time_module(dense, x, selected_experts, routing_weights, warmup=args.warmup, iters=args.iters)
    sparse_ms = _time_module(sparse, x, selected_experts, routing_weights, warmup=args.warmup, iters=args.iters)

    print(f"DenseMoeBaseline: {dense_ms:.3f} ms/iter")
    print(f"SIMPASparseMoe:   {sparse_ms:.3f} ms/iter")
    print(f"Speedup:          {dense_ms / sparse_ms:.2f}x")


if __name__ == "__main__":
    run_benchmark()
