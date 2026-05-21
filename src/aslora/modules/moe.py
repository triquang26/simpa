import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from mixlora.lora_linear import LoraLinear  # noqa: F401

from .sparsify import SparsifyModule
from .scalar_gate import InputDependentScalarGate


class MixLoraSparseMoe(nn.Module):
    def __init__(self, base_layer: nn.Module, config: Any, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        object.__setattr__(self, "base_layer_", base_layer)

        self.hidden_size = getattr(config, 'hidden_size', 4096)
        self.total_experts = getattr(config, 'num_experts_', 8)
        self.routed_experts_count = self.total_experts - 1

        self.moe_topk = getattr(config, 'moe_topk', 2)
        self.temperature = 0.1

        self.experts_ = nn.ModuleList([nn.ModuleDict() for _ in range(self.total_experts)])
        self.gate_ = nn.Parameter(torch.randn(self.routed_experts_count, self.hidden_size) * 0.01)
        self.gate_norm = nn.LayerNorm(self.hidden_size)

        self.act_fn_ = nn.SiLU()

        self.branch_mode = getattr(config, 'branch_mode', 'sparse_only')
        target_sparsity = getattr(config, 'target_sparsity', 0.3)
        self.sparsify_module = SparsifyModule(target_sparsity)

        if self.branch_mode == 'dual_scalar_gate':
            self.scalar_gate = InputDependentScalarGate(self.hidden_size)

        self.register_buffer("last_bal_loss", torch.tensor(0.0))
        self.last_hidden: torch.Tensor | None = None
        self.last_routing: torch.Tensor | None = None

    def _apply_lora_with_branch(self, lora_module, base_out: torch.Tensor, x_dense: torch.Tensor, x_sparse: torch.Tensor) -> torch.Tensor:
        # LoRA input: dense for 'dense_only' mode, sparse otherwise. The dual_scalar_gate
        # branching is handled at the y1/y2 level in forward(), not per-projection.
        if self.branch_mode == 'dense_only':
            return lora_module.lora_forward(base_out, x_dense)
        return lora_module.lora_forward(base_out, x_sparse)

    def _apply_expert(self, x_dense: torch.Tensor, x_sparse: torch.Tensor, expert_dict: nn.ModuleDict, g_base=None, u_base=None) -> torch.Tensor:
        if g_base is None:
            g_base = self.base_layer_.gate_proj(x_dense)
        if u_base is None:
            u_base = self.base_layer_.up_proj(x_dense)

        g = self._apply_lora_with_branch(expert_dict["gate_proj"], g_base, x_dense, x_sparse) if "gate_proj" in expert_dict else g_base
        u = self._apply_lora_with_branch(expert_dict["up_proj"], u_base, x_dense, x_sparse) if "up_proj" in expert_dict else u_base
        act = self.act_fn_(g) * u
        d_base = self.base_layer_.down_proj(act)
        d = self._apply_lora_with_branch(expert_dict["down_proj"], d_base, act, act) if "down_proj" in expert_dict else d_base
        return d

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_shape = hidden_states.shape
        x_flat = hidden_states.view(-1, self.hidden_size)

        g_base_full = self.base_layer_.gate_proj(x_flat)
        u_base_full = self.base_layer_.up_proj(x_flat)

        if self.branch_mode == 'dense_only':
            x_sparse = x_flat
        else:
            x_sparse = self.sparsify_module(x_flat)

        # Shared expert (Index 0):
        # - dual_scalar_gate: shared expert acts as the dense path (LoRA on x_flat)
        # - sparse_only / dense_only: shared expert uses the same LoRA input as routed experts
        shared_lora_input = x_flat if self.branch_mode == 'dual_scalar_gate' else x_sparse
        y1 = self._apply_expert(x_flat, shared_lora_input, self.experts_[0], g_base_full, u_base_full)

        x_norm = self.gate_norm(x_flat.to(self.gate_.dtype))
        logits = F.linear(x_norm, self.gate_)
        probs = F.softmax(logits / self.temperature, dim=-1)
        weights, selected = torch.topk(probs, self.moe_topk, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        if not self.training:
            self.last_hidden = x_flat.detach().float().cpu()
            self.last_routing = selected.detach().cpu()

        y2 = torch.zeros_like(x_flat)
        current_routed_count = self.gate_.shape[0]

        for router_idx in range(current_routed_count):
            token_indices, topk_indices = torch.where(selected == router_idx)

            curr_x_dense = x_flat[token_indices]
            curr_x_sparse = x_sparse[token_indices]

            curr_g_base = g_base_full[token_indices]
            curr_u_base = u_base_full[token_indices]

            target_expert_idx = (router_idx + 1) % len(self.experts_)

            out_expert = self._apply_expert(
                curr_x_dense, curr_x_sparse, self.experts_[target_expert_idx], curr_g_base, curr_u_base
            )

            w = weights[token_indices, topk_indices].unsqueeze(1)
            y2 = y2.index_add(0, token_indices, out_expert * w.to(out_expert.dtype))

        # Mix shared (y1, dense path) and routed (y2, sparse path):
        # - dual_scalar_gate: scalar gate g ∈ R per token, broadcast over feature dim (Jacobian rank=1)
        # - sparse_only / dense_only: plain sum (no mixing gate)
        if self.branch_mode == 'dual_scalar_gate':
            gate_val = self.scalar_gate(x_flat).to(x_flat.dtype)  # [T, 1]
            final_output = gate_val * y1 + (1.0 - gate_val) * y2
        else:
            final_output = y1 + y2

        if self.training:
            density = F.one_hot(selected, current_routed_count).float().mean(dim=0).mean(dim=0)
            self.last_bal_loss = (density * probs.mean(dim=0)).sum() * current_routed_count

        return final_output.view(batch_shape)
