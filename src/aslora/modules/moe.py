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

    def _apply_lora_with_branch(self, lora_module, base_out: torch.Tensor, x_dense: torch.Tensor, x_sparse: torch.Tensor, gate_input: torch.Tensor | None = None) -> torch.Tensor:
        if self.branch_mode == 'sparse_only':
            return lora_module.lora_forward(base_out, x_sparse)
        if self.branch_mode == 'dense_only':
            return lora_module.lora_forward(base_out, x_dense)
        # dual_scalar_gate — gate always uses original hidden state (hidden_size-dim), not intermediate acts
        delta_dense = lora_module.lora_forward(base_out, x_dense) - base_out
        delta_sparse = lora_module.lora_forward(base_out, x_sparse) - base_out
        g = self.scalar_gate(gate_input if gate_input is not None else x_dense).to(base_out.dtype)
        return base_out + g * delta_dense + (1.0 - g) * delta_sparse

    # OPTIMIZATION: Accept pre-computed base projections to prevent re-calculation
    def _apply_expert(self, x_dense: torch.Tensor, x_sparse: torch.Tensor, expert_dict: nn.ModuleDict, g_base=None, u_base=None) -> torch.Tensor:
        if g_base is None:
            g_base = self.base_layer_.gate_proj(x_dense)
        if u_base is None:
            u_base = self.base_layer_.up_proj(x_dense)

        g = self._apply_lora_with_branch(expert_dict["gate_proj"], g_base, x_dense, x_sparse) if "gate_proj" in expert_dict else g_base
        u = self._apply_lora_with_branch(expert_dict["up_proj"], u_base, x_dense, x_sparse) if "up_proj" in expert_dict else u_base
        act = self.act_fn_(g) * u
        d_base = self.base_layer_.down_proj(act)
        # down_proj LoRA input is `act` (intermediate, 11008-dim); gate must use original hidden state (4096-dim)
        d = self._apply_lora_with_branch(expert_dict["down_proj"], d_base, act, act, gate_input=x_dense) if "down_proj" in expert_dict else d_base
        return d

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_shape = hidden_states.shape
        x_flat = hidden_states.view(-1, self.hidden_size)

        # OPTIMIZATION: Compute heavy base matrices ONCE for all tokens
        g_base_full = self.base_layer_.gate_proj(x_flat)
        u_base_full = self.base_layer_.up_proj(x_flat)

        if self.branch_mode == 'dense_only':
            x_sparse = x_flat
        else:
            x_sparse = self.sparsify_module(x_flat)

        # Shared expert (Index 0)
        y1 = self._apply_expert(x_flat, x_sparse, self.experts_[0], g_base_full, u_base_full)

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

            # WE REMOVED THE IF STATEMENT HERE. NO MORE CPU SYNCS!

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

        final_output = y1 + y2

        if self.training:
            density = F.one_hot(selected, current_routed_count).float().mean(dim=0).mean(dim=0)
            self.last_bal_loss = (density * probs.mean(dim=0)).sum() * current_routed_count

        return final_output.view(batch_shape)
