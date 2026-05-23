"""HydraLoRA: asymmetric LoRA with 1 shared A + N B-heads gated by a softmax router.

Reference: Tian et al., "HydraLoRA: An Asymmetric LoRA Architecture for Efficient
Fine-Tuning" (NeurIPS 2024 Oral). https://github.com/Clin0212/HydraLoRA

Per target nn.Linear (in, out):
    lora_route : Linear(in, N, bias=False)   shared router (softmax over N heads)
    lora_A     : Linear(in, r, bias=False)   shared down-projection
    lora_B{i}  : Linear(r, out, bias=False)  N independent up-projections
Forward:
    y = base(x) + scaling * sum_i softmax(route(x))_i * B_i(A(x))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HydraLoRAConfig:
    base_model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_nums: int = 8
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    def to_dict(self) -> dict:
        return {**asdict(self), "peft_type": "HYDRALORA"}


class HydraLoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds the HydraLoRA delta."""

    def __init__(self, base_linear: nn.Linear, r: int, lora_alpha: int,
                 lora_dropout: float, lora_nums: int):
        super().__init__()
        if r <= 0:
            raise ValueError("HydraLoRA rank r must be > 0")
        if lora_nums <= 0:
            raise ValueError("HydraLoRA lora_nums must be > 0")

        self.base_layer = base_linear
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_nums = lora_nums
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        device = base_linear.weight.device
        dtype = base_linear.weight.dtype

        self.lora_route = nn.Linear(in_features, lora_nums, bias=False, device=device, dtype=dtype)
        self.lora_A = nn.Linear(in_features, r, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.ModuleList([
            nn.Linear(r, out_features, bias=False, device=device, dtype=dtype)
            for _ in range(lora_nums)
        ])

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))
        for b in self.lora_B:
            nn.init.zeros_(b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        dropped = self.lora_dropout(x)

        route_logits = self.lora_route(dropped)
        route_w = F.softmax(route_logits, dim=-1, dtype=torch.float32).to(result.dtype)

        a_out = self.lora_A(dropped)
        delta = result.new_zeros(result.shape)
        for i, b in enumerate(self.lora_B):
            delta = delta + route_w[..., i:i + 1] * b(a_out)
        return result + self.scaling * delta


def inject_hydralora_into_llama(model: nn.Module, config: HydraLoRAConfig) -> nn.Module:
    """Replace target Linear modules inside every decoder layer with HydraLoRALinear."""
    attn_targets = [m for m in config.target_modules if m in {"q_proj", "k_proj", "v_proj", "o_proj"}]
    mlp_targets = [m for m in config.target_modules if m in {"gate_proj", "up_proj", "down_proj"}]

    for layer in model.model.layers:
        if hasattr(layer, "self_attn"):
            for name in attn_targets:
                if hasattr(layer.self_attn, name):
                    base = getattr(layer.self_attn, name)
                    if isinstance(base, HydraLoRALinear):
                        continue
                    wrapped = HydraLoRALinear(
                        base_linear=base,
                        r=config.r, lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout, lora_nums=config.lora_nums,
                    )
                    setattr(layer.self_attn, name, wrapped)

        if hasattr(layer, "mlp"):
            for name in mlp_targets:
                if hasattr(layer.mlp, name):
                    base = getattr(layer.mlp, name)
                    if isinstance(base, HydraLoRALinear):
                        continue
                    wrapped = HydraLoRALinear(
                        base_linear=base,
                        r=config.r, lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout, lora_nums=config.lora_nums,
                    )
                    setattr(layer.mlp, name, wrapped)

    model._hydralora_config = config
    return model


def freeze_base_for_hydralora(model: nn.Module) -> nn.Module:
    """Freeze every parameter except those belonging to HydraLoRALinear adapters."""
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, HydraLoRALinear):
            for p in module.lora_route.parameters():
                p.requires_grad = True
            for p in module.lora_A.parameters():
                p.requires_grad = True
            for b in module.lora_B:
                for p in b.parameters():
                    p.requires_grad = True
    return model


def extract_hydralora_state_dict(model: nn.Module) -> dict:
    """Collect only HydraLoRA trainable tensors (CPU, contiguous)."""
    out = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" not in name:
            continue
        out[name] = param.detach().cpu().contiguous().clone()
    return out


def load_hydralora_state_dict(model: nn.Module, state_dict: dict) -> tuple:
    """Load trained HydraLoRA weights into an injected model. Returns (missing, unexpected)."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected
