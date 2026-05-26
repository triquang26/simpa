"""Forward-pre-hook registration for SIMPA adapter-input capture."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from measure_jaccard_aslora import MixLoraSparseMoe  # noqa: E402


_DOWN_FIRED: dict[int, bool] = {}


def _make_pre_hook(loc_key: str, accumulator, dedupe: bool = False):
    """Build a forward_pre_hook closure binding loc_key and accumulator."""
    def hook(module, inputs):
        if not inputs:
            return
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return
        if dedupe:
            key = id(module)
            if _DOWN_FIRED.get(key, False):
                return
            _DOWN_FIRED[key] = True
        accumulator.add(loc_key, x)
    return hook


def register_adapter_hooks(model, accumulator) -> tuple[list, dict[str, list[str]]]:
    """Register forward_pre_hooks on each transformer layer.

    For each transformer layer i (model.model.layers[i]):
      - q_proj input -> "layer_i/attn_qkv_in" (aliased by Q/K/V)
      - o_proj input -> "layer_i/attn_o_in"
      - MoE input    -> "layer_i/mlp_in"     (aliased by gate_proj/up_proj)
      - down_proj input (base) -> "layer_i/mlp_down_in" (dedup via fired flag)

    Returns (handles, name_map). name_map keys are the 7 logical adapter names
    and values are the per-layer loc_key lists.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(
            "register_adapter_hooks expects a LLaMA-like model with model.model.layers"
        )

    layers = model.model.layers
    n_layers = len(layers)

    # Discover MoE modules indexed by layer position.
    moe_by_layer: dict[int, nn.Module] = {}
    for name, m in model.named_modules():
        if isinstance(m, MixLoraSparseMoe):
            # Names look like "model.layers.<i>.mlp.mixlora_moes.default"
            parts = name.split(".")
            try:
                idx = parts.index("layers")
                layer_i = int(parts[idx + 1])
                moe_by_layer[layer_i] = m
            except (ValueError, IndexError):
                continue

    handles: list = []
    name_map: dict[str, list[str]] = {
        "q_proj": [],
        "k_proj": [],
        "v_proj": [],
        "o_proj": [],
        "gate_proj": [],
        "up_proj": [],
        "down_proj": [],
    }

    for i in range(n_layers):
        layer = layers[i]
        attn_qkv_key = f"layer_{i}/attn_qkv_in"
        attn_o_key = f"layer_{i}/attn_o_in"
        mlp_in_key = f"layer_{i}/mlp_in"
        mlp_down_key = f"layer_{i}/mlp_down_in"

        # Q/K/V share input — hook q_proj only.
        q_proj = layer.self_attn.q_proj
        handles.append(q_proj.register_forward_pre_hook(_make_pre_hook(attn_qkv_key, accumulator)))

        o_proj = layer.self_attn.o_proj
        handles.append(o_proj.register_forward_pre_hook(_make_pre_hook(attn_o_key, accumulator)))

        moe = moe_by_layer.get(i)
        if moe is not None:
            handles.append(moe.register_forward_pre_hook(_make_pre_hook(mlp_in_key, accumulator)))
            base_down = moe.base_layer_.down_proj
            handles.append(
                base_down.register_forward_pre_hook(
                    _make_pre_hook(mlp_down_key, accumulator, dedupe=True)
                )
            )

        name_map["q_proj"].append(attn_qkv_key)
        name_map["k_proj"].append(attn_qkv_key)
        name_map["v_proj"].append(attn_qkv_key)
        name_map["o_proj"].append(attn_o_key)
        name_map["gate_proj"].append(mlp_in_key)
        name_map["up_proj"].append(mlp_in_key)
        name_map["down_proj"].append(mlp_down_key)

    return handles, name_map


def reset_down_fired_flags(model) -> None:
    """Clear the per-batch 'fired' dedupe state of down_proj hooks.
    Call this BEFORE each batch's forward pass."""
    _DOWN_FIRED.clear()
