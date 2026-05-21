from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from transformers import PreTrainedModel
from mixlora.config import MixLoraConfig
from mixlora.lora_linear import LoraLinear
from .moe import MixLoraSparseMoe


def _inject_llama_attn_module(layer_idx, self_attn, config, weights: Optional[Dict] = None):
    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        if proj_name not in config.target_modules_ or not config.target_modules_[proj_name]:
            continue
        if not hasattr(self_attn, proj_name):
            continue

        base_layer = getattr(self_attn, proj_name)
        key_prefix = f"mixlora.layers.{layer_idx}.self_attn.{proj_name}"

        loaded_weights = (None, None)
        if weights is not None:
            w_a = weights.get(f"{key_prefix}.lora_A.weight")
            w_b = weights.get(f"{key_prefix}.lora_B.weight")
            if w_a is not None and w_b is not None:
                loaded_weights = (w_a, w_b)

        setattr(self_attn, proj_name, LoraLinear(base_layer, config, loaded_weights))


def _inject_llama_mlp_module(layer_idx, mlp, config, weights: Optional[Dict] = None):
    if hasattr(mlp, 'mixlora_moes') and config.adapter_name_ in mlp.mixlora_moes:
        return

    current_device = mlp.gate_proj.weight.device
    target_dtype = mlp.gate_proj.weight.dtype
    real_hidden_size = mlp.gate_proj.in_features

    moe_layer = MixLoraSparseMoe(mlp, config, layer_idx=layer_idx).to(current_device, dtype=target_dtype)

    gate_key = f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"
    if weights is not None and gate_key in weights:
        gate_w = weights[gate_key].to(device=current_device, dtype=target_dtype)
        if gate_w.shape != (config.num_experts_, real_hidden_size):
            gate_w = gate_w.T
        moe_layer.gate_ = nn.Parameter(gate_w)
    else:
        nn.init.normal_(moe_layer.gate_, std=0.02)

    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        if proj_name not in config.target_modules_ or not config.target_modules_[proj_name]:
            continue
        base_proj = getattr(mlp, proj_name)

        for expert_idx in range(config.num_experts_):
            prefix = f"mixlora.layers.{layer_idx}.mlp.{proj_name}.experts.{expert_idx}"
            expert_weights = (None, None)

            if weights is not None:
                w_a = weights.get(f"{prefix}.lora_A.weight")
                w_b = weights.get(f"{prefix}.lora_B.weight")
                if w_a is not None and w_b is not None:
                    expert_weights = (w_a.to(current_device, dtype=target_dtype),
                                      w_b.to(current_device, dtype=target_dtype))

            moe_layer.experts_[expert_idx][proj_name] = LoraLinear(base_proj, config, expert_weights)

    if not hasattr(mlp, "mixlora_moes"):
        mlp.mixlora_moes = nn.ModuleDict()
    mlp.mixlora_moes[config.adapter_name_] = moe_layer
    if not hasattr(mlp, '_original_forward'):
        mlp._original_forward = mlp.forward
    mlp.forward = lambda x, **kwargs: moe_layer(x)


def inject_mixlora_into_llama(model, config, weights: Optional[Dict] = None):
    config.model_type_ = model.config.model_type
    model._mixlora_config = config

    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn'):
            _inject_llama_attn_module(idx, layer.self_attn, config, weights)
        if hasattr(layer, 'mlp'):
            _inject_llama_mlp_module(idx, layer.mlp, config, weights)

    return model


def initialize_mixlora_weights_for_llama(model: PreTrainedModel, config: MixLoraConfig) -> Dict[str, torch.Tensor]:
    weights = {}
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        mlp = model.model.layers[layer_idx].mlp
        hidden_size = mlp.gate_proj.in_features

        weights[f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"] = torch.randn(config.num_experts_, hidden_size) * config.router_init_range_

        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            if proj_name not in config.target_modules_ or not config.target_modules_[proj_name]:
                continue
            if not hasattr(mlp, proj_name):
                continue

            base_layer = getattr(mlp, proj_name)
            in_features = base_layer.weight.shape[1]
            out_features = base_layer.weight.shape[0]

            for expert_idx in range(config.num_experts_):
                layer_prefix = f"mixlora.layers.{layer_idx}.mlp.{proj_name}.experts.{expert_idx}"
                weights[f"{layer_prefix}.lora_A.weight"] = torch.randn(config.lora_r_, in_features) * 0.01
                weights[f"{layer_prefix}.lora_B.weight"] = torch.zeros(out_features, config.lora_r_)

    return weights
