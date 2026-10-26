"""Shared adapter-only checkpoint loader (run_dense.py pattern).

`aslora.cli._load_model_with_adapter` uses `transformers.load_sharded_checkpoint`,
which expects a sharded-checkpoint index file. Adapters saved by
`aslora.training.saving.save_mixlora_adapter_safetensors` (and every checkpoint
pushed to the twanghcmut HF Hub namespace) only ship `adapter_model.safetensors`
+ `adapter_config.json` — no index — so that loader raises. This module
factors out the working pattern from run_dense.py into a reusable helper for
eval/measurement scripts.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from aslora.config import create_mixlora_config_for_llama
from aslora.modules.injection import (
    initialize_mixlora_weights_for_llama,
    inject_mixlora_into_llama,
)


@dataclass
class LoadedAdapter:
    model: Any
    tokenizer: Any
    mixlora_config: Any
    adapter_cfg: dict


def _resolve_file(adapter: str, filename: str, token: str | None) -> str:
    local_path = os.path.join(adapter, filename)
    if os.path.isdir(adapter) and os.path.exists(local_path):
        return local_path
    return hf_hub_download(repo_id=adapter, filename=filename, token=token)


def load_model_with_simpa_adapter(
    adapter_repo_or_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    device_map: str = "auto",
    token: str | None = None,
    overrides: dict | None = None,
    strict_adapter_keys: bool = True,
) -> LoadedAdapter:
    """Load a base model + inject SIMPA/MixLoRA + restore adapter-only weights.

    Mirrors run_dense.py's loading path exactly (hf_hub_download + safe_load_file
    + load_state_dict(strict=False)), but asserts every tensor present in the
    adapter safetensors file is actually consumed by the model — a silent
    partial load (e.g. from a hidden_size/branch_mode mismatch) would otherwise
    produce a model that "loads" but scores near-random.
    """
    cfg_path = _resolve_file(adapter_repo_or_dir, "adapter_config.json", token)
    with open(cfg_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)

    overrides = overrides or {}
    branch_mode = overrides.get("branch_mode", adapter_cfg.get("branch_mode", "sparse_only"))
    target_sparsity = float(overrides.get("target_sparsity", adapter_cfg.get("target_sparsity", 0.3)))
    gate_rank = int(overrides.get("gate_rank", adapter_cfg.get("gate_rank", 16)))
    base_model = overrides.get("base_model", adapter_cfg.get("base_model_name_or_path", "meta-llama/Llama-2-7b-hf"))
    r = int(overrides.get("r", adapter_cfg.get("r", 16)))
    lora_alpha = int(overrides.get("lora_alpha", adapter_cfg.get("lora_alpha", 32)))
    mask_type = overrides.get("mask_type", adapter_cfg.get("mask_type", "magnitude"))
    block_aligned = bool(overrides.get("block_aligned", adapter_cfg.get("block_aligned", False)))
    block_size = int(overrides.get("block_size", adapter_cfg.get("block_size", 32)))

    tokenizer = AutoTokenizer.from_pretrained(adapter_repo_or_dir, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        trust_remote_code=True,
    )

    mixlora_cfg = create_mixlora_config_for_llama(
        base_model=base_model,
        target_sparsity=target_sparsity,
        branch_mode=branch_mode,
        gate_rank=gate_rank,
        r=r,
        lora_alpha=lora_alpha,
        mask_type=mask_type,
        block_aligned=block_aligned,
        block_size=block_size,
    )
    init_weights = initialize_mixlora_weights_for_llama(model, mixlora_cfg)
    model = inject_mixlora_into_llama(model, mixlora_cfg, init_weights)

    weights_path = _resolve_file(adapter_repo_or_dir, "adapter_model.safetensors", token)
    adapter_sd = safe_load_file(weights_path)
    model_keys = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())
    unloadable = [k for k in adapter_sd if k not in model_keys]
    if strict_adapter_keys and unloadable:
        raise RuntimeError(
            f"{len(unloadable)}/{len(adapter_sd)} adapter tensors have no matching "
            f"parameter in the injected model (first 5: {unloadable[:5]}). This "
            "usually means branch_mode/hidden_size/target_modules mismatch between "
            "adapter_config.json and the reconstructed MixLoRA config."
        )
    missing, unexpected = model.load_state_dict(adapter_sd, strict=False)
    print(
        f"[adapter_loading] {adapter_repo_or_dir}: loaded {len(adapter_sd)} tensors "
        f"| missing={len(missing)} | unexpected={len(unexpected)}"
    )
    model.eval()

    return LoadedAdapter(model=model, tokenizer=tokenizer, mixlora_config=mixlora_cfg, adapter_cfg=adapter_cfg)
