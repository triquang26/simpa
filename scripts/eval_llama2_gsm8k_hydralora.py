"""Evaluate the pushed HydraLoRA adapter on GSM8K test (batch 32).

Loads adapter_config.json + adapter_model.safetensors from the HF Hub repo
created by scripts/train_llama2_gsm8k_hydralora.py, re-injects HydraLoRA into a
freshly loaded Llama-2-7b-hf base, then runs greedy generation on GSM8K test.

Run from repo root:
    uv run python scripts/eval_llama2_gsm8k_hydralora.py
"""
import json
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from aslora.eval.gsm8k import run_offline_eval
from aslora.modules.hydralora import (
    HydraLoRAConfig,
    inject_hydralora_into_llama,
    load_hydralora_state_dict,
)
from aslora.secrets import load_secrets, login_hf_and_wandb

# ── Config ─────────────────────────────────────────────────────────────
HF_REPO         = "twanghcmut/Llama-2-7b-hf-hydralora-r50-metamath"
EVAL_BATCH_SIZE = 32
MAX_NEW_TOKENS  = 512
# ───────────────────────────────────────────────────────────────────────


def main():
    secrets = load_secrets()
    login_hf_and_wandb(secrets)
    token = secrets.get("HF_TOKEN") or None

    cfg_path = hf_hub_download(repo_id=HF_REPO, filename="adapter_config.json", token=token)
    with open(cfg_path) as f:
        ada = json.load(f)
    print(f"[cfg] {ada}")

    hydra_cfg = HydraLoRAConfig(
        base_model_name_or_path=ada.get("base_model_name_or_path", "meta-llama/Llama-2-7b-hf"),
        r=int(ada["r"]),
        lora_alpha=int(ada["lora_alpha"]),
        lora_dropout=float(ada.get("lora_dropout", 0.05)),
        lora_nums=int(ada["lora_nums"]),
        target_modules=list(ada["target_modules"]),
    )

    print(f"[load] {hydra_cfg.base_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        hydra_cfg.base_model_name_or_path,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", trust_remote_code=True,
    )
    model = inject_hydralora_into_llama(model, hydra_cfg)

    weights_path = hf_hub_download(repo_id=HF_REPO, filename="adapter_model.safetensors", token=token)
    sd = safe_load_file(weights_path)
    missing, unexpected = load_hydralora_state_dict(model, sd)
    print(f"[adapter] loaded {len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print(f"[warn] unexpected keys (first 5): {unexpected[:5]}")

    model.eval()
    acc = run_offline_eval(
        model, tokenizer,
        batch_size=EVAL_BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(f"\n{'='*55}\n  GSM8K Accuracy [HydraLoRA] = {acc * 100:.2f}%\n{'='*55}")


if __name__ == "__main__":
    main()
