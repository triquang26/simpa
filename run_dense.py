# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Dense LoRA eval — load checkpoint + run GSM8K                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── CONFIG (chỉnh ở đây) ─────────────────────────────────────────────
ADAPTER_PATH    = "twanghcmut/Llama-2-7b-hf-densense-gsm8k"
ASLORA_SRC      = "/root/aslora-code/src"     # nơi chứa aslora package
MIXLORA_SRC     = "/root/aslora-code/third_party/mixlora"
EVAL_BATCH_SIZE = 64
MAX_NEW_TOKENS  = 512
# ─────────────────────────────────────────────────────────────────────

import sys, os, json
sys.path.insert(0, ASLORA_SRC)
sys.path.insert(0, MIXLORA_SRC)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load_file

from aslora.config import create_mixlora_config_for_llama
from aslora.modules.injection import (
    initialize_mixlora_weights_for_llama,
    inject_mixlora_into_llama,
)
from aslora.eval.gsm8k import run_offline_eval

# ── Đọc adapter_config.json ──────────────────────────────────────────
cfg_path = hf_hub_download(repo_id=ADAPTER_PATH, filename="adapter_config.json")
with open(cfg_path) as f:
    ada = json.load(f)

branch_mode     = ada.get("branch_mode",            "dense")
target_sparsity = float(ada.get("target_sparsity",  0.3))
gate_rank       = int(ada.get("gate_rank",          16))
base_model      = ada.get("base_model_name_or_path","meta-llama/Llama-2-7b-hf")
r               = int(ada.get("r",                  16))
lora_alpha      = int(ada.get("lora_alpha",         32))

print(f"branch_mode : {branch_mode}")
print(f"base_model  : {base_model}")
print(f"sparsity    : {target_sparsity}  |  gate_rank: {gate_rank}  |  r={r} α={lora_alpha}")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)
print("Base model loaded.")

# ── Inject MixLoRA, load adapter weights ─────────────────────────────
mixlora_cfg = create_mixlora_config_for_llama(
    base_model=base_model,
    target_sparsity=target_sparsity,
    branch_mode=branch_mode,
    gate_rank=gate_rank,
    r=r,
    lora_alpha=lora_alpha,
)
init_weights = initialize_mixlora_weights_for_llama(model, mixlora_cfg)
model = inject_mixlora_into_llama(model, mixlora_cfg, init_weights)

# Load actual trained weights
adapter_weights_path = hf_hub_download(repo_id=ADAPTER_PATH, filename="adapter_model.safetensors")
adapter_sd = safe_load_file(adapter_weights_path)
missing, unexpected = model.load_state_dict(adapter_sd, strict=False)
print(f"Adapter loaded: {len(adapter_sd)} tensors | missing={len(missing)} | unexpected={len(unexpected)}")

model.eval()

# ── GSM8K eval ────────────────────────────────────────────────────────
accuracy = run_offline_eval(
    model, tokenizer,
    batch_size=EVAL_BATCH_SIZE,
    max_new_tokens=MAX_NEW_TOKENS,
)

print(f"\n{'='*55}")
print(f"  GSM8K Accuracy  [{branch_mode}]  →  {accuracy*100:.2f}%")
print(f"{'='*55}")
