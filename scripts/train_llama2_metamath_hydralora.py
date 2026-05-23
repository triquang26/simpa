"""Train HydraLoRA on MetaMathQA-100k with Llama-2-7b-hf, then eval GSM8K.

Matches the ASLoRA repo's canonical pipeline (configs/dual_scalar_gate.yaml,
src/aslora/cli.py::cmd_train) but swaps the parameter-efficient module from
MixLoRA/ASLoRA to HydraLoRA (Tian et al., NeurIPS 2024).

Pipeline (same as sibling scripts in this repo):
  1. Inject HydraLoRA into Llama-2-7b-hf.
  2. Train on MetaMathQA 100k samples (Alpaca template, instruction-masked).
  3. Save adapter to disk (outputs/<run>/final/).
  4. Push to HF Hub.
  5. Eval GSM8K test (batch 32) — separate script, after the push.

HydraLoRA architecture (paper) with rank scaled up to ~2.9% trainable so the
comparison vs ASLoRA is at an equal parameter budget. We keep N=4 B-heads and
the MLP-only target set (gate/up/down_proj) per the paper's design and only
scale rank: r=50 → 202.3M trainable / 6.94B ≈ 2.91%.

Run from repo root:
    uv run python scripts/train_llama2_metamath_hydralora.py
"""
import json
import os
import sys
import time

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file as safe_save_file
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from aslora.data.metamath import create_meta_math_dataset
from aslora.modules.hydralora import (
    HydraLoRAConfig,
    extract_hydralora_state_dict,
    freeze_base_for_hydralora,
    inject_hydralora_into_llama,
)
from aslora.secrets import load_secrets, login_hf_and_wandb, preflight_check
from aslora.training.freezing import count_trainable_parameters

torch.set_float32_matmul_precision("high")

# ── Config (mirrors configs/dual_scalar_gate.yaml) ─────────────────────
MODEL_ID     = "meta-llama/Llama-2-7b-hf"
HF_USERNAME  = "twanghcmut"
HF_REPO_NAME = "Llama-2-7b-hf-hydralora-r50-metamath"
RUN_DIR      = os.path.join("outputs", HF_REPO_NAME)

# HydraLoRA architecture (paper): asymmetric LoRA, N=4 B-heads, MLP-only targets
# (gate/up/down_proj). Rank scaled UP to match ASLoRA's ~2.9% trainable budget
# for an equal-parameter-budget comparison (per-layer params on Llama2-7B:
# 76800 + 123648·r; with r=50 over 32 layers → 202.3M / 6.94B ≈ 2.91%).
HYDRA_R         = 50
HYDRA_ALPHA     = 32
HYDRA_DROPOUT   = 0.05
HYDRA_NUM_HEADS = 4
HYDRA_TARGETS   = ["gate_proj", "up_proj", "down_proj"]

# MetaMathQA training params (match configs/dual_scalar_gate.yaml)
DATA_NAME     = "meta-math/MetaMathQA"
NUM_SAMPLES   = 100_000
MAX_SEQ_LEN   = 1024
NUM_EPOCHS    = 1
PER_DEVICE_BS = 32
GRAD_ACC      = 2
LR            = 2e-4
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.03
SEED          = 42
# ───────────────────────────────────────────────────────────────────────


def save_adapter(model, config: HydraLoRAConfig, save_dir: str, tokenizer):
    os.makedirs(save_dir, exist_ok=True)
    weights = extract_hydralora_state_dict(model)
    safe_save_file(weights, os.path.join(save_dir, "adapter_model.safetensors"))

    with open(os.path.join(save_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    tokenizer.save_pretrained(save_dir)
    print(f"[save] {len(weights)} tensors → {save_dir}")


def write_readme(local_dir: str, repo_id: str, config: HydraLoRAConfig,
                 trainable: int, total: int, num_steps: int):
    card = (
        f"# {repo_id}\n\n"
        f"HydraLoRA adapter for `{config.base_model_name_or_path}` fine-tuned on "
        f"MetaMathQA-100k for GSM8K-style math reasoning.\n\n"
        f"## Architecture\n\n"
        f"HydraLoRA (Tian et al., NeurIPS 2024): 1 shared A matrix + N B-heads with softmax router.\n\n"
        f"- rank `r` = {config.r}\n"
        f"- alpha = {config.lora_alpha}\n"
        f"- dropout = {config.lora_dropout}\n"
        f"- N B-heads = {config.lora_nums}\n"
        f"- target modules = {config.target_modules}\n"
        f"- trainable params = {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)\n\n"
        f"## Training\n\n"
        f"- dataset = `{DATA_NAME}` (first {NUM_SAMPLES:,} samples, Alpaca template, instruction-masked)\n"
        f"- epochs = {NUM_EPOCHS}\n"
        f"- batch size = {PER_DEVICE_BS} × grad_accum {GRAD_ACC} (effective {PER_DEVICE_BS * GRAD_ACC})\n"
        f"- lr = {LR} (cosine, warmup_ratio={WARMUP_RATIO}, wd={WEIGHT_DECAY})\n"
        f"- max_seq_length = {MAX_SEQ_LEN}\n"
        f"- steps trained = {num_steps}\n"
        f"- precision = bf16 + tf32 + gradient checkpointing\n\n"
        f"## Evaluation\n\n"
        f"Run `scripts/eval_llama2_gsm8k_hydralora.py` (batch 32) to evaluate on GSM8K test.\n"
    )
    with open(os.path.join(local_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)


def main():
    set_seed(SEED)
    os.makedirs(RUN_DIR, exist_ok=True)

    secrets = load_secrets()
    preflight_check(secrets)
    login_hf_and_wandb(secrets)
    os.environ.setdefault("WANDB_PROJECT", "aslora-hydralora")

    print(f"[load] {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hydra_cfg = HydraLoRAConfig(
        base_model_name_or_path=MODEL_ID,
        r=HYDRA_R, lora_alpha=HYDRA_ALPHA, lora_dropout=HYDRA_DROPOUT,
        lora_nums=HYDRA_NUM_HEADS, target_modules=HYDRA_TARGETS,
    )
    model = inject_hydralora_into_llama(model, hydra_cfg)
    model = freeze_base_for_hydralora(model)
    trainable, total = count_trainable_parameters(model)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print(f"[data] building {DATA_NAME} ({NUM_SAMPLES:,} samples, max_len={MAX_SEQ_LEN})")
    train_dataset = create_meta_math_dataset(
        tokenizer,
        max_length=MAX_SEQ_LEN,
        dataset_name=DATA_NAME,
        num_samples=NUM_SAMPLES,
        cache_dir="data_cache",
    )
    print(f"[data] {len(train_dataset)} samples")

    training_args = TrainingArguments(
        output_dir=RUN_DIR,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        logging_steps=50,
        logging_first_step=True,
        save_strategy="no",
        report_to="wandb" if secrets.get("WANDB_API_KEY") else "none",
        run_name=HF_REPO_NAME,
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8),
    )

    t0 = time.time()
    trainer.train()
    dur = time.time() - t0
    steps = int(trainer.state.global_step)
    print(f"\n[train] done — {dur/60:.1f} min, {steps} steps, {dur/max(steps,1):.2f} s/step")

    # ── Save locally, then push ────────────────────────────────────────
    final_dir = os.path.join(RUN_DIR, "final")
    save_adapter(model, hydra_cfg, final_dir, tokenizer)
    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    write_readme(final_dir, repo_id, hydra_cfg, trainable, total, steps)

    print(f"[hf] pushing → {repo_id}")
    token = secrets.get("HF_TOKEN") or None
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, private=secrets.get("HF_PRIVATE", False), token=token)
    api.upload_folder(
        folder_path=final_dir, repo_id=repo_id, token=token,
        commit_message=f"HydraLoRA r={HYDRA_R} N={HYDRA_NUM_HEADS} on MetaMathQA-{NUM_SAMPLES} ({steps} steps)",
    )
    print(f"[hf] done → https://huggingface.co/{repo_id}")
    print("\nNext: scripts/eval_llama2_gsm8k_hydralora.py (batch=32)")


if __name__ == "__main__":
    main()
