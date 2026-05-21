import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainingArguments,
)
from transformers.modeling_utils import load_sharded_checkpoint

from aslora.config import (
    ExperimentConfig,
    create_mixlora_config_for_llama,
    load_experiment_config,
)
from aslora.data.metamath import create_meta_math_dataset
from aslora.eval.gsm8k import GSM8KEvalCallback, run_offline_eval
from aslora.modules.injection import (
    initialize_mixlora_weights_for_llama,
    inject_mixlora_into_llama,
)
from aslora.secrets import load_secrets, login_hf_and_wandb
from aslora.training.freezing import (
    count_trainable_parameters,
    freeze_base_model,
    verify_freezing,
)
from aslora.training.saving import (
    push_to_hub_with_card,
    save_full_adapter_for_resume,
    save_mixlora_adapter_safetensors,
)
from aslora.training.trainer import MixLoraTrainer


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_short(name: str) -> str:
    return name.split("/")[-1]


def _resolve_dtype(name: str):
    return _DTYPES.get(str(name).lower(), torch.bfloat16)


class EpochSaveCallback(TrainerCallback):
    def __init__(self, model, config, run_dir: str):
        self.model = model
        self.config = config
        self.run_dir = run_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        save_dir = os.path.join(self.run_dir, f"epoch_{epoch}")
        save_mixlora_adapter_safetensors(self.model, self.config, save_dir)


def _build_mixlora_config(cfg: ExperimentConfig):
    return create_mixlora_config_for_llama(
        base_model=cfg.model.name,
        target_sparsity=cfg.sparsity.target_sparsity,
        branch_mode=cfg.branch_mode,
        gate_rank=cfg.moe.gate_rank,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        num_experts=cfg.moe.num_experts,
        top_k=cfg.moe.top_k,
        router_aux_loss_coef=cfg.moe.router_aux_loss_coef,
    )


def _build_training_args(cfg: ExperimentConfig, run_name: str, output_dir: str) -> TrainingArguments:
    t = cfg.train
    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_train_epochs=t.num_train_epochs,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        warmup_ratio=t.warmup_ratio,
        optim=t.optim,
        lr_scheduler_type=t.lr_scheduler_type,
        bf16=t.bf16,
        tf32=t.tf32,
        logging_steps=t.logging_steps,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=t.dataloader_num_workers,
        dataloader_pin_memory=True,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
        save_strategy="no",
    )
    if getattr(t, "max_steps", None):
        kwargs["max_steps"] = t.max_steps
    return TrainingArguments(**kwargs)


def cmd_train(args: argparse.Namespace) -> None:
    cfg: ExperimentConfig = load_experiment_config(args.config, overrides=args.override or [])

    _set_seed(42)

    secrets = load_secrets()
    login_hf_and_wandb(secrets)

    if cfg.wandb.project:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)

    model_short = _model_short(cfg.model.name)
    run_name = cfg.wandb.run_name_template.format(model_short=model_short, branch_mode=cfg.branch_mode)
    run_dir = os.path.join("outputs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    dtype = _resolve_dtype(cfg.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mixlora_config = _build_mixlora_config(cfg)
    mixlora_weights = initialize_mixlora_weights_for_llama(model, mixlora_config)
    model = inject_mixlora_into_llama(model, mixlora_config, mixlora_weights)

    model = freeze_base_model(model)
    verify_freezing(model)
    count_trainable_parameters(model)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    train_dataset = create_meta_math_dataset(
        tokenizer,
        max_length=cfg.data.max_length,
        dataset_name=cfg.data.dataset,
        num_samples=cfg.data.num_samples,
        cache_dir=cfg.data.cache_dir,
    )

    training_args = _build_training_args(cfg, run_name=run_name, output_dir=run_dir)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    epoch_save_cb = EpochSaveCallback(model=model, config=mixlora_config, run_dir=run_dir)

    trainer = MixLoraTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[epoch_save_cb],
    )

    trainer.train()

    final_dir = os.path.join(run_dir, "final")
    save_full_adapter_for_resume(model, mixlora_config, tokenizer, final_dir)

    # Free training state before eval so we can use a larger batch size without OOM.
    global_step = int(trainer.state.global_step)
    try:
        if getattr(trainer, "optimizer", None) is not None:
            del trainer.optimizer
            trainer.optimizer = None
        if getattr(trainer, "lr_scheduler", None) is not None:
            del trainer.lr_scheduler
            trainer.lr_scheduler = None
    except Exception:
        pass
    for p in model.parameters():
        p.grad = None
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    model.eval()
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gsm8k_callback = GSM8KEvalCallback(
        tokenizer,
        eval_batch_size=cfg.eval.gsm8k_batch_size,
        max_new_tokens=cfg.eval.max_new_tokens,
    )
    last_acc = None
    try:
        last_acc = float(gsm8k_callback.evaluate(model))
        print(f"\n{'='*50}\nFinal GSM8K Accuracy = {100*last_acc:.2f}%\n{'='*50}\n")
    except Exception as e:
        print(f"GSM8K eval failed: {e!r}")

    if cfg.hub.push_after_train and secrets.get("HF_TOKEN"):
        repo_name = cfg.hub.repo_template.format(model_short=model_short, branch_mode=cfg.branch_mode)
        repo_id = f"{cfg.hub.username}/{repo_name}"
        push_to_hub_with_card(
            local_dir=final_dir,
            repo_id=repo_id,
            branch_mode=cfg.branch_mode,
            dataset_name=cfg.data.dataset,
            num_steps=global_step,
            gsm8k_accuracy=last_acc,
            token=secrets["HF_TOKEN"],
            private=secrets.get("HF_PRIVATE", False),
        )


def _resolve_adapter_path(adapter: str, token: str | None) -> str:
    if os.path.isdir(adapter):
        return adapter
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=adapter, token=token or None)


def _load_adapter_overrides(cfg: ExperimentConfig, adapter_path: str) -> None:
    adapter_cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        if "branch_mode" in adapter_cfg:
            cfg.branch_mode = adapter_cfg["branch_mode"]
        if "target_sparsity" in adapter_cfg:
            cfg.sparsity.target_sparsity = float(adapter_cfg["target_sparsity"])
        if "gate_rank" in adapter_cfg:
            cfg.moe.gate_rank = int(adapter_cfg["gate_rank"])


def cmd_eval(args: argparse.Namespace) -> None:
    cfg: ExperimentConfig = load_experiment_config(args.config, overrides=args.override or [])

    _set_seed(42)
    secrets = load_secrets()
    login_hf_and_wandb(secrets)

    adapter_path = _resolve_adapter_path(args.adapter, secrets.get("HF_TOKEN") or None)
    _load_adapter_overrides(cfg, adapter_path)

    dtype = _resolve_dtype(cfg.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mixlora_config = _build_mixlora_config(cfg)
    mixlora_weights = initialize_mixlora_weights_for_llama(model, mixlora_config)
    model = inject_mixlora_into_llama(model, mixlora_config, mixlora_weights)

    load_sharded_checkpoint(model, adapter_path, strict=False)
    model.eval()

    acc = run_offline_eval(model, tokenizer, batch_size=cfg.eval.gsm8k_batch_size)
    print(f"Final GSM8K accuracy: {acc:.4f}")


def cmd_measure(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForSeq2Seq
    from aslora.eval.jaccard_e1 import measure_jaccard_e1

    cfg: ExperimentConfig = load_experiment_config(args.config, overrides=args.override or [])
    _set_seed(42)
    secrets = load_secrets()
    login_hf_and_wandb(secrets)

    adapter_path = _resolve_adapter_path(args.adapter, secrets.get("HF_TOKEN") or None)
    _load_adapter_overrides(cfg, adapter_path)

    dtype = _resolve_dtype(cfg.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mixlora_config = _build_mixlora_config(cfg)
    mixlora_weights = initialize_mixlora_weights_for_llama(model, mixlora_config)
    model = inject_mixlora_into_llama(model, mixlora_config, mixlora_weights)
    load_sharded_checkpoint(model, adapter_path, strict=False)
    model.eval()

    dataset = create_meta_math_dataset(
        tokenizer,
        max_length=cfg.data.max_length,
        dataset_name=cfg.data.dataset,
        num_samples=args.n_samples,
        cache_dir=cfg.data.cache_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8),
        shuffle=False,
    )

    p = args.p
    result = measure_jaccard_e1(model, dataloader, p=p, n_batches=args.n_batches, n_pairs=args.n_pairs)

    d = 4096
    k = max(1, int((1 - p) * d))
    bound = (k * k) / (2 * k * d - k * k)

    print(f"\n{'='*55}")
    print(f"  Jaccard E1 — branch_mode={cfg.branch_mode}, p={p}")
    print(f"{'='*55}")
    print(f"  Mean Jaccard (cross-expert):  {result['mean_jaccard']:.4f}")
    print(f"  Theoretical bound:            {bound:.4f}")
    print(f"  Assumption holds (J <= bound): {result['mean_jaccard'] <= bound}")
    print(f"  Batches processed:            {result['n_batches_processed']}")
    print(f"\n  Per-layer breakdown:")
    for layer_name, j in sorted(result["per_layer"].items()):
        print(f"    {layer_name:50s}  J={j:.4f}")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(prog="aslora")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--override", nargs="*", default=[])
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--adapter", required=True)
    p_eval.add_argument("--override", nargs="*", default=[])
    p_eval.set_defaults(func=cmd_eval)

    p_measure = sub.add_parser("measure")
    p_measure.add_argument("--config", required=True)
    p_measure.add_argument("--adapter", required=True)
    p_measure.add_argument("--p", type=float, default=0.7)
    p_measure.add_argument("--n-batches", type=int, default=20, dest="n_batches")
    p_measure.add_argument("--n-pairs", type=int, default=200, dest="n_pairs")
    p_measure.add_argument("--n-samples", type=int, default=2048, dest="n_samples")
    p_measure.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    p_measure.add_argument("--override", nargs="*", default=[])
    p_measure.set_defaults(func=cmd_measure)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
