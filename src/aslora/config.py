from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional, Dict, List, Any

import torch
import yaml

from mixlora.config import MixLoraConfig


def create_mixlora_config_for_llama(
    base_model: str = "meta-llama/Llama-2-7b-hf",
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_experts: int = 8,
    top_k: int = 2,
    router_aux_loss_coef: float = 0.001,
    target_modules: Optional[Dict[str, bool]] = None,

    # Sparsity Params
    target_sparsity: float = 0.3,
    models_path: str = "/root/simpa/models/Llama-2-7B",
    gate_rank: int = 16,
    branch_mode: str = "sparse_only",
    mask_type: str = "magnitude",
    block_aligned: bool = False,
    block_size: int = 32,
):
    if target_modules is None:
        target_modules = {
            "q_proj": True, "k_proj": True, "v_proj": True, "o_proj": True,
            "gate_proj": True, "up_proj": True, "down_proj": True,
        }

    config_dict = {
        "peft_type": "MIXLORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "routing_strategy": "mixlora",
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": [k for k, v in target_modules.items() if v],
        "num_experts": num_experts,
        "top_k": top_k,
        "router_init_range": 0.02,
        "jitter_noise": 0.0,
        "router_loss": True,
        "router_aux_loss_coef": router_aux_loss_coef,
    }

    config = MixLoraConfig.from_config(config_dict)

    if config.target_modules_ is None:
        config.target_modules_ = target_modules
    else:
        for k, v in target_modules.items():
            if v and k not in config.target_modules_:
                config.target_modules_[k] = True

    config.target_sparsity = target_sparsity
    config.models_path = models_path
    config.gate_rank = gate_rank
    config.mask_type = mask_type
    config.block_aligned = block_aligned
    config.block_size = block_size
    config.adapter_name_ = "default"
    config.dtype_ = torch.float32
    config.act_fn_ = "silu"
    config.top_k_ = 1
    config.branch_mode = branch_mode
    return config


@dataclass
class ModelCfg:
    name: str = "meta-llama/Llama-2-7b-hf"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"


@dataclass
class LoraCfg:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class MoeCfg:
    num_experts: int = 8
    top_k: int = 2
    router_aux_loss_coef: float = 0.001
    gate_rank: int = 16


@dataclass
class SparsityCfg:
    target_sparsity: float = 0.3
    mask_type: str = "magnitude"  # magnitude | random | none
    block_aligned: bool = False
    block_size: int = 32


@dataclass
class DataCfg:
    dataset: str = "meta-math/MetaMathQA"
    max_length: int = 1024
    num_samples: int = 100000
    cache_dir: str = "data_cache"


@dataclass
class TrainCfg:
    per_device_train_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    optim: str = "adamw_torch_fused"
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    tf32: bool = True
    logging_steps: int = 50
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 8
    max_steps: int = -1
    seed: int = 42


@dataclass
class EvalCfg:
    gsm8k_batch_size: int = 4
    max_new_tokens: int = 512
    arc_easy_batch_size: int = 4


@dataclass
class HubCfg:
    username: str = "twanghcmut"
    repo_template: str = "{model_short}-simpa-{branch_mode}"
    push_after_train: bool = True


@dataclass
class WandbCfg:
    project: str = "simpa-emnlp"
    run_name_template: str = "{model_short}-{branch_mode}"


@dataclass
class ExperimentConfig:
    model: ModelCfg = field(default_factory=ModelCfg)
    lora: LoraCfg = field(default_factory=LoraCfg)
    moe: MoeCfg = field(default_factory=MoeCfg)
    sparsity: SparsityCfg = field(default_factory=SparsityCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    hub: HubCfg = field(default_factory=HubCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)
    branch_mode: str = "sparse_only"


def _instantiate_from_dict(cls, data: Optional[dict]):
    if data is None:
        return cls()
    field_map = {f.name: f for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for k, v in data.items():
        if k not in field_map:
            continue
        f = field_map[k]
        if is_dataclass(f.type) if isinstance(f.type, type) else False:
            kwargs[k] = _instantiate_from_dict(f.type, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def _build_experiment_config(raw: dict) -> ExperimentConfig:
    section_map = {
        "model": ModelCfg,
        "lora": LoraCfg,
        "moe": MoeCfg,
        "sparsity": SparsityCfg,
        "data": DataCfg,
        "train": TrainCfg,
        "eval": EvalCfg,
        "hub": HubCfg,
        "wandb": WandbCfg,
    }
    kwargs: Dict[str, Any] = {}
    for name, cls in section_map.items():
        kwargs[name] = _instantiate_from_dict(cls, raw.get(name))
    if "branch_mode" in raw:
        kwargs["branch_mode"] = raw["branch_mode"]
    return ExperimentConfig(**kwargs)


def _apply_override(cfg: ExperimentConfig, key: str, value: Any) -> None:
    parts = key.split(".")
    obj: Any = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def load_experiment_config(yaml_path: str, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _build_experiment_config(raw)
    if overrides:
        for ov in overrides:
            if "=" not in ov:
                continue
            k, v = ov.split("=", 1)
            parsed = yaml.safe_load(v)
            _apply_override(cfg, k.strip(), parsed)
    return cfg
