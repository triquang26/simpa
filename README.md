# SIMPA — Sparse / Dense / Dual-Gate LoRA-MoE (EMNLP)

SIMPA is a reproducible research repo for sparse / dense / dual-gate LoRA-MoE
experiments. Three ablations:

1. **Sparse-only** — LoRA path operates on a top-k sparsified input.
2. **Dense-only** — Standard LoRA (sparsify bypassed).
3. **Dual scalar gate** — Mixes dense and sparse LoRA outputs via an input-dependent scalar
   gate `g = sigmoid(linear(x))`.

## Quickstart

```bash
# 1. install deps (creates .venv, installs vendored mixlora as path dep)
uv sync

# 2. set up secrets (HF_TOKEN, WANDB_API_KEY, etc.)
cp .env.example .env
$EDITOR .env

# 3. run the three ablations
uv run aslora train --config configs/sparse_only.yaml
uv run aslora train --config configs/dense_only.yaml
uv run aslora train --config configs/dual_scalar_gate.yaml
```

## Evaluation

After training, the adapter is pushed to `twanghcmut/<model>-simpa-<branch_mode>` on the HF
Hub. To evaluate a saved adapter (from HF Hub or a local directory):

```bash
uv run aslora eval --config configs/sparse_only.yaml \
  --adapter twanghcmut/Llama-2-7b-hf-simpa-sparse_only
```

`--adapter` accepts an HF repo id or a local path. The eval skeleton infers `branch_mode`
from the adapter's `adapter_config.json`.

## Layout

- `configs/` — YAML configs (base + one per ablation).
- `src/aslora/` — package source (config, modules, data, training, eval, CLI).
- `third_party/mixlora/` — vendored copy of [TUDB-Labs/MixLoRA](https://github.com/TUDB-Labs/MixLoRA),
  installed as an editable path dep.
- `scripts/` — convenience launchers wrapping the three `uv run aslora train ...` commands.

## Triton Sparse MoE Benchmark

The standalone Triton benchmark lives at `benchmarks/triton_sparse_moe_benchmark.py`.
It does not import the training package, so it can be inspected or run independently on a GPU
machine.

```bash
uv run python benchmarks/triton_sparse_moe_benchmark.py
```

Requirements: CUDA GPU, a PyTorch CUDA build, and Triton. This repo update did not run the
benchmark locally because no GPU is available in the current environment.

See `PLAN.md` for the full design notes.
