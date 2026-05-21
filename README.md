# ASLoRA2 — Sparse / Dense / Dual-Gate LoRA-MoE (EMNLP)

Refactor of `ASLoRA2.ipynb` into a reproducible research repo. Three ablations:

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

After training, the adapter is pushed to `twanghcmut/<model>-aslora-<branch_mode>` on the HF
Hub. To evaluate a saved adapter (from HF Hub or a local directory):

```bash
uv run aslora eval --config configs/sparse_only.yaml \
  --adapter twanghcmut/Llama-2-7b-hf-aslora-sparse_only
```

`--adapter` accepts an HF repo id or a local path. The eval skeleton infers `branch_mode`
from the adapter's `adapter_config.json`.

## Layout

- `configs/` — YAML configs (base + one per ablation).
- `src/aslora/` — package source (config, modules, data, training, eval, CLI).
- `third_party/mixlora/` — vendored copy of [TUDB-Labs/MixLoRA](https://github.com/TUDB-Labs/MixLoRA),
  installed as an editable path dep.
- `scripts/` — convenience launchers wrapping the three `uv run aslora train ...` commands.

See `PLAN.md` for the full design notes.
