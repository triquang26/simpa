<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/emnlp2026-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/emnlp2026.png">
  <img src="assets/emnlp2026.png" alt="EMNLP 2026" width="640">
</picture>

# SIMPA: Restricting What Experts See, Not What They Produce, in MoE-PEFT

**S**parse-**I**nput **M**agnitude **P**artitioning **A**daptation

<p>
<b>Pham Tri Quang</b><sup>1*</sup> &nbsp;
<b>Khoa Tran</b><sup>1</sup> &nbsp;
<b>Giang Do</b><sup>2</sup> &nbsp;
<b>Hung Le</b><sup>2</sup> &nbsp;
<b>Truyen Tran</b><sup>2</sup> &nbsp;
<b>Tho T. Quan</b><sup>1†</sup>
</p>

<sup>1</sup>Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT), VNU-HCM, Vietnam<br>
<sup>2</sup>Applied Artificial Intelligence Initiative (A2I2), Deakin University, Australia

<p>
<a href="https://2026.emnlp.org/"><img src="https://img.shields.io/badge/EMNLP_2026-Main_Conference-ed1c24?style=for-the-badge" alt="EMNLP 2026 Main"></a>
<a href="https://aclanthology.org/2026.emnlp-main.726.pdf"><img src="https://img.shields.io/badge/Paper-ACL_Anthology-1f6feb?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Paper"></a>
<a href="https://huggingface.co/twanghcmut"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Checkpoints-ffd21e?style=for-the-badge" alt="Hugging Face"></a>
<img src="https://img.shields.io/badge/Python-3.12-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
</p>

</div>

## 🔥 News

- **[2026/10]** 🎉 SIMPA is accepted to the **EMNLP 2026 Main Conference**!
- **[2026/10]** Code released. Paper available on [ACL Anthology](https://aclanthology.org/2026.emnlp-main.726.pdf).

## Overview

MoE routing collapses in parameter-efficient fine-tuning: on LLaMA-3 8B with MixLoRA, a single
expert absorbs 82.5% of tokens. Prior MoE-PEFT remedies intervene at the *output*. **SIMPA inverts
the intervention site: rather than constraining what experts *produce*, it restricts what they
*see*.** A magnitude-thresholded input drives a sparse routed-LoRA branch, a parallel dense LoRA
branch (a DeepSeekMoE-style shared expert adapted to PEFT) acts as an information safeguard, and
a column-wise gate fuses the two.

- **Accuracy** at a matched 2.9% trainable-parameter budget: 86.2% avg on seven LLaMA-3 8B
  reasoning tasks (+3.9 over MixDoRA), 38.4 HumanEval Pass@1 / 59.4 GSM8K, and 88.7 GLUE /
  77.7 SuperGLUE on T5-base.
- **Expert diversity**: expert-input Jaccard of 0.184 vs. MixLoRA's 1.000 (5.4× reduction).
- **Speed**: the block-aligned mask drives a Triton kernel with 1.46× end-to-end inference
  speedup on A100.

This repo supports three ablations:

1. **Sparse-only**: LoRA path operates on a top-k sparsified input.
2. **Dense-only**: standard LoRA (sparsify bypassed).
3. **Dual scalar gate**: mixes dense and sparse LoRA outputs via an input-dependent scalar
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

Requirements: CUDA GPU, a PyTorch CUDA build, and Triton.

## Citation

If you find SIMPA useful, please cite:

```bibtex
@inproceedings{quang-etal-2026-simpa,
  title     = {{SIMPA}: Restricting What Experts See, Not What They Produce, in {MoE-PEFT}},
  author    = {Pham, Tri Quang and Tran, Khoa and Do, Giang and Le, Hung and Tran, Truyen and Quan, Tho T.},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  url       = {https://aclanthology.org/2026.emnlp-main.726/}
}
```

## Acknowledgements

Built on top of [MixLoRA](https://github.com/TUDB-Labs/MixLoRA).
