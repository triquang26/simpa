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

# block-aligned magnitude mask (paper setting: block size 32)
uv run aslora train --config configs/dual_scalar_gate.yaml \
  --override sparsity.block_aligned=true sparsity.block_size=32
```

Sparsity options (`sparsity:` section of the config, saved into `adapter_config.json`):

| key | default | meaning |
|---|---|---|
| `target_sparsity` | `0.3` | fraction `p` of the input dropped before the routed LoRA branch |
| `block_aligned` | `false` | rank and keep contiguous blocks instead of single coordinates |
| `block_size` | `32` | block width `B` when `block_aligned` is on |
| `mask_type` | `magnitude` | `magnitude`, `random` (control at the same density), or `none` |

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
- `scripts/` — training launchers and measurement scripts (Jaccard, Hill estimator).
- `benchmarks/` — standalone Triton kernel benchmark.
- `tests/` — CPU-only unit tests.

## Block-Aligned Mask and Triton Kernel

### 1. The mask

For a token's hidden state $x \in \mathbb{R}^d$, split the $d$ coordinates into $n = d/B$
contiguous blocks of width $B$ (default $B = 32$). Score each block by its energy and keep the
top $k_b = \mathrm{round}\big((1-p)\,n\big)$ blocks:

$$
s_b = \lVert x_{[bB:(b+1)B]} \rVert_2^2, \qquad
\mathcal{K}(x) = \mathrm{TopK}_{k_b}(s_1, \dots, s_n), \qquad
\tilde{x}_j = x_j \cdot \mathbb{1}\big[\lfloor j/B \rfloor \in \mathcal{K}(x)\big].
$$

The sparse routed-LoRA branch sees $\tilde{x}$; the parallel dense LoRA branch still sees $x$.
Since $\mathcal{K}(x)$ depends on the token, different tokens (and therefore the experts they
are routed to) see different slices of the input.

```
x (d = 4096)   [b0][b1][b2][b3][b4][b5] ... [b127]     128 blocks of 32 coordinates
block energy    .2  3.1  .1  2.4  .3  1.9 ...   .1
p = 0.7         --  [b1] --  [b3] --  [b5] ...  --     keep top round(0.3*128) = 38 blocks
```

Training-side implementation: `SparsifyModule._block_mask` in
[`src/aslora/modules/sparsify.py`](src/aslora/modules/sparsify.py).

### 2. Why it is cheap: only kept rows of W are read

For a projection $W \in \mathbb{R}^{d \times h}$, zeroed coordinates contribute nothing, so

$$
y = \tilde{x}\,W = \sum_{j \in \mathrm{kept}(x)} x_j\, W_{j,:} .
$$

Instead of a dense matmul over all $d$ rows, the kernel gathers only the $(1-p)\,d$ kept rows.
With the block-aligned mask those rows come in runs of $B$ consecutive rows, so every kept block
is one contiguous slab of $W$ in memory rather than scattered single rows.

```
W (d x h, row-major)
row   0..31   [..............]  dropped  -> never loaded
row  32..63   [##############]  kept     -> one contiguous read
row  64..95   [..............]  dropped
row  96..127  [##############]  kept     -> one contiguous read
...
```

### 3. The kernel

[`benchmarks/triton_sparse_moe_benchmark.py`](benchmarks/triton_sparse_moe_benchmark.py)

| piece | role |
|---|---|
| `BlockColumnSparsifyModule` | builds the block-aligned mask in `(values, indices)` form: `values[t] = x[t, kept]`, `indices[t]` = kept coordinates, blocks sorted by position |
| `simpa_sparse_indexed_gemm_kernel` | grid = `(tokens, ceil(h / block_out))`; each program loads one token's `k_keep` (value, index) pairs, gathers `W[indices, tile]`, and reduces in fp32 to produce `y[t, tile]` |
| `simpa_linear` | launch wrapper (`block_out = 32`, `num_warps = 4`) |
| `SIMPASparseMoe` | routes tokens to experts and calls `simpa_linear` per expert, vs. `DenseMoeBaseline` (dense PyTorch matmul) |

At $p = 0.7$ the kernel executes 30% of the dense FLOPs and gives a **1.46× end-to-end
inference speedup**. The gain is below the FLOP ratio ($1/0.3 \approx 3.3\times$) because the
indexed gather is memory-bound: dropping rows cuts arithmetic faster than it cuts memory traffic.

```bash
uv run python benchmarks/triton_sparse_moe_benchmark.py                 # block-aligned, B = 32
uv run python benchmarks/triton_sparse_moe_benchmark.py --block-size 1  # coordinate-wise mask
```

Requirements: CUDA GPU, a PyTorch CUDA build, and Triton. The benchmark does not import the
training package. CPU-only tests check that the kernel-side mask matches the training-side mask
and that the indexed product equals $\tilde{x}W$:

```bash
uv run python tests/test_kernel_mask.py
uv run python tests/test_sparsify_masks.py
```

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
