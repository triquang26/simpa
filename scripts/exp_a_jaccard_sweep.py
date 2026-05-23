"""
Experiment A: Jaccard + Retention Ratio sweep on SIMPA (column-wise output_gate).

Measures J̄_p and ρ(p) for p ∈ {0.3, 0.5, 0.7, 0.8, 0.9}.
Single forward pass per batch — all p values computed from same last_hidden.

Outputs:
  {out_prefix}_results.json   — per-layer and mean J̄_p, ρ(p), S, U per p
  {out_prefix}_table.md       — Table 14-equivalent markdown

Usage:
    cd /mnt/data/sftp/data/quangpt3/emnlp
    uv run python scripts/exp_a_jaccard_sweep.py
    uv run python scripts/exp_a_jaccard_sweep.py --adapter /path/to/local/ckpt
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "mixlora"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from measure_jaccard_aslora import MixLoraSparseMoe, inject_full_simpa
from aslora.data.metamath import create_meta_math_dataset
from aslora.secrets import load_secrets, login_hf_and_wandb

HIDDEN_SIZE = 4096
P_SWEEP = [0.3, 0.5, 0.7, 0.8, 0.9]
DEFAULT_REPO_ID = "twanghcmut/Llama-2-7b-hf-gsm8k"


def theoretical_bound(p: float, d: int = HIDDEN_SIZE) -> float:
    k = max(1, int((1 - p) * d))
    return (k * k) / (2 * k * d - k * k)


def run_sweep(
    model: torch.nn.Module,
    dataloader: DataLoader,
    p_values: list[float],
    n_batches: int,
    n_pairs: int,
) -> list[dict]:
    moe_modules = [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, MixLoraSparseMoe)
    ]
    if not moe_modules:
        raise RuntimeError("No MixLoraSparseMoe modules found — was inject_full_simpa called?")

    jac_scores = {name: {p: [] for p in p_values} for name, _ in moe_modules}
    rho_accum  = {name: {p: {"num": 0.0, "den": 0.0} for p in p_values} for name, _ in moe_modules}

    device = next(model.parameters()).device
    model.eval()
    processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if processed >= n_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            B, seq_len = batch["attention_mask"].shape
            valid_mask = batch["attention_mask"].view(-1).bool()  # [T]

            model(**batch)
            processed += 1
            print(f"  [batch {processed}/{n_batches}] computing metrics...", flush=True)

            for name, moe in moe_modules:
                hidden  = moe.last_hidden   # [T, d] GPU bf16
                routing = moe.last_routing  # [T, K] GPU int64
                if hidden is None or routing is None:
                    continue

                T, d = hidden.shape
                K = routing.shape[1]
                hidden_f = hidden.float()
                seq_idx = torch.arange(T, device=device) // seq_len

                # Pre-compute expert→token lists once (shared across p values)
                routing_flat = routing.view(-1)
                tok_rep   = torch.arange(T, device=device).repeat_interleave(K)
                valid_rep = valid_mask[tok_rep]
                expert_lists: dict[int, torch.Tensor] = {}
                for e in routing_flat.unique().tolist():
                    sel = (routing_flat == e) & valid_rep
                    expert_lists[e] = tok_rep[sel]
                experts_sorted = sorted(expert_lists)

                for p in p_values:
                    k_keep = max(1, int((1 - p) * d))

                    _, topk_idx = torch.topk(hidden_f.abs(), k_keep, dim=-1)  # [T, k]
                    mask = torch.zeros(T, d, dtype=torch.bool, device=device)
                    mask.scatter_(1, topk_idx, True)  # [T, d]

                    # ρ(p)
                    x_sp = hidden_f * mask.float()
                    rho_accum[name][p]["num"] += x_sp.pow(2).sum().item()
                    rho_accum[name][p]["den"] += hidden_f.pow(2).sum().item()

                    # Cross-prompt Jaccard
                    pair_scores: list[float] = []
                    for i in range(len(experts_sorted)):
                        for j in range(i + 1, len(experts_sorted)):
                            ti = expert_lists[experts_sorted[i]]
                            tj = expert_lists[experts_sorted[j]]
                            if len(ti) == 0 or len(tj) == 0:
                                continue
                            n_cand = min(n_pairs * 4, len(ti) * len(tj))
                            ii = ti[torch.randint(len(ti), (n_cand,), device=device)]
                            jj = tj[torch.randint(len(tj), (n_cand,), device=device)]
                            cross = seq_idx[ii] != seq_idx[jj]
                            ii = ii[cross][:n_pairs]
                            jj = jj[cross][:n_pairs]
                            if len(ii) == 0:
                                continue
                            inter = (mask[ii] & mask[jj]).sum(-1).float()
                            pair_scores.append((inter / (2 * k_keep - inter)).mean().item())

                    if pair_scores:
                        jac_scores[name][p].append(float(np.mean(pair_scores)))

                    del mask, topk_idx, x_sp

                moe.last_hidden  = None
                moe.last_routing = None

            torch.cuda.empty_cache()

    results = []
    for p in p_values:
        per_j = {
            n: float(np.mean(v))
            for n, v in jac_scores.items() if jac_scores[n][p]
        }
        per_rho = {
            n: acc[p]["num"] / acc[p]["den"] if acc[p]["den"] > 0 else float("nan")
            for n, acc in rho_accum.items()
        }
        mean_j   = float(np.mean(list(per_j.values()))) if per_j else float("nan")
        mean_rho = float(np.mean(list(per_rho.values())))
        results.append({
            "p":                  p,
            "mean_jaccard":       mean_j,
            "mean_rho":           mean_rho,
            "S":                  1 - mean_j,
            "U":                  (1 - mean_j) * mean_rho,
            "theoretical_bound":  theoretical_bound(p),
            "per_layer_jaccard":  per_j,
            "per_layer_rho":      per_rho,
        })
    return results


def write_json(results: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON → {path}")


def write_markdown_table(results: list[dict], path: str) -> None:
    lines = [
        "| p   | J̄_p   | ρ(p)  | S(p)=1−J̄ | U(p)=S·ρ | Bound  |",
        "|-----|--------|-------|-----------|-----------|--------|",
    ]
    for r in results:
        lines.append(
            f"| {r['p']:.1f} | {r['mean_jaccard']:.4f} | {r['mean_rho']:.3f}"
            f" | {r['S']:.4f}    | {r['U']:.4f}    | {r['theoretical_bound']:.4f} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved table → {path}")


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 62)
    print("  Experiment A — SIMPA (column-wise output_gate)")
    print("=" * 62)
    print(f"  {'p':>5}  {'J̄':>8}  {'ρ(p)':>8}  {'S=1-J̄':>8}  {'U=S·ρ':>8}  {'Bound':>8}")
    print("  " + "-" * 58)
    for r in results:
        marker = " ←" if r["U"] == max(x["U"] for x in results) else ""
        print(
            f"  {r['p']:>5.1f}  {r['mean_jaccard']:>8.4f}  {r['mean_rho']:>8.4f}"
            f"  {r['S']:>8.4f}  {r['U']:>8.4f}  {r['theoretical_bound']:>8.4f}{marker}"
        )
    best = max(results, key=lambda r: r["U"])
    print("=" * 62)
    print(f"  argmax U(p) = {best['p']}  (U = {best['U']:.4f})")
    print("=" * 62 + "\n")


def load_checkpoint(ckpt_dir: str, model: torch.nn.Module) -> None:
    idx_path = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    if not os.path.exists(idx_path):
        # Try safetensors fallback
        from safetensors.torch import load_file as safe_load
        st_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
        if os.path.exists(st_path):
            sd = safe_load(st_path)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"  Loaded safetensors: {len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
            return
        raise FileNotFoundError(f"No pytorch_model.bin.index.json or adapter_model.safetensors in {ckpt_dir}")

    with open(idx_path) as f:
        idx = json.load(f)
    shards = sorted(set(idx["weight_map"].values()))
    state_dict: dict[str, torch.Tensor] = {}
    for sf in shards:
        print(f"  loading shard {sf}...", flush=True)
        shard = torch.load(os.path.join(ckpt_dir, sf), map_location="cpu", weights_only=True)
        state_dict.update(shard)
        del shard
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded {len(state_dict)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"  first missing: {missing[:3]}")
    del state_dict
    gc.collect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment A: Jaccard + retention ratio sweep")
    p.add_argument("--adapter", default=None,
                   help="Local checkpoint path or HF repo ID (default: twanghcmut/Llama-2-7b-hf-gsm8k)")
    p.add_argument("--n-batches", type=int, default=20, dest="n_batches")
    p.add_argument("--n-pairs",   type=int, default=200, dest="n_pairs")
    p.add_argument("--out-prefix", default="runs/exp_a", dest="out_prefix")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    secrets = load_secrets()
    login_hf_and_wandb(secrets)

    print("[1/5] Loading base Llama-2-7b-hf...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] Injecting SIMPA architecture (column-wise output_gate)...")
    inject_full_simpa(model)
    device = next(model.parameters()).device
    model.to(device=device, dtype=torch.bfloat16)

    adapter = args.adapter or DEFAULT_REPO_ID
    print(f"[3/5] Loading checkpoint: {adapter}")
    if os.path.isdir(adapter):
        ckpt_dir = adapter
    else:
        from huggingface_hub import snapshot_download
        ckpt_dir = snapshot_download(repo_id=adapter, token=secrets.get("HF_TOKEN") or None)
    load_checkpoint(ckpt_dir, model)
    torch.cuda.empty_cache()
    model.eval()

    print("[4/5] Building MetaMathQA dataloader...")
    dataset = create_meta_math_dataset(
        tokenizer,
        max_length=1024,
        dataset_name="meta-math/MetaMathQA",
        num_samples=2048,
        cache_dir="data_cache",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8),
        shuffle=False,
    )

    print(f"[5/5] Running sweep over p={P_SWEEP} ({args.n_batches} batches × 8 seqs)...")
    results = run_sweep(model, dataloader, p_values=P_SWEEP, n_batches=args.n_batches, n_pairs=args.n_pairs)

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    write_json(results,           f"{args.out_prefix}_results.json")
    write_markdown_table(results, f"{args.out_prefix}_table.md")
    print_table(results)


if __name__ == "__main__":
    main()
