"""Measure block-norm tail index beta and predict optimal sparsity p* (SIMPA).

Loads twanghcmut/Llama-2-7b-hf-gsm8k (LLaMA-2-7B + ASLoRA SIMPA architecture),
forwards a held-out MetaMathQA slice, captures the input activation at the
7 logical adapter locations per layer (Q, K, V, O, gate_proj, up_proj, down_proj)
via forward-pre-hooks, and computes the block-norm squared distribution
(block size 32). For each (layer, location) the Hill estimator yields a tail
index beta, which feeds the closed-form p* solver.

Output: outputs/hill_pstar/hill_results.json with per-location beta_hat,
predicted p*, predicted retention at p* and at the configured grid p (default
0.7), plus per-location Hill-plot data for offline plateau diagnosis.

Run:
    uv run python scripts/measure_hill_pstar.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "mixlora"))

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from aslora.data.metamath import create_meta_math_dataset
from aslora.eval.block_norms import BlockNormAccumulator
from aslora.eval.hill import (
    hill_estimator,
    hill_plot_data,
    predicted_retention,
    sanity_check_synthetic,
    solve_pstar,
)
from aslora.eval.hook_registry import register_adapter_hooks, reset_down_fired_flags
from aslora.secrets import load_secrets, login_hf_and_wandb

from measure_jaccard_aslora import (  # noqa: E402
    P_SPARSIFY,
    REPO_ID,
    MixLoraSparseMoe,
    _make_lora_config,
    inject_full_aslora,
)


LOGICAL_NAMES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--k-frac", type=float, default=0.05)
    p.add_argument(
        "--k-frac-list",
        type=str,
        default="",
        help="Comma-separated extra k_frac values (e.g. '0.02,0.05,0.10') to "
             "additionally estimate beta_hat at, per (location,layer), for "
             "plateau diagnosis. The primary beta_hat/pstar always use --k-frac.",
    )
    p.add_argument("--max-samples-per-loc", type=int, default=2_000_000)
    p.add_argument("--n-batches-warmup", type=int, default=1)
    p.add_argument("--out", type=str, default="outputs/hill_pstar/hill_results.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-grid-optimum", type=float, default=0.7)
    p.add_argument(
        "--mode",
        choices=("trained", "pretrained"),
        default="trained",
        help="trained: load twanghcmut SIMPA checkpoint; pretrained: plain HF backbone only.",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HF base model id (used as the backbone in both modes).",
    )
    p.add_argument(
        "--auto-plot",
        action="store_true",
        help="After writing JSON, also render Hill curves / heatmap / pstar figures.",
    )
    return p.parse_args()


def _load_model(secrets: dict, mode: str, base_model: str):
    print(f"[1/4] Loading base {base_model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    if mode == "pretrained":
        print("[2/4] Pretrained mode: skipping ASLoRA injection and checkpoint load.", flush=True)
        print("[3/4] No checkpoint to load.", flush=True)
        model.eval()
        return model, tokenizer, device

    print("[2/4] Injecting full ASLoRA (attn LoRA + mlp.mixlora_moes.default + output_gate)...", flush=True)
    inject_full_aslora(model)
    model.to(device=device, dtype=torch.bfloat16)

    print(f"[3/4] Downloading checkpoint {REPO_ID}...", flush=True)
    ckpt_dir = snapshot_download(repo_id=REPO_ID, token=secrets.get("HF_TOKEN") or None)
    idx_path = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    shard_files = sorted(set(idx["weight_map"].values()))
    state_dict: dict[str, torch.Tensor] = {}
    for sf in shard_files:
        shard_path = os.path.join(ckpt_dir, sf)
        print(f"      loading {sf}...", flush=True)
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        state_dict.update(shard)
        del shard
    print(f"      Total state dict keys: {len(state_dict)}", flush=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"      missing keys:    {len(missing)}", flush=True)
    print(f"      unexpected keys: {len(unexpected)}", flush=True)
    if missing[:5]:
        print(f"      first missing:   {missing[:5]}", flush=True)
    if unexpected[:5]:
        print(f"      first unexpected: {unexpected[:5]}", flush=True)
    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    return model, tokenizer, device


def _clear_moe_buffers(model) -> None:
    for _, m in model.named_modules():
        if isinstance(m, MixLoraSparseMoe):
            m.last_hidden = None
            m.last_routing = None


def _print_beta_table(per_location: list[dict], n_layers: int) -> None:
    print()
    print("=" * 78)
    print(f"  beta_hat per (location, layer) -- {n_layers} layers")
    print("=" * 78)
    header = f"  {'location':12s} " + " ".join(f"L{i:02d}" for i in range(n_layers))
    print(header)
    grid: dict[str, dict[int, float]] = {n: {} for n in LOGICAL_NAMES}
    for row in per_location:
        grid[row["location"]][row["layer"]] = row["beta_hat"]
    for name in LOGICAL_NAMES:
        cells = []
        for i in range(n_layers):
            v = grid[name].get(i, float("nan"))
            cells.append(f"{v:4.2f}" if not np.isnan(v) else " nan")
        print(f"  {name:12s} " + " ".join(cells))
    print("=" * 78)


def main(args: argparse.Namespace) -> None:
    secrets = load_secrets()
    login_hf_and_wandb(secrets)

    print("[sanity] Running Hill synthetic check (beta_true=1.5, n=100000)...", flush=True)
    sanity = sanity_check_synthetic(beta_true=1.5, n=100_000, seed=args.seed)
    print(f"[sanity] result: {sanity}", flush=True)
    rel_err = sanity.get("rel_err")
    if rel_err is None or rel_err > 0.05:
        print(f"[sanity] FAIL: rel_err={rel_err} > 0.05; aborting", flush=True)
        sys.exit(2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, tokenizer, device = _load_model(secrets, mode=args.mode, base_model=args.base_model)

    print("[4/4] Building MetaMathQA dataloader...", flush=True)
    n_total = args.n_batches * args.batch_size + args.batch_size + args.n_batches_warmup * args.batch_size
    dataset = create_meta_math_dataset(
        tokenizer,
        max_length=1024,
        dataset_name="meta-math/MetaMathQA",
        num_samples=max(n_total, 2048),
        cache_dir="data_cache",
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False,
    )

    # Warmup forward passes WITHOUT hooks, to let caches settle.
    warmup_done = 0
    if args.n_batches_warmup > 0:
        print(f"[warmup] running {args.n_batches_warmup} warmup batch(es) without hooks...", flush=True)
        with torch.no_grad():
            for batch in dataloader:
                if warmup_done >= args.n_batches_warmup:
                    break
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                model(**batch)
                if args.mode == "trained":
                    _clear_moe_buffers(model)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                warmup_done += 1

    accumulator = BlockNormAccumulator(
        block_size=args.block_size,
        max_samples_per_loc=args.max_samples_per_loc,
        seed=args.seed,
    )
    handles, name_map = register_adapter_hooks(model, accumulator, mode=args.mode)
    print(f"[hooks] registered {len(handles)} forward_pre_hooks (mode={args.mode})", flush=True)

    processed = 0
    skipped = 0
    for batch in dataloader:
        if skipped < warmup_done:
            skipped += 1
            continue
        if processed >= args.n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        accumulator.set_attention_mask(batch.get("attention_mask"))
        reset_down_fired_flags(model)
        with torch.no_grad():
            model(**batch)
        if args.mode == "trained":
            _clear_moe_buffers(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        processed += 1
        print(f"  batch {processed}/{args.n_batches} done (total samples so far: {accumulator.total_samples()})", flush=True)

    for h in handles:
        h.remove()

    n_layers = len(model.model.layers)
    real_hidden_size = int(model.config.hidden_size)

    k_frac_list = [float(s) for s in args.k_frac_list.split(",") if s.strip()]

    per_location: list[dict] = []
    for logical_name in LOGICAL_NAMES:
        loc_keys = name_map[logical_name]
        for i, loc_key in enumerate(loc_keys):
            samples = accumulator.get(loc_key).astype(np.float64, copy=False)
            n_s = int(samples.size)
            if n_s == 0:
                beta = float("nan")
                pstar = float("nan")
                rho_at_pstar = float("nan")
                rho_at_grid = float("nan")
                hp = {"k": [], "beta_hat": []}
                beta_by_kfrac = {kf: float("nan") for kf in k_frac_list}
            else:
                beta = float(hill_estimator(samples, k_frac=args.k_frac))
                if np.isnan(beta):
                    pstar = float("nan")
                    rho_at_pstar = float("nan")
                    rho_at_grid = float("nan")
                else:
                    pstar = float(solve_pstar(beta))
                    rho_at_pstar = float(predicted_retention(beta, pstar))
                    rho_at_grid = float(predicted_retention(beta, args.p_grid_optimum))
                hp = hill_plot_data(samples, n_points=40) if n_s >= 400 else {"k": [], "beta_hat": []}
                beta_by_kfrac = {kf: float(hill_estimator(samples, k_frac=kf)) for kf in k_frac_list}
            per_location.append({
                "layer": i,
                "location": logical_name,
                "loc_key": loc_key,
                "n_samples": n_s,
                "beta_hat": beta,
                "beta_hat_by_kfrac": beta_by_kfrac,
                "pstar_pred": pstar,
                "retention_at_pstar": rho_at_pstar,
                "retention_at_0.7": rho_at_grid,
                "hill_plot": hp,
            })

    # Aggregates
    finite_betas = np.array([r["beta_hat"] for r in per_location if not np.isnan(r["beta_hat"])])
    finite_pstars = np.array([r["pstar_pred"] for r in per_location if not np.isnan(r["pstar_pred"])])
    finite_rho_grid = np.array([r["retention_at_0.7"] for r in per_location if not np.isnan(r["retention_at_0.7"])])
    aggregate = {
        "beta_mean": float(finite_betas.mean()) if finite_betas.size else float("nan"),
        "beta_std": float(finite_betas.std(ddof=0)) if finite_betas.size else float("nan"),
        "beta_median": float(np.median(finite_betas)) if finite_betas.size else float("nan"),
        "pstar_pred_mean": float(finite_pstars.mean()) if finite_pstars.size else float("nan"),
        "pstar_grid": float(args.p_grid_optimum),
        "gap": (float(finite_pstars.mean()) - float(args.p_grid_optimum)) if finite_pstars.size else float("nan"),
        "retention_at_0.7_mean": float(finite_rho_grid.mean()) if finite_rho_grid.size else float("nan"),
    }

    # Q/K/V identity check: they alias the same loc_key, so beta is identical by construction.
    qkv_diffs: list[float] = []
    for i in range(n_layers):
        bs = []
        for nm in ("q_proj", "k_proj", "v_proj"):
            rows = [r for r in per_location if r["location"] == nm and r["layer"] == i]
            if rows and not np.isnan(rows[0]["beta_hat"]):
                bs.append(rows[0]["beta_hat"])
        if len(bs) >= 2:
            qkv_diffs.append(max(bs) - min(bs))
    qkv_max_abs_diff_beta = float(max(qkv_diffs)) if qkv_diffs else 0.0

    # Range check
    n_out_of_range = int(sum(
        1 for b in finite_betas if (b < 1.0 or b > 3.0)
    ))
    if n_out_of_range:
        print(f"[warn] {n_out_of_range} locations have beta_hat outside [1.0, 3.0]", flush=True)

    # p* sanity
    pmean = aggregate["pstar_pred_mean"]
    if not (np.isnan(pmean) or (0.5 < pmean < 0.9)):
        print(f"[warn] pstar_pred_mean={pmean:.4f} outside (0.5, 0.9)", flush=True)

    sanity_block = {
        "synthetic": sanity,
        "qkv_max_abs_diff_beta": qkv_max_abs_diff_beta,
        "n_out_of_range_beta": n_out_of_range,
    }

    results = {
        "config": {
            "n_batches": args.n_batches,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "k_frac": args.k_frac,
            "max_samples_per_loc": args.max_samples_per_loc,
            "n_batches_warmup": args.n_batches_warmup,
            "seed": args.seed,
            "p_grid_optimum": args.p_grid_optimum,
            "model_repo_id": REPO_ID if args.mode == "trained" else None,
            "base_model": args.base_model,
            "mode": args.mode,
            "hidden_size": real_hidden_size,
            "k_frac_list": k_frac_list,
            "p_sparsify_train": P_SPARSIFY if args.mode == "trained" else None,
            "n_layers": n_layers,
        },
        "sanity": sanity_block,
        "aggregate": aggregate,
        "per_location": per_location,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {out_path}", flush=True)

    _print_beta_table(per_location, n_layers)
    print()
    print(f"  beta_mean              = {aggregate['beta_mean']:.4f}")
    print(f"  beta_median            = {aggregate['beta_median']:.4f}")
    print(f"  pstar_pred_mean        = {aggregate['pstar_pred_mean']:.4f}  (grid={aggregate['pstar_grid']:.2f}, gap={aggregate['gap']:+.4f})")
    print(f"  retention_at_0.7_mean  = {aggregate['retention_at_0.7_mean']:.4f}")

    if args.auto_plot:
        try:
            _SCRIPTS = Path(__file__).resolve().parent
            if str(_SCRIPTS) not in sys.path:
                sys.path.insert(0, str(_SCRIPTS))
            import plot_hill_curves  # noqa: E402

            plot_out_dir = out_path.parent / out_path.stem
            print(f"\n[auto-plot] rendering figures into {plot_out_dir}/ ...", flush=True)
            plot_args = argparse.Namespace(
                in_path=str(out_path),
                out_dir=str(plot_out_dir),
                curve_layers=[0, 8, 15, 23, n_layers - 1],
                curve_locations=["q_proj", "o_proj", "gate_proj", "down_proj"],
            )
            # plot_hill_curves.main() reads its own argparse; call its building blocks directly.
            results_loaded = plot_hill_curves.load_results(Path(plot_args.in_path))
            grouped = plot_hill_curves.group_by_location(results_loaded["per_location"])
            out_d = Path(plot_args.out_dir)
            out_d.mkdir(parents=True, exist_ok=True)
            plot_hill_curves.plot_hill_curves(
                grouped, plot_args.curve_layers, plot_args.curve_locations,
                out_d / "fig1_hill_curves.pdf",
            )
            plot_hill_curves.plot_beta_heatmap(grouped, n_layers, out_d / "fig2_beta_heatmap.pdf")
            plot_hill_curves.plot_pstar_per_location(
                grouped,
                float(results_loaded.get("config", {}).get("p_grid_optimum", 0.7)),
                out_d / "fig3_pstar_per_loc.pdf",
            )
        except Exception as e:  # noqa: BLE001
            print(f"[auto-plot] skipped due to error: {e}", flush=True)


if __name__ == "__main__":
    main(parse_args())
