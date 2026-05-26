"""Plot Hill curves + beta_hat heatmap from hill_results.json.

Reads outputs/hill_pstar/hill_results.json (produced by measure_hill_pstar.py)
and renders three figures:

  1. fig1_hill_curves.pdf   -- Hill plot (beta_hat vs k) for representative
     (layer, location) pairs. A flat plateau supports Pareto-tail assumption;
     continuous drift contradicts it.
  2. fig2_beta_heatmap.pdf  -- 7x32 grid of beta_hat values per logical
     location and transformer layer.
  3. fig3_pstar_per_loc.pdf -- predicted p* vs grid-search optimum (0.7),
     plotted per logical location across all 32 layers.

Run:
    uv run python scripts/plot_hill_curves.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOCATION_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="outputs/hill_pstar/hill_results.json")
    ap.add_argument("--out-dir", default="outputs/hill_pstar")
    ap.add_argument(
        "--curve-layers",
        type=int,
        nargs="+",
        default=[0, 8, 15, 23, 31],
        help="Layers to overlay in the Hill-curve figure",
    )
    ap.add_argument(
        "--curve-locations",
        type=str,
        nargs="+",
        default=["q_proj", "o_proj", "gate_proj", "down_proj"],
        help="Logical locations to render Hill curves for (one subplot each)",
    )
    return ap.parse_args()


def load_results(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def group_by_location(per_loc: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in per_loc:
        out[row["location"]].append(row)
    for loc in out:
        out[loc].sort(key=lambda r: r["layer"])
    return out


def plot_hill_curves(grouped: dict[str, list[dict]], layers: list[int],
                     locations: list[str], out_path: Path):
    n_loc = len(locations)
    fig, axes = plt.subplots(1, n_loc, figsize=(4.2 * n_loc, 3.6), sharey=True)
    if n_loc == 1:
        axes = [axes]

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(layers) - 1)) for i in range(len(layers))]

    for ax, loc in zip(axes, locations):
        rows = grouped.get(loc, [])
        if not rows:
            ax.set_title(f"{loc}\n(no data)")
            continue
        rows_by_layer = {r["layer"]: r for r in rows}
        for color, layer in zip(colors, layers):
            r = rows_by_layer.get(layer)
            if r is None:
                continue
            hp = r.get("hill_plot", {})
            ks, betas = hp.get("k", []), hp.get("beta_hat", [])
            if not ks:
                continue
            ax.plot(ks, betas, color=color, lw=1.5, label=f"L{layer:02d}")
            beta_final = r["beta_hat"]
            if np.isfinite(beta_final):
                ax.axhline(beta_final, color=color, lw=0.6, ls=":", alpha=0.4)

        ax.set_xscale("log")
        ax.set_xlabel("k (upper order statistics)")
        ax.set_title(loc)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=1)

    axes[0].set_ylabel(r"$\hat{\beta}$ (Hill estimate)")
    fig.suptitle(r"Hill plots: $\hat{\beta}(k)$. Flat plateau = Pareto tail; drift = non-Pareto.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] wrote {out_path}")


def plot_beta_heatmap(grouped: dict[str, list[dict]], n_layers: int, out_path: Path):
    matrix = np.full((len(LOCATION_ORDER), n_layers), np.nan)
    for i, loc in enumerate(LOCATION_ORDER):
        for r in grouped.get(loc, []):
            if 0 <= r["layer"] < n_layers:
                matrix[i, r["layer"]] = r["beta_hat"]

    fig, ax = plt.subplots(figsize=(11, 3.6))
    vmin = np.nanpercentile(matrix, 2)
    vmax = np.nanpercentile(matrix, 98)
    im = ax.imshow(matrix, aspect="auto", cmap="magma_r", vmin=vmin, vmax=vmax)

    ax.set_yticks(range(len(LOCATION_ORDER)))
    ax.set_yticklabels(LOCATION_ORDER)
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_xticklabels([f"L{i:02d}" for i in range(0, n_layers, 2)], fontsize=8)
    ax.set_xlabel("Transformer layer")
    ax.set_title(r"$\hat{\beta}$ by location $\times$ layer (Hill, k=0.05$\cdot$n)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                color = "white" if (v - vmin) / max(1e-9, (vmax - vmin)) > 0.55 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.5, color=color)

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label=r"$\hat{\beta}$")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] wrote {out_path}")


def plot_pstar_per_location(grouped: dict[str, list[dict]], p_grid: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cmap = plt.get_cmap("tab10")

    for i, loc in enumerate(LOCATION_ORDER):
        rows = grouped.get(loc, [])
        if not rows:
            continue
        xs = [r["layer"] for r in rows]
        ys = [r["pstar_pred"] for r in rows]
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.2,
                color=cmap(i), label=loc, alpha=0.85)

    ax.axhline(p_grid, color="red", ls="--", lw=1, label=f"grid p*={p_grid}")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel(r"Predicted $p^*$")
    ax.set_title(r"Closed-form $p^*$ prediction per location vs grid-search optimum")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] wrote {out_path}")


def main():
    args = parse_args()
    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found. Run scripts/measure_hill_pstar.py first.")

    results = load_results(in_path)
    per_loc = results["per_location"]
    grouped = group_by_location(per_loc)

    n_layers = 1 + max((r["layer"] for r in per_loc), default=0)
    p_grid = float(results.get("config", {}).get("p_grid_optimum", 0.7))

    plot_hill_curves(grouped, args.curve_layers, args.curve_locations,
                     out_dir / "fig1_hill_curves.pdf")
    plot_beta_heatmap(grouped, n_layers, out_dir / "fig2_beta_heatmap.pdf")
    plot_pstar_per_location(grouped, p_grid, out_dir / "fig3_pstar_per_loc.pdf")

    agg = results.get("aggregate", {})
    print()
    print("=" * 60)
    print(f"  beta_mean              = {agg.get('beta_mean', float('nan')):.4f}")
    print(f"  beta_median            = {agg.get('beta_median', float('nan')):.4f}")
    print(f"  pstar_pred_mean        = {agg.get('pstar_pred_mean', float('nan')):.4f}"
          f"  (grid={p_grid}, gap={agg.get('gap', float('nan')):+.4f})")
    print(f"  retention_at_0.7_mean  = {agg.get('retention_at_0.7_mean', float('nan')):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
