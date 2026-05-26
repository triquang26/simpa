# Repo-root path setup so the test can import without install
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from aslora.eval.hill import (
    hill_estimator, hill_plot_data, solve_pstar,
    predicted_retention, sanity_check_synthetic,
)

def test_hill_recovers_pareto_beta():
    """Hill estimator recovers β=1.5 within ±0.1 on 200k Pareto samples."""
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 1, 200_000)
    samples = u ** (-1.0 / 1.5)
    beta_hat = hill_estimator(samples, k_frac=0.05)
    assert 1.4 <= beta_hat <= 1.6, f"β̂={beta_hat} not in [1.4, 1.6]"

def test_solve_pstar_matches_numpy_roots():
    """Closed-form root matches numpy.roots for several β values."""
    for beta in [1.2, 1.34, 1.5, 2.0, 3.0]:
        theta = (beta - 1) / beta
        coeffs = [theta, 1 + theta, -1]
        roots = np.roots(coeffs)
        positive = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
        expected = min(p for p in positive if 0 < p < 1)
        got = solve_pstar(beta)
        assert abs(got - expected) < 1e-6, f"β={beta}: solve_pstar={got}, expected={expected}"

def test_solve_pstar_matches_simpa_anchor():
    """At β = 1.34 (Pareto fit on LLaMA-3 8B from SIMPA paper), p* ≈ 0.699."""
    p = solve_pstar(1.34)
    assert abs(p - 0.699) < 0.01, f"p*(1.34) = {p}, expected ≈ 0.699"

def test_retention_matches_pareto_simulation():
    """rho(p) = (1-p)^((β-1)/β) matches energy-retention from a Pareto sample."""
    rng = np.random.default_rng(1)
    beta = 1.5
    u = rng.uniform(0, 1, 1_000_000)
    samples = u ** (-1.0 / beta)
    # Energy retained when keeping top (1-p) fraction of samples
    p = 0.7
    sorted_desc = np.sort(samples)[::-1]
    k_keep = int(len(samples) * (1 - p))
    retained_energy = sorted_desc[:k_keep].sum() / sorted_desc.sum()
    predicted = predicted_retention(beta, p)
    # Empirical converges slowly; tolerance 5%
    assert abs(retained_energy - predicted) < 0.05, \
        f"retention: empirical={retained_energy}, predicted={predicted}"

def test_sanity_check_synthetic():
    """sanity_check_synthetic returns correct keys and small relative error."""
    result = sanity_check_synthetic(beta_true=1.5, n=100_000, seed=42)
    assert set(result.keys()) == {"beta_true", "beta_hat", "rel_err", "pstar_pred"}
    assert result["beta_true"] == 1.5
    assert result["rel_err"] < 0.05

def test_hill_plot_data_shape():
    """hill_plot_data returns equal-length arrays, monotone k grid."""
    rng = np.random.default_rng(2)
    samples = rng.uniform(0, 1, 50_000) ** (-1.0 / 2.0)
    out = hill_plot_data(samples, n_points=20)
    assert len(out["k"]) == len(out["beta_hat"])
    assert len(out["k"]) > 5
    assert list(out["k"]) == sorted(out["k"])

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  RUN  {t.__name__}")
        t()
        print(f"  PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
