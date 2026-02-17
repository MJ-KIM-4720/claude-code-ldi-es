"""
Monte Carlo 시뮬레이션 실행
============================
ES, VaR, Merton 전략 하에서 펀딩 비율(Y) 경로 시뮬레이션.

Figures:
  1. Fan chart (median + quantile bands) — 3 models side-by-side
  2. Terminal distribution histogram
  3. Shortfall probability over time
  4. Sample paths
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldi import params as P
from ldi import monte_carlo as MC

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

# ── Simulation settings ──────────────────────────────────
N_PATHS  = 10000
N_STEPS  = 250       # ~10 steps per year for T=10 (25/year)
SEED     = 42
MODELS   = ['es', 'var', 'merton']
LABELS   = {'es': 'ES', 'var': 'VaR', 'merton': 'Merton'}
COLORS   = {'es': 'C0', 'var': 'C1', 'merton': 'C2'}


def run_scenario(y0, tag=""):
    """Run MC for all models and generate figures for a given y0."""
    print(f"\n{'='*55}")
    print(f"  Scenario: y0 = {y0}  {tag}")
    print(f"{'='*55}")

    results = {}
    for model in MODELS:
        t0 = time.time()
        paths, t_grid = MC.simulate_paths(
            y0, n_paths=N_PATHS, n_steps=N_STEPS, model=model, seed=SEED
        )
        elapsed = time.time() - t0
        stats = MC.compute_path_stats(paths)
        tstats = MC.compute_terminal_stats(paths)
        sf_prob = MC.shortfall_prob_over_time(paths)
        results[model] = {
            'paths': paths, 't_grid': t_grid,
            'stats': stats, 'tstats': tstats, 'sf_prob': sf_prob,
        }
        print(f"  {LABELS[model]:>6}: "
              f"E[Y_T]={tstats['mean']:.3f}  "
              f"P(Y_T<k)={tstats['shortfall_prob']:.3f}  "
              f"ES={tstats['expected_shortfall']:.4f}  "
              f"({elapsed:.1f}s)")

    suffix = f"_y0{y0:.1f}".replace(".", "")

    # ── Figure 1: Fan charts ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        s = results[model]['stats']
        t = results[model]['t_grid']
        ax.fill_between(t, s['q05'], s['q95'], alpha=0.15, color=COLORS[model])
        ax.fill_between(t, s['q25'], s['q75'], alpha=0.3, color=COLORS[model])
        ax.plot(t, s['median'], color=COLORS[model], lw=2, label='Median')
        ax.plot(t, s['mean'], color=COLORS[model], lw=1, ls='--', label='Mean')
        ax.axhline(P.k, color='gray', ls=':', lw=0.8)
        ax.set_title(f"{LABELS[model]}  (y₀={y0})")
        ax.set_xlabel('t (years)')
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(3.0, s['q95'].max() * 1.2))
    axes[0].set_ylabel('Funding Ratio Y')
    fig.suptitle(f'Fan Chart: Funding Ratio Paths (y₀={y0})', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"mc_fan{suffix}.png"), dpi=150)
    plt.close(fig)

    # ── Figure 2: Terminal distribution ───────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODELS:
        Y_T = results[model]['paths'][:, -1]
        # Clip for histogram visibility
        Y_T_clip = np.clip(Y_T, 0, np.quantile(Y_T, 0.99))
        ax.hist(Y_T_clip, bins=80, alpha=0.4, color=COLORS[model],
                label=f"{LABELS[model]} (μ={results[model]['tstats']['mean']:.2f})",
                density=True)
    ax.axvline(P.k, color='gray', ls=':', lw=1.5, label=f'k={P.k}')
    ax.set_xlabel('Terminal Funding Ratio Y_T')
    ax.set_ylabel('Density')
    ax.set_title(f'Terminal Distribution (y₀={y0}, T={P.T:.0f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"mc_terminal{suffix}.png"), dpi=150)
    plt.close(fig)

    # ── Figure 3: Shortfall probability over time ─────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODELS:
        t = results[model]['t_grid']
        sf = results[model]['sf_prob']
        ax.plot(t, sf, color=COLORS[model], lw=2, label=LABELS[model])
    ax.axhline(P.alpha, color='gray', ls=':', lw=0.8, label=f'α={P.alpha}')
    ax.set_xlabel('t (years)')
    ax.set_ylabel('P(Y_t < k)')
    ax.set_title(f'Shortfall Probability Over Time (y₀={y0})')
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"mc_shortfall{suffix}.png"), dpi=150)
    plt.close(fig)

    # ── Figure 4: Sample paths ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    n_sample = 20
    for ax, model in zip(axes, MODELS):
        t = results[model]['t_grid']
        for j in range(n_sample):
            ax.plot(t, results[model]['paths'][j, :],
                    alpha=0.4, lw=0.6, color=COLORS[model])
        ax.axhline(P.k, color='gray', ls=':', lw=0.8)
        ax.set_title(f"{LABELS[model]}  (y₀={y0})")
        ax.set_xlabel('t (years)')
        ax.set_ylim(0, max(3.0, results[model]['paths'][:n_sample].max() * 1.1))
    axes[0].set_ylabel('Funding Ratio Y')
    fig.suptitle(f'Sample Paths (y₀={y0}, {n_sample} paths)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"mc_samples{suffix}.png"), dpi=150)
    plt.close(fig)

    return results


def print_summary_table(all_results):
    """Print comparison table across scenarios."""
    print(f"\n{'='*70}")
    print("  Summary Table")
    print(f"{'='*70}")
    header = f"  {'y0':>4} | {'Model':>6} | {'E[Y_T]':>7} | {'Std':>6} | {'P(short)':>8} | {'E[short]':>8} | {'Q05':>6} | {'Q95':>6}"
    print(header)
    print("  " + "-" * 66)
    for y0, results in all_results.items():
        for model in MODELS:
            ts = results[model]['tstats']
            print(f"  {y0:>4.1f} | {LABELS[model]:>6} | "
                  f"{ts['mean']:>7.3f} | {ts['std']:>6.3f} | "
                  f"{ts['shortfall_prob']:>8.3f} | {ts['expected_shortfall']:>8.4f} | "
                  f"{ts['q05']:>6.3f} | {ts['q95']:>6.3f}")
        print("  " + "-" * 66)


def main():
    P.print_params()

    scenarios = {
        0.8: "(Underfunded — VaR gambling incentive)",
        1.0: "(Fully funded — baseline)",
        1.2: "(Overfunded — constraints non-binding)",
    }

    all_results = {}
    for y0, tag in scenarios.items():
        all_results[y0] = run_scenario(y0, tag)

    print_summary_table(all_results)

    print(f"\nAll MC figures saved to: {os.path.abspath(OUT)}/")


if __name__ == "__main__":
    main()
