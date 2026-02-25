"""
Monte Carlo 시뮬레이션 실행
============================
ES, VaR, Merton 전략 하에서 펀딩 비율(Y) 경로 시뮬레이션.

Figures:
  1. Fan chart (median + quantile bands) — 3 models side-by-side
  2. Terminal distribution histogram
  3. Shortfall probability over time
  4. Sample paths
  5. Welfare analysis terminal distribution (F0=1.0 only)
"""

import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldi import params as P
from ldi import monte_carlo as MC
from ldi.style import (apply_style, COLORS, FIGSIZES, DPI, FAN_ALPHA,
                        HIST_ALPHA, MERTON_LINE, LEGEND, setup_grid, savefig)

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(OUT, exist_ok=True)

# ── Simulation settings ──────────────────────────────────
N_PATHS  = 10000
N_STEPS  = 250       # ~10 steps per year for T=10 (25/year)
SEED     = 42
MODELS   = ['es', 'var', 'merton']
LABELS   = {'es': 'ES', 'var': 'VaR', 'merton': 'Merton'}
MC_COLORS = {'es': COLORS['ES'], 'var': COLORS['VaR'], 'merton': COLORS['Merton']}


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
        ce = MC.certainty_equivalent(paths)
        tstats['CE'] = ce
        results[model] = {
            'paths': paths, 't_grid': t_grid,
            'stats': stats, 'tstats': tstats, 'sf_prob': sf_prob,
        }
        print(f"  {LABELS[model]:>6}: "
              f"E[Y_T]={tstats['mean']:.3f}  "
              f"P(Y_T<k)={tstats['shortfall_prob']:.3f}  "
              f"ES={tstats['expected_shortfall']:.4f}  "
              f"CE={ce:.4f}  "
              f"({elapsed:.1f}s)")

    suffix = f"_y0{y0:.1f}".replace(".", "")

    # ── Figure 1: Fan charts ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZES['triple'], sharey=True)
    for ax, model in zip(axes, MODELS):
        s = results[model]['stats']
        t = results[model]['t_grid']
        c = MC_COLORS[model]
        ax.fill_between(t, s['q05'], s['q95'], alpha=FAN_ALPHA['outer'], color=c)
        ax.fill_between(t, s['q25'], s['q75'], alpha=FAN_ALPHA['middle'], color=c)
        ax.plot(t, s['median'], color=c, lw=2, label='Median')
        ax.plot(t, s['mean'], color=c, lw=1, ls='--', label='Mean')
        ax.axhline(P.k, **MERTON_LINE, label=f'k={P.k}')
        ax.set_title(f"{LABELS[model]}  ($F_0$={y0})")
        ax.set_xlabel('$t$ (years)')
        ax.legend(**LEGEND)
        ax.set_ylim(0, max(3.0, s['q95'].max() * 1.2))
        setup_grid(ax)
    axes[0].set_ylabel('Funding Ratio $F(t)$')
    fig.suptitle(f'Fan Chart: Funding Ratio Paths ($F_0$={y0})')
    fig.tight_layout()
    path = os.path.join(OUT, f"mc_fan{suffix}.png")
    savefig(fig, path)
    print(f"  Saved mc_fan{suffix}.png")

    # ── Figure 2: Terminal distribution with KDE overlay ──
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    kde_linestyles = {'es': '-', 'var': '--', 'merton': ':'}
    for model in MODELS:
        Y_T = results[model]['paths'][:, -1]
        ts = results[model]['tstats']
        # Clip outliers to [0, 1.6] for visibility
        Y_T_clip = np.clip(Y_T, 0, 1.6)
        # Histogram with low alpha (KDE is the primary visual)
        ax.hist(Y_T_clip, bins=80, alpha=0.3, color=MC_COLORS[model],
                density=True)
        # KDE overlay
        kde = gaussian_kde(Y_T_clip)
        x_kde = np.linspace(0.3, 1.6, 500)
        ax.plot(x_kde, kde(x_kde), color=MC_COLORS[model], lw=2,
                linestyle=kde_linestyles[model],
                label=(f"{LABELS[model]} "
                       f"(med={ts['median']:.2f}, "
                       f"shortfall={ts['expected_shortfall']:.3f})"))
    ax.axvline(P.k, **MERTON_LINE, label=f'k={P.k}')
    ax.set_xlabel('Terminal Funding Ratio $F_T$')
    ax.set_ylabel('Density')
    ax.set_xlim(0.3, 1.6)
    ax.set_title(f'Terminal Funding Ratio Distribution '
                 f'($F_0 = {y0}$, $N = 10\\,000$, $T = {P.T:.0f}$)')
    ax.legend(**LEGEND)
    setup_grid(ax)
    fig.tight_layout()
    path = os.path.join(OUT, f"mc_terminal{suffix}.png")
    savefig(fig, path)
    print(f"  Saved mc_terminal{suffix}.png")

    # ── Figure 3: Shortfall probability over time ─────────
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for model in MODELS:
        t = results[model]['t_grid']
        sf = results[model]['sf_prob']
        ax.plot(t, sf, color=MC_COLORS[model], lw=2, label=LABELS[model])
    ax.axhline(P.alpha, **MERTON_LINE, label=f'$\\alpha$={P.alpha}')
    ax.set_xlabel('$t$ (years)')
    ax.set_ylabel('$P(F_t < k)$')
    ax.set_title(f'Shortfall Probability Over Time ($F_0$={y0})')
    ax.legend(**LEGEND)
    ax.set_ylim(-0.02, 1.02)
    setup_grid(ax)
    fig.tight_layout()
    path = os.path.join(OUT, f"mc_shortfall{suffix}.png")
    savefig(fig, path)
    print(f"  Saved mc_shortfall{suffix}.png")

    # ── Figure 4: Sample paths ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZES['triple'], sharey=True)
    n_sample = 20
    for ax, model in zip(axes, MODELS):
        t = results[model]['t_grid']
        c = MC_COLORS[model]
        for j in range(n_sample):
            ax.plot(t, results[model]['paths'][j, :],
                    alpha=0.4, lw=0.6, color=c)
        ax.axhline(P.k, **MERTON_LINE)
        ax.set_title(f"{LABELS[model]}  ($F_0$={y0})")
        ax.set_xlabel('$t$ (years)')
        ax.set_ylim(0, max(3.0, results[model]['paths'][:n_sample].max() * 1.1))
        setup_grid(ax)
    axes[0].set_ylabel('Funding Ratio $F(t)$')
    fig.suptitle(f'Sample Paths ($F_0$={y0}, {n_sample} paths)')
    fig.tight_layout()
    path = os.path.join(OUT, f"mc_samples{suffix}.png")
    savefig(fig, path)
    print(f"  Saved mc_samples{suffix}.png")

    return results


def print_summary_table(all_results):
    """Print comparison table across scenarios (with CE column)."""
    print(f"\n{'='*80}")
    print("  Summary Table")
    print(f"{'='*80}")
    header = (f"  {'y0':>4} | {'Model':>6} | {'E[F_T]':>7} | {'Std':>6} | "
              f"{'P(short)':>8} | {'E[short]':>8} | {'Median':>6} | {'CE':>7}")
    print(header)
    print("  " + "-" * 76)
    for y0, results in all_results.items():
        for model in MODELS:
            ts = results[model]['tstats']
            print(f"  {y0:>4.1f} | {LABELS[model]:>6} | "
                  f"{ts['mean']:>7.3f} | {ts['std']:>6.3f} | "
                  f"{ts['shortfall_prob']:>8.3f} | {ts['expected_shortfall']:>8.4f} | "
                  f"{ts['median']:>6.3f} | {ts['CE']:>7.4f}")
        print("  " + "-" * 76)


def print_welfare_cost(results):
    """Print welfare cost analysis for a single scenario.

    Welfare cost = (CE_Merton - CE_model) / CE_Merton * 100 (%).
    """
    ce_merton = results['merton']['tstats']['CE']
    print(f"\n{'='*55}")
    print("  Welfare Analysis (Certainty Equivalent)")
    print(f"{'='*55}")
    print(f"  {'Model':>6} | {'CE':>8} | {'CE Loss (%)':>12}")
    print("  " + "-" * 35)
    for model in MODELS:
        ce = results[model]['tstats']['CE']
        if ce_merton > 0:
            loss = (ce_merton - ce) / ce_merton * 100.0
        else:
            loss = 0.0
        print(f"  {LABELS[model]:>6} | {ce:>8.4f} | {loss:>11.2f}%")
    print("  " + "-" * 35)


def plot_welfare_terminal(results):
    """Generate terminal distribution figure with CE values in legend.

    F0=1.0 only. Saves to fig_mc_terminal_F10_welfare.png.
    """
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for model in MODELS:
        Y_T = results[model]['paths'][:, -1]
        ce = results[model]['tstats']['CE']
        Y_T_clip = np.clip(Y_T, 0, np.quantile(Y_T, 0.99))
        ax.hist(Y_T_clip, bins=80, alpha=HIST_ALPHA, color=MC_COLORS[model],
                label=f"{LABELS[model]} (CE={ce:.3f})",
                density=True)
    ax.axvline(P.k, color='green', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'$k$={P.k}')
    ax.set_xlabel('Terminal Funding Ratio $F_T$')
    ax.set_ylabel('Density')
    ax.set_title(f'Terminal Distribution with Welfare ($F_0$=1.0, $T$={P.T:.0f})')
    ax.legend(**LEGEND)
    setup_grid(ax)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_mc_terminal_F10_welfare.png")
    savefig(fig, path)
    print(f"  Saved fig_mc_terminal_F10_welfare.png")


def main():
    apply_style()
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

    # Welfare analysis for F0=1.0
    print_welfare_cost(all_results[1.0])
    plot_welfare_terminal(all_results[1.0])

    print(f"\nAll MC figures saved to: {os.path.abspath(OUT)}/")


if __name__ == "__main__":
    main()
