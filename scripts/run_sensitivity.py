"""
Sensitivity Analysis
====================
Generates figures for Groups A-E and a summary CSV table.
Outputs saved to results/figures/ at DPI 300.

Groups:
  A: Risk aversion (gamma)
  B: ES budget (epsilon)
  C: Expected inflation (mu_I)
  D: Investment horizon (T)
  E: Stock-IIB correlation (rho)
"""

import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldi import params as P, es_model as ES, var_model as VaR
from ldi.params import override_params
from ldi.style import (apply_style, COLORS, LINE_STYLES, FIGSIZES, DPI,
                        LEGEND, MERTON_LINE, GAMBLING_REGION,
                        setup_grid, add_merton_hline, savefig)

# ── Constants ──────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
F_RANGE = np.linspace(0.5, 1.3, 500)


# ═══════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════

def safe_es_A(y0, eps=None):
    """Compute ES adjustment factor, returning NaN on failure."""
    try:
        return ES.cross_sectional_A(y0, eps)
    except Exception:
        return np.nan


def safe_var_A(y0, alpha=None):
    """Compute VaR adjustment factor, returning NaN on failure."""
    try:
        return VaR.cross_sectional_A(y0, alpha)
    except Exception:
        return np.nan


def safe_es_portfolio(y0, eps=None):
    """Return (pi_S, pi_I, total, A) for ES model."""
    try:
        A = ES.cross_sectional_A(y0, eps)
        pi_S = A * P.Pi_star[0]
        pi_I = A * P.Pi_star[1]
        return pi_S, pi_I, pi_S + pi_I, A
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def safe_var_portfolio(y0, alpha=None):
    """Return (pi_S, pi_I, total, A) for VaR model."""
    try:
        A = VaR.cross_sectional_A(y0, alpha)
        pi_S = A * P.Pi_star[0]
        pi_I = A * P.Pi_star[1]
        return pi_S, pi_I, pi_S + pi_I, A
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def compute_es_totals(F_range, eps=None):
    """Compute total risky allocation (π_S + π_I) for ES across funding ratios."""
    totals = np.full(len(F_range), np.nan)
    for i, f in enumerate(F_range):
        A = safe_es_A(f, eps)
        totals[i] = A * P.Pi_star.sum()
    return totals


def compute_var_totals(F_range, alpha=None):
    """Compute total risky allocation (π_S + π_I) for VaR across funding ratios."""
    totals = np.full(len(F_range), np.nan)
    for i, f in enumerate(F_range):
        A = safe_var_A(f, alpha)
        totals[i] = A * P.Pi_star.sum()
    return totals


def compute_es_components(F_range, eps=None):
    """Compute (pi_S_array, pi_I_array) for ES across funding ratios."""
    pi_S = np.full(len(F_range), np.nan)
    pi_I = np.full(len(F_range), np.nan)
    for i, f in enumerate(F_range):
        A = safe_es_A(f, eps)
        pi_S[i] = A * P.Pi_star[0]
        pi_I[i] = A * P.Pi_star[1]
    return pi_S, pi_I


def compute_es_A_array(F_range, eps=None):
    """Compute adjustment factor A array for ES."""
    return np.array([safe_es_A(f, eps) for f in F_range])


def compute_var_A_array(F_range, alpha=None):
    """Compute adjustment factor A array for VaR."""
    return np.array([safe_var_A(f, alpha) for f in F_range])


# ═══════════════════════════════════════════════════════════
# Group A: Risk Aversion (gamma)
# ═══════════════════════════════════════════════════════════

def plot_group_A():
    gammas = [2, 3, 5, 7]
    colors = COLORS['param_values'][:4]

    # --- Figure A1: ES-only total risky allocation ---
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for gamma, color in zip(gammas, colors):
        with override_params(GAMMA=float(gamma)):
            totals = compute_es_totals(F_RANGE)
            ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                    label=fr'$\gamma = {gamma}$')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.set_title(r'ES Constraint: Effect of Risk Aversion ($\gamma$)')
    ax.legend(**LEGEND)
    setup_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_A1_gamma_es.png')
    savefig(fig, path)
    print("  Saved fig_A1_gamma_es.png")

    # --- Figure A2: ES vs VaR total allocation (2x2 subplots) ---
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZES['quad'])
    for ax, gamma in zip(axes.flat, gammas):
        with override_params(GAMMA=float(gamma)):
            es_tot = compute_es_totals(F_RANGE)
            var_tot = compute_var_totals(F_RANGE)
            merton = P.Pi_star.sum()
            ax.plot(F_RANGE, es_tot, label='ES', **LINE_STYLES['ES'])
            ax.plot(F_RANGE, var_tot, label='VaR', **LINE_STYLES['VaR'])
            add_merton_hline(ax, merton, f'Merton ({merton:.2f})')
            ax.set_title(fr'$\gamma = {gamma}$')
            ax.set_xlabel('Funding Ratio $F(t)$')
            ax.set_ylabel(r'Total Risky Allocation')
            ax.legend(**LEGEND)
            setup_grid(ax)
            ax.set_xlim(0.5, 1.3)
    fig.suptitle(r'ES vs VaR: Total Risky Allocation by Risk Aversion ($\gamma$)')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_A2_gamma_compare.png')
    savefig(fig, path)
    print("  Saved fig_A2_gamma_compare.png")

    # --- Figure A3: Adjustment factor A(F) (2x2 subplots) ---
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZES['quad'])
    for ax, gamma in zip(axes.flat, gammas):
        with override_params(GAMMA=float(gamma)):
            es_A = compute_es_A_array(F_RANGE)
            var_A = compute_var_A_array(F_RANGE)
            ax.plot(F_RANGE, es_A, label='ES', **LINE_STYLES['ES'])
            ax.plot(F_RANGE, var_A, label='VaR', **LINE_STYLES['VaR'])
            add_merton_hline(ax, 1.0, 'A = 1 (Merton)')
            # Highlight A > 1 region for VaR
            gambling_mask = var_A > 1.0
            if np.any(gambling_mask):
                ax.fill_between(F_RANGE, 1.0, var_A, where=gambling_mask,
                                **GAMBLING_REGION, label='VaR gambling')
            ax.set_title(fr'$\gamma = {gamma}$')
            ax.set_xlabel('Funding Ratio $F(t)$')
            ax.set_ylabel('Adjustment Factor $A(F)$')
            ax.legend(**LEGEND)
            setup_grid(ax)
            ax.set_xlim(0.5, 1.3)
    fig.suptitle(r'Adjustment Factor $A(F)$: ES vs VaR by Risk Aversion ($\gamma$)')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_A3_gamma_A_factor.png')
    savefig(fig, path)
    print("  Saved fig_A3_gamma_A_factor.png")


# ═══════════════════════════════════════════════════════════
# Group B: ES Budget (epsilon)
# ═══════════════════════════════════════════════════════════

def plot_group_B():
    epsilons = [0.02, 0.03, 0.05, 0.08, 0.10]
    colors = COLORS['param_values'][:5]

    # --- Figure B1: ES-only total allocation by epsilon ---
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for eps, color in zip(epsilons, colors):
        totals = compute_es_totals(F_RANGE, eps=eps)
        ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                label=fr'$\varepsilon = {eps}$')
    merton = P.Pi_star.sum()
    add_merton_hline(ax, merton, f'Merton ({merton:.2f})')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.set_title(r'ES Constraint: Effect of ES Budget ($\varepsilon$)')
    ax.legend(**LEGEND)
    setup_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_B1_epsilon_es.png')
    savefig(fig, path)
    print("  Saved fig_B1_epsilon_es.png")

    # --- Figure B2: ES vs VaR comparison ---
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    # VaR reference (single dashed line at baseline alpha)
    var_tot = compute_var_totals(F_RANGE)
    ax.plot(F_RANGE, var_tot, label=fr'VaR ($\alpha = {P.alpha}$)',
            **LINE_STYLES['VaR'])
    # ES lines by epsilon
    for eps, color in zip(epsilons, colors):
        totals = compute_es_totals(F_RANGE, eps=eps)
        ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                label=fr'ES ($\varepsilon = {eps}$)')
    merton = P.Pi_star.sum()
    add_merton_hline(ax, merton, f'Merton ({merton:.2f})')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.set_title(r'ES vs VaR: Total Risky Allocation for Various ES Budgets')
    ax.legend(**LEGEND)
    setup_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_B2_epsilon_compare.png')
    savefig(fig, path)
    print("  Saved fig_B2_epsilon_compare.png")


# ═══════════════════════════════════════════════════════════
# Group C: Expected Inflation (mu_I)
# ═══════════════════════════════════════════════════════════

def plot_group_C():
    mu_Is = [0.01, 0.023, 0.035, 0.05]
    colors = COLORS['param_values'][:4]

    # --- Figure C1: ES total allocation by mu_I ---
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for mu_I, color in zip(mu_Is, colors):
        with override_params(MU_I=mu_I):
            totals = compute_es_totals(F_RANGE)
            ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                    label=fr'$\mu_I = {mu_I}$')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.set_title(r'ES Constraint: Effect of Expected Inflation ($\mu_I$)')
    ax.legend(**LEGEND)
    setup_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_C1_muI_es.png')
    savefig(fig, path)
    print("  Saved fig_C1_muI_es.png")

    # --- Figure C2: ES vs VaR total allocation (2x2 subplots) ---
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZES['quad'])
    for ax, mu_I in zip(axes.flat, mu_Is):
        with override_params(MU_I=mu_I):
            es_tot = compute_es_totals(F_RANGE)
            var_tot = compute_var_totals(F_RANGE)
            merton = P.Pi_star.sum()
            ax.plot(F_RANGE, es_tot, label='ES', **LINE_STYLES['ES'])
            ax.plot(F_RANGE, var_tot, label='VaR', **LINE_STYLES['VaR'])
            add_merton_hline(ax, merton, f'Merton ({merton:.2f})')
            ax.set_title(fr'$\mu_I = {mu_I}$')
            ax.set_xlabel('Funding Ratio $F(t)$')
            ax.set_ylabel('Total Risky Allocation')
            ax.legend(**LEGEND)
            setup_grid(ax)
            ax.set_xlim(0.5, 1.3)
    fig.suptitle(r'ES vs VaR: Total Risky Allocation by Expected Inflation ($\mu_I$)')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_C2_muI_compare.png')
    savefig(fig, path)
    print("  Saved fig_C2_muI_compare.png")

    # --- Figure C2 appendix: Stock vs IIB allocation (1x2 subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZES['triple'])
    for mu_I, color in zip(mu_Is, colors):
        with override_params(MU_I=mu_I):
            pi_S, pi_I = compute_es_components(F_RANGE)
            ax1.plot(F_RANGE, pi_S, '-', color=color, lw=2,
                     label=fr'$\mu_I = {mu_I}$')
            ax2.plot(F_RANGE, pi_I, '-', color=color, lw=2,
                     label=fr'$\mu_I = {mu_I}$')

    ax1.set_xlabel('Funding Ratio $F(t)$')
    ax1.set_ylabel(r'Stock Allocation ($\pi_S$)')
    ax1.set_title(r'Stock Allocation ($\pi_S$)')
    ax1.legend(**LEGEND)
    setup_grid(ax1)
    ax1.set_xlim(0.5, 1.3)

    ax2.set_xlabel('Funding Ratio $F(t)$')
    ax2.set_ylabel(r'IIB Allocation ($\pi_I$)')
    ax2.set_title(r'IIB Allocation ($\pi_I$)')
    ax2.legend(**LEGEND)
    setup_grid(ax2)
    ax2.set_xlim(0.5, 1.3)

    fig.suptitle(r'ES Constraint: Stock vs IIB by Expected Inflation ($\mu_I$)')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_C2_muI_components_appendix.png')
    savefig(fig, path)
    print("  Saved fig_C2_muI_components_appendix.png")


# ═══════════════════════════════════════════════════════════
# Group D: Investment Horizon (T)
# ═══════════════════════════════════════════════════════════

def plot_group_D():
    Ts = [5, 10, 15, 20]
    colors = COLORS['param_values'][:4]

    # --- Figure D1: ES total allocation by T ---
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for T_val, color in zip(Ts, colors):
        with override_params(T=float(T_val)):
            totals = compute_es_totals(F_RANGE)
            ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                    label=f'$T = {T_val}$')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.set_title('ES Constraint: Effect of Investment Horizon ($T$)')
    ax.legend(**LEGEND)
    setup_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_D1_T_es.png')
    savefig(fig, path)
    print("  Saved fig_D1_T_es.png")

    # --- Figure D2: ES vs VaR (2x2 subplots) ---
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZES['quad'])
    for ax, T_val in zip(axes.flat, Ts):
        with override_params(T=float(T_val)):
            es_tot = compute_es_totals(F_RANGE)
            var_tot = compute_var_totals(F_RANGE)
            merton = P.Pi_star.sum()
            ax.plot(F_RANGE, es_tot, label='ES', **LINE_STYLES['ES'])
            ax.plot(F_RANGE, var_tot, label='VaR', **LINE_STYLES['VaR'])
            add_merton_hline(ax, merton, f'Merton ({merton:.2f})')
            ax.set_title(f'$T = {T_val}$')
            ax.set_xlabel('Funding Ratio $F(t)$')
            ax.set_ylabel('Total Risky Allocation')
            ax.legend(**LEGEND)
            setup_grid(ax)
            ax.set_xlim(0.5, 1.3)
    fig.suptitle('ES vs VaR: Total Risky Allocation by Investment Horizon ($T$)')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_D2_T_compare.png')
    savefig(fig, path)
    print("  Saved fig_D2_T_compare.png")


# ═══════════════════════════════════════════════════════════
# Group E: Stock-IIB Correlation (rho)
# ═══════════════════════════════════════════════════════════

def plot_group_E():
    rhos = [-0.30, -0.15, -0.05]
    colors = COLORS['param_values'][:3]

    # --- Figure E1: ES total allocation by rho ---
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for rho, color in zip(rhos, colors):
        with override_params(RHO=rho):
            totals = compute_es_totals(F_RANGE)
            ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                    label=fr'$\rho = {rho}$')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.set_title(r'ES Constraint: Effect of Stock-IIB Correlation ($\rho$)')
    ax.legend(**LEGEND)
    setup_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_E1_rho_es.png')
    savefig(fig, path)
    print("  Saved fig_E1_rho_es.png")

    # --- Figure E2: Stock vs IIB allocation (1x2 subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZES['triple'])
    for rho, color in zip(rhos, colors):
        with override_params(RHO=rho):
            pi_S, pi_I = compute_es_components(F_RANGE)
            ax1.plot(F_RANGE, pi_S, '-', color=color, lw=2,
                     label=fr'$\rho = {rho}$')
            ax2.plot(F_RANGE, pi_I, '-', color=color, lw=2,
                     label=fr'$\rho = {rho}$')

    ax1.set_xlabel('Funding Ratio $F(t)$')
    ax1.set_ylabel(r'Stock Allocation ($\pi_S$)')
    ax1.set_title(r'Stock Allocation ($\pi_S$)')
    ax1.legend(**LEGEND)
    setup_grid(ax1)
    ax1.set_xlim(0.5, 1.3)

    ax2.set_xlabel('Funding Ratio $F(t)$')
    ax2.set_ylabel(r'IIB Allocation ($\pi_I$)')
    ax2.set_title(r'IIB Allocation ($\pi_I$)')
    ax2.legend(**LEGEND)
    setup_grid(ax2)
    ax2.set_xlim(0.5, 1.3)

    fig.suptitle(r'ES Constraint: Stock vs IIB by Correlation ($\rho$)')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_E2_rho_components.png')
    savefig(fig, path)
    print("  Saved fig_E2_rho_components.png")


# ═══════════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════════

def generate_summary_table():
    """Generate CSV with allocations for all parameter variations at key F values."""
    F_sample = [0.7, 0.8, 0.9, 1.0]

    header = ['param_name', 'param_value', 'F',
              'pi_S_es', 'pi_I_es', 'total_es',
              'pi_S_var', 'pi_I_var', 'total_var',
              'A_es', 'A_var']

    rows = []

    # Group A: gamma
    for gamma in [2, 3, 5, 7]:
        with override_params(GAMMA=float(gamma)):
            for f in F_sample:
                es = safe_es_portfolio(f)
                va = safe_var_portfolio(f)
                rows.append(['gamma', gamma, f,
                             es[0], es[1], es[2],
                             va[0], va[1], va[2],
                             es[3], va[3]])

    # Group B: epsilon (pass directly, no override needed)
    for eps in [0.02, 0.03, 0.05, 0.08, 0.10]:
        for f in F_sample:
            es = safe_es_portfolio(f, eps=eps)
            va = safe_var_portfolio(f)
            rows.append(['epsilon', eps, f,
                         es[0], es[1], es[2],
                         va[0], va[1], va[2],
                         es[3], va[3]])

    # Group C: mu_I
    for mu_I in [0.01, 0.023, 0.035, 0.05]:
        with override_params(MU_I=mu_I):
            for f in F_sample:
                es = safe_es_portfolio(f)
                va = safe_var_portfolio(f)
                rows.append(['mu_I', mu_I, f,
                             es[0], es[1], es[2],
                             va[0], va[1], va[2],
                             es[3], va[3]])

    # Group D: T
    for T_val in [5, 10, 15, 20]:
        with override_params(T=float(T_val)):
            for f in F_sample:
                es = safe_es_portfolio(f)
                va = safe_var_portfolio(f)
                rows.append(['T', T_val, f,
                             es[0], es[1], es[2],
                             va[0], va[1], va[2],
                             es[3], va[3]])

    # Group E: rho
    for rho in [-0.30, -0.15, -0.05]:
        with override_params(RHO=rho):
            for f in F_sample:
                es = safe_es_portfolio(f)
                va = safe_var_portfolio(f)
                rows.append(['rho', rho, f,
                             es[0], es[1], es[2],
                             va[0], va[1], va[2],
                             es[3], va[3]])

    csv_path = os.path.join(OUT, 'summary_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f'{v:.6f}' if isinstance(v, float) else v for v in row])
    print(f"  Saved summary_table.csv ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    apply_style()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 55)
    print("  Sensitivity Analysis")
    print("=" * 55)
    P.print_params()
    print()

    print("Group A: Risk Aversion (gamma)...")
    plot_group_A()

    print("Group B: ES Budget (epsilon)...")
    plot_group_B()

    print("Group C: Expected Inflation (mu_I)...")
    plot_group_C()

    print("Group D: Investment Horizon (T)...")
    plot_group_D()

    print("Group E: Stock-IIB Correlation (rho)...")
    plot_group_E()

    print("\nGenerating summary table...")
    generate_summary_table()

    print(f"\nAll outputs saved to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
