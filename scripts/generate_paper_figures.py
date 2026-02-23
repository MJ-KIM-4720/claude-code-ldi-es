"""
Paper Figure Generator
======================
Generates journal-submission-quality figures to paper/figures/.

Style rules:
  - No fig.suptitle() — captions handled in LaTeX
  - Subplot titles (panel labels) retained
  - ES: red solid lw=2.5, VaR: blue dashed lw=2.5, Merton: gray dotted lw=1.5
  - A=1 reference: gray dotted lw=1.0
  - ES-only sensitivity: warm palette (dark red → bright yellow)
  - White background, light gray grid (#CCCCCC)
  - F_grid: 500 points, DPI: 300, PNG, bbox_inches='tight'
  - MC colors: ES=red, VaR=blue, Merton=gray

Usage:
    python scripts/generate_paper_figures.py
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldi import params as P, es_model as ES, var_model as VaR
from ldi.params import override_params
from ldi.bs_utils import bs_put, bs_digital_put
from ldi import monte_carlo as MC
from ldi.style import (
    COLORS, OPTION_DECOMP, FIGSIZES, FAN_ALPHA, HIST_ALPHA, LEGEND, K_LINE,
    WARM_PALETTE, PAPER_LINE_STYLES, PAPER_REF_LINE, PAPER_GRID,
    PAPER_GAMBLING, apply_paper_style, paper_grid, paper_hline, paper_savefig,
)

# ── Constants ─────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
F_RANGE = np.linspace(0.5, 1.3, 500)

LS = PAPER_LINE_STYLES   # shorthand


# ═══════════════════════════════════════════════════════════════
# Safe wrappers (same as existing scripts)
# ═══════════════════════════════════════════════════════════════

def safe_es_threshold(y0):
    try:
        return ES.solve_threshold(y0)
    except Exception:
        return None


def safe_var_threshold(y0):
    try:
        return VaR.solve_threshold(y0, alpha=0.1)
    except Exception:
        return None


def safe_es_A(y0, eps=None):
    try:
        return ES.cross_sectional_A(y0, eps)
    except Exception:
        return np.nan


def safe_var_A(y0, alpha=None):
    try:
        return VaR.cross_sectional_A(y0, alpha)
    except Exception:
        return np.nan


def compute_es_totals(F_range, eps=None):
    totals = np.full(len(F_range), np.nan)
    for i, f in enumerate(F_range):
        A = safe_es_A(f, eps)
        totals[i] = A * P.Pi_star.sum()
    return totals


def compute_var_totals(F_range, alpha=None):
    totals = np.full(len(F_range), np.nan)
    for i, f in enumerate(F_range):
        A = safe_var_A(f, alpha)
        totals[i] = A * P.Pi_star.sum()
    return totals


def compute_es_components(F_range, eps=None):
    pi_S = np.full(len(F_range), np.nan)
    pi_I = np.full(len(F_range), np.nan)
    for i, f in enumerate(F_range):
        A = safe_es_A(f, eps)
        pi_S[i] = A * P.Pi_star[0]
        pi_I[i] = A * P.Pi_star[1]
    return pi_S, pi_I


def compute_es_A_array(F_range, eps=None):
    return np.array([safe_es_A(f, eps) for f in F_range])


def compute_var_A_array(F_range, alpha=None):
    return np.array([safe_var_A(f, alpha) for f in F_range])


# ═══════════════════════════════════════════════════════════════
# Baseline Figures (5)
# ═══════════════════════════════════════════════════════════════

def plot_baseline_claim_function():
    """Fig 1: Terminal claim function g(y) for ES and VaR at Y₀ = 1.0."""
    y0 = 1.0
    k = P.k
    k_eps, c, _ = ES.solve_threshold(y0)
    k_alpha, _ = VaR.solve_threshold(y0, alpha=0.1)

    y = np.linspace(0, 1.5, 2000)
    g_es = np.where(y < k_eps, c * y, np.where(y < k, k, y))
    g_var = np.where(y < k_alpha, y, np.where(y < k, k, y))
    g_unc = y

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    ax.plot(y, g_es, label='ES constraint', **LS['ES'])
    ax.plot(y, g_var, label='VaR constraint', **LS['VaR'])
    ax.plot(y, g_unc, label='Unconstrained (45° line)', **LS['Merton'])

    ax.axvline(k, **K_LINE)
    ax.text(k + 0.02, ax.get_ylim()[0] + 0.05, f'k = {k:.1f}',
            fontsize=10, color='green')
    ax.axvline(k_eps, color=COLORS['ES'], ls='--', alpha=0.5, lw=1.2)
    ax.text(k_eps + 0.02, 0.15, f'$k_\\varepsilon$ = {k_eps:.3f}',
            fontsize=10, color=COLORS['ES'])
    ax.axvline(k_alpha, color=COLORS['VaR'], ls='--', alpha=0.5, lw=1.2)
    ax.text(k_alpha - 0.15, 0.05, f'$k_\\alpha$ = {k_alpha:.3f}',
            fontsize=10, color=COLORS['VaR'])

    ax.set_xlabel('$y$ (terminal funding ratio)')
    ax.set_ylabel('$g(y)$ (claim function)')
    ax.legend(loc='upper left', framealpha=LEGEND['framealpha'],
              edgecolor=LEGEND['edgecolor'])
    paper_grid(ax)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_baseline_claim_function.png'))
    print("  Saved fig_baseline_claim_function.png")


def plot_baseline_present_value():
    """Fig 2: Present value Ψ(F) for ES and VaR."""
    psi_es = np.full(len(F_RANGE), np.nan)
    psi_var = np.full(len(F_RANGE), np.nan)

    for i, F in enumerate(F_RANGE):
        res = safe_es_threshold(F)
        if res is not None:
            k_eps, c, binding = res
            psi_es[i] = ES.psi(F, k_eps, c) if binding else F
        res = safe_var_threshold(F)
        if res is not None:
            k_alpha, binding = res
            psi_var[i] = VaR.psi(F, k_alpha) if binding else F

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    ax.plot(F_RANGE, psi_es, label=r'$\Psi_{ES}$', **LS['ES'])
    ax.plot(F_RANGE, psi_var, label=r'$\Psi_{VaR}$', **LS['VaR'])
    ax.plot(F_RANGE, F_RANGE, label=r'Unconstrained ($\Psi = F$)', **LS['Merton'])

    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'$\Psi(t, F)$ (present value)')
    ax.legend(**LEGEND)
    paper_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_baseline_present_value.png'))
    print("  Saved fig_baseline_present_value.png")


def plot_baseline_adjustment_factor():
    """Fig 3: Adjustment factor A(F) with gambling region shading."""
    A_es = np.full(len(F_RANGE), np.nan)
    A_var = np.full(len(F_RANGE), np.nan)

    for i, F in enumerate(F_RANGE):
        try:
            A_es[i] = ES.cross_sectional_A(F)
        except Exception:
            pass
        try:
            A_var[i] = VaR.cross_sectional_A(F, alpha=0.1)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    ax.plot(F_RANGE, A_es, label=r'ES ($\varepsilon$ = 0.05)', **LS['ES'])
    ax.plot(F_RANGE, A_var, label=r'VaR ($\alpha$ = 0.1)', **LS['VaR'])
    paper_hline(ax, 1.0, 'A = 1 (Merton)')

    gambling_mask = A_var > 1.0
    if np.any(gambling_mask):
        ax.fill_between(F_RANGE, 1.0, A_var, where=gambling_mask,
                        **PAPER_GAMBLING, label='Gambling region (VaR)')
        gambling_idx = np.where(gambling_mask)[0]
        if len(gambling_idx) > 0:
            peak_idx = gambling_idx[np.argmax(A_var[gambling_idx])]
            ax.annotate('Gambling\nRegion',
                        xy=(F_RANGE[peak_idx], A_var[peak_idx]),
                        xytext=(F_RANGE[peak_idx] + 0.08, A_var[peak_idx] + 0.15),
                        fontsize=10, color=COLORS['ES'], fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=COLORS['ES'], lw=1.5),
                        ha='center')

    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel('Adjustment Factor $A(F)$')
    ax.legend(**LEGEND)
    paper_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_baseline_adjustment_factor.png'))
    print("  Saved fig_baseline_adjustment_factor.png")


def plot_baseline_allocation():
    """Fig 4: Optimal allocation 1×3 (Stock, IIB, Total Risky)."""
    pi_S_es = np.full(len(F_RANGE), np.nan)
    pi_I_es = np.full(len(F_RANGE), np.nan)
    pi_S_var = np.full(len(F_RANGE), np.nan)
    pi_I_var = np.full(len(F_RANGE), np.nan)

    for i, F in enumerate(F_RANGE):
        res = safe_es_threshold(F)
        if res is not None:
            k_eps, c, binding = res
            if binding:
                ps, pi = ES.optimal_portfolio(F, k_eps, c)
            else:
                ps, pi = P.Pi_star[0], P.Pi_star[1]
            pi_S_es[i] = ps
            pi_I_es[i] = pi
        res = safe_var_threshold(F)
        if res is not None:
            k_alpha, binding = res
            if binding:
                ps, pi = VaR.optimal_portfolio(F, k_alpha)
            else:
                ps, pi = P.Pi_star[0], P.Pi_star[1]
            pi_S_var[i] = ps
            pi_I_var[i] = pi

    total_es = pi_S_es + pi_I_es
    total_var = pi_S_var + pi_I_var
    merton_S = P.Pi_star[0]
    merton_I = P.Pi_star[1]
    merton_total = P.Pi_star.sum()

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZES['triple'])

    titles = ['Stock', 'Inflation-Indexed Bond', 'Total Risky']
    ylabels = [r'$\pi_S$', r'$\pi_I$', r'$\pi_S + \pi_I$']
    es_data = [pi_S_es, pi_I_es, total_es]
    var_data = [pi_S_var, pi_I_var, total_var]
    merton_vals = [merton_S, merton_I, merton_total]

    for ax, title, ylabel, es_d, var_d, m_val in zip(
            axes, titles, ylabels, es_data, var_data, merton_vals):
        ax.plot(F_RANGE, es_d, label='ES', **LS['ES'])
        ax.plot(F_RANGE, var_d, label='VaR', **LS['VaR'])
        paper_hline(ax, m_val, f'Merton ({m_val:.2f})')
        ax.set_xlabel('Funding Ratio $F(t)$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(**LEGEND)
        paper_grid(ax)
        ax.set_xlim(0.5, 1.3)

    # NO suptitle
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_baseline_allocation.png'))
    print("  Saved fig_baseline_allocation.png")


def plot_baseline_option_decomposition():
    """Fig 5: Option decomposition 1×2 (ES panel, VaR panel)."""
    tau = P.T

    es_put_k = np.full(len(F_RANGE), np.nan)
    es_neg_c_put_ke = np.full(len(F_RANGE), np.nan)
    es_net = np.full(len(F_RANGE), np.nan)

    var_put_k = np.full(len(F_RANGE), np.nan)
    var_neg_put_ka = np.full(len(F_RANGE), np.nan)
    var_neg_digital = np.full(len(F_RANGE), np.nan)
    var_net = np.full(len(F_RANGE), np.nan)

    for i, F in enumerate(F_RANGE):
        res = safe_es_threshold(F)
        if res is not None:
            k_eps, c, binding = res
            pk = bs_put(F, P.k, P.r_tilde, P.sigma_Y, tau)
            if binding:
                pke = bs_put(F, k_eps, P.r_tilde, P.sigma_Y, tau)
                es_put_k[i] = pk
                es_neg_c_put_ke[i] = -c * pke
                es_net[i] = pk - c * pke
            else:
                es_put_k[i] = pk
                es_neg_c_put_ke[i] = -pk
                es_net[i] = 0.0
        res = safe_var_threshold(F)
        if res is not None:
            k_alpha, binding = res
            pk = bs_put(F, P.k, P.r_tilde, P.sigma_Y, tau)
            if binding:
                pka = bs_put(F, k_alpha, P.r_tilde, P.sigma_Y, tau)
                dka = bs_digital_put(F, k_alpha, P.r_tilde, P.sigma_Y, tau)
                var_put_k[i] = pk
                var_neg_put_ka[i] = -pka
                var_neg_digital[i] = -(P.k - k_alpha) * dka
                var_net[i] = pk - pka - (P.k - k_alpha) * dka
            else:
                var_put_k[i] = pk
                var_neg_put_ka[i] = -pk
                var_neg_digital[i] = 0.0
                var_net[i] = 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZES['triple'])

    # Left: ES decomposition
    ax1.plot(F_RANGE, es_put_k, label='Put(F, k)', **OPTION_DECOMP['put_k'])
    ax1.plot(F_RANGE, es_neg_c_put_ke,
             label=r'$-c \cdot$ Put(F, $k_\varepsilon$)',
             **OPTION_DECOMP['neg_put'])
    ax1.plot(F_RANGE, es_net, label=r'Net ($\Psi_{ES} - F$)',
             **OPTION_DECOMP['net'])
    ax1.axhline(0, color=COLORS['Merton'], ls=':', alpha=0.4)
    ax1.set_xlabel('Funding Ratio $F(t)$')
    ax1.set_ylabel('Option value')
    ax1.set_title('ES: Option Decomposition')
    ax1.legend(**LEGEND)
    paper_grid(ax1)
    ax1.set_xlim(0.5, 1.3)

    # Right: VaR decomposition
    ax2.plot(F_RANGE, var_put_k, label='Put(F, k)', **OPTION_DECOMP['put_k'])
    ax2.plot(F_RANGE, var_neg_put_ka,
             label=r'$-$Put(F, $k_\alpha$)',
             **OPTION_DECOMP['neg_put'])
    ax2.plot(F_RANGE, var_neg_digital,
             label=r'$-(k-k_\alpha) \cdot$ Digital($k_\alpha$)',
             **OPTION_DECOMP['digital'])
    ax2.plot(F_RANGE, var_net, label=r'Net ($\Psi_{VaR} - F$)',
             **OPTION_DECOMP['net'])
    ax2.axhline(0, color=COLORS['Merton'], ls=':', alpha=0.4)
    ax2.set_xlabel('Funding Ratio $F(t)$')
    ax2.set_ylabel('Option value')
    ax2.set_title('VaR: Option Decomposition')
    ax2.legend(**LEGEND)
    paper_grid(ax2)
    ax2.set_xlim(0.5, 1.3)

    # NO suptitle
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_baseline_option_decomposition.png'))
    print("  Saved fig_baseline_option_decomposition.png")


# ═══════════════════════════════════════════════════════════════
# Sensitivity Figures (11)
# ═══════════════════════════════════════════════════════════════

# ── Group A: Risk Aversion (gamma) ────────────────────────────

def plot_group_A():
    gammas = [2, 3, 5, 7]
    warm = WARM_PALETTE[:4]

    # A1: ES-only by gamma — warm palette
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for gamma, color in zip(gammas, warm):
        with override_params(GAMMA=float(gamma)):
            totals = compute_es_totals(F_RANGE)
            ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                    label=fr'$\gamma = {gamma}$')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.legend(**LEGEND)
    paper_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_A1_gamma_es.png'))
    print("  Saved fig_A1_gamma_es.png")

    # A2: ES vs VaR 2×2 — ES red / VaR blue / Merton gray
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZES['quad'])
    for ax, gamma in zip(axes.flat, gammas):
        with override_params(GAMMA=float(gamma)):
            es_tot = compute_es_totals(F_RANGE)
            var_tot = compute_var_totals(F_RANGE)
            merton = P.Pi_star.sum()
            ax.plot(F_RANGE, es_tot, label='ES', **LS['ES'])
            ax.plot(F_RANGE, var_tot, label='VaR', **LS['VaR'])
            paper_hline(ax, merton, f'Merton ({merton:.2f})')
            ax.set_title(fr'$\gamma = {gamma}$')
            ax.set_xlabel('Funding Ratio $F(t)$')
            ax.set_ylabel('Total Risky Allocation')
            ax.legend(**LEGEND)
            paper_grid(ax)
            ax.set_xlim(0.5, 1.3)
    # NO suptitle
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_A2_gamma_compare.png'))
    print("  Saved fig_A2_gamma_compare.png")

    # A3: A(F) 2×2 — ES red / VaR blue, gambling shading
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZES['quad'])
    for ax, gamma in zip(axes.flat, gammas):
        with override_params(GAMMA=float(gamma)):
            es_A = compute_es_A_array(F_RANGE)
            var_A = compute_var_A_array(F_RANGE)
            ax.plot(F_RANGE, es_A, label='ES', **LS['ES'])
            ax.plot(F_RANGE, var_A, label='VaR', **LS['VaR'])
            paper_hline(ax, 1.0, 'A = 1 (Merton)')
            gambling_mask = var_A > 1.0
            if np.any(gambling_mask):
                ax.fill_between(F_RANGE, 1.0, var_A, where=gambling_mask,
                                **PAPER_GAMBLING, label='VaR gambling')
            ax.set_title(fr'$\gamma = {gamma}$')
            ax.set_xlabel('Funding Ratio $F(t)$')
            ax.set_ylabel('Adjustment Factor $A(F)$')
            ax.legend(**LEGEND)
            paper_grid(ax)
            ax.set_xlim(0.5, 1.3)
    # NO suptitle
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_A3_gamma_A_factor.png'))
    print("  Saved fig_A3_gamma_A_factor.png")


# ── Group B: ES Budget (epsilon) ─────────────────────────────

def plot_group_B():
    epsilons = [0.02, 0.03, 0.05, 0.08, 0.10]
    warm = WARM_PALETTE[:5]

    # B1: ES-only by epsilon — warm palette
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for eps, color in zip(epsilons, warm):
        totals = compute_es_totals(F_RANGE, eps=eps)
        ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                label=fr'$\varepsilon = {eps}$')
    merton = P.Pi_star.sum()
    paper_hline(ax, merton, f'Merton ({merton:.2f})')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.legend(**LEGEND)
    paper_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_B1_epsilon_es.png'))
    print("  Saved fig_B1_epsilon_es.png")

    # B2: ES (warm) + VaR (blue dashed) comparison
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    var_tot = compute_var_totals(F_RANGE)
    ax.plot(F_RANGE, var_tot, label=fr'VaR ($\alpha = {P.alpha}$)', **LS['VaR'])
    for eps, color in zip(epsilons, warm):
        totals = compute_es_totals(F_RANGE, eps=eps)
        ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                label=fr'ES ($\varepsilon = {eps}$)')
    merton = P.Pi_star.sum()
    paper_hline(ax, merton, f'Merton ({merton:.2f})')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.legend(**LEGEND)
    paper_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_B2_epsilon_compare.png'))
    print("  Saved fig_B2_epsilon_compare.png")


# ── Group C: Expected Inflation (mu_I) ───────────────────────

def plot_group_C():
    mu_Is = [0.015, 0.023, 0.035, 0.05]
    warm = WARM_PALETTE[:4]

    # C1: ES total allocation by mu_I — warm palette
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for mu_I, color in zip(mu_Is, warm):
        with override_params(MU_I=mu_I):
            totals = compute_es_totals(F_RANGE)
            ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                    label=fr'$\mu_I = {mu_I}$')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.legend(**LEGEND)
    paper_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_C1_muI_es.png'))
    print("  Saved fig_C1_muI_es.png")

    # C2: Stock vs IIB 1×2 — warm palette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZES['triple'])
    for mu_I, color in zip(mu_Is, warm):
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
    paper_grid(ax1)
    ax1.set_xlim(0.5, 1.3)

    ax2.set_xlabel('Funding Ratio $F(t)$')
    ax2.set_ylabel(r'IIB Allocation ($\pi_I$)')
    ax2.set_title(r'IIB Allocation ($\pi_I$)')
    ax2.legend(**LEGEND)
    paper_grid(ax2)
    ax2.set_xlim(0.5, 1.3)

    # NO suptitle
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_C2_muI_components.png'))
    print("  Saved fig_C2_muI_components.png")


# ── Group D: Investment Horizon (T) ──────────────────────────

def plot_group_D():
    Ts = [5, 10, 15, 20]
    warm = WARM_PALETTE[:4]

    # D1: ES total allocation by T — warm palette
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for T_val, color in zip(Ts, warm):
        with override_params(T=float(T_val)):
            totals = compute_es_totals(F_RANGE)
            ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                    label=f'$T = {T_val}$')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.legend(**LEGEND)
    paper_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_D1_T_es.png'))
    print("  Saved fig_D1_T_es.png")

    # D2: ES vs VaR 2×2 — ES red / VaR blue / Merton gray
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZES['quad'])
    for ax, T_val in zip(axes.flat, Ts):
        with override_params(T=float(T_val)):
            es_tot = compute_es_totals(F_RANGE)
            var_tot = compute_var_totals(F_RANGE)
            merton = P.Pi_star.sum()
            ax.plot(F_RANGE, es_tot, label='ES', **LS['ES'])
            ax.plot(F_RANGE, var_tot, label='VaR', **LS['VaR'])
            paper_hline(ax, merton, f'Merton ({merton:.2f})')
            ax.set_title(f'$T = {T_val}$')
            ax.set_xlabel('Funding Ratio $F(t)$')
            ax.set_ylabel('Total Risky Allocation')
            ax.legend(**LEGEND)
            paper_grid(ax)
            ax.set_xlim(0.5, 1.3)
    # NO suptitle
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_D2_T_compare.png'))
    print("  Saved fig_D2_T_compare.png")


# ── Group E: Stock-IIB Correlation (rho) ─────────────────────

def plot_group_E():
    rhos = [-0.30, -0.15, -0.05]
    warm = WARM_PALETTE[:3]

    # E1: ES total allocation by rho — warm palette
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for rho, color in zip(rhos, warm):
        with override_params(RHO=rho):
            totals = compute_es_totals(F_RANGE)
            ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                    label=fr'$\rho = {rho}$')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.legend(**LEGEND)
    paper_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_E1_rho_es.png'))
    print("  Saved fig_E1_rho_es.png")

    # E2: Stock vs IIB 1×2 — warm palette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZES['triple'])
    for rho, color in zip(rhos, warm):
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
    paper_grid(ax1)
    ax1.set_xlim(0.5, 1.3)

    ax2.set_xlabel('Funding Ratio $F(t)$')
    ax2.set_ylabel(r'IIB Allocation ($\pi_I$)')
    ax2.set_title(r'IIB Allocation ($\pi_I$)')
    ax2.legend(**LEGEND)
    paper_grid(ax2)
    ax2.set_xlim(0.5, 1.3)

    # NO suptitle
    plt.tight_layout()
    paper_savefig(fig, os.path.join(OUT, 'fig_E2_rho_components.png'))
    print("  Saved fig_E2_rho_components.png")


# ═══════════════════════════════════════════════════════════════
# Monte Carlo Figures (12)
# ═══════════════════════════════════════════════════════════════

N_PATHS = 10000
N_STEPS = 250
SEED = 42
MODELS = ['es', 'var', 'merton']
LABELS = {'es': 'ES', 'var': 'VaR', 'merton': 'Merton'}
MC_COLORS = {
    'es':     COLORS['ES'],      # red
    'var':    COLORS['VaR'],     # blue
    'merton': COLORS['Merton'],  # gray
}
MC_LINE_STYLES = {
    'es':     {'color': COLORS['ES'],     'linestyle': '-',  'linewidth': 2},
    'var':    {'color': COLORS['VaR'],    'linestyle': '--', 'linewidth': 2},
    'merton': {'color': COLORS['Merton'], 'linestyle': ':',  'linewidth': 1.5},
}
MERTON_REF = {'color': COLORS['Merton'], 'linestyle': ':', 'linewidth': 1.5}


def run_mc_scenario(y0):
    """Run MC for all models and generate 4 figures for a given y0."""
    results = {}
    for model in MODELS:
        t0 = time.time()
        paths, t_grid = MC.simulate_paths(
            y0, n_paths=N_PATHS, n_steps=N_STEPS, model=model, seed=SEED)
        elapsed = time.time() - t0
        stats = MC.compute_path_stats(paths)
        tstats = MC.compute_terminal_stats(paths)
        sf_prob = MC.shortfall_prob_over_time(paths)
        results[model] = {
            'paths': paths, 't_grid': t_grid,
            'stats': stats, 'tstats': tstats, 'sf_prob': sf_prob,
        }
        print(f"    {LABELS[model]:>6}: E[Y_T]={tstats['mean']:.3f}  "
              f"P(short)={tstats['shortfall_prob']:.3f}  ({elapsed:.1f}s)")

    suffix = f"_y0{y0:.1f}".replace(".", "")

    # ── Fan charts (1×3) ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZES['triple'], sharey=True)
    for ax, model in zip(axes, MODELS):
        s = results[model]['stats']
        t = results[model]['t_grid']
        c = MC_COLORS[model]
        ax.fill_between(t, s['q05'], s['q95'], alpha=FAN_ALPHA['outer'], color=c)
        ax.fill_between(t, s['q25'], s['q75'], alpha=FAN_ALPHA['middle'], color=c)
        ax.plot(t, s['median'], color=c, lw=2, label='Median')
        ax.plot(t, s['mean'], color=c, lw=1, ls='--', label='Mean')
        ax.axhline(P.k, **MERTON_REF, label=f'k={P.k}')
        ax.set_title(f"{LABELS[model]}  ($F_0$={y0})")
        ax.set_xlabel('$t$ (years)')
        ax.legend(**LEGEND)
        ax.set_ylim(0, max(3.0, s['q95'].max() * 1.2))
        paper_grid(ax)
    axes[0].set_ylabel('Funding Ratio $F(t)$')
    # NO suptitle
    fig.tight_layout()
    paper_savefig(fig, os.path.join(OUT, f"mc_fan{suffix}.png"))
    print(f"  Saved mc_fan{suffix}.png")

    # ── Terminal distribution ───────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for model in MODELS:
        Y_T = results[model]['paths'][:, -1]
        Y_T_clip = np.clip(Y_T, 0, np.quantile(Y_T, 0.99))
        ax.hist(Y_T_clip, bins=80, alpha=HIST_ALPHA, color=MC_COLORS[model],
                label=f"{LABELS[model]} ($\\mu$={results[model]['tstats']['mean']:.2f})",
                density=True)
    ax.axvline(P.k, **MERTON_REF, label=f'k={P.k}')
    ax.set_xlabel('Terminal Funding Ratio $F_T$')
    ax.set_ylabel('Density')
    ax.legend(**LEGEND)
    paper_grid(ax)
    fig.tight_layout()
    paper_savefig(fig, os.path.join(OUT, f"mc_terminal{suffix}.png"))
    print(f"  Saved mc_terminal{suffix}.png")

    # ── Shortfall probability over time ─────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for model in MODELS:
        t = results[model]['t_grid']
        sf = results[model]['sf_prob']
        ax.plot(t, sf, label=LABELS[model], **MC_LINE_STYLES[model])
    ax.axhline(P.alpha, **MERTON_REF, label=f'$\\alpha$={P.alpha}')
    ax.set_xlabel('$t$ (years)')
    ax.set_ylabel('$P(F_t < k)$')
    ax.legend(**LEGEND)
    ax.set_ylim(-0.02, 1.02)
    paper_grid(ax)
    fig.tight_layout()
    paper_savefig(fig, os.path.join(OUT, f"mc_shortfall{suffix}.png"))
    print(f"  Saved mc_shortfall{suffix}.png")

    # ── Sample paths (1×3) ──────────────────────────────────
    n_sample = 20
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZES['triple'], sharey=True)
    for ax, model in zip(axes, MODELS):
        t = results[model]['t_grid']
        c = MC_COLORS[model]
        for j in range(n_sample):
            ax.plot(t, results[model]['paths'][j, :],
                    alpha=0.4, lw=0.6, color=c)
        ax.axhline(P.k, **MERTON_REF)
        ax.set_title(f"{LABELS[model]}  ($F_0$={y0})")
        ax.set_xlabel('$t$ (years)')
        ax.set_ylim(0, max(3.0, results[model]['paths'][:n_sample].max() * 1.1))
        paper_grid(ax)
    axes[0].set_ylabel('Funding Ratio $F(t)$')
    # NO suptitle
    fig.tight_layout()
    paper_savefig(fig, os.path.join(OUT, f"mc_samples{suffix}.png"))
    print(f"  Saved mc_samples{suffix}.png")

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    apply_paper_style()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 60)
    print("  Paper Figure Generator — Journal Submission Quality")
    print("=" * 60)
    P.print_params()
    print(f"\n  Output: {os.path.abspath(OUT)}")
    print(f"  Style:  white bg, #CCCCCC grid, DPI 300")
    print(f"  Colors: ES=red, VaR=blue, Merton=gray")
    print(f"  Warm palette: {WARM_PALETTE}")
    print()

    # ── Baseline (5 figures) ────────────────────────────────
    print("Baseline Figures:")
    plot_baseline_claim_function()
    plot_baseline_present_value()
    plot_baseline_adjustment_factor()
    plot_baseline_allocation()
    plot_baseline_option_decomposition()

    # ── Sensitivity (11 figures) ────────────────────────────
    print("\nSensitivity Figures:")

    print("  Group A: Risk Aversion (gamma)...")
    plot_group_A()

    print("  Group B: ES Budget (epsilon)...")
    plot_group_B()

    print("  Group C: Expected Inflation (mu_I)...")
    plot_group_C()

    print("  Group D: Investment Horizon (T)...")
    plot_group_D()

    print("  Group E: Stock-IIB Correlation (rho)...")
    plot_group_E()

    # ── Monte Carlo (12 figures) ────────────────────────────
    print("\nMonte Carlo Figures:")
    for y0 in [0.8, 1.0, 1.2]:
        print(f"  Scenario y0 = {y0}:")
        run_mc_scenario(y0)

    print(f"\n{'='*60}")
    print(f"  All 28 paper figures saved to {os.path.abspath(OUT)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
