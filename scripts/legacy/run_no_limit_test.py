"""
No-Limit Test: Verify A_ES behavior without clamping
=====================================================
Generates figures to outputs/no_limit/ with 500-point grid.
Prints max A_ES values for verification.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldi import params as P, es_model as ES, var_model as VaR
from ldi.params import override_params
from ldi.bs_utils import bs_put, bs_digital_put
from ldi.style import (apply_style, COLORS, LINE_STYLES, OPTION_DECOMP,
                        FIGSIZES, DPI, LEGEND, MERTON_LINE, K_LINE,
                        GAMBLING_REGION, setup_grid, add_merton_hline,
                        add_k_vline, savefig)

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "no_limit")
F_RANGE = np.linspace(0.5, 1.3, 500)


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


# ═══════════════════════════════════════════════════════════
# Fig 1: Claim Function
# ═══════════════════════════════════════════════════════════

def plot_claim_function():
    y0 = 1.0
    k = P.k
    k_eps, c, _ = ES.solve_threshold(y0)
    k_alpha, _ = VaR.solve_threshold(y0, alpha=0.1)

    y = np.linspace(0, 1.5, 2000)
    g_es = np.where(y < k_eps, c * y, np.where(y < k, k, y))
    g_var = np.where(y < k_alpha, y, np.where(y < k, k, y))
    g_unc = y

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    ax.plot(y, g_es, label='ES constraint', **LINE_STYLES['ES'])
    ax.plot(y, g_var, label='VaR constraint', **LINE_STYLES['VaR'])
    ax.plot(y, g_unc, label='Unconstrained (45° line)', **LINE_STYLES['Merton'])
    add_k_vline(ax, k)
    ax.axvline(k_eps, color=COLORS['ES'], ls='--', alpha=0.5, lw=1.2)
    ax.axvline(k_alpha, color=COLORS['VaR'], ls='--', alpha=0.5, lw=1.2)
    ax.set_xlabel('$y$ (terminal funding ratio)')
    ax.set_ylabel('$g(y)$ (claim function)')
    ax.set_title(r'Terminal Claim Function: ES vs VaR ($Y_0$ = 1.0)')
    ax.legend(loc='upper left', framealpha=LEGEND['framealpha'],
              edgecolor=LEGEND['edgecolor'])
    setup_grid(ax)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(bottom=0)
    plt.suptitle(f'$R$={P.R}, $r$={P.r}, $\\gamma$={P.GAMMA}, $T$={P.T}')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_baseline_claim_function.png')
    savefig(fig, path)
    print(f"  Saved {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════
# Fig 2: Adjustment Factor
# ═══════════════════════════════════════════════════════════

def plot_adjustment_factor():
    A_es = np.full(len(F_RANGE), np.nan)
    A_var = np.full(len(F_RANGE), np.nan)

    for i, F in enumerate(F_RANGE):
        A_es[i] = safe_es_A(F)
        A_var[i] = safe_var_A(F, alpha=0.1)

    # Print max A_ES
    valid = ~np.isnan(A_es)
    print(f"  max A_ES = {A_es[valid].max():.6f}")
    print(f"  min A_ES = {A_es[valid].min():.6f}")

    # Find transition point (last binding F)
    binding_mask = A_es < 1.0 - 1e-10
    if np.any(binding_mask):
        last_binding_idx = np.where(binding_mask)[0][-1]
        print(f"  Last binding F = {F_RANGE[last_binding_idx]:.4f}, A = {A_es[last_binding_idx]:.6f}")
        if last_binding_idx + 1 < len(F_RANGE):
            print(f"  Next F = {F_RANGE[last_binding_idx + 1]:.4f}, A = {A_es[last_binding_idx + 1]:.6f}")

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    ax.plot(F_RANGE, A_es, label=r'ES ($\varepsilon$ = 0.05)', **LINE_STYLES['ES'])
    ax.plot(F_RANGE, A_var, label=r'VaR ($\alpha$ = 0.1)', **LINE_STYLES['VaR'])
    add_merton_hline(ax, 1.0, 'A = 1 (Merton)')

    gambling_mask = A_var > 1.0
    if np.any(gambling_mask):
        ax.fill_between(F_RANGE, 1.0, A_var, where=gambling_mask,
                        **GAMBLING_REGION, label='Gambling region (VaR)')

    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel('Adjustment Factor $A(F)$')
    ax.set_title('Portfolio Adjustment Factor: ES vs VaR Constraint')
    ax.legend(**LEGEND)
    setup_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.suptitle(f'$R$={P.R}, $r$={P.r}, $\\gamma$={P.GAMMA}, $T$={P.T}')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_baseline_adjustment_factor.png')
    savefig(fig, path)
    print(f"  Saved {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════
# Fig 3: Allocation
# ═══════════════════════════════════════════════════════════

def plot_allocation():
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

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZES['triple'])
    titles = ['Stock', 'Inflation-Indexed Bond', 'Total Risky']
    ylabels = [r'$\pi_S$', r'$\pi_I$', r'$\pi_S + \pi_I$']
    es_data = [pi_S_es, pi_I_es, total_es]
    var_data = [pi_S_var, pi_I_var, total_var]
    merton_vals = [P.Pi_star[0], P.Pi_star[1], P.Pi_star.sum()]

    for ax, title, ylabel, es_d, var_d, m_val in zip(
            axes, titles, ylabels, es_data, var_data, merton_vals):
        ax.plot(F_RANGE, es_d, label='ES', **LINE_STYLES['ES'])
        ax.plot(F_RANGE, var_d, label='VaR', **LINE_STYLES['VaR'])
        add_merton_hline(ax, m_val, f'Merton ({m_val:.2f})')
        ax.set_xlabel('Funding Ratio $F(t)$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(**LEGEND)
        setup_grid(ax)
        ax.set_xlim(0.5, 1.3)

    fig.suptitle('Optimal Asset Allocation: ES vs VaR Constraint')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_baseline_allocation.png')
    savefig(fig, path)
    print(f"  Saved {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════
# Fig 4: Present Value
# ═══════════════════════════════════════════════════════════

def plot_present_value():
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
    ax.plot(F_RANGE, psi_es, label=r'$\Psi_{ES}$', **LINE_STYLES['ES'])
    ax.plot(F_RANGE, psi_var, label=r'$\Psi_{VaR}$', **LINE_STYLES['VaR'])
    ax.plot(F_RANGE, F_RANGE, label=r'Unconstrained ($\Psi = F$)',
            **LINE_STYLES['Merton'])
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'$\Psi(t, F)$ (present value)')
    ax.set_title('Present Value of Constrained Portfolio: ES vs VaR')
    ax.legend(**LEGEND)
    setup_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.suptitle(f'$R$={P.R}, $r$={P.r}, $\\gamma$={P.GAMMA}, $T$={P.T}')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_baseline_present_value.png')
    savefig(fig, path)
    print(f"  Saved {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════
# Fig 5: Option Decomposition
# ═══════════════════════════════════════════════════════════

def plot_option_decomposition():
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
    setup_grid(ax1)
    ax1.set_xlim(0.5, 1.3)

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
    setup_grid(ax2)
    ax2.set_xlim(0.5, 1.3)

    fig.suptitle('Option Portfolio Decomposition: ES vs VaR')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_baseline_option_decomposition.png')
    savefig(fig, path)
    print(f"  Saved {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════
# Sensitivity: A1 gamma ES
# ═══════════════════════════════════════════════════════════

def plot_A1_gamma_es():
    gammas = [2, 3, 5, 7]
    colors = COLORS['param_values'][:4]

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for gamma, color in zip(gammas, colors):
        with override_params(GAMMA=float(gamma)):
            totals = np.full(len(F_RANGE), np.nan)
            for i, f in enumerate(F_RANGE):
                A = safe_es_A(f)
                totals[i] = A * P.Pi_star.sum()
            ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                    label=fr'$\gamma = {gamma}$')
            # Print max A for each gamma
            A_arr = np.array([safe_es_A(f) for f in F_RANGE])
            valid = ~np.isnan(A_arr)
            print(f"  gamma={gamma}: max A_ES = {A_arr[valid].max():.6f}")

    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.set_title(r'ES Constraint: Effect of Risk Aversion ($\gamma$)')
    ax.legend(**LEGEND)
    setup_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.suptitle(f'$\\varepsilon$={P.epsilon}, $T$={P.T}')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_A1_gamma_es.png')
    savefig(fig, path)
    print(f"  Saved fig_A1_gamma_es.png")


# ═══════════════════════════════════════════════════════════
# Sensitivity: B1 epsilon ES
# ═══════════════════════════════════════════════════════════

def plot_B1_epsilon_es():
    epsilons = [0.02, 0.03, 0.05, 0.08, 0.10]
    colors = COLORS['param_values'][:5]

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for eps, color in zip(epsilons, colors):
        totals = np.full(len(F_RANGE), np.nan)
        for i, f in enumerate(F_RANGE):
            A = safe_es_A(f, eps=eps)
            totals[i] = A * P.Pi_star.sum()
        ax.plot(F_RANGE, totals, '-', color=color, lw=2,
                label=fr'$\varepsilon = {eps}$')
        # Print max A for each epsilon
        A_arr = np.array([safe_es_A(f, eps=eps) for f in F_RANGE])
        valid = ~np.isnan(A_arr)
        print(f"  eps={eps}: max A_ES = {A_arr[valid].max():.6f}")

    merton = P.Pi_star.sum()
    add_merton_hline(ax, merton, f'Merton ({merton:.2f})')
    ax.set_xlabel('Funding Ratio $F(t)$')
    ax.set_ylabel(r'Total Risky Allocation ($\pi_S + \pi_I$)')
    ax.set_title(r'ES Constraint: Effect of ES Budget ($\varepsilon$)')
    ax.legend(**LEGEND)
    setup_grid(ax)
    ax.set_xlim(0.5, 1.3)
    plt.suptitle(f'$\\gamma$={P.GAMMA}, $T$={P.T}')
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_B1_epsilon_es.png')
    savefig(fig, path)
    print(f"  Saved fig_B1_epsilon_es.png")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    apply_style()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 55)
    print("  No-Limit Test (Clamping Verification)")
    print("=" * 55)
    P.print_params()
    print(f"\n  Output: {os.path.abspath(OUT)}")
    print(f"  F_grid: {len(F_RANGE)} points, [{F_RANGE[0]}, {F_RANGE[-1]}]")
    print()

    print("Baseline Figures:")
    print("  Fig 1: Claim Function...")
    plot_claim_function()

    print("  Fig 2: Adjustment Factor...")
    plot_adjustment_factor()

    print("  Fig 3: Allocation...")
    plot_allocation()

    print("  Fig 4: Present Value...")
    plot_present_value()

    print("  Fig 5: Option Decomposition...")
    plot_option_decomposition()

    print("\nSensitivity Figures:")
    print("  A1: Gamma ES...")
    plot_A1_gamma_es()

    print("  B1: Epsilon ES...")
    plot_B1_epsilon_es()

    print(f"\nAll no_limit figures saved to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
