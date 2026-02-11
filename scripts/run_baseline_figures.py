"""
Baseline Figures
================
Generates 5 baseline figures comparing ES vs VaR constrained strategies.
All figures saved to results/ at DPI 300.

Figures:
  1. Terminal Claim Function (Y₀ = 1.0 fixed)
  2. Present Value Function (cross-sectional)
  3. Adjustment Factor (cross-sectional)
  4. Optimal Asset Allocation (cross-sectional, 1×3 subplots)
  5. Option Decomposition (cross-sectional, 1×2 subplots)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldi import params as P, es_model as ES, var_model as VaR
from ldi.bs_utils import bs_put, bs_digital_put

# ── Constants ──────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
DPI = 300


# ═══════════════════════════════════════════════════════════
# Safe wrappers
# ═══════════════════════════════════════════════════════════

def safe_es_threshold(y0):
    """Solve ES threshold, returning (k_eps, c, binding) or None on failure."""
    try:
        return ES.solve_threshold(y0)
    except Exception:
        return None


def safe_var_threshold(y0):
    """Solve VaR threshold with alpha=0.1, returning (k_alpha, binding) or None."""
    try:
        return VaR.solve_threshold(y0, alpha=0.1)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
# Fig 1: Terminal Claim Function
# ═══════════════════════════════════════════════════════════

def plot_claim_function():
    """Terminal claim function g(y) for ES and VaR at Y₀ = 1.0."""
    y0 = 1.0
    k = P.k

    # Solve thresholds at Y₀ = 1.0
    k_eps, c, es_bind = ES.solve_threshold(y0)
    k_alpha, var_bind = VaR.solve_threshold(y0, alpha=0.1)

    y = np.linspace(0, 1.5, 2000)

    # ES claim: g(y) = c·y for y < k_eps, k for k_eps ≤ y < k, y for y ≥ k
    g_es = np.where(y < k_eps, c * y,
                    np.where(y < k, k, y))

    # VaR claim: g(y) = y for y < k_alpha, k for k_alpha ≤ y < k, y for y ≥ k
    g_var = np.where(y < k_alpha, y,
                     np.where(y < k, k, y))

    # Unconstrained: g(y) = y (45-degree line)
    g_unc = y

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(y, g_es, 'r-', lw=2.5, label='ES constraint')
    ax.plot(y, g_var, 'b--', lw=2.5, label='VaR constraint')
    ax.plot(y, g_unc, ':', color='gray', lw=1.5, label='Unconstrained (45° line)')

    # Vertical dashed lines at thresholds
    ax.axvline(k, color='green', ls='--', alpha=0.6, lw=1.2)
    ax.text(k + 0.02, ax.get_ylim()[0] + 0.05, f'k = {k:.1f}',
            fontsize=10, color='green')

    ax.axvline(k_eps, color='red', ls='--', alpha=0.5, lw=1.2)
    ax.text(k_eps + 0.02, 0.15, f'$k_\\varepsilon$ = {k_eps:.3f}',
            fontsize=10, color='red')

    ax.axvline(k_alpha, color='blue', ls='--', alpha=0.5, lw=1.2)
    ax.text(k_alpha - 0.15, 0.05, f'$k_\\alpha$ = {k_alpha:.3f}',
            fontsize=10, color='blue')

    ax.set_xlabel('y (terminal funding ratio)', fontsize=12)
    ax.set_ylabel('g(y) (claim function)', fontsize=12)
    ax.set_title(r'Terminal Claim Function: ES vs VaR Constraint ($Y_0$ = 1.0)',
                 fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(OUT, 'fig_baseline_claim_function.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {os.path.basename(path)}")
    return fig


# ═══════════════════════════════════════════════════════════
# Fig 2: Present Value Function
# ═══════════════════════════════════════════════════════════

def plot_present_value():
    """Present value Ψ(F) for ES and VaR (cross-sectional)."""
    F_range = np.linspace(0.5, 1.3, 500)

    psi_es = np.full(len(F_range), np.nan)
    psi_var = np.full(len(F_range), np.nan)

    for i, F in enumerate(F_range):
        # ES
        res = safe_es_threshold(F)
        if res is not None:
            k_eps, c, binding = res
            if binding:
                psi_es[i] = ES.psi(F, k_eps, c)
            else:
                psi_es[i] = F  # non-binding → Ψ = F
        # VaR
        res = safe_var_threshold(F)
        if res is not None:
            k_alpha, binding = res
            if binding:
                psi_var[i] = VaR.psi(F, k_alpha)
            else:
                psi_var[i] = F

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(F_range, psi_es, 'r-', lw=2.5, label=r'$\Psi_{ES}$')
    ax.plot(F_range, psi_var, 'b--', lw=2.5, label=r'$\Psi_{VaR}$')
    ax.plot(F_range, F_range, ':', color='gray', lw=1.5,
            label=r'Unconstrained ($\Psi = F$)')

    ax.set_xlabel('F(t) (current funding ratio)', fontsize=12)
    ax.set_ylabel(r'$\Psi(t, F)$ (present value)', fontsize=12)
    ax.set_title('Present Value of Constrained Portfolio: ES vs VaR',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.3)

    plt.tight_layout()
    path = os.path.join(OUT, 'fig_baseline_present_value.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {os.path.basename(path)}")
    return fig


# ═══════════════════════════════════════════════════════════
# Fig 3: Adjustment Factor
# ═══════════════════════════════════════════════════════════

def plot_adjustment_factor():
    """Adjustment factor A(F) for ES and VaR (cross-sectional)."""
    F_range = np.linspace(0.5, 1.3, 500)

    A_es = np.full(len(F_range), np.nan)
    A_var = np.full(len(F_range), np.nan)

    for i, F in enumerate(F_range):
        try:
            A_es[i] = ES.cross_sectional_A(F)
        except Exception:
            pass
        try:
            A_var[i] = VaR.cross_sectional_A(F, alpha=0.1)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(F_range, A_es, 'r-', lw=2.5, label=r'ES ($\varepsilon$ = 0.05)')
    ax.plot(F_range, A_var, 'b--', lw=2.5, label=r'VaR ($\alpha$ = 0.1)')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.6, lw=1.5, label='A = 1 (Merton)')

    # Shade gambling region (A > 1 for VaR)
    gambling_mask = A_var > 1.0
    if np.any(gambling_mask):
        ax.fill_between(F_range, 1.0, A_var, where=gambling_mask,
                        color='red', alpha=0.12, label='Gambling region (VaR)')
        # Add annotation at peak of gambling region
        gambling_idx = np.where(gambling_mask)[0]
        if len(gambling_idx) > 0:
            peak_idx = gambling_idx[np.argmax(A_var[gambling_idx])]
            ax.annotate('Gambling\nRegion',
                        xy=(F_range[peak_idx], A_var[peak_idx]),
                        xytext=(F_range[peak_idx] + 0.08, A_var[peak_idx] + 0.15),
                        fontsize=10, color='darkred', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                        ha='center')

    ax.set_xlabel('F(t) (current funding ratio)', fontsize=12)
    ax.set_ylabel('Adjustment Factor A(F)', fontsize=12)
    ax.set_title('Portfolio Adjustment Factor: ES vs VaR Constraint',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.3)

    plt.tight_layout()
    path = os.path.join(OUT, 'fig_baseline_adjustment_factor.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {os.path.basename(path)}")
    return fig


# ═══════════════════════════════════════════════════════════
# Fig 4: Optimal Asset Allocation (1×3 subplots)
# ═══════════════════════════════════════════════════════════

def plot_allocation():
    """Optimal asset allocation: Stock, IIB, Total Risky (1×3 subplots)."""
    F_range = np.linspace(0.5, 1.3, 500)

    pi_S_es = np.full(len(F_range), np.nan)
    pi_I_es = np.full(len(F_range), np.nan)
    pi_S_var = np.full(len(F_range), np.nan)
    pi_I_var = np.full(len(F_range), np.nan)

    for i, F in enumerate(F_range):
        # ES
        res = safe_es_threshold(F)
        if res is not None:
            k_eps, c, binding = res
            if binding:
                ps, pi = ES.optimal_portfolio(F, k_eps, c)
            else:
                ps, pi = P.Pi_star[0], P.Pi_star[1]
            pi_S_es[i] = ps
            pi_I_es[i] = pi

        # VaR
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

    # Merton unconstrained
    merton_S = P.Pi_star[0]
    merton_I = P.Pi_star[1]
    merton_total = P.Pi_star.sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: Stock
    ax = axes[0]
    ax.plot(F_range, pi_S_es, 'r-', lw=2.5, label='ES')
    ax.plot(F_range, pi_S_var, 'b--', lw=2.5, label='VaR')
    ax.axhline(merton_S, color='gray', ls=':', alpha=0.6, lw=1.5,
               label=f'Merton ({merton_S:.2f})')
    ax.set_xlabel('F(t)', fontsize=12)
    ax.set_ylabel(r'$\pi_S$', fontsize=12)
    ax.set_title('Stock', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.3)

    # Subplot 2: IIB
    ax = axes[1]
    ax.plot(F_range, pi_I_es, 'r-', lw=2.5, label='ES')
    ax.plot(F_range, pi_I_var, 'b--', lw=2.5, label='VaR')
    ax.axhline(merton_I, color='gray', ls=':', alpha=0.6, lw=1.5,
               label=f'Merton ({merton_I:.2f})')
    ax.set_xlabel('F(t)', fontsize=12)
    ax.set_ylabel(r'$\pi_I$', fontsize=12)
    ax.set_title('Inflation-Indexed Bond', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.3)

    # Subplot 3: Total Risky
    ax = axes[2]
    ax.plot(F_range, total_es, 'r-', lw=2.5, label='ES')
    ax.plot(F_range, total_var, 'b--', lw=2.5, label='VaR')
    ax.axhline(merton_total, color='gray', ls=':', alpha=0.6, lw=1.5,
               label=f'Merton ({merton_total:.2f})')
    ax.set_xlabel('F(t)', fontsize=12)
    ax.set_ylabel(r'$\pi_S + \pi_I$', fontsize=12)
    ax.set_title('Total Risky', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.3)

    fig.suptitle('Optimal Asset Allocation: ES vs VaR Constraint', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_baseline_allocation.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {os.path.basename(path)}")
    return fig


# ═══════════════════════════════════════════════════════════
# Fig 5: Option Decomposition (1×2 subplots)
# ═══════════════════════════════════════════════════════════

def plot_option_decomposition():
    """Option portfolio decomposition for ES and VaR (1×2 subplots)."""
    F_range = np.linspace(0.5, 1.3, 500)
    tau = P.T

    # ES components: Put(F, k), -c·Put(F, k_eps), net = Ψ_ES - F
    es_put_k = np.full(len(F_range), np.nan)
    es_neg_c_put_ke = np.full(len(F_range), np.nan)
    es_net = np.full(len(F_range), np.nan)

    # VaR components: Put(F, k), -Put(F, k_alpha), -(k-k_alpha)·Digital(F, k_alpha), net
    var_put_k = np.full(len(F_range), np.nan)
    var_neg_put_ka = np.full(len(F_range), np.nan)
    var_neg_digital = np.full(len(F_range), np.nan)
    var_net = np.full(len(F_range), np.nan)

    for i, F in enumerate(F_range):
        # ES
        res = safe_es_threshold(F)
        if res is not None:
            k_eps, c, binding = res
            pk = bs_put(F, P.k, P.r_tilde, P.sigma_Y, tau)
            if binding:
                pke = bs_put(F, k_eps, P.r_tilde, P.sigma_Y, tau)
                es_put_k[i] = pk
                es_neg_c_put_ke[i] = -c * pke
                es_net[i] = pk - c * pke  # Ψ_ES - F
            else:
                es_put_k[i] = pk
                es_neg_c_put_ke[i] = -pk  # c=1, k_eps=k → cancels
                es_net[i] = 0.0

        # VaR
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
                var_net[i] = pk - pka - (P.k - k_alpha) * dka  # Ψ_VaR - F
            else:
                var_put_k[i] = pk
                var_neg_put_ka[i] = -pk
                var_neg_digital[i] = 0.0
                var_net[i] = 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: ES decomposition
    ax1.plot(F_range, es_put_k, 'g-', lw=2, label='Put(F, k)')
    ax1.plot(F_range, es_neg_c_put_ke, 'm--', lw=2,
             label=r'$-c \cdot$ Put(F, $k_\varepsilon$)')
    ax1.plot(F_range, es_net, 'k-', lw=2.5, label=r'Net ($\Psi_{ES} - F$)')
    ax1.axhline(0, color='gray', ls=':', alpha=0.4)
    ax1.set_xlabel('F(t)', fontsize=12)
    ax1.set_ylabel('Option value', fontsize=12)
    ax1.set_title('ES: Option Decomposition', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 1.3)

    # Right: VaR decomposition
    ax2.plot(F_range, var_put_k, 'g-', lw=2, label='Put(F, k)')
    ax2.plot(F_range, var_neg_put_ka, 'm--', lw=2,
             label=r'$-$Put(F, $k_\alpha$)')
    ax2.plot(F_range, var_neg_digital, 'c-.', lw=2,
             label=r'$-(k-k_\alpha) \cdot$ Digital($k_\alpha$)')
    ax2.plot(F_range, var_net, 'k-', lw=2.5, label=r'Net ($\Psi_{VaR} - F$)')
    ax2.axhline(0, color='gray', ls=':', alpha=0.4)
    ax2.set_xlabel('F(t)', fontsize=12)
    ax2.set_ylabel('Option value', fontsize=12)
    ax2.set_title('VaR: Option Decomposition', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, 1.3)

    fig.suptitle('Option Portfolio Decomposition: ES vs VaR', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_baseline_option_decomposition.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {os.path.basename(path)}")
    return fig


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT, exist_ok=True)

    print("=" * 55)
    print("  Baseline Figures")
    print("=" * 55)
    P.print_params()
    print()

    print("Fig 1: Terminal Claim Function...")
    plot_claim_function()

    print("Fig 2: Present Value Function...")
    plot_present_value()

    print("Fig 3: Adjustment Factor...")
    plot_adjustment_factor()

    print("Fig 4: Optimal Asset Allocation...")
    plot_allocation()

    print("Fig 5: Option Decomposition...")
    plot_option_decomposition()

    print(f"\nAll baseline figures saved to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
