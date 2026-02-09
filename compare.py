"""
VaR vs ES Comparison
=====================
Cross-sectional and time-series comparison of VaR and ES constrained strategies.
Generates publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt

from . import params as P
from . import es_model as ES
from . import var_model as VaR


# ═══════════════════════════════════════════════════════════
# Cross-sectional comparison
# ═══════════════════════════════════════════════════════════

def cross_sectional_table(y0_list=None, eps=None, alpha=None):
    """Print cross-sectional comparison table."""
    if y0_list is None:
        y0_list = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

    mt = P.Pi_star.sum() * 100   # Merton total %

    print(f"{'y0':>6} | {'VaR A':>7} | {'ES A':>7} | {'VaR%':>7} | {'ES%':>7}")
    print("-" * 48)
    for y0 in y0_list:
        av = VaR.cross_sectional_A(y0, alpha)
        ae = ES.cross_sectional_A(y0, eps)
        print(f"{y0:>6.2f} | {av:>7.4f} | {ae:>7.4f} | {av*mt:>6.1f}% | {ae*mt:>6.1f}%")


def plot_cross_sectional(y0_range=(0.05, 2.5), n_points=1000,
                         eps=None, alpha=None, save_path=None):
    """Plot cross-sectional A(y0) for VaR and ES.

    Shows adjustment factor (left) and total allocation % (right).
    """
    y0_vals = np.linspace(*y0_range, n_points)
    Av = [VaR.cross_sectional_A(y0, alpha) for y0 in y0_vals]
    Ae = [ES.cross_sectional_A(y0, eps) for y0 in y0_vals]

    _eps = eps if eps is not None else P.epsilon
    _alpha = alpha if alpha is not None else P.alpha
    mt = P.Pi_star.sum() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Adjustment factor
    ax1.plot(y0_vals, Av, 'b-', lw=2.5, label=f'VaR (α={_alpha})')
    ax1.plot(y0_vals, Ae, 'r-', lw=2.5, label=f'ES (ε={_eps})')
    ax1.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Merton (A=1)')
    ax1.axvline(P.k, color='green', ls=':', alpha=0.3)
    ax1.set_xlabel('y₀ (initial funding ratio)', fontsize=12)
    ax1.set_ylabel('Adjustment Factor A(y₀)', fontsize=12)
    ax1.set_title('Cross-Sectional Adjustment Factor', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(y0_range)

    # Right: Total allocation
    ax2.plot(y0_vals, [a * mt for a in Av], 'b-', lw=2.5, label='VaR')
    ax2.plot(y0_vals, [a * mt for a in Ae], 'r-', lw=2.5, label='ES')
    ax2.axhline(mt, color='gray', ls='--', alpha=0.5, label=f'Merton ({mt:.0f}%)')
    ax2.axvline(P.k, color='green', ls=':', alpha=0.3)
    ax2.set_xlabel('y₀ (initial funding ratio)', fontsize=12)
    ax2.set_ylabel('Total Risky Allocation (%)', fontsize=12)
    ax2.set_title('Cross-Sectional Total Allocation', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(y0_range)

    plt.suptitle(f'R={P.R}, r={P.r}, γ={P.GAMMA}, T={P.T}', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════
# Time-series comparison
# ═══════════════════════════════════════════════════════════

def plot_time_series(y0_fund=None, Y_range=(0.1, 2.5), n_points=1000,
                     eps=None, alpha=None, save_path=None):
    """Plot time-series A(Y) for a single fund.

    The fund starts at y0_fund, solves its threshold once,
    then A is plotted as Y varies over time.
    """
    if y0_fund is None:
        y0_fund = P.y0
    _eps = eps if eps is not None else P.epsilon
    _alpha = alpha if alpha is not None else P.alpha

    # Solve thresholds for this fund
    k_eps, c, es_bind = ES.solve_threshold(y0_fund, _eps)
    k_alpha, var_bind = VaR.solve_threshold(y0_fund, _alpha)

    Y_vals = np.linspace(*Y_range, n_points)
    Av = [VaR.adjustment_factor(Y, k_alpha) if var_bind else 1.0 for Y in Y_vals]
    Ae = [ES.adjustment_factor(Y, k_eps, c) if es_bind else 1.0 for Y in Y_vals]

    mt = P.Pi_star.sum() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Adjustment factor
    ax1.plot(Y_vals, Av, 'b-', lw=2.5, label=f'VaR (k_α={k_alpha:.3f})')
    ax1.plot(Y_vals, Ae, 'r-', lw=2.5, label=f'ES (k_ε={k_eps:.3f})')
    ax1.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax1.axvline(k_alpha, color='blue', ls=':', alpha=0.4, label=f'k_α={k_alpha:.3f}')
    ax1.axvline(k_eps, color='red', ls=':', alpha=0.4, label=f'k_ε={k_eps:.3f}')
    ax1.axvline(P.k, color='green', ls=':', alpha=0.3)
    ax1.set_xlabel('Y (current funding ratio)', fontsize=12)
    ax1.set_ylabel('A(Y)', fontsize=12)
    ax1.set_title(f'Time-Series: Fund starting at y₀={y0_fund}', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(Y_range)

    # Right: Total allocation
    ax2.plot(Y_vals, [a * mt for a in Av], 'b-', lw=2.5, label='VaR')
    ax2.plot(Y_vals, [a * mt for a in Ae], 'r-', lw=2.5, label='ES')
    ax2.axhline(mt, color='gray', ls='--', alpha=0.5, label=f'Merton ({mt:.0f}%)')
    ax2.set_xlabel('Y (current funding ratio)', fontsize=12)
    ax2.set_ylabel('Total Risky Allocation (%)', fontsize=12)
    ax2.set_title('Total Allocation over Time', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(Y_range)

    plt.suptitle(f'y₀={y0_fund}, α={_alpha}, ε={_eps}', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════
# Sensitivity analysis
# ═══════════════════════════════════════════════════════════

def plot_eps_sensitivity(eps_list=None, y0_range=(0.05, 2.5), n_points=1000,
                         save_path=None):
    """Plot ES cross-sectional A(y0) for different epsilon values,
    with VaR as reference."""
    if eps_list is None:
        eps_list = [0.01, 0.05, 0.10, 0.15]

    y0_vals = np.linspace(*y0_range, n_points)
    Av = [VaR.cross_sectional_A(y0) for y0 in y0_vals]

    colors = ['darkred', 'red', 'orangered', 'orange']

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(y0_vals, Av, 'b-', lw=2.5, label=f'VaR (α={P.alpha})')

    for eps, color in zip(eps_list, colors):
        Ae = [ES.cross_sectional_A(y0, eps) for y0 in y0_vals]
        ls = '-' if eps >= 0.10 else '--' if eps >= 0.05 else ':'
        ax.plot(y0_vals, Ae, color=color, ls=ls, lw=2, label=f'ES (ε={eps})')

    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax.axvline(P.k, color='green', ls=':', alpha=0.3)
    ax.set_xlabel('y₀ (initial funding ratio)', fontsize=12)
    ax.set_ylabel('Adjustment Factor A(y₀)', fontsize=12)
    ax.set_title('Effect of ES Budget (ε) on Adjustment Factor', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(y0_range)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    out = "/mnt/user-data/outputs"
    P.print_params()
    print()

    print("=== Cross-Sectional Table ===")
    cross_sectional_table()
    print()

    plot_cross_sectional(save_path=os.path.join(out, "cross_sectional.png"))
    plot_time_series(save_path=os.path.join(out, "time_series.png"))
    plot_eps_sensitivity(save_path=os.path.join(out, "eps_sensitivity.png"))

    print("\nAll figures generated.")
