"""
VaR-Constrained LDI Model
===========================
Based on Jo et al. (2025) and Kraft & Steffensen (2013).

Constraint:
    P(F_T < k) <= alpha

Claim function:
    g_VaR(y) = y      if y < k_alpha       (protection abandoned)
               k      if k_alpha <= y < k   (boost to target)
               y      if y >= k             (unconstrained)

Option decomposition:
    g(y) = y + (k-y)^+ - (k_alpha-y)^+ - (k-k_alpha)·1_{y<k_alpha}
         = y + Put(k) - Put(k_alpha) - (k-k_alpha)·Digital(k_alpha)

Threshold (under P-measure):
    P^P(Y_T < k_alpha) = alpha
    → k_alpha = y0 · exp(m_P·T + sigma_Y·sqrt(T)·Phi^{-1}(alpha))

Key property: A(y0) > 1 possible for underfunded (gambling incentive).
"""

import numpy as np
from scipy.stats import norm

from .bs_utils import (bs_put, bs_d1, bs_d2,
                       bs_digital_put, bs_digital_put_delta)
from . import params as P


# ═══════════════════════════════════════════════════════════
# Threshold solver
# ═══════════════════════════════════════════════════════════

def solve_threshold(y0, alpha=None):
    """Solve for k_alpha given initial funding ratio y0.

    Under P-measure:
        k_alpha = y0 · exp(m_P·T + sigma_Y·sqrt(T)·Phi^{-1}(alpha))

    Returns:
        (k_alpha, is_binding)
    """
    if alpha is None:
        alpha = P.alpha

    # Check binding: P(Y_T < k | Y_0 = y0) > alpha?
    prob_under_k = norm.cdf(
        (np.log(P.k / y0) - P.m_P * P.T) / (P.sigma_Y * np.sqrt(P.T))
    )
    if prob_under_k <= alpha:
        return P.k, False       # non-binding → Merton

    # k_alpha from quantile of lognormal
    k_alpha = y0 * np.exp(
        P.m_P * P.T + P.sigma_Y * np.sqrt(P.T) * norm.ppf(alpha)
    )
    return k_alpha, True


# ═══════════════════════════════════════════════════════════
# Present value & derivatives
# ═══════════════════════════════════════════════════════════

def psi(Y, k_alpha, tau=None):
    """Present value of VaR claim:
    Psi = Y + Put(k) - Put(k_alpha) - (k-k_alpha)·Digital(k_alpha)

    Args:
        Y: current funding ratio
        k_alpha: VaR threshold from solve_threshold
        tau: time to maturity (default: T)
    """
    if tau is None:
        tau = P.T
    P_k  = bs_put(Y, P.k,     P.r_tilde, P.sigma_Y, tau)
    P_ka = bs_put(Y, k_alpha,  P.r_tilde, P.sigma_Y, tau)
    D_ka = bs_digital_put(Y, k_alpha, P.r_tilde, P.sigma_Y, tau)
    return Y + P_k - P_ka - (P.k - k_alpha) * D_ka


def dpsi_dy(Y, k_alpha, tau=None):
    """∂Psi/∂y = 1 + ∂Put(k)/∂y - ∂Put(k_alpha)/∂y - (k-k_alpha)·∂Digital(k_alpha)/∂y

    = 1 - N(-d1(k)) + N(-d1(k_alpha)) - (k-k_alpha)·∂Digital/∂y
    """
    if tau is None:
        tau = P.T
    d1_k  = bs_d1(Y, P.k,    P.r_tilde, P.sigma_Y, tau)
    d1_ka = bs_d1(Y, k_alpha, P.r_tilde, P.sigma_Y, tau)
    dD_ka = bs_digital_put_delta(Y, k_alpha, P.r_tilde, P.sigma_Y, tau)

    return 1.0 - norm.cdf(-d1_k) + norm.cdf(-d1_ka) - (P.k - k_alpha) * dD_ka


# ═══════════════════════════════════════════════════════════
# Adjustment factor & optimal strategy
# ═══════════════════════════════════════════════════════════

def adjustment_factor(Y, k_alpha, tau=None):
    """A = (Y / Psi) · (dPsi/dy)"""
    psi_val = psi(Y, k_alpha, tau)
    if psi_val <= 0:
        return 1.0
    dpsi_val = dpsi_dy(Y, k_alpha, tau)
    return (Y / psi_val) * dpsi_val


def optimal_portfolio(Y, k_alpha, tau=None):
    """pi*_S, pi*_I = A · Pi_star"""
    A = adjustment_factor(Y, k_alpha, tau)
    return A * P.Pi_star[0], A * P.Pi_star[1]


# ═══════════════════════════════════════════════════════════
# Cross-sectional convenience
# ═══════════════════════════════════════════════════════════

def cross_sectional_A(y0, alpha=None):
    """Adjustment factor for a fund starting at y0.

    Each fund solves its own threshold.
    """
    k_alpha, binding = solve_threshold(y0, alpha)
    if not binding:
        return 1.0
    return adjustment_factor(y0, k_alpha)


# ═══════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    P.print_params()
    print()

    # Solve for default y0
    k_alpha, binding = solve_threshold(P.y0)
    prob = norm.cdf(
        (np.log(P.k / P.y0) - P.m_P * P.T) / (P.sigma_Y * np.sqrt(P.T))
    )

    print(f"VaR Constraint (alpha = {P.alpha})")
    print(f"  k_alpha    = {k_alpha:.6f}")
    print(f"  binding    = {binding}")
    print(f"  P(Y_T < k) = {prob:.4f}  (Merton shortfall prob)")
    print()

    # Time-series
    print("Time-series allocation (fixed threshold from y0=1.0):")
    print(f"  {'Y':>6} | {'A':>7} | {'pi_S':>7} | {'pi_I':>7} | {'Total':>7}")
    print("  " + "-" * 46)
    for Y in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
        A = adjustment_factor(Y, k_alpha)
        ps, pi = A * P.Pi_star[0], A * P.Pi_star[1]
        print(f"  {Y:>6.2f} | {A:>7.4f} | {ps:>6.1%} | {pi:>6.1%} | {ps+pi:>6.1%}")

    # Cross-sectional
    print()
    print("Cross-sectional allocation (each y0 solves own threshold):")
    print(f"  {'y0':>6} | {'A':>7} | {'Total':>7} | {'k_alpha':>8} | {'bind':>5}")
    print("  " + "-" * 46)
    for y0 in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
        ka, b = solve_threshold(y0)
        A = cross_sectional_A(y0)
        print(f"  {y0:>6.2f} | {A:>7.4f} | {A*P.Pi_star.sum():>6.1%} | {ka:>8.4f} | {'Y' if b else 'N':>5}")
