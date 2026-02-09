"""
ES-Constrained LDI Model
==========================
Based on Kraft & Steffensen (2013) option-based approach.

Constraint:
    E^Q[e^{-r̃T}(k - F_T)^+] <= epsilon

Claim function:
    g_ES(y) = c·y      if y < k_eps       (partial protection)
              k         if k_eps <= y < k   (boost to target)
              y         if y >= k           (unconstrained)

    where c = k / k_eps > 1

Option decomposition:
    g(y) = y + Put(k) - c·Put(k_eps)

Present value:
    Psi_ES = y + Put(y, k) - c·Put(y, k_eps)

Binding condition:
    c · Put(y0, k_eps) = epsilon

Key property: A(y0) <= 1 always (no gambling incentive).
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from .bs_utils import bs_put, bs_d1
from . import params as P


# ═══════════════════════════════════════════════════════════
# Threshold solver
# ═══════════════════════════════════════════════════════════

def solve_threshold(y0, eps=None):
    """Solve for k_eps given initial funding ratio y0.

    Binding condition: c · Put(y0, k_eps) = eps
      where c = k / k_eps

    Returns:
        (k_eps, c, is_binding)
    """
    if eps is None:
        eps = P.epsilon

    # Check if constraint binds: Put(y0, k) > eps?
    put_k = bs_put(y0, P.k, P.r_tilde, P.sigma_Y, P.T)
    if put_k <= eps:
        return P.k, 1.0, False     # non-binding → Merton

    # Solve: (k/k_eps) · Put(y0, k_eps) = eps  on (0, k)
    def residual(ke):
        return (P.k / ke) * bs_put(y0, ke, P.r_tilde, P.sigma_Y, P.T) - eps

    k_eps = brentq(residual, 1e-12, P.k - 1e-12, xtol=1e-14)
    c = P.k / k_eps
    return k_eps, c, True


# ═══════════════════════════════════════════════════════════
# Present value & derivatives
# ═══════════════════════════════════════════════════════════

def psi(Y, k_eps, c, tau=None):
    """Present value of ES claim: Psi = Y + Put(Y,k) - c·Put(Y,k_eps)

    Args:
        Y: current funding ratio
        k_eps, c: threshold and multiplier from solve_threshold
        tau: time to maturity (default: T)
    """
    if tau is None:
        tau = P.T
    P_k  = bs_put(Y, P.k,  P.r_tilde, P.sigma_Y, tau)
    P_ke = bs_put(Y, k_eps, P.r_tilde, P.sigma_Y, tau)
    return Y + P_k - c * P_ke


def dpsi_dy(Y, k_eps, c, tau=None):
    """∂Psi/∂y = 1 - N(-d1(k)) + c·N(-d1(k_eps))

    = 1 + Delta_Put(k) - c·Delta_Put(k_eps)
    """
    if tau is None:
        tau = P.T
    d1_k  = bs_d1(Y, P.k,  P.r_tilde, P.sigma_Y, tau)
    d1_ke = bs_d1(Y, k_eps, P.r_tilde, P.sigma_Y, tau)
    return 1.0 - norm.cdf(-d1_k) + c * norm.cdf(-d1_ke)


# ═══════════════════════════════════════════════════════════
# Adjustment factor & optimal strategy
# ═══════════════════════════════════════════════════════════

def adjustment_factor(Y, k_eps, c, tau=None):
    """A = (Y / Psi) · (dPsi/dy)

    Multiply by Pi_star to get constrained portfolio:
        pi* = A · Pi_star
    """
    psi_val = psi(Y, k_eps, c, tau)
    dpsi_val = dpsi_dy(Y, k_eps, c, tau)
    return (Y / psi_val) * dpsi_val


def optimal_portfolio(Y, k_eps, c, tau=None):
    """pi*_S, pi*_I = A · Pi_star"""
    A = adjustment_factor(Y, k_eps, c, tau)
    return A * P.Pi_star[0], A * P.Pi_star[1]


# ═══════════════════════════════════════════════════════════
# Cross-sectional convenience
# ═══════════════════════════════════════════════════════════

def cross_sectional_A(y0, eps=None):
    """Adjustment factor for a fund starting at y0.

    Each fund solves its own threshold.
    """
    k_eps, c, binding = solve_threshold(y0, eps)
    if not binding:
        return 1.0
    return adjustment_factor(y0, k_eps, c)


# ═══════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    P.print_params()
    print()

    # Solve for default y0
    k_eps, c, binding = solve_threshold(P.y0)
    verify = c * bs_put(P.y0, k_eps, P.r_tilde, P.sigma_Y, P.T)

    print(f"ES Constraint (epsilon = {P.epsilon})")
    print(f"  k_eps   = {k_eps:.6f}")
    print(f"  c       = {c:.6f}")
    print(f"  binding = {binding}")
    print(f"  verify: c·Put(y0,k_eps) = {verify:.8f}")
    print()

    # Time-series: fixed threshold, varying Y
    print("Time-series allocation (fixed threshold from y0=1.0):")
    print(f"  {'Y':>6} | {'A':>7} | {'pi_S':>7} | {'pi_I':>7} | {'Total':>7}")
    print("  " + "-" * 46)
    for Y in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
        A = adjustment_factor(Y, k_eps, c)
        ps, pi = A * P.Pi_star[0], A * P.Pi_star[1]
        print(f"  {Y:>6.2f} | {A:>7.4f} | {ps:>6.1%} | {pi:>6.1%} | {ps+pi:>6.1%}")

    # Cross-sectional: each fund has own threshold
    print()
    print("Cross-sectional allocation (each y0 solves own threshold):")
    print(f"  {'y0':>6} | {'A':>7} | {'Total':>7} | {'k_eps':>8} | {'bind':>5}")
    print("  " + "-" * 46)
    for y0 in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
        ke, cc, b = solve_threshold(y0)
        A = cross_sectional_A(y0)
        print(f"  {y0:>6.2f} | {A:>7.4f} | {A*P.Pi_star.sum():>6.1%} | {ke:>8.4f} | {'Y' if b else 'N':>5}")
