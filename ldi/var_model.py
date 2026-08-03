"""
VaR-Constrained LDI Model  (joint budget + constraint system)
==============================================================
Jo, Kim & Jang (2025) / Kraft & Steffensen (2013).

Constraint:
    P(F_T < k) <= alpha

Claim function (fixed at t=0):
    g_VaR(y) = y      if y < k_alpha       (protection abandoned)
               k      if k_alpha <= y < k  (boost to target)
               y      if y >= k            (unconstrained)

Present value:
    Psi_VaR(t,y) = y + Put(y,k) - Put(y,k_alpha)
                     - (k - k_alpha)·Digital(y, k_alpha)

────────────────────────────────────────────────────────────
THE JOINT SYSTEM
────────────────────────────────────────────────────────────
Exactly as in the ES model, g_VaR(y) >= y, so the claim costs more than
the reference process and Y0 != F0. The pair (Y0, k_alpha) solves

    (budget)   Psi_VaR(0, Y0) = F0
    (binding)  P(Y_T < k_alpha) = alpha
               <=> k_alpha = Y0·exp{m_P·T + sigma_Y·sqrt(T)·Phi^{-1}(alpha)}
                          =: lambda(alpha)·Y0                (P-measure)

Since k_alpha is proportional to Y0, substitute it into the budget and
solve a single 1-D root problem in Y0. By homogeneity of the put and the
scale-invariance of the digital,

    Psi_VaR(0,Y0) = Y0·[1 + N(-d1(1,lambda))] + Put(Y0,k) - k·D(lambda)

whose derivative in Y0 is 1 + N(-d1(1,lambda)) + N(d1(Y0,k)) > 0, so the
root is unique.

FEASIBILITY (quantile hedging).  The cheapest way to satisfy a
probability constraint is to fund k on the Q-cheapest states carrying
P-probability 1-alpha. Since dQ/dP is decreasing in Y_T, those are the
states {Y_T >= k_alpha}, giving the floor

    C_VaR(alpha) = k·e^{-r̃T}·N(d2(1, lambda(alpha)))

(independent of Y0). The constraint is attainable iff F0 > C_VaR(alpha).
Unlike the ES floor eps_min, this floor is far below F0 at the baseline
— that asymmetry is itself a result.

Key property: A_VaR > 1 is possible for underfunded states — the digital
term makes the exposure blow up near k_alpha (gambling incentive).
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from .bs_utils import (bs_put, bs_d1, bs_d2,
                       bs_digital_put, bs_digital_put_delta)
from . import params as P


class InfeasibleError(ValueError):
    """Raised when alpha is below the quantile-hedging floor for this F0."""


# ═══════════════════════════════════════════════════════════
# Threshold ratio & feasibility floor
# ═══════════════════════════════════════════════════════════

def lambda_ratio(alpha=None):
    """k_alpha / Y0 = exp{m_P·T + sigma_Y·sqrt(T)·Phi^{-1}(alpha)} (P-measure)."""
    if alpha is None:
        alpha = P.alpha
    return float(np.exp(P.m_P * P.T + P.sigma_Y * np.sqrt(P.T) * norm.ppf(alpha)))


def quantile_hedge_cost(alpha=None):
    """Minimum initial capital funding P(F_T >= k) >= 1 - alpha.

    C_VaR(alpha) = k·e^{-r̃T}·Q(Y_T >= k_alpha) = k·e^{-r̃T}·N(d2(1, lambda)).
    """
    lam = lambda_ratio(alpha)
    return float(P.k * np.exp(-P.r_tilde * P.T)
                 * norm.cdf(bs_d2(1.0, lam, P.r_tilde, P.sigma_Y, P.T)))


def alpha_min(F0=None):
    """Smallest attainable alpha at budget F0 (C_VaR(alpha_min) = F0).

    Returns 0.0 if even alpha -> 0 is affordable.
    """
    if F0 is None:
        F0 = P.F0
    if quantile_hedge_cost(1e-12) <= F0:
        return 0.0
    return float(brentq(lambda a: quantile_hedge_cost(a) - F0,
                        1e-12, 1 - 1e-12, xtol=1e-14, rtol=8.9e-16))


def is_feasible(F0=None, alpha=None):
    if F0 is None:
        F0 = P.F0
    return quantile_hedge_cost(alpha) < F0


# ═══════════════════════════════════════════════════════════
# Joint solver
# ═══════════════════════════════════════════════════════════

def solve_var(F0=None, alpha=None, strict=True):
    """Solve the joint (budget, binding-VaR) system for (Y0, k_alpha).

    Returns dict with keys:
        Y0, k_alpha, cost_min, feasible, binding, alpha, F0
    """
    if F0 is None:
        F0 = P.F0
    if alpha is None:
        alpha = P.alpha

    lam = lambda_ratio(alpha)
    cost_min = quantile_hedge_cost(alpha)
    out = dict(F0=F0, alpha=alpha, cost_min=cost_min, lam=lam)

    if cost_min >= F0:
        if strict:
            raise InfeasibleError(
                f"alpha = {alpha:.6f} needs at least C_VaR = {cost_min:.6f} "
                f"> F0 = {F0:.6f} (quantile-hedging floor)."
            )
        out.update(Y0=np.nan, k_alpha=np.nan, feasible=False, binding=True)
        return out

    # Non-binding: does Merton (Y0 = F0) already satisfy P(Y_T < k) <= alpha?
    prob_merton = norm.cdf((np.log(P.k / F0) - P.m_P * P.T)
                           / (P.sigma_Y * np.sqrt(P.T)))
    if prob_merton <= alpha:
        out.update(Y0=F0, k_alpha=lam * F0, feasible=True, binding=False)
        return out

    # Binding: solve Psi_VaR(0, Y0; k_alpha = lam·Y0) = F0
    def f_budget(y):
        return float(psi(y, lam * y, P.T)) - F0

    lo = 1e-12 * F0
    while f_budget(lo) > 0 and lo > 1e-300:
        lo *= 1e-3
    Y0 = brentq(f_budget, lo, F0, xtol=1e-16, rtol=8.9e-16, maxiter=200)

    out.update(Y0=Y0, k_alpha=lam * Y0, feasible=True, binding=True)
    return out


def solve_threshold(F0=None, alpha=None):
    """Convenience wrapper: (Y0, k_alpha, binding)."""
    s = solve_var(F0, alpha)
    return s['Y0'], s['k_alpha'], s['binding']


# ═══════════════════════════════════════════════════════════
# Claim, present value & derivatives
# ═══════════════════════════════════════════════════════════

def claim(y, k_alpha):
    """g_VaR(y) = y if y < k_alpha; k if k_alpha <= y < k; y if y >= k."""
    y = np.asarray(y, dtype=float)
    return np.where((y >= k_alpha) & (y < P.k), P.k, y)


def psi(Y, k_alpha, tau=None):
    """Psi_VaR(t,y) = y + Put(y,k) - Put(y,k_a) - (k-k_a)·Digital(y,k_a)."""
    if tau is None:
        tau = P.T
    Y = np.asarray(Y, dtype=float)
    if np.isscalar(tau) and tau <= 0:
        return claim(Y, k_alpha)
    P_k = bs_put(Y, P.k, P.r_tilde, P.sigma_Y, tau)
    P_ka = bs_put(Y, k_alpha, P.r_tilde, P.sigma_Y, tau)
    D_ka = bs_digital_put(Y, k_alpha, P.r_tilde, P.sigma_Y, tau)
    return Y + P_k - P_ka - (P.k - k_alpha) * D_ka


def dpsi_dy(Y, k_alpha, tau=None):
    """dPsi/dy = N(d1(k)) + N(-d1(k_a)) - (k-k_a)·dDigital/dy."""
    if tau is None:
        tau = P.T
    Y = np.asarray(Y, dtype=float)
    d1_k = bs_d1(Y, P.k, P.r_tilde, P.sigma_Y, tau)
    d1_ka = bs_d1(Y, k_alpha, P.r_tilde, P.sigma_Y, tau)
    dD_ka = bs_digital_put_delta(Y, k_alpha, P.r_tilde, P.sigma_Y, tau)
    return norm.cdf(d1_k) + norm.cdf(-d1_ka) - (P.k - k_alpha) * dD_ka


# ═══════════════════════════════════════════════════════════
# Adjustment factor & optimal strategy  (FIXED CLAIM)
# ═══════════════════════════════════════════════════════════

def adjustment_factor(Y, k_alpha, tau=None):
    """A_VaR(t,y) = y·Psi_y / Psi  for the claim fixed at t=0."""
    Y = np.asarray(Y, dtype=float)
    psi_val = psi(Y, k_alpha, tau)
    return Y * dpsi_dy(Y, k_alpha, tau) / psi_val


def optimal_portfolio(Y, k_alpha, tau=None):
    """pi*_S, pi*_I = A · Pi_star."""
    A = adjustment_factor(Y, k_alpha, tau)
    return A * P.Pi_star[0], A * P.Pi_star[1]


# ═══════════════════════════════════════════════════════════
# Cross-sectional convenience (t=0, one fund per F0)
# ═══════════════════════════════════════════════════════════

def cross_sectional_A(F0, alpha=None, strict=True):
    """Initial adjustment factor A_VaR(0, Y0) for a fund with budget F0."""
    s = solve_var(F0, alpha, strict=strict)
    if not s['feasible']:
        return np.nan
    if not s['binding']:
        return 1.0
    return float(adjustment_factor(s['Y0'], s['k_alpha'], P.T))


# ═══════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    P.print_params()
    print()

    s = solve_var()
    print(f"VaR joint solution (F0={s['F0']}, alpha={s['alpha']})")
    for key in ('lam', 'cost_min', 'Y0', 'k_alpha'):
        print(f"  {key:9s} = {s[key]:.6f}")
    print(f"  feasible  = {s['feasible']}  (C_VaR = {s['cost_min']:.6f} "
          f"{'<' if s['feasible'] else '>='} F0 = {s['F0']})")
    print(f"  alpha_min = {alpha_min():.3e}")
    print(f"  budget residual = {float(psi(s['Y0'], s['k_alpha'])) - s['F0']:+.3e}")
    print()

    print("Fixed-claim exposure A_VaR(t,y):")
    print(f"  {'y':>6} |" + "".join(f" {'t='+str(t):>9}" for t in (0, 2.5, 5, 7.5)))
    print("  " + "-" * 48)
    for y in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5]:
        row = "".join(f" {float(adjustment_factor(y, s['k_alpha'], P.T - t)):>9.4f}"
                      for t in (0, 2.5, 5, 7.5))
        print(f"  {y:>6.2f} |{row}")
