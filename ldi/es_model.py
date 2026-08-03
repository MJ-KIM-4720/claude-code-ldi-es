"""
ES-Constrained LDI Model  (joint budget + constraint system)
=============================================================
Kraft & Steffensen (2013) option-based approach, solved as the *joint*
system that the fund actually faces.

Constraint:
    E^Q[e^{-r̃T}(k - F_T)^+] <= epsilon

Claim function (fixed at t=0):
    g_ES(y) = c·y      if y < k_eps       (partial linear protection)
              k        if k_eps <= y < k  (boost to target)
              y        if y >= k          (unconstrained)
    with c = k / k_eps > 1

Present value of the claim:
    Psi_ES(t,y) = y + Put(y,k) - c·Put(y,k_eps)          (tau = T-t)

────────────────────────────────────────────────────────────
THE JOINT SYSTEM  (this replaces the old single-equation solver)
────────────────────────────────────────────────────────────
The reference process start Y0 is NOT the funding ratio F0. The pair
(Y0, k_eps) is pinned down by two equations:

    (budget)   Psi_ES(0, Y0) = F0
    (binding)  (k/k_eps)·Put(Y0, k_eps) = epsilon

They decouple: substituting (binding) into (budget) gives

    (A)  Y0 + Put(0, Y0, k) = F0 + epsilon

whose LHS is strictly increasing in Y0 (derivative N(d1) > 0) with
infimum k·e^{-r̃T} as Y0 -> 0.  Hence:

  * Step 1: solve (A) for Y0.
  * Step 2: solve (binding) for k_eps at that fixed Y0.
            LHS is strictly increasing in k_eps since
            d/dK [Put(y,K)/K] = y·N(-d1)/K² > 0.

FEASIBILITY.  (A) has a root iff F0 + epsilon > k·e^{-r̃T}, i.e.

    epsilon > eps_min := max(k·e^{-r̃T} - F0, 0)

which is exactly the budget-implied floor on the Q-expected shortfall
(see params.eps_min). Below the floor the problem has NO solution — the
old baseline epsilon = 0.05 sat there.

SLACK.  If epsilon >= eps_M := Put(F0, k), the Merton claim already
satisfies the constraint: Y0 = F0, k_eps = k, c = 1, A ≡ 1.

Key structural property (wedge identity, exact):
    Psi - y·Psi_y = k·e^{-r̃τ}·[N(-d2(y,k)) - N(-d2(y,k_eps))] > 0
  =>  0 < A_ES = y·Psi_y / Psi < 1  everywhere: no gambling incentive.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from .bs_utils import bs_put, bs_d1, bs_d2
from . import params as P
from .params import eps_min, eps_merton, eps_band     # re-export


class InfeasibleError(ValueError):
    """Raised when the ES budget is below the budget-implied floor eps_min."""


# ═══════════════════════════════════════════════════════════
# Joint solver
# ═══════════════════════════════════════════════════════════

def solve_es(F0=None, eps=None, strict=True):
    """Solve the joint (budget, binding-ES) system for (Y0, k_eps).

    Args:
        F0:   initial funding ratio (budget). Default P.F0.
        eps:  ES budget. Default P.epsilon.
        strict: if False, an infeasible eps returns feasible=False instead
                of raising (useful for scanning sensitivity grids).

    Returns dict with keys:
        Y0, k_eps, c, eps_min, eps_M, feasible, binding, eps, F0
    """
    if F0 is None:
        F0 = P.F0
    if eps is None:
        eps = P.epsilon

    k, rt, sig, T = P.k, P.r_tilde, P.sigma_Y, P.T
    e_lo = P.eps_min(F0)
    e_hi = P.eps_merton(F0)

    out = dict(F0=F0, eps=eps, eps_min=e_lo, eps_M=e_hi)

    # ── Infeasible: no admissible strategy attains this budget ──
    if eps <= e_lo:
        if strict:
            raise InfeasibleError(
                f"eps = {eps:.6f} <= eps_min = {e_lo:.6f} "
                f"(= max(k·e^-r̃T - F0, 0) with k={k}, r̃={rt:.6f}, "
                f"T={T}, F0={F0}). No admissible strategy can attain a "
                f"Q-expected shortfall this low: the budget constraint "
                f"E^Q[e^-r̃T F_T] <= F0 forbids it."
            )
        out.update(Y0=np.nan, k_eps=np.nan, c=np.nan,
                   feasible=False, binding=True)
        return out

    # ── Slack: Merton already satisfies the constraint ──
    if eps >= e_hi:
        out.update(Y0=F0, k_eps=k, c=1.0, feasible=True, binding=False)
        return out

    # ── Step 1: budget-substituted equation (A) for Y0 ──
    def f_budget(y):
        return y + bs_put(y, k, rt, sig, T) - (F0 + eps)

    hi = F0 + eps                       # f(hi) = Put(hi,k) > 0
    lo = 1e-14 * hi
    while f_budget(lo) > 0 and lo > 1e-300:
        lo *= 1e-3
    Y0 = brentq(f_budget, lo, hi, xtol=1e-16, rtol=8.9e-16, maxiter=200)

    # ── Step 2: binding ES condition for k_eps at fixed Y0 ──
    def f_constraint(ke):
        return (k / ke) * bs_put(Y0, ke, rt, sig, T) - eps

    hi_k = k * (1.0 - 1e-14)            # f(k) = Put(Y0,k) - eps > 0 since Y0 < F0
    lo_k = min(1e-10, Y0 * 1e-6)
    while f_constraint(lo_k) > 0 and lo_k > 1e-300:
        lo_k *= 1e-3
    k_eps = brentq(f_constraint, lo_k, hi_k, xtol=1e-16, rtol=8.9e-16,
                   maxiter=200)

    out.update(Y0=Y0, k_eps=k_eps, c=k / k_eps, feasible=True, binding=True)
    return out


def solve_threshold(F0=None, eps=None):
    """Convenience wrapper: (Y0, k_eps, c, binding)."""
    s = solve_es(F0, eps)
    return s['Y0'], s['k_eps'], s['c'], s['binding']


# ═══════════════════════════════════════════════════════════
# Claim, present value & derivatives
# ═══════════════════════════════════════════════════════════

def claim(y, k_eps, c):
    """Terminal payoff g_ES(y) = min(c·y, k) for y < k, else y."""
    y = np.asarray(y, dtype=float)
    return np.where(y >= P.k, y, np.minimum(c * y, P.k))


def psi(Y, k_eps, c, tau=None):
    """Psi_ES(t,y) = y + Put(y,k) - c·Put(y,k_eps),  tau = T - t."""
    if tau is None:
        tau = P.T
    Y = np.asarray(Y, dtype=float)
    if np.isscalar(tau) and tau <= 0:
        return claim(Y, k_eps, c)
    P_k = bs_put(Y, P.k, P.r_tilde, P.sigma_Y, tau)
    P_ke = bs_put(Y, k_eps, P.r_tilde, P.sigma_Y, tau)
    return Y + P_k - c * P_ke


def dpsi_dy(Y, k_eps, c, tau=None):
    """dPsi/dy = 1 - N(-d1(k)) + c·N(-d1(k_eps)) = N(d1(k)) + c·N(-d1(k_eps))."""
    if tau is None:
        tau = P.T
    Y = np.asarray(Y, dtype=float)
    d1_k = bs_d1(Y, P.k, P.r_tilde, P.sigma_Y, tau)
    d1_ke = bs_d1(Y, k_eps, P.r_tilde, P.sigma_Y, tau)
    return norm.cdf(d1_k) + c * norm.cdf(-d1_ke)


def wedge(Y, k_eps, c, tau=None):
    """Psi - y·Psi_y = k·e^{-r̃τ}·[N(-d2(y,k)) - N(-d2(y,k_eps))] >= 0.

    Exact identity (uses c·k_eps = k). Positivity of the wedge is what
    forces A_ES < 1.
    """
    if tau is None:
        tau = P.T
    Y = np.asarray(Y, dtype=float)
    d2_k = bs_d2(Y, P.k, P.r_tilde, P.sigma_Y, tau)
    d2_ke = bs_d2(Y, k_eps, P.r_tilde, P.sigma_Y, tau)
    return P.k * np.exp(-P.r_tilde * tau) * (norm.cdf(-d2_k) - norm.cdf(-d2_ke))


# ═══════════════════════════════════════════════════════════
# Adjustment factor & optimal strategy  (FIXED CLAIM)
# ═══════════════════════════════════════════════════════════

def adjustment_factor(Y, k_eps, c, tau=None):
    """A_ES(t,y) = y·Psi_y / Psi, evaluated as 1 - wedge/Psi.

    This is the delta-hedging exposure of a claim FIXED at t=0. Do NOT
    re-solve k_eps as y moves — that would price a different claim at
    every state.

    The wedge form is algebraically identical to y·Psi_y/Psi but better
    conditioned: the direct form differences two nearly equal quantities
    in the tails and can round to 1 + O(1e-15), whereas the closed-form
    wedge k·e^{-r̃τ}[N(-d2(k)) - N(-d2(k_eps))] is computed without
    cancellation, so A <= 1 holds to machine precision rather than by
    clamping.
    """
    return 1.0 - wedge(Y, k_eps, c, tau) / psi(Y, k_eps, c, tau)


def optimal_portfolio(Y, k_eps, c, tau=None):
    """pi*_S, pi*_I = A · Pi_star."""
    A = adjustment_factor(Y, k_eps, c, tau)
    return A * P.Pi_star[0], A * P.Pi_star[1]


# ═══════════════════════════════════════════════════════════
# Cross-sectional convenience (t=0, one fund per F0)
# ═══════════════════════════════════════════════════════════

def cross_sectional_A(F0, eps=None, strict=True):
    """Initial adjustment factor A_ES(0, Y0) for a fund with budget F0.

    Each fund solves its own joint system. Returns np.nan when the fund's
    (F0, eps) pair is infeasible and strict=False.
    """
    s = solve_es(F0, eps, strict=strict)
    if not s['feasible']:
        return np.nan
    if not s['binding']:
        return 1.0
    return float(adjustment_factor(s['Y0'], s['k_eps'], s['c'], P.T))


# ═══════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    P.print_params()
    print()

    s = solve_es()
    print(f"ES joint solution (F0={s['F0']}, eps={s['eps']})")
    for key in ('eps_min', 'eps_M', 'Y0', 'k_eps', 'c'):
        print(f"  {key:8s} = {s[key]:.6f}")
    print(f"  budget residual     = "
          f"{psi(s['Y0'], s['k_eps'], s['c']) - s['F0']:+.3e}")
    print(f"  constraint residual = "
          f"{s['c'] * bs_put(s['Y0'], s['k_eps'], P.r_tilde, P.sigma_Y, P.T) - s['eps']:+.3e}")
    print()

    print("Fixed-claim exposure A_ES(t,y) (claim fixed at t=0):")
    print(f"  {'y':>6} |" + "".join(f" {'t='+str(t):>9}" for t in (0, 2.5, 5, 7.5)))
    print("  " + "-" * 48)
    for y in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5]:
        row = "".join(f" {float(adjustment_factor(y, s['k_eps'], s['c'], P.T - t)):>9.4f}"
                      for t in (0, 2.5, 5, 7.5))
        print(f"  {y:>6.2f} |{row}")
