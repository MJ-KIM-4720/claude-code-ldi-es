"""
Model-Implied Exact Statistics (closed form, no simulation)
============================================================
Every terminal statistic in Table 2 is available in closed form: the
reference process is lognormal under P and each claim is piecewise
(linear | flat | linear) in it, so all moments, probabilities, quantiles
and tail means reduce to truncated-lognormal building blocks.

Under P,  ln Y_T ~ N(m, s²)  with

    m = ln(Y_0) + (r̃ + gamma·sigma_Y² - sigma_Y²/2)·T = ln(Y_0) + m_P·T
    s = sigma_Y·sqrt(T)

Building blocks (Lambda = lower partial moment, Upsilon = upper):

    Lambda(a,K) = E[Y_T^a·1{Y_T <  K}] = exp(am + a²s²/2)·Phi((lnK - m - as²)/s)
    Upsilon(a,K) = E[Y_T^a·1{Y_T >= K}] = exp(am + a²s²/2)·(1 - Phi(...))
    P(Y_T < K)  = Phi((lnK - m)/s)

────────────────────────────────────────────────────────────
ONE CLAIM FORM COVERS ALL THREE STRATEGIES
────────────────────────────────────────────────────────────
    g(y) = c·y   if y <  k_low
           k     if k_low <= y < k
           y     if y >= k

    ES     : c = k/k_eps > 1, k_low = k_eps      (partial protection)
    VaR    : c = 1,           k_low = k_alpha    (protection abandoned)
    Merton : c = 1,           k_low = k          (middle region empty
                                                  -> g is the identity)

Note c·k_low = k for ES and c·k_low = k_low <= k otherwise, so in every
case the lower region is exactly the shortfall region {F_T < k} and
P(F_T < k) = P(Y_T < k_low).

No quadrature, no Monte Carlo — scipy.stats.norm only.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from . import params as P
from . import es_model as ES
from . import var_model as VaR


# ═══════════════════════════════════════════════════════════
# Lognormal building blocks
# ═══════════════════════════════════════════════════════════

def log_moments(Y0, T=None):
    """(m, s) of ln Y_T under the P-measure."""
    if T is None:
        T = P.T
    m = np.log(Y0) + P.m_P * T
    s = P.sigma_Y * np.sqrt(T)
    return float(m), float(s)


def Lambda(a, K, m, s):
    """E[Y_T^a · 1{Y_T < K}] (lower partial moment)."""
    if K <= 0:
        return 0.0
    return float(np.exp(a * m + a ** 2 * s ** 2 / 2)
                 * norm.cdf((np.log(K) - m - a * s ** 2) / s))


def Upsilon(a, K, m, s):
    """E[Y_T^a · 1{Y_T >= K}] (upper partial moment)."""
    if K <= 0:
        return float(np.exp(a * m + a ** 2 * s ** 2 / 2))
    return float(np.exp(a * m + a ** 2 * s ** 2 / 2)
                 * (1.0 - norm.cdf((np.log(K) - m - a * s ** 2) / s)))


def prob_below(K, m, s):
    """P(Y_T < K)."""
    if K <= 0:
        return 0.0
    return float(norm.cdf((np.log(K) - m) / s))


def y_quantile(q, m, s):
    """q-quantile of Y_T."""
    return float(np.exp(m + s * norm.ppf(q)))


# ═══════════════════════════════════════════════════════════
# Generic piecewise claim  g(y) = c·y | k | y
# ═══════════════════════════════════════════════════════════

def _moment(a, c, k_low, m, s, k=None):
    """E[g(Y_T)^a] for the piecewise claim.

    = c^a·Lambda(a,k_low) + k^a·[P(Y<k) - P(Y<k_low)] + Upsilon(a,k)
    """
    if k is None:
        k = P.k
    lower = c ** a * Lambda(a, k_low, m, s)
    middle = k ** a * (prob_below(k, m, s) - prob_below(k_low, m, s))
    upper = Upsilon(a, k, m, s)
    return lower + middle + upper


def _partial_mean(q, c, k_low, m, s, k=None):
    """E[g(Y_T)·1{Y_T < y_q}] where y_q is the q-quantile of Y_T.

    Branches on which segment of the claim the q-quantile falls in.
    """
    if k is None:
        k = P.k
    y_q = y_quantile(q, m, s)
    p1 = prob_below(k_low, m, s)
    p2 = prob_below(k, m, s)

    if q <= p1:                                   # still inside c·y
        return c * Lambda(1, y_q, m, s)
    acc = c * Lambda(1, k_low, m, s)              # whole lower segment
    if q <= p2:                                   # inside the flat part
        return acc + k * (q - p1)
    acc += k * (p2 - p1)                          # whole flat part
    return acc + (Lambda(1, y_q, m, s) - Lambda(1, k, m, s))


def claim_stats(Y0, c=1.0, k_low=None, k=None, gamma=None, T=None,
                tail_q=0.05, label=None):
    """All exact terminal statistics for one piecewise claim.

    Args:
        Y0:    reference-process start from the joint solver (NOT F0)
        c:     multiplier on the protected segment
        k_low: upper end of the protected segment (k_eps / k_alpha / k)
        tail_q: tail level for the quantile and bottom-q mean (0.05)

    Returns dict of point statistics (no standard errors — these are exact).
    """
    if k is None:
        k = P.k
    if gamma is None:
        gamma = P.GAMMA
    if k_low is None:
        k_low = k
    m, s = log_moments(Y0, T)

    p1 = prob_below(k_low, m, s)                  # = P(F_T < k)
    p2 = prob_below(k, m, s)

    mean = _moment(1, c, k_low, m, s, k)
    second = _moment(2, c, k_low, m, s, k)
    var = max(second - mean ** 2, 0.0)

    # shortfall statistics live entirely on the lower segment
    exp_sf = k * p1 - c * Lambda(1, k_low, m, s)
    cond_sf = exp_sf / p1 if p1 > 0 else 0.0

    # CRRA certainty equivalent
    omg = 1.0 - gamma
    e_util = _moment(omg, c, k_low, m, s, k)
    ce = float(e_util ** (1.0 / omg))

    # tail quantile: g is non-decreasing, so Q_q(F) = g(Q_q(Y))
    y_q = y_quantile(tail_q, m, s)
    if tail_q < p1:
        q_tail = c * y_q
    elif tail_q < p2:
        q_tail = k
    else:
        q_tail = y_q
    bottom_mean = _partial_mean(tail_q, c, k_low, m, s, k) / tail_q

    return {
        'label': label,
        'Y0': float(Y0), 'c': float(c), 'k_low': float(k_low),
        'mean': float(mean),
        'std': float(np.sqrt(var)),
        'prob_shortfall': float(p1),
        'exp_shortfall': float(exp_sf),
        'cond_shortfall': float(cond_sf),
        'q5': float(q_tail),
        'bottom5_mean': float(bottom_mean),
        'ce': ce,
    }


# ═══════════════════════════════════════════════════════════
# Strategy wrappers — parameters come from the joint solvers
# ═══════════════════════════════════════════════════════════

def merton_stats(F0=None, **kw):
    if F0 is None:
        F0 = P.F0
    return claim_stats(F0, c=1.0, k_low=P.k, label='Merton', **kw)


def es_stats(F0=None, eps=None, sol=None, **kw):
    """Exact statistics for the ES strategy (claim from ES.solve_es)."""
    if sol is None:
        sol = ES.solve_es(F0, eps)
    return claim_stats(sol['Y0'], c=sol['c'], k_low=sol['k_eps'],
                       label=f"ES (eps={sol['eps']:.4g})", **kw)


def var_stats(F0=None, alpha=None, sol=None, label=None, **kw):
    """Exact statistics for the VaR strategy (claim from VaR.solve_var)."""
    if sol is None:
        sol = VaR.solve_var(F0, alpha)
    return claim_stats(sol['Y0'], c=1.0, k_low=sol['k_alpha'],
                       label=label or f"VaR (alpha={sol['alpha']:.4g})", **kw)


def ce_loss_pct(stats, ce_merton):
    """Certainty-equivalent loss relative to the unconstrained benchmark."""
    return 100.0 * (ce_merton - stats['ce']) / ce_merton


# ═══════════════════════════════════════════════════════════
# Exact equal-CE calibration  (replaces the MC/seed-dependent search)
# ═══════════════════════════════════════════════════════════

def ce_var(alpha, F0=None):
    """Exact certainty equivalent of the VaR strategy at level alpha."""
    return var_stats(F0, alpha)['ce']


def match_alpha_equal_ce(F0=None, eps=None, bracket=(0.05, 0.10), tol=1e-12):
    """alpha such that the exact CE of the VaR strategy equals the ES one.

    Deterministic: no seed, no sampling error. The MC-based counterpart in
    simulate.py is retained for the histogram figures but is no longer the
    number reported in the paper.

    tol defaults to 1e-12 rather than the 1e-5 the brief allows: the extra
    precision is free here and 1e-5 in alpha is not enough to pin the fifth
    decimal of the reported level.
    """
    if F0 is None:
        F0 = P.F0
    ce_target = es_stats(F0, eps)['ce']
    f = lambda a: ce_var(a, F0) - ce_target

    lo, hi = bracket
    a_floor = alpha_min(F0)
    for _ in range(8):                       # widen until the root is bracketed
        lo = max(lo, a_floor * (1 + 1e-9) + 1e-12)
        if f(lo) * f(hi) <= 0:
            break
        lo = max(a_floor * (1 + 1e-9) + 1e-12, lo / 2)
        hi = min(0.99, hi * 1.5)
    else:
        return dict(alpha=np.nan, ce_target=ce_target, bracketed=False,
                    bracket=(lo, hi))

    a = brentq(f, lo, hi, xtol=tol, rtol=8.9e-16, maxiter=200)
    return dict(alpha=float(a), ce_target=ce_target, bracketed=True,
                ce_achieved=ce_var(a, F0), bracket=(lo, hi))


def match_alpha_threshold(F0=None, eps=None):
    """Robustness calibration: alpha implied by fixing k_alpha = k_eps.

    With k_alpha held at k_eps the budget equation pins Y0, and the level
    follows from the P-measure quantile:
        alpha = Phi( (ln(k_alpha) - ln(Y0) - m_P·T) / (sigma_Y·sqrt(T)) ).
    """
    if F0 is None:
        F0 = P.F0
    k_eps = ES.solve_es(F0, eps)['k_eps']

    # budget with the threshold FIXED (not proportional to Y0)
    f = lambda y: float(VaR.psi(y, k_eps, P.T)) - F0
    Y0 = brentq(f, 1e-12 * F0, F0, xtol=1e-16, rtol=8.9e-16, maxiter=200)

    m, s = log_moments(Y0)
    a = float(norm.cdf((np.log(k_eps) - m) / s))
    return dict(alpha=a, k_alpha=float(k_eps), k_eps=float(k_eps),
                Y0=float(Y0))


# ═══════════════════════════════════════════════════════════
# VaR feasibility bound  (closed form — lives in var_model)
# ═══════════════════════════════════════════════════════════

from .var_model import (market_price_of_risk, alpha_min,      # noqa: E402
                        quantile_hedge_cost as cost_var)


# ═══════════════════════════════════════════════════════════
# Table 2 assembly
# ═══════════════════════════════════════════════════════════

def table2(F0=None, eps=None, alpha_nominal=None, alpha_eqce=None,
           alpha_thr=None):
    """The five rows of Table 2 with exact statistics."""
    if F0 is None:
        F0 = P.F0
    if alpha_nominal is None:
        alpha_nominal = P.alpha
    if alpha_eqce is None:
        alpha_eqce = match_alpha_equal_ce(F0, eps)['alpha']
    if alpha_thr is None:
        alpha_thr = match_alpha_threshold(F0, eps)['alpha']

    rows = [
        merton_stats(F0),
        es_stats(F0, eps),
        var_stats(F0, alpha_nominal,
                  label=f'VaR (alpha={alpha_nominal:.4g})'),
        var_stats(F0, alpha_eqce,
                  label=f'VaR equal-CE (alpha={alpha_eqce:.5f})'),
        var_stats(F0, alpha_thr,
                  label=f'VaR threshold-matched (alpha={alpha_thr:.5f})'),
    ]
    ce_m = rows[0]['ce']
    for r in rows:
        r['ce_loss_pct'] = ce_loss_pct(r, ce_m)
    return rows


if __name__ == "__main__":
    rows = table2()
    hdr = (f"{'strategy':<40}{'mean':>8}{'std':>8}{'P(F<k)':>9}{'E[(k-F)+]':>11}"
           f"{'CondSF':>9}{'Q5':>8}{'Bot5':>8}{'CE':>9}{'CEloss%':>9}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['label']:<40}{r['mean']:>8.5f}{r['std']:>8.5f}"
              f"{r['prob_shortfall']:>9.5f}{r['exp_shortfall']:>11.5f}"
              f"{r['cond_shortfall']:>9.5f}{r['q5']:>8.5f}"
              f"{r['bottom5_mean']:>8.5f}{r['ce']:>9.6f}{r['ce_loss_pct']:>9.4f}")
