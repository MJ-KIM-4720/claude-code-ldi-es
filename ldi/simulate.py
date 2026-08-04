"""
Fixed-Claim Monte Carlo for the LDI Models
===========================================
The claim is decided ONCE at t=0. The simulation never re-solves a
threshold along a path.

    1. Solve the joint system at t=0  ->  (Y0, k_eps) or (Y0, k_alpha).
    2. Simulate the reference process under P (exact GBM):
           d ln Y = m_P dt + sigma_Y dW,   m_P = r̃ + gamma·sigma_Y² - sigma_Y²/2
    3. Funding ratio  F_t = Psi(t, Y_t);  allocation  pi_t = A(t,Y_t)·Pi*.

REPLICATION CHECK.  Because Pi* is a fixed direction, the strategy is
"fraction A_t in the aggregate risky fund, 1-A_t in the riskless asset",
and the aggregate risky fund's value process IS the reference process Y.
So a genuinely self-financing discrete-rebalancing wealth path is

    F^repl_{i+1} = F^repl_i·[1 + A_i·(Y_{i+1}/Y_i - 1) + (1-A_i)·(e^{r̃Δ}-1)]

and max_t |F^repl_t - Psi(t,Y_t)| -> 0 as the step size shrinks.

MEASURE CHANGE.  The Q-expected shortfall is recovered from the same
P-paths via  dQ/dP = exp(-gamma·sigma_Y·W_T - gamma²sigma_Y²T/2), since
the market price of risk of the aggregate fund is gamma·sigma_Y.
"""

import numpy as np
from scipy.optimize import brentq

from . import params as P
from . import es_model as ES
from . import var_model as VaR


DEFAULT_N_PATHS = 10_000
DEFAULT_N_STEPS = 120          # monthly over T = 10y (12 steps/year)
DEFAULT_SEED = 20260803
SCHEME = "exact GBM (log-Euler is exact for the reference process)"

# Terminal-only sampling (Table 2). The claim depends on the path ONLY through
# Y_T, so the reported table needs no path at all: draw Y_T straight from its
# exact lognormal law. That is what makes N = 10^6 cheap.
DEFAULT_N_TERMINAL = 1_000_000
TERMINAL_SCHEME = "exact lognormal terminal draws (no path discretisation)"


# ═══════════════════════════════════════════════════════════
# Reference process
# ═══════════════════════════════════════════════════════════

def reference_paths(Y0, n_paths=DEFAULT_N_PATHS, n_steps=DEFAULT_N_STEPS,
                    seed=DEFAULT_SEED):
    """Exact-scheme GBM paths of the reference process Y under P.

    Returns (t_grid, Y, Z) with Y of shape (n_paths, n_steps+1).
    """
    rng = np.random.default_rng(seed)
    dt = P.T / n_steps
    t_grid = np.linspace(0.0, P.T, n_steps + 1)
    Z = rng.standard_normal((n_paths, n_steps))

    incr = P.m_P * dt + P.sigma_Y * np.sqrt(dt) * Z
    log_Y = np.concatenate(
        [np.full((n_paths, 1), np.log(Y0)),
         np.log(Y0) + np.cumsum(incr, axis=1)], axis=1)
    return t_grid, np.exp(log_Y), Z


# ═══════════════════════════════════════════════════════════
# Terminal-only sampling  (Table 2)
# ═══════════════════════════════════════════════════════════

def terminal_draws(Y0, n=DEFAULT_N_TERMINAL, seed=DEFAULT_SEED, Z=None):
    """Exact lognormal draws of Y_T. No path, no discretisation error.

    Pass Z to reuse the same standard normals across strategies (common
    random numbers) — the strategies then differ only through Y0 and the
    claim, which is what makes differences such as the CE loss comparable.
    """
    if Z is None:
        Z = np.random.default_rng(seed).standard_normal(n)
    return Y0 * np.exp(P.m_P * P.T + P.sigma_Y * np.sqrt(P.T) * Z)


def claim_from_params(y, c, k_low, k=None):
    """Generic piecewise claim g(y) = c·y | k | y (same form as exact_stats).

    In the middle segment the payoff is EXACTLY k, so the probability atom
    P(F_T = k) can be counted by equality testing.
    """
    if k is None:
        k = P.k
    y = np.asarray(y, dtype=float)
    return np.where(y < k_low, c * y, np.where(y < k, k, y))


def terminal_sample(sols, n=DEFAULT_N_TERMINAL, seed=DEFAULT_SEED):
    """Terminal funding-ratio samples for several claims on common normals.

    Args:
        sols: {label: (Y0, c, k_low)}
    Returns {label: F_T array} plus the shared Z under key '_Z'.
    """
    Z = np.random.default_rng(seed).standard_normal(n)
    out = {'_Z': Z}
    for label, (Y0, c, k_low) in sols.items():
        out[label] = claim_from_params(terminal_draws(Y0, n, Z=Z), c, k_low)
    return out


def atom_fraction(F_T, k=None):
    """Sample P(F_T = k) — the mass sitting exactly on the target."""
    if k is None:
        k = P.k
    return float(np.mean(np.asarray(F_T) == k))


# ═══════════════════════════════════════════════════════════
# Strategy evaluation on a fixed claim
# ═══════════════════════════════════════════════════════════

def evaluate(model, sol, t_grid, Y):
    """Map reference paths to (F, A) for a fixed claim.

    Args:
        model: 'es', 'var' or 'merton'
        sol:   dict from ES.solve_es / VaR.solve_var (ignored for merton)
    Returns dict(F=(n,ns+1), A=(n,ns+1)).
    """
    n_paths, n_pts = Y.shape
    F = np.empty_like(Y)
    A = np.empty_like(Y)

    for i, t in enumerate(t_grid):
        tau = P.T - t
        y = Y[:, i]
        if model == 'merton' or not sol.get('binding', True):
            F[:, i] = y
            A[:, i] = 1.0
        elif model == 'es':
            F[:, i] = ES.psi(y, sol['k_eps'], sol['c'], tau)
            A[:, i] = (ES.adjustment_factor(y, sol['k_eps'], sol['c'], tau)
                       if tau > 0 else np.nan)
        elif model == 'var':
            F[:, i] = VaR.psi(y, sol['k_alpha'], tau)
            A[:, i] = (VaR.adjustment_factor(y, sol['k_alpha'], tau)
                       if tau > 0 else np.nan)
        else:
            raise ValueError(f"Unknown model: {model}")
    return dict(F=F, A=A)


def replicate(F0, A, Y, n_steps):
    """Discrete self-financing wealth path (see module docstring)."""
    dt = P.T / n_steps
    rf = np.exp(P.r_tilde * dt) - 1.0
    F = np.empty_like(Y)
    F[:, 0] = F0
    for i in range(n_steps):
        risky = Y[:, i + 1] / Y[:, i] - 1.0
        a = A[:, i]
        F[:, i + 1] = F[:, i] * (1.0 + a * risky + (1.0 - a) * rf)
    return F


def radon_nikodym(Y, Y0):
    """dQ/dP evaluated at T from the terminal reference state."""
    W_T = (np.log(Y[:, -1] / Y0) - P.m_P * P.T) / P.sigma_Y
    theta = P.GAMMA * P.sigma_Y          # market price of risk of the agg. fund
    return np.exp(-theta * W_T - 0.5 * theta ** 2 * P.T)


# ═══════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════

def certainty_equivalent(F_T, gamma=None):
    """CE = ((1-g)·E[F^{1-g}/(1-g)])^{1/(1-g)}."""
    if gamma is None:
        gamma = P.GAMMA
    F_T = np.asarray(F_T, dtype=float)
    F_T = F_T[F_T > 0]
    if F_T.size == 0:
        return 0.0
    omg = 1.0 - gamma
    return float((omg * np.mean(F_T ** omg / omg)) ** (1.0 / omg))


def cvar(F_T, level=0.05):
    """CVaR_level: mean of the worst `level` fraction of terminal outcomes."""
    F_T = np.asarray(F_T, dtype=float)
    q = np.quantile(F_T, level)
    tail = F_T[F_T <= q]
    return float(np.mean(tail)) if tail.size else float(q)


def terminal_stats(F_T, k=None):
    """Point estimates of the terminal statistics reported in Table 2."""
    if k is None:
        k = P.k
    F_T = np.asarray(F_T, dtype=float)
    short = F_T < k
    return {
        'mean': float(np.mean(F_T)),
        'std': float(np.std(F_T, ddof=1)),
        'median': float(np.median(F_T)),
        'q05': float(np.quantile(F_T, 0.05)),
        'q95': float(np.quantile(F_T, 0.95)),
        'min': float(np.min(F_T)),
        'shortfall_prob': float(np.mean(short)),
        'exp_shortfall': float(np.mean(np.maximum(k - F_T, 0.0))),
        'cond_shortfall': float(np.mean(k - F_T[short])) if short.any() else 0.0,
        'cvar05': cvar(F_T, 0.05),
        'CE': certainty_equivalent(F_T),
    }


def bootstrap_se(F_T, n_boot=500, seed=7, k=None):
    """Bootstrap standard errors for every entry of terminal_stats."""
    rng = np.random.default_rng(seed)
    n = len(F_T)
    keys = terminal_stats(F_T[:10], k).keys()
    draws = {key: np.empty(n_boot) for key in keys}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        s = terminal_stats(F_T[idx], k)
        for key in keys:
            draws[key][b] = s[key]
    return {key: float(np.std(v, ddof=1)) for key, v in draws.items()}


# ═══════════════════════════════════════════════════════════
# Full run
# ═══════════════════════════════════════════════════════════

def run(F0=None, eps=None, alpha=None, n_paths=DEFAULT_N_PATHS,
        n_steps=DEFAULT_N_STEPS, seed=DEFAULT_SEED, n_boot=500,
        with_se=True, models=('merton', 'es', 'var')):
    """Fixed-claim MC for all three strategies on common random numbers.

    Every model is driven by the SAME standard normals; only Y0 and the
    claim differ. Returns a dict keyed by model name plus 'meta'.
    """
    if F0 is None:
        F0 = P.F0

    sols = {}
    if 'es' in models:
        sols['es'] = ES.solve_es(F0, eps)
    if 'var' in models:
        sols['var'] = VaR.solve_var(F0, alpha)
    if 'merton' in models:
        sols['merton'] = dict(Y0=F0, binding=False)

    out = {'meta': dict(F0=F0, n_paths=n_paths, n_steps=n_steps, seed=seed,
                        scheme=SCHEME, dt=P.T / n_steps,
                        eps=P.epsilon if eps is None else eps,
                        alpha=P.alpha if alpha is None else alpha)}

    for name in models:
        sol = sols[name]
        t_grid, Y, Z = reference_paths(sol['Y0'], n_paths, n_steps, seed)
        paths = evaluate(name, sol, t_grid, Y)
        F = paths['F']
        F_repl = replicate(F0, paths['A'], Y, n_steps)
        rn = radon_nikodym(Y, sol['Y0'])

        rec = dict(sol=sol, t_grid=t_grid, Y=Y, F=F, A=paths['A'],
                   F_repl=F_repl)
        rec['stats'] = terminal_stats(F[:, -1])
        if with_se:
            rec['se'] = bootstrap_se(F[:, -1], n_boot=n_boot, seed=seed + 1)
        # replication diagnostics
        err = np.abs(F_repl - F)
        rec['repl_err_max'] = float(np.max(err))
        rec['repl_err_terminal'] = float(np.max(err[:, -1]))
        rec['repl_err_mean'] = float(np.mean(err))
        # realized Q-expected shortfall  E^Q[e^{-r̃T}(k - F_T)^+]
        disc = np.exp(-P.r_tilde * P.T)
        sf = np.maximum(P.k - F[:, -1], 0.0)
        rec['q_shortfall'] = float(disc * np.mean(rn * sf))
        rec['q_shortfall_se'] = float(disc * np.std(rn * sf, ddof=1)
                                      / np.sqrt(n_paths))
        out[name] = rec

    # welfare relative to Merton
    if 'merton' in models:
        ce_m = out['merton']['stats']['CE']
        for name in models:
            out[name]['ce_loss_pct'] = 100.0 * (ce_m - out[name]['stats']['CE']) / ce_m
    return out


# ═══════════════════════════════════════════════════════════
# VaR calibration:  equal-CE  and  threshold matching
# ═══════════════════════════════════════════════════════════

def _ce_loss_var(alpha, F0, ce_merton, n_paths, n_steps, seed):
    sol = VaR.solve_var(F0, alpha)
    _, Y, _ = reference_paths(sol['Y0'], n_paths, n_steps, seed)
    F_T = VaR.psi(Y[:, -1], sol['k_alpha'], 0.0)
    return 100.0 * (ce_merton - certainty_equivalent(F_T)) / ce_merton


def match_alpha_equal_ce(F0=None, eps=None, n_paths=DEFAULT_N_PATHS,
                         n_steps=DEFAULT_N_STEPS, seed=DEFAULT_SEED,
                         tol=1e-10):
    """Find alpha whose MC certainty-equivalent loss equals the ES model's.

    Common random numbers make CE_loss_VaR(alpha) a smooth deterministic
    function of alpha, so plain bisection converges cleanly.
    """
    if F0 is None:
        F0 = P.F0

    # Merton and ES reference losses (same normals)
    _, Y_m, _ = reference_paths(F0, n_paths, n_steps, seed)
    ce_merton = certainty_equivalent(Y_m[:, -1])

    s_es = ES.solve_es(F0, eps)
    _, Y_e, _ = reference_paths(s_es['Y0'], n_paths, n_steps, seed)
    F_T_es = ES.psi(Y_e[:, -1], s_es['k_eps'], s_es['c'], 0.0)
    target = 100.0 * (ce_merton - certainty_equivalent(F_T_es)) / ce_merton

    a_lo = VaR.alpha_min(F0) * (1 + 1e-6) + 1e-9      # tightest affordable
    a_hi = 0.5
    f = lambda a: _ce_loss_var(a, F0, ce_merton, n_paths, n_steps, seed) - target

    f_lo, f_hi = f(a_lo), f(a_hi)
    if f_lo * f_hi > 0:
        return dict(alpha=np.nan, target_ce_loss=target, bracketed=False,
                    f_lo=f_lo, f_hi=f_hi, alpha_lo=a_lo, alpha_hi=a_hi)
    a = brentq(f, a_lo, a_hi, xtol=tol, rtol=8.9e-16, maxiter=200)
    return dict(alpha=float(a), target_ce_loss=target, bracketed=True,
                achieved_ce_loss=target + f(a), alpha_lo=a_lo, alpha_hi=a_hi)


def match_alpha_threshold(F0=None, eps=None):
    """Robustness calibration: alpha such that k_alpha = k_eps."""
    if F0 is None:
        F0 = P.F0
    s_es = ES.solve_es(F0, eps)
    k_eps = s_es['k_eps']

    def f(a):
        s = VaR.solve_var(F0, a, strict=False)
        if not s['feasible']:
            return np.inf
        return s['k_alpha'] - k_eps

    a_lo = VaR.alpha_min(F0) * (1 + 1e-6) + 1e-9
    a_hi = 0.5
    if f(a_lo) * f(a_hi) > 0:
        return dict(alpha=np.nan, k_eps=k_eps, bracketed=False)
    a = brentq(f, a_lo, a_hi, xtol=1e-14, rtol=8.9e-16, maxiter=200)
    s = VaR.solve_var(F0, a)
    return dict(alpha=float(a), k_eps=k_eps, k_alpha=s['k_alpha'],
                bracketed=True)


# ═══════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    res = run()
    m = res['meta']
    print(f"Fixed-claim MC: N={m['n_paths']}, steps={m['n_steps']}, "
          f"seed={m['seed']}, dt={m['dt']:.4f}")
    print(f"{'':8s} {'mean':>8} {'std':>8} {'P(F<k)':>8} {'E[(k-F)+]':>10} "
          f"{'CVaR5':>8} {'CE':>8} {'CEloss%':>8} {'replerr':>9}")
    for name in ('merton', 'es', 'var'):
        s = res[name]['stats']
        print(f"{name:8s} {s['mean']:8.4f} {s['std']:8.4f} "
              f"{s['shortfall_prob']:8.4f} {s['exp_shortfall']:10.4f} "
              f"{s['cvar05']:8.4f} {s['CE']:8.4f} "
              f"{res[name]['ce_loss_pct']:8.3f} {res[name]['repl_err_max']:9.2e}")
    print()
    for name in ('merton', 'es', 'var'):
        print(f"  {name:8s} Q-shortfall = {res[name]['q_shortfall']:.6f} "
              f"(± {res[name]['q_shortfall_se']:.6f})")
