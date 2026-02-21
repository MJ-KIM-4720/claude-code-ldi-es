"""
Monte Carlo Simulation for LDI Models
=======================================
Simulates funding ratio paths under ES, VaR, and Merton strategies.

Dynamics under constrained strategy (P-measure):
    d ln(Y) = [r̃ + A·γ·σ²_Y - A²·σ²_Y/2] dt + A·σ_Y dW^P

where A = adjustment_factor(Y, tau) depends on model:
    - Merton: A = 1 always
    - ES:     A = ES.adjustment_factor(Y, k_eps, c, tau)
    - VaR:    A = VaR.adjustment_factor(Y, k_alpha, tau)

Threshold is solved once at t=0 (time-series approach).
"""

import numpy as np
from scipy.stats import norm

from .bs_utils import bs_put, bs_d1, bs_d2, bs_digital_put, bs_digital_put_delta
from . import params as P
from . import es_model as ES
from . import var_model as VaR


# ═══════════════════════════════════════════════════════════
# Vectorized adjustment factors (array of Y → array of A)
# ═══════════════════════════════════════════════════════════

def _es_adjustment_vec(Y, k_eps, c, tau):
    """Vectorized ES adjustment factor for array Y."""
    # Psi = Y + Put(Y, k) - c * Put(Y, k_eps)
    P_k  = bs_put(Y, P.k,   P.r_tilde, P.sigma_Y, tau)
    P_ke = bs_put(Y, k_eps, P.r_tilde, P.sigma_Y, tau)
    psi_val = Y + P_k - c * P_ke

    # dPsi/dy = 1 - N(-d1(k)) + c * N(-d1(k_eps))
    d1_k  = bs_d1(Y, P.k,   P.r_tilde, P.sigma_Y, tau)
    d1_ke = bs_d1(Y, k_eps, P.r_tilde, P.sigma_Y, tau)
    dpsi_val = 1.0 - norm.cdf(-d1_k) + c * norm.cdf(-d1_ke)

    return (Y / psi_val) * dpsi_val


def _var_adjustment_vec(Y, k_alpha, tau):
    """Vectorized VaR adjustment factor for array Y."""
    # Psi = Y + Put(k) - Put(k_alpha) - (k - k_alpha) * Digital(k_alpha)
    P_k  = bs_put(Y, P.k,      P.r_tilde, P.sigma_Y, tau)
    P_ka = bs_put(Y, k_alpha,  P.r_tilde, P.sigma_Y, tau)
    D_ka = bs_digital_put(Y, k_alpha, P.r_tilde, P.sigma_Y, tau)
    psi_val = Y + P_k - P_ka - (P.k - k_alpha) * D_ka

    # dPsi/dy = 1 - N(-d1(k)) + N(-d1(k_alpha)) - (k-k_alpha) * dDigital/dy
    d1_k  = bs_d1(Y, P.k,      P.r_tilde, P.sigma_Y, tau)
    d1_ka = bs_d1(Y, k_alpha,  P.r_tilde, P.sigma_Y, tau)
    dD_ka = bs_digital_put_delta(Y, k_alpha, P.r_tilde, P.sigma_Y, tau)
    dpsi_val = 1.0 - norm.cdf(-d1_k) + norm.cdf(-d1_ka) - (P.k - k_alpha) * dD_ka

    # Guard: if psi_val <= 0 (should not happen normally), return 1
    safe = psi_val > 0
    A = np.where(safe, (Y / psi_val) * dpsi_val, 1.0)
    return A


# ═══════════════════════════════════════════════════════════
# Main simulation
# ═══════════════════════════════════════════════════════════

def simulate_paths(y0, n_paths=10000, n_steps=250, model='es', seed=42):
    """Simulate funding ratio paths via Euler-Maruyama on log(Y).

    Args:
        y0: initial funding ratio
        n_paths: number of MC paths
        n_steps: time steps (dt = T / n_steps)
        model: 'es', 'var', or 'merton'
        seed: random seed

    Returns:
        paths: ndarray (n_paths, n_steps+1) — funding ratio paths
        t_grid: ndarray (n_steps+1,) — time points [0, T]
    """
    rng = np.random.default_rng(seed)
    dt = P.T / n_steps
    sqrt_dt = np.sqrt(dt)
    t_grid = np.linspace(0, P.T, n_steps + 1)

    # Solve threshold once at t=0
    if model == 'es':
        k_eps, c, binding = ES.solve_threshold(y0)
    elif model == 'var':
        k_alpha, binding = VaR.solve_threshold(y0)
    elif model == 'merton':
        binding = False
    else:
        raise ValueError(f"Unknown model: {model}")

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = y0

    Z = rng.standard_normal((n_paths, n_steps))

    for i in range(n_steps):
        Y_curr = paths[:, i]
        tau = P.T - t_grid[i]

        if tau < 1e-10:
            paths[:, i + 1] = Y_curr
            continue

        # Compute A vectorized
        if model == 'merton' or not binding:
            A = np.ones(n_paths)
        elif model == 'es':
            A = _es_adjustment_vec(Y_curr, k_eps, c, tau)
        elif model == 'var':
            A = _var_adjustment_vec(Y_curr, k_alpha, tau)

        # Clamp A to avoid extreme values
        A = np.clip(A, 0.0, 5.0)

        # Euler-Maruyama on log(Y)
        drift = (P.r_tilde + A * P.GAMMA * P.sigma_Y**2
                 - A**2 * P.sigma_Y**2 / 2) * dt
        diffusion = A * P.sigma_Y * sqrt_dt * Z[:, i]

        log_Y_next = np.log(Y_curr) + drift + diffusion
        paths[:, i + 1] = np.exp(log_Y_next)

    return paths, t_grid


# ═══════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════

def compute_path_stats(paths):
    """Compute quantile statistics across paths at each time step.

    Returns dict with keys: mean, median, q05, q25, q75, q95
    Each value is ndarray of length n_steps+1.
    """
    return {
        'mean':   np.mean(paths, axis=0),
        'median': np.median(paths, axis=0),
        'q05':    np.quantile(paths, 0.05, axis=0),
        'q25':    np.quantile(paths, 0.25, axis=0),
        'q75':    np.quantile(paths, 0.75, axis=0),
        'q95':    np.quantile(paths, 0.95, axis=0),
    }


def compute_terminal_stats(paths, k=None):
    """Compute terminal distribution statistics.

    Args:
        paths: (n_paths, n_steps+1)
        k: target funding ratio (default: P.k)

    Returns dict with:
        mean, std, median, shortfall_prob, expected_shortfall,
        q05, q95, min, max
    """
    if k is None:
        k = P.k
    Y_T = paths[:, -1]
    shortfall = Y_T < k

    stats = {
        'mean':             np.mean(Y_T),
        'std':              np.std(Y_T),
        'median':           np.median(Y_T),
        'shortfall_prob':   np.mean(shortfall),
        'q05':              np.quantile(Y_T, 0.05),
        'q95':              np.quantile(Y_T, 0.95),
        'min':              np.min(Y_T),
        'max':              np.max(Y_T),
    }

    # Expected shortfall: E[(k - Y_T)^+ | Y_T < k]
    if shortfall.any():
        stats['expected_shortfall'] = np.mean(k - Y_T[shortfall])
    else:
        stats['expected_shortfall'] = 0.0

    return stats


def certainty_equivalent(paths, gamma=None):
    """Compute certainty equivalent of terminal funding ratio.

    CE = ((1-gamma) * E[F_T^{1-gamma} / (1-gamma)])^{1/(1-gamma)}

    For gamma > 1, (1-gamma) < 0, so careful sign handling is needed.
    Paths where F_T <= 0 are excluded (would give infinity for gamma > 1).

    Args:
        paths: (n_paths, n_steps+1)
        gamma: risk aversion (default: P.GAMMA)

    Returns:
        CE value (scalar)
    """
    if gamma is None:
        gamma = P.GAMMA
    Y_T = paths[:, -1]

    # Exclude zero/negative terminal values (gamma > 1 → 0^{1-gamma} = inf)
    mask = Y_T > 0
    Y_T = Y_T[mask]

    if len(Y_T) == 0:
        return 0.0

    one_minus_gamma = 1.0 - gamma
    # U_i = F_T_i^{1-gamma} / (1-gamma)
    U = Y_T ** one_minus_gamma / one_minus_gamma
    EU = np.mean(U)
    # CE = ((1-gamma) * EU)^{1/(1-gamma)}
    CE = (one_minus_gamma * EU) ** (1.0 / one_minus_gamma)
    return float(CE)


def shortfall_prob_over_time(paths, k=None):
    """Compute P(Y_t < k) at each time step.

    Returns ndarray of length n_steps+1.
    """
    if k is None:
        k = P.k
    return np.mean(paths < k, axis=0)
