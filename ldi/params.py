"""
LDI Parameters & Derived Quantities
=====================================
Baseline: Jo et al. (2025), Kraft & Steffensen (2013).

IMPORTANT: R=0.02 (real rate), r=0.04 (nominal rate).
  → IIB excess = mu_I + R - r = 0.003
  → Merton total ≈ 80.4%
  → r_tilde = -0.0084    (liability grows faster than assets)
"""

import contextlib
import numpy as np

# ── Market ──────────────────────────────────────────────
MU_S    = 0.08      # Stock expected return
SIGMA_S = 0.18      # Stock volatility
MU_I    = 0.023     # Expected inflation rate
SIGMA_I = 0.07      # IIB volatility
R       = 0.02      # Real interest rate
r       = 0.04      # Nominal risk-free rate
RHO     = -0.15     # Stock-IIB correlation

# ── Liability ───────────────────────────────────────────
BETA_0  = 0.03      # Base liability growth rate
BETA_1  = 0.8       # Inflation sensitivity

# ── Preferences ─────────────────────────────────────────
GAMMA   = 3.0       # CRRA risk aversion

# ── Constraint ──────────────────────────────────────────
k       = 1.0       # Target funding ratio
alpha   = 0.10      # VaR confidence level  P(F_T < k) <= alpha
epsilon = 0.05      # ES budget             E^Q[(k-F_T)^+ e^{-r̃T}] <= epsilon
T       = 10.0      # Horizon (years)

# ── Initial condition ───────────────────────────────────
y0      = 1.0       # Initial funding ratio (default)


# ═══════════════════════════════════════════════════════════
# Derived quantities (computed once at import time)
# ═══════════════════════════════════════════════════════════

# Liability-adjusted discount rate
r_tilde = r - (BETA_0 + BETA_1 * MU_I)

# Covariance matrix & inverse
Sigma_mat = np.array([
    [SIGMA_S**2,              RHO * SIGMA_S * SIGMA_I],
    [RHO * SIGMA_S * SIGMA_I, SIGMA_I**2             ]
])
Sigma_inv = np.linalg.inv(Sigma_mat)

# Excess return vector:  [mu_S - r,  mu_I + R - r]
mu_excess = np.array([MU_S - r, MU_I + R - r])

# Sharpe ratio squared: theta^T theta = mu_exc^T Sigma^{-1} mu_exc
theta_sq = mu_excess @ Sigma_inv @ mu_excess

# Aggregate portfolio volatility
sigma_Y = np.sqrt(theta_sq) / GAMMA

# Unconstrained Merton weights: Pi* = Sigma^{-1} mu_exc / gamma
Pi_star = Sigma_inv @ mu_excess / GAMMA     # [pi*_S, pi*_I]

# P-measure drift of ln(Y):  m_P = r̃ + γσ²_Y - σ²_Y/2
m_P = r_tilde + GAMMA * sigma_Y**2 - sigma_Y**2 / 2


# ═══════════════════════════════════════════════════════════
# Parameter recomputation (for sensitivity analysis)
# ═══════════════════════════════════════════════════════════

def recompute_derived():
    """Recompute all derived quantities from current primitive parameters.

    Call after modifying any primitive parameter (MU_S, SIGMA_S, MU_I,
    SIGMA_I, R, r, RHO, BETA_0, BETA_1, GAMMA).
    """
    global r_tilde, Sigma_mat, Sigma_inv, mu_excess, theta_sq
    global sigma_Y, Pi_star, m_P

    r_tilde = r - (BETA_0 + BETA_1 * MU_I)

    Sigma_mat = np.array([
        [SIGMA_S**2,              RHO * SIGMA_S * SIGMA_I],
        [RHO * SIGMA_S * SIGMA_I, SIGMA_I**2             ]
    ])
    Sigma_inv = np.linalg.inv(Sigma_mat)

    mu_excess = np.array([MU_S - r, MU_I + R - r])
    theta_sq = mu_excess @ Sigma_inv @ mu_excess
    sigma_Y = np.sqrt(theta_sq) / GAMMA
    Pi_star = Sigma_inv @ mu_excess / GAMMA
    m_P = r_tilde + GAMMA * sigma_Y**2 - sigma_Y**2 / 2


@contextlib.contextmanager
def override_params(**kwargs):
    """Temporarily override parameters and recompute derived quantities.

    Usage:
        with override_params(GAMMA=5.0, T=10.0):
            A = ES.cross_sectional_A(y0=0.8)

    On exit, all parameters (primitive and derived) are restored.
    """
    _derived_names = ['r_tilde', 'Sigma_mat', 'Sigma_inv', 'mu_excess',
                      'theta_sq', 'sigma_Y', 'Pi_star', 'm_P']

    g = globals()

    # Save primitive overrides
    saved_prims = {}
    for name in kwargs:
        if name not in g:
            raise KeyError(f"Unknown parameter: {name}")
        saved_prims[name] = g[name]

    # Save derived quantities (deep copy arrays)
    saved_derived = {}
    for name in _derived_names:
        val = g[name]
        saved_derived[name] = val.copy() if isinstance(val, np.ndarray) else val

    try:
        for name, value in kwargs.items():
            g[name] = value
        recompute_derived()
        yield
    finally:
        for name, value in saved_prims.items():
            g[name] = value
        for name, value in saved_derived.items():
            g[name] = value


# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════

def print_params():
    """Print all parameters and derived quantities."""
    print("=" * 55)
    print("  LDI Model Parameters")
    print("=" * 55)
    print(f"  Market:  mu_S={MU_S}, sigma_S={SIGMA_S}, mu_I={MU_I}, sigma_I={SIGMA_I}")
    print(f"           R={R} (real), r={r} (nominal), rho={RHO}")
    print(f"  Liab:    beta_0={BETA_0}, beta_1={BETA_1}")
    print(f"  Pref:    gamma={GAMMA}")
    print(f"  Constr:  k={k}, alpha={alpha}, epsilon={epsilon}, T={T}")
    print("-" * 55)
    print(f"  Derived: r_tilde  = {r_tilde:.4f}")
    print(f"           sigma_Y  = {sigma_Y:.4f}")
    print(f"           m_P      = {m_P:.4f}")
    print(f"           Pi*_S    = {Pi_star[0]:.4f}  ({Pi_star[0]*100:.1f}%)")
    print(f"           Pi*_I    = {Pi_star[1]:.4f}  ({Pi_star[1]*100:.1f}%)")
    print(f"           Total    = {Pi_star.sum():.4f}  ({Pi_star.sum()*100:.1f}%)")
    print("=" * 55)


if __name__ == "__main__":
    print_params()
