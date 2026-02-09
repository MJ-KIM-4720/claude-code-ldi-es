"""
Black-Scholes Utilities (Liability-Adjusted)
==============================================
All functions use r_tilde (liability-adjusted rate) and sigma_Y (aggregate vol).

Convention:
  - y: current value of unconstrained funding ratio
  - K: strike price
  - tau: time to maturity (T - t)
  - r_t: liability-adjusted discount rate (r_tilde)
  - sig: aggregate portfolio volatility (sigma_Y)
"""

import numpy as np
from scipy.stats import norm


def bs_d1(y, K, r_t, sig, tau):
    """d₁ = [ln(y/K) + (r̃ + σ²/2)τ] / (σ√τ)"""
    return (np.log(y / K) + (r_t + sig**2 / 2) * tau) / (sig * np.sqrt(tau))


def bs_d2(y, K, r_t, sig, tau):
    """d₂ = d₁ - σ√τ"""
    return bs_d1(y, K, r_t, sig, tau) - sig * np.sqrt(tau)


def bs_put(y, K, r_t, sig, tau):
    """Black-Scholes put price with liability-adjusted rate.

    Put(y, K) = K·e^{-r̃τ}·N(-d₂) - y·N(-d₁)
    """
    d1 = bs_d1(y, K, r_t, sig, tau)
    d2 = d1 - sig * np.sqrt(tau)
    return K * np.exp(-r_t * tau) * norm.cdf(-d2) - y * norm.cdf(-d1)


def bs_put_delta(y, K, r_t, sig, tau):
    """∂Put/∂y = -N(-d₁)"""
    d1 = bs_d1(y, K, r_t, sig, tau)
    return -norm.cdf(-d1)


def bs_digital_put(y, K, r_t, sig, tau):
    """Digital put price: e^{-r̃τ}·N(-d₂)

    Pays 1 if Y_T < K at maturity.
    """
    d2 = bs_d2(y, K, r_t, sig, tau)
    return np.exp(-r_t * tau) * norm.cdf(-d2)


def bs_digital_put_delta(y, K, r_t, sig, tau):
    """∂Digital/∂y = -e^{-r̃τ}·φ(d₂) / (y·σ·√τ)"""
    d2 = bs_d2(y, K, r_t, sig, tau)
    return -np.exp(-r_t * tau) * norm.pdf(d2) / (y * sig * np.sqrt(tau))
