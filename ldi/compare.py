"""
VaR vs ES Comparison — figures for the joint-system / fixed-claim model
=======================================================================
Two mutually exclusive presentation modes; the paper picks one.

  Mode A "cross-sectional": x-axis = F0. Each point is a DIFFERENT fund
      solving its own joint system at t=0.
        A-1  fixed eps: only the feasible range F0 > k·e^{-r̃T} - eps is
             drawn — the curve is deliberately short.
        A-2  eps(F0) = eps_min(F0) + delta: every fund gets the same
             slack delta above its own attainable floor, so the whole
             range is covered.

  Mode B "fixed-claim": ONE fund. (Y0, k_eps) solved once at t=0; the
      x-axis is the reference state y at time snapshots. This is the
      object A(t,y) = y·Psi_y/Psi actually describes.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from . import params as P
from . import es_model as ES
from . import var_model as VaR
from .style import (COLORS, LINE_STYLES, FIGSIZES, LEGEND,
                    setup_grid, add_merton_hline, add_k_vline, savefig)

DELTA_DEFAULT = 0.05          # slack above eps_min in mode A-2


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _A_es(F0, eps):
    return ES.cross_sectional_A(F0, eps, strict=False)


def _A_var(F0, alpha):
    return VaR.cross_sectional_A(F0, alpha, strict=False)


def es_curve_fixed_eps(F0_vals, eps):
    """A_ES(0,Y0) across funds at a common eps (NaN where infeasible)."""
    return np.array([_A_es(f, eps) for f in F0_vals])


def es_curve_slack(F0_vals, delta=DELTA_DEFAULT):
    """A_ES with eps(F0) = eps_min(F0) + delta."""
    return np.array([_A_es(f, P.eps_min(f) + delta) for f in F0_vals])


def feasible_F0_min(eps):
    """Smallest F0 for which a common eps is attainable: F0 > k·e^{-r̃T} - eps."""
    return P.k * np.exp(-P.r_tilde * P.T) - eps


# ═══════════════════════════════════════════════════════════
# Mode A — cross-sectional
# ═══════════════════════════════════════════════════════════

def plot_cross_sectional(F0_range=(0.5, 2.0), n_points=400, eps=None,
                         alpha=None, variant='A1', delta=DELTA_DEFAULT,
                         save_path=None):
    """Cross-sectional A(F0) and total allocation for VaR and ES."""
    if eps is None:
        eps = P.epsilon
    if alpha is None:
        alpha = P.alpha

    F0_vals = np.linspace(*F0_range, n_points)
    Av = np.array([_A_var(f, alpha) for f in F0_vals])
    if variant == 'A1':
        Ae = es_curve_fixed_eps(F0_vals, eps)
        es_label = rf'ES ($\varepsilon$={eps}, feasible range only)'
        sub = (rf'A-1: fixed $\varepsilon$={eps}; infeasible for '
               rf'$F_0 \leq k e^{{-\tilde r T}}-\varepsilon = '
               rf'{feasible_F0_min(eps):.4f}$')
    else:
        Ae = es_curve_slack(F0_vals, delta)
        es_label = rf'ES ($\varepsilon=\varepsilon_{{\min}}(F_0)+{delta}$)'
        sub = (rf'A-2: each fund allowed slack $\delta$={delta} above its own '
               rf'attainable floor $\varepsilon_{{\min}}(F_0)$')

    mt = P.Pi_star.sum() * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(F0_vals, Av, label=rf'VaR ($\alpha$={alpha})', **LINE_STYLES['VaR'])
    ax1.plot(F0_vals, Ae, label=es_label, **LINE_STYLES['ES'])
    add_merton_hline(ax1, 1.0, 'Merton ($A=1$)')
    add_k_vline(ax1)
    if variant == 'A1':
        ax1.axvspan(F0_range[0], max(F0_range[0], feasible_F0_min(eps)),
                    color='0.6', alpha=0.25, label='ES infeasible')
    ax1.set_xlabel('Initial funding ratio $F_0$')
    ax1.set_ylabel('Adjustment factor $A(0,Y_0)$')
    ax1.set_title('Cross-sectional adjustment factor')
    ax1.legend(**LEGEND)
    setup_grid(ax1)
    ax1.set_xlim(F0_range)

    ax2.plot(F0_vals, Av * mt, label='VaR', **LINE_STYLES['VaR'])
    ax2.plot(F0_vals, Ae * mt, label='ES', **LINE_STYLES['ES'])
    add_merton_hline(ax2, mt, f'Merton ({mt:.0f}%)')
    add_k_vline(ax2)
    if variant == 'A1':
        ax2.axvspan(F0_range[0], max(F0_range[0], feasible_F0_min(eps)),
                    color='0.6', alpha=0.25)
    ax2.set_xlabel('Initial funding ratio $F_0$')
    ax2.set_ylabel('Total risky allocation (%)')
    ax2.set_title('Cross-sectional total allocation')
    ax2.legend(**LEGEND)
    setup_grid(ax2)
    ax2.set_xlim(F0_range)

    plt.suptitle(sub, fontsize=13)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


def plot_solution_map(F0_range=(0.5, 2.0), n_points=400, eps=None,
                      variant='A1', delta=DELTA_DEFAULT, save_path=None):
    """(Y0, k_eps, c) of the joint solution across funds."""
    if eps is None:
        eps = P.epsilon
    F0_vals = np.linspace(*F0_range, n_points)

    Y0, ke, cc = [], [], []
    for f in F0_vals:
        e = eps if variant == 'A1' else P.eps_min(f) + delta
        s = ES.solve_es(f, e, strict=False)
        Y0.append(s['Y0'] if s['feasible'] else np.nan)
        ke.append(s['k_eps'] if s['feasible'] else np.nan)
        cc.append(s['c'] if s['feasible'] else np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(F0_vals, Y0, color=COLORS['ES'], lw=2.5, label='$Y_0$ (reference start)')
    ax1.plot(F0_vals, F0_vals, color='0.4', ls=':', lw=1.5, label='$F_0$ (45°)')
    ax1.plot(F0_vals, ke, color=COLORS['VaR'], lw=2.0, ls='--',
             label=r'$k_\varepsilon$')
    ax1.set_xlabel('Initial funding ratio $F_0$')
    ax1.set_ylabel('Level')
    ax1.set_title(r'Joint solution: $Y_0 < F_0$ and $k_\varepsilon$')
    ax1.legend(**LEGEND)
    setup_grid(ax1)

    ax2.plot(F0_vals, cc, color=COLORS['ES'], lw=2.5)
    ax2.axhline(1.0, color='0.4', ls=':', lw=1.5)
    ax2.set_xlabel('Initial funding ratio $F_0$')
    ax2.set_ylabel(r'$c = k/k_\varepsilon$')
    ax2.set_title('Tail protection multiplier')
    setup_grid(ax2)

    tag = (rf'$\varepsilon$={eps}' if variant == 'A1'
           else rf'$\varepsilon=\varepsilon_{{\min}}(F_0)+{delta}$')
    plt.suptitle(f'ES joint solution across funds ({tag})', fontsize=13)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


def slack_binding_range(delta=DELTA_DEFAULT, F0_range=(0.2, 4.0), n_scan=4000):
    """(F0_lo, F0_hi) over which the A-2 budget eps_min(F0)+delta actually binds.

    The gap eps_M(F0) - eps_min(F0) is hump-shaped in F0: it vanishes both for
    deeply underfunded funds (where the Merton put premium is almost exactly
    the discounted intrinsic value) and for well funded ones (where the put is
    worthless). A fixed slack delta therefore binds only on an interior band —
    there are TWO crossings, not one. Outside the band the constraint is slack
    and A ≡ 1, which is an artefact of the delta design, not of the model.

    Returns (nan, nan) if the constraint never binds at this delta.
    """
    from scipy.optimize import brentq

    f = lambda x: (P.eps_min(x) + delta) - P.eps_merton(x)
    xs = np.linspace(*F0_range, n_scan)
    fs = np.array([f(x) for x in xs])
    sign_changes = np.where(np.diff(np.sign(fs)))[0]
    if len(sign_changes) < 2:
        return (np.nan, np.nan)
    lo = brentq(f, xs[sign_changes[0]], xs[sign_changes[0] + 1], xtol=1e-12)
    hi = brentq(f, xs[sign_changes[-1]], xs[sign_changes[-1] + 1], xtol=1e-12)
    return float(lo), float(hi)


def plot_slack_grid(F0_range=(0.5, 2.0), n_points=400, deltas=(0.01, 0.02, 0.05),
                    save_path=None):
    """Mode A-2 for several slack levels delta (delta=0.05 is the ordered one)."""
    F0_vals = np.linspace(*F0_range, n_points)
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(deltas)))
    for d, col in zip(deltas, cols):
        lo, hi = slack_binding_range(d)
        ax.plot(F0_vals, es_curve_slack(F0_vals, d), color=col, lw=2.2,
                label=rf'$\delta$={d} (binds on {lo:.3f}–{hi:.3f})')
    add_merton_hline(ax, 1.0, 'Merton ($A=1$)')
    add_k_vline(ax)
    ax.set_xlabel('Initial funding ratio $F_0$')
    ax.set_ylabel('$A_{ES}(0,Y_0)$')
    ax.set_title(r'Mode A-2: sensitivity to the slack $\delta$')
    ax.text(0.02, 0.03,
            r'outside the quoted band $\varepsilon_{\min}(F_0)+\delta \geq '
            r'\varepsilon_M(F_0)$:' '\n' r'the constraint is slack, so $A\equiv1$',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(fc='0.85', ec='0.4', alpha=0.9))
    ax.legend(**LEGEND)
    setup_grid(ax)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


def plot_eps_sensitivity(eps_list=None, F0_range=(0.9, 2.0), n_points=400,
                         save_path=None):
    """Cross-sectional A_ES for several eps inside the feasible band."""
    if eps_list is None:
        eps_list = P.EPS_GRID
    F0_vals = np.linspace(*F0_range, n_points)

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    ax.plot(F0_vals, [_A_var(f, P.alpha) for f in F0_vals],
            label=rf'VaR ($\alpha$={P.alpha})', **LINE_STYLES['VaR'])
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(eps_list)))
    for e, col in zip(eps_list, cols):
        ax.plot(F0_vals, es_curve_fixed_eps(F0_vals, e), color=col, lw=2,
                label=rf'ES ($\varepsilon$={e})')
    add_merton_hline(ax, 1.0, 'Merton ($A=1$)')
    add_k_vline(ax)
    ax.set_xlabel('Initial funding ratio $F_0$')
    ax.set_ylabel('Adjustment factor $A(0,Y_0)$')
    ax.set_title(r'Effect of the ES budget $\varepsilon$ (feasible band only)')
    ax.legend(**LEGEND, ncol=2)
    setup_grid(ax)
    ax.set_xlim(F0_range)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════
# Mode B — fixed claim
# ═══════════════════════════════════════════════════════════

def plot_fixed_claim_A(sol_es, sol_var=None, y_range=(0.2, 2.5), n_points=600,
                       snapshots=(0.0, 2.5, 5.0, 7.5), save_path=None):
    """A(t,y) for the claim fixed at t=0, at several time snapshots."""
    y = np.linspace(*y_range, n_points)
    fig, axes = plt.subplots(1, 2 if sol_var else 1,
                             figsize=(15, 6) if sol_var else FIGSIZES['single'],
                             squeeze=False)
    axes = axes[0]

    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(snapshots)))
    for t, col in zip(snapshots, cols):
        tau = P.T - t
        axes[0].plot(y, ES.adjustment_factor(y, sol_es['k_eps'], sol_es['c'], tau),
                     color=col, lw=2.2, label=f'$t$={t:g}')
    add_merton_hline(axes[0], 1.0, 'Merton ($A=1$)')
    axes[0].axvline(sol_es['k_eps'], color=COLORS['ES'], ls=':', alpha=0.6,
                    label=rf"$k_\varepsilon$={sol_es['k_eps']:.3f}")
    axes[0].axvline(sol_es['Y0'], color='0.3', ls='-.', alpha=0.6,
                    label=rf"$Y_0$={sol_es['Y0']:.3f}")
    add_k_vline(axes[0])
    axes[0].set_xlabel('Reference state $y$')
    axes[0].set_ylabel('$A_{ES}(t,y)$')
    axes[0].set_title('ES: fixed-claim exposure')
    axes[0].legend(**LEGEND, ncol=2)
    setup_grid(axes[0])
    axes[0].set_ylim(0, 1.15)

    if sol_var:
        for t, col in zip(snapshots, cols):
            tau = P.T - t
            axes[1].plot(y, VaR.adjustment_factor(y, sol_var['k_alpha'], tau),
                         color=col, lw=2.2, label=f'$t$={t:g}')
        add_merton_hline(axes[1], 1.0, 'Merton ($A=1$)')
        axes[1].axvline(sol_var['k_alpha'], color=COLORS['VaR'], ls=':',
                        alpha=0.6, label=rf"$k_\alpha$={sol_var['k_alpha']:.3f}")
        add_k_vline(axes[1])
        axes[1].set_xlabel('Reference state $y$')
        axes[1].set_ylabel('$A_{VaR}(t,y)$')
        axes[1].set_title('VaR: fixed-claim exposure (gambling region $A>1$)')
        axes[1].axhspan(1.0, 2.0, **{'color': COLORS['ES'], 'alpha': 0.10})
        axes[1].legend(**LEGEND, ncol=2)
        setup_grid(axes[1])
        axes[1].set_ylim(0, 1.6)

    plt.suptitle(rf"Fixed claim at $F_0$={sol_es['F0']}, "
                 rf"$\varepsilon$={sol_es['eps']}: exposure vs. reference state",
                 fontsize=13)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


def plot_fixed_claim_F(sol_es, sol_var=None, y_range=(0.2, 2.5), n_points=600,
                       snapshots=(0.0, 2.5, 5.0, 7.5), save_path=None):
    """Auxiliary axis: A against the funding ratio F = Psi(t,y)."""
    y = np.linspace(*y_range, n_points)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(snapshots)))
    for t, col in zip(snapshots, cols):
        tau = P.T - t
        F = ES.psi(y, sol_es['k_eps'], sol_es['c'], tau)
        A = ES.adjustment_factor(y, sol_es['k_eps'], sol_es['c'], tau)
        axes[0].plot(F, A, color=col, lw=2.2, label=f'$t$={t:g}')
        axes[1].plot(y, F, color=col, lw=2.2, label=f'$t$={t:g}')

    add_merton_hline(axes[0], 1.0, 'Merton ($A=1$)')
    add_k_vline(axes[0])
    axes[0].set_xlabel('Funding ratio $F=\\Psi(t,y)$')
    axes[0].set_ylabel('$A_{ES}$')
    axes[0].set_title('ES exposure vs. funding ratio')
    axes[0].legend(**LEGEND)
    setup_grid(axes[0])
    axes[0].set_xlim(0, 2.5)
    axes[0].set_ylim(0, 1.15)

    axes[1].plot(y, ES.claim(y, sol_es['k_eps'], sol_es['c']), color='k',
                 ls='--', lw=1.5, label='payoff $g_{ES}(y)$')
    axes[1].plot(y, y, color='0.5', ls=':', lw=1.5, label='45°')
    axes[1].set_xlabel('Reference state $y$')
    axes[1].set_ylabel('$\\Psi_{ES}(t,y)$')
    axes[1].set_title('Present value of the fixed claim')
    axes[1].legend(**LEGEND)
    setup_grid(axes[1])

    plt.suptitle(rf"Fixed claim at $F_0$={sol_es['F0']}, "
                 rf"$\varepsilon$={sol_es['eps']}", fontsize=13)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


def plot_claim_functions(sol_es, sol_var, y_range=(0.2, 1.8), n_points=800,
                         save_path=None):
    """The two claim functions side by side — the core economic contrast."""
    y = np.linspace(*y_range, n_points)
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    ax.plot(y, ES.claim(y, sol_es['k_eps'], sol_es['c']),
            label=rf"$g_{{ES}}$ ($c$={sol_es['c']:.3f})", **LINE_STYLES['ES'])
    ax.plot(y, VaR.claim(y, sol_var['k_alpha']),
            label=rf"$g_{{VaR}}$ ($k_\alpha$={sol_var['k_alpha']:.3f})",
            **LINE_STYLES['VaR'])
    ax.plot(y, y, color='0.5', ls=':', lw=1.5, label='Merton $g(y)=y$')
    ax.axvline(sol_es['k_eps'], color=COLORS['ES'], ls=':', alpha=0.5)
    ax.axvline(sol_var['k_alpha'], color=COLORS['VaR'], ls=':', alpha=0.5)
    add_k_vline(ax)
    ax.set_xlabel('Reference state $y$')
    ax.set_ylabel('Terminal funding ratio $g(y)$')
    ax.set_title('Claim functions: partial linear protection vs. abandonment')
    ax.legend(**LEGEND)
    setup_grid(ax)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════
# Feasibility-floor figure (new proposition)
# ═══════════════════════════════════════════════════════════

def plot_eps_min_muI(mu_range=(0.01, 0.05), n_points=300, F0=None,
                     save_path=None):
    """eps_min(mu_I) — visualisation of the feasibility floor.

    beta_0, beta_1 fixed; mu_I moves r̃ = r - (beta_0 + beta_1·mu_I), hence
    the discounted target k·e^{-r̃T} and therefore the floor.
    """
    if F0 is None:
        F0 = P.F0
    mus = np.linspace(*mu_range, n_points)
    floor, disc, merton = [], [], []
    for mu in mus:
        with P.override_params(MU_I=mu):
            floor.append(P.eps_min(F0))
            disc.append(P.k * np.exp(-P.r_tilde * P.T))
            merton.append(P.eps_merton(F0))

    floor, disc, merton = map(np.array, (floor, disc, merton))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(mus, disc, color=COLORS['ES'], lw=2.5,
             label=r'$k\,e^{-\tilde r(\mu_I) T}$')
    ax1.axhline(F0, color='0.3', ls='--', lw=1.5, label=f'$F_0$={F0}')
    cross = np.where(np.diff(np.sign(disc - F0)))[0]
    for i in cross:
        mu_c = np.interp(F0, disc[i:i + 2], mus[i:i + 2])
        ax1.plot(mu_c, F0, 'o', ms=9, color='k', zorder=5)
        ax1.annotate(rf'$\mu_I^*$={mu_c:.4f}', (mu_c, F0),
                     textcoords='offset points', xytext=(10, -18))
    ax1.axvline(P.MU_I, color='0.5', ls=':', lw=1.2,
                label=rf'baseline $\mu_I$={P.MU_I}')
    ax1.set_xlabel(r'Expected inflation $\mu_I$')
    ax1.set_ylabel('Discounted target')
    ax1.set_title(r'Where the target becomes unaffordable')
    ax1.legend(**LEGEND)
    setup_grid(ax1)

    ax2.plot(mus, floor, color=COLORS['ES'], lw=2.5,
             label=r'$\varepsilon_{\min}(\mu_I)$')
    ax2.plot(mus, merton, color=COLORS['VaR'], lw=2.0, ls='--',
             label=r'$\varepsilon_M(\mu_I) = \mathrm{Put}(F_0,k)$')
    ax2.fill_between(mus, floor, merton, where=merton > floor, alpha=0.15,
                     color=COLORS['ES'], label='feasible & binding band')
    ax2.axhline(P.epsilon, color='0.3', ls='--', lw=1.5,
                label=rf'baseline $\varepsilon$={P.epsilon}')
    ax2.axvline(P.MU_I, color='0.5', ls=':', lw=1.2)
    bad = np.where(floor >= P.epsilon)[0]
    if bad.size:
        mu_b = mus[bad[0]]
        ax2.axvspan(mu_b, mus[-1], color='0.6', alpha=0.25)
        ax2.annotate(rf'$\varepsilon$={P.epsilon} infeasible' '\n'
                     rf'for $\mu_I>{mu_b:.4f}$', (mu_b, P.epsilon),
                     textcoords='offset points', xytext=(8, 30))
    ax2.set_xlabel(r'Expected inflation $\mu_I$')
    ax2.set_ylabel('ES budget')
    ax2.set_title(r'Feasibility floor $\varepsilon_{\min}$ vs. slack bound $\varepsilon_M$')
    ax2.legend(**LEGEND)
    setup_grid(ax2)

    plt.suptitle(f'Feasibility floor of the ES constraint ($F_0$={F0}, '
                 rf'$\beta_0$={P.BETA_0}, $\beta_1$={P.BETA_1}, $T$={P.T})',
                 fontsize=13)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════
# Parameter sensitivity with feasibility flags
# ═══════════════════════════════════════════════════════════

SENS_CONFIGS = {
    'GAMMA': [2.0, 3.0, 5.0, 8.0],
    'MU_I':  [0.010, 0.015, 0.023, 0.030],
    'T':     [5.0, 10.0, 15.0, 20.0],
    'RHO':   [-0.5, -0.15, 0.0, 0.5],
}

# Code-style parameter names must never reach a figure: every title, legend
# entry and annotation uses the mathtext symbol.
PARAM_LABELS = {
    'GAMMA':  r'$\gamma$',
    'MU_I':   r'$\mu_I$',
    'T':      r'$T$',
    'RHO':    r'$\rho$',
    'BETA_0': r'$\beta_0$',
    'EPS':    r'$\varepsilon$',
    'DELTA_L': r'$\delta_L$',
}


def param_label(param):
    """Mathtext symbol for a parameter name (falls back to the raw name)."""
    return PARAM_LABELS.get(param, param)


def sensitivity_scan(param, values=None, F0=None, eps=None):
    """Recompute the joint solution for each parameter value.

    Returns a list of records including the RECOMPUTED eps_min / eps_M —
    these move with the parameter (dramatically so for mu_I and T), which
    is why a fixed baseline eps can become infeasible.
    """
    if values is None:
        values = SENS_CONFIGS[param]
    if F0 is None:
        F0 = P.F0
    if eps is None:
        eps = P.epsilon

    recs = []
    for v in values:
        with P.override_params(**{param: v}):
            e_lo, e_hi = P.eps_band(F0)
            s = ES.solve_es(F0, eps, strict=False)
            sv = VaR.solve_var(F0, P.alpha, strict=False)
            rec = dict(param=param, value=v, eps=eps, eps_min=e_lo, eps_M=e_hi,
                       r_tilde=P.r_tilde, sigma_Y=P.sigma_Y,
                       feasible=s['feasible'], binding=s['binding'],
                       Y0=s['Y0'], k_eps=s['k_eps'], c=s['c'],
                       var_feasible=sv['feasible'], var_Y0=sv['Y0'],
                       k_alpha=sv['k_alpha'], var_cost_min=sv['cost_min'])
            rec['A0'] = (float(ES.adjustment_factor(s['Y0'], s['k_eps'], s['c'], P.T))
                         if s['feasible'] and s['binding']
                         else (1.0 if s['feasible'] else np.nan))
            rec['A0_var'] = (float(VaR.adjustment_factor(sv['Y0'], sv['k_alpha'], P.T))
                             if sv['feasible'] and sv['binding']
                             else (1.0 if sv['feasible'] else np.nan))
        recs.append(rec)
    return recs


def plot_sensitivity_cross(param, values=None, F0_range=(0.9, 2.0),
                           n_points=300, eps=None, save_path=None):
    """Cross-sectional counterpart: A_ES(0,Y0) vs F0 for each parameter value.

    Each (F0, parameter) pair re-solves its own joint system; points where
    the fixed eps sits below that configuration's floor are dropped.
    """
    if eps is None:
        eps = P.epsilon
    values = values if values is not None else SENS_CONFIGS[param]
    F0_vals = np.linspace(*F0_range, n_points)

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(values)))
    notes = []
    for v, col in zip(values, cols):
        with P.override_params(**{param: v}):
            A = es_curve_fixed_eps(F0_vals, eps)
            lo = feasible_F0_min(eps)
        n_ok = int(np.sum(~np.isnan(A)))
        ax.plot(F0_vals, A, color=col, lw=2.2,
                label=f'{param_label(param)}={v:g} ($F_0>${lo:.3f})')
        if n_ok == 0:
            notes.append(f'{param_label(param)}={v:g}: infeasible over the '
                         f'whole range')
    add_merton_hline(ax, 1.0, 'Merton ($A=1$)')
    add_k_vline(ax)
    ax.set_xlabel('Initial funding ratio $F_0$')
    ax.set_ylabel('$A_{ES}(0,Y_0)$')
    ax.set_title(f'Cross-sectional sensitivity to {param_label(param)} '
                 rf'($\varepsilon$={eps})')
    if notes:
        ax.text(0.02, 0.02, '\n'.join(notes), transform=ax.transAxes,
                fontsize=10, va='bottom',
                bbox=dict(fc='0.85', ec='0.4', alpha=0.9))
    ax.legend(**LEGEND)
    setup_grid(ax)
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


def plot_sensitivity(param, values=None, F0=None, eps=None,
                     y_range=(0.2, 2.5), n_points=400, save_path=None):
    """A_ES(0,y) curves per parameter value; infeasible configs omitted+labelled."""
    recs = sensitivity_scan(param, values, F0, eps)
    y = np.linspace(*y_range, n_points)

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(recs)))
    skipped = []
    for rec, col in zip(recs, cols):
        if not rec['feasible']:
            skipped.append(rec)
            continue
        with P.override_params(**{param: rec['value']}):
            A = (ES.adjustment_factor(y, rec['k_eps'], rec['c'], P.T)
                 if rec['binding'] else np.ones_like(y))
        ax.plot(y, A, color=col, lw=2.2,
                label=f"{param_label(param)}={rec['value']:g} "
                      rf"($\varepsilon_{{\min}}$={rec['eps_min']:.3f})")
    add_merton_hline(ax, 1.0, 'Merton ($A=1$)')
    add_k_vline(ax)
    ax.set_xlabel('Reference state $y$')
    ax.set_ylabel('$A_{ES}(0,y)$')
    ax.set_title(f'Sensitivity to {param_label(param)} '
                 rf'($F_0$={F0 or P.F0}, $\varepsilon$={eps or P.epsilon})')
    if skipped:
        txt = '\n'.join(f"{param_label(param)}={r['value']:g}: INFEASIBLE "
                        rf"($\varepsilon_{{\min}}$={r['eps_min']:.4f}"
                        rf"$\geq\varepsilon$)" for r in skipped)
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=10,
                va='bottom', bbox=dict(fc='0.85', ec='0.4', alpha=0.9))
    ax.legend(**LEGEND)
    setup_grid(ax)
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig, recs
