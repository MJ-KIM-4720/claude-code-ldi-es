"""
Manuscript-referenced sensitivity figures
==========================================
Regenerates the three figures the manuscript cites, keeping the composition
of the existing ones (2x2 panels, one per parameter value; ES vs VaR series;
Merton reference line; shaded VaR gambling region) and writing them straight
to paper/figures/ under the SAME file names:

    fig_A3_gamma_A_factor.png   adjustment factor A, one panel per gamma
    fig_C2_muI_compare.png      total risky allocation, one panel per mu_I
    fig_D2_T_compare.png        total risky allocation, one panel per T

Two things differ from the files these replace, both unavoidable:

  * The curves are recomputed with the corrected joint-system / fixed-claim
    model. The originals came from the discarded single-equation solver at
    the infeasible baseline eps = 0.05 (and a gamma = 7 panel that is not in
    the current grid).
  * The x-axis is the reference state y of ONE fund whose claim is fixed at
    t = 0, not a cross-section of funds indexed by their funding ratio.
    A(t,y) is by definition the delta of a claim fixed at t=0, so re-solving
    the threshold at every x — which is what "A(F) across funding ratios"
    used to mean — prices a different claim at every point. The Mode A
    counterparts (x = F0, one fund per point) live in
    outputs/cross_sectional/sens_*.png if the manuscript prefers them.

Every title, panel label and legend entry is mathtext.

Run:  python3 scripts/run_paper_figures.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldi import params as P
from ldi import es_model as ES
from ldi import var_model as VaR
from ldi import compare as C
from ldi.style import (apply_paper_style, COLORS, PAPER_LINE_STYLES, LEGEND,
                       FIGSIZES, PAPER_GAMBLING, paper_grid, paper_hline,
                       paper_savefig)

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "paper", "figures")

Y_RANGE = (0.5, 1.3)          # same window as the figures being replaced
N_POINTS = 400


def _curves(y):
    """(A_ES, A_VaR) for the current parameter state; NaN where infeasible."""
    s = ES.solve_es(strict=False)
    sv = VaR.solve_var(strict=False)

    if not s['feasible']:
        a_es = np.full_like(y, np.nan)
    elif not s['binding']:
        a_es = np.ones_like(y)
    else:
        a_es = np.asarray(ES.adjustment_factor(y, s['k_eps'], s['c'], P.T))

    if not sv['feasible']:
        a_var = np.full_like(y, np.nan)
    elif not sv['binding']:
        a_var = np.ones_like(y)
    else:
        a_var = np.asarray(VaR.adjustment_factor(y, sv['k_alpha'], P.T))
    return a_es, a_var, s, sv


def _panel(ax, y, a_es, a_var, s, scale, ylabel, merton_total, band):
    es_c = a_es * scale
    var_c = a_var * scale
    ax.plot(y, es_c, label='ES', **PAPER_LINE_STYLES['ES'])
    ax.plot(y, var_c, label='VaR', **PAPER_LINE_STYLES['VaR'])
    ref = 1.0 * scale if scale == 1.0 else merton_total
    paper_hline(ax, ref, 'Merton' if scale != 1.0 else '$A = 1$ (Merton)')

    gambling = var_c > ref
    if np.any(gambling):
        ax.fill_between(y, ref, var_c, where=gambling,
                        label='VaR gambling', **PAPER_GAMBLING)
    if not s['feasible']:
        # band = (eps_min, eps_M) of THIS panel's parameter state, captured
        # inside the override block — P.eps_min() here would report baseline.
        ax.text(0.5, 0.30, 'ES infeasible\n'
                           rf'($\varepsilon_{{\min}}$={band[0]:.4f}'
                           rf'$\,>\,\varepsilon$={P.epsilon:g})',
                transform=ax.transAxes, ha='center', va='center', fontsize=12,
                bbox=dict(fc='0.9', ec='0.4', alpha=0.95))
    ax.set_xlabel('Reference state $y$')
    ax.set_ylabel(ylabel)
    ax.legend(**LEGEND)
    paper_grid(ax)
    ax.set_xlim(Y_RANGE)


def figure(param, values, kind, suptitle, fname):
    """2x2 comparison panel; kind = 'A' (adjustment factor) or 'alloc'."""
    y = np.linspace(*Y_RANGE, N_POINTS)
    sym = C.param_label(param)
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZES['quad'])

    for ax, v in zip(axes.flat, values):
        with P.override_params(**{param: v}):
            a_es, a_var, s, sv = _curves(y)
            merton_total = float(P.Pi_star.sum())
            band = P.eps_band()
        scale = 1.0 if kind == 'A' else merton_total
        ylabel = ('Adjustment factor $A(y)$' if kind == 'A'
                  else r'Total risky allocation $\pi_S + \pi_I$')
        _panel(ax, y, a_es, a_var, s, scale, ylabel, merton_total, band)
        fmt = f'{v:g}' if param != 'MU_I' else f'{v:.3f}'
        ax.set_title(f'{sym} $= {fmt}$')

    fig.suptitle(suptitle)
    plt.tight_layout()
    path = os.path.join(FIG, fname)
    paper_savefig(fig, path)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


def main():
    apply_paper_style()
    os.makedirs(FIG, exist_ok=True)
    print(f"Manuscript figures (F0={P.F0}, eps={P.epsilon}, alpha={P.alpha}, "
          f"claim fixed at t=0)")

    figure('GAMMA', C.SENS_CONFIGS['GAMMA'], 'A',
           r'Adjustment factor $A(y)$: ES vs VaR by risk aversion $\gamma$',
           'fig_A3_gamma_A_factor.png')

    figure('MU_I', C.SENS_CONFIGS['MU_I'], 'alloc',
           r'ES vs VaR: total risky allocation by expected inflation $\mu_I$',
           'fig_C2_muI_compare.png')

    figure('T', C.SENS_CONFIGS['T'], 'alloc',
           r'ES vs VaR: total risky allocation by investment horizon $T$',
           'fig_D2_T_compare.png')


if __name__ == "__main__":
    main()
