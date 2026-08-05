"""
Manuscript-referenced sensitivity figures
==========================================
Writes the three figures the manuscript cites straight to paper/figures/:

    fig_A3_gamma_A_factor.png   A_ES overlaid by gamma
    fig_C2_muI_compare.png      A_ES overlaid by mu_I
    fig_D2_T_compare.png        A_ES overlaid by T

COMPOSITION (fixed — do not redesign):
  * a SINGLE panel per file;
  * ES only: one A_ES(0,y) curve per parameter value, overlaid;
  * dotted Merton reference at A = 1;
  * a slack configuration is drawn as the constant curve A == 1, since the
    unconstrained claim already complies;
  * an infeasible configuration gets NO curve, only an annotation box;
  * y (the reference state) spans 0.2 - 2.5.
Only the titles and legend entries are mathtext ($\\gamma$, $\\mu_I$, $T$)
rather than code-style parameter names.

The 2x2 ES-vs-VaR panels are NOT part of these files; they are written to
outputs/alt/ as candidates for an appendix figure.

One label differs from the pre-2026-08 files: the horizontal axis is
"Reference state y", not "Funding Ratio F(t)". A(t,y) is the delta of a
claim fixed at t=0, so the curve is indexed by the reference state; the old
axis re-solved the threshold at every point, which prices a different claim
at each x. The cross-sectional object (x = F0, one fund per point) is in
outputs/cross_sectional/sens_*.png.

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
from ldi.style import (apply_paper_style, WARM_PALETTE, PAPER_LINE_STYLES,
                       LEGEND, FIGSIZES, PAPER_GAMBLING, paper_grid,
                       paper_hline, paper_savefig)

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "paper", "figures")
ALT = os.path.join(ROOT, "outputs", "alt")

Y_RANGE = (0.2, 2.5)
N_POINTS = 500


def _a_es(y):
    """A_ES(0,y) for the current parameter state, plus the solution dict.

    Returns (curve or None, solution). A slack configuration returns the
    constant-1 curve; an infeasible one returns None.
    """
    s = ES.solve_es(strict=False)
    if not s['feasible']:
        return None, s
    if not s['binding']:
        return np.ones_like(y), s
    return np.asarray(ES.adjustment_factor(y, s['k_eps'], s['c'], P.T)), s


def _a_var(y):
    sv = VaR.solve_var(strict=False)
    if not sv['feasible']:
        return None, sv
    if not sv['binding']:
        return np.ones_like(y), sv
    return np.asarray(VaR.adjustment_factor(y, sv['k_alpha'], P.T)), sv


def _fmt(param, v):
    return f'{v:.3f}' if param == 'MU_I' else f'{v:g}'


# ═══════════════════════════════════════════════════════════
# The manuscript figures: single panel, ES only
# ═══════════════════════════════════════════════════════════

def es_overlay(param, values, title, fname):
    y = np.linspace(*Y_RANGE, N_POINTS)
    sym = C.param_label(param)
    colors = WARM_PALETTE[:len(values)]

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    skipped = []
    for v, col in zip(values, colors):
        with P.override_params(**{param: v}):
            curve, s = _a_es(y)
            band = P.eps_band()
        if curve is None:
            skipped.append((v, band[0]))
            continue
        ax.plot(y, curve, color=col, lw=2.0,
                label=f'{sym} $= {_fmt(param, v)}$')

    paper_hline(ax, 1.0, '$A = 1$ (Merton)')
    if skipped:
        txt = '\n'.join(
            f'{sym} $= {_fmt(param, v)}$: infeasible '
            rf'($\varepsilon_{{\min}} = {lo:.4f} > \varepsilon = {P.epsilon:g}$)'
            for v, lo in skipped)
        ax.text(0.98, 0.04, txt, transform=ax.transAxes, fontsize=10,
                ha='right', va='bottom',
                bbox=dict(fc='0.92', ec='0.4', alpha=0.95))

    ax.set_xlabel('Reference state $y$')
    ax.set_ylabel('Adjustment factor $A_{\\mathrm{ES}}(0,y)$')
    ax.set_title(title)
    ax.legend(**LEGEND)
    paper_grid(ax)
    ax.set_xlim(Y_RANGE)
    ax.set_ylim(0, 1.08)
    plt.tight_layout()
    path = os.path.join(FIG, fname)
    paper_savefig(fig, path)
    print(f"  wrote {os.path.relpath(path, ROOT)}"
          + (f"   [infeasible: {', '.join(_fmt(param, v) for v, _ in skipped)}]"
             if skipped else ""))


# ═══════════════════════════════════════════════════════════
# Appendix candidates: 2x2 ES vs VaR panels  ->  outputs/alt/
# ═══════════════════════════════════════════════════════════

def es_var_panels(param, values, kind, suptitle, fname):
    y = np.linspace(0.5, 1.3, 400)
    sym = C.param_label(param)
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZES['quad'])

    for ax, v in zip(axes.flat, values):
        with P.override_params(**{param: v}):
            a_es, s = _a_es(y)
            a_var, sv = _a_var(y)
            merton_total = float(P.Pi_star.sum())
            band = P.eps_band()
        scale = 1.0 if kind == 'A' else merton_total
        ref = 1.0 * scale
        if a_es is not None:
            ax.plot(y, a_es * scale, label='ES', **PAPER_LINE_STYLES['ES'])
        if a_var is not None:
            var_c = a_var * scale
            ax.plot(y, var_c, label='VaR', **PAPER_LINE_STYLES['VaR'])
            if np.any(var_c > ref):
                ax.fill_between(y, ref, var_c, where=var_c > ref,
                                label='VaR gambling', **PAPER_GAMBLING)
        paper_hline(ax, ref, '$A = 1$ (Merton)' if kind == 'A' else 'Merton')
        if a_es is None:
            ax.text(0.5, 0.30, 'ES infeasible\n'
                               rf'($\varepsilon_{{\min}}$={band[0]:.4f}'
                               rf'$\,>\,\varepsilon$={P.epsilon:g})',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, bbox=dict(fc='0.9', ec='0.4', alpha=0.95))
        ax.set_title(f'{sym} $= {_fmt(param, v)}$')
        ax.set_xlabel('Reference state $y$')
        ax.set_ylabel('Adjustment factor $A(y)$' if kind == 'A'
                      else r'Total risky allocation $\pi_S + \pi_I$')
        ax.legend(**LEGEND)
        paper_grid(ax)
        ax.set_xlim(0.5, 1.3)

    fig.suptitle(suptitle)
    plt.tight_layout()
    path = os.path.join(ALT, fname)
    paper_savefig(fig, path)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


def main():
    apply_paper_style()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(ALT, exist_ok=True)
    print(f"Manuscript figures (F0={P.F0}, eps={P.epsilon}, claim fixed at t=0)")

    es_overlay('GAMMA', C.SENS_CONFIGS['GAMMA'],
               r'ES Constraint: Effect of Risk Aversion ($\gamma$)',
               'fig_A3_gamma_A_factor.png')
    es_overlay('MU_I', C.SENS_CONFIGS['MU_I'],
               r'ES Constraint: Effect of Expected Inflation ($\mu_I$)',
               'fig_C2_muI_compare.png')
    es_overlay('T', C.SENS_CONFIGS['T'],
               r'ES Constraint: Effect of Investment Horizon ($T$)',
               'fig_D2_T_compare.png')

    print("Appendix candidates (ES vs VaR panels)")
    es_var_panels('GAMMA', C.SENS_CONFIGS['GAMMA'], 'A',
                  r'Adjustment factor $A(y)$: ES vs VaR by risk aversion $\gamma$',
                  'alt_gamma_A_factor_panels.png')
    es_var_panels('MU_I', C.SENS_CONFIGS['MU_I'], 'alloc',
                  r'ES vs VaR: total risky allocation by expected inflation $\mu_I$',
                  'alt_muI_compare_panels.png')
    es_var_panels('T', C.SENS_CONFIGS['T'], 'alloc',
                  r'ES vs VaR: total risky allocation by investment horizon $T$',
                  'alt_T_compare_panels.png')


if __name__ == "__main__":
    main()
