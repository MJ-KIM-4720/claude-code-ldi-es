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
from ldi import exact_stats as X
from ldi import simulate as SIM
from ldi.style import (apply_paper_style, WARM_PALETTE, PAPER_LINE_STYLES,
                       COLORS, LEGEND, FIGSIZES, PAPER_GAMBLING, K_LINE,
                       paper_grid, paper_hline, paper_savefig)

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
# Baseline figures
# ═══════════════════════════════════════════════════════════

def claim_function(fname='fig_baseline_claim_function.png'):
    """g_ES vs g_VaR at the current calibration, with both thresholds."""
    s, sv = ES.solve_es(), VaR.solve_var()
    ke, c, ka = s['k_eps'], s['c'], sv['k_alpha']
    y = np.linspace(0.0, 1.5, 1500)

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    ax.plot(y, ES.claim(y, ke, c), label=rf'$g_{{ES}}$  ($c$={c:.3f})',
            **PAPER_LINE_STYLES['ES'])

    # g_VaR jumps at k_alpha: draw the branches separately, then mark the jump
    lo, mid = y[y < ka], y[(y >= ka) & (y < P.k)]
    ax.plot(lo, lo, **PAPER_LINE_STYLES['VaR'], label=r'$g_{VaR}$')
    ax.plot(mid, np.full_like(mid, P.k), **PAPER_LINE_STYLES['VaR'])
    hi = y[y >= P.k]
    ax.plot(hi, hi, **PAPER_LINE_STYLES['VaR'])
    ax.plot([ka, ka], [ka, P.k], color=COLORS['VaR'], ls='--', lw=1.2,
            alpha=0.55)
    ax.plot([ka], [ka], marker='o', ms=6, mfc='white', mec=COLORS['VaR'],
            mew=1.6, zorder=5)
    ax.plot([ka], [P.k], marker='o', ms=6, color=COLORS['VaR'], zorder=5)
    ax.annotate(rf'jump at $k_\alpha$: $g_{{VaR}}\!\to\!k$',
                xy=(ka, (ka + P.k) / 2), xytext=(10, -6),
                textcoords='offset points', fontsize=10, color=COLORS['VaR'])

    ax.plot(y, y, color='0.45', ls=':', lw=1.5, label=r'45$^\circ$  ($g(y)=y$)')
    ax.axvline(P.k, **K_LINE)
    ax.axvline(ke, color=COLORS['ES'], ls=':', lw=1.2, alpha=0.7)
    ax.axvline(ka, color=COLORS['VaR'], ls=':', lw=1.2, alpha=0.7)
    ax.annotate(rf'$k_\varepsilon$ = {ke:.3f}', xy=(ke, 0.42), xytext=(1.02, 0.42),
                fontsize=11, color=COLORS['ES'], va='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['ES'], lw=1.1))
    ax.annotate(rf'$k_\alpha$ = {ka:.3f}', xy=(ka, 0.24), xytext=(1.02, 0.24),
                fontsize=11, color=COLORS['VaR'], va='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['VaR'], lw=1.1))
    ax.annotate(rf'$k$ = {P.k:g}', xy=(P.k, 0.06), xytext=(6, 0),
                textcoords='offset points', rotation=90, va='bottom',
                fontsize=11, color='green')

    ax.set_xlabel('Reference state $y$')
    ax.set_ylabel('Terminal funding ratio $g(y)$')
    ax.set_title('Claim functions: partial protection vs. abandonment')
    ax.legend(**LEGEND)
    paper_grid(ax)
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(0.0, 1.5)
    plt.tight_layout()
    path = os.path.join(FIG, fname)
    paper_savefig(fig, path)
    print(f"  wrote {os.path.relpath(path, ROOT)}"
          f"   [k_eps={ke:.6f}, k_alpha={ka:.6f}, c={c:.6f}]")


SNAPSHOTS = (0.0, 2.5, 5.0, 7.5)


def adjustment_factor(fname='fig_baseline_adjustment_factor.png'):
    """A_ES and A_VaR for the claim FIXED at t=0, at four time snapshots."""
    s, sv = ES.solve_es(), VaR.solve_var()
    y = np.linspace(*Y_RANGE, 700)
    reds = plt.cm.Reds(np.linspace(0.45, 0.95, len(SNAPSHOTS)))
    blues = plt.cm.Blues(np.linspace(0.45, 0.95, len(SNAPSHOTS)))

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    # ES curves first, then VaR, so the two-column legend reads as one
    # column per constraint rather than interleaving them.
    for t, rc in zip(SNAPSHOTS, reds):
        ax.plot(y, np.asarray(ES.adjustment_factor(y, s['k_eps'], s['c'],
                                                   P.T - t)),
                color=rc, lw=2.0, label=rf'ES, $t$={t:g}')
    var_max = np.full_like(y, 1.0)
    for t, bc in zip(SNAPSHOTS, blues):
        a_var = np.asarray(VaR.adjustment_factor(y, sv['k_alpha'], P.T - t))
        ax.plot(y, a_var, color=bc, lw=2.0, ls='--', label=rf'VaR, $t$={t:g}')
        var_max = np.maximum(var_max, a_var)

    ax.fill_between(y, 1.0, var_max, where=var_max > 1.0, color=COLORS['ES'],
                    alpha=0.12, label=r'VaR gambling ($A>1$)')
    paper_hline(ax, 1.0, '$A = 1$ (Merton)')
    ax.axvline(s['k_eps'], color=COLORS['ES'], ls=':', lw=1.0, alpha=0.6)
    ax.axvline(sv['k_alpha'], color=COLORS['VaR'], ls=':', lw=1.0, alpha=0.6)

    ax.set_xlabel('Reference state $y$')
    ax.set_ylabel('Adjustment factor $A(t,y)$')
    ax.set_title('Fixed-claim exposure: ES returns to Merton in both tails')
    ax.legend(**LEGEND, ncol=2, fontsize=10)
    paper_grid(ax)
    ax.set_xlim(Y_RANGE)
    ax.set_ylim(0.0, 1.75)
    plt.tight_layout()
    path = os.path.join(FIG, fname)
    paper_savefig(fig, path)
    a_es0 = float(ES.adjustment_factor(s['Y0'], s['k_eps'], s['c'], P.T))
    a_var0 = float(VaR.adjustment_factor(sv['Y0'], sv['k_alpha'], P.T))
    print(f"  wrote {os.path.relpath(path, ROOT)}"
          f"   [A_ES(0,{s['Y0']:.4f})={a_es0:.4f}, "
          f"A_VaR(0,{sv['Y0']:.4f})={a_var0:.4f}]")


def feasibility_map(fname='fig_feasibility_map.png'):
    """(F0, eps) phase diagram: Infeasible / Binding / Slack."""
    F0 = np.linspace(0.7, 1.3, 800)
    e_lo = np.array([P.eps_min(f) for f in F0])
    e_hi = np.array([P.eps_merton(f) for f in F0])
    y_top = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    ax.fill_between(F0, 0.0, e_lo, color='0.55', alpha=0.30)
    ax.fill_between(F0, e_lo, e_hi, color=COLORS['ES'], alpha=0.16)
    ax.fill_between(F0, e_hi, y_top, color=COLORS['Merton'], alpha=0.12)

    ax.plot(F0, e_lo, color=COLORS['ES'], lw=2.5,
            label=r'$\varepsilon_{\min}(F_0) = (k e^{-\tilde r T} - F_0)^{+}$')
    ax.plot(F0, e_hi, color=COLORS['VaR'], lw=2.0, ls='--',
            label=r'$\varepsilon_M(F_0) = \mathrm{Put}(0,F_0,\tilde r,\sigma_Y,k)$')

    kink = P.k * np.exp(-P.r_tilde * P.T)
    if F0[0] <= kink <= F0[-1]:
        ax.plot([kink], [0.0], marker='v', ms=8, color=COLORS['ES'], zorder=5)
        ax.annotate(rf'kink at $F_0 = k e^{{-\tilde r T}}$ = {kink:.4f}',
                    xy=(kink, 0.0), xytext=(-8, 14),
                    textcoords='offset points', ha='right', fontsize=10)

    ax.text(0.76, 0.055, 'Infeasible', fontsize=13, weight='bold',
            color='0.25')
    ax.text(1.02, 0.115, 'Binding', fontsize=13, weight='bold',
            color=COLORS['ES'])
    ax.text(0.95, 0.305, 'Slack (Merton)', fontsize=13, weight='bold',
            color='0.35')

    ax.plot([P.F0], [P.epsilon], marker='*', ms=17, color='k', zorder=6,
            label=rf'baseline $(F_0,\varepsilon)$ = ({P.F0:g}, {P.epsilon:g})')

    f_mark = 0.8
    ax.axvline(f_mark, color='0.25', ls=':', lw=1.4)
    ax.annotate(rf'$F_0$={f_mark:g}: $\varepsilon_{{\min}}$ = {P.eps_min(f_mark):.3f}',
                xy=(f_mark, P.eps_min(f_mark)), xytext=(8, 10),
                textcoords='offset points', fontsize=11,
                bbox=dict(fc='white', ec='0.5', alpha=0.9))

    ax.set_xlabel('Initial funding ratio $F_0$')
    ax.set_ylabel(r'ES budget $\varepsilon$')
    ax.set_title('Feasibility map of the ES constraint')
    ax.legend(loc='upper right', framealpha=0.92, edgecolor='gray',
              fontsize=10)
    paper_grid(ax)
    ax.set_xlim(0.7, 1.3)
    ax.set_ylim(0.0, y_top)
    plt.tight_layout()
    path = os.path.join(FIG, fname)
    paper_savefig(fig, path)
    print(f"  wrote {os.path.relpath(path, ROOT)}"
          f"   [eps_min(1.0)={P.eps_min(1.0):.4f}, "
          f"eps_M(1.0)={P.eps_merton(1.0):.4f}, "
          f"eps_min(0.8)={P.eps_min(0.8):.4f}]")



# ═══════════════════════════════════════════════════════════
# Figure 5 — terminal CDF at the EQUAL-CE calibration
# ═══════════════════════════════════════════════════════════

TAIL_Q = 0.05


def _ecdf(x_sorted, grid):
    return np.searchsorted(x_sorted, grid, side='right') / len(x_sorted)


def terminal_cdf(fname='mc_terminal_y010.png', n=SIM.DEFAULT_N_TERMINAL,
                 seed=SIM.DEFAULT_SEED):
    """Two stacked panels: full CDF (atom) and left tail (Q5 / bottom-5%).

    The VaR series is the EQUAL-CE calibration, so the two constrained
    strategies carry the same certainty-equivalent cost and the tail
    comparison is like-for-like.
    """
    s = ES.solve_es()
    a_eq = X.match_alpha_equal_ce()['alpha']
    sv = VaR.solve_var(alpha=a_eq)

    specs = {'Merton': (P.F0, 1.0, P.k),
             'ES': (s['Y0'], s['c'], s['k_eps']),
             'VaR': (sv['Y0'], 1.0, sv['k_alpha'])}
    smp = SIM.terminal_sample(specs, n=n, seed=seed)
    F = {kk: smp[kk] for kk in specs}
    srt = {kk: np.sort(v) for kk, v in F.items()}
    n_tail = int(round(TAIL_Q * n))
    q5 = {kk: float(srt[kk][n_tail - 1]) for kk in specs}
    bot5 = {kk: float(srt[kk][:n_tail].mean()) for kk in specs}

    labels = {'Merton': 'Merton',
              'ES': rf"ES ($\varepsilon$={s['eps']:g})",
              'VaR': rf'VaR equal-CE ($\alpha$={a_eq:.4f})'}

    big = {'axes.labelsize': 17, 'axes.titlesize': 18, 'legend.fontsize': 14,
           'xtick.labelsize': 15, 'ytick.labelsize': 15}
    with plt.rc_context(big):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 13))

        # ── top: full CDF with the atom jumps ──
        grid = np.unique(np.concatenate(
            [np.linspace(0.4, 1.8, 1600), [P.k - 1e-12, P.k, P.k + 1e-12]]))
        for kk in specs:
            ax.plot(grid, _ecdf(srt[kk], grid), label=labels[kk],
                    **PAPER_LINE_STYLES[kk if kk != 'VaR' else 'VaR'])
        ax.axvline(P.k, **K_LINE)
        ax.annotate(rf'$k$ = {P.k:g}', xy=(P.k, 0.90), xytext=(6, 0),
                    textcoords='offset points', rotation=90, va='bottom',
                    fontsize=13, color='green')

        for kk, dx in (('ES', 0.06), ('VaR', 0.06)):
            lo = _ecdf(srt[kk], np.array([P.k - 1e-12]))[0]
            hi = _ecdf(srt[kk], np.array([P.k]))[0]
            ax.annotate('', xy=(P.k, hi), xytext=(P.k, lo),
                        arrowprops=dict(arrowstyle='<->', color=COLORS[kk],
                                        lw=2.2, shrinkA=0, shrinkB=0))
            ax.annotate(rf'{kk}: $P(F_T\!=\!k)$ = {hi - lo:.3f}',
                        xy=(P.k, (lo + hi) / 2), xytext=(dx * 100, -6),
                        textcoords='offset points', fontsize=14,
                        color=COLORS[kk])

        # VaR's flat stretch: no mass between k_alpha and k
        ka = sv['k_alpha']
        ax.annotate('', xy=(ka, a_eq), xytext=(P.k, a_eq),
                    arrowprops=dict(arrowstyle='<->', color=COLORS['VaR'],
                                    lw=1.8, shrinkA=0, shrinkB=0))
        ax.annotate(rf'VaR: no mass on $({ka:.3f},\,{P.k:g})$,'
                    '\n' rf'plateau height $\alpha$ = {a_eq:.4f}',
                    xy=((ka + P.k) / 2, a_eq), xytext=(0.47, 0.30),
                    ha='left', va='center', fontsize=13,
                    color=COLORS['VaR'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['VaR'],
                                    lw=1.2))
        ax.set_xlim(0.4, 1.8)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('Terminal funding ratio $F_T$')
        ax.set_ylabel('$P(F_T \\leq x)$')
        ax.set_title('Empirical CDF with the probability atom at $k$')
        ax.legend(loc='lower right', framealpha=0.92, edgecolor='gray')
        paper_grid(ax)

        # ── bottom: left tail ──
        gridL = np.linspace(0.4, 0.9, 900)
        for kk in specs:
            ax2.plot(gridL, _ecdf(srt[kk], gridL), label=labels[kk],
                     **PAPER_LINE_STYLES[kk])
        ax2.axhline(TAIL_Q, color='0.4', ls=':', lw=1.4)
        for kk in specs:
            ax2.plot(q5[kk], TAIL_Q, 'o', ms=11, color=COLORS[kk], zorder=6)
            ax2.axvline(bot5[kk], color=COLORS[kk], ls='--', lw=1.8, alpha=0.8)
            ax2.annotate(f'{bot5[kk]:.3f}', xy=(bot5[kk], 0.006),
                         xytext=(4, 0), textcoords='offset points',
                         rotation=90, va='bottom', fontsize=13,
                         color=COLORS[kk])
            ax2.annotate(f'{q5[kk]:.3f}', xy=(q5[kk], TAIL_Q), xytext=(0, 12),
                         textcoords='offset points', ha='center', fontsize=13,
                         color=COLORS[kk])

        xs = np.linspace(0.45, 0.98, 3000)
        d = _ecdf(srt['ES'], xs) - _ecdf(srt['VaR'], xs)
        sc = np.where(np.diff(np.sign(d)))[0]
        cross = float(xs[sc[-1]]) if sc.size else float('nan')
        if np.isfinite(cross):
            yc = float(_ecdf(srt['ES'], np.array([cross]))[0])
            ax2.plot(cross, yc, marker='X', ms=13, color='k', zorder=7)
            ax2.annotate(rf'ES–VaR crossing at $F_T$ = {cross:.3f}',
                         xy=(cross, yc), xytext=(-14, 26),
                         textcoords='offset points', ha='right', fontsize=13,
                         arrowprops=dict(arrowstyle='->', color='0.3', lw=1.2))
        ax2.set_xlim(0.4, 0.9)
        ax2.set_xlabel('Terminal funding ratio $F_T$')
        ax2.set_ylabel('$P(F_T \\leq x)$')
        ax2.set_title('Left tail $F_T \\leq 0.9$ '
                      '(dots: $Q_5$, dashed: bottom-5% mean)')
        ax2.legend(loc='upper left', framealpha=0.92, edgecolor='gray')
        paper_grid(ax2)

        plt.tight_layout()
        path = os.path.join(FIG, fname)
        paper_savefig(fig, path)

    atom_var = SIM.atom_fraction(F['VaR'])
    p_short = float((F['VaR'] < P.k).mean())
    print(f"  wrote {os.path.relpath(path, ROOT)}")
    print(f"      equal-CE alpha = {a_eq:.6f}, Y0_VaR = {sv['Y0']:.6f}, "
          f"k_alpha = {sv['k_alpha']:.6f}")
    print(f"      sample atom_VaR   = {atom_var:.4f} "
          f"(theory {X.var_stats(sol=sv)['atom_mass']:.4f})")
    print(f"      sample P(F_T < 1) = {p_short:.4f} (alpha = {a_eq:.4f})")
    print(f"      Q5        ES {q5['ES']:.3f} / Merton {q5['Merton']:.3f} "
          f"/ VaR {q5['VaR']:.3f}")
    print(f"      bottom-5% ES {bot5['ES']:.3f} / Merton {bot5['Merton']:.3f} "
          f"/ VaR {bot5['VaR']:.3f}")
    print(f"      ES-VaR CDF crossing at F_T = {cross:.3f}")


# ═══════════════════════════════════════════════════════════
# eps-robustness of the equal-CE comparison
# ═══════════════════════════════════════════════════════════

EPS_ROBUST = (0.10, 0.12, 0.14)
EPS_ROBUST_EXPECTED = {           # (alpha, CE loss %, ES bot5, VaR bot5, ratio)
    0.10: (0.0812, 2.275, 0.713, 0.574, 3.02),
    0.12: (0.1419, 0.753, 0.685, 0.615, 2.20),
    0.14: (0.2415, 0.103, 0.660, 0.638, 1.52),
}


def eps_robustness(path=None):
    """Re-run the equal-CE calibration at several eps and tabulate the tails."""
    import csv
    if path is None:
        path = os.path.join(ROOT, 'results', 'table_eps_robust.csv')
    mert = X.merton_stats()
    rows = []
    print("eps-robustness of the equal-CE comparison")
    print(f"    {'eps':>5}{'alpha':>9}{'CEloss%':>9}{'ES bot5':>9}"
          f"{'VaR bot5':>10}{'ratio':>7}   vs expected")
    for e in EPS_ROBUST:
        a = X.match_alpha_equal_ce(eps=e)['alpha']
        E, V = X.es_stats(eps=e), X.var_stats(alpha=a)
        se = ES.solve_es(eps=e)
        loss = 100.0 * (mert['ce'] - E['ce']) / mert['ce']
        ratio = V['cond_shortfall'] / E['cond_shortfall']
        rows.append(dict(eps=e, alpha_eqCE=a, es_ce_loss_pct=loss,
                         es_bottom5=E['bottom5_mean'],
                         var_bottom5=V['bottom5_mean'],
                         cond_shortfall_ratio=ratio,
                         es_q5=E['q5'], var_q5=V['q5'],
                         es_cond_shortfall=E['cond_shortfall'],
                         var_cond_shortfall=V['cond_shortfall'],
                         es_prob_shortfall=E['prob_shortfall'],
                         k_eps=se['k_eps'], c=se['c'], Y0_es=se['Y0'],
                         bottom5_gain_pct=100.0 * (E['bottom5_mean']
                                                   / V['bottom5_mean'] - 1)))
        exp = EPS_ROBUST_EXPECTED[e]
        got = (a, loss, E['bottom5_mean'], V['bottom5_mean'], ratio)
        # each expected value is quoted at its own precision, so compare
        # within one unit in the last displayed place
        ulp = (1e-4, 1e-3, 1e-3, 1e-3, 1e-2)
        devs = [abs(g - x) for g, x in zip(got, exp)]
        ok = all(d <= u for d, u in zip(devs, ulp))
        print(f"    {e:>5.2f}{a:>9.4f}{loss:>9.3f}{E['bottom5_mean']:>9.3f}"
              f"{V['bottom5_mean']:>10.3f}{ratio:>7.2f}   "
              f"{'match' if ok else 'DIFFERS'} "
              f"(max dev {max(d / u for d, u in zip(devs, ulp)):.2f} ulp)")
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"    wrote {os.path.relpath(path, ROOT)}")


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

    claim_function()
    adjustment_factor()
    feasibility_map()
    terminal_cdf()
    eps_robustness()

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
