"""
Exact-statistics deliverables (review round 2)
==============================================
Replaces the Monte Carlo numbers in Table 2 with model-implied closed-form
statistics, recalibrates the equal-CE VaR level exactly, adds the VaR
feasibility bound alpha_min to Table 3, and runs the delta_L (liability
channel) comparative statics.

The MC pipeline is NOT removed — it is retained for the Figure 8 histogram
and is used here only as an independent cross-check of the exact formulas.

Outputs
    results/table2_exact.csv          Table 2, exact
    results/table_exact_summary.tex   Table 2 body (no SE rows)
    results/exact_vs_mc.md            exact vs MC (+-3 SE) reconciliation
    results/headline_numbers.md       manuscript-ready headline figures
    results/table_sensitivity_v2.csv/.tex   Table 3 + alpha_min column
    results/table_deltaL.csv/.tex     delta_L comparative statics
    outputs/common/eps_min_muI_v2.png       Figure 6 draft with alpha_min
    outputs/fixed_claim/mc_terminal_y010_inset.png   Figure 8 draft (a)
    outputs/fixed_claim/mc_terminal_y010_cdf.png     Figure 8 draft (b)

Run:  python3 scripts/run_exact.py [--quick]
"""

import argparse
import csv
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
from ldi import exact_stats as X
from ldi import simulate as SIM
from ldi import compare as C
from ldi.style import (apply_style, COLORS, LINE_STYLES, LEGEND, FIGSIZES,
                       setup_grid, add_k_vline, savefig, HIST_ALPHA)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
OUT_C = os.path.join(ROOT, "outputs", "common")
OUT_F = os.path.join(ROOT, "outputs", "fixed_claim")

TAIL_Q = 0.05


# ═══════════════════════════════════════════════════════════
# Verification against the expected values supplied in the order
# ═══════════════════════════════════════════════════════════

# (label, got, expected, kind)  kind: 'rel' -> 1e-3 relative, 'prob' -> 1e-4 abs
def verify(rows, sol_es, sol_var, a_eqce, a_thr):
    m = {r['label'].split(' (')[0]: r for r in rows}
    es, mert = m['ES'], m['Merton']
    checks = [
        ('Y0 (ES)',              sol_es['Y0'],       0.803420,  'rel'),
        ('k_eps',                sol_es['k_eps'],    0.727553,  'rel'),
        ('c',                    sol_es['c'],        1.374471,  'rel'),
        ('eps_min',              sol_es['eps_min'],  0.087629,  'rel'),
        ('eps_M',                sol_es['eps_M'],    0.152614,  'rel'),
        ('CE_Merton',            mert['ce'],         1.008236,  'rel'),
        ('CE_ES',                es['ce'],           0.985302,  'rel'),
        ('ES CE loss %',         es['ce_loss_pct'],  2.275,     'rel'),
        ('ES E[(k-F)^+]',        es['exp_shortfall'], 0.032497, 'rel'),
        ('ES P(F<k)',            es['prob_shortfall'], 0.24803, 'prob'),
        ('ES Q5',                es['q5'],           0.78746,   'rel'),
        ('ES bottom-5% mean',    es['bottom5_mean'], 0.71285,   'rel'),
        ('ES mean',              es['mean'],         1.01514,   'rel'),
        ('ES std',               es['std'],          0.14712,   'rel'),
        ('Merton mean',          mert['mean'],       1.10562,   'rel'),
        ('Merton std',           mert['std'],        0.27839,   'rel'),
        ('Merton P(F<k)',        mert['prob_shortfall'], 0.38935, 'prob'),
        ('Merton Q5',            mert['q5'],         0.71309,   'rel'),
        ('Merton bottom-5%',     mert['bottom5_mean'], 0.64556, 'rel'),
        ('VaR(0.10) k_alpha',    sol_var['k_alpha'], 0.71477,   'rel'),
        ('alpha_equal_CE',       a_eqce,             0.08118,   'rel'),
        ('alpha_threshold',      a_thr,              0.1067,    'rel'),
        ('alpha_min baseline',   VaR.alpha_min(),    0.01597,   'prob'),
    ]
    # equal-CE VaR cross-checks (order gives these at 1e-2 relative)
    eq = [r for r in rows if r['label'].startswith('VaR equal-CE')][0]
    loose = [
        ('eqCE P(F<k)',      eq['prob_shortfall'], 0.0812,   1e-2),
        ('eqCE E[(k-F)^+]',  eq['exp_shortfall'],  0.03208,  1e-2),
        ('eqCE CondSF',      eq['cond_shortfall'], 0.39524,  1e-2),
        ('eqCE Q5',          eq['q5'],             0.63363,  1e-2),
        ('eqCE bottom-5%',   eq['bottom5_mean'],   0.57360,  1e-2),
        ('eqCE CE',          eq['ce'],             0.985302, 1e-2),
    ]

    lines, failures = [], []
    lines.append(f"{'item':<24}{'computed':>13}{'expected':>13}"
                 f"{'deviation':>13}  status")
    lines.append('-' * 78)
    for label, got, exp, kind in checks:
        tol = 1e-4 if kind == 'prob' else 1e-3 * abs(exp)
        dev = got - exp
        ok = abs(dev) <= tol
        status = 'OK' if ok else 'DEVIATION'
        if not ok:
            failures.append((label, got, exp, dev, kind))
        lines.append(f"{label:<24}{got:>13.6f}{exp:>13.6f}{dev:>+13.2e}  {status}")
    lines.append('')
    lines.append('equal-CE VaR cross-checks (1e-2 relative)')
    for label, got, exp, tol in loose:
        dev = got - exp
        ok = abs(dev) <= tol * abs(exp)
        if not ok:
            failures.append((label, got, exp, dev, 'loose'))
        lines.append(f"{label:<24}{got:>13.6f}{exp:>13.6f}{dev:>+13.2e}  "
                     f"{'OK' if ok else 'DEVIATION'}")
    return '\n'.join(lines), failures


def consistency_audit(sol_var):
    """Explain the deviations by checking the ORDER's own numbers internally.

    The order states Merton mean = 1.10562 and CE = 1.008236. For a lognormal
    terminal value these two pin (m_P·T, s) exactly:
        ln CE   = m_P·T - s²          (gamma = 3)
        ln mean = m_P·T + s²/2
    Everything else in Table 2 follows deterministically from those plus the
    stated (Y0, k_eps, c) — so any expected value inconsistent with them is an
    arithmetic slip on the supplying side, not a disagreement about the model.
    """
    from scipy.stats import norm
    ln_ce, ln_mean = np.log(1.008236), np.log(1.10562)
    s2 = (ln_mean - ln_ce) / 1.5
    s = np.sqrt(s2)
    mT = ln_ce + s2

    Y0, ke, c = 0.803420, 0.727553, 1.374471
    p_es = float(norm.cdf((np.log(ke) - np.log(Y0) - mT) / s))
    st = X.claim_stats(Y0, c=c, k_low=ke)

    lam = float(np.exp(mT + s * norm.ppf(0.10)))
    Y0v_implied = 0.71477 / lam
    budget_at_implied = float(VaR.psi(Y0v_implied, 0.71477, P.T))

    return {
        'implied_mPT': mT, 'implied_s': s,
        'ours_mPT': P.m_P * P.T, 'ours_s': P.sigma_Y * np.sqrt(P.T),
        'ES_prob_from_order_inputs': p_es,
        'ES_std_from_order_inputs': st['std'],
        'VaR_Y0_implied_by_0.71477': Y0v_implied,
        'VaR_budget_at_that_Y0': budget_at_implied,
        'VaR_budget_at_ours': float(VaR.psi(sol_var['Y0'], sol_var['k_alpha'], P.T)),
    }


# ═══════════════════════════════════════════════════════════
# LaTeX helpers
# ═══════════════════════════════════════════════════════════

def _f(x, nd=4, dash='---'):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f'{x:.{nd}f}'


def write_table2_tex(rows, path):
    body = []
    for r in rows:
        body.append(
            f"  {r['tex_label']} & {_f(r['mean'],4)} & {_f(r['std'],4)} & "
            f"{_f(r['prob_shortfall'],4)} & {_f(r['exp_shortfall'],4)} & "
            f"{_f(r['cond_shortfall'],4)} & {_f(r['q5'],4)} & "
            f"{_f(r['bottom5_mean'],4)} & {_f(r['ce'],4)} & "
            f"{_f(r['ce_loss_pct'],3)} \\\\")
    tex = r"""\begin{table}[htbp]
\centering
\caption{Model-implied terminal statistics ($F_0 = %s$, $T = %s$)}
\label{tab:exact_summary}
\begin{tabular}{l ccccccccc}
\toprule
 & & & & Expected & Conditional & & Bottom-5\%% & & CE \\
Strategy & $\mathrm{E}[F_T]$ & $\mathrm{Std}[F_T]$ & $\mathrm{P}(F_T<k)$
 & shortfall & shortfall & $Q_{5}$ & mean $F_T$ & CE & loss (\%%) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (_f(P.F0, 1), _f(P.T, 0), '\n'.join(body))
    with open(path, 'w') as fh:
        fh.write(tex)


def write_sensitivity_tex(panels, path):
    blocks = []
    for pname, sym, rows in panels:
        lines = []
        for i, r in enumerate(rows):
            head = (rf"  \multirow{{{len(rows)}}}{{*}}{{{sym}}} & "
                    if i == 0 else "   & ")
            val = f"{r['value']:g}"
            if r['feasible']:
                ke, aes, tot_es = _f(r['k_eps']), _f(r['A0'], 3), _f(r['tot_es'], 3)
            else:
                ke = aes = tot_es = '---'
            lines.append(
                f"{head}{val} & {ke} & {aes} & {_f(r['A0_var'],3)} & "
                f"{_f(r['alpha_min'],5)} & {tot_es} & {_f(r['tot_var'],3)} \\\\")
        blocks.append('\n'.join(lines))
    tex = r"""\begin{table}[htbp]
\centering
\caption{Sensitivity of the joint solution, with the VaR feasibility bound
($F_0 = %s$, $\varepsilon = %s$, $\alpha = %s$)}
\label{tab:sensitivity_v2}
\begin{tabular}{lc c cc c cc}
\toprule
 & & & \multicolumn{2}{c}{Adjustment factor} & & \multicolumn{2}{c}{Total allocation} \\
\cmidrule(lr){4-5} \cmidrule(lr){7-8}
Parameter & Value & $k_\varepsilon$ & $A_{\mathrm{ES}}$ & $A_{\mathrm{VaR}}$
 & $\alpha_{\min}$ & ES & VaR \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (_f(P.F0, 1), _f(P.epsilon, 2), _f(P.alpha, 2),
       '\n  \\midrule\n'.join(blocks))
    with open(path, 'w') as fh:
        fh.write(tex)


def write_deltaL_tex(rows, path):
    lines = []
    for r in rows:
        star = r'$^{*}$' if r['baseline'] else ''
        if r['feasible']:
            ke, aes, tot_es = _f(r['k_eps']), _f(r['A0'], 3), _f(r['tot_es'], 3)
        else:
            ke = aes = tot_es = '---'
        lines.append(
            f"  {r['value']:.4f}{star} & {_f(r['r_tilde'],5)} & "
            f"{_f(r['eps_min'],4)} & {_f(r['eps_M'],4)} & {r['status']} & "
            f"{ke} & {aes} & {_f(r['A0_var'],3)} & {_f(r['alpha_min'],5)} & "
            f"{tot_es} & {_f(r['tot_var'],3)} \\\\")
    tex = r"""\begin{table}[htbp]
\centering
\caption{Liability-channel comparative statics: $\delta_L = \beta_0 + \beta_1\mu_I$
with all asset-side parameters held at baseline ($F_0 = %s$, $\varepsilon = %s$)}
\label{tab:deltaL}
\begin{tabular}{cc cc l c cc c cc}
\toprule
 & & & & & & \multicolumn{2}{c}{Adjustment factor} & & \multicolumn{2}{c}{Total allocation} \\
\cmidrule(lr){7-8} \cmidrule(lr){10-11}
$\delta_L$ & $\tilde r$ & $\varepsilon_{\min}$ & $\varepsilon_M$ & Status
 & $k_\varepsilon$ & $A_{\mathrm{ES}}$ & $A_{\mathrm{VaR}}$ & $\alpha_{\min}$
 & ES & VaR \\
\midrule
%s
\bottomrule
\end{tabular}

\vspace{2pt}
{\footnotesize $^{*}$ baseline. $\sigma_Y$, $\lambda$ and $\Pi^{*}$ are
identical in every row by construction: only $\tilde r = r - \delta_L$ moves.}
\end{table}
""" % (_f(P.F0, 1), _f(P.epsilon, 2), '\n'.join(lines))
    with open(path, 'w') as fh:
        fh.write(tex)


def write_csv(rows, path, fields=None):
    fields = fields or list(rows[0].keys())
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


# ═══════════════════════════════════════════════════════════
# Panels
# ═══════════════════════════════════════════════════════════

def _row_from_state(value, F0, eps):
    """Solve both models under the CURRENT parameter state."""
    s = ES.solve_es(F0, eps, strict=False)
    sv = VaR.solve_var(F0, P.alpha, strict=False)
    lo, hi = P.eps_band(F0)
    A0 = (float(ES.adjustment_factor(s['Y0'], s['k_eps'], s['c'], P.T))
          if s['feasible'] and s['binding'] else (1.0 if s['feasible'] else np.nan))
    A0v = (float(VaR.adjustment_factor(sv['Y0'], sv['k_alpha'], P.T))
           if sv['feasible'] and sv['binding'] else (1.0 if sv['feasible'] else np.nan))
    tot = float(P.Pi_star.sum())
    return dict(value=value, r_tilde=P.r_tilde, sigma_Y=P.sigma_Y,
                eps_min=lo, eps_M=hi, feasible=s['feasible'],
                binding=s['binding'], Y0=s['Y0'], k_eps=s['k_eps'], c=s['c'],
                A0=A0, tot_es=A0 * tot, Y0_var=sv['Y0'],
                k_alpha=sv['k_alpha'], A0_var=A0v, tot_var=A0v * tot,
                alpha_min=VaR.alpha_min(F0), merton_total=tot,
                status=('Infeasible' if not s['feasible']
                        else ('Slack' if not s['binding'] else 'Binding')))


def sensitivity_panels(F0=None, eps=None):
    if F0 is None:
        F0 = P.F0
    if eps is None:
        eps = P.epsilon
    panels = []
    for pname, sym in [('GAMMA', r'$\gamma$'), ('EPS', r'$\varepsilon$'),
                       ('MU_I', r'$\mu_I$'), ('T', '$T$'), ('RHO', r'$\rho$')]:
        rows = []
        if pname == 'EPS':
            for e in P.EPS_GRID:
                r = _row_from_state(e, F0, e)
                r['param'] = 'epsilon'
                rows.append(r)
        else:
            for v in C.SENS_CONFIGS[pname]:
                with P.override_params(**{pname: v}):
                    r = _row_from_state(v, F0, eps)
                r['param'] = pname
                rows.append(r)
        panels.append((pname, sym, rows))
    return panels


DELTA_L_GRID = [0.040, 0.043, 0.046, 0.0484, 0.052]


def deltaL_panel(F0=None, eps=None):
    if F0 is None:
        F0 = P.F0
    if eps is None:
        eps = P.epsilon
    base = P.delta_L()
    rows = []
    for d in DELTA_L_GRID:
        with P.override_delta_L(d):
            r = _row_from_state(d, F0, eps)
        r['param'] = 'delta_L'
        r['baseline'] = abs(d - base) < 1e-9
        rows.append(r)
    return rows


# ═══════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════

def fig_eps_min_muI_v2(mu_range=(0.01, 0.05), n=300, F0=None, save_path=None):
    """Figure 6 draft: eps_min(mu_I) with the VaR bound alpha_min(mu_I) beside it.

    The two live on incompatible scales (an ES budget in funding-ratio units
    vs a probability), so alpha_min goes on a twin axis rather than sharing
    the left one.
    """
    if F0 is None:
        F0 = P.F0
    mus = np.linspace(*mu_range, n)
    floor, merton, amin = [], [], []
    for mu in mus:
        with P.override_params(MU_I=mu):
            floor.append(P.eps_min(F0))
            merton.append(P.eps_merton(F0))
            amin.append(VaR.alpha_min(F0))
    floor, merton, amin = map(np.array, (floor, merton, amin))

    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    ax.plot(mus, floor, color=COLORS['ES'], lw=2.5,
            label=r'ES floor $\varepsilon_{\min}(\mu_I)$')
    ax.plot(mus, merton, color=COLORS['ES'], lw=2.0, ls='--', alpha=0.7,
            label=r'$\varepsilon_M(\mu_I)$')
    ax.fill_between(mus, floor, merton, where=merton > floor, alpha=0.13,
                    color=COLORS['ES'], label='feasible & binding band')
    ax.axhline(P.epsilon, color='0.3', ls='--', lw=1.5,
               label=rf'baseline $\varepsilon$={P.epsilon}')
    ax.axvline(P.MU_I, color='0.5', ls=':', lw=1.2)
    bad = np.where(floor >= P.epsilon)[0]
    if bad.size:
        ax.axvspan(mus[bad[0]], mus[-1], color='0.6', alpha=0.20)
        ax.annotate(rf'ES infeasible ($\mu_I>{mus[bad[0]]:.4f}$)',
                    (mus[bad[0]], P.epsilon), textcoords='offset points',
                    xytext=(6, 26), fontsize=11)
    ax.set_xlabel(r'Expected inflation $\mu_I$')
    ax.set_ylabel(r'ES budget')
    ax.set_ylim(bottom=0)
    setup_grid(ax)

    ax2 = ax.twinx()
    ax2.plot(mus, amin, color=COLORS['VaR'], lw=2.5, ls='-.',
             label=r'VaR bound $\alpha_{\min}(\mu_I)$')
    ax2.axhline(P.alpha, color=COLORS['VaR'], ls=':', lw=1.2,
                label=rf'baseline $\alpha$={P.alpha}')
    ax2.set_ylabel(r'VaR level $\alpha_{\min}$', color=COLORS['VaR'])
    ax2.tick_params(axis='y', labelcolor=COLORS['VaR'])
    ax2.set_ylim(bottom=0)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', framealpha=0.9,
              edgecolor='gray', fontsize=10)
    ax.set_title(r'Feasibility: ES floor binds long before the VaR bound')
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


def fig_terminal_variants(res, out_dir):
    """Figure 8 drafts: (a) left-tail zoom inset, (b) CDF version."""
    keys = {'merton': 'Merton', 'es': 'ES', 'var': 'VaR'}
    bins = np.linspace(0.3, 2.2, 90)

    # (a) histogram with a left-tail inset
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for name, key in keys.items():
        ax.hist(res[name]['F'][:, -1], bins=bins, alpha=HIST_ALPHA,
                color=COLORS[key],
                label=f"{key} (CVaR$_5$={res[name]['stats']['cvar05']:.3f})")
    add_k_vline(ax)
    ax.set_xlabel('Terminal funding ratio $F_T$')
    ax.set_ylabel('Frequency')
    ax.set_title('Terminal distribution with left-tail detail')
    ax.legend(**LEGEND)
    setup_grid(ax)

    # shade the source region instead of drawing zoom connectors: the region
    # is a thin sliver at the baseline, so connector lines cross the whole
    # panel and read as clutter.
    ax.axvspan(0.4, 0.9, color='0.5', alpha=0.08, zorder=0)
    y_top = ax.get_ylim()[1]
    ax.annotate('atom at $F_T=k$\n(protected states)',
                xy=(1.0, 0.20 * y_top), xytext=(1.30, 0.17 * y_top),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='0.35', lw=1.2))

    axin = ax.inset_axes([0.09, 0.50, 0.34, 0.38])
    zbins = np.linspace(0.4, 0.9, 45)
    for name, key in keys.items():
        axin.hist(res[name]['F'][:, -1], bins=zbins, alpha=HIST_ALPHA,
                  color=COLORS[key])
    axin.set_xlim(0.4, 0.9)
    axin.set_title('left tail (shaded region)', fontsize=10)
    axin.tick_params(labelsize=9)
    axin.grid(True, alpha=0.3, lw=0.5)
    for sp in axin.spines.values():
        sp.set_edgecolor('0.4')
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, 'mc_terminal_y010_inset.png'))

    # (b) empirical CDF
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    cdfs = {}
    for name, key in keys.items():
        x = np.sort(res[name]['F'][:, -1])
        y = np.arange(1, len(x) + 1) / len(x)
        cdfs[name] = (x, y)
        ax.plot(x, y, label=key, **LINE_STYLES[key])
    add_k_vline(ax)
    ax.axhline(TAIL_Q, color='0.4', ls=':', lw=1.2, label='5% level')

    # where the ES and VaR CDFs cross: below it ES is first-order better
    grid = np.linspace(0.45, 0.98, 2000)
    d = (np.interp(grid, *cdfs['es']) - np.interp(grid, *cdfs['var']))
    sign = np.where(np.diff(np.sign(d)))[0]
    if sign.size:
        xc = grid[sign[-1]]
        yc = float(np.interp(xc, *cdfs['es']))
        ax.plot(xc, yc, 'o', ms=8, color='k', zorder=5)
        ax.annotate(f'CDFs cross at $F_T$={xc:.3f}\nbelow: ES has less mass',
                    (xc, yc), textcoords='offset points', xytext=(12, -34),
                    fontsize=10)
    ax.set_xlim(0.4, 1.8)
    ax.set_xlabel('Terminal funding ratio $F_T$')
    ax.set_ylabel('$P(F_T \\leq x)$')
    ax.set_title('Terminal distribution (empirical CDF)')
    ax.legend(**LEGEND)
    setup_grid(ax)
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, 'mc_terminal_y010_cdf.png'))


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    apply_style()
    for d in (RES, OUT_C, OUT_F):
        os.makedirs(d, exist_ok=True)
    n_paths = 2000 if args.quick else SIM.DEFAULT_N_PATHS

    P.print_params()
    print()

    # ── Calibration ───────────────────────────────────────
    sol_es = ES.solve_es()
    sol_var = VaR.solve_var()
    eq = X.match_alpha_equal_ce()
    thr = X.match_alpha_threshold()
    print(f"exact equal-CE alpha        = {eq['alpha']:.6f}  "
          f"(target CE {eq['ce_target']:.6f}, achieved {eq['ce_achieved']:.6f})")
    print(f"exact threshold-matched alpha = {thr['alpha']:.6f}  "
          f"(k_alpha = k_eps = {thr['k_eps']:.6f}, Y0 = {thr['Y0']:.6f})")
    print(f"alpha_min (closed form)     = {VaR.alpha_min():.6f}  "
          f"| numeric {VaR.alpha_min_numeric():.6f}")
    print()

    # ── Table 2 exact ─────────────────────────────────────
    rows = X.table2(alpha_eqce=eq['alpha'], alpha_thr=thr['alpha'])
    tex_labels = ['Merton',
                  rf"ES ($\varepsilon={P.epsilon:g}$)",
                  rf"VaR ($\alpha={P.alpha:g}$)",
                  rf"VaR equal-CE ($\alpha={eq['alpha']:.5f}$)",
                  rf"VaR threshold-matched ($\alpha={thr['alpha']:.5f}$)"]
    for r, tl in zip(rows, tex_labels):
        r['tex_label'] = tl

    fields = ['label', 'mean', 'std', 'prob_shortfall', 'exp_shortfall',
              'cond_shortfall', 'q5', 'bottom5_mean', 'ce', 'ce_loss_pct',
              'Y0', 'c', 'k_low']
    write_csv(rows, os.path.join(RES, 'table2_exact.csv'), fields)
    write_table2_tex(rows, os.path.join(RES, 'table_exact_summary.tex'))

    hdr = (f"{'strategy':<42}{'mean':>9}{'std':>9}{'P(F<k)':>9}{'E[(k-F)+]':>11}"
           f"{'CondSF':>9}{'Q5':>9}{'Bot5':>9}{'CE':>10}{'CEloss%':>9}")
    print("Table 2 — exact (model-implied)")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['label']:<42}{r['mean']:>9.5f}{r['std']:>9.5f}"
              f"{r['prob_shortfall']:>9.5f}{r['exp_shortfall']:>11.5f}"
              f"{r['cond_shortfall']:>9.5f}{r['q5']:>9.5f}"
              f"{r['bottom5_mean']:>9.5f}{r['ce']:>10.6f}"
              f"{r['ce_loss_pct']:>9.4f}")
    print()

    # ── Verification ──────────────────────────────────────
    report, failures = verify(rows, sol_es, sol_var, eq['alpha'], thr['alpha'])
    print("Verification against the expected values in the order")
    print(report)
    audit = consistency_audit(sol_var)
    if failures:
        print()
        print("DEVIATIONS FOUND — internal-consistency audit of the expected values:")
        for kk, v in audit.items():
            print(f"    {kk:<32} {v:.8f}" if isinstance(v, float)
                  else f"    {kk:<32} {v}")
    print()

    # ── MC cross-check ────────────────────────────────────
    print(f"MC cross-check (N={n_paths}, seed={SIM.DEFAULT_SEED}) ...")
    mc = SIM.run(n_paths=n_paths, alpha=P.alpha)
    mc_map = {'Merton': 'merton', 'ES': 'es', 'VaR': 'var'}
    mc_lines = ["| statistic | exact | MC | SE | (exact-MC)/SE | within 3 SE |",
                "|---|---|---|---|---|---|"]
    mc_flags = []
    for r in rows[:3]:
        name = mc_map[r['label'].split(' (')[0]]
        st, se = mc[name]['stats'], mc[name]['se']
        pairs = [('mean', 'mean'), ('std', 'std'),
                 ('prob_shortfall', 'shortfall_prob'),
                 ('exp_shortfall', 'exp_shortfall'),
                 ('cond_shortfall', 'cond_shortfall'),
                 ('q5', 'q05'), ('bottom5_mean', 'cvar05'), ('ce', 'CE')]
        for ex_key, mc_key in pairs:
            e_val, m_val, s_val = r[ex_key], st[mc_key], se[mc_key]
            z = (e_val - m_val) / s_val if s_val > 0 else 0.0
            ok = abs(z) <= 3
            mc_lines.append(f"| {r['label']} {ex_key} | {e_val:.5f} | "
                            f"{m_val:.5f} | {s_val:.5f} | {z:+.2f} | "
                            f"{'yes' if ok else '**NO**'} |")
            if not ok:
                mc_flags.append((r['label'], ex_key, z))
    with open(os.path.join(RES, 'exact_vs_mc.md'), 'w') as fh:
        fh.write(f"# Exact vs Monte Carlo reconciliation\n\n"
                 f"MC: N={n_paths}, steps={SIM.DEFAULT_N_STEPS}, "
                 f"seed={SIM.DEFAULT_SEED}, bootstrap SE.\n\n"
                 + '\n'.join(mc_lines) + '\n')
    for lbl, kk, z in mc_flags:
        print(f"    WARNING (not a failure): {lbl} {kk} is {z:+.2f} SE from MC")
    if not mc_flags:
        print("    all statistics within 3 SE")
    print()

    # ── Sensitivity + alpha_min ───────────────────────────
    print("Table 3 — sensitivity with alpha_min")
    panels = sensitivity_panels()
    all_rows = [r for _, _, rs in panels for r in rs]
    write_csv(all_rows, os.path.join(RES, 'table_sensitivity_v2.csv'))
    write_sensitivity_tex(panels, os.path.join(RES, 'table_sensitivity_v2.tex'))
    for pname, _, rs in panels:
        for r in rs:
            print(f"    {pname:<6}{r['value']:<8g} eps_min={r['eps_min']:.4f} "
                  f"A_ES={r['A0']:.4f} A_VaR={r['A0_var']:.4f} "
                  f"alpha_min={r['alpha_min']:.5f}  {r['status']}")

    # alpha_min sanity checks
    base_amin = VaR.alpha_min()
    for pname in ('GAMMA', 'EPS'):
        rs = dict((p, r) for p, _, r in [(x[0], x[1], x[2]) for x in panels])[pname]
        assert all(abs(r['alpha_min'] - base_amin) < 1e-12 for r in rs), \
            f'alpha_min must be constant across the {pname} panel'
    for a in (P.alpha, eq['alpha'], thr['alpha']):
        assert a > base_amin, f'alpha={a} below alpha_min={base_amin}'
    print(f"    [ok] alpha_min constant across gamma and epsilon panels "
          f"(= {base_amin:.5f}); all three calibrations exceed it")
    print()

    # ── delta_L ───────────────────────────────────────────
    print("delta_L comparative statics (liability channel isolated)")
    dl = deltaL_panel()
    write_csv(dl, os.path.join(RES, 'table_deltaL.csv'))
    write_deltaL_tex(dl, os.path.join(RES, 'table_deltaL.tex'))
    for r in dl:
        print(f"    delta_L={r['value']:.4f} r_tilde={r['r_tilde']:+.5f} "
              f"eps_min={r['eps_min']:.4f} eps_M={r['eps_M']:.4f} "
              f"alpha_min={r['alpha_min']:.5f}  {r['status']}")
    sig = {round(r['sigma_Y'], 12) for r in dl}
    tot = {round(r['merton_total'], 12) for r in dl}
    assert len(sig) == 1 and len(tot) == 1, 'asset side must be invariant'
    print(f"    [ok] sigma_Y and Merton total identical in every row "
          f"({sig.pop():.6f}, {tot.pop():.6f})")
    print()

    # ── Headline numbers ──────────────────────────────────
    es_r = rows[1]
    eq_r = [r for r in rows if r['label'].startswith('VaR equal-CE')][0]
    nom_r = rows[2]
    mert_r = rows[0]
    bottom_gain = es_r['bottom5_mean'] / eq_r['bottom5_mean'] - 1
    cond_ratio = eq_r['cond_shortfall'] / es_r['cond_shortfall']
    sf_cut = 1 - es_r['exp_shortfall'] / mert_r['exp_shortfall']

    hl = f"""# Headline numbers (exact, model-implied)

All values closed form — no simulation, no seed. Generated by
`scripts/run_exact.py`; full tables in `results/table2_exact.csv`.

## Calibration

| quantity | value |
|---|---|
| `alpha_eqCE` (equal-CE VaR level) | **{eq['alpha']:.5f}** |
| `alpha_thr` (threshold-matched, k_alpha = k_eps) | {thr['alpha']:.5f} |
| `alpha_min` (VaR feasibility bound) | {base_amin:.5f} |
| exact CE loss of ES (= of equal-CE VaR) | **{es_r['ce_loss_pct']:.3f}%** |
| (Y0, k_eps, c) | ({sol_es['Y0']:.6f}, {sol_es['k_eps']:.6f}, {sol_es['c']:.6f}) |
| (eps_min, eps_M) | ({sol_es['eps_min']:.6f}, {sol_es['eps_M']:.6f}) |

## Equal-CE comparison (welfare cost held identical at {es_r['ce_loss_pct']:.3f}%)

| statistic | ES | VaR equal-CE | contrast |
|---|---|---|---|
| bottom-5% mean $F_T$ | **{es_r['bottom5_mean']:.5f}** | {eq_r['bottom5_mean']:.5f} | **+{bottom_gain*100:.2f}%** |
| $Q_5$ | {es_r['q5']:.5f} | {eq_r['q5']:.5f} | +{(es_r['q5']-eq_r['q5']):.5f} |
| conditional shortfall | {es_r['cond_shortfall']:.5f} | {eq_r['cond_shortfall']:.5f} | **x{cond_ratio:.3f}** |
| $P(F_T<k)$ | {es_r['prob_shortfall']:.5f} | {eq_r['prob_shortfall']:.5f} | ES higher |
| $E[(k-F_T)^+]$ | {es_r['exp_shortfall']:.5f} | {eq_r['exp_shortfall']:.5f} | |
| CE | {es_r['ce']:.6f} | {eq_r['ce']:.6f} | matched |

* bottom-5% mean improvement: **{bottom_gain*100:.2f}%**
* conditional-shortfall multiple: **{cond_ratio:.3f}x** deeper under VaR
* ES cuts $E[(k-F_T)^+]$ vs Merton by **{sf_cut*100:.2f}%**
  ({mert_r['exp_shortfall']:.5f} -> {es_r['exp_shortfall']:.5f})

### Nuance to state explicitly in the text

Against the *equal-CE* VaR, ES does **not** win on the unconditional
expected shortfall: {es_r['exp_shortfall']:.5f} vs {eq_r['exp_shortfall']:.5f}, i.e.
{(1-es_r['exp_shortfall']/eq_r['exp_shortfall'])*100:+.2f}% — the two are
within {abs(es_r['exp_shortfall']-eq_r['exp_shortfall'])/eq_r['exp_shortfall']*100:.1f}% of each other,
and VaR is marginally lower. The same is true of $P(F_T<k)$, where VaR is
much lower by construction ({eq_r['prob_shortfall']:.4f} vs {es_r['prob_shortfall']:.4f}).

The entire ES advantage is in the SHAPE of the shortfall, not its mean:
ES accepts shortfalls that are frequent and shallow, VaR produces ones that
are rare and deep. That is exactly what the bottom-5% mean
({bottom_gain*100:+.1f}%) and the conditional shortfall ({cond_ratio:.2f}x)
measure, and it is why a mean-based summary understates the difference.
Leading with $E[(k-F_T)^+]$ against the equal-CE VaR would invite the
objection that the two constraints are equivalent; leading with the tail
statistics is both stronger and more accurate.

## Nominal alpha = {P.alpha:g} (not welfare-matched)

| statistic | ES | VaR |
|---|---|---|
| bottom-5% mean | {es_r['bottom5_mean']:.5f} | {nom_r['bottom5_mean']:.5f} |
| conditional shortfall | {es_r['cond_shortfall']:.5f} | {nom_r['cond_shortfall']:.5f} |
| CE loss (%) | {es_r['ce_loss_pct']:.3f} | {nom_r['ce_loss_pct']:.3f} |

VaR's bottom-5% mean ({nom_r['bottom5_mean']:.5f}) is below Merton's
({mert_r['bottom5_mean']:.5f}): the gambling incentive makes the deep tail
worse than no constraint at all.
"""
    with open(os.path.join(RES, 'headline_numbers.md'), 'w') as fh:
        fh.write(hl)
    print(f"headline: bottom-5% gain {bottom_gain*100:.2f}%, "
          f"CondSF ratio {cond_ratio:.3f}x, ES CE loss {es_r['ce_loss_pct']:.3f}%")

    # ── Figures ───────────────────────────────────────────
    print()
    print("Figures ...")
    fig_eps_min_muI_v2(save_path=os.path.join(OUT_C, 'eps_min_muI_v2.png'))
    fig_terminal_variants(mc, OUT_F)
    print(f"    {os.path.relpath(os.path.join(OUT_C, 'eps_min_muI_v2.png'), ROOT)}")
    print(f"    {os.path.relpath(os.path.join(OUT_F, 'mc_terminal_y010_inset.png'), ROOT)}")
    print(f"    {os.path.relpath(os.path.join(OUT_F, 'mc_terminal_y010_cdf.png'), ROOT)}")

    if failures:
        print()
        print(f"{len(failures)} expected value(s) deviated — see the audit above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
