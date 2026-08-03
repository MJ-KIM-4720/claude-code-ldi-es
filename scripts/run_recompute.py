"""
Full numerical recompute under the joint-system / fixed-claim model
====================================================================
Entry point that regenerates EVERYTHING downstream of the theory fix:

  outputs/cross_sectional/   Mode A figures (x-axis = F0, one fund per point)
  outputs/fixed_claim/       Mode B figures + Monte Carlo (one fund, y-axis state)
  outputs/common/            mode-independent figures (feasibility floor, claims)
  results/diagnostics.csv    residuals, replication error, baseline solution
  results/table2_mc.csv      Table 2 with bootstrap standard errors
  results/sensitivity.csv    per-config eps_min / eps_M / feasibility

Run:  python3 scripts/run_recompute.py [--quick]
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldi import params as P
from ldi import es_model as ES
from ldi import var_model as VaR
from ldi import compare as C
from ldi import simulate as SIM
from ldi.style import (apply_style, COLORS, LINE_STYLES, LEGEND, FIGSIZES,
                       setup_grid, add_k_vline, savefig, FAN_ALPHA, HIST_ALPHA)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_X = os.path.join(ROOT, "outputs", "cross_sectional")
OUT_F = os.path.join(ROOT, "outputs", "fixed_claim")
OUT_C = os.path.join(ROOT, "outputs", "common")
RES = os.path.join(ROOT, "results")


def _mk():
    for d in (OUT_X, OUT_F, OUT_C, RES):
        os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# Monte Carlo figures (fixed claim only)
# ═══════════════════════════════════════════════════════════

def mc_figures(res):
    t = res['merton']['t_grid']

    # Fan chart of the funding ratio
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZES['triple'], sharey=True)
    for ax, name in zip(axes, ('merton', 'es', 'var')):
        F = res[name]['F']
        col = COLORS[{'merton': 'Merton', 'es': 'ES', 'var': 'VaR'}[name]]
        for lo, hi, a in ((5, 95, FAN_ALPHA['outer']),
                          (25, 75, FAN_ALPHA['middle']),
                          (40, 60, FAN_ALPHA['inner'])):
            ax.fill_between(t, np.percentile(F, lo, axis=0),
                            np.percentile(F, hi, axis=0), color=col, alpha=a)
        ax.plot(t, np.median(F, axis=0), color=col, lw=2.2, label='median')
        ax.axhline(P.k, color='green', ls='--', alpha=0.6, label=f'$k$={P.k}')
        ax.set_title(name.upper())
        ax.set_xlabel('$t$')
        setup_grid(ax)
        ax.legend(**LEGEND)
    axes[0].set_ylabel('Funding ratio $F_t$')
    plt.suptitle('Fixed-claim Monte Carlo: funding-ratio fan charts', fontsize=14)
    plt.tight_layout()
    savefig(fig, os.path.join(OUT_F, 'mc_fan.png'))

    # Terminal distribution
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    bins = np.linspace(0.3, 2.2, 90)
    for name in ('merton', 'es', 'var'):
        col = COLORS[{'merton': 'Merton', 'es': 'ES', 'var': 'VaR'}[name]]
        ax.hist(res[name]['F'][:, -1], bins=bins, alpha=HIST_ALPHA, color=col,
                label=f"{name.upper()} (CVaR$_5$="
                      f"{res[name]['stats']['cvar05']:.3f})")
    add_k_vline(ax)
    ax.set_xlabel('Terminal funding ratio $F_T$')
    ax.set_ylabel('Frequency')
    ax.set_title('Terminal distribution (fixed claim)')
    ax.legend(**LEGEND)
    setup_grid(ax)
    plt.tight_layout()
    savefig(fig, os.path.join(OUT_F, 'mc_terminal.png'))

    # Shortfall probability over time + mean exposure over time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for name in ('merton', 'es', 'var'):
        key = {'merton': 'Merton', 'es': 'ES', 'var': 'VaR'}[name]
        ax1.plot(t, np.mean(res[name]['F'] < P.k, axis=0),
                 label=key, **LINE_STYLES[key])
        A = res[name]['A'][:, :-1]
        ax2.plot(t[:-1], np.nanmean(A, axis=0), label=key, **LINE_STYLES[key])
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$P(F_t < k)$')
    ax1.set_title('Shortfall probability over time')
    ax1.legend(**LEGEND)
    setup_grid(ax1)
    ax2.axhline(1.0, color='0.4', ls=':', lw=1.5)
    ax2.set_xlabel('$t$')
    ax2.set_ylabel(r'mean $A(t,Y_t)$')
    ax2.set_title('Average risky exposure over time')
    ax2.legend(**LEGEND)
    setup_grid(ax2)
    plt.tight_layout()
    savefig(fig, os.path.join(OUT_F, 'mc_shortfall_exposure.png'))

    # Replication error
    fig, ax = plt.subplots(figsize=FIGSIZES['single'])
    for name in ('es', 'var'):
        key = {'es': 'ES', 'var': 'VaR'}[name]
        err = np.abs(res[name]['F_repl'] - res[name]['F'])
        ax.plot(t, np.mean(err, axis=0), label=f'{key} mean', **LINE_STYLES[key])
        ax.plot(t, np.max(err, axis=0), color=COLORS[key], ls=':', lw=1.5,
                label=f'{key} max')
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$|F^{repl}_t - \Psi(t,Y_t)|$')
    ax.set_yscale('log')
    ax.set_title('Discrete self-financing replication error')
    ax.legend(**LEGEND)
    setup_grid(ax)
    plt.tight_layout()
    savefig(fig, os.path.join(OUT_F, 'mc_replication.png'))


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='smaller MC (fast smoke run)')
    args = ap.parse_args()

    apply_style()
    _mk()
    t_start = time.time()

    n_paths = 2000 if args.quick else SIM.DEFAULT_N_PATHS
    n_boot = 100 if args.quick else 500

    P.print_params()
    print()

    # ── Baseline joint solutions ──────────────────────────
    s_es = ES.solve_es()
    s_var = VaR.solve_var()
    print("Baseline ES joint solution")
    for kk in ('eps_min', 'eps_M', 'Y0', 'k_eps', 'c'):
        print(f"    {kk:8s} = {s_es[kk]:.8f}")
    budget_res = float(ES.psi(s_es['Y0'], s_es['k_eps'], s_es['c'])) - s_es['F0']
    from ldi.bs_utils import bs_put
    constr_res = (s_es['c'] * bs_put(s_es['Y0'], s_es['k_eps'], P.r_tilde,
                                     P.sigma_Y, P.T) - s_es['eps'])
    print(f"    budget residual     = {budget_res:+.3e}")
    print(f"    constraint residual = {constr_res:+.3e}")
    print()
    print("Baseline VaR joint solution")
    for kk in ('Y0', 'k_alpha', 'cost_min'):
        print(f"    {kk:8s} = {s_var[kk]:.8f}")
    var_budget_res = float(VaR.psi(s_var['Y0'], s_var['k_alpha'])) - s_var['F0']
    print(f"    budget residual = {var_budget_res:+.3e}")
    print(f"    VaR feasible    = {s_var['feasible']} "
          f"(C_VaR={s_var['cost_min']:.6f} vs F0={s_var['F0']}), "
          f"alpha_min={VaR.alpha_min():.6f}")
    print()

    # ── Mode A: cross-sectional ───────────────────────────
    print("[Mode A] cross-sectional figures ...")
    C.plot_cross_sectional(variant='A1',
                           save_path=os.path.join(OUT_X, 'A1_fixed_eps.png'))
    C.plot_cross_sectional(variant='A2',
                           save_path=os.path.join(OUT_X, 'A2_slack_eps.png'))
    C.plot_solution_map(variant='A1',
                        save_path=os.path.join(OUT_X, 'A1_solution_map.png'))
    C.plot_solution_map(variant='A2',
                        save_path=os.path.join(OUT_X, 'A2_solution_map.png'))
    C.plot_slack_grid(save_path=os.path.join(OUT_X, 'A2_delta_grid.png'))
    C.plot_eps_sensitivity(save_path=os.path.join(OUT_X, 'eps_sensitivity.png'))
    for d in (0.01, 0.02, 0.05):
        lo, hi = C.slack_binding_range(d)
        print(f"    A-2 delta={d:.2f}: constraint binds only on "
              f"F0 in ({lo:.4f}, {hi:.4f}); outside, eps_min+delta >= eps_M "
              f"-> slack, A=1")

    # ── Mode B: fixed claim ───────────────────────────────
    print("[Mode B] fixed-claim figures ...")
    C.plot_fixed_claim_A(s_es, s_var,
                         save_path=os.path.join(OUT_F, 'B1_A_vs_y.png'))
    C.plot_fixed_claim_F(s_es, s_var,
                         save_path=os.path.join(OUT_F, 'B2_A_vs_F.png'))

    # ── Common ────────────────────────────────────────────
    print("[Common] feasibility floor + claim functions ...")
    C.plot_claim_functions(s_es, s_var,
                           save_path=os.path.join(OUT_C, 'claim_functions.png'))
    C.plot_eps_min_muI(save_path=os.path.join(OUT_C, 'eps_min_muI.png'))

    # ── Sensitivity (both modes) ──────────────────────────
    print("[Sensitivity] scanning ...")
    sens_rows = []
    for param in C.SENS_CONFIGS:
        _, recs = C.plot_sensitivity(
            param, save_path=os.path.join(OUT_F, f'sens_{param}.png'))
        C.plot_sensitivity_cross(
            param, save_path=os.path.join(OUT_X, f'sens_{param}.png'))
        sens_rows.extend(recs)
        for r in recs:
            flag = 'OK' if r['feasible'] else 'INFEASIBLE'
            print(f"    {param}={r['value']:<6g} eps_min={r['eps_min']:.4f} "
                  f"eps_M={r['eps_M']:.4f}  {flag}")

    with open(os.path.join(RES, 'sensitivity.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(sens_rows[0].keys()))
        w.writeheader()
        w.writerows(sens_rows)

    # ── Monte Carlo (fixed claim) ─────────────────────────
    print("[MC] equal-CE matching ...")
    match_ce = SIM.match_alpha_equal_ce(n_paths=n_paths)
    match_th = SIM.match_alpha_threshold()
    print(f"    equal-CE alpha   = {match_ce['alpha']:.6f} "
          f"(target CE loss {match_ce['target_ce_loss']:.4f}%)")
    print(f"    threshold-matched alpha = {match_th['alpha']:.6f} "
          f"(k_alpha={match_th.get('k_alpha', float('nan')):.6f} "
          f"vs k_eps={match_th['k_eps']:.6f})")

    print("[MC] main run ...")
    res = SIM.run(n_paths=n_paths, n_boot=n_boot)
    res_ce = SIM.run(n_paths=n_paths, n_boot=n_boot, alpha=match_ce['alpha'],
                     models=('merton', 'var'))
    res_th = SIM.run(n_paths=n_paths, n_boot=n_boot, alpha=match_th['alpha'],
                     models=('merton', 'var'))
    mc_figures(res)

    # ── Table 2 ───────────────────────────────────────────
    rows = []
    def _row(label, rec):
        s, se = rec['stats'], rec['se']
        d = dict(strategy=label)
        for kk in s:
            d[kk] = s[kk]
            d[kk + '_se'] = se[kk]
        d['ce_loss_pct'] = rec['ce_loss_pct']
        d['q_shortfall'] = rec['q_shortfall']
        d['q_shortfall_se'] = rec['q_shortfall_se']
        d['repl_err_mean'] = rec['repl_err_mean']
        d['repl_err_max'] = rec['repl_err_max']
        return d

    rows.append(_row('Merton', res['merton']))
    rows.append(_row(f"ES (eps={P.epsilon})", res['es']))
    rows.append(_row(f"VaR (alpha={P.alpha})", res['var']))
    rows.append(_row(f"VaR equal-CE (alpha={match_ce['alpha']:.4f})",
                     res_ce['var']))
    rows.append(_row(f"VaR threshold-matched (alpha={match_th['alpha']:.4f})",
                     res_th['var']))
    with open(os.path.join(RES, 'table2_mc.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("Table 2 (fixed-claim MC)")
    hdr = (f"{'strategy':<38}{'mean':>8}{'std':>8}{'P(F<k)':>9}"
           f"{'E[(k-F)+]':>11}{'CVaR5':>8}{'CE':>8}{'CEloss%':>9}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['strategy']:<38}{r['mean']:>8.4f}{r['std']:>8.4f}"
              f"{r['shortfall_prob']:>9.4f}{r['exp_shortfall']:>11.4f}"
              f"{r['cvar05']:>8.4f}{r['CE']:>8.4f}{r['ce_loss_pct']:>9.3f}")

    # ── Convergence check ─────────────────────────────────
    print()
    print("[MC] convergence check (N x2, steps x2) ...")
    conv = []
    base = dict(n_paths=n_paths, n_steps=SIM.DEFAULT_N_STEPS)
    variants = [('base', base),
                ('N x2', dict(n_paths=2 * n_paths, n_steps=SIM.DEFAULT_N_STEPS)),
                ('steps x2', dict(n_paths=n_paths, n_steps=2 * SIM.DEFAULT_N_STEPS))]
    for label, kw in variants:
        r = SIM.run(with_se=False, **kw)
        for name in ('es', 'var'):
            conv.append(dict(variant=label, model=name, **kw,
                             mean=r[name]['stats']['mean'],
                             shortfall_prob=r[name]['stats']['shortfall_prob'],
                             exp_shortfall=r[name]['stats']['exp_shortfall'],
                             cvar05=r[name]['stats']['cvar05'],
                             CE=r[name]['stats']['CE'],
                             q_shortfall=r[name]['q_shortfall'],
                             repl_err_mean=r[name]['repl_err_mean'],
                             repl_err_max=r[name]['repl_err_max']))
    with open(os.path.join(RES, 'mc_convergence.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(conv[0].keys()))
        w.writeheader()
        w.writerows(conv)
    for c in conv:
        print(f"    {c['variant']:<9}{c['model']:<5} mean={c['mean']:.4f} "
              f"P(F<k)={c['shortfall_prob']:.4f} CE={c['CE']:.4f} "
              f"Qsf={c['q_shortfall']:.4f} repl={c['repl_err_mean']:.2e}")

    # ── Diagnostics ───────────────────────────────────────
    diag = [
        ('budget_residual_ES', budget_res),
        ('constraint_residual_ES', constr_res),
        ('budget_residual_VaR', var_budget_res),
        ('terminal_replication_error_ES_max', res['es']['repl_err_terminal']),
        ('terminal_replication_error_ES_mean', res['es']['repl_err_mean']),
        ('terminal_replication_error_VaR_max', res['var']['repl_err_terminal']),
        ('terminal_replication_error_VaR_mean', res['var']['repl_err_mean']),
        ('eps_min_baseline', s_es['eps_min']),
        ('eps_M_baseline', s_es['eps_M']),
        ('epsilon_baseline', P.epsilon),
        ('Y0_ES', s_es['Y0']),
        ('k_eps', s_es['k_eps']),
        ('c', s_es['c']),
        ('Y0_VaR', s_var['Y0']),
        ('k_alpha', s_var['k_alpha']),
        ('VaR_quantile_hedge_cost', s_var['cost_min']),
        ('VaR_alpha_min', VaR.alpha_min()),
        ('alpha_equal_CE', match_ce['alpha']),
        ('alpha_threshold_matched', match_th['alpha']),
        ('MC_N', n_paths),
        ('MC_steps', SIM.DEFAULT_N_STEPS),
        ('MC_seed', SIM.DEFAULT_SEED),
        ('MC_scheme', SIM.SCHEME),
        ('MC_bootstrap_reps', n_boot),
        ('Q_shortfall_ES_realized', res['es']['q_shortfall']),
        ('Q_shortfall_ES_se', res['es']['q_shortfall_se']),
        ('Q_shortfall_ES_geq_eps_min',
         bool(res['es']['q_shortfall'] >= s_es['eps_min'])),
        ('Q_shortfall_VaR_realized', res['var']['q_shortfall']),
        ('Q_shortfall_Merton_realized', res['merton']['q_shortfall']),
        ('r_tilde', P.r_tilde),
        ('sigma_Y', P.sigma_Y),
        ('m_P', P.m_P),
        ('Merton_total_weight', float(P.Pi_star.sum())),
    ]
    with open(os.path.join(RES, 'diagnostics.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['item', 'value'])
        for kk, v in diag:
            w.writerow([kk, v])

    print()
    print("Diagnostics -> results/diagnostics.csv")
    for kk, v in diag:
        print(f"    {kk:38s} {v}")
    print(f"\nDone in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
