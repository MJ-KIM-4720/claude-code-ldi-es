"""
논문용 LaTeX 테이블 생성
========================
Table 1: Baseline parameters (params.py에서 읽음)
Table 2: Sensitivity analysis summary (summary_table.csv에서 읽음)
Table 3: Monte Carlo summary statistics (MC 시뮬레이션 실행)

Usage:
    python scripts/generate_tables.py              # 전체 생성
    python scripts/generate_tables.py --no-mc      # MC 제외 (Table 1, 2만)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldi import params as P
from ldi import es_model as ES
from ldi import monte_carlo as MC

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

# CSV 경로 (sensitivity 결과)
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "result", "sensitivity",
                        "summary_table.csv")


# ═══════════════════════════════════════════════════════════
# Table 1: Baseline Parameters
# ═══════════════════════════════════════════════════════════

def generate_table_parameters():
    """params.py에서 값을 읽어 LaTeX 테이블 생성."""

    def fmt(v):
        """숫자를 적절한 형식으로 포맷."""
        if isinstance(v, float):
            if v == int(v):
                return str(int(v))
            # 작은 소수 → 소수점 유지
            return f"{v:g}"
        return str(v)

    tex = r"""\begin{table}[htbp]
\centering
\caption{Baseline parameter values}
\label{tab:parameters}
\begin{tabular}{llcl}
\toprule
Category & Parameter & Symbol & Value \\
\midrule
\multirow{2}{*}{Stock}
  & Expected return   & $\mu_S$    & """ + fmt(P.MU_S) + r""" \\
  & Volatility        & $\sigma_S$ & """ + fmt(P.SIGMA_S) + r""" \\
\midrule
\multirow{3}{*}{IIB}
  & Expected inflation & $\mu_I$    & """ + fmt(P.MU_I) + r""" \\
  & Volatility         & $\sigma_I$ & """ + fmt(P.SIGMA_I) + r""" \\
  & Real interest rate  & $R$        & """ + fmt(P.R) + r""" \\
\midrule
\multirow{2}{*}{Market}
  & Nominal risk-free rate  & $r$    & """ + fmt(P.r) + r""" \\
  & Stock--IIB correlation  & $\rho$ & $""" + fmt(P.RHO) + r"""$ \\
\midrule
\multirow{2}{*}{Liability}
  & Base growth rate       & $\beta_0$ & """ + fmt(P.BETA_0) + r""" \\
  & Inflation sensitivity  & $\beta_1$ & """ + fmt(P.BETA_1) + r""" \\
\midrule
\multirow{5}{*}{Optimization}
  & Risk aversion      & $\gamma$      & """ + fmt(P.GAMMA) + r""" \\
  & Target funding ratio & $k$         & """ + fmt(P.k) + r""" \\
  & ES budget           & $\varepsilon$ & """ + fmt(P.epsilon) + r""" \\
  & VaR level           & $\alpha$      & """ + fmt(P.alpha) + r""" \\
  & Investment horizon   & $T$          & """ + fmt(P.T) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    path = os.path.join(OUT, "table_parameters.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"  Table 1 saved: {path}")
    return tex


# ═══════════════════════════════════════════════════════════
# Table 2: Sensitivity Analysis Summary
# ═══════════════════════════════════════════════════════════

# Parameter display mapping: param_name → (LaTeX symbol, value formatter)
PARAM_DISPLAY = {
    "gamma":   (r"$\gamma$",      lambda v: f"{v:g}"),
    "epsilon": (r"$\varepsilon$", lambda v: f"{v:.2f}"),
    "mu_I":    (r"$\mu_I$",       lambda v: f"{v:.3f}"),
    "T":       (r"$T$",           lambda v: f"{v:g}"),
    "rho":     (r"$\rho$",        lambda v: f"{v:.2f}"),
}

# Desired parameter order and values
PARAM_ORDER = [
    ("gamma",   [2, 3, 5, 7]),
    ("epsilon", [0.02, 0.03, 0.05, 0.08, 0.10]),
    ("mu_I",    [0.01, 0.023, 0.035, 0.05]),
    ("T",       [5, 10, 15, 20]),
    ("rho",     [-0.30, -0.15, -0.05]),
]

# Map param_name to the keyword for override_params
PARAM_KWARG = {
    "gamma":   "GAMMA",
    "epsilon": "epsilon",
    "mu_I":    "MU_I",
    "T":       "T",
    "rho":     "RHO",
}


def _compute_k_eps(param_name, param_value, y0=0.8):
    """Compute ES threshold k_eps for given parameter override at y0."""
    kwarg = PARAM_KWARG[param_name]
    with P.override_params(**{kwarg: param_value}):
        k_eps, c, binding = ES.solve_threshold(y0)
    return k_eps


def generate_table_sensitivity():
    """summary_table.csv에서 F=0.8 데이터를 읽어 LaTeX 테이블 생성."""

    if not os.path.exists(CSV_PATH):
        print(f"  WARNING: {CSV_PATH} not found. "
              "Run scripts/run_sensitivity.py first.")
        return None

    df = pd.read_csv(CSV_PATH)

    # Build LaTeX rows
    rows = []
    target_F = 0.8

    for param_name, values in PARAM_ORDER:
        symbol, val_fmt = PARAM_DISPLAY[param_name]
        n_vals = len(values)

        for i, pval in enumerate(values):
            # Find closest F to target
            sub = df[(df["param_name"] == param_name) &
                     (np.isclose(df["param_value"], pval, atol=1e-6))]
            if sub.empty:
                continue
            idx = (sub["F"] - target_F).abs().idxmin()
            row = sub.loc[idx]

            # Compute k_eps
            k_eps = _compute_k_eps(param_name, pval, y0=target_F)

            # Format values
            A_es = f"{row['A_es']:.3f}"
            A_var = f"{row['A_var']:.3f}"
            total_es = f"{row['total_es']:.3f}"
            total_var = f"{row['total_var']:.3f}"
            k_eps_str = f"{k_eps:.4f}"

            # Build row
            if i == 0:
                param_cell = (r"\multirow{" + str(n_vals) + "}{*}{"
                              + symbol + "}")
            else:
                param_cell = ""

            rows.append(
                f"  {param_cell} & {val_fmt(pval)} & {k_eps_str} "
                f"& {A_es} & {A_var} & {total_es} & {total_var} \\\\"
            )

        rows.append(r"  \midrule")

    # Remove trailing \midrule and replace with \bottomrule
    if rows and rows[-1].strip() == r"\midrule":
        rows.pop()

    body = "\n".join(rows)

    tex = r"""\begin{table}[htbp]
\centering
\caption{Sensitivity of optimal allocations to model parameters ($F_0 = 0.8$)}
\label{tab:sensitivity}
\begin{tabular}{lc c cc cc}
\toprule
 & & & \multicolumn{2}{c}{Adjustment Factor} & \multicolumn{2}{c}{Total Allocation} \\
\cmidrule(lr){4-5} \cmidrule(lr){6-7}
Parameter & Value & $k_\varepsilon$ & $A_{\text{ES}}$ & $A_{\text{VaR}}$ & ES & VaR \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    path = os.path.join(OUT, "table_sensitivity_summary.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"  Table 2 saved: {path}")
    return tex


# ═══════════════════════════════════════════════════════════
# Table 3: Monte Carlo Summary Statistics
# ═══════════════════════════════════════════════════════════

N_PATHS = 10000
N_STEPS = 250
SEED = 42


def generate_table_mc():
    """MC 시뮬레이션을 실행하고 요약 통계 LaTeX 테이블 생성."""

    models = ["es", "var", "merton"]
    labels = {"es": "ES", "var": "VaR", "merton": "Merton"}
    y0_list = [1.0, 0.8]

    rows = []
    for y0 in y0_list:
        n_models = len(models)
        for i, model in enumerate(models):
            print(f"    MC: {labels[model]}, y0={y0} ...", end=" ", flush=True)
            paths, t_grid = MC.simulate_paths(
                y0, n_paths=N_PATHS, n_steps=N_STEPS,
                model=model, seed=SEED
            )
            ts = MC.compute_terminal_stats(paths)
            print(f"E[F_T]={ts['mean']:.3f}")

            # Format row
            if i == 0:
                y0_cell = (r"\multirow{" + str(n_models) + "}{*}{"
                           + f"{y0:.1f}" + "}")
            else:
                y0_cell = ""

            rows.append(
                f"  {y0_cell} & {labels[model]} "
                f"& {ts['mean']:.3f} & {ts['std']:.3f} "
                f"& {ts['shortfall_prob']:.3f} "
                f"& {ts['expected_shortfall']:.4f} "
                f"& {ts['median']:.3f} \\\\"
            )

        rows.append(r"  \midrule")

    # Remove trailing \midrule
    if rows and rows[-1].strip() == r"\midrule":
        rows.pop()

    body = "\n".join(rows)

    tex = r"""\begin{table}[htbp]
\centering
\caption{Monte Carlo summary statistics ($N = """ + f"{N_PATHS:,}" + r"""$, $T = """ + f"{P.T:g}" + r"""$)}
\label{tab:mc_summary}
\begin{tabular}{cl ccccc}
\toprule
 & & & & & Expected & \\
$F_0$ & Model & $\mathrm{E}[F_T]$ & $\mathrm{Std}[F_T]$ & $\mathrm{P}(F_T < k)$ & Shortfall & Median \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    path = os.path.join(OUT, "table_mc_summary.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"  Table 3 saved: {path}")
    return tex


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for JEDC paper")
    parser.add_argument("--no-mc", action="store_true",
                        help="Skip Monte Carlo table (Table 3)")
    args = parser.parse_args()

    print("=" * 55)
    print("  Generating LaTeX Tables")
    print("=" * 55)

    print("\n[Table 1] Baseline Parameters")
    generate_table_parameters()

    print("\n[Table 2] Sensitivity Analysis Summary")
    generate_table_sensitivity()

    if not args.no_mc:
        print("\n[Table 3] Monte Carlo Summary Statistics")
        generate_table_mc()
    else:
        print("\n[Table 3] Skipped (--no-mc)")

    print("\n" + "=" * 55)
    print(f"  All tables saved to: {os.path.abspath(OUT)}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
