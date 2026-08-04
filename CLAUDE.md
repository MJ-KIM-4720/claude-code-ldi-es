# CLAUDE.md

## Project Overview

**ES-Constrained Liability-Driven Investment (LDI) Model**

Academic research extending Jo et al. (2025) VaR-LDI model to Expected Shortfall (ES) constraints using the option-based approach of Kraft & Steffensen (2013). Target journal: JEDC.

The core contribution: ES eliminates gambling incentives that VaR creates for underfunded pension funds, by providing partial linear protection in the tail (g = cy, c > 1) instead of abandoning protection entirely (g = y).

## Repository Structure

```
.
├── CLAUDE.md                  # 이 파일 — Claude Code가 매 세션마다 읽는 프로젝트 정의서
├── NOTES.md                   # 2026-08 재계산 결과 총정리 (원고 반영용 숫자)
├── README.md                  # GitHub용 일반 프로젝트 설명
│
├── ldi/                       # 핵심 패키지
│   ├── __init__.py
│   ├── params.py              # 파라미터 + 파생 값 + eps_min/eps_merton/eps_band
│   ├── bs_utils.py            # Black-Scholes 함수 (put, digital put, deltas)
│   ├── es_model.py            # ES joint solver + fixed-claim A (InfeasibleError)
│   ├── var_model.py           # VaR joint solver + quantile-hedging feasibility
│   ├── simulate.py            # fixed-claim MC + 복제 검증 (검증/Figure 8 전용)
│   ├── exact_stats.py         # ★ closed-form terminal 통계 + exact equal-CE
│   ├── compare.py             # Mode A / Mode B / 민감도 / feasibility 그림
│   └── style.py               # 공통 figure 스타일
│
├── scripts/
│   ├── run_recompute.py       # ★ 전체 재계산 진입점 (그림 + CSV 전부)
│   ├── run_exact.py           # ★ exact 통계/표 + α_min + δ_L + Figure 시안
│   └── legacy/                # 구 방법론(단일 식) 스크립트 — 참고용, 실행 금지
│
├── tests/                     # pytest 기반 테스트 (146 passed)
│   ├── test_params.py         # 파라미터 + feasibility band 검증
│   ├── test_es_model.py       # joint system, wedge identity, no-gambling
│   ├── test_var_model.py      # joint system, quantile-hedging, gambling
│   ├── test_simulate.py       # fixed claim, 복제 오차, 제약 복원
│   └── test_exact_stats.py    # closed form vs 수치적분/MC, α_min, δ_L
│
├── notes/                     # 연구 노트 (매 작업 후 업데이트)
│   ├── decisions.md           # 모델링 결정과 근거
│   ├── bugs.md                # 버그 원인과 해결법
│   └── todo.md                # 작업 트래킹
│
├── paper/                     # JEDC 논문
│   ├── main.tex
│   ├── figures/               # 논문용 고해상도 figure (git 포함, 구 방법론 산출물)
│   └── tables/
│
├── results/                   # CSV 산출물 (git 포함)
│   ├── diagnostics.csv        # residual, 복제오차, baseline 해, MC 설정
│   ├── table2_exact.csv       # ★ Table 2 (exact, 원고용)
│   ├── headline_numbers.md    # ★ 원고 반영용 headline
│   ├── exact_vs_mc.md         # exact vs MC (±3 SE) 대조
│   ├── table_sensitivity_v2.* # Table 3 + α_min 열
│   ├── table_deltaL.*         # δ_L comparative statics
│   ├── table2_mc.csv          # Table 2 (MC, 검증용)
│   ├── sensitivity.csv        # config별 eps_min/eps_M/feasibility
│   ├── mc_convergence.csv     # N×2, steps×2 수렴 확인
│   └── legacy/                # 폐기된 구 결과 백업
│
├── outputs/                   # 그림 (cross_sectional/ fixed_claim/ common/ 만 git 포함)
│
├── .devcontainer/
│   └── devcontainer.json      # Codespaces 환경 정의
├── .claude/
│   └── commands/              # Claude Code 슬래시 커맨드
│       ├── init.md            # /init — 프로젝트 부트스트랩
│       ├── done.md            # /done — 작업 마무리 체크리스트
│       └── validate.md        # /validate — Known Results 검증
├── .gitignore
└── pyproject.toml
```

---

## Key Parameters (CRITICAL)

```
R = 0.02  (real interest rate)
r = 0.04  (nominal risk-free rate)
```

**DO NOT swap R and r.** With correct values: Merton total ≈ 80.4%, r_tilde = -0.0084, sigma_Y = 0.0784. Swapping gives unrealistic Merton explosion.

Default constraint parameters: `alpha = 0.10` (VaR), `epsilon = 0.10` (ES), `T = 10`, `k = 1.0`, `gamma = 3.0`, `F0 = 1.0`.
(`epsilon = 0.05` was the pre-2026-08 value and is **infeasible** — below the floor `eps_min = 0.0876`.)

## Model API

Both models solve a **joint system** (budget + binding constraint) — the
reference-process start `Y0` is NOT the funding ratio `F0`.

```python
from ldi import es_model as ES, var_model as VaR, params as P, simulate as SIM

# Joint solve at t=0 → dict(Y0, k_eps, c, eps_min, eps_M, feasible, binding)
s = ES.solve_es(F0=1.0, eps=0.10)       # raises InfeasibleError if eps <= eps_min
s = ES.solve_es(F0=1.0, eps=0.05, strict=False)   # → feasible=False instead

sv = VaR.solve_var(F0=1.0, alpha=0.10)  # dict(Y0, k_alpha, cost_min, ...)

# FIXED CLAIM: solve once at t=0, then A varies with the reference state only.
A = ES.adjustment_factor(y, s['k_eps'], s['c'], tau=6.0)
A = VaR.adjustment_factor(y, sv['k_alpha'], tau=6.0)
pi_S, pi_I = ES.optimal_portfolio(y, s['k_eps'], s['c'], tau)

# Cross-sectional: one fund per F0, each solving its own joint system
A = ES.cross_sectional_A(F0=1.05, eps=0.10, strict=False)   # nan if infeasible

# Fixed-claim Monte Carlo (검증 + Figure 8 전용; 원고 수치는 exact 사용)
res = SIM.run(F0=1.0, n_paths=10_000, n_steps=120, seed=20260803)

# EXACT (closed form) — 원고 Table 2는 전부 여기서 나온다
from ldi import exact_stats as X
X.merton_stats(); X.es_stats(sol=s); X.var_stats(alpha=0.10)
X.match_alpha_equal_ce()['alpha']      # 0.081178 (seed 무관)
X.match_alpha_threshold()['alpha']     # 0.106663
VaR.alpha_min()                        # 0.015975  VaR feasibility bound
with P.override_delta_L(0.046):        # liability 채널만 이동 (자산 고정)
    ...
```

## Mathematical Notes

- **Joint system (ES):** `Psi_ES(0,Y0) = F0` and `(k/k_eps)·Put(0,Y0,k_eps) = eps`.
  They decouple: substituting the second into the first gives
  `Y0 + Put(0,Y0,k) = F0 + eps` — solve for `Y0` first, then `k_eps`.
- **Feasibility floor:** `eps_min = max(k·e^{-r̃T} - F0, 0)`. NO admissible
  strategy can go below it. Slack above `eps_M = Put(F0,k)`.
  Baseline band = (0.087629, 0.152614); **eps = 0.05 is infeasible**.
- **Joint system (VaR):** `Psi_VaR(0,Y0) = F0` with
  `k_alpha = Y0·exp(m_P·T + sigma_Y·sqrt(T)·Phi^{-1}(alpha))` (P-measure);
  substitute and solve one 1-D root in `Y0`.
- **VaR feasibility:** quantile-hedging cost
  `C_VaR(alpha) = k·e^{-r̃T}·N(d2(1,lambda))` must be < F0. Baseline 0.766 — far
  more slack than the ES floor, which is itself a result.
- **Adjustment factor:** `A = y·Psi_y/Psi` for a claim FIXED at t=0.
  Never re-solve the threshold as `y` moves.
- **Wedge identity (exact):**
  `Psi - y·Psi_y = k·e^{-r̃τ}[N(-d2(y,k)) - N(-d2(y,k_eps))] > 0`.
  `A_ES` is computed as `1 - wedge/Psi` — algebraically identical to the direct
  quotient but free of cancellation, so `0 < A_ES <= 1` holds to machine
  precision without clamping. U-shaped: `A → 1` at both `y→0` and `y→∞`.
- **VaR key property:** A > 1 possible for underfunded states (gambling
  incentive from the digital option); survives the joint-system correction.
- All Black-Scholes pricing uses liability-adjusted rate `r_tilde = r - (beta_0 + beta_1 * mu_I)`

## Conventions

- **Language:** Python 3.10+, numpy, scipy, matplotlib
- **Cross-sectional analysis** = different pension funds at t=0 with varying y0, each solving own threshold
- **Time-series analysis** = single fund over time, threshold fixed at t=0, Y evolves stochastically
- Figures saved to `outputs/` at 150 dpi
- Use `brentq` for ES threshold solving, closed-form for VaR threshold
- All monetary values are in funding ratio units (F = X/L, dimensionless)

## Known Results (Regression Baseline)

**2026-08 재계산 기준.** 이전 표(y0별 VaR A / ES A)는 단일 constraint 식만
풀던 구 방법론 산출물이라 폐기되었다 — `results/legacy/` 참조.

Baseline: `F0 = 1.0`, `eps = 0.10`, `alpha = 0.10`, `T = 10`, `k = 1`, `gamma = 3`.

| 항목 | 값 |
|------|-----|
| `eps_min` (feasibility floor) | 0.0876289 |
| `eps_M` (slack bound) | 0.1526135 |
| ES `Y0` | 0.8034200 |
| ES `k_eps` | 0.7275526 |
| ES `c` | 1.3744710 |
| ES `A(0, Y0)` | 0.5804 |
| VaR `Y0` | 0.9162059 |
| VaR `k_alpha` | 0.7149262 |
| VaR quantile-hedge cost | 0.7663776 |
| equal-CE matched alpha (**exact**) | 0.0811781 |
| threshold-matched alpha | 0.1066267 |
| `alpha_min` (VaR feasibility bound) | 0.0159749 |

**Exact (closed form) — 원고 Table 2 기준값:**

| 항목 | Merton | ES (ε=0.10) | VaR equal-CE (α=0.081178) |
|---|---|---|---|
| mean | 1.10562 | 1.01516 | 1.05716 |
| P(F_T<k) | 0.38935 | 0.24791 | 0.08118 |
| E[(k−F)⁺] | 0.05941 | 0.03249 | 0.03208 |
| 조건부 부족분 | 0.15258 | 0.13106 | 0.39524 |
| Q05 | 0.71310 | 0.78746 | 0.63363 |
| **Bottom-5% 평균** | 0.64554 | **0.71285** | 0.57360 |
| CE | 1.008236 | 0.985302 | 0.985302 |

Headline: 동일 CE loss 2.275% 에서 bottom-5% **+24.28%**, 조건부 부족분
**3.016배**. 전체는 `NOTES.md` §11, `results/headline_numbers.md`.

**이 값들은 regression test의 기준이다. 코드 수정 후 반드시 `pytest tests/ -v`
(146 passed) 로 확인할 것.**

---

## Workflow Rules

### Git 규칙
- 브랜치 네이밍: `feat/`, `fix/`, `paper/` 접두사 사용
- 커밋 메시지: 한글 OK, 간결하게 (예: "ES threshold solver 버그 수정")
- **작업 끝나면 반드시 commit & push** (Codespaces 환경이라 로컬 저장 안 됨)
- 태깅: 논문 제출 시점에 `v1.0-submission`, 리비전 시 `v1.1-revision` 등

### Notes 관리 (매 작업 후 업데이트 필수)
- `notes/decisions.md` — 모델링 결정과 근거 (왜 이 접근을 택했는지)
- `notes/bugs.md` — 버그 발견 시 증상, 원인, 해결법 기록
- `notes/todo.md` — 완료 항목 체크, 새 항목 추가

### Testing 규칙
- 코드 수정 후 `pytest tests/ -v` 통과 필수
- Known Results 테이블은 **절대 기준** — 값이 달라지면 코드가 잘못된 것
- 새 기능 추가 시 해당 테스트도 함께 작성

### Common Mistakes (실수 발생 시 여기 추가)
1. **R과 r 스왑 금지** — R=0.02(real), r=0.04(nominal). 바꾸면 Merton=650%로 폭발
2. **ES threshold solving** — brentq 구간을 너무 좁게 잡으면 수렴 실패. 초기 구간 [1e-6, k] 사용
3. **P-measure vs Q-measure 혼동** — VaR threshold는 P-measure, BS pricing은 Q-measure(r_tilde)
4. **Y0 ≠ F0** — reference process 시작점은 적립률이 아니다. claim이 `g(y) ≥ y`
   라 보호비용만큼 Y0 < F0. budget 식을 빼먹으면 예산 초과 전략을 "해"로 반환한다
5. **Feasibility 먼저 확인** — `eps <= eps_min` 이면 해가 존재하지 않는다.
   숫자가 나왔다고 해가 있는 것이 아니다 (`notes/bugs.md` 2026-08-03 참조)
6. **경로/그림 위에서 threshold 재계산 금지** — `A(t,y)`는 t=0에 고정된 claim의
   delta다. y마다 k_eps를 다시 풀면 매 상태에서 다른 claim을 가격하는 셈
7. **MC 수치를 원고에 쓰지 말 것** — equal-CE α가 seed에 따라 0.0856(MC) vs
   0.0812(exact)로 달라진다. terminal 통계는 전부 closed form이 있다
   (`ldi/exact_stats.py`)
8. **matplotlib에 LaTeX 이스케이프 금지** — usetex를 쓰지 않으므로 `\&`, `\%`가
   리터럴로 찍힌다. 일반 텍스트엔 `&`, `%` 그대로
9. **μ_I 민감도는 dual channel** — 부채(β₁μ_I)와 IIB 초과수익을 동시에 움직인다.
   liability 채널만 보려면 `P.override_delta_L()` (β₀만 이동)
10. **파라미터를 바꾸면 ε_min이 바뀐다** — 특히 μ_I, T (r̃ 경유). 민감도 분석에서
   baseline ε=0.10이 infeasible해지는 config가 실제로 존재 (μ_I≥0.03, T≥15)

<!-- Claude: 새로운 실수를 발견하면 번호를 이어서 여기에 추가해라 -->

### Before Finishing Any Task (체크리스트)
1. `pytest tests/ -v` 통과 확인
2. `notes/` 관련 파일 업데이트 (decisions, bugs, todo 중 해당하는 것)
3. Common Mistakes에 새로 발견한 이슈 추가 (해당 시)
4. `git add -A && git commit` (메시지는 한글, 간결하게)
5. `git push`

### 복잡한 작업 시 Plan First
- 새 기능 구현이나 큰 변경은 **먼저 계획을 세우고** 승인 받은 후 구현
- 작업이 꼬이면 무리하게 밀어붙이지 말고 **계획 단계로 돌아가서 재설계**
- 구현 후 "이게 맞는지 증명해봐" — Known Results와 비교, edge case 확인

---

## Common Tasks

- **Change parameters:** Edit `ldi/params.py` — derived quantities auto-compute on import.
  파라미터를 바꾸면 `eps_min`/`eps_M`이 함께 움직이므로 baseline ε이 여전히
  feasible한지 `P.print_params()` 로 확인할 것
- **Add sensitivity analysis:** `ldi/compare.py`의 `SENS_CONFIGS`에 항목 추가 →
  `sensitivity_scan()`이 config별로 ε_min/ε_M을 재계산하고 infeasible을 표시
- **Add new constraint model:** Create `ldi/new_model.py` mirroring `es_model.py` —
  반드시 **joint system** (budget + binding)으로 풀고 feasibility floor를 명시할 것
- **Monte Carlo:** `ldi/simulate.py`. reference process Y는 A와 무관한 exact GBM
  (drift `m_P`, vol `sigma_Y`), 적립률은 `F_t = Psi(t, Y_t)`. 경로 위 threshold 재계산 금지
- **전체 재계산:** `python3 scripts/run_recompute.py` (약 40초) → `outputs/` 확인 →
  채택 결정 후 `paper/figures/`로 복사
- **exact 표/headline 재생성:** `python3 scripts/run_exact.py` (약 40초).
  오더 기대값 23항목을 자동 검증하고 어긋나면 내부 정합성 audit을 출력한다
- **원고 수치는 MC가 아니라 exact를 쓸 것** — MC는 seed 의존적이다.
  `simulate.py`는 검증과 Figure 8 histogram 용도로만 유지한다

## References

- Jo, Kim, Jang (2025) — VaR + LDI + inflation risk (Applied Economics Letters)
- Kraft & Steffensen (2013) — Option-based VaR/ES (European J. Operational Research)
- Basak & Shapiro (2001) — VaR + ES constraints (Review of Financial Studies)
- Gabih, Grecksch, Wunderlich (2005) — Expected Loss constraint (Stochastic Analysis and Applications)
