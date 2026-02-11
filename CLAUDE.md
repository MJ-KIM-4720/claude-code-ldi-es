# CLAUDE.md

## Project Overview

**ES-Constrained Liability-Driven Investment (LDI) Model**

Academic research extending Jo et al. (2025) VaR-LDI model to Expected Shortfall (ES) constraints using the option-based approach of Kraft & Steffensen (2013). Target journal: JEDC.

The core contribution: ES eliminates gambling incentives that VaR creates for underfunded pension funds, by providing partial linear protection in the tail (g = cy, c > 1) instead of abandoning protection entirely (g = y).

## Repository Structure

```
.
├── CLAUDE.md                  # 이 파일 — Claude Code가 매 세션마다 읽는 프로젝트 정의서
├── README.md                  # GitHub용 일반 프로젝트 설명
│
├── ldi/                       # 핵심 패키지
│   ├── __init__.py
│   ├── params.py              # 모든 파라미터 + 파생 값 (r_tilde, sigma_Y, Pi_star 등)
│   ├── bs_utils.py            # Black-Scholes 함수 (put, digital put, deltas)
│   ├── es_model.py            # ES constrained model
│   ├── var_model.py           # VaR constrained model (Jo et al. 2025)
│   └── compare.py             # 비교 분석 & plotting
│
├── scripts/                   # 실행 스크립트
│   ├── run_all.py             # 전체 실행 진입점
│   ├── run_sensitivity.py     # 민감도 분석
│   └── run_monte_carlo.py     # MC 시뮬레이션
│
├── tests/                     # pytest 기반 테스트
│   ├── test_params.py         # 파라미터 검증 (Merton ≈ 80.4%, r_tilde ≈ -0.0084)
│   ├── test_es_model.py       # ES 모델 단위 테스트
│   └── test_var_model.py      # VaR 모델 단위 테스트
│
├── notes/                     # 연구 노트 (매 작업 후 업데이트)
│   ├── decisions.md           # 모델링 결정과 근거
│   ├── bugs.md                # 버그 원인과 해결법
│   └── todo.md                # 작업 트래킹
│
├── paper/                     # JEDC 논문
│   ├── main.tex
│   ├── figures/               # 논문용 고해상도 figure (git 포함)
│   └── tables/
│
├── outputs/                   # 코드 생성 figure/결과 (gitignore 대상)
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

Default constraint parameters: `alpha = 0.10` (VaR), `epsilon = 0.05` (ES), `T = 10`, `k = 1.0`, `gamma = 3.0`.

## Model API

Both ES and VaR models expose the same interface:

```python
from ldi import es_model as ES, var_model as VaR, params as P

# Cross-sectional: each fund solves its own threshold
A = ES.cross_sectional_A(y0=0.8, eps=0.10)    # returns scalar
A = VaR.cross_sectional_A(y0=0.8, alpha=0.05)

# Time-series: fund solves threshold once, A varies as Y evolves
k_eps, c, binding = ES.solve_threshold(y0=1.0)
A = ES.adjustment_factor(Y=0.9, k_eps=k_eps, c=c, tau=4.0)

k_alpha, binding = VaR.solve_threshold(y0=1.0)
A = VaR.adjustment_factor(Y=0.9, k_alpha=k_alpha, tau=4.0)

# Optimal portfolio weights: pi* = A · Pi_star
pi_S, pi_I = ES.optimal_portfolio(Y, k_eps, c, tau)
```

## Mathematical Notes

- **ES constraint:** `c · Put(y0, k_eps) = epsilon` where `c = k / k_eps`
- **VaR threshold:** `k_alpha = y0 · exp(m_P·T + sigma_Y·sqrt(T)·Phi^{-1}(alpha))` (P-measure)
- **Adjustment factor:** `A = (Y / Psi) · (dPsi/dy)` — multiplies Merton weights
- **ES key property:** A ≤ 1 always (structural, because g_ES ≥ Y everywhere)
- **VaR key property:** A > 1 possible for underfunded funds (gambling incentive from digital option)
- All Black-Scholes pricing uses liability-adjusted rate `r_tilde = r - (beta_0 + beta_1 * mu_I)`

## Conventions

- **Language:** Python 3.10+, numpy, scipy, matplotlib
- **Cross-sectional analysis** = different pension funds at t=0 with varying y0, each solving own threshold
- **Time-series analysis** = single fund over time, threshold fixed at t=0, Y evolves stochastically
- Figures saved to `outputs/` at 150 dpi
- Use `brentq` for ES threshold solving, closed-form for VaR threshold
- All monetary values are in funding ratio units (F = X/L, dimensionless)

## Known Results (Regression Baseline)

| y0   | VaR A | ES A  | Interpretation                          |
|------|-------|-------|-----------------------------------------|
| 0.1  | 1.80  | 0.29  | VaR gambles, ES conservative            |
| 0.9  | 0.89  | 0.50  | VaR near Merton, ES moderate            |
| 1.0  | 0.92  | 0.65  | Both constrained, ES more so            |
| 1.5  | 1.00  | 1.00  | Both non-binding → Merton              |

**이 값들은 regression test의 기준이다. 코드 수정 후 반드시 `/validate`로 확인할 것.**

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

- **Change parameters:** Edit `ldi/params.py` — derived quantities auto-compute on import
- **Add sensitivity analysis:** Add function in `ldi/compare.py`, follow `plot_eps_sensitivity` pattern
- **Add new constraint model:** Create `ldi/new_model.py` mirroring `es_model.py` API
- **Monte Carlo simulation:** Use P-measure drift `m_P` and vol `A · sigma_Y` for fund dynamics
- **논문 figure 업데이트:** `scripts/run_all.py` 실행 → `outputs/`에서 확인 → 확정되면 `paper/figures/`로 복사

## References

- Jo, Kim, Jang (2025) — VaR + LDI + inflation risk (Applied Economics Letters)
- Kraft & Steffensen (2013) — Option-based VaR/ES (European J. Operational Research)
- Basak & Shapiro (2001) — VaR + ES constraints (Review of Financial Studies)
- Gabih, Grecksch, Wunderlich (2005) — Expected Loss constraint (Stochastic Analysis and Applications)
