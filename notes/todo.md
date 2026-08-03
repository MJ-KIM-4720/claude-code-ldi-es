# 작업 트래킹

## 완료
- [x] 기본 파라미터 모듈 (`params.py`)
- [x] Black-Scholes 유틸리티 (`bs_utils.py`)
- [x] ES 모델 구현 (`es_model.py`)
- [x] VaR 모델 구현 (`var_model.py`)
- [x] 비교 분석 및 plotting (`compare.py`)
- [x] 프로젝트 구조 보완 (tests, notes, scripts)

## 2026-08 전면 재계산 (완료)
- [x] Feasibility floor `ε_min` 도출·구현, ε=0.05가 infeasible임을 확인
- [x] ES joint solver (budget + binding 연립, sequential) + `InfeasibleError`
- [x] VaR joint solver + quantile-hedging feasibility (`C_VaR`, `alpha_min`)
- [x] Baseline 재설계: `F0` 파라미터 추가, ε 0.05 → 0.10, ε 그리드 [0.09,0.15]
- [x] Fixed-claim MC 재작성 (`ldi/simulate.py`) + 복제오차 검증
- [x] equal-CE matching (α=0.0856) + threshold matching (α=0.1067)
- [x] Mode A / Mode B 그림 세트 + ε_min(μ_I) 신규 그림
- [x] Table 2 재계산 (SE, CVaR₅ 포함), convergence check
- [x] `results/diagnostics.csv`, `NOTES.md`
- [x] 구 결과 `results/legacy/`, 구 스크립트 `scripts/legacy/` 백업
- [x] 테스트 107개 통과

## 향후 작업 (재계산 후 남은 것)
- [ ] **Mode A / Mode B 중 논문 채택 결정** (사람이 판단 — `NOTES.md` §8)
- [ ] A-2를 쓸 경우 δ=0.05는 저적립 구간을 slack으로 만듦 → δ 재선택 검토
- [ ] baseline μ_I=0.023이 infeasibility 경계(0.0244)에 근접 — 본문에서 다룰지 결정
- [ ] 새 Proposition (feasibility floor) 논문 본문에 반영
- [ ] 채택된 그림을 `paper/figures/` 로 복사 (기존 figure는 구 방법론 산출물)
- [ ] `paper/tables/` 의 table_mc_summary.tex 등 재생성 (구 Table 2 기반)

## 이전 작업
- [x] Monte Carlo 시뮬레이션 (`ldi/monte_carlo.py` + `scripts/run_monte_carlo.py`) — 벡터화 구현, 3 시나리오 × 4 figure = 12개 출력
- [x] 민감도 분석 스크립트 (`scripts/run_sensitivity.py`) — 11개 figure + summary CSV 생성 완료
- [x] baseline 파라미터 업데이트 (sigma_S=0.18, sigma_I=0.07, rho=-0.15, beta0=0.03, beta1=0.8, T=10, eps=0.05, alpha=0.1)
- [x] Welfare analysis (CE, welfare cost) 추가 — `monte_carlo.py`에 `certainty_equivalent()`, `run_monte_carlo.py`에 welfare 출력/figure
- [ ] 논문 figure 생성 및 `paper/figures/`로 복사
- [ ] 논문 본문 작성 (`paper/main.tex`)
