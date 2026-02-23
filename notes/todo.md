# 작업 트래킹

## 완료
- [x] 기본 파라미터 모듈 (`params.py`)
- [x] Black-Scholes 유틸리티 (`bs_utils.py`)
- [x] ES 모델 구현 (`es_model.py`)
- [x] VaR 모델 구현 (`var_model.py`)
- [x] 비교 분석 및 plotting (`compare.py`)
- [x] 프로젝트 구조 보완 (tests, notes, scripts)

## 향후 작업
- [x] Monte Carlo 시뮬레이션 (`ldi/monte_carlo.py` + `scripts/run_monte_carlo.py`) — 벡터화 구현, 3 시나리오 × 4 figure = 12개 출력
- [x] 민감도 분석 스크립트 (`scripts/run_sensitivity.py`) — 11개 figure + summary CSV 생성 완료
- [x] baseline 파라미터 업데이트 (sigma_S=0.18, sigma_I=0.07, rho=-0.15, beta0=0.03, beta1=0.8, T=10, eps=0.05, alpha=0.1)
- [x] Welfare analysis (CE, welfare cost) 추가 — `monte_carlo.py`에 `certainty_equivalent()`, `run_monte_carlo.py`에 welfare 출력/figure
- [ ] 논문 figure 생성 및 `paper/figures/`로 복사
- [ ] 논문 본문 작성 (`paper/main.tex`)
