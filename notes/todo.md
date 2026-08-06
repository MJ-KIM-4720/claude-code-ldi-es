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

## 2026-08 리뷰 2차 (완료)
- [x] Table 2 exact 전환 (`ldi/exact_stats.py`, truncated lognormal closed form)
- [x] equal-CE α exact 재산정 (0.085618 MC → **0.081178** exact)
- [x] α_min closed form + Table 3 열 추가 (γ-free 검증 포함)
- [x] δ_L comparative statics (liability 채널 분리, 자산 불변 assert)
- [x] Figure 표기 mathtext 일괄 정리 + Figure 6/8 시안 3종
- [x] 오더 기대값 23항목 자동 검증 (22 OK / 1 외부 반올림 불일치 규명)
- [x] 테스트 146개 통과
- [x] (오더 수정) 원고 Table 2를 terminal-draw MC N=10⁶ 로 재생성, exact는
      캘리브레이션·검증 전용으로 격하, `table_exact_summary.tex` 취소
- [x] atom 검산 (ES 0.4784 / VaR 0.4287 이론값과 표본비율 대조)
- [x] Figure 8 (c) 시안 추가 (CDF + atom 점프 + 좌측꼬리 소패널), 3종 모두 N=10⁶
- [x] 테스트 153개 통과
- [x] (후속) Table 3 원고 레이아웃 + α_min 열, ε 그리드 step 0.01 원복
- [x] (후속) Table 2의 CE·CE loss 두 열 closed form + 각주 교체
- [x] (후속) diagnostics.csv 재생성 (alpha_equal_CE → exact 0.081178)
- [x] (후속) Fig 8 (c) 및 sens γ/μ_I/T 를 paper/figures/ 로 승격
- [x] (마무리) Table 3 Slack 행 k_ε·c "---", 각주 ε < ε_min
- [x] (마무리) Table 2 행 라벨 α 4자리 (0.0812 / 0.1067)
- [x] (마무리) 원고 참조 그림 3종 재생성 (`scripts/run_paper_figures.py`)
- [x] (교정) 원고 3종을 원본 구성(단일 패널 · A_ES 오버레이)으로 재생성,
      2×2 ES-vs-VaR 버전은 `outputs/alt/` 로 이동 (appendix 후보)
- [x] (batch 3) fig_baseline_claim_function / fig_baseline_adjustment_factor
      재생성 (현행 캘리브레이션, 검산 통과)
- [x] (batch 3) fig_feasibility_map 신규 — (F₀, ε) phase diagram
- [x] (batch 3) mc_terminal_y010 우측 패널: Q5 점 + bottom-5% mean 수직 파선 분리
- [x] (batch 3) fig_B2_epsilon_compare → outputs/alt/
- [x] (Fig 5) mc_terminal_y010 을 equal-CE(α=0.081178) 기준·세로 2단·큰 폰트로 재생성
- [x] (Fig 5) results/table_eps_robust.csv — ε ∈ {0.10,0.12,0.14} equal-CE 재계산

## 향후 작업 (재계산 후 남은 것)
- [ ] **Mode A / Mode B 중 논문 채택 결정** (사람이 판단 — `NOTES.md` §8)
- [ ] A-2를 쓸 경우 δ=0.05는 저적립 구간을 slack으로 만듦 → δ 재선택 검토
- [ ] baseline μ_I=0.023이 infeasibility 경계(0.0244)에 근접 — 본문에서 다룰지 결정
- [ ] 새 Proposition (feasibility floor) 논문 본문에 반영
- [ ] **Figure 6 시안 채택 결정** (`outputs/common/eps_min_muI_v2.png`)
- [ ] 승격·재생성한 그림이 모두 Mode B (x = reference state y) 버전임을 확인 —
      Mode A 채택 시 `outputs/cross_sectional/sens_*.png` 로 교체 (NOTES §12, §12.5)
- [ ] ρ 패널(`fig_E1_rho_es.png`) 승격 여부 결정
- [ ] `outputs/alt/` 2×2 ES-vs-VaR 패널의 appendix 채택 여부 결정
- [ ] `paper/figures/` 잔여 구 방법론 그림 정리 — 아직 구 산출물:
      fig_A2_gamma_compare, fig_B1_epsilon_es, fig_C2_muI_components(+appendix),
      fig_E1_rho_es, fig_E2_rho_components, fig_baseline_allocation,
      fig_baseline_option_decomposition, fig_baseline_present_value,
      cross_sectional, time_series, eps_sensitivity, mc_fan/samples/shortfall
- [ ] 채택된 그림을 `paper/figures/` 로 복사 (기존 figure는 구 방법론 산출물)
- [ ] 본문에서 ES의 우위를 tail 지표로 서술할 것 — equal-CE VaR 대비 무조건부
      E[(k−F)⁺]는 ES가 오히려 근소하게 높다 (NOTES §11.2 뉘앙스 항목)
- [ ] `paper/tables/` 의 table_mc_summary.tex 등 재생성 (구 Table 2 기반)

## 이전 작업
- [x] Monte Carlo 시뮬레이션 (`ldi/monte_carlo.py` + `scripts/run_monte_carlo.py`) — 벡터화 구현, 3 시나리오 × 4 figure = 12개 출력
- [x] 민감도 분석 스크립트 (`scripts/run_sensitivity.py`) — 11개 figure + summary CSV 생성 완료
- [x] baseline 파라미터 업데이트 (sigma_S=0.18, sigma_I=0.07, rho=-0.15, beta0=0.03, beta1=0.8, T=10, eps=0.05, alpha=0.1)
- [x] Welfare analysis (CE, welfare cost) 추가 — `monte_carlo.py`에 `certainty_equivalent()`, `run_monte_carlo.py`에 welfare 출력/figure
- [ ] 논문 figure 생성 및 `paper/figures/`로 복사
- [ ] 논문 본문 작성 (`paper/main.tex`)
