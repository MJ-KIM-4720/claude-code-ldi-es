# 모델링 결정 기록

## 1. Option-Based Approach (Kraft & Steffensen 2013)
- VaR/ES 제약을 option payoff로 분해하여 closed-form 해 도출
- 장점: Monte Carlo 없이 정확한 최적 전략 계산 가능
- ES claim: `g(y) = y + Put(k) - c·Put(k_eps)` → 부분적 선형 보호

## 2. Liability-Adjusted Rate (`r_tilde`)
- `r_tilde = r - (beta_0 + beta_1 * mu_I) = -0.0084` (updated baseline)
- 부채 성장률이 명목금리보다 높아 음수 → 부채가 자산보다 빠르게 성장
- 모든 BS pricing에 r_tilde 사용 (Q-measure)

## 3. P-measure vs Q-measure 분리
- VaR threshold (`k_alpha`): P-measure drift `m_P` 사용 (실제 확률)
- Option pricing: Q-measure `r_tilde` 사용 (위험중립 가격)
- 혼동 시 결과가 완전히 달라짐 → Common Mistakes #3

## 4. ES Threshold Solver
- `brentq` 사용, 구간 `[1e-12, k-1e-12]`
- binding condition: `(k/k_eps) · Put(y0, k_eps) = epsilon`
- 구간이 좁으면 수렴 실패 → Common Mistakes #2

## 5. Cross-Sectional vs Time-Series 분석
- Cross-sectional: 각 펀드가 자체 threshold 풀어서 A 계산 (다른 y0)
- Time-series: 하나의 펀드가 t=0에서 threshold 고정, Y 변화에 따라 A 변화

## 6. Baseline 파라미터 업데이트 (2026-02)
- 변경: sigma_S=0.18, sigma_I=0.07, rho=-0.15, beta0=0.03, beta1=0.8, T=10, eps=0.05, alpha=0.1
- 유지: mu_S=0.08, mu_I=0.023, R=0.02, r=0.04, gamma=3, k=1.0
- 결과: r_tilde=-0.0084, sigma_Y=0.0784, Merton=80.4%
- 이유: 더 현실적인 파라미터 조합으로 업데이트

## 7. Parameter Recomputation (`override_params`)
- 민감도 분석을 위해 `params.py`에 `recompute_derived()`와 `override_params()` context manager 추가
- `override_params(GAMMA=5.0)`으로 임시 파라미터 변경 → 모든 파생량 자동 재계산
- context manager 종료 시 원래 값 복원 (finally 블록으로 안전하게)

## 8. Monte Carlo 시뮬레이션 설계 (2026-02)
- **Time-series 방식**: threshold를 t=0에서 한번만 solve, 이후 Y 변화에 따라 A 동적 계산
- **P-measure dynamics**: `d ln(Y) = [r̃ + A·γ·σ²_Y - A²·σ²_Y/2] dt + A·σ_Y dW^P`
  - A가 1이면 Merton GBM, A<1이면 변동성 축소 (ES), A>1이면 변동성 확대 (VaR gambling)
- **벡터화**: adjustment factor를 numpy 배열 연산으로 구현 (스칼라 루프 대비 ~1000배 속도 향상)
- **설정**: 10,000 paths, 250 steps (25 steps/year), 3 시나리오 (y0=0.8, 1.0, 1.2)
- **핵심 발견**:
  - y0=0.8(underfunded): ES는 매우 보수적 (std=0.04), VaR는 공격적 (std=0.11) → gambling incentive 확인
  - y0=1.2(overfunded): 세 모델 거의 동일 → 제약 비결합 확인
- **A clamp**: A를 [0, 5] 범위로 제한하여 수치 안정성 확보

## 10. A_ES Clamping 조사 (2026-02)
- **Explicit clamping 없음**: `min(A, 1.0)`, `np.clip(..., 1.0)` 등 ES A에 대한 상한 제한 코드 없음
- **MC의 `np.clip(A, 0.0, 5.0)`** (monte_carlo.py:120)은 시뮬레이션 안정화 목적, 정적 그래프와 무관
- **Implicit clamping**: `cross_sectional_A`에서 `if not binding: return 1.0` — 수학적으로 correct하지만 binding/non-binding 경계에서 미세한 불연속 발생 (0.999732 → 1.000000)
- **Kink 원인**: explicit clamping이 아닌 Put option의 내재적 비선형성과 cross-sectional 분석의 독립적 threshold solving 구조에서 비롯
- **A_ES 최대값**: 모든 파라미터 조합에서 A_ES <= 1.0 확인 (이론과 일치)

## 9. Welfare Analysis — Certainty Equivalent (2026-02)
- **CE 정의**: `CE = ((1-γ) · E[F_T^{1-γ}/(1-γ)])^{1/(1-γ)}`
  - γ=3일 때 (1-γ)=-2, 음수 거듭제곱 처리 필요
  - F_T=0인 path 제외 (γ>1이면 0^{1-γ}=∞)
- **Welfare cost**: `CE_loss = (CE_Merton - CE_model) / CE_Merton × 100%`
- **F0=1.0 baseline 결과**:
  - CE_ES = 0.9906, CE_VaR = 1.0055, CE_Merton = 1.0113
  - CE loss: ES = 2.05%, VaR = 0.57%
- **해석**: ES 제약이 VaR보다 welfare cost가 높지만, 이는 tail risk를 더 효과적으로 관리하는 대가
  - ES는 expected shortfall이 0.0878로 VaR(0.1023)이나 Merton(0.1508)보다 낮음
  - 즉, ES는 약간의 welfare를 희생하여 대규모 손실 방지
