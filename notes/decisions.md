# 모델링 결정 기록

## 1. Option-Based Approach (Kraft & Steffensen 2013)
- VaR/ES 제약을 option payoff로 분해하여 closed-form 해 도출
- 장점: Monte Carlo 없이 정확한 최적 전략 계산 가능
- ES claim: `g(y) = y + Put(k) - c·Put(k_eps)` → 부분적 선형 보호

## 2. Liability-Adjusted Rate (`r_tilde`)
- `r_tilde = r - (beta_0 + beta_1 * mu_I) = -0.023`
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
