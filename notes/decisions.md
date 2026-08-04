# 모델링 결정 기록

## 12. Exact statistics 전환 + α_min + δ_L (2026-08-04, 리뷰 2차)

**(a) Table 2를 MC → closed form 으로.** reference process가 lognormal이고
세 claim이 전부 `g(y)= c·y | k | y` 구간별 선형이므로 모든 terminal 통계가
truncated lognormal 조립으로 닫힌다. `ldi/exact_stats.py` 는 **하나의 공식**
으로 세 전략을 처리한다 (Merton은 c=1, k_low=k → 중간 구간이 비어 identity).
MC 파이프라인은 삭제하지 않고 검증 + Figure 8 용도로 유지.

**(b) equal-CE α를 exact 기준으로 재산정.** MC 기반 0.085618은 seed에
의존한다. exact CE로 풀면 **0.081178**. root-finder tol은 오더가 허용한
1e-5 대신 1e-12로 뒀다 — 공짜이고, 1e-5로는 보고할 5번째 소수가 안 잡힌다.

**(c) α_min closed form.** `λ = γσ_Y = √θᵀθ` 가 γ-free 라는 점이 핵심이라
γ 패널 전 행에서 α_min이 동일하다. brentq 해와 2e-17 이내 일치하는 것을
테스트로 고정했고, 기존 numeric 버전은 `alpha_min_numeric`으로 남겨 교차검증.

**(d) δ_L 채널 분리.** μ_I는 부채(β₁μ_I)와 IIB 초과수익(μ_I+R−r)을 동시에
움직이는 dual channel이다. β₀만 이동시키면 (μ_I 고정) 자산 쪽 θ, σ_Y, Π* 가
전부 불변이고 r̃만 움직인다 → `params.override_delta_L()`. 테스트에서 자산
불변을 assert 한다.

**(e) 원고 Table 2는 MC(N=10⁶ terminal draws), 캘리브레이션은 exact.**
(사용자 결정) claim이 경로에 `Y_T`로만 의존하므로 terminal 분포에서 직접
뽑는다 — 이산화 오차 0, N=10⁶이 저렴. 전 전략이 같은 표준정규를 공유(CRN).
전 항목이 closed form 대비 <10⁻³ (max 7.76e−4) 임을 assert 한다.
`table_exact_summary.tex` 생성은 취소.
**단 CE loss 열만 예외** — 퍼센트 포인트라 오차가 100배 증폭돼 ≈0.017 pp가
되고, 정의상 동일해야 할 ES/equal-CE VaR 행이 다르게 찍힌다. 그 열만
closed-form 값을 싣고 각주로 밝혔다 (CSV엔 둘 다 보관).

**(f) A_ES 계산식은 wedge form 유지** (결정 11-d). exact_stats는 A를 쓰지
않지만, 두 경로가 같은 claim 파라미터를 쓰므로 일관성이 유지된다.

## 11. Joint system + fixed claim 으로 전면 재설계 (2026-08) ★
아래 1~10 중 ES/VaR 해법과 MC 관련 항목은 이 결정으로 **대체되었다**.
상세 숫자는 저장소 루트 `NOTES.md` 참조.

**(a) Feasibility floor.** budget `E^Q[e^{-r̃T}F_T] ≤ F0` 와
`(k-F_T)^+ ≥ k-F_T` 로부터 `ε ≥ ε_min = max(k·e^{-r̃T} − F0, 0)`.
baseline에서 ε_min = 0.087629 이므로 **기존 baseline ε=0.05는 해가 없는
문제였다**. 새 baseline ε=0.10 (feasible 구간 (0.087629, 0.152614)).
→ `params.eps_min/eps_merton/eps_band`, `es_model.InfeasibleError`.

**(b) Joint system.** Y0는 F0가 아니다. `(Y0, k_ε)`는
`Ψ_ES(0,Y0)=F0` 와 `(k/k_ε)·Put(0,Y0,k_ε)=ε` 의 연립해.
두 식은 decouple된다: binding을 budget에 대입하면
`Y0 + Put(0,Y0,k) = F0 + ε` (LHS가 Y0에 대해 strictly increasing,
하한 k·e^{-r̃T}) → Y0를 먼저 풀고, 고정된 Y0에서 k_ε를 푼다.
nested bisection 불필요. VaR도 동일 (k_α = λ(α)·Y0 를 budget에 대입).

**(c) Fixed claim.** `A(t,y)=y·Ψ_y/Ψ` 는 t=0에 확정된 claim의 delta다.
경로 위나 그림 위에서 y가 바뀔 때마다 threshold를 재계산하면 매 상태에서
다른 claim을 가격하는 셈이 된다 → 금지. MC는 reference process Y만
exact GBM으로 굴리고 `F_t = Ψ(t,Y_t)` 로 매핑한다.

**(d) A_ES를 wedge form으로 계산.** `A = 1 − wedge/Ψ`,
`wedge = Ψ − y·Ψ_y = k·e^{-r̃τ}[N(−d2(k)) − N(−d2(k_ε))]` (정확한 항등식).
직접식 `y·Ψ_y/Ψ` 은 tail에서 상쇄오차로 1+O(1e−15)를 낼 수 있다.
wedge form은 상쇄가 없어 clamping 없이 A ≤ 1 이 기계정밀도로 성립.

**(e) VaR feasibility = quantile-hedging cost.**
`C_VaR(α) = k·e^{-r̃T}·N(d2(1,λ(α)))` (Neyman–Pearson: Q-측도에서 가장 싼
(1−α) 상태 집합 = {Y_T ≥ k_α}). baseline에서 0.766 < F0=1 로 여유롭게
feasible — ES의 빠듯한 floor와 대비되며 그 자체가 논문 포인트.

**(f) VaR matching = equal-CE 기본.** MC CE loss를 ES와 같게 만드는 α를
common random numbers 위에서 이분법으로 탐색 (α=0.0856). 기존
threshold matching (k_α=k_ε → α=0.1067)은 robustness로 병행 보고.

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
