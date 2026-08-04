# NOTES — 2026-08 전면 재계산 (joint system + fixed claim)

원고 반영용 숫자를 한 곳에 모은 문서. 모든 값은
`python3 scripts/run_recompute.py` 로 재생성되며, 원자료는
`results/diagnostics.csv`, `results/table2_mc.csv`,
`results/sensitivity.csv`, `results/mc_convergence.csv` 에 있다.

---

## 0. 무엇이 바뀌었나

| | 이전 (폐기) | 현재 |
|---|---|---|
| 해법 | constraint 식 하나만 풀어 `k_ε(y)` 를 매 점에서 재계산, `Y0 = F0` 대입 | budget + binding ES **연립해** `(Y0, k_ε)` |
| Baseline ε | 0.05 — **infeasible** (해 없음) | 0.10 |
| A(t,y) | y가 바뀔 때마다 threshold 재계산 | t=0에 고정된 claim의 delta |
| 산출물 | `k_ε`=0.796, `A(0.8)`=0.39, 구 Table 2 | 아래 전부 |

폐기된 결과는 `results/legacy/`, 그 결과를 만든 스크립트는
`scripts/legacy/` 에 백업되어 있다.

---

## 1. Feasibility floor (신규 Proposition)

Budget `E^Q[e^{-r̃T} F_T] ≤ F0` 와 `(k-F_T)^+ ≥ k-F_T` 로부터

```
E^Q[e^{-r̃T}(k-F_T)^+] ≥ k·e^{-r̃T} - F0    ⟹    ε_min = max(k·e^{-r̃T} - F0, 0)
```

어떤 admissible 전략도 이 아래로 갈 수 없다. 상한은 Merton claim이
소비하는 예산 `ε_M = Put(F0, k)` — 그 위로는 제약이 slack이다.

**Baseline (F0=1, k=1, r̃=−0.0084, T=10):**

| 항목 | 값 |
|---|---|
| `k·e^{-r̃T}` | 1.087629 |
| **ε_min** | **0.0876289** |
| **ε_M** | **0.1526135** |
| feasible & binding 구간 | **(0.087629, 0.152614)** |
| 채택한 baseline ε | **0.10** |

이전 baseline ε=0.05는 이 구간 밖 (floor보다 낮음) — 해가 존재하지 않는
문제를 풀고 있었다. `tests/test_es_model.py::test_old_baseline_is_infeasible`
가 이를 회귀 테스트로 고정한다.

## 2. Baseline joint solution

**ES (F0=1.0, ε=0.10):**

| 항목 | 값 |
|---|---|
| **Y0** | **0.8034200** |
| **k_ε** | **0.7275526** |
| **c = k/k_ε** | **1.3744710** |
| A_ES(0, Y0) | 0.5804 |
| budget residual | 0.0 |
| constraint residual | 1.1e−16 |

`Y0 = 0.803 < F0 = 1.0`: 보호가 붙은 claim이 reference process보다 비싸므로
reference는 F0보다 낮은 점에서 출발한다. 이 구분이 이번 수정의 핵심이다.

**VaR (F0=1.0, α=0.10):**

| 항목 | 값 |
|---|---|
| **Y0** | **0.9162059** |
| **k_α** | **0.7149262** |
| λ = k_α/Y0 | 0.7803119 |
| quantile-hedge cost `C_VaR(0.10)` | 0.7663776 |
| α_min (C_VaR = F0 가 되는 α) | 0.0159749 |
| budget residual | −1.1e−16 |

## 3. VaR feasibility — 논문 포인트가 되는 비대칭

확률 제약의 최소 비용은 quantile-hedging 비용
`C_VaR(α) = k·e^{-r̃T}·N(d2(1, λ(α)))` (Q-측도에서 가장 싼 (1−α) 상태에
k를 배달하는 비용; Neyman–Pearson).

- baseline에서 **C_VaR(0.10) = 0.766 < F0 = 1.0 → 여유롭게 feasible**.
  α를 0.016까지 조여야 비로소 예산에 걸린다.
- 반면 ES는 baseline에서 floor 0.0876이 예산의 8.8%를 이미 잡아먹고 있고,
  feasible 폭은 (0.0876, 0.1526) 밖에 안 된다.
- α→0 극한에서 `C_VaR → k·e^{-r̃T} = 1.0876` — 즉 ES의 floor와 정확히 같은
  full-defeasance 비용으로 수렴한다 (`test_cost_limit_is_full_defeasance`).

**해석:** 확률 제약은 "몇 %는 포기해도 된다"는 탈출구가 있어서 예산 제약과
잘 충돌하지 않는다. 손실 크기를 재는 ES에는 그 탈출구가 없다. VaR이
gambling을 유발한다는 기존 비판에, "VaR은 애초에 지키기 쉬운 제약"이라는
비용 측면의 대비가 추가된다.

## 4. VaR matching

> **[갱신됨 — §11 참조]** equal-CE α는 exact CE 기준 **0.081178** 로 대체되었다.
> 아래 0.0856181은 MC(seed 20260803) 기준값으로, seed에 의존하므로 원고에는
> 쓰지 않는다.

| 방식 | α (MC, 구) | α (exact, 채택) | 근거 |
|---|---|---|---|
| **equal-CE (기본)** | 0.0856181 | **0.081178** | CE loss를 ES와 동일하게 맞춤 |
| threshold matching (robustness) | 0.1066627 | 0.1066627 | `k_α = k_ε = 0.7275526` (MC 무관) |
| 명목 동일값 (참고) | 0.10 | 0.10 | — |

## 5. Table 2 — fixed-claim MC (SE 포함)

> **[갱신됨 — §11 참조]** 원고 Table 2는 exact(closed-form) 값으로 대체되었다.
> 아래 MC 표는 (a) exact 공식의 독립 검증, (b) Figure 8 histogram 용도로 유지한다.
> 모든 통계가 exact ± 3 SE 안에 들어온다 (`results/exact_vs_mc.md`).

N = 10,000 / steps = 120 (월별, T=10) / seed = 20260803 /
exact GBM scheme / bootstrap 500회. 괄호 안이 SE.

| 전략 | mean | std | P(F_T<k) | E[(k−F_T)^+] | 조건부 부족분 | Q05 | **CVaR₅** | CE | CE loss % |
|---|---|---|---|---|---|---|---|---|---|
| Merton | 1.1097 (0.0030) | 0.2807 (0.0027) | 0.3756 (0.0049) | 0.0586 (0.0010) | 0.1560 (0.0018) | 0.7083 (0.0038) | 0.6423 (0.0038) | 1.0105 (0.0028) | 0.000 |
| **ES (ε=0.10)** | 1.0160 (0.0016) | 0.1500 (0.0023) | 0.2409 (0.0043) | **0.0325** (0.0008) | **0.1351** (0.0021) | **0.7822** (0.0042) | **0.7093** (0.0041) | 0.9850 (0.0016) | 2.522 |
| VaR (α=0.10) | 1.0713 (0.0024) | 0.2270 (0.0026) | 0.1004 (0.0031) | 0.0365 (0.0011) | 0.3638 (0.0019) | 0.6490 (0.0035) | 0.5885 (0.0034) | 0.9917 (0.0029) | 1.865 |
| **VaR equal-CE (α=0.0856)** | 1.0609 (0.0023) | 0.2157 (0.0026) | 0.0887 (0.0030) | 0.0343 (0.0011) | 0.3871 (0.0019) | 0.6347 (0.0034) | 0.5755 (0.0034) | 0.9850 (0.0029) | 2.522 |
| VaR threshold-matched (α=0.1067) | 1.0749 (0.0025) | 0.2318 (0.0026) | 0.1074 (0.0031) | 0.0379 (0.0011) | 0.3531 (0.0019) | 0.6544 (0.0035) | 0.5934 (0.0035) | 0.9935 (0.0029) | 1.688 |

### Headline 후보 (equal-CE 비교 = 복지 비용을 통제한 비교)

1. **CVaR₅: 0.7093 vs 0.5755 — ES가 +0.1338 (+23.3%) 높다.**
   같은 CE 비용(2.52%)을 내고 최악 5% 구간의 평균 적립률이 이만큼 개선된다.
   SE가 각각 0.004 수준이므로 30 SE 이상 떨어진 차이.
2. **조건부 손실 깊이: 0.1351 vs 0.3871 — VaR 부족분이 2.9배 깊다.**
   VaR은 부족 사건 *빈도*(0.089 vs 0.241)는 낮추지만, 일단 터지면
   훨씬 크게 터진다. ES는 정반대로 빈도를 받아들이고 깊이를 자른다.
3. **Q05: 0.7822 vs 0.6347 (+0.1475).**
4. **무조건부 기대부족: Merton 대비 ES −44.5% (0.0586 → 0.0325).**
   equal-CE VaR 대비로는 −5.2% (0.0343 → 0.0325).
5. **VaR은 Merton보다 tail이 나쁘다**: CVaR₅ 0.5885 < Merton 0.6423.
   gambling incentive가 실제 tail 통계로 관측된다.

### 제약 복원 sanity check

| 검증 | 결과 |
|---|---|
| ES 실현 Q-shortfall | 0.10239 (SE 0.00354) vs ε = 0.10 — 0.7 SE 이내 ✓ |
| ES Q-shortfall ≥ ε_min | True (0.1024 ≥ 0.0876) ✓ |
| Merton 실현 Q-shortfall | 0.15403 (SE 0.0042) vs ε_M = 0.15261 ✓ |
| VaR 실현 P(F_T<k) | 0.1004 (SE 0.0031) vs α = 0.10 ✓ |

## 6. Replication & convergence

Discrete self-financing 경로
`F_{i+1} = F_i·[1 + A_i·(Y_{i+1}/Y_i − 1) + (1−A_i)(e^{r̃Δ}−1)]`
를 별도로 굴려 `Ψ(t,Y_t)` 와 비교.

| | mean err | max err |
|---|---|---|
| Merton | 4.0e−16 | 0 (A≡1이면 복제가 정확) |
| ES | 2.74e−03 | 0.0517 |
| VaR | 3.46e−03 | 0.2794 |

스텝 수를 늘리면 mean error가 정확히 1/√2 씩 감소 (60→960 steps:
3.93e−3 → 2.74e−3 → 1.94e−3 → 1.38e−3 → 9.68e−4).

**VaR의 max error는 스텝을 16배 늘려도 ~0.18 아래로 내려가지 않는다.**
digital option의 불연속 때문에 만기 근처에서 delta가 발산하고, 어떤
현실적 리밸런싱 빈도로도 복제되지 않는다. ES claim은 연속(kinked)이라
전 구간에서 수렴한다. 실행가능성 측면의 추가 대비로 쓸 수 있다.

**Convergence check 주의:** exact GBM scheme이므로 terminal 분포는
스텝 수와 무관하다 (law가 동일). `mc_convergence.csv` 의 steps×2 행에서
보이는 통계 변화는 discretization error가 아니라 서로 다른 난수 draw에서
오는 MC noise (모두 2 SE 이내). 스텝 수가 실제로 개선하는 것은 복제 오차뿐.
N×2 행 역시 모든 통계가 1 SE 내에서 일치.

## 7. Sensitivity — feasibility가 깨지는 config

파라미터가 바뀌면 r̃가 바뀌고 따라서 ε_min이 바뀐다. baseline ε=0.10
기준으로:

| param | value | r̃ | ε_min | ε_M | 상태 |
|---|---|---|---|---|---|
| γ | 2.0 | −0.0084 | 0.0876 | 0.2017 | OK (binding) |
| γ | 3.0 | −0.0084 | 0.0876 | 0.1526 | OK (baseline) |
| γ | 5.0 | −0.0084 | 0.0876 | 0.1153 | OK (binding) |
| γ | 8.0 | −0.0084 | 0.0876 | 0.0973 | **slack** (ε > ε_M → Merton) |
| μ_I | 0.010 | +0.0020 | 0.0000 | 0.0935 | **slack** |
| μ_I | 0.015 | −0.0020 | 0.0202 | 0.1060 | OK |
| μ_I | 0.023 | −0.0084 | 0.0876 | 0.1526 | OK (baseline) |
| μ_I | 0.030 | −0.0140 | 0.1503 | 0.2170 | **INFEASIBLE** |
| T | 5 | −0.0084 | 0.0429 | 0.0948 | **slack** |
| T | 10 | −0.0084 | 0.0876 | 0.1526 | OK (baseline) |
| T | 15 | −0.0084 | 0.1343 | 0.2068 | **INFEASIBLE** |
| T | 20 | −0.0084 | 0.1829 | 0.2603 | **INFEASIBLE** |
| ρ | −0.5 / −0.15 / 0 / 0.5 | −0.0084 | 0.0876 | 0.1731 / 0.1526 / 0.1490 / 0.1529 | 모두 OK |

- **μ_I > 0.024414 에서 baseline ε=0.10 이 infeasible** 이 된다.
  baseline μ_I = 0.023 은 이 경계에서 겨우 0.0014 떨어져 있다 —
  민감도 그림에 반드시 표시할 것. (`outputs/common/eps_min_muI.png`)
- `k·e^{-r̃T} = F0` 가 되는 임계 인플레이션은 **μ_I\* = 0.0125**.
  이보다 낮으면 floor 자체가 0 (부채가 자산보다 느리게 자라므로 target을
  무위험으로 완전히 defease 가능).
- γ, ρ 는 r̃를 건드리지 않으므로 ε_min을 바꾸지 않는다. 다만 σ_Y를 통해
  ε_M을 움직여서 slack 여부는 바뀐다 (γ=8).
- VaR는 위 config 전부에서 feasible.

## 8. 그림 인벤토리 (두 모드 모두 생성, 채택은 사람이 결정)

```
outputs/cross_sectional/   Mode A — x축 = F0, 각 점이 서로 다른 펀드
  A1_fixed_eps.png         A-1: ε=0.10 고정, feasible 구간(F0 > 0.9876)만
  A1_solution_map.png      (Y0, k_ε, c) vs F0
  A2_slack_eps.png         A-2: ε = ε_min(F0) + 0.05
  A2_solution_map.png
  A2_delta_grid.png        δ ∈ {0.01, 0.02, 0.05} 비교
  eps_sensitivity.png      ε ∈ [0.09, 0.15] 6개 점
  sens_{GAMMA,MU_I,T,RHO}.png

outputs/fixed_claim/       Mode B — 단일 펀드, x축 = reference state y
  B1_A_vs_y.png            A(t,y), t ∈ {0, 2.5, 5, 7.5} — U-shape
  B2_A_vs_F.png            F = Ψ(t,y) 보조축 + claim PV
  mc_fan.png               적립률 fan chart
  mc_terminal.png          terminal 분포
  mc_shortfall_exposure.png
  mc_replication.png       복제 오차 (log scale)
  sens_{GAMMA,MU_I,T,RHO}.png

outputs/common/            모드 구분이 없는 그림
  claim_functions.png      g_ES vs g_VaR
  eps_min_muI.png          신규 Proposition 시각화
```

### A-2 (δ-slack) 사용 시 주의

간격 `ε_M(F0) − ε_min(F0)` 는 F0에 대해 봉우리 모양이다 — 심하게
저적립인 펀드에서는 Merton put 프리미엄이 할인된 내재가치와 거의 같아
간격이 0으로 닫히고, 충분히 적립된 펀드에서는 put 자체가 무가치해진다.
따라서 고정 δ는 **양쪽 끝이 아니라 가운데 구간에서만 binding** 이다
(교차점이 두 개다):

| δ | binding 구간 |
|---|---|
| 0.01 | F0 ∈ (0.7846, 1.5676) |
| 0.02 | F0 ∈ (0.8501, 1.4388) |
| 0.03 | F0 ∈ (0.8950, 1.3604) |
| **0.05 (지시된 값)** | **F0 ∈ (0.9610, 1.2570)** |

δ=0.05 에서는 A-2 곡선이 F0 < 0.961 과 F0 > 1.257 에서 A≡1 (Merton)로
눕는다. 즉 "전 구간 커버"라는 A-2의 의도가 δ=0.05 에서는 달성되지
않는다 — 정작 관심 대상인 저적립 펀드들이 slack 영역에 들어간다.
모형의 성질이 아니라 δ 설계의 부작용이므로, A-2를 논문에 쓸 경우
δ를 0.01~0.02로 줄이는 편이 낫다 (`A2_delta_grid.png` 참조).

## 9. Diagnostics 요약

| 항목 | 값 |
|---|---|
| budget residual (ES) | 0.0 |
| constraint residual (ES) | 1.11e−16 |
| budget residual (VaR) | −1.11e−16 |
| terminal replication error (ES) | mean 2.74e−03 / max 5.17e−02 |
| terminal replication error (VaR) | mean 3.46e−03 / max 2.79e−01 |
| ε_min, ε_M (baseline) | 0.0876289, 0.1526135 |
| (Y0, k_ε, c) baseline | (0.8034200, 0.7275526, 1.3744710) |
| MC: N, steps, seed, scheme | 10,000 / 120 (월별) / 20260803 / exact GBM |
| bootstrap reps | 500 |

## 10. 재현

```bash
python3 -m pytest tests/ -v          # 107 passed
python3 scripts/run_recompute.py     # 약 37초, 그림 + CSV 전부 재생성
python3 scripts/run_recompute.py --quick   # 축소 MC 스모크 런
```

---

# 11. Review round 2 — exact statistics, VaR bound, δ_L (2026-08-04)

재생성: `python3 scripts/run_exact.py` (약 1분).
산출물: `results/table2_mc1e6.csv` + `table_mc_summary.tex` (**원고 Table 2**),
`table2_exact.csv` (closed-form 벤치마크), `table2_mc_vs_exact.md`,
`headline_numbers.md`, `exact_vs_mc.md`, `table_sensitivity_v2.{csv,tex}`,
`table_deltaL.{csv,tex}`.

> **오더 수정 반영 (사용자 결정).** 원고 Table 2는 **MC (N=10⁶ terminal draws)**
> 이다. `exact_stats`는 표를 직접 만들지 않고 (a) equal-CE α 캘리브레이션,
> (b) MC 검증 기준값, (c) `headline_numbers.md` 산출에만 쓴다.
> `table_exact_summary.tex` 생성은 취소했다 (§11.9).

## 11.1 왜 exact인가

Reference process가 P-측도에서 lognormal이고 세 claim이 모두
`g(y) = c·y | k | y` 형태의 구간별 선형함수이므로, terminal 통계 전부가
truncated lognormal 조립으로 닫힌 해를 갖는다. MC/수치적분 불필요.

```
Λ(a,K) = E[Y_T^a 1{Y_T<K}] = exp(am + a²s²/2)·Φ((lnK − m − as²)/s)
m = ln Y_0 + m_P·T,   s = σ_Y√T
ES: c = k/k_ε, k_low = k_ε | VaR: c = 1, k_low = k_α | Merton: c = 1, k_low = k
```

세 전략이 **하나의 공식**으로 처리된다 (`ldi/exact_stats.py`).
Merton은 중간 구간이 비어 identity claim이 된다.

## 11.2 Closed-form 기준값 (원고 Table 2는 §11.9의 MC 버전)

| 전략 | mean | std | P(F_T<k) | E[(k−F)⁺] | 조건부 | Q05 | **Bottom-5%** | CE | CE loss % |
|---|---|---|---|---|---|---|---|---|---|
| Merton | 1.10562 | 0.27838 | 0.38935 | 0.05941 | 0.15258 | 0.71310 | 0.64554 | 1.008236 | 0.000 |
| **ES (ε=0.10)** | 1.01516 | 0.14701 | 0.24791 | 0.03249 | **0.13106** | **0.78746** | **0.71285** | 0.985302 | **2.275** |
| VaR (α=0.10) | 1.06981 | 0.22387 | 0.10000 | 0.03611 | 0.36109 | 0.65335 | 0.59144 | 0.992132 | 1.597 |
| **VaR equal-CE (α=0.081178)** | 1.05716 | 0.20735 | 0.08118 | 0.03208 | 0.39524 | 0.63363 | 0.57360 | 0.985302 | **2.275** |
| VaR thr-matched (α=0.106663) | 1.07338 | 0.22858 | 0.10666 | 0.03741 | 0.35076 | 0.65880 | 0.59638 | 0.993970 | 1.415 |

**Headline (동일 CE 비용 2.275% 통제):**

1. **Bottom-5% 평균: 0.71285 vs 0.57360 → +24.28%**
2. **조건부 부족분: 0.13106 vs 0.39524 → VaR이 3.016배 깊다**
3. Q05: 0.78746 vs 0.63363 (+0.15383)
4. ES는 Merton 대비 E[(k−F)⁺]를 **45.31%** 줄인다 (0.05941 → 0.03249)
5. VaR의 bottom-5% (0.59144)는 **Merton (0.64554)보다도 나쁘다**

### 원고에서 반드시 짚어야 할 뉘앙스

equal-CE VaR 대비로는 ES가 **무조건부** E[(k−F)⁺]에서 이기지 못한다
(0.03249 vs 0.03208, VaR이 1.3% 낮다). P(F_T<k)도 VaR이 훨씬 낮다
(0.0812 vs 0.2479). ES의 우위는 부족의 **평균이 아니라 모양**에 있다 —
ES는 얕고 잦은 부족을, VaR은 드물고 깊은 부족을 만든다. 그래서 평균 기반
지표로 요약하면 두 제약이 동등해 보이고, tail 지표(bottom-5%, 조건부
부족분)에서만 차이가 드러난다. 본문은 tail 지표를 앞세우는 편이
정확하고 방어에도 유리하다.

## 11.3 오더 기대값과의 대조 — 3건 불일치, 전부 외부 반올림

`scripts/run_exact.py`가 23개 항목을 자동 검증한다. 22개 OK, 1개 DEVIATION.
외부 재계산치가 자기 자신의 다른 값과 모순되는 것이 원인이다.

오더가 제시한 Merton mean(1.10562)과 CE(1.008236)는 lognormal에서
(m_P·T, s)를 정확히 고정한다:
`ln CE = m_P·T − s²`, `ln mean = m_P·T + s²/2` → m_P·T = 0.0696716,
s = 0.2479301 (우리 값 0.0696703 / 0.2479277과 일치).

| 항목 | 오더 기대 | 우리 값 | 오더 자신의 입력으로 재계산 | 판정 |
|---|---|---|---|---|
| ES P(F<k) | 0.24803 | 0.247907 | **0.247908** | 오더 값이 자기 입력과 불일치 |
| ES std | 0.14712 | 0.147011 | **0.147011** | 〃 (허용오차 내라 OK 처리) |
| VaR k_α | 0.71477 | 0.714926 | Ψ_VaR = **0.99990** ≠ 1 | 예산식 위반 |

- ES P(F<k)/std: 오더가 준 (Y0, k_ε, c) = (0.803420, 0.727553, 1.374471) 과
  위 (m_P·T, s)를 그대로 넣으면 0.247908 / 0.147011 이 나온다. 즉 우리 값과
  같고, 기대값 쪽이 어긋난다. **원고에는 0.24791 / 0.14701 을 쓸 것.**
- VaR k_α = 0.71477 은 Y0_var = 0.916007 을 함의하는데, 그 점에서 예산식
  Ψ_VaR(0,Y0) = 0.99990 으로 1을 만족하지 않는다. 우리 값 0.714926
  (Y0 = 0.916206) 은 기계정밀도로 예산을 만족한다.

## 11.4 α_min — VaR feasibility bound

```
λ = γ·σ_Y = sqrt(θᵀθ)                          (γ와 무관!)
C_VaR(α) = k·e^{-r̃T}·Φ(Φ^{-1}(1−α) − λ√T)
α_min    = 1 − Φ(λ√T + Φ^{-1}(F0·e^{r̃T}/k))    if F0 < k·e^{-r̃T}, else 0
```

- baseline **α_min = 0.015975** (closed form과 brentq 해가 2e-17 이내 일치)
- σ_Y = √θ²/γ 이므로 λ = γσ_Y = √θ² 는 γ-free → **γ 패널·ε 패널 전 행에서
  α_min이 0.015975로 동일** (assert로 고정)
- 세 calibration (0.10, 0.081178, 0.106663) 전부 α_min 초과 ✓
- μ_I=0.010 → r̃ > 0 이라 target을 무위험으로 완전 defease 가능 → **α_min = 0**
- **μ_I=0.030 → ES는 infeasible (ε_min=0.1503 > 0.10) 이지만 VaR는
  α_min=0.02148 로 여유롭게 feasible** — 두 제약의 비대칭을 보여주는 핵심 행

Table 3 = `results/table_sensitivity_v2.tex` (기존 열 순서 유지 + A_VaR 뒤에
α_min 열 추가). 패널: γ, ε, μ_I, T, ρ. infeasible 행은 ES 열을 `---` 처리.
ε_min/ε_M/status 등 전체 필드는 동반 CSV에 있다.

## 11.5 δ_L comparative statics (liability 채널 분리)

μ_I 패널은 dual channel이다 — μ_I는 부채 성장률(β₁μ_I)과 IIB 초과수익
(μ_I+R−r)을 동시에 움직여 r̃, θ, σ_Y, Π* 가 전부 바뀐다. δ_L = β₀+β₁μ_I 를
직접 변수로 삼되 **β₀만 이동**시키면 (μ_I 고정) 자산 쪽은 전부 고정되고
r̃ = r − δ_L 만 움직인다 → `P.override_delta_L()`.

| δ_L | r̃ | ε_min | ε_M | status | α_min |
|---|---|---|---|---|---|
| 0.0400 | +0.00000 | 0.0000 | 0.0987 | **Slack** (ε_M < 0.10) | 0.00000 |
| 0.0430 | −0.00300 | 0.0305 | 0.1161 | Binding | 0.00425 |
| 0.0460 | −0.00600 | 0.0618 | 0.1356 | Binding | 0.01035 |
| 0.0484* | −0.00840 | 0.0876 | 0.1526 | Binding (baseline) | 0.01597 |
| 0.0520 | −0.01200 | 0.1275 | 0.1807 | **Infeasible** | 0.02534 |

전 행에서 σ_Y = 0.078402, Merton total = 0.804337 로 **완전히 동일**
(assert로 고정). 즉 위 변화는 순수하게 liability 채널이다.
ε_min은 δ_L에 대해 `e^{(δ_L−r)T} − 1` 로 거의 선형에 가깝게 증가하며,
δ_L이 baseline에서 0.0036 (약 7%) 오르는 것만으로 baseline ε=0.10이
infeasible이 된다 — 제약의 실행가능성이 부채 가정에 매우 민감함을 보여준다.

## 11.6 Figure 시안 (확정 보류, 기존 파일 유지)

| 파일 | 내용 |
|---|---|
| `outputs/common/eps_min_muI_v2.png` | Figure 6 + α_min(μ_I) twin axis. 스케일이 달라(예산 vs 확률) 오른쪽 축 분리. ES floor는 μ_I=0.0244에서 ε=0.10을 뚫지만 α_min은 전 구간 α=0.10 훨씬 아래 |
| `outputs/fixed_claim/mc_terminal_y010_inset.png` | Figure 8 (a): 좌측꼬리 zoom inset. connector 대신 원본 구간을 음영 처리 (connector가 패널을 가로질러 지저분함), F_T=k의 atom을 화살표로 명시 |
| `outputs/fixed_claim/mc_terminal_y010_cdf.png` | Figure 8 (b): empirical CDF. VaR의 α=0.10 평평한 구간(보호 포기 영역에 질량 없음)이 그대로 보이고, ES/VaR CDF 교차점 F_T≈0.861 아래에서 ES 질량이 더 적다 |
| `outputs/fixed_claim/mc_terminal_y010_cdf_atom.png` | Figure 8 (c): **추천.** CDF의 atom 점프를 양방향 화살표로 표시하고 수치를 병기 (ES 0.4781 / VaR 0.4283), 우측에 좌측꼬리 F_T ≤ 0.9 소패널 + CVaR₅ 마커. atom·꼬리·교차를 한 장에 담는다 |

세 시안 모두 **N = 10⁶ terminal draws** 로 생성한다.

경로 주의: 오더는 `outputs/` 직하를 지정했으나, `.gitignore`가 
`outputs/{cross_sectional,fixed_claim,common}` 만 추적하므로 직하 파일은
커밋되지 않는다. 원본 Figure 8과 같은 디렉터리에 두었다.

## 11.7 Figure 표기 정리

`ldi/compare.py`에 `PARAM_LABELS` / `param_label()` 추가. 제목·범례·infeasible
주석 박스까지 `GAMMA` → `$\gamma$`, `MU_I` → `$\mu_I$` 등 mathtext로 일괄 변환.
matplotlib은 usetex를 쓰지 않으므로 `\&`, `\%` 같은 LaTeX 이스케이프는 리터럴로
출력된다 — 일반 텍스트에는 `&`, `%` 를 그대로 쓸 것 (2건 수정).

## 11.8 MC 파이프라인

**삭제하지 않았다.** `ldi/simulate.py`, `scripts/run_recompute.py` 전부 유지.
용도가 (원고 수치) → (검증 + Figure 8 histogram)으로 바뀌었을 뿐이다.
`results/exact_vs_mc.md` 에 exact vs MC(±3 SE) 대조표가 있고 전 통계가 3 SE
이내다. 오더가 예고한 Merton P(F<k) 예외는 **+2.83 SE** 로 3 SE를 넘지 않는다
(exact 0.38935 vs MC 0.37560, SE 0.00487).

## 11.9 원고 Table 2 — terminal-draw MC (N = 10⁶)

경로가 필요 없다. claim은 경로에 오직 `Y_T`를 통해서만 의존하므로 terminal
분포에서 직접 뽑는다 (`SIM.terminal_draws`, exact lognormal, 이산화 오차 0).
그래서 N=10⁶이 저렴하다. 전 전략이 **같은 표준정규**를 공유한다 (CRN) —
Y0와 claim만 다르다.

seed = 20260803, equal-CE 행은 **exact α = 0.081178** 사용.

| 전략 | mean | std | P(F_T<k) | E[(k−F)⁺] | 조건부 | Q05 | Bottom-5% | CE | CE loss % |
|---|---|---|---|---|---|---|---|---|---|
| Merton | 1.1062 | 0.2784 | 0.3889 | 0.0593 | 0.1524 | 0.7137 | 0.6462 | 1.0088 | 0.000 |
| **ES (ε=0.10)** | 1.0155 | 0.1469 | 0.2473 | 0.0324 | **0.1308** | **0.7882** | **0.7136** | 0.9857 | 2.275 |
| VaR (α=0.10) | 1.0703 | 0.2238 | 0.0996 | 0.0359 | 0.3608 | 0.6539 | 0.5921 | 0.9928 | 1.597 |
| **VaR equal-CE (α=0.081178)** | 1.0576 | 0.2072 | 0.0807 | 0.0319 | 0.3950 | 0.6342 | 0.5742 | 0.9860 | 2.275 |
| VaR thr-matched (α=0.106663) | 1.0738 | 0.2286 | 0.1063 | 0.0373 | 0.3504 | 0.6594 | 0.5970 | 0.9946 | 1.415 |

### 정밀도 — 표 각주 문구를 한 군데 손봐야 한다

closed form 대비 **max |MC − exact| = 7.76e−4 < 10⁻³** (9개 열 × 5행 전부,
`results/table2_mc_vs_exact.md`). 오더가 준 문구
*"Simulation standard errors are below 10^{-3} for all entries and are omitted."*
는 **CE loss 열만 빼면 그대로 참**이다.

CE loss는 퍼센트 포인트라 100배 증폭돼 MC 오차가 **≈0.017 pp** 다 (10⁻³ 초과).
그대로 두면 정의상 동일해야 할 ES와 equal-CE VaR의 CE loss가 2.29 / 2.27 처럼
**다르게 찍혀 캘리브레이션이 실패한 것처럼 보인다.** 그래서 CE loss 열만
closed-form 값(2.275)으로 싣고 각주에 그 이유를 명시했다. CSV에는
`ce_loss_pct_mc`, `ce_loss_pct_exact` 둘 다 있으니 원하면 바꿀 수 있다.

현재 각주 문구:
> Simulation standard errors are below 10⁻³ for all entries and are omitted.
> The certainty-equivalent loss is reported at its closed-form value: it is the
> calibration target that defines the equal-CE row, and its Monte Carlo
> counterpart carries an error of about 2×10⁻² percentage points, which would
> make the two matched rows print differently.

### Atom 검산

`P(F_T = k)` 는 중간 구간(보호 성공 상태)의 질량이다. 이 구간 payoff가
정확히 k라 표본비율을 등식으로 셀 수 있다.

| 전략 | 표본비율 | 이론값 (Φ((ln k − m)/s) − p₁) | \|차\| |
|---|---|---|---|
| Merton | 0.000000 | 0.000000 | 0 (중간 구간 없음) |
| **ES** | 0.478075 | **0.478446** | 3.7e−4 |
| **VaR (0.10)** | 0.428345 | **0.428688** | 3.4e−4 |
| VaR equal-CE | 0.496474 | 0.496341 | 1.3e−4 |
| VaR thr-matched | 0.408258 | 0.408668 | 4.1e−4 |

오더가 제시한 ES 0.4784 / VaR 0.4287 과 일치한다.
**ES는 47.8%의 확률로 정확히 목표에 착지**하고 VaR는 42.9%다 — 보호가
성공하는 상태의 비중 자체가 ES 쪽이 크다.
