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

| 방식 | α | 근거 |
|---|---|---|
| **equal-CE (기본)** | **0.0856181** | MC CE loss를 ES와 동일 (2.5219%)하게 맞춤 |
| threshold matching (robustness) | 0.1066627 | `k_α = k_ε = 0.7275526` 정확히 일치 |
| 명목 동일값 (참고) | 0.10 | — |

equal-CE 탐색은 common random numbers 위에서 이분법 — CE_loss(α)가 매끄러운
결정론적 함수가 되어 수렴이 안정적이다.

## 5. Table 2 — fixed-claim MC (SE 포함)

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
