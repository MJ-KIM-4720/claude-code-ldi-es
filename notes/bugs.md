# 버그 기록

<!-- 형식: ## 날짜 - 제목 / 증상 / 원인 / 해결 -->

## 2026-08-03 — infeasible한 ES 예산을 계속 풀고 있었다 ★

**증상**
`solve_threshold(y0=1.0, eps=0.05)` 가 아무 경고 없이 `k_eps=0.796` 을
반환. 겉보기에 정상적인 숫자라 오랫동안 발견되지 않았다.

**원인**
두 가지가 겹쳤다.
1. constraint 식 `(k/k_ε)·Put(y0,k_ε)=ε` **하나만** 풀고 budget 식
   `Ψ_ES(0,Y0)=F0` 를 쓰지 않았다. 그래서 `Y0=F0` 를 암묵적으로 대입한
   꼴이 되는데, claim이 `g(y) ≥ y` 라 Ψ(0,F0) > F0 — 예산을 초과하는
   전략을 "해"라고 부르고 있었다.
2. budget을 함께 쓰면 `ε > ε_min = max(k·e^{-r̃T} − F0, 0) = 0.0876` 이
   필요하다. ε=0.05 는 이 아래라 **애초에 해가 존재하지 않는다**.
   식 하나만 풀었기 때문에 그 사실이 드러나지 않았다.

**해결**
- `es_model.solve_es()` 가 연립해를 푼다 (binding을 budget에 대입 →
  Y0 먼저, 그 다음 k_ε).
- ε ≤ ε_min 이면 `InfeasibleError` 를 명시적으로 발생.
- `tests/test_es_model.py::test_old_baseline_is_infeasible` 로 회귀 고정.
- baseline ε을 0.10으로 변경. 구 결과 전부 `results/legacy/` 로 백업.

**교훈**
제약 최적화 문제에서 "해가 나왔다"는 것이 "해가 존재한다"의 증거가
아니다. 미지수 개수와 식 개수가 맞는지 먼저 확인할 것.

## 2026-08-03 — 시뮬레이션이 경로 위에서 threshold를 재계산

**증상**
MC 통계가 이론값과 어긋남 (특히 실현 Q-shortfall이 ε 근처로 오지 않음).

**원인**
`monte_carlo.simulate_paths` 가 Y 자체를 A에 의존하는 drift/vol로 굴렸다
(`d ln Y = [r̃ + A·γσ² − A²σ²/2]dt + A·σ dW`). 이는 reference process Y와
적립률 F를 한 프로세스로 뭉갠 것이다. 올바른 구조는 Y가 A와 무관한
exact GBM이고, `F_t = Ψ(t,Y_t)` 로 매핑되는 것.

**해결**
`ldi/simulate.py` 로 재작성. Y는 exact GBM (drift m_P, vol σ_Y), F는 Ψ로
매핑. 별도로 이산 자기금융 경로를 굴려 복제 오차를 보고하며, 스텝 2배마다
평균 오차가 1/√2로 감소함을 확인
(`test_replication_error_shrinks_with_steps`). 수정 후 실현 Q-shortfall
0.1024 (SE 0.0035) ≈ ε=0.10, 실현 P(F_T<k) 0.1004 ≈ α=0.10 으로 복원됨.

## 2026-08-03 — A-2 slack 교차점 solver가 nan 반환

**증상**
`slack_goes_binding_at(0.05)` 가 nan → 그림 범례에 "binds for F0 > nan".

**원인**
`ε_M(F0) − ε_min(F0)` 간격이 F0에 대해 봉우리 모양이라 교차점이 **두 개**
인데, 단일 bracket `[0.3, 3.0]` 으로 brentq를 호출했다. 양 끝에서 부호가
같아 bracket 실패.

**해결**
`slack_binding_range()` 로 교체 — 구간을 스캔해 부호 변화 지점을 모두 찾고
가장 바깥 두 개를 반환. δ=0.05 → (0.9610, 1.2570).

## 2026-08-03 — (버그 아님) A_ES가 1을 1e-15 넘는 현상

극단적 y에서 `N(-d2(k))` 와 `N(-d2(k_ε))` 가 둘 다 double precision에서
정확히 1.0(또는 0.0)이 되어 wedge가 0으로 underflow하고, 직접식
`y·Ψ_y/Ψ` 이 상쇄오차로 1+2e-16을 냈다. 모형의 성질이 아니라 부동소수점
한계. clamping 대신 대수적으로 동치인 wedge form `A = 1 − wedge/Ψ` 로
계산식을 바꿔 해결 (상쇄 없음 → A ≤ 1 이 기계정밀도로 성립).
