# Known Results Validation

CLAUDE.md의 Known Results 테이블과 현재 코드 출력을 비교 검증한다.

## 1. 파라미터 검증
```python
from ldi import params as P
```
확인 항목:
- Merton total weight ≈ 0.84
- r_tilde ≈ -0.023
- sigma_Y ≈ 0.0711

## 2. Cross-sectional A 검증
아래 값들을 계산하고 Known Results와 비교한다 (허용 오차 ±0.02):

| y0  | 기대 VaR A | 기대 ES A |
|-----|-----------|----------|
| 0.1 | 1.34      | 0.62     |
| 0.9 | 0.61      | 0.70     |
| 1.0 | 0.67      | 0.85     |
| 1.5 | 1.00      | 1.00     |

## 3. 핵심 성질 확인
- ES A ≤ 1 for all y0 (구조적 성질)
- VaR A > 1 가능 (underfunded일 때 gambling incentive)
- y0가 충분히 크면 (e.g., 1.5) 둘 다 non-binding → A = 1.0

## 4. 결과 보고
검증 결과를 테이블 형태로 출력하고, 불일치가 있으면 원인을 분석한다.
불일치 발견 시 코드를 수정하되, Known Results 테이블은 수정하지 않는다.
