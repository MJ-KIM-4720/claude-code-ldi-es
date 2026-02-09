# Project Bootstrap

CLAUDE.md를 읽고 아래 작업을 순서대로 수행해라.

## 1. 디렉토리 구조 생성
CLAUDE.md의 Repository Structure 섹션을 참고하여 전체 디렉토리와 파일을 생성한다.
- ldi/ 패키지: __init__.py, params.py, bs_utils.py, es_model.py, var_model.py, compare.py
- scripts/: run_all.py, run_sensitivity.py, run_monte_carlo.py
- tests/: test_params.py, test_es_model.py, test_var_model.py
- notes/: decisions.md, bugs.md, todo.md (초기 템플릿 포함)
- paper/figures/, paper/tables/
- outputs/

## 2. 핵심 코드 구현
CLAUDE.md의 Key Parameters, Model API, Mathematical Notes를 기반으로 구현한다.
- params.py부터 시작 (R, r, 파생값 자동계산)
- bs_utils.py (put, digital put, deltas)
- es_model.py, var_model.py (Model API 섹션의 인터페이스를 정확히 따를 것)
- compare.py (cross-sectional, time-series 비교 플롯)
- 구현 순서는 의존성 순서를 따른다

## 3. 환경 설정 파일
- pyproject.toml (numpy, scipy, matplotlib, pytest 의존성)
- .gitignore (outputs/, __pycache__/, *.pyc, .ipynb_checkpoints/)

## 4. 테스트 작성
Known Results 테이블을 기반으로 regression test를 작성한다:
- test_params.py: Merton total ≈ 0.84 (±0.02), r_tilde ≈ -0.023 (±0.001), sigma_Y ≈ 0.0711 (±0.001)
- test_es_model.py: y0별 ES A 값 검증 (허용 오차 ±0.02)
- test_var_model.py: y0별 VaR A 값 검증 (허용 오차 ±0.02)

## 5. Notes 초기화
- notes/decisions.md: "# Modeling Decisions" 헤더 + 초기 설계 결정 기록
- notes/bugs.md: "# Bug Log" 헤더 + 빈 템플릿
- notes/todo.md: 초기 TODO 항목 (sensitivity analysis, MC simulation 등)

## 6. 검증 & 마무리
- `pytest tests/ -v` 실행하여 전부 통과 확인
- Known Results 값이 정확히 재현되는지 확인
- git add -A && git commit -m "프로젝트 초기 세팅: 전체 구조 + 핵심 모델 구현"
