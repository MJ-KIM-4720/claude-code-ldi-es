# Project Bootstrap (기존 코드 기반)

기존 ldi/ 패키지 코드가 이미 있다. 이 코드를 먼저 읽고 이해한 뒤,
빠진 부분만 보완한다. 기존 코드를 덮어쓰지 않는다.

## 1. 기존 코드 파악
- ldi/ 디렉토리의 모든 .py 파일을 읽는다
- params.py의 파라미터 값이 CLAUDE.md와 일치하는지 확인한다
- 각 모델의 API가 CLAUDE.md의 Model API 섹션과 일치하는지 확인한다

## 2. 빠진 디렉토리/파일만 생성
이미 존재하는 파일은 건드리지 않는다. 없는 것만 만든다:
- scripts/ (run_all.py를 기존 위치에서 이동하거나, 없으면 생성)
- tests/ (test_params.py, test_es_model.py, test_var_model.py)
- notes/ (decisions.md, bugs.md, todo.md)
- paper/figures/, paper/tables/
- outputs/
- pyproject.toml, .gitignore (없으면 생성)

## 3. 테스트 작성 (기존 코드 기반)
기존 코드를 실제로 실행해서 현재 출력값을 확인한 뒤,
Known Results 테이블과 비교하는 regression test를 작성한다:
- test_params.py: Merton ≈ 0.84, r_tilde ≈ -0.023, sigma_Y ≈ 0.0711
- test_es_model.py: y0별 ES A 값
- test_var_model.py: y0별 VaR A 값

## 4. Notes 초기화
- notes/decisions.md: 기존 코드의 설계 결정을 분석하여 기록
- notes/bugs.md: 빈 템플릿
- notes/todo.md: 향후 작업 항목

## 5. 검증
- pytest tests/ -v 실행하여 전부 통과 확인
- 기존 코드가 Known Results와 일치하는지 확인
- 불일치 시 원인을 분석하여 보고 (코드를 임의로 수정하지 않는다)

## 6. 마무리
- git add -A && git commit -m "프로젝트 구조 보완: 테스트, notes, 환경설정 추가"