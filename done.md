# Task Completion Checklist

작업 마무리 시 아래를 순서대로 수행한다.

## 1. 테스트
```bash
pytest tests/ -v
```
실패하는 테스트가 있으면 수정한다. 전부 통과할 때까지 진행하지 않는다.

## 2. Notes 업데이트
해당하는 항목만 업데이트한다:
- 모델링 결정을 내렸으면 → `notes/decisions.md`에 날짜와 함께 기록
- 버그를 발견/수정했으면 → `notes/bugs.md`에 증상, 원인, 해결법 기록
- 작업을 완료했거나 새 작업이 생겼으면 → `notes/todo.md` 업데이트

## 3. Common Mistakes 업데이트
이번 작업에서 실수나 주의사항을 발견했으면 CLAUDE.md의 Common Mistakes 섹션에 추가한다.

## 4. Commit & Push
```bash
git add -A
git commit -m "<한글로 간결한 커밋 메시지>"
git push
```
