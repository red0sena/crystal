# LLM 병합 비교: 단순한 Task Arithmetic만 꾸준히 이득

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md)

- 논문: **A Systematic Study of In-the-Wild Model Merging for Large Language Models** (Hitit, Girrbach, Akata; arXiv 판 제목 "A Systematic Study of Model Merging Techniques in Large Language Models")
- 연도: 2026 (TMLR), arXiv 2025-11 · 주제: 병합 방법 비교
- 원문: [arXiv](https://arxiv.org/abs/2511.21437)
- 조사 기록: 2026-10-18 웹 검색으로 초록 확인. 본문의 실험 절차는 확인하지 않았다.

## 확인한 실험 근거

- 오픈 LLM 4종에 대해, 모델마다 미세조정 체크포인트 12개를 병합했다. 병합 방법 6개를 벤치마크 16개로 비교했다.
- 가장 단순한 Task Arithmetic만 꾸준히 이득을 냈다. 간섭 정리·부분공간 방법(TIES 등)은 원본 대비 개선이 없거나 떨어졌다.

## 한계와 해석 범위

"in-the-wild" 조건이다. 즉 목표가 겹치거나 충돌하는 체크포인트들이다. 과제가 분명히 나뉜 경우에는 결과가 다를 수 있다.

## 우리 Qwen에 대한 의미

복잡한 병합(TIES, DARE, Breadcrumbs, TSV)은 시험하지 않았다. 단순 평균과 원본 보간을 기준선으로 유지한 근거다.

## 연결 문서

- [체크포인트 평균·Soup](checkpoint-averaging.md) · [Model Stock](model-stock.md)
