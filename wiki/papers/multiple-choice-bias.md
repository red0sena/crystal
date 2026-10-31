# 객관식 코드·위치 편향

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **Large Language Models Are Not Robust Multiple Choice Selectors**
- 연도: 2024 · 주제: 코드 prior / 객관식 안정성
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2309.03882) · [arXiv](https://arxiv.org/abs/2309.03882)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

A/B/C 확률은 선택지 내용만 반영하지 않으며 코드 prior와 serialization 효과가 섞일 수 있다.

## 논문에서 제안한 방식

선택지 표기와 순서에 따른 LLM 객관식 선택 편향을 분석하고, 추정한 선택 토큰 prior를 보정하는 PriDe 접근을 제시한다.

## 확인한 실험 근거

여러 LLM·객관식 benchmark에서 순서 변경·prior correction을 평가한다. 기존 대화에서 확인한 편향 연구의 배경 자료이며 이번 wiki 작성 시 재평가하지 않았다.

## 한계와 해석 범위

PriDe는 추론 보정 방식이다. 선택지 섞기 학습이나 새로운 content head의 효과를 이 논문의 직접 결과로 소개하지 않는다.

## 우리 Qwen에 적용한다면

내용 순서와 코드 매핑을 독립적으로 교차 변경하고 canonical option ID로 결과를 비교하는 진단의 근거다.

**측정함:** `bias_eval.py`(10-17)가 4순서 뒤집힘과 위치 쏠림(RStd)을 따로 잰다. 위치 쏠림은 작았다(g1 0.025, d3-lite 0.020, [두 요인 카드](option-bias-two-factors.md)). 10-25 순서 측정에서 selection 답 바뀜은 10.2%였고, 코드 자리 편향은 선택지 50개 과제에서만 보였다([순서 측정](../project/order-probe.md)). PriDe는 하지 않았다. 비슷한 prior 보정은 10-19에 −0.39~−4.69%p였다([다른 방법](../project/other-methods.md)).

## 연결 문서

- [candidate-independent-attention ](candidate-independent-attention.md)
- [evaluation ](../designs/evaluation.md)

