# Ordinal scale-utilization bias

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **More Choices, Fewer Decisions: Ordinal-Scale Bias in JEV-like Direct-Decision Models**
- 연도: 2026 · 주제: 등급 편향 / 진단
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2609.38827) · [arXiv](https://arxiv.org/abs/2609.38827)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

정확도가 괜찮아도 일부 등급만 반복 선택할 수 있다. 선택 순서를 섞는 것으로는 의미상 척도 활용 편향이 모두 사라지지 않는다.

## 논문에서 제안한 방식

JEV와 KEV 모델에서 ordinal 등급의 실제 사용 분포를 분석한다. 표시 위치와 정답 분포를 통제하고 같은 문항의 척도를 K=2…14로 세분화한다. targeted BA-LoRA post-training도 비교한다.

## 확인한 실험 근거

36 ordinal datasets에서 effective gold support 대비 최종 선택의 활용이 67–76%, 4 nominal tasks에서는 87–102%라고 보고한다. supervised scales에서 post-training으로 회복을 보고한다. 근거: 초록, §4.

## 한계와 해석 범위

Gold 빈도와 같은 비율로 예측하는 것이 개별 정확도를 항상 개선한다는 뜻은 아니다. 데이터·척도·모델별 효과를 따로 평가한다. 이 논문은 새 Backbone 제안이 아닌 진단·학습 연구다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** 난이도·품질 score에서 confusion matrix, 등급별 recall, MAE, 기대 점수 오차와 predicted/gold 분포를 함께 측정한다. 분야 분류는 nominal, 난이도는 ordinal로 구분한다.

## 연결 문서

- [reasoning-and-calibration ](../concepts/reasoning-and-calibration.md)
- [evaluation ](../designs/evaluation.md)

