# RankT5

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses**
- 연도: 2022 · 주제: 스칼라 점수 / ranking loss
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2210.10634) · [arXiv](https://arxiv.org/abs/2210.10634)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

문장 생성 loss와 실제 사용할 ranking score 사이의 불일치를 줄인다. 단순히 출력 한 행을 읽는 것보다 점수 목적에 맞는 학습 목표가 중요하다는 사례다.

## 논문에서 제안한 방식

T5 encoder-decoder에서 생성 없이 unused token의 unnormalized logit를 점수로 사용한다. Encoder-only pooling+dense scalar 변형도 비교한다. Pointwise CE, pairwise logistic, listwise softmax CE, Poly1 ranking losses를 평가한다.

## 확인한 실험 근거

MS MARCO와 out-of-domain ranking 평가. 근거: §4, 그림 1 및 실험.

## 한계와 해석 범위

각 query–document pair를 처리하므로 후보 전체가 한 Backbone pass로 끝나거나 공통 문맥 공유가 자동 제공되는 구조는 아니다. Encoder-decoder T5의 decoder 제거를 decoder-only Qwen에 그대로 적용할 수 없다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** 공유 스칼라 scorer를 학습할 때 후보 점수에 직접 CE를 적용하는 기준과 복수 정답·관련성 등급의 ranking objectives를 비교한다.

## 연결 문서

- [architecture-map ](../concepts/architecture-map.md)
- [qwen-next ](../designs/qwen-next.md)

