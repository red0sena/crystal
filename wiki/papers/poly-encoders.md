# Poly-encoders

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring**
- 연도: 2020 · 주제: 공통 문맥 / 후보 비교
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/1905.01969) · [arXiv](https://arxiv.org/abs/1905.01969)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

Bi-encoder의 후보 캐시와 Cross-encoder의 후보별 문맥 비교 사이의 절충이다. 문맥 전체를 후보마다 다시 인코딩하는 비용을 줄인다.

## 논문에서 제안한 방식

문맥과 후보를 별도 Transformer로 인코딩한다. 후보는 하나의 벡터, 문맥은 학습된 m개 context code로 집계한 여러 벡터로 표현한다. 각 후보가 문맥 벡터에 Attention하여 후보에 맞는 문맥 표현을 얻고 최종 점수를 계산한다.

## 확인한 실험 근거

ICLR 2020. ConvAI2, DSTC7, Ubuntu V2, Wikipedia Article Search의 4개 응답 선택·검색 과제에서 비교. Cross-encoder보다 빠르고 Bi-encoder보다 높은 품질을 보고한다. 근거: §4.2–4.4 및 실험.

## 한계와 해석 범위

문맥 요약 벡터 개수 m에 따른 품질·비용 절충이 있다. 논문은 BERT 계열 기반이며 MMLU·Qwen3.5 장문 reasoning의 검증이 아니다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현이다.** 가까운 시험(g4b 위, 10-19): [Pointer head](../project/pointer-pilot.md) −5.05%p, [대체 readout·MLP](../project/readout-mlp.md)와 [잔차 Set Transformer](../project/architecture-search.md)는 개선 없음, [후보별 판정 Query](../project/decision-native-next.md) +0.05%p [−0.33, +0.41]. 이 제안 그대로는 미구현이다. Qwen이 읽은 문맥을 여러 벡터로 압축하고 후보별 작은 Attention과 scorer를 붙이는 설계의 참고다. 긴 문서의 세부 근거가 압축 중 소실되는지 검증해야 한다.

## 연결 문서

- [colbert ](colbert.md)
- [architecture-map ](../concepts/architecture-map.md)

