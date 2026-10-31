# Soft Contextualized Encoder

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **Soft Contextualized Encoder For User Defined Text Classification**
- 연도: 2026 · 주제: 새 판단 아키텍처 / 집합 비교
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2601.03450) · [arXiv](https://arxiv.org/abs/2601.03450)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

본문은 한 번 읽고, 가변 라벨 집합은 함께 비교한다. 표시 순서를 입력 특징으로 쓰지 않고, 라벨 관계와 본문 조건을 반영한다. 정답 라벨의 동의어를 학습 중 교체한다.

## 논문에서 제안한 방식

Frozen 외부 임베딩 모델이 본문 벡터 q를 만든다. 학습 가능한 affine map으로 폭을 맞추고, 라벨 embedding들과 함께 위치 정보 없는 Transformer Encoder에 넣는다. contextualized query–label dot product와 softmax로 분류한다. CE로 encoder와 adapter를 학습한다.

## 확인한 실험 근거

Jina-embedding-v3(570M)+RoBERTa-base(110M), 합계 약 680M. 뉴스 기반 학습 및 미학습 분류 데이터 평가. 근거: §2–3.

## 한계와 해석 범위

실제 연구는 단일 토큰 라벨을 선택했고 주로 뉴스 분류를 평가했다. 논문은 파라미터 수가 비슷한 모델의 latency가 비슷하다고 가정하므로 정확한 성능·속도 실측으로 확대해석하면 안 된다. 긴 본문을 하나의 frozen 벡터로 압축하는 병목이 있다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현이다.** 가까운 시험(g4b 위, 10-19): [Pointer head](../project/pointer-pilot.md) −5.05%p, [대체 readout·MLP](../project/readout-mlp.md)와 [잔차 Set Transformer](../project/architecture-search.md)는 개선 없음, [후보별 판정 Query](../project/decision-native-next.md) +0.05%p [−0.33, +0.41]. 이 제안 그대로는 미구현이다. Qwen 본문 표현과 독립 후보 표현을 작은 위치 없는 Transformer에서 비교하는 새 판단층의 가장 직접적인 참고다. 여러 토큰 후보 encoder와 다중 문맥 벡터로 확장하는 것은 우리의 제안이며 논문 결과와 구분한다.

## 연결 문서

- [set-transformer ](set-transformer.md)
- [qwen-next ](../designs/qwen-next.md)

