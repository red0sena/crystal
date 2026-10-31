# GLiClass

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **GLiClass: Generalist Lightweight Model for Sequence Classification Tasks**
- 연도: 2025 · 주제: 가변 라벨 내용 / 공동 인코딩
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2508.07662) · [arXiv](https://arxiv.org/abs/2508.07662)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

라벨을 고정된 출력 차원이 아닌 자연어 입력으로 표현한다. 본문–라벨 및 라벨–라벨 상호작용을 한 공동 인코딩에 반영한다.

## 논문에서 제안한 방식

Uni-encoder는 각 라벨 앞에 LABEL marker를 붙이고 본문과 함께 양방향 Encoder에 넣는다. 본문과 라벨 표현을 별도로 pooling하고 dot product 또는 공유 MLP로 점수를 얻는다. Bi-encoder·fused bi-encoder·encoder-decoder 변형도 제시한다. 층별 재가중 모듈과 보조 표현 학습도 설명한다.

## 확인한 실험 근거

DeBERTa v3와 ModernBERT 등을 사용한 zero/few-shot 분류 연구. Multi-label 처리와 PPO 학습도 다룬다. 근거: §2.1 및 논문 실험.

## 한계와 해석 범위

Joint encoder라고 해서 label 순열 대칭성을 자동으로 보장하지 않는다. 분류 연구이며 복잡한 객관식 reasoning 향상을 보장하지 않는다. main uni-encoder는 bidirectional이므로 causal Qwen에 marker만 추가해서는 동일 구조가 되지 않는다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현이다.** 가까운 시험(g4b 위, 10-19): [Pointer head](../project/pointer-pilot.md) −5.05%p, [대체 readout·MLP](../project/readout-mlp.md)와 [잔차 Set Transformer](../project/architecture-search.md)는 개선 없음, [후보별 판정 Query](../project/decision-native-next.md) +0.05%p [−0.33, +0.41]. 이 제안 그대로는 미구현이다. 분야·감정·문의 의도를 같은 가변 후보 인터페이스로 처리하는 데 적합한 참고다. Qwen 뒤에 candidate encoder와 공유 scorer를 붙이거나 제한된 공동 비교층을 두는 실험을 제안한다.

## 연결 문서

- [soft-contextualized-encoder ](soft-contextualized-encoder.md)
- [architecture-map ](../concepts/architecture-map.md)

