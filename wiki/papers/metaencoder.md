# MetaEncoder

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **MetaEncoder: Exploring the Limit of Bi-Encoders for Multimodal System One Decision Making with Natural Language Interface**
- 연도: 2026 · 주제: 후보 캐시 / 대조 학습
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2610.11316) · [arXiv](https://arxiv.org/abs/2610.11316)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

후보를 자연어 내용으로 표현하고 재사용 가능한 embedding을 만든다. Closed-set에서는 request에 후보 목록도 넣어 전역 비교를 지원하고, 학습 중 일부 목록은 dropout한다. Open-set에서는 목록을 request에 넣지 않고 캐시·인덱스를 사용한다.

## 논문에서 제안한 방식

Muse-Glimmer 30B causal multimodal Backbone의 마지막 토큰 벡터를 정규화한다. 같은 네트워크로 request와 candidate를 별도로 인코딩하고 cosine score/temperature를 후보 사이 softmax로 정규화한다. request→candidate InfoNCE 학습을 사용한다.

## 확인한 실험 근거

11 benchmark suite, 190 tasks, closed-set decision과 multimodal/open-set retrieval을 비교. 근거: §3–4. 본문은 reasoning-intensive 과제의 한계를 함께 보고한다.

## 한계와 해석 범위

논문에서 decoder→encoder라는 역할 변경은 causal mask 유지와 양립한다. Cached 후보라면 request forward 한 번과 dot products로 점수화할 수 있지만 신규 후보의 인코딩은 별도 비용이다. Closed-set request에 후보 목록을 넣으면 전체 순서 독립성을 자동 보장하지 않는다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** 현재 Qwen causal Backbone을 유지하면서 LM Head를 content similarity로 대체하는 대조군이다. 후보 캐시가 유용한 반복 분류와 매번 후보가 다른 문제 풀이를 따로 측정해야 한다. 30B 결과를 0.8B에 전이된 성능으로 보지 않는다.

## 연결 문서

- [poly-encoders ](poly-encoders.md)
- [colbert ](colbert.md)

