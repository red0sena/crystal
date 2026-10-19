# ColBERT

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**
- 연도: 2020 · 주제: 토큰 단위 late interaction
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2004.12832) · [arXiv](https://arxiv.org/abs/2004.12832)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

오프라인 document 표현을 캐시하면서 세밀한 token 대응을 유지한다. 모든 query–document 쌍을 무거운 Cross-encoder로 다시 처리하는 비용을 줄인다.

## 논문에서 제안한 방식

Query와 document를 별도로 인코딩하되 한 벡터로 압축하지 않고 token embeddings를 유지한다. Query token마다 가장 관련된 document token과의 유사도를 모으는 MaxSim late interaction으로 점수를 만든다.

## 확인한 실험 근거

SIGIR 2020. MS MARCO 등 passage retrieval 평가. 논문은 BERT ranker에 비해 큰 속도·FLOPs 감소를 보고한다. 근거: 구조·retrieval 실험.

## 한계와 해석 범위

검색 관련성과 정답 타당성은 다르다. MaxSim만으로 부정·조건·수량·논리 관계를 충분히 판단한다고 가정하면 안 된다. 캐시 크기와 token-pair 비교 비용도 남는다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** 긴 문맥을 단일 벡터로 압축하지 않고 후보 내용과 token 수준 근거를 비교하는 참고다. 필요하면 late interaction 뒤 작은 학습 scorer를 두는 확장을 별도로 검증한다.

## 연결 문서

- [poly-encoders ](poly-encoders.md)
- [metaencoder ](metaencoder.md)

