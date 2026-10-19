# dQwen3.5

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **dQwen3.5: Hybrid-Attention Diffusion Language Models**
- 연도: 2026 · 주제: hybrid 정보 흐름 / 주변 연구
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2609.20751) · [arXiv](https://arxiv.org/abs/2609.20751)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

Hybrid Backbone에서 모든 계층을 양방향으로 바꾸지 않고도 비인과 정보 흐름을 추가할 수 있다는 실제 연구 사례다.

## 논문에서 제안한 방식

Qwen3.5의 일부 full Attention을 bidirectionalize하고 Gated DeltaNet 등 순환 계층은 causal하게 유지하여 diffusion language model로 적응시킨다. minority bidirectional Attention으로 미래 정보를 전달하는 hybrid 구조다.

## 확인한 실험 근거

0.8B·2B·4B·9B를 포함한 모델군. 50B/100B adaptation tokens를 평가하며 hybrid 2B와 full Attention control을 비교한다. 근거: 구조 및 §4–5.

## 한계와 해석 범위

Diffusion 출력은 여러 denoising forward를 사용하는 생성 방식이다. 한 번 prefill로 선택 확률만 반환하는 decision model 논문이 아니다. 수십 B 토큰 적응 비용은 작은 선택층 학습과 다른 규모다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** Qwen3.5를 양방향으로 만드는 논의를 위한 주변 근거로만 사용한다. 이번 text decision 실험에 diffusion 모델을 도입하자는 추천은 아니다.

## 연결 문서

- [llm2vec ](llm2vec.md)
- [architecture-map ](../concepts/architecture-map.md)

