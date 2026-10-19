# LLM2Vec

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders**
- 연도: 2024 · 주제: Decoder를 표현 모델로 전환
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2404.05961) · [arXiv](https://arxiv.org/abs/2404.05961)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

사전학습한 생성형 Backbone의 지식을 문장 인코딩·분류·검색에 활용한다. 단순히 causal mask만 제거하는 것과 적응 학습을 구분한다.

## 논문에서 제안한 방식

Decoder-only 모델의 Attention을 양방향으로 변경하고 Masked Next Token Prediction(MNTP), 이어서 SimCSE 대조 학습으로 문장 표현을 적응시킨다. supervised contrastive 학습도 비교한다.

## 확인한 실험 근거

COLM 2024. Llama·Mistral 등에서 단어 수준 과제와 MTEB 문장 표현 과제를 평가. 양방향 연결만 추가한 경우와 MNTP로 적응한 경우를 비교한다. 근거: 방법 및 §3–5.

## 한계와 해석 범위

임베딩 성능 개선이 객관식 reasoning 개선을 의미하지 않는다. 일반 Attention decoder 결과를 hybrid DeltaNet에 그대로 적용할 수 없다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** Qwen3.5의 일부 full Attention을 양방향으로 바꾸는 실험은 가능하지만 순환·convolution 계층이 여전히 causal하다. 전체 Encoder로 전환했다고 부르면 안 된다. 추가 적응 학습과 별도 비교가 필요하다.

## 연결 문서

- [dqwen35 ](dqwen35.md)
- [metaencoder ](metaencoder.md)

