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

**2026-10-26~31 시험함.** full attention 6층만 양방향으로 바꾸고(`noncausal`) 원래 Qwen에서 full FT로 비교했다. 원본 대비 selection +0.51, 장문 −0.21로 정답률 검정은 통과하지 못했다([구조 비교](../project/from-base-structures.md)). 정답 NLL로는 짧은 문항에서 작지만 일관된 이득이 있어, 지금까지 최선 레시피에 들어 있다. 학습 없이 켜기만 해도 원본보다 높았다(1회 측정, [LoRA 위치 비교](../project/lora-placement.md)). MNTP 전환 학습은 10-21에 중단했고([MNTP 전환](../project/mntp-conversion.md)), MNTP 보조 loss는 답에 효과가 없었다([Loss 비교](../project/loss-screen.md)). DeltaNet 역방향 scan은 −0.51%p였다.

순환·convolution 계층은 여전히 causal하므로 전체 Encoder로 전환했다고 부르면 안 된다.

## 연결 문서

- [dqwen35 ](dqwen35.md)
- [metaencoder ](metaencoder.md)

