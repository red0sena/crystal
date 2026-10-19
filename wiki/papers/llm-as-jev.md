# LLM-as-Jev

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **LLM-as-Jev: LLMs Are Already Jev-Style Decision Models—When and How to Fine-Tune Them**
- 연도: 2026 · 주제: 기존 LM Head / 후보 문자열 점수
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2610.02076) · [arXiv](https://arxiv.org/abs/2610.02076)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

닫는 괄호로 후보 suffix를 prefix-free하게 구성하여 단일 코드 토큰 목록보다 후보 수를 확장한다. 학습에서는 tree-factorized listwise loss와 원본 모델 예측을 유지하는 KL anchors를 사용한다.

## 논문에서 제안한 방식

채팅 입력에 [1]…[K] 선택지를 제공하고 assistant prefix를 'Best answer: ['로 고정한다. 각 알려진 후보 suffix 'k]'의 토큰별 조건부 log probability를 합해 후보 사이에서 정규화한다. 공유 prefix와 후보 suffix branch를 사용한다.

## 확인한 실험 근거

Qwen3.5-4B와 Qwen3-0.6B를 평가. 학습 없는 numeric suffix readout, letter readout, full fine-tuning·LoRA를 비교하며 강한 원본 모델에서 광범위 학습의 이익은 제한적일 수 있다고 보고한다. 근거: §3–4.

## 한계와 해석 범위

후보 문자열이 여러 토큰이면 마지막 입력 logits 하나만 읽는 방식과 계산량이 다르다. 자유 문장 생성이 없어도 suffix 점수화 forward가 추가된다. public JevBench 결과를 Jev Decision Index의 공식 Full 점수와 혼동하지 않는다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** 현재 letter/code readout과 비교할 구조 보존 대조군이다. tokenizer별 suffix 길이, 숫자 표기의 prior, candidate count별 성능·보정·지연시간을 비교한다.

## 연결 문서

- [innerjev ](innerjev.md)
- [current-qwen ](../project/current-qwen.md)

