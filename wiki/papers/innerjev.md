# InnerJev / JEVal

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **General Decision Models: Benchmarking and Insights Beyond Jev**
- 연도: 2026 · 주제: reasoning 분포 증류 / 기존 LM Head
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2610.03935) · [arXiv](https://arxiv.org/abs/2610.03935)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

정답 one-hot뿐 아니라 reasoning 후의 전체 후보 분포를 직접 판단 출력에 전달한다. 별도 강한 교사 없이 자기 reasoning으로 학습하는 절차이며, teacher target 생성에는 reasoning 비용이 든다.

## 논문에서 제안한 방식

First-token readout은 유효 단일 토큰 코드의 LM Head 점수만 정규화한다. 같은 원본 모델이 reasoning chain을 두 번 생성하고 각 reasoning 끝의 후보 분포를 평균하여 교사 q를 만든다. 학생은 reasoning 없이 q를 CE로 맞추며 frozen 원본 분포에 대한 KL anchor를 추가한다. score는 기대 등급 오차 항도 사용한다.

## 확인한 실험 근거

Qwen3.5-4B와 Qwen3.8-27B에서 InnerJev를 학습. JEVal은 영어·중국어 11,257개 사례, 36 datasets, 10 domains. 단일 H100의 짧은 입력 latency를 보고한다. 근거: §3, §6.1–6.3.

## 한계와 해석 범위

교사 분포에 gold를 사용하거나 정답 여부로 필터링하지 않아 오류도 전달될 수 있다. JEVal 점수와 Jev Decision Index Full 점수는 다른 평가다. 출력 생성 없음은 온라인 inference 조건이며 데이터 준비는 reasoning을 생성한다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** 기존 readout을 그대로 둔 증류 대조군으로 유용하다. embedding·LM Head는 고정하고 언어 계층 projection을 full-rank 또는 LoRA로 학습한 논문 방식과, 우리의 v1 full language/tied embedding 학습을 구분한다. 0.8B의 자기 reasoning 품질을 먼저 평가한다.

## 연결 문서

- [dual-head-reasoning-distillation ](dual-head-reasoning-distillation.md)
- [reasoning-and-calibration ](../concepts/reasoning-and-calibration.md)

