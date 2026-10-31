# Dual-Head Reasoning Distillation

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning**
- 연도: 2025 · 주제: 학습 때만 reasoning / 분류층
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/2509.21487) · [arXiv](https://arxiv.org/abs/2509.21487)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

교사 설명을 학습하면서 추론에서는 원래 입력과 분류층만 사용한다. causal 입력 표현이 미래의 설명을 읽지 못하므로 설명·정답의 입력 누출을 방지한다.

## 논문에서 제안한 방식

공통 causal decoder에 분류용 2층 MLP와 학습 전용 LM Head를 사용한다. 학습 입력은 [원래 입력, REASON, 교사 설명, ANS, 정답]이다. 분류는 REASON 직전 마지막 실제 입력 hidden state만 읽고, LM loss는 전체 유효 시퀀스에 적용한다. L=β L_cls+α L_reason.

## 확인한 실험 근거

7개 SuperGLUE 과제. Qwen3-4B 평균 84.50→85.05, Llama3.2-3B 77.34→81.57. 논문은 동일 Backbone의 CoT decoding 대비 96–142배 QPS를 보고한다. 근거: §3, 표 1, §4.2.

## 한계와 해석 범위

속도 수치는 논문의 입력·생성 길이와 환경에 따른 비교이며 우리 모델 예상 속도가 아니다. 3–8B 고정 클래스 실험이고 0.8B·가변 후보 구조의 직접 검증은 아니다. 교사 설명 품질과 α에 민감하며 설정별 단일 실행 한계가 있다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** 후보별 공통 scorer를 사용하는 모델에도 보조 설명 loss를 결합하는 실험을 제안한다. 설명은 학습에서만 사용하며 분류 표현이 설명 이후 토큰을 참조하지 않도록 해야 한다. 2026-10-17부터 교사 설명·분포를 학습하지 않는 방침이라 보류 중이다.

## 연결 문서

- [reasoning-and-calibration ](../concepts/reasoning-and-calibration.md)
- [innerjev ](innerjev.md)

