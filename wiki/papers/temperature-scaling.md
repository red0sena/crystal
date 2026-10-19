# Temperature scaling

[위키 홈](../README.md) · [구조 비교](../concepts/architecture-map.md)

- 논문: **On Calibration of Modern Neural Networks**
- 연도: 2017 · 주제: 확률 보정
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/1706.04599) · [arXiv](https://arxiv.org/abs/1706.04599)
- 조사 기록: 2026-10-13. 아래는 공개 본문을 바탕으로 작성한 요약이다. 우리 모델에서 재현한 결과가 아니다.

## 해결하려는 문제

최고 확률의 숫자를 실제 정답률과 더 잘 맞추기 위한 도구다. T는 모델마다·평가 조건마다 다시 검증해야 한다.

## 논문에서 제안한 방식

별도 validation에서 softmax(z/T)의 양수 scalar T를 NLL 기준으로 적합한다. 모델의 decision scores를 그대로 두고 확신의 크기를 사후 조정한다.

## 확인한 실험 근거

ICML 2017. 현대 neural classifier의 calibration 및 temperature scaling을 비교한다. 기존 대화에서 확인한 확률 보정의 기본 참고다.

## 한계와 해석 범위

모든 후보에 같은 양수 T를 적용하면 argmax와 후보 순위가 바뀌지 않는다. 오답 수정·순서 편향 제거·새 지식 학습 방법이 아니다.

## 우리 Qwen에 적용한다면

**아래는 적용 제안이며 미구현·미검증이다.** 모델 선택용 dev와 calibration용 dev를 분리하고 NLL·Brier·ECE를 test에서 보고한다. 후보 수·질문 유형별 T는 우리 v2 설계이며 논문의 기본 scalar 설정과 구분한다.

## 연결 문서

- [reasoning-and-calibration ](../concepts/reasoning-and-calibration.md)
- [current-qwen ](../project/current-qwen.md)

