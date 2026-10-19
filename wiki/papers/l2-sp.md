# L2-SP: 사전학습 가중치 쪽으로의 L2 제약

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md)

- 논문: **Explicit Inductive Bias for Transfer Learning with Convolutional Networks** (Xuhong Li, Yves Grandvalet, Franck Davoine)
- 연도: 2018 (ICML) · 주제: 전이학습 정규화 / 망각 완화
- 원문: [alphaXiv](https://www.alphaxiv.org/abs/1802.01483) · [arXiv](https://arxiv.org/abs/1802.01483)
- 조사 기록: 2026-10-17 arXiv 초록 확인. 본문 실험 수치는 이 카드에 옮기지 않았다.

## 해결하려는 문제

미세조정은 초기화와 조기 종료만으로 사전학습 해에 가까이 머무르게 하는데, 이 귀납 편향을 명시적으로 걸 수 있는가.

## 논문에서 제안한 방식

0 쪽으로 줄이는 weight decay 대신, 사전학습 가중치를 기준점으로 하는 L2 벌점 λ‖θ−θ₀‖²를 쓴다. 이 단순한 벌점을 다른 전이학습 정규화의 기준선으로 권한다.

## 한계와 해석 범위

합성곱 신경망 이미지 전이 실험이다. Adam 계열 옵티마이저에서는 손실에 더한 L2 벌점이 좌표별로 다시 스케일되므로, 손실 벌점과 분리형(decoupled) 구현은 다르게 동작한다.

## 우리 Qwen에 적용한다면 (구현됨, 미평가)

`train.py --anchor`(2026-10-17): Adam 업데이트 뒤 각 FP32 master 가중치를 원본 쪽으로 `lr × anchor`만큼 보간하는 분리형 구현이다. 켜면 0 방향 weight decay는 끈다. 원본 사본에 약 3GB가 든다. 강도는 짧은 비교로 골라야 하며 아직 학습에 쓰지 않았다(g3는 anchor 0).

## 연결 문서

- [WiSE-FT](wise-ft.md) · [LoRA Learns Less and Forgets Less](lora-learns-less.md)
