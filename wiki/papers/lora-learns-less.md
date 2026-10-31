# LoRA Learns Less and Forgets Less

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md)

- 논문: **LoRA Learns Less and Forgets Less** (Dan Biderman 외, Columbia University·Databricks Mosaic Research)
- 연도: 2024 (TMLR), ICLR 2025 포스터 · 주제: 미세조정 방식과 망각
- 원문: [arXiv](https://arxiv.org/abs/2405.09673)
- 조사 기록: 2026-10-17 초록·학회 페이지 기준.

## 해결하려는 문제

LoRA와 full fine-tuning이 새 분야 학습량과 원래 능력 망각 사이에서 어떻게 다른가.

## 논문에서 제안한 방식

프로그래밍·수학에서 지시 미세조정과 continued pretraining으로 LoRA와 full fine-tuning을 비교한다.

## 확인한 실험 근거

표준 저랭크 설정의 LoRA는 대부분 조건에서 full fine-tuning보다 덜 배우지만, 목표 밖 과제의 원래 능력은 더 잘 지킨다. weight decay·dropout보다 망각을 더 줄인다. full fine-tuning이 학습하는 변화의 랭크가 일반 LoRA 설정보다 10~100배 높다는 분석을 설명으로 든다. continued pretraining 토큰 수는 판본마다 10B/20B로 다르다.

## 한계와 해석 범위

생성형 과제 7B급 실험이다. 우리 판정 readout의 정답 CE 학습에 그대로 옮길 근거는 아니다.

## 우리 Qwen에 대한 의미

g1(full fine-tune)이 d3-lite보다 원본에서 1.5~2.4배 더 이동했고 원본과 섞으면 대리셋이 회복됐다. 다만 망각 보고에서 격차의 큰 몫은 "덜 새로 맞힘"이었다. 덜 잊는 대신 덜 배울 수 있는 LoRA는 새 과제 성능을 함께 비교해야 하는 후보로만 뒀다.

**2026-10-28~29 시험함:** 원래 Qwen에서 MLP를 포함한 전 층 LoRA r16이 양방향 full FT보다 짧은 문항과 장문 모두 높았다(69.4/63.1% 대 66.6/58.3%, 1.6만 행, 5 seed). "덜 배움"은 나타나지 않았다. 장문 상승은 보존이 아니라 전이였다(학습 전 49.1%). 지금까지 최선 레시피다([LoRA 위치 비교](../project/lora-placement.md), [Attention-only LoRA](../project/lora-attention.md)). 10-19에 g4b 위에서 한 r8 소규모 시험은 개선이 없었다([조합 실험](../project/robust-combinations.md)).

## 연결 문서

- [WiSE-FT](wise-ft.md) · [L2-SP](l2-sp.md) · [LP-FT](lp-ft.md)
