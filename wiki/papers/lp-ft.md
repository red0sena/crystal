# LP-FT: 미세조정이 사전학습 특징을 왜곡한다

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md)

- 논문: **Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution** (Kumar 외)
- 연도: 2022 (ICLR) · 주제: 미세조정 / 분포 밖 성능 / 출력층 초기화
- 원문: [arXiv](https://arxiv.org/abs/2202.10054)
- 조사 기록: 2026-10-17 웹 검색 요약 기준.

## 해결하려는 문제

full fine-tuning은 분포 안에서는 linear probing보다 낫지만 분포 밖에서는 못할 수 있다.

## 논문에서 제안한 방식

먼저 출력층만 학습(linear probe)한 뒤 전체를 미세조정(LP-FT)한다. 잘 맞춘 출력층에서 시작하면 하위 특징을 덜 왜곡한다.

## 확인한 실험 근거

분포 이동 벤치마크 10개 평균으로 fine-tuning은 linear probing보다 분포 안 +2%, 분포 밖 −7%였다. LP-FT는 fine-tuning 대비 분포 안 +1%, 분포 밖 +10%였다. 조기 종료로는 이 왜곡이 고쳐지지 않는다고 보고한다. 2층 선형망 이론으로도 무작위·고정 출력층에서 시작한 미세조정의 분포 밖 오차가 크다는 것을 보인다.

## 한계와 해석 범위

비전 분류 실험이 중심이다. 우리 readout은 무작위가 아니라 원본 LM head 코드 행에서 시작하므로 조건이 다르다.

## 우리 Qwen에 대한 의미

g1의 코드 행은 원본에서 4.5% 이동했고 d3-lite readout은 1.1%였다. 출력층을 먼저 맞추는 단계는 미구현 후보다.

## 연결 문서

- [WiSE-FT](wise-ft.md) · [LoRA Learns Less and Forgets Less](lora-learns-less.md)
