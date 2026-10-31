# WiSE-FT: 원본과 미세조정 가중치 섞기

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md) · [전략](../designs/beat-d3-lite.md)

- 논문: **Robust fine-tuning of zero-shot models** (Wortsman, Ilharco 외)
- 연도: 2021 arXiv, CVPR 2022(NeurIPS 2021 워크숍 판도 있음) · 주제: 가중치 보간 / 분포 밖 강건성
- 원문: [arXiv](https://arxiv.org/abs/2109.01903) · [코드](https://github.com/mlfoundations/wise-ft)
- 조사 기록: 2026-10-17 웹 검색 요약 기준. 논문 수치는 판본마다 다르다.

## 해결하려는 문제

미세조정은 목표 분포 정확도를 올리지만 분포가 바뀐 입력에서는 원래 모델의 강건성을 잃는다.

## 논문에서 제안한 방식

미세조정한 가중치와 원래(zero-shot) 가중치를 원소별로 가중 평균한다. 섞는 비율 α를 0~1에서 고른다. 추가 학습이 없고 몇 줄로 구현된다.

## 확인한 실험 근거

CLIP 계열에서 ImageNet과 다섯 개 분포 이동 변형에 대해 분포 밖 정확도가 4~6%p 오르고(arXiv 초록) 분포 안 정확도도 비슷하거나 약간 오른다고 보고한다. 원래 모델과 미세조정 모델 사이 가중치 경로에서 정확도가 높게 유지된다는 관찰이 근거다. 워크숍 판은 분포 밖 2~10%p로 적는다.

## 한계와 해석 범위

이미지·텍스트 대조 모델 실험이다. 언어 모델의 판정 readout에서 같은 폭을 기대할 근거는 아니다. 최적 α는 평가 기준마다 다를 수 있다.

## 우리 Qwen에 적용한 결과 (우리 실험)

`qwen_decision/v3/interpolate.py`로 g1과 원본을 섞었다(2026-10-17). JEVal v2 dev 매크로 skill이 g1 0.348에서 α 0.3일 때 0.432로 올랐고, 새 분야는 0.288→0.399(d3-lite 0.409)였다. 공개 샘플 지수는 α 0.5가 35.30으로 가장 높았고 α 0.3은 29.97이었다. 대리셋과 공개 점수의 최적 α가 다르다. 상세는 [작업 기록](../project/v3-log.md). 10-19에는 g4b 위 LoRA 업데이트를 원본과 섞어 봤지만 개선이 없었다([조합 실험](../project/robust-combinations.md)).

## 연결 문서

- [L2-SP](l2-sp.md) · [LoRA Learns Less and Forgets Less](lora-learns-less.md) · [LP-FT](lp-ft.md)
