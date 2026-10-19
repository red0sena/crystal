# Model Stock: 미세조정 모델 두 개의 각도로 섞는 비율 정하기

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md) · [v3 요약](../project/v3-summary.md)

- 논문: **Model Stock: All we need is just a few fine-tuned models** (Jang 외, NAVER AI Lab)
- 연도: 2024 (ECCV) · 주제: 가중치 병합 / 하이퍼파라미터 없는 보간 비율
- 원문: [arXiv](https://arxiv.org/abs/2403.19522)
- 조사 기록: 2026-10-18 arXiv HTML 본문에서 식을 확인했다.

## 논문에서 제안한 방식

같은 사전학습 가중치 w₀에서 미세조정한 두 모델 w₁, w₂의 평균 w₁₂를 원본 쪽으로 당긴다. 비율 t는 두 변화량 사이 각도 θ로 정한다.
- 두 모델: w_H = t·w₁₂ + (1−t)·w₀, t = 2cosθ / (1+cosθ).
- N개 모델: t = N·cosθ / (1+(N−1)cosθ).
- 층마다 따로 계산한다. 논문은 bias·분류층은 각도가 작아 t가 크고, attention·MLP 가중치는 원본에 더 기대게 된다고 보고한다.

## 확인한 실험 근거

CLIP ViT 계열만 실험했다. ViT-B/32에서 모델 2개로 ImageNet 79.89%, 분포 이동 평균 50.99%였다. 균등 Model Soup(48개)은 79.76%, 52.08%였다. 언어 모델 실험은 없다.

## 우리 Qwen에 적용한 결과 (우리 실험)

`merge.py --stock`으로 구현했다(텐서마다 코사인을 구하고 [0, 1]로 자른다).
- g1과 g3 epoch-1의 변화량은 코사인 0.10~0.33으로 거의 수직이었다. t는 임베딩 0.18, 앞층 0.27, 23층 0.50이었다.
- 결과는 JEVal new가 가장 높았다(g1+g3e2 0.404, g4b+g3e2 0.414로 d3-lite 0.409 수준). 대신 공개 샘플이 29~31로 크게 떨어졌다.
- 서로 다른 데이터·학습률로 학습한 모델이라 논문의 같은 조건(시드만 다름)과 다르다.

## 연결 문서

- [WiSE-FT](wise-ft.md) · [LiNeS](lines.md) · [체크포인트 평균](checkpoint-averaging.md)
