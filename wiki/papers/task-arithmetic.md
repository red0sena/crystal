# Task Arithmetic: 미세조정 변화량을 더하고 빼기

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md) · [전략](../designs/beat-d3-lite.md)

- 논문: **Editing Models with Task Arithmetic** (Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi)
- 연도: 2023 (ICLR) · 주제: 가중치 병합 / 모델 편집
- 원문: [arXiv](https://arxiv.org/abs/2212.04089)
- 확인 수준: 초록·2차 자료(본문 완독 아님). 정의와 연산은 여러 자료가 일치한다.

## 논문에서 제안한 방식

원본 가중치를 θ₀, 어떤 과제로 미세조정한 가중치를 θᵢ라 할 때 **task vector** Δᵢ = θᵢ − θ₀를 그 과제에서 배운 것으로 본다.
- **더하기:** θ = θ₀ + λ Σᵢ Δᵢ. 여러 과제의 능력을 한 모델에 합친다.
- **빼기:** θ = θ₀ − λΔ. 예를 들어 독성 데이터로 학습한 벡터를 빼서 독성 출력을 줄인다.
- **λ:** 벡터를 반영하는 배율이다. 여러 벡터를 그대로(λ = 1) 더하면 원본에서 너무 멀어지기 쉬워 dev로 고른다.

TIES, DARE, AdaMerging 등 뒤의 병합 방법은 모두 task vector를 단위로 쓰고, 벡터를 거르거나(TIES·DARE) 층마다 λ를 다르게 둔다(AdaMerging).

## 우리 Qwen에 적용한 결과 (2026-10-21)

`merge.py --tuned <a> <b> --weights wₐ w_b --alpha α`는 θ₀ + α(wₐΔₐ + w_bΔ_b)를 만든다. 벡터별 λ가 α·wᵢ인 Task Arithmetic이고, WiSE-FT(원본 쪽으로 당기기)와 같은 식이다.
- g6 + g7 에폭 2, 벡터별 λ = 0.4(합 0.8, 원본 20%)가 22개 병합 후보 중 1위였다: JEVal same / new / 전체 0.575 / 0.424 / 0.489, 공개 44.95, fresh 0.541. d3-lite(0.578 / 0.409 / 0.482, 39.65, 0.509) 대비 Full 추정 +1.49.
- λ 합에 따른 변화(g6 : g7e2 = 1 : 1): 1.0(단순 평균) +0.13, 0.9 +1.02, 0.85 +1.11, **0.8 +1.49**, 0.75 +1.46, 0.7 +0.79. 합을 줄일수록 JEVal new가 오르고 same은 0.9 아래에서 내린다.
- 두 벡터의 코사인 유사도 중앙값은 0.29였다(Model Stock 레시피에서 역산). v3-gold7이 v3-gold6의 출처를 모두 포함하고 학습 설정도 같아, 서로 다른 과제를 합치는 논문 조건보다 같은 과제의 Model Soup에 가깝다. 그래서 벡터를 거르는 TIES·DARE보다 λ만 고르는 쪽이 나았다.
- 1위는 22개 후보를 같은 JEVal dev로 비교해 고른 것이라 낙관적이다. JEVal restricted 확인 전이다.

## 연결 문서

- [WiSE-FT](wise-ft.md) · [TIES-Merging](ties-merging.md) · [DARE](dare.md) · [LiNeS](lines.md) · [Model Stock](model-stock.md) · [체크포인트 평균·Soup](checkpoint-averaging.md)
