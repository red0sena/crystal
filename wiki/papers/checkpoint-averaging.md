# 체크포인트 평균(LAWA)과 Model Soup

[위키 홈](../README.md) · [작업 기록](../project/v3-log.md)

- 논문: **Early Weight Averaging meets High Learning Rates for LLM Pre-training** (Sanyal 외, 2023), 원형은 **Stop Wasting My Time! … Latest Weight Averaging** (Kaddour, 2022)
- 주제: 한 학습의 최근 체크포인트 평균
- 원문: [Sanyal 외 arXiv](https://arxiv.org/abs/2306.03241) · [Kaddour arXiv](https://arxiv.org/abs/2209.14981)
- 조사 기록: 2026-10-18 웹 검색 요약 기준.

## 논문에서 제안한 방식

일정 간격으로 저장한 최근 k개 체크포인트를 평균한다(k 기본 6, 16 넘게 평균하면 나빠졌다고 보고).

## 확인한 실험 근거

- nanoGPT 125M~770M과 Pythia 1B~12B 사전학습에서 EMA·SWA보다 검증 손실이 낮았다. 예: 770M에서 2.977 대 LAWA 2.908.
- 학습률이 높을수록, 체크포인트 간격이 넓을수록 이득이 컸다.

## 우리 Qwen에 적용한 결과 (우리 실험)

- g3(lr 3e-6) epoch-2와 epoch-3의 평균은 공개 샘플 41.14로 가장 높았다. JEVal 전체는 0.404로 거의 그대로였다.
- 서로 다른 학습의 평균(Model Soup 방식)인 g4b + g3 e2는 두 모델의 장점을 합쳤다(공개 39.80, fresh 0.483, JEVal same 0.548).
- g1을 넣은 세 모델 평균은 나빠졌다.
- 한 학습 안의 평균(10-23, 4,096 update): EMA −0.54%p, 마지막 구간 평균 −0.56%p로 둘 다 나빴다([레시피 변경](../project/recipe-ema-lr.md)).
- 지금 최선 완성 모델도 서로 다른 학습의 평균이다. g6 67% + g3 e2 33%가 새 1,847문항에서 g4b보다 +6.58%p 높았다(65.39 → 71.97%, [g6·g3 조합](../project/g6-soup-replication.md)).

## 연결 문서

- [Model Stock](model-stock.md) · [WiSE-FT](wise-ft.md) · [In-the-Wild 병합 연구](merging-in-the-wild.md)
