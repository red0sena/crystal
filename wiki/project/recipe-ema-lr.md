# 학습 레시피 변경: 가중치 평균(EMA·구간 평균)과 학습률

[위키 홈](../README.md) · [작업 기록](v3-log.md) · [학습 예산 확대](long-training.md)

2026-10-23. **결론: 두 레시피 모두 native를 넘지 못했고, 사전 규칙에 따라 확인셋 비교는 열지 않았다.**

## 설계

- 기준: [학습 예산 확대](long-training.md)의 native seed41 4,096 update(selection 4,147문항 출처 평균 79.14%).
- 같은 캐시 입력(`encoded-seed41.jsonl`)과 순서를 그대로 읽고 학습 방법만 바꿨다. step마다 입력 해시를 native 기록과 대조했다.
- `ema`: native의 2,048 step 체크포인트(optimizer 포함)에서 재개해 EMA(감쇠 .999)와 3,073~4,096 step 단순 평균을 함께 쌓았다. 원래 가중치는 native와 비트 단위로 같아야 하며 실제로 일치했다(결정론 확인).
- `lr1e5`: body LR 3e-6 → 1e-5, 같은 일정과 같은 두 평균.
- 사전 규칙(`runs/recipe/64k-4096-v1/preregistration.json`, 결과 전 작성): selection +0.5%p 이상인 후보 중 최선 하나만 확인셋 8,823문항에서 한 번 비교한다. 이 확인셋은 10-19 [블록 반복](recurrent-depth.md) 비교에 한 번 쓴 것이다(레시피 선택에는 쓰지 않았다). 기준을 넘은 후보가 없어 열지 않았다.

## 결과 (selection, native 대비)

| 후보 | 차이 | 95% 구간 | 회복 / 악화 |
|---|---:|---|---|
| ema · 원래 가중치 | 0.000%p | native와 동일 | 0 / 0 |
| ema · EMA | −0.54%p | [−1.39, +0.14] | 51 / 60 |
| ema · 구간 평균 | −0.56%p | [−1.41, +0.10] | 35 / 52 |
| lr1e5 · 원래 가중치 | −3.76%p | [−5.31, −2.32] | 203 / 334 |
| lr1e5 · EMA | −3.60%p | [−5.17, −2.23] | 204 / 323 |
| lr1e5 · 구간 평균 | −3.60%p | [−5.19, −2.22] | 210 / 344 |

## 해석

- 평균이 손해였던 이유: 마지막까지 점수가 빠르게 오르고 있었다(2,048 step 75.9% → 4,096 step 79.1%). 지난 구간을 평균하면 덜 학습된 가중치가 섞여 뒤처진다. cosine으로 이미 학습률을 10%까지 내리고 있어 EMA의 잡음 감소 이득도 작다([EMA of weights, TMLR 2024](https://arxiv.org/abs/2411.18704v1)).
- LR 1e-5는 학습 loss가 처음부터 끝까지 native보다 높았다(마지막 512 step 0.57 대 0.48). 학습 probe도 90.2% 대 93.9%였다. 과적합이 아니라 최적화가 나빠졌다. 3e-6이 이 조건에서 이미 좋은 값에 가깝다.
- 학습률이 높을수록 평균의 이득은 컸다(2,048 step에서 원래 70.1% 대 EMA 75.1%). [높은 LR에서 체크포인트 평균 이득이 크다는 보고](https://arxiv.org/abs/2306.03241)와 같은 방향이지만 최종 점수를 뒤집지는 못했다.
- 한 seed·한 학습률 값만 시험했다. 학습률 2~3개 값을 비교하지 않은 것은 설계의 한계다.

## 파일

- 코드: `qwen_decision/v3/recipe_experiment.py`, 테스트 `tests/test_v3_recipe.py`.
- 실행: `runs/recipe/64k-4096-v1/` — `preregistration.json`, `report.json`, `confirm.json`(후보 없음), 레시피별 `train.jsonl`·`selection-*.jsonl`·`model-{raw,ema,tail}/`.
