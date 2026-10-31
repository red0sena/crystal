# Loss 비교: 전체 LoRA에서 loss만 바꾸기

[위키 홈](../README.md) · [LoRA 위치 비교](lora-placement.md) · [인사이트](insights.md) · [loss 문헌 검토](../papers/loss-alternatives.md)

**상태: 완료(2026-10-31 00:18 KST).** 판정 규칙은 첫 fit 전에 `plan.json`으로 고정했다.
- 1단계(한 번 forward하는 ls, focal, brier)는 10-29 09:06~17:33에 돌았다.
- 2단계(두 번 forward하는 twoview, consist, mntp)는 10-29 17:33에 시작했다. fit당 학습 50~52분, 평가 포함 약 60분이다.
- **10-29 22:01에 기계가 비정상 종료돼 2단계가 멈췄다.** 정상 종료 기록 없이 10-30 12:37에 다시 부팅됐고, 학습 코드의 오류 기록은 없다.
  - 그때까지 끝난 fit은 twoview 131/151, consist 131, mntp 131이다. consist 151은 672 step에서 끊겼다.
  - 10-30 13:17에 같은 `plan.json`으로 이어서 시작했다. 완료된 fit은 건너뛰었고, 끊긴 폴더는 `consist-seed151.incomplete-*`로 보존한 뒤 consist 151을 처음부터 다시 돌렸다. 입력은 seed와 step의 hash로 정해지므로 중단 전과 같다.

**한 줄 결론: 사전 등록 검정 16개 중 CE를 이긴 것은 없다.**
- consist(두 순서 + 일관성 벌점)가 평균으로는 가장 좋았다. CE 대비 selection NLL −0.023(4/5 seed, 구간이 0을 넘음)이다.
- 다만 이 비교는 보고만 하기로 한 비교다. 이를 나눈 두 검정(두 순서를 본 효과, 일관성 벌점의 효과)은 각각 규칙에 조금씩 못 미쳤다.
- 두 번 forward하는 팔은 학습 계산이 2배다. 확인하려면 새 seed로 consist − all을 미리 등록해 다시 재야 한다.

**1단계 한 줄 결론: label smoothing과 Brier는 CE와 같고, focal은 짧은 문항을 해친다.**

## 1단계 결과 (seed 5개)

5 seed 평균이다. 정답 NLL이고, 괄호 안은 정답률(%)이다.

| 팔 | selection | 장문 | BABILong | JEVal 장문 | QASPER | DocNLI 장문 |
|---|---|---|---|---|---|---|
| all (CE, 대조군) | 0.750 (69.4) | 0.952 (63.1) | 1.381 (51.7) | 0.971 (74.4) | 0.588 (71.1) | 0.868 (55.2) |
| ls | 0.760 (69.6) | 0.938 (63.4) | 1.340 (51.9) | 0.999 (77.6) | 0.593 (68.9) | 0.820 (55.1) |
| focal | 0.799 (67.9) | 0.920 (62.4) | 1.349 (51.0) | 1.026 (74.6) | 0.601 (69.3) | 0.706 (54.6) |
| brier | 0.746 (69.2) | 0.944 (63.5) | 1.381 (52.3) | 0.917 (76.2) | 0.627 (69.5) | 0.851 (56.2) |
| frozen (참고) | 0.772 (68.0) | 1.056 (60.5) | 1.411 (48.8) | 1.215 (72.7) | 0.638 (66.7) | 0.960 (54.0) |

사전 등록 검정 10개다. 값은 후보 − all이다. NLL은 음수가 나은 쪽이고, 정답률은 점 단위다.

| 검정 | 평균 | 95% 구간 | 후보가 나은 seed | 판정 |
|---|---|---|---|---|
| ls selection NLL | +0.010 | [−0.008, +0.029] | 1/5 | 차이 없음 |
| ls 장문 NLL | −0.014 | [−0.066, +0.028] | 2/5 | 차이 없음 |
| ls selection 정답률 | +0.2 | [−0.9, +1.3] | 3/5 | 차이 없음 |
| ls 장문 정답률 | +0.3 | [−1.3, +1.6] | 3/5 | 차이 없음 |
| focal selection NLL | +0.050 | [+0.032, +0.067] | 0/5 | **나쁨** |
| focal 장문 NLL | −0.031 | [−0.084, +0.028] | 4/5 | 차이 없음 |
| focal selection 정답률 | −1.5 | [−2.8, −0.2] | 0/5 | **나쁨** |
| focal 장문 정답률 | −0.7 | [−2.0, +0.5] | 1/5 | 차이 없음 |
| brier selection NLL | −0.003 | [−0.016, +0.010] | 4/5 | 차이 없음 |
| brier 장문 NLL | −0.008 | [−0.048, +0.027] | 3/5 | 차이 없음 |

**보고만 하는 비교**
- 세 팔 모두 장문 NLL에서 frozen보다 낫다(5/5). 장문 정답률은 ls와 brier가 5/5로 낫고, focal은 4/5 seed지만 구간 하한이 −0.06이라 규칙에 못 미쳤다.
- selection NLL에서는 brier만 frozen보다 낫다(5/5). focal은 frozen보다도 나쁘다(0/5).
- 학습 데이터의 CE(마지막 64 step 평균)는 all 0.726, brier 0.738, ls 0.748, focal 0.785다. CE를 바로 줄이는 all이 가장 낮고, focal이 가장 덜 맞췄다.
- 셋별로 보면(사후 관찰) ls의 JEVal 장문 정답률이 +3.2점(4/5 seed)인 대신 QASPER는 −2.3점(1/5 seed)이다. 둘 다 seed마다 크게 흔들리는 셋이라 새 seed로 확인하기 전에는 신호로 보지 않는다.

**해석**
- **CE가 이미 맞는 목표다.** 주 지표는 정답 확률(NLL)이고, 지수의 대부분은 argmax 정답률이다. CE는 NLL을 바로 줄이므로, 다른 loss가 이기려면 CE가 못 잡는 무언가를 따로 줘야 한다.
- **focal은 이미 맞히는 문항을 덜 학습해서 짧은 문항 정답률까지 잃은 것으로 보인다.** 이것은 우리 실험에서 직접 나온 결과다. [Beyond Log Likelihood](https://arxiv.org/abs/2510.00526)는 반대 방향(−p, 확신 있는 문항에 비중을 더 줌)만 시험했다. 그러니 "그 논문의 예측과 같다"고 했던 10-29의 설명은 맞지 않아 고쳤다.
  - 그 논문의 기준으로 보면, 학습 전 Qwen의 평균 정답 확률(0.41~0.43)은 "loss 선택이 거의 차이를 내지 않는" 중간 구간에 가깝다. ls, brier ≈ CE와 맞는다.
  - 문헌 검토는 [CE가 정말 최선인가](../papers/loss-alternatives.md)에 있다.
- focal의 장문 NLL이 4/5 seed에서 낮은 것은 덜 확신하게 만든 결과로 보인다. DocNLI 장문은 NLL이 0.868 → 0.706(5/5 seed)으로 크게 내렸지만 정답률은 55.2 → 54.6으로 그대로다. 정답률이 같고 NLL만 낮아지는 것은 보정 효과다.
- **brier는 full FT에 이어 LoRA에서도 효과가 없었다.** CE에 Brier를 더해도 CE의 최적점이 바뀌지 않는다.
- **ls는 정답률과 NLL 모두 CE와 같다.** 0.1만큼 분포를 펴는 것은 이 규모에서 과확신 문제를 바꾸지 못했다.
- 1단계 fit 시간은 학습 25분, 평가 포함 약 34분으로 추정(33분)과 맞았다. 최대 메모리는 6.1GB다.

## 2단계 결과 (seed 5개)

5 seed 평균이다. 정답 NLL이고, 괄호 안은 정답률(%)이다.

| 팔 | selection | 장문 | BABILong | JEVal 장문 | QASPER | DocNLI 장문 |
|---|---|---|---|---|---|---|
| all (CE, 대조군) | 0.750 (69.4) | 0.952 (63.1) | 1.381 (51.7) | 0.971 (74.4) | 0.588 (71.1) | 0.868 (55.2) |
| twoview | 0.736 (69.4) | 0.959 (63.2) | 1.424 (52.3) | 0.902 (75.8) | 0.634 (67.8) | 0.877 (56.8) |
| **consist** | **0.727 (69.9)** | **0.937 (63.8)** | 1.403 (52.9) | 0.871 (76.7) | 0.605 (70.1) | 0.868 (55.4) |
| mntp | 0.737 (69.6) | 0.956 (62.3) | 1.385 (51.0) | 0.947 (74.7) | 0.620 (67.8) | 0.870 (55.8) |
| frozen (참고) | 0.772 (68.0) | 1.056 (60.5) | 1.411 (48.8) | 1.215 (72.7) | 0.638 (66.7) | 0.960 (54.0) |

사전 등록 검정 6개다. NLL은 음수가 나은 쪽이다.

| 검정 | 평균 | 95% 구간 | 후보가 나은 seed | 판정 |
|---|---|---|---|---|
| twoview − all, selection NLL | −0.013 | [−0.027, +0.000] | 5/5 | 차이 없음(구간 상단이 +0.0005) |
| twoview − all, 장문 NLL | +0.007 | [−0.044, +0.048] | 1/5 | 차이 없음 |
| consist − twoview, selection NLL | −0.009 | [−0.028, +0.006] | 4/5 | 차이 없음 |
| consist − twoview, 장문 NLL | −0.023 | [−0.052, +0.007] | 4/5 | 차이 없음 |
| mntp − all, selection NLL | −0.012 | [−0.029, +0.002] | 5/5 | 차이 없음 |
| mntp − all, 장문 NLL | +0.004 | [−0.037, +0.045] | 2/5 | 차이 없음 |

**보고만 하는 비교**

| 비교 | selection NLL | selection 정답률 | 장문 NLL | 장문 정답률 |
|---|---|---|---|---|
| consist − all | **−0.023 [−0.040, −0.005], 4/5** | +0.5, 3/5 | −0.015, 3/5 | +0.7, 3/5 |
| twoview − frozen | −0.036, 5/5 | +1.3, 5/5 | −0.096, 5/5 | +2.6, 5/5 |
| consist − frozen | −0.045, 5/5 | +1.9, 5/5 | −0.119, 5/5 | +3.2, 5/5 |
| mntp − frozen | −0.035, 5/5 | +1.6, 4/5 | −0.100, 5/5 | +1.8, 4/5 |

- consist − all의 selection NLL만 규칙(4/5 seed + 구간이 0을 넘음)을 만족한다. 하지만 보고만 하기로 한 비교 7개 중 하나라서 판정으로 쓰지 않는다. 정답률과 장문은 3/5 seed로 차이가 없다.
- 2단계 세 팔은 모두 장문과 selection에서 frozen보다 낫다. 1단계와 같이, 전체 LoRA가 frozen보다 낫다는 결론은 그대로다.
- 학습 loss 성분(5 seed 평균): CE는 twoview 0.712, consist 0.712, mntp 0.718이다. 1단계 all의 마지막 64 step 평균은 0.726이었다. 두 순서의 Jensen-Shannon은 twoview 0.010, consist 0.008이다. MNTP loss는 1.88이다.

**해석**
- **CE를 넘은 loss는 없다. 다만 두 순서를 보는 쪽은 방향이 일관되게 좋다.** selection NLL에서 twoview는 5/5, consist는 all 대비 4/5, twoview 대비 4/5 seed가 낮다. 효과가 작아(0.01~0.02) seed 5개로는 하나씩 갈라지지 않았다.
- **일관성 벌점이 할 일이 거의 없었다.** 벌점이 없는 twoview도 두 순서의 Jensen-Shannon이 0.010밖에 안 된다. CE 학습이 이미 step마다 선택지 순서를 섞기 때문이다. 벌점은 이것을 0.008로 내렸다. 전체 loss에서 차지하는 몫은 약 1%다. λ = 1을 더 키우면 달라질 수는 있지만, 10-25 [순서 측정](order-probe.md)에서 잰 4순서 평균 이득(짧은 문항 +0.7점, g4b에서 출발한 모델)도 작다.
- **계산량이 2배다.** twoview와 consist는 한 update에 같은 행을 순서만 바꿔 두 번 본다. 학습 시간이 fit당 25분에서 50분으로 늘었다. 이득이 "두 순서"에서 왔는지 "더 많은 계산"에서 왔는지는 이 실험으로 가를 수 없다.
- **MNTP는 가린 토큰은 배웠지만 답에는 옮겨지지 않았다.** MNTP loss는 학습 초 4.70(seed 131)에서 끝 1.88(5 seed 평균)까지 내려 원본 구조의 다음 토큰 loss 3.40보다 낮아졌다. 그런데 selection NLL은 −0.012(구간이 0을 포함)이고, 장문 정답률은 −0.8점(1/5 seed)이다.
- 셋별로 보면(사후 관찰) consist의 JEVal 장문 정답률이 all 대비 +2.3점(4/5 seed)이다. 1단계 ls의 같은 관찰처럼 seed마다 크게 흔들리는 셋이라 신호로 보지 않는다.

## 왜 하나

- [LoRA 위치 비교](lora-placement.md)에서 전체 LoRA r16(양방향)이 지금까지 최선이었다. 짧은 문항 69.4%, 장문 63.1%다.
- 지금까지 바꿔 본 loss는 둘이고(CE + Brier, 두 번 읽기 학습), 둘 다 full FT에서 효과가 없었다.
- 질문: **모델과 데이터는 그대로 두고 loss만 바꾸면 달라지나?**

## 설정

모든 팔은 전체 LoRA r16과 같은 모델이다. 데이터(1.6만 행), seed 5개, 입력이 같고, 입력은 step마다 hash로 대조한다. 바뀌는 것은 loss뿐이다.

| 팔 | loss | 근거 | 계산량 |
|---|---|---|---|
| all (대조군, 완료) | CE | 위치 비교의 all fit을 다시 쓴다 | 1배 |
| **ls** | 정답을 보인 선택지에 고르게 0.1만큼 나눈 CE (label smoothing) | Szegedy et al. 2016 | 1배 |
| **focal** | focal loss, γ = 2. 이미 맞히는 문항의 비중을 낮춘다 | Lin et al. 2017, 보정은 Mukhoti et al. 2020 | 1배 |
| **brier** | CE + 1 × Brier | 거르기의 brier를 LoRA에 다시 적용 | 1배 |
| **twoview** | 같은 문항을 선택지 순서를 다르게 두 번 넣고 CE 평균 | consist의 대조군 | 2배 |
| **consist** | twoview + 1 × 두 출력 분포 사이 Jensen-Shannon | 순서 일관성. g4b에서 λ = 0.5로 해 봤을 때는 확인된 변화가 없었다([이전 실험](robust-combinations.md)) | 2배 |
| **mntp** | CE + 0.5 × 가린 토큰 맞히기. 자연어 토큰 20%를 `_`로 바꾸고, 바로 앞 위치에서 원래 토큰을 맞힌다 | LLM2Vec이 decoder를 양방향 attention에 적응시킨 방법 | 2배 |

- consist와 twoview의 두 번째 입력은 seed, step, 행 위치의 hash로 정한다. 순서를 바꿀 수 없는 문항(순서가 있는 선택지, 위치를 가리키는 질문, noul)은 같은 입력이 두 번 들어간다. 이 경우 두 출력은 dropout만큼만 다르다.
- mntp의 가중치 0.5, consist의 λ = 1은 조정하지 않은 값 하나씩이다.
- 학습 정답은 gold 라벨만 쓴다. teacher는 없다.

## 판정 규칙 (`plan.json`에 고정)

- 주 지표는 정답 NLL(selection, 장문)이고, 규칙은 지금까지와 같다(5 seed 중 4개 이상 + 계층 bootstrap 95% 구간이 0을 넘음).
- 검정 16개:
  - ls, focal, brier, twoview, mntp 각각 − all: selection NLL, 장문 NLL
  - consist − twoview: 일관성 항만의 효과
  - ls, focal − all: selection 정답률, 장문 정답률도 검정한다. 이 둘은 loss가 확신 정도를 바꾸기 때문에, NLL에 순서 맞히기와 보정이 섞인다.
- 보고만 하는 비교: consist − all, 각 팔 − frozen, 셋별 점수, 학습 loss 성분.

## 구현 검사 (`check.json`, `smoke.json`)

- 위치 비교의 구현 hash가 그대로이고, 대조군 fit이 모두 완료돼 있다.
- loss 값을 알려진 입력으로 확인했다.
  - label smoothing: `[1, 0, 0]`(두 선택지만 보임) → `[0.95, 0.05, 0]`
  - focal: 손으로 계산한 값과 같다. γ = 0이면 CE와 같다.
- 두 번째 입력을 원래 선택지 순서로 되돌리면 정답 위치가 같다. 학습 전 모델의 두 입력 사이 Jensen-Shannon은 0.10~0.66으로, 순서에 따라 답이 흔들린다.
- 학습 데이터 16,384행 모두에 가릴 자연어 토큰이 있다.
- MNTP 구현 확인(학습 전 모델, 32행). 단위는 토큰당 nats다.

  | 구조 | 다음 토큰 그대로 | 20% 가림 |
  |---|---|---|
  | 원본 구조 | 3.40 | 4.93 |
  | 양방향 | 0.23 | 7.53 |

  - 양방향에서 가리지 않으면 정답 토큰이 미리 보여 loss가 0.23까지 떨어진다. 그래서 가려야 한다.
  - 가리면 7.53으로 원본보다 높다. 학습 전 양방향 구조가 아직 적응하지 않았다는 뜻이고, MNTP가 겨냥하는 부분이다.
- 여섯 팔 모두 한 step 뒤 base 가중치는 그대로이고 LoRA B만 움직였다. LR은 2e-4다.
- 처음 12 step의 입력 hash가 거르기 smoke와 같았다.
- smoke 추정 학습 시간: 한 번 forward하는 팔 14분, 두 번 forward하는 팔 29분. 최대 메모리는 7.6GB 이하다.
  - 위치 비교의 실제/추정 비율(1.73배)을 적용하면 fit당 약 33분, 57분이다(평가 8분 포함).
  - 전체는 약 23시간이다.

## 재현

```bash
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.loss_experiment check
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.loss_experiment smoke
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.loss_experiment freeze
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.loss_experiment run
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.loss_experiment report
```

코드는 [loss_experiment.py](../../qwen_decision/v3/loss_experiment.py)이고, 출력은 `runs/from-base/loss-v1/`이다.
