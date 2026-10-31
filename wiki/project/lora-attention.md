# Attention-only LoRA: 논문 설정을 학습 방식 거르기 데이터로

[위키 홈](../README.md) · [학습 방식 거르기](training-methods.md) · [근거 논문](../papers/decision-training-recipes.md)

**상태: 완료(2026-10-28 13:27 ~ 17:42 KST).** 판정 규칙은 첫 fit 전에 `plan.json`으로 고정했다. 판정은 `runs/from-base/lora-v1/report.json`이다.

**한 줄 결론:** attention-only LoRA는 짧은 문항에서 full FT와 같았고, 장문에서는 full FT보다 확실히 좋았다(+3.9~4.8점, 5/5 seed). DeltaNet 얼리기와 비교하면 짧은 문항은 졌고(−2.2점, 5/5 seed) 장문은 이겼다(+2.6점).

## 결과: 5 seed 평균

정답 NLL(낮을수록 좋음)이고, 괄호 안은 정답률(%)이다. causal, noncausal, frozen은 [학습 방식 거르기](training-methods.md)의 같은 seed fit이다.

| 레시피 | 학습 파라미터 | selection | 장문 평균 | BABILong | JEVal 장문 | QASPER | DocNLI 장문 | 장문 정답률 seed 표준편차 |
|---|---|---|---|---|---|---|---|---|
| causal (full FT) | 7.5억 | 0.840 (65.3) | 1.140 (57.7) | 1.411 (49.4) | 1.617 (63.0) | 0.624 (65.5) | 0.907 (52.9) | 3.1 |
| noncausal (full FT) | 7.5억 | 0.808 (66.6) | 1.154 (58.3) | 1.543 (48.6) | 1.445 (66.2) | 0.647 (65.9) | 0.979 (52.3) | 1.6 |
| frozen (full FT, DeltaNet 얼림) | 5.6억 | **0.772 (68.0)** | 1.056 (60.5) | 1.411 (48.8) | 1.215 (72.7) | 0.638 (66.7) | 0.960 (54.0) | 2.4 |
| lora-causal | 108만 | 0.850 (64.9) | 1.042 (61.6) | 1.346 (**53.3**) | 1.344 (71.1) | 0.591 (68.8) | 0.889 (53.2) | 0.6 |
| lora-noncausal | 108만 | 0.814 (65.8) | **0.985 (63.1)** | 1.412 (53.1) | **1.093 (73.1)** | **0.568 (71.2)** | **0.865 (54.9)** | 1.0 |

## 판정 (사전 등록 6개 검정, 정답 NLL)

| 검정 | selection NLL | 장문 NLL |
|---|---|---|
| lora-causal − causal | +0.010 [−0.017, +0.038], 1/5 → 차이 없음 | −0.097 [−0.166, −0.036], 5/5 → **나음** |
| lora-noncausal − noncausal | +0.006 [−0.020, +0.030], 2/5 → 차이 없음 | −0.169 [−0.231, −0.111], 5/5 → **나음** |
| lora-noncausal − frozen | +0.042 [+0.018, +0.066], 0/5 → **나쁨** | −0.071 [−0.152, −0.002], 4/5 → **나음** |

보조 지표(정답률)도 같은 방향이었다.

- lora-causal − causal: selection −0.4점(차이 없음), 장문 +3.9점 [+1.7, +6.6]
- lora-noncausal − noncausal: selection −0.8점(차이 없음), 장문 +4.8점 [+3.2, +6.6]
- lora-noncausal − frozen: selection −2.2점 [−3.8, −0.7], 장문 +2.6점 [+0.9, +4.3]
- 보고만 하는 비교(lora-noncausal − lora-causal): LoRA에서도 양방향이 낫다. selection NLL −0.037(5/5), 장문 정답률 +1.5점(5/5)

## 해석

- **짧은 문항은 108만 파라미터로도 full FT만큼 배운다.** 원본 구조와 양방향 모두 LoRA와 full FT의 차이가 0에 가깝다.
- **장문은 덜 바꿀수록 좋다(이 실험 안에서).** 장문 정답률은 바꾼 파라미터 양의 순서와 정확히 반대였다. 다만 10-29 [위치 비교](lora-placement.md)에서 10배를 학습한 전체 LoRA도 장문이 같았고, 같은 양을 attention 한 곳에 몬 rank 160만 떨어졌다. 총량보다 한 모듈의 큰 변화가 문제로 보인다.
  - full FT 57.7~58.3
  - DeltaNet만 얼림 60.5
  - attention LoRA 61.6~63.1
  - 장문은 학습 데이터에 적은 분포라서, 원래 Qwen이 가진 장문 능력을 덜 망가뜨리는 쪽이 이기는 것으로 보인다. 논문이 순차형 hybrid에서 보고한 망각과 같은 방향이다.
  - 학습 전 점수를 재 보니(10-28, [LoRA 위치 비교](lora-placement.md)) 장문은 49.1%로 낮았다. "보존"보다는 "적게 바꾸면 장문으로 더 잘 옮겨 간다"에 가깝다. 학습 전보다 떨어진 것은 DocNLI 장문(full FT)뿐이었다.
- **LoRA는 seed 흔들림도 작았다.** 장문 정답률 표준편차가 full FT의 1.6~3.2점에서 0.6~1.0점으로 줄었다. 다만 다른 위치의 LoRA는 1.4~1.8점이었고([LoRA 위치 비교](lora-placement.md)), 같은 전체 LoRA에서 loss만 바꾼 팔은 1.0~1.2점이었다([Loss 비교](loss-screen.md)). 위치의 성질인지는 판단을 보류한다.
- **DeltaNet 얼리기와는 서로 장단이 있었다.** 짧은 문항은 frozen이 2.2점 높고, 장문은 LoRA가 2.6점 높다. 10-29 [위치 비교](lora-placement.md)에서 MLP를 포함한 LoRA r16이 둘 다 frozen보다 나아, 이 장단은 attention-only가 짧은 문항을 덜 배운 탓으로 정리됐다.
- **학습 비용:** fit당 17~19분(full FT 21~23분), 최대 메모리 5.6GB(full FT 16.7GB)다.


## 왜 하나

- [학습 방식 거르기](training-methods.md)에서 DeltaNet token mixer를 얼린 full FT(frozen)만 양방향 full FT를 이겼다.
- 그 근거였던 [Where Should LoRA Go?](https://arxiv.org/abs/2604.22127)는 full FT가 아니라 LoRA였고, attention에만 붙인 LoRA가 가장 좋았다.
  - attention만: GSM8K 0.398
  - 전체 LoRA: 0.375
  - 학습 전: 0.297
- 질문: **우리 판정 과제에서도 attention만 LoRA로 학습하는 것이 full FT보다 나은가?**

## 설정

| | 논문 (softmax_only) | 이 실험 |
|---|---|---|
| 붙인 곳 | softmax attention 24개 모듈, 1.08M 파라미터 | 6개 full-attention 층의 q/k/v/o = 24개 모듈, 1,081,344 파라미터 (논문 숫자와 정확히 일치) |
| 얼린 곳 | 나머지 전부 | 나머지 전부. DeltaNet, MLP, norm, embedding, code readout(tied LM head) |
| rank / alpha / dropout | 16 / 32 / 0.05 | 같음 |
| LR | 2e-4 cosine | 2e-4, warmup 64, cosine 10%까지 (거르기와 같은 모양) |
| 데이터 | 과제당 2천 문항 × 3 epoch, 1,024토큰 자름 | 거르기와 같은 16,384행 한 바퀴, 토큰 자르지 않음 |
| optimizer | 8-bit Adam | FP32-master AdamW (거르기와 같음) |
| seed | 1개 | 5개 (131, 151, 173, 191, 211) |

**두 팔**

- **lora-causal:** 원본 구조. 논문 설정 그대로다.
- **lora-noncausal:** 양방향 구조. 거르기의 noncausal, frozen과 같은 구조다.

**대조군:** 거르기에서 이미 돌린 causal, noncausal, frozen fit(full FT, LR 1e-5)이다. 같은 seed를 쓰고, 매 step 입력 hash를 거르기의 causal fit과 대조한다. 거르기의 평가 코드 hash가 그때와 같은지도 확인했다.

## 판정 규칙 (`plan.json`에 고정)

- 주 지표는 정답 NLL이다(selection과 장문). 정답률과 Brier는 보조 지표다.
- 검정 6개:
  - lora-causal − causal: 논문 설정의 LoRA 대 full FT
  - lora-noncausal − noncausal: 양방향에서 LoRA 대 full FT
  - lora-noncausal − frozen: LoRA 대 지금 최선 방식
- 규칙은 거르기와 같다. 5 seed 중 4개 이상 + 계층 bootstrap 95% 구간이 0을 넘으면 "낫다"이다.
- full FT 대조군을 이긴 LoRA 팔은 6.55만 행 확인으로 넘어간다(10-29 이후 후보는 전체·MLP LoRA 대 frozen으로 바뀌었고, 아직 안 돌렸다).

## 구현 검사 (`check.json`, `smoke.json`)

- 학습되는 파라미터는 LoRA 1,081,344개뿐이다. 한 step 뒤 base 가중치는 그대로였고 B는 움직였다.
- 학습 전에는 LoRA 모델과 base 모델의 logit 차이가 0이다(B를 0으로 초기화했기 때문).
- 처음 12 step의 입력 hash가 거르기 smoke와 같았다.
- smoke 추정은 fit당 학습 10.3분이었다. 실제로는 lora-causal 17.3분, lora-noncausal 19.4분이었고, 최대 메모리는 5.6GB였다.

## 재현

```bash
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_experiment check
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_experiment smoke
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_experiment freeze
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_experiment run
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_experiment report
```

코드는 [lora_experiment.py](../../qwen_decision/v3/lora_experiment.py)이고, 출력은 `runs/from-base/lora-v1/`이다. fit마다 adapter만 저장한다(`adapter.safetensors`, BF16 값과 FP32 master).
