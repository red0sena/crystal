# LoRA 위치 비교: 어디에 붙이나, 얼마나 붙이나

[위키 홈](../README.md) · [Attention-only LoRA](lora-attention.md) · [학습 방식 거르기](training-methods.md) · [인사이트](insights.md)

**상태: 완료(2026-10-28 18:56 ~ 10-29 07:45 KST).** 판정 규칙은 첫 fit 전에 `plan.json`으로 고정했다. 판정은 `runs/from-base/lora-placement-v1/report.json`이다.

**한 줄 결론:** attention에만 붙인 LoRA가 r16 위치 중 짧은 문항에서 가장 약했다. DeltaNet이나 MLP에 붙이면 짧은 문항이 +1.9~3.6점 올랐고(5/5 seed), 장문은 attention LoRA와 같았다. MLP를 포함한 세 팔(mlp, attn-mlp, all)은 지금까지 최선이던 DeltaNet 얼리기를 짧은 문항과 장문 **둘 다** 이겼다(NLL 5/5 seed, 보고만 한 비교). 파라미터 수가 같아도 attention 한 곳의 rank를 160으로 키우면 장문이 full FT 수준으로 떨어졌다.

## 결과: 5 seed 평균

정답 NLL(낮을수록 좋음)이고, 괄호 안은 정답률(%)이다. attn은 [Attention-only LoRA](lora-attention.md)의 lora-noncausal, noncausal과 frozen은 [학습 방식 거르기](training-methods.md)의 같은 seed fit이다. 모두 양방향 구조다.

| 레시피 | 학습 파라미터 | selection | 장문 평균 | BABILong | JEVal 장문 | QASPER | DocNLI 장문 | 장문 정답률 seed 표준편차 |
|---|---|---|---|---|---|---|---|---|
| 학습 전 | 0 | 1.404 (47.9) | 1.770 (49.1) | 1.536 (45.8) | 4.152 (34.3) | 0.664 (62.4) | 0.729 (54.0) | – |
| full FT (noncausal) | 7.5억 | 0.808 (66.6) | 1.154 (58.3) | 1.543 (48.6) | 1.445 (66.2) | 0.647 (65.9) | 0.979 (52.3) | 1.6 |
| DeltaNet 얼리기 (frozen) | 5.6억 | 0.772 (68.0) | 1.056 (60.5) | 1.411 (48.8) | 1.215 (72.7) | 0.638 (66.7) | 0.960 (54.0) | 2.4 |
| attn LoRA | 108만 | 0.814 (65.8) | 0.985 (63.1) | 1.412 (53.1) | 1.093 (73.1) | **0.568** (71.2) | 0.865 (54.9) | 1.0 |
| gdn LoRA | 443만 | 0.768 (67.7) | 0.967 (**63.6**) | 1.414 (**53.7**) | **0.897** (76.1) | 0.593 (69.9) | 0.965 (54.6) | 1.4 |
| mlp LoRA | 531만 | 0.745 (68.6) | 0.964 (**63.6**) | 1.438 (51.4) | 0.899 (**76.7**) | 0.595 (70.9) | 0.923 (**55.3**) | 1.8 |
| attn-mlp LoRA | 639만 | **0.743** (69.1) | 0.957 (63.0) | 1.413 (51.8) | 0.902 (76.4) | 0.626 (68.7) | 0.887 (55.0) | 1.8 |
| all LoRA | 1,082만 | 0.750 (**69.4**) | **0.952** (63.1) | **1.381** (51.7) | 0.971 (74.4) | 0.588 (71.1) | 0.868 (55.2) | 1.8 |
| attn-r160 LoRA | 1,081만 | 0.813 (65.7) | 1.088 (58.4) | 1.483 (47.3) | 1.420 (66.8) | 0.649 (65.2) | **0.801** (54.3) | 1.2 |

- fit당 학습 시간은 19.6~24.9분이고, 최대 메모리는 6.1GB 이하였다.
- 학습 전 행은 seed 없이 한 번 잰 값이다.

## 판정 (사전 등록 10개 검정, 정답 NLL)

| 검정 | selection NLL | 장문 NLL |
|---|---|---|
| gdn − attn | −0.046 [−0.060, −0.030], 5/5 → **나음** | −0.017 [−0.057, +0.031], 4/5 → 차이 없음 |
| mlp − attn | −0.069 [−0.086, −0.051], 5/5 → **나음** | −0.021 [−0.057, +0.015], 4/5 → 차이 없음 |
| attn-mlp − attn | −0.071 [−0.087, −0.054], 5/5 → **나음** | −0.028 [−0.066, +0.006], 4/5 → 차이 없음 |
| all − attn | −0.064 [−0.085, −0.043], 5/5 → **나음** | −0.033 [−0.072, +0.006], 3/5 → 차이 없음 |
| all − attn-r160 | −0.063 [−0.084, −0.043], 5/5 → **나음** | −0.137 [−0.182, −0.093], 5/5 → **나음** |

보조 지표(정답률)도 같은 방향이었다.

- 각 위치 − attn: selection +1.9 / +2.8 / +3.3 / +3.6점(모두 5/5), 장문 −0.1~+0.5점(차이 없음)
- all − attn-r160: selection +3.7점 [+2.5, +4.8], 장문 +4.7점 [+3.3, +6.0], 모두 5/5

## 보고만 한 비교 (정답 NLL, 괄호는 정답률)

| 비교 | selection | 장문 |
|---|---|---|
| gdn − frozen | −0.004, 3/5 (−0.3점) → 차이 없음 | −0.088, 4/5 (+3.0점) → 나음 |
| mlp − frozen | −0.027, 5/5 (+0.6점) → 나음 | −0.092, 5/5 (+3.0점) → 나음 |
| attn-mlp − frozen | −0.029, 5/5 (+1.1점) → 나음 | −0.099, 5/5 (+2.4점) → 나음 |
| all − frozen | −0.023, 5/5 (+1.4점, 정답률도 통과) → 나음 | −0.104, 5/5 (+2.5점) → 나음 |
| attn-r160 − attn | −0.001, 3/5 (−0.1점) → 차이 없음 | +0.104, 0/5 (−4.7점) → **나쁨** |
| attn-r160 − frozen | +0.041, 0/5 (−2.3점) → 나쁨 | +0.033, 1/5 (−2.1점) → 차이 없음 |

- 네 위치(gdn, mlp, attn-mlp, all) 모두 full FT(noncausal)를 selection NLL(−0.040~−0.065, 5/5)과 장문 NLL(−0.186~−0.202, 5/5)에서 이겼다. 장문 정답률은 +4.7~5.3점이다.
- "나음/나쁨"은 사전 등록 규칙을 똑같이 적용한 것이지만, 이 비교들은 사전 등록 검정이 아니다.

## 해석

- **짧은 문항과 장문을 둘 다 잡을 수 있다.** DeltaNet 얼리기는 짧은 문항, attention LoRA는 장문이 강했다. MLP를 포함한 r16 LoRA는 둘 다 frozen보다 낫고, 장문은 attention LoRA와 같다. 장단이 있던 이유는 attention-only가 짧은 문항을 덜 배운 탓이었다.
- **"위치"도 "양"도 단독으로는 맞지 않았다.** 사전 등록한 두 가설(위치라면 attn-r160이 낫고, 양이라면 둘이 비슷하다)이 모두 틀렸다. 같은 1,080만 파라미터로 all이 attn-r160을 둘 다 이겼다.
  - 장문을 해친 것은 전체 파라미터 수가 아니었다. all은 attention LoRA의 10배를 학습했지만 장문이 같았다.
  - 장문을 해친 것은 **한 모듈을 크게 바꾸는 것**으로 보인다. attention의 rank를 16에서 160으로 올리자 장문이 63.1%에서 58.4%로 떨어져 full FT(58.3%)와 같아졌다.
  - 주의: alpha를 rank의 2배로 두어 scale이 2로 같다. 그래서 rank 160은 같은 LR에서 모듈당 변화량이 더 클 수 있다. "용량이 커서"인지 "한 step의 변화가 커서"인지는 이 실험으로 가를 수 없다.
- **논문과 다른 점:** 논문(GSM8K)에서는 attention-only가 가장 좋았다. 우리 과제에서는 짧은 문항에서 가장 약했다. DeltaNet에 붙인 LoRA도 해롭지 않았다(짧은 문항 +1.9점, 장문 같음). full FT에서 DeltaNet을 얼린 이득은 "DeltaNet을 건드리면 안 된다"보다 "full FT가 너무 많이 바꾼다"에 가까워 보인다.
- **네 위치 사이의 차이는 작다.** NLL 차이가 0.01 안팎이고 서로의 검정은 하지 않았다. 셋별로는 엇갈린다.
  - BABILong: attn·gdn 53~54%, mlp·all 51~52%
  - JEVal 장문: mlp·attn-mlp·gdn 76~77%, all 74%
- **seed 흔들림 감소는 attention LoRA에서만 보였다.** 다른 위치의 장문 정답률 표준편차는 1.4~1.8점으로 full FT(1.6~3.2)와 비슷했다. 다만 10-31 [Loss 비교](loss-screen.md)에서 같은 전체 LoRA로 loss만 바꾼 팔은 1.0~1.2점이었다. 위치의 성질인지는 판단을 보류한다.

## 왜 하나

- [Attention-only LoRA](lora-attention.md)는 짧은 문항에서 full FT와 같았고, 장문에서는 +4.8점 나았다.
- 이유는 두 가지로 읽을 수 있다.
  1. **위치:** attention 경로가 맞는 자리다. [논문](https://arxiv.org/abs/2604.22127)의 주장이다.
  2. **양:** 적게 바꿔서 원래 Qwen의 장문 능력을 덜 망가뜨렸다.
- 논문의 다른 위치 설정을 같은 데이터로 돌리고, 파라미터 수를 맞춘 대조군을 하나 더 넣어 이 둘을 가른다.

## 설정

모두 양방향(d3) 구조이고, 거르기와 같은 1.6만 행, 같은 seed 5개, 같은 입력(step마다 hash 대조)을 쓴다. rank 16, alpha 32, dropout 0.05, LR 2e-4이다.

| 팔 | 붙이는 곳 | 모듈 | 파라미터 | 논문 이름 |
|---|---|---|---|---|
| attn (완료, 대조군) | 6개 attention 층의 q/k/v/o | 24 | 108만 | softmax_only |
| **gdn** | 18개 DeltaNet 층의 in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj | 90 | 443만 | gdn_only |
| **mlp** | 24개 층 MLP의 gate/up/down | 72 | 531만 | mlp_only |
| **attn-mlp** | attn + mlp | 96 | 639만 | softmax_plus_mlp |
| **all** | 위 전부 | 186 | 1,082만 | all_layers |
| **attn-r160** | attn과 같은 자리, rank 160 (alpha 320) | 24 | 1,081만 | 없음 (all과 파라미터 수를 맞춘 대조군) |

모듈 수와 파라미터 수는 논문 표와 정확히 같다.

**학습 전 기준점:** 학습하지 않은 원래 Qwen을 원본 구조와 양방향 구조로 한 번씩 먼저 평가한다(보고만 함).

**대조군:** attn은 [Attention-only LoRA](lora-attention.md)의 lora-noncausal fit이다. noncausal과 frozen은 [학습 방식 거르기](training-methods.md)의 fit이다. 두 실험의 평가 코드 hash가 그때와 같은지 확인했다.

## 학습 전 기준점 (10-28 19:10 측정, 보고만 함)

정답 NLL이고, 괄호 안은 정답률(%)이다. 학습한 방식들은 5 seed 평균이다.

| 모델 | selection | 장문 평균 | BABILong | JEVal 장문 | QASPER | DocNLI 장문 |
|---|---|---|---|---|---|---|
| 학습 전 원본 구조 | 1.431 (45.3) | 2.004 (46.6) | 1.712 (45.0) | 4.618 (35.9) | 0.845 (55.2) | 0.843 (50.4) |
| 학습 전 양방향 | 1.404 (47.9) | 1.770 (49.1) | 1.536 (45.8) | 4.152 (34.3) | 0.664 (62.4) | **0.729 (54.0)** |
| 양방향 full FT | 0.808 (66.6) | 1.154 (58.3) | 1.543 (48.6) | 1.445 (66.2) | 0.647 (65.9) | 0.979 (52.3) |
| DeltaNet 얼리기 | **0.772 (68.0)** | 1.056 (60.5) | 1.411 (48.8) | 1.215 (72.7) | 0.638 (66.7) | 0.960 (54.0) |
| 양방향 attention LoRA | 0.814 (65.8) | **0.985 (63.1)** | 1.412 (53.1) | 1.093 (73.1) | 0.568 (71.2) | 0.865 (54.9) |

- **"LoRA가 원래 장문 점수를 지킨다"는 맞지 않았다.** 학습 전 장문 정답률은 49.1%로 낮다. 모든 방식이 학습 전보다 올랐고, LoRA가 가장 많이 올랐다(+14.0점, full FT +9.2점).
- **망가짐은 DocNLI 장문에서만 보였다.**
  - full FT는 학습 전보다 정답률이 낮아졌다(54.0 → 52.3).
  - 정답 NLL은 세 방식 모두 학습 전보다 나빠졌다(0.729 → 0.865~0.979).
  - LoRA가 가장 덜 나빠졌다.
- **BABILong과 QASPER는 LoRA만 크게 올랐다.** BABILong은 45.8 → 53.1(full FT 48.6), QASPER는 62.4 → 71.2(full FT 65.9)다.
- JEVal 장문은 모든 방식이 크게 올랐다(34% → 66~73%).
- 학습 없이 양방향으로만 바꿔도 원본 구조보다 높았다(selection +2.6, 장문 +2.5점). 하나의 측정이고 seed가 없다.

## 판정 규칙 (`plan.json`에 고정)

- 주 지표는 정답 NLL(selection, 장문)이고, 규칙은 지금까지와 같다(5 seed 중 4개 이상 + 계층 bootstrap 95% 구간이 0을 넘음).
- 검정 10개:
  - gdn, mlp, attn-mlp, all 각각 − attn: 위치에 따라 어떻게 달라지나
  - all − attn-r160: **위치 대 양**. 파라미터 수가 같으니, all이 attn-r160과 비슷하면 "양"이 핵심이고 attn-r160이 나으면 "위치"가 핵심이다.
- 보고만 하는 비교: 각 팔 − noncausal full FT, 각 팔 − frozen, attn-r160 − attn(rank만 키운 효과), 학습 전 기준점.

## 구현 검사 (`check.json`, `smoke.json`)

- 다섯 팔 모두 학습되는 파라미터가 LoRA뿐이고, 개수가 위 표와 같다.
- 학습 전 logit이 base 모델과 차이 0이다.
- 한 step 뒤 base 가중치는 그대로이고 B는 움직였다.
- 처음 12 step의 입력 hash가 거르기 smoke와 같았다.
- smoke 추정 학습 시간은 fit당 10.8~14.4분이고, 최대 메모리는 6.1GB 이하다.
  - 이전 LoRA 실험의 실제/추정 비율(1.85배)을 적용하면 seed 하나에 약 2.5시간(학습 약 1.9시간 + 평가 약 35분)이다.
  - 전체는 약 12.5시간이다.
- 실제 학습 시간은 fit당 19.6~24.9분(smoke의 1.7~1.8배)이었고, 전체는 12시간 49분이 걸렸다.

## 재현

```bash
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_placement_experiment check
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_placement_experiment smoke
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_placement_experiment freeze
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_placement_experiment run
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_placement_experiment report
```

코드는 [lora_placement_experiment.py](../../qwen_decision/v3/lora_placement_experiment.py)이고, 출력은 `runs/from-base/lora-placement-v1/`이다.
