# 학습 방식 거르기: 원래 Qwen에서 다섯 레시피, fit당 1.6만 행 × seed 5

[위키 홈](../README.md) · [쉬운 정리](v3-easy-guide.md) · [근거 논문](../papers/decision-training-recipes.md) · [구조 비교(이전 실험)](from-base-structures.md) · [평가 점검](evaluation-validity.md)

**상태: 완료(2026-10-27 14:01 ~ 10-28 04:33 KST).** 판정 규칙은 첫 fit 전에 `plan.json`으로 고정했다. 판정은 `runs/from-base/methods-v1/report.json`이다.

**한 줄 결론:** DeltaNet token mixer를 얼린 frozen이 거르기 6개 지표를 모두 통과했다. 6.55만 행 확인 실험으로 넘어간다. brier와 hoptrain은 noncausal과 차이가 없었다.

## 결과: 5 seed 평균

정답 NLL(낮을수록 좋음)이고, 괄호 안은 정답률(%)이다.

| 레시피 | selection | 장문 평균 | BABILong | JEVal 장문 | QASPER | DocNLI 장문 |
|---|---|---|---|---|---|---|
| causal | 0.840 (65.3) | 1.140 (57.7) | 1.411 (49.4) | 1.617 (63.0) | 0.624 (65.5) | 0.907 (52.9) |
| noncausal | 0.808 (66.6) | 1.154 (58.3) | 1.543 (48.6) | 1.445 (66.2) | 0.647 (65.9) | 0.979 (52.3) |
| **frozen** | **0.772 (68.0)** | **1.056 (60.5)** | 1.411 (48.8) | **1.215 (72.7)** | 0.638 (66.7) | 0.960 (**54.0**) |
| brier | 0.813 (66.2) | 1.112 (59.1) | 1.466 (48.8) | 1.435 (68.1) | 0.633 (66.7) | 0.912 (53.0) |
| hoptrain | 0.802 (66.4) | 1.100 (59.4) | 1.519 (48.6) | 1.382 (68.8) | 0.633 (66.3) | 0.866 (53.8) |

seed별 selection 정답률(131/151/173/191/211)은 다음과 같다.

- causal 64.5 / 64.3 / 65.0 / 66.2 / 66.5
- noncausal 66.5 / 66.5 / 67.2 / 65.8 / 67.3
- frozen 68.9 / 68.3 / 67.4 / 67.5 / 68.0

## 판정 (사전 등록 규칙: 5 seed 중 4개 이상 + 계층 bootstrap 95% 구간이 0을 넘음)

| 검정 | 지표 | 평균 차이 | 95% 구간 | 후보가 나은 seed | 판정 |
|---|---|---|---|---|---|
| 확인: noncausal−causal | 장문 NLL | +0.014 | [−0.085, +0.104] | 2/5 | 차이 없음 (재현 실패) |
| 확인: hoptrain−causal | selection NLL | −0.038 | [−0.064, −0.008] | 4/5 | **나음** |
| 거르기: frozen−noncausal | selection NLL | −0.036 | [−0.054, −0.020] | 5/5 | **나음** |
| 거르기: frozen−noncausal | 장문 NLL | −0.098 | [−0.182, −0.013] | 4/5 | **나음** |
| 거르기: brier−noncausal | selection NLL | +0.005 | [−0.014, +0.022] | 2/5 | 차이 없음 |
| 거르기: brier−noncausal | 장문 NLL | −0.042 | [−0.135, +0.060] | 3/5 | 차이 없음 |
| 거르기: hoptrain−noncausal | selection NLL | −0.006 | [−0.022, +0.010] | 3/5 | 차이 없음 |
| 거르기: hoptrain−noncausal | 장문 NLL | −0.054 | [−0.143, +0.031] | 3/5 | 차이 없음 |

보조 지표(frozen−noncausal)는 다음과 같다. 모두 같은 규칙을 통과했다.

- selection 정답률 +1.38점 [+0.07, +2.62], 5/5 seed
- selection Brier −0.017 [−0.027, −0.008], 5/5 seed
- 장문 정답률 +2.29점 [+0.12, +4.52], 4/5 seed
- 장문 Brier −0.033 [−0.057, −0.007], 4/5 seed

## 해석

- **frozen이 유일하게 통과했다.**
  - 지표 6개(NLL·정답률·Brier × selection·장문)가 모두 같은 방향이다.
  - 장문의 이득은 주로 JEVal 장문(정답률 +6.5점)에서 나왔다. BABILong과 QASPER는 차이가 작다.
  - 학습 메모리도 16.7GB에서 14.0GB로 줄었다.
  - LoRA placement 논문의 "Qwen3.5-0.8B에서 DeltaNet을 학습시키면 해롭다"와 같은 방향이다.
  - 다만 1.6만 행에서의 결과다. 이득이 "DeltaNet을 건드리면 해롭다"에서 오는지, "적은 데이터에서 파라미터를 줄이면 과적합이 준다"에서 오는지는 아직 구분하지 못한다. 그래서 6.55만 행 확인이 필요하다.
- **hoptrain−causal은 확인됐지만, hop 학습의 효과가 아니다.**
  - hoptrain은 noncausal과 차이가 없다(selection −0.006).
  - 사전 등록 검정이 아닌 noncausal−causal selection NLL도 −0.032 [−0.054, −0.005], 4/5 seed다.
  - 이전 실험의 hop 팔도 양방향 구조였다. 그러니 "hop이 짧은 문항에서 원본보다 낫다"는 신호는 양방향 attention에서 온 것이다.
  - hoptrain의 두 번째 읽기는 첫 읽기보다 장문 NLL이 +0.007 나빴다(5/5 seed). 다시 읽기는 여전히 정보를 더하지 못한다.
- **noncausal−causal 장문 NLL은 재현되지 않았다.** seed별 차이가 −0.156 ~ +0.150으로 흔들렸다. 이전 실험의 장문 신호는 seed 운이었을 가능성이 크다. 장문 평가(특히 JEVal 장문)는 seed마다 크게 흔들린다. 예를 들어 noncausal의 JEVal 장문 정답률은 seed에 따라 55~71%다.
- **brier는 효과가 없었다.** 확률 보정 지표(Brier)도 noncausal과 차이가 없었다.

## 다음

- **frozen 6.55만 행 확인 실험**(사전 등록된 다음 단계): frozen 대 noncausal, 원래 Qwen에서 출발한다.
  - 거르기와 같은 평가셋을 쓰고, 새 seed를 쓴다.
  - fit 하나에 약 1시간 40분이 든다. 학습이 약 1.5시간이고(이전 6.55만 행 실험에서 noncausal 1.53시간), 평가가 약 8분이다.
  - seed 3개면 6 fit, 약 10시간이다. seed 5개면 10 fit, 약 17시간이다.
- 통과하면 Jev Decision Index에서 d3-lite와 맞대결한다.

## 실행 기록

실제 fit당 학습 시간은 causal 20.8분, noncausal 22.8분, frozen 21.3분, brier 22.8분, hoptrain 44.4분이었다. smoke에서 잰 추정치(13~26분)보다 1.6배 길었다.
최대 메모리는 16.8GB 이하(frozen 14.0GB)였다. 다섯 레시피의 매 step 입력 hash도 일치했다.
seed 하나(학습 + 평가)에 2시간 54분, 전체 14시간 32분이 걸렸다.

## 왜 하나

- 이전 [구조 비교](from-base-structures.md)에서 정답률로는 원본을 넘은 구조가 없었다.
- 사후에 정답 NLL로 다시 재 보니 두 신호가 있었다.
  - 양방향이 원본보다 장문 NLL이 낮았다.
  - hop 학습(두 번 읽고 loss를 반씩)이 원본보다 짧은 문항 NLL이 낮았다.
- 논문 조사([근거](../papers/decision-training-recipes.md))에서 학습 방식 후보 두 개를 더 얻었다.
  - Qwen3.5-0.8B에서 DeltaNet 경로를 학습시키면 해롭다는 보고.
  - CE에 Brier를 더하는 손실.
- 판정 모델 논문들은 fit당 2천~7.6만 문항으로 학습했다([평가 점검 5절](evaluation-validity.md)). 학습 방식을 거르는 데는 70만 행 전체가 필요 없다. 그래서 fit당 1.6만 행으로 줄이고, 그 대신 seed를 5개로 늘린다.

## 다섯 레시피

| 이름 | 구조 | 학습 방식 | 근거 | 학습 비용 |
|---|---|---|---|---|
| causal | 원래 Qwen (전부 앞→뒤) | 정답 CE | 기준 | 1배 |
| noncausal | d3식 양방향 | 정답 CE | 기준 (방식 비교의 대조군) | 1배 |
| frozen | 양방향 | DeltaNet token mixer(`linear_attn`, 162개 텐서, 1.9억 파라미터)를 얼림. MLP·norm·attention·embedding은 학습 | LoRA placement 논문: DeltaNet만 학습하면 GSM8K −14.8점 | 1배 이하 |
| brier | 양방향 | 정답 CE + 1×Brier (보이는 선택지 위의 확률과 정답의 제곱 오차) | NDM(log-loss + Brier), DKnown-Jev, OneJev | 1배 |
| hoptrain | 양방향 | hop처럼 두 번 읽고 loss를 반씩. 단 **답은 첫 읽기에서** 내서 추론은 forward 한 번 | 이전 실험: hop 모델의 첫 읽기 NLL이 두 번째 읽기와 같았고 원본보다 낮았다 | 약 2배 |

## 통제 조건

- **출발점:** `models/qwen3.5-0.8b`(판정 학습 전). 모든 fit이 같은 가중치에서 시작한다.
- **데이터:** `data/longctx/train-nocap104-16k.jsonl`. 104개 출처 풀에서 출처당 157~158행씩 16,384행을 뽑았고, 토큰은 자르지 않았다.
- **학습량:** 1,024 update × 16행으로 풀을 정확히 한 바퀴 돈다. LR 1e-5, warmup 64, cosine 10%, FP32-master AdamW, clip 1.
- **같은 입력:** 매 step의 행·선택지 순열·코드 배정이 다섯 레시피에서 같은지 hash로 확인한다.
- **seed:** 131, 151, 173, 191, 211. 이전 from-base 실험과 겹치지 않는 새 seed다.
- **평가:** selection 4,147, BABILong 0k~16k 600, JEVal 장문 684, QASPER 284, DocNLI 장문 1,188.
  - BABILong 32k~128k는 뺐다. 평가 시간의 대부분을 차지하는데, 모든 구조가 바닥 근처라 학습 방식을 거르는 데 정보가 적다.
  - LongBench v2도 뺐다.

## 판정 규칙 (`plan.json`에 고정)

- **주 지표: 정답 NLL**(정답 선택지 확률의 −log, 출처 평균). 장문 NLL은 네 장문 셋의 출처 평균을 다시 평균한 값이다.
- 정답률과 Brier는 보조 지표로 함께 보고한다.
- **확인 검정 2개.** 이전 실험의 사후 발견을 새 seed로 다시 잰다.
  - noncausal−causal: 장문 NLL
  - hoptrain−causal: selection NLL
- **거르기 검정 6개.** frozen·brier·hoptrain 각각을 noncausal과 비교하고, 지표는 selection NLL과 장문 NLL이다.
- **"낫다"의 기준:** seed 5개 중 4개 이상에서 후보가 낫고, seed × 문항 계층 bootstrap 95% 구간이 0을 넘는다. "나쁘다"는 반대 방향으로 같은 규칙을 적용한다.
- 거르기에서 통과한 레시피는 주장이 아니라 **6.55만 행 규모 확인 실험으로 넘어가는 후보**다.

## 구현 검사 (`check.json`)

- Brier와 정답 NLL이 손으로 계산한 값과 일치했다(예: 정답 확률 0.2 → NLL 1.609).
- frozen
  - DeltaNet mixer 162개 텐서가 optimizer에 들어가지 않았다.
  - 한 step 뒤 BF16 가중치가 그대로였다.
  - 다른 층(MLP·attention)의 FP32 master는 바뀌었다.
  - 학습 파라미터는 5.63억, 얼린 것은 1.90억이다.
- hoptrain의 답(첫 읽기)이 학습 전 noncausal 출력과 정확히 같았다(차이 0).

## 재현

```bash
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.training_method_experiment pool
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.training_method_experiment check
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.training_method_experiment smoke
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.training_method_experiment freeze
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.training_method_experiment run
PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.training_method_experiment report
```

코드는 [training_method_experiment.py](../../qwen_decision/v3/training_method_experiment.py)이고, 출력은 `runs/from-base/methods-v1/`에 쌓인다.
