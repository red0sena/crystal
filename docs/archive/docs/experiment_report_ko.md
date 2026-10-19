# Qwen3.5 선택지 판단 모델 개발과 평가 기록

작성 기준: 2026년 10월 10일, 한국 시간.

Qwen3.5-0.8B에 문맥·질문·선택지를 넣고, 답변 문장을 생성하지 않고 선택지 점수를 반환하는 모델을 구현했다. 이후 영어 텍스트 16개 과제로 언어 모델 부분을 미세조정하고, 원본과 학습 모델을 여러 평가 조건에서 비교했다.

현재 결과는 선택지 판단 전용 학습의 효과를 보여준다. 같은 출처에서 따로 남겨 둔 평가 데이터에서는 분야·인지 과정·의도 등의 분류 성능과 선택지 순서 변경에 대한 일관성이 개선됐다. 새 데이터에서는 개선이 작업마다 달랐고, 한국어 KMMLU에서 응답 코드 편향과 난이도 기준의 불일치가 드러났다. 범용 난이도 분류나 45개 선택지 지원이 검증됐다고 보기는 어렵다.

초기 구현은 이미지 입력도 지원하지만, 이번 미세조정과 본 평가의 중심은 텍스트다. 이미지 학습, 상태를 공유하는 여러 질문 처리, 장문 성능 평가와 서비스 최적화는 후속 작업으로 남아 있다.

## 목표와 d1에서 참고한 방식

목표는 이미지나 긴 문서를 읽고, 객관식 답·분야·작업 종류·난이도처럼 제공된 후보 중 하나를 판단하는 작은 모델을 만드는 것이다. 먼저 원본 모델에서 생성 없는 점수 반환을 구현하고, 이후 판단 과제에 맞게 학습했다.

Liquid AI의 d1-3B는 LFM2.5-VL-3B에서 만든 결정 모델로, 상태와 질문을 받아 출력 토큰 없이 한 번의 forward로 답을 반환한다. [d1 공식 모델 카드](https://huggingface.co/LiquidAI/d1-3B)

공개된 학습 설명에서 참고한 요소는 긴 입력 학습, 선택지 섞기, 데이터의 지름길 제거, 서로 다른 난수 시드와 데이터 혼합, 가중치 평균이다. 우리 학습은 이를 Qwen에 적용한 실험이며, 공개되지 않은 d1의 정확한 손실·데이터 혼합·학습 설정을 재현한 것은 아니다. [Liquid AI 공식 학습 설명](https://www.liquid.ai/blog/d1-open)

d1은 같은 상태에 붙은 여러 질문에서 상태를 한 번 읽는 기능을 제공한다. 현재 Qwen 구현은 질문마다 전체 입력을 다시 읽는다. 두 모델의 출력 방향은 비슷하지만, 기반 모델·상태 공유·확률 보정·추론 최적화는 동일하지 않다. [d1 사용 인터페이스](https://huggingface.co/LiquidAI/d1-3B#-how-to-use)

## 생성 없이 선택지 점수를 계산하는 구조

외부 입력은 `state`, `question`, `options`로 나뉜다. 모델 내부에서는 이를 하나의 chat 입력으로 직렬화한다. 선택지의 이름은 결과를 식별하는 키이고, 설명은 실제 입력에 들어간다. 이름이나 설명이 여러 토큰으로 구성돼도, 응답 위치에서는 A·B·C 같은 단일 토큰 코드의 점수를 읽는다.

```json
{
  "state": "한 주문에 두 번 결제되어 환불을 요청했다.",
  "question": "어느 팀이 담당해야 하는가?",
  "options": {
    "billing": "결제와 환불 담당",
    "technical": "앱 오류 담당",
    "sales": "새 구매와 상품 문의 담당"
  }
}
```

위 입력은 내부적으로 `A. 결제와 환불 담당`, `B. 앱 오류 담당`, `C. 새 구매와 상품 문의 담당`으로 표시된다. 결과는 다시 `billing`, `technical`, `sales` 이름으로 매핑된다. 후보는 요청마다 달라질 수 있으므로 고정된 클래스 전용 출력층을 새로 붙이지 않았다.

계산 과정은 다음과 같다.

1. 공식 chat template을 `enable_thinking=False`로 적용한다.
2. 각 응답 코드가 실제 답변 경계에서 단일 토큰인지 검사한다.
3. backbone을 한 번 실행하고 마지막 실제 입력 위치의 hidden state를 읽는다.
4. 기존 LM Head에서 응답 코드에 대응하는 행만 계산한다.
5. 후보 logits를 softmax로 정규화하고 가장 큰 값을 선택한다.

```text
z_i = LM_Head[응답 코드 i] · 마지막 입력 hidden state
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
선택 = argmax_i p_i
```

`generate()`를 호출하지 않고, 이 점수 반환 경로의 출력 토큰은 0개다. 모든 입력을 처리하는 비용은 그대로 있으므로 긴 문서를 넣으면 forward 시간이 늘어난다. softmax는 제공된 후보끼리 정규화한 점수이며, 실제 정답률로 보정된 신뢰도는 아니다.

한 번의 forward 안에서도 여러 신경망 층이 문맥을 처리한다. 다만 단계별 추론 문장을 생성하는 CoT는 수행하지 않으며, CoT와 같은 문제 해결 성능을 보장하지 않는다.

구현: [선택지 scorer](/home/khkim/qwen-decision/scorer.py) · [초기 구현과 실행 설명](/home/khkim/qwen-decision/README.md)

## 실행 환경과 구현 검증

기반 가중치는 `Qwen/Qwen3.5-0.8B`, GPU는 NVIDIA RTX A6000이다. 학습 manifest에 기록된 PyTorch는 2.10.0+cu128이다. 추론은 GPU에서 BF16을 사용했고, 학습은 FP32 master 가중치와 optimizer 상태에 BF16 autocast를 적용했다.

초기 scorer는 2~26개 선택지와 이미지 입력을 지원하며 기본 입력 제한은 32,768 tokens다. 한도를 넘는 입력은 자동으로 잘라내지 않고 오류를 반환한다. 후속 KMMLU 평가는 별도 프로토콜로 45개 코드까지 확장했다. 기본 scorer 자체의 제한을 45개로 바꾼 것은 아니다.

| 검증 항목 | 확인한 내용 |
| --- | --- |
| 선택 LM Head와 전체 LM Head | 검사한 입력의 logits 최대 차이 0.0 |
| 텍스트 입력 | 문의 라우팅 예제에서 점수 반환 경로 확인 |
| 이미지 입력 | 빨강·파랑 이미지 교체에 따른 출력 변경 확인 |
| 긴 입력 | 9,137 tokens 입력 실행 확인 |
| 생성 제거 | 단일 backbone forward와 출력 토큰 0개 검사 |
| 입력 한도 | 한도 초과 시 추론 전 거부 검사 |

이미지 예제와 긴 입력 예제는 기능 검사다. 일반 이미지 이해 정확도, 32K 장문 정확도 또는 32K 지연시간의 검증을 의미하지 않는다. 현재 선택적 DeltaNet 고속 커널을 설치하지 않은 기준 PyTorch 경로를 사용한다. Liquid AI의 컴파일·CUDA Graph 기반 공식 지연시간과 직접 비교할 수 없다.

검증 결과: [초기 smoke 검사](/home/khkim/qwen-decision/smoke-results.json)

## 학습 전 원본 모델 평가

MMLU-Pro 전체 test 12,032문항을 0-shot·reasoning OFF 조건으로 평가했다. 점수 반환 방식과 일반 답 생성 방식을 각각 5회 실제로 실행했다. 모든 회차는 고정 입력과 선택지 순서를 사용했다.

| 방식 | 회차 수 | 매 회 정답 수 | 정확도 | 회차 간 차이 |
| --- | --- | --- | --- | --- |
| 선택지 logits와 softmax argmax | 5 | 2,337 / 12,032 | 19.42% | 없음 |
| 전체 vocabulary에서 greedy 답 생성 | 5 | 2,340 / 12,032 | 19.45% | 없음 |

답 생성은 A~J 중 답 문자와 종료 토큰을 생성했고, 설명이나 reasoning 문장은 생성하지 않았다. 형식 오류는 0건이었다. 두 방식의 정확도 차이는 약 0.025%p이며, padding과 BF16 연산 경로 차이도 포함되므로 방법의 우열로 해석하지 않는다.

동일한 결정적 실행을 5번 반복한 결과이므로, 다섯 개의 독립 확률 실험이나 서로 다른 추론 경로를 평가한 것은 아니다. 이 0-shot 직접 답변 조건은 공식 5-shot CoT 평가 조건과 다르다.

결과: [MMLU-Pro 점수 반환 5회](/home/khkim/qwen-decision/results/mmlu-pro-0shot-5runs/summary.json) · [MMLU-Pro 답 생성 5회](/home/khkim/qwen-decision/results/mmlu-pro-generate-5runs/summary.json)

다른 객관식 데이터도 원본 모델로 각 1회씩 평가했다. CommonsenseQA와 PIQA는 공개 정답이 있는 validation, ARC-Challenge는 test를 사용했다.

| 벤치마크 | 문항 수 | 답 생성 | 점수 반환 | 균등 무작위 기대값 |
| --- | --- | --- | --- | --- |
| ARC-Challenge | 1,172 | 57.00% | 57.42% | 25.02% |
| CommonsenseQA | 1,221 | 42.42% | 42.34% | 20.00% |
| PIQA | 1,838 | 58.92% | 59.09% | 50.00% |

무작위 기대값은 각 문항에서 후보를 같은 확률로 선택했을 때의 이론값이다. 별도의 무작위 모델을 실행한 점수가 아니다.

결과: [세 벤치마크 답 생성](/home/khkim/qwen-decision/results/choice-benchmarks-generate-1run/summary.json) · [같은 문항의 점수 반환](/home/khkim/qwen-decision/results/choice-benchmarks-probabilities-1run/summary.json)

## 선택지 위치와 응답 표기 편향

내용 순서 변경, 코드 배정 변경, 문자·숫자 표기 변경, 지시문 변경을 대조했다. 내용 순서와 코드를 독립적으로 섞어, 첫 번째 보기를 선호하는 현상과 A 또는 1이라는 응답 코드를 선호하는 현상을 구분했다.

내용과 코드 배정을 모두 섞은 원본 모델의 결과는 다음과 같다.

| 벤치마크 | 문자 조건 첫 위치 | 문자 조건 A | 숫자 조건 첫 위치 | 숫자 조건 1 |
| --- | --- | --- | --- | --- |
| ARC-Challenge | 19.88% | 29.69% | 20.48% | 36.18% |
| CommonsenseQA | 7.21% | 28.75% | 6.47% | 38.98% |
| PIQA | 69.31% | 48.64% | 47.77% | 67.03% |

A와 1이 첫 위치에 고정되지 않았는데도 표기 선호가 남았다. PIQA 문자 조건에서는 첫 위치 선택이 높았다. 동률 문항을 제외해도 일부 편향이 유지됐으므로, argmax 동률 처리만으로 설명할 수 없다.

같은 원래 선택지 순서에서 지시문을 중립적으로 바꿨을 때 ARC 62.12%, CommonsenseQA 53.40%, PIQA 64.20%가 나왔다. 지시문은 예시 A와 답 안내 표현을 함께 바꿨으므로, 개선을 예시 한 부분의 삭제 효과로 단정하지 않는다. 이 결과는 프롬프트와 응답 표기가 평가에 영향을 준다는 근거다.

결과: [편향 대조 실험](/home/khkim/qwen-decision/results/choice-bias-controls/summary.json) · [내용과 코드의 독립 변경](/home/khkim/qwen-decision/results/choice-bias-crossed/summary.json)

## 텍스트 학습 데이터 구성

학습 예제 63,054개, 개발 예제 14,078개, 평가 예제 30,801개를 구성했다. 데이터는 영어 텍스트다. 한 문제에서 정답 선택과 분야 분류를 각각 만들 수 있으므로, 이 수는 서로 다른 원본 문제 수와 같지 않다.

정답 선택뿐 아니라 분류·라우팅·문장 관계·인지 과정·상대 난이도를 공통 `state / question / options` 형식으로 바꿨다.

| 과제 | 학습 출처와 조건 | 학습 예제 수 |
| --- | --- | --- |
| 객관식 정답 선택 | MMLU-Pro | 6,000 |
| 과학 객관식 정답 선택 | ARC Easy + Challenge | 3,350 |
| 일반 상식 정답 선택 | CommonsenseQA | 6,000 |
| 물리적 상식 정답 선택 | PIQA | 6,000 |
| 지문 독해 | RACE | 4,000 |
| 뉴스 주제 분류 | AG News | 4,000 |
| 긍정과 부정 분류 | SST-2 | 4,000 |
| 감정 종류 분류 | Emotion | 4,000 |
| 고객 문의 의도 선택 | BANKING77에서 후보 16개 | 4,000 |
| 문장 간 함의 | GLUE RTE | 2,375 |
| 문장 의미 동등성 | GLUE MRPC | 3,505 |
| 영어 문법적 수용성 | GLUE CoLA | 4,000 |
| 학문 분야 분류 | MMLU-Pro category 14개 | 4,000 |
| 요청한 작업 종류 분류 | Natural Instructions Categories 8개 | 1,930 |
| Bloom 인지 과정 분류 | CogBench 6개 과정 | 5,000 |
| AMC 내부 상대 난이도 분류 | Easy2Hard-Bench E2H-AMC | 894 |

분야 분류와 난이도 분류도 학습에 포함했다. 분야는 MMLU-Pro에서 만든 4,000개, 난이도는 AMC에서 만든 894개다. 따라서 이와 같은 출처의 보류 데이터 점수 향상을 미학습 과제에 대한 성능으로 표현하면 안 된다.

MMLU-Pro 공식 test를 문제 텍스트 그룹 기준 약 80/10/10으로 다시 나눈 뒤, 정답 선택과 분야 분류를 만들었다. 같은 문제의 파생 예제는 같은 분할에 둔다. 학습 이후 공식 test 전체를 다시 평가하면 학습 문제를 포함하게 된다. 비교에는 별도로 남긴 custom test를 사용했다.

RACE는 지문 단위, Natural Instructions는 출처 계열 단위로 분리했고, 데이터 전체에서 같은 정규화 입력을 분할 사이에 겹치지 않게 묶었다. 정답·해설·분야 메타데이터·원래 수치 난이도는 모델 입력에서 제외했다.

BANKING77은 전체 77개 의도를 출처로 쓰되, 요청마다 정답과 오답 15개를 제공하는 후보 16개 선택 과제다. 표준 77개 전체 클래스 분류 점수와는 다르다.

CogBench의 Bloom 라벨은 Remember·Understand·Apply·Analyze·Evaluate·Create라는 인지 과정이다. 사람 라벨 595개와 분류기 silver 라벨 4,405개를 학습에 사용했다. silver는 공개 confidence 0.8 이상만 사용하고 손실 가중치를 0.5로 낮췄다. Bloom 인지 과정과 풀이 난이도는 별도 과제로 구분했다.

AMC 난이도는 학습에 남은 문제의 normalized IRT rating을 삼분위로 나눈 것이다. 경계는 0.279354, 0.560366이고, 학습의 low·medium·high는 각각 298개다. 이 기준은 AMC 안에서의 상대 난이도이며, 모든 분야나 응시자 집단에 적용되는 범용 상·중·하 기준은 아니다.

데이터 설명: [학습 데이터 구성](/home/khkim/qwen-decision/data/text-decisions-v1/README.md) · [분할과 개수 manifest](/home/khkim/qwen-decision/data/text-decisions-v1/manifest.json) · [AMC 난이도 기준](/home/khkim/qwen-decision/data/text-decisions-v1/difficulty_rubric.json)

## 실제 학습 방식과 완료 상태

언어 backbone과 tied embedding/LM Head의 752,393,024개 파라미터를 전체 미세조정했다. vision encoder는 동결했고 이미지 학습은 수행하지 않았다. 새로운 분류 head나 LoRA를 붙인 학습은 아니다.

각 입력에서 기존 LM Head의 후보 코드 logits만 계산하고, 정답 코드에 대한 sample-weighted cross-entropy를 최소화했다. 답변 문장이나 CoT 정답을 생성하도록 학습하지 않았다.

```text
loss = 예제 가중치 × cross_entropy(제공된 후보 코드 logits, 현재 정답 코드)
```

선택지 내용과 코드 배정을 독립적으로 섞고 매번 정답 매핑을 갱신했다. 후보가 9개 이하인 예제의 25%에는 숫자 코드를 사용했다. 실제 학습 데이터의 후보 수는 최대 16개였고, 알파벳 응답 코드는 A~P까지였다.

RACE 학습 예제의 25%에는 학습 분할에서만 가져온 무관한 지문 최대 3개를 삽입하고, 관련 지문의 위치를 섞었다. 학습 입력 한도는 8,192 tokens이며, 증강이 한도를 넘으면 되돌리고 원본이 한도를 넘으면 기록 후 제외하도록 구성했다. 이것은 32K 입력 학습을 완료했다는 뜻이 아니다.

| 설정 | 값 |
| --- | --- |
| GPU | NVIDIA RTX A6000 |
| 학습 정밀도 | FP32 master 가중치·optimizer 상태, BF16 autocast |
| optimizer | AdamW |
| 최대 학습률 | 2e-5 |
| 학습률 스케줄 | 5% warmup, cosine decay |
| microbatch와 유효 batch | 최대 4개, 유효 32개 |
| 메모리 절약 | gradient checkpointing, padded token budget 8,192 |
| 첫 학습 | seed 41, 자연적인 과제 빈도 |
| 두 번째 학습 | seed 73, 과제 크기 제곱근에 비례한 sampling |
| 학습량 | 각 실행 63,054개 draw, 1,971 optimizer steps |
| 모델 선택 | 과제별 32개인 고정 개발 표본 512개, 개발 macro accuracy |

두 실행은 원본 모델에서 각각 시작했다. 두 번째 실행은 복원 추출이므로 개별 예제를 반복하거나 생략할 수 있다. 최종 두 가중치의 평균과 각 실행의 마지막·개발 최적 checkpoint를 비교했고, 평균 모델이 개발 정확도 74.41%로 선택됐다. test 데이터는 모델 선택에 사용하지 않았다.

한국 시간 2026년 10월 10일 12:31:59에 학습을 시작했고, 모델 선택과 초기 평가까지의 파이프라인은 16:00:53에 완료됐다. 이 구간은 약 3시간 29분이며, 순수 optimizer 실행 시간만을 뜻하지 않는다.

학습 결과: [학습 설정](/home/khkim/qwen-decision/training/text-decision-full-v1/manifest.json) · [개발 데이터 기반 모델 선택](/home/khkim/qwen-decision/training/text-decision-full-v1/selection.json) · [완료 상태](/home/khkim/qwen-decision/training/text-decision-full-v1/pipeline.json)

## 학습 후 같은 출처의 보류 데이터 비교

각 과제에서 보류 예제 120개씩, 총 1,920개를 원본과 학습 모델로 비교했다. 아래 정확도는 기준 선택지 순서인 canonical 조건이다. 공식 전체 벤치마크 점수는 아니다.

| 과제 | 원본 | 학습 모델 | 변화 | 학습 모델의 순서 변경 일치율 |
| --- | --- | --- | --- | --- |
| 객관식 정답 선택 | 25.83% | 36.67% | +10.83%p | 73.33% |
| 과학 객관식 정답 선택 | 74.17% | 82.50% | +8.33%p | 92.50% |
| 일반 상식 정답 선택 | 47.50% | 66.67% | +19.17%p | 85.83% |
| 물리적 상식 정답 선택 | 76.67% | 77.50% | +0.83%p | 75.83% |
| 지문 독해 | 70.00% | 70.83% | +0.83%p | 94.17% |
| 뉴스 주제 분류 | 79.17% | 85.83% | +6.67%p | 98.33% |
| 긍정과 부정 분류 | 83.33% | 84.17% | +0.83%p | 99.17% |
| 감정 종류 분류 | 52.50% | 95.00% | +42.50%p | 100.00% |
| 고객 문의 의도 선택 | 32.50% | 91.67% | +59.17%p | 97.50% |
| 문장 간 함의 | 65.83% | 76.67% | +10.83%p | 100.00% |
| 문장 의미 동등성 | 52.50% | 79.17% | +26.67%p | 100.00% |
| 영어 문법적 수용성 | 65.83% | 71.67% | +5.83%p | 95.00% |
| 학문 분야 분류 | 49.17% | 85.00% | +35.83%p | 97.50% |
| 요청한 작업 종류 분류 | 90.00% | 78.33% | -11.67%p | 90.83% |
| Bloom 인지 과정 분류 | 25.83% | 87.50% | +61.67%p | 98.33% |
| AMC 내부 상대 난이도 분류 | 33.33% | 65.83% | +32.50%p | 95.00% |

순서 변경 일치율은 한 예제에 고정 permutation을 적용했을 때 같은 의미의 선택지를 골랐는지의 비율이다. 모든 순서와 모든 응답 표기에 대한 불변성을 보장하지 않는다.

분야·의도·인지 과정·감정 등의 개선이 컸다. 작업 종류 라우팅은 90.00%에서 78.33%로 하락했다. 따라서 이번 미세조정이 모든 판단 과제를 일괄 개선했다고 볼 수는 없다. MMLU-Pro custom 보류 표본은 25.83%에서 36.67%로 올랐지만, 120문항 결과를 공식 12,032문항 점수로 대체해서는 안 된다.

결과: [16개 과제 비교와 shuffled 조건](/home/khkim/qwen-decision/training/text-decision-full-v1/evaluation/summary.json)

## 학습에 사용하지 않은 새 객관식 데이터 비교

HellaSwag·WinoGrande·OpenBookQA를 추가로 내려받아, 로컬 미세조정 자료와 겹치는 입력과 중복을 정리한 뒤 전체 공개 정답 분할을 각 모델에서 1회 평가했다. 후보 점수의 argmax를 사용하고 출력 토큰은 0개다. 기반 모델의 사전학습에서 이 데이터를 봤는지는 확인할 수 없다.

| 벤치마크 | 사용한 분할 | 평가 수 | 원본 | 학습 모델 | 변화 |
| --- | --- | --- | --- | --- | --- |
| HellaSwag | validation | 10,042 | 34.46% | 39.10% | +4.64%p |
| WinoGrande | validation | 1,267 | 50.04% | 48.93% | -1.10%p |
| OpenBookQA | test | 500 | 51.60% | 60.20% | +8.60%p |

HellaSwag는 자연스러운 상황의 다음 전개 선택, WinoGrande는 문맥에 맞는 대명사 지시 대상 선택, OpenBookQA는 기초 과학 지식과 추론을 요구하는 객관식 문제를 평가했다. HellaSwag와 OpenBookQA는 개선됐고 WinoGrande는 소폭 하락했다. 이 점수도 선택지 코드 logits 방식이며, 다른 completion likelihood나 few-shot 평가 점수와 직접 동일시하지 않는다.

결과: [새 객관식 데이터 평가](/home/khkim/qwen-decision/results/unseen-text-v1/summary.json) · [독립 검산](/home/khkim/qwen-decision/results/unseen-text-v1/audit.json)

## 분야와 난이도 분류의 전체 보류 데이터 평가

처음에는 분야별·난이도별 예제와 소수의 거친 분야 분류 데모를 확인했다. 소수 예제의 결과는 일반 성능의 근거로 삼지 않고, 이후 로컬 보류 분할의 분야·난이도 전체를 평가했다.

| 과제 | 평가 수 | 원본 | 학습 모델 | 변화 |
| --- | --- | --- | --- | --- |
| 학문 분야 분류 | 1,181 | 58.09% | 85.86% | +27.77%p |
| AMC 내부 상대 난이도 분류 | 2,974 | 35.31% | 67.38% | +32.08%p |

| AMC 등급 | 원본 적중률 | 학습 모델 적중률 |
| --- | --- | --- |
| low | 18.76% | 80.86% |
| medium | 57.19% | 31.54% |
| high | 31.29% | 88.77% |

분야 분류와 AMC 상대 난이도는 학습에 사용한 과제·출처에서 따로 남긴 데이터다. 난이도 전체 정확도는 개선됐지만 medium 적중률은 하락했다. 이 결과만으로 다른 분야나 다른 기준의 난이도를 잘 분류한다고 주장할 수 없다.

결과: [초기 예제 확인](/home/khkim/qwen-decision/results/classification-examples-v1/results.json) · [전체 보류 분할 평가](/home/khkim/qwen-decision/results/full-classification-v1/summary.json) · [독립 검산](/home/khkim/qwen-decision/results/full-classification-v1/audit.json)

## 별도 분야와 난이도 데이터 조사

문제풀이 데이터의 메타데이터도 분류 정답으로 사용할 수 있도록 SuperGPQA·KMMLU·MATH·Omni-MATH의 원본과 라벨을 조사했다.

| 자료 | 원본 규모 | 활용 가능한 메타데이터 | 현재 상태 |
| --- | --- | --- | --- |
| SuperGPQA | 26,529문항 | 학문 분야 13개와 easy·middle·hard | 원본과 학습 모델 평가 완료 |
| KMMLU | test 35,030문항 | 분야 45개와 Human Accuracy | 원본과 학습 모델 평가 완료 |
| MATH | test 5,000문항 | 수학 유형과 level | 원본 다운로드와 라벨 조사 완료 |
| Omni-MATH | 4,428문항 | 수학 분야와 수치 difficulty | 원본 다운로드와 라벨 조사 완료 |

SuperGPQA에는 기존 MMLU-Pro 등과 겹치는 문제가 있어 전체를 완전히 독립적인 평가 자료로 간주하지 않았다. 원래 저장소가 미세조정 출처 목록에 없다는 것과, 포함된 문제를 한 번도 학습에 쓰지 않았다는 것은 별개다.

외부 평가용 입력에서는 기존 학습 자료뿐 아니라 앞서 사용한 MMLU-Pro·AMC 원본 전체도 대조했다. 정규화한 정확한 일치, 포함 관계, 높은 5-word shingle 중복과 일부 문자열 유사도를 검사했다. 이 절차는 모든 번역·의미상 중복이나 기반 모델의 사전학습 노출을 배제한 검증은 아니다.

SuperGPQA에서 1,302개를 제외해 25,227문항을 남겼다. KMMLU에서는 데이터 내부 중복과 분야 라벨 충돌 59개를 제외해 34,971문항을 남겼고, 그중 사람 정답률이 없는 76개를 난이도 평가에서 제외했다. SuperGPQA의 Hub split 이름은 train이지만 여기서는 외부 평가에만 사용했다.

자료 조사: [SuperGPQA와 수학 자료 조사](/home/khkim/qwen-decision/data/external-classification-candidates-v1/inspection.json) · [KMMLU 조사](/home/khkim/qwen-decision/data/external-classification-candidates-v1/kmmlu-inspection.json) · [외부 평가 데이터와 제외 조건](/home/khkim/qwen-decision/data/external-classification-eval-v1/manifest.json)

## SuperGPQA와 KMMLU의 외부 분류 평가

문제와 원래 객관식 보기만 입력으로 제공하고 분야나 난이도를 분류했다. 원래 문제의 정답을 맞히는 시험이 아니다. 정답 분야·난이도·사람 정답률은 모델 입력에 넣지 않았다.

SuperGPQA 분야는 원래 discipline 13개, 난이도는 원래 easy·middle·hard 라벨을 사용했다. KMMLU 분야는 원래 45개 과목을 사용했다. KMMLU에는 공식 상·중·하 라벨이 없어, 사람 정답률에서 쉬움 ≥80%, 중간 ≥50% 및 <80%, 어려움 <50% 구간을 평가 전에 고정했다.

두 모델은 같은 입력 토큰·선택지 순서·배치를 사용했다. 선택지 순서는 문제별로 고정된 난수로 섞었고, 모델당 1회씩 실행했다. 전체 요청은 모델당 120,320개, 총 예측은 240,640개다.

| 분류 작업 | 평가 수 | 원본 | 학습 모델 | 변화 |
| --- | --- | --- | --- | --- |
| SuperGPQA 분야 13개 | 25,227 | 61.50% | 67.26% | +5.76%p |
| SuperGPQA 난이도 | 25,227 | 35.60% | 45.22% | +9.62%p |
| KMMLU 분야 45개 | 34,971 | 12.31% | 12.45% | +0.14%p |
| KMMLU 난이도 | 34,895 | 34.12% | 12.29% | -21.83%p |

전체 정확도는 문항 수에 비례한 점수다. 클래스별 문항 수가 다르므로, 각 클래스 적중률을 같은 비중으로 평균한 balanced accuracy와 각 클래스 F1의 평균인 macro F1도 확인했다.

| 분류 작업 | 등급이나 분야 균등 평균 원본 → 학습 | macro F1 원본 → 학습 | 항상 최빈 클래스 기준 |
| --- | --- | --- | --- |
| SuperGPQA 분야 13개 | 52.13% → 66.06% | 46.72% → 59.68% | 36.88% |
| SuperGPQA 난이도 | 34.41% → 40.47% | 34.17% → 39.49% | 43.43% |
| KMMLU 분야 45개 | 14.09% → 15.44% | 11.52% → 12.34% | 2.86% |
| KMMLU 난이도 | 33.20% → 34.90% | 24.03% → 13.22% | 82.57% |

SuperGPQA 분야는 전체 정확도뿐 아니라 분야별 평균 적중률도 52.13%에서 66.06%로 상승했다. 역사·교육·사회학 등의 적중률이 개선됐고, 과학은 74.50%에서 70.63%로 하락했다. 과학과 공학 사이의 혼동은 계속 남았다.

KMMLU 분야 정확도는 거의 변하지 않았으며, 아래의 응답 코드 편향 때문에 순수한 분야 이해 능력으로 해석할 수 없다. 난이도는 두 자료 모두 세부 등급별로 다른 변화가 나타났다.

| 자료 | 쉬움 적중률 원본 → 학습 | 중간 적중률 원본 → 학습 | 어려움 적중률 원본 → 학습 |
| --- | --- | --- | --- |
| SuperGPQA | 24.77% → 35.21% | 43.36% → 69.82% | 35.11% → 16.38% |
| KMMLU | 31.47% → 56.47% | 33.97% → 42.43% | 34.17% → 5.82% |

SuperGPQA 난이도 전체 정확도는 상승했지만, 어려움 적중률은 35.11%에서 16.38%로 하락했다. 학습 모델 45.22%는 항상 middle을 고르는 43.43%보다 약 1.79%p 높은 수준이다.

외부 평가에서 추론과 예측 기록을 포함한 평가 루프 시간의 합계는 약 63.8분이다. 이 값은 여러 길이의 입력을 batch로 처리한 실행 구간의 합이며, 모델 로딩·입력 인코딩·진단을 모두 포함한 벽시계 시간이나 단일 요청의 응답 지연시간은 아니다.

결과: [외부 분류 전체 결과](/home/khkim/qwen-decision/results/external-classification-v1/summary.json) · [모든 분야와 등급 상세 점수](/home/khkim/qwen-decision/results/external-classification-v1/per-class.csv) · [240,640개 예측 검산](/home/khkim/qwen-decision/results/external-classification-v1/audit.json)

## KMMLU 분야 평가에서 드러난 응답 코드 문제

45개 후보를 한 번의 forward로 비교하기 위해 단일 토큰 코드 `A–Z, a–s`를 사용했다. 모든 코드가 서로 다른 단일 토큰임을 검사했지만, 코드가 계산 가능하다는 사실이 모델이 그 코드를 잘 사용한다는 뜻은 아니다.

학습 모델은 소문자 응답 코드를 한 번도 선택하지 않았다. 선택지 순서를 섞었기 때문에 정답이 소문자 위치에 배정된 14,800문항은 전부 오답이 됐다.

| 정답 코드 구간 | 문항 수 | 원본 적중률 | 학습 모델 적중률 | 학습 모델 선택 비율 |
| --- | --- | --- | --- | --- |
| A-P | 12,436 | 28.07% | 29.56% | 92.00% |
| Q-Z | 7,735 | 6.75% | 8.77% | 8.00% |
| a-s | 14,800 | 1.97% | 0.00% | 0.00% |

학습 모델 응답의 92.00%가 A~P에 몰렸다. 학습 후보가 최대 16개였다는 사실과 일치하는 패턴이지만, 코드와 표시 위치가 함께 바뀌므로 이것만으로 원인을 완전히 분리할 수는 없다. 한국어·세부 분야 변화와 후보 수·코드 변화의 효과도 각각 따로 측정해야 한다.

따라서 KMMLU 분야 12.45%는 현재 45개 후보 프로토콜에서의 실제 점수이며, 응답 코드 편향을 포함한다. 후보 수만 늘어난 것이 유일한 원인이라고 단정하지 않는다.

분석: [정답 코드 구간별 분석](/home/khkim/qwen-decision/results/external-classification-v1/code-position-analysis.json)

## KMMLU 난이도 급락의 해석

난이도는 선택지가 3개였으므로, 45개 분야의 소문자 코드 문제와 구분해야 한다. 확인된 현상은 학습 모델이 문제를 쉬움으로 과도하게 분류했다는 것이다.

| 등급 | 정답 비율 | 원본 선택 비율 | 학습 모델 선택 비율 |
| --- | --- | --- | --- |
| easy | 0.66% | 31.90% | 59.03% |
| middle | 16.76% | 33.91% | 34.83% |
| hard | 82.57% | 34.19% | 6.14% |

원본 모델은 응답 코드 B를 95.01% 선택했다. 코드가 대응하는 등급을 문제별로 섞었기 때문에 정확도 34.12%는 약 1/3을 맞히는 수준이다. 원본이 안정적인 난이도 분류를 수행했다는 근거로 보기 어렵다.

학습 모델은 쉬움을 59.03% 선택했지만, 정답의 쉬움은 0.66%뿐이다. 어려움은 정답의 82.57%인데 모델은 6.14%만 어려움으로 분류했다. 특히 실제 어려움의 적중률이 5.82%로 낮아, 문항 수로 계산한 전체 정확도가 12.29%까지 내려갔다.

세 등급 적중률의 균등 평균은 33.20%에서 34.90%로 비슷했다. 전체 정확도 하락과 모든 등급의 성능 하락을 동일하게 볼 수는 없다. 다만 macro F1도 24.03%에서 13.22%로 하락했고, 현재 모델은 이 사람 정답률 기준에 잘 맞지 않는다.

가장 유력한 설명은 학습과 평가의 난이도 정의가 달랐다는 것이다. AMC 내부의 상대 난이도 894개를 영어로 학습한 모델에, 한국어의 여러 전문시험 문제를 보고 사람 정답률 구간을 예측하도록 요구했다. 기준·대상 집단·분야·언어가 함께 바뀌었다.

KMMLU 논문은 사람 정답률이 낮은 문제를 test에 모았다고 설명하며, 시험마다 응시자 집단이 달라 정답률의 직접 비교에도 주의가 필요하다고 밝힌다. 우리 상·중·하 구간은 공식 난이도 라벨이 아니라 이 수치를 변환한 평가 기준이다. [KMMLU 원 논문](https://arxiv.org/html/2402.11548v2)

기준 불일치가 원인이라는 설명은 현재 관찰에 근거한 추정이다. 언어 변화, 프롬프트 표현, 분야 변화 또는 미세조정 후 능력 변화의 기여도를 분리한 인과 실험을 수행한 것은 아니다. 학습 후 전체 한국어 이해 능력이 손상됐다고 단정할 수 없다.

## 확률과 재현성 검증의 범위

평가 파일에는 데이터 revision, 가중치·코드·입력 해시, 응답 코드의 토큰 ID, 배치 구성과 문항별 logits·확률·정답 여부를 저장했다. 주요 전체 평가에는 원본 라벨·코드 매핑·softmax·argmax·집계를 별도로 대조하는 audit가 있다.

BF16은 batch 모양에 따라 수치가 달라질 수 있다. 외부 평가 중 초기 진단에서 한 문항의 단건·배치 확률 차이가 임의의 5%p 기준을 조금 넘어 중단됐다. 별도 진단의 세 문항에서는 최종 선택이 모두 같았고, 같은 배치를 반복한 결과는 완전히 같았다. 완료된 결과를 보존하고, 차이를 기록하는 후속 실행기로 나머지를 실행했다.

최종 비교에서 두 모델의 입력·선택지·배치를 일치시켰다. 이것은 단건과 모든 batch의 확률이 항상 같다는 검증은 아니다. 후보 수와 입력 길이가 다른 조건의 시간을 단일 요청 지연시간처럼 해석하지 않는다.

| 검산 대상 | 상태 | 검사 예측 수 또는 범위 | 파일 |
| --- | --- | --- | --- |
| MMLU-Pro 점수 반환 | 통과 | 회차별 원본 대조 | [audit](/home/khkim/qwen-decision/results/mmlu-pro-0shot-5runs/audit.json) |
| MMLU-Pro 답 생성 | 통과 | 회차별 원본 대조 | [audit](/home/khkim/qwen-decision/results/mmlu-pro-generate-5runs/audit.json) |
| 새 객관식 데이터 | 통과 | 23,618 | [audit](/home/khkim/qwen-decision/results/unseen-text-v1/audit.json) |
| 전체 보류 분야와 난이도 | 통과 | 8,310 | [audit](/home/khkim/qwen-decision/results/full-classification-v1/audit.json) |
| 외부 분야와 난이도 | 통과 | 240,640 | [audit](/home/khkim/qwen-decision/results/external-classification-v1/audit.json) |

## 다음 학습과 평가에서 바꿀 사항

텍스트 판단 모델을 확장할 때는 다음 항목을 우선 다룬다. 아래 항목은 후속 작업이며 이번 실행에서 새로 학습하거나 평가한 결과가 아니다.

1. **후보 수와 응답 코드 범위를 확장한다.** 최대 16개를 넘어서는 후보 수, 코드 종류와 위치를 다양하게 학습하고, 코드별·위치별 적중률을 검증한다. 많은 후보를 계층적으로 나눠 고르는 방식은 추가 forward가 필요하므로 별도 성능·시간 조건으로 평가한다.
2. **난이도의 대상과 기준을 입력에 넣는다.** `문제 + 대상 집단 + 평가 기준 → 등급` 형태로 구성한다. 일반인·전공자·시험 응시자, 사람 정답률·상대 순위·필요 지식·추론 과정 등을 구분한다.
3. **난이도 자료를 여러 분야와 언어로 늘린다.** 영어 AMC 수학만으로 한국어 전문시험 난이도까지 일반화할 수 있다고 가정하지 않는다. 같은 라벨 이름을 써도 원래 기준이 다른 자료는 출처와 기준을 유지한다.
4. **클래스 불균형을 함께 평가한다.** 정확도와 함께 등급별 recall, balanced accuracy, macro F1, 최빈 클래스 기준을 확인한다. 현재 KMMLU 난이도는 항상 어려움만 고르는 기준이 82.57%다.
5. **개발 데이터에서 확률을 보정한다.** 선택지 softmax를 바로 실제 정답 가능성으로 쓰지 않고, temperature 보정과 신뢰도 검증을 개발 분할에서 수행한다.
6. **독립 평가 분할을 고정한다.** 새 학습 데이터로 사용할 자료는 평가 분할을 먼저 보호하고, 기준·코드·모델 선택을 테스트 점수에 맞춰 조정하지 않는다.
7. **장문과 이미지 기능을 별도로 평가한다.** 긴 입력의 근거 위치·길이·무관 문맥에 따른 정확도와 지연시간을 측정한다. 이미지 판단과 여러 질문의 상태 공유는 텍스트 결과와 구분해 검증한다.

## 현재 모델과 결과 파일 사용

프로젝트는 `/home/khkim/qwen-decision`, 원본 가중치는 `model`, 선택된 학습 모델은 `training/text-decision-full-v1/model`에 있다. 데이터 준비 manifest의 `training_has_run: false`는 준비 당시 기록이며, 실제 학습 완료 상태는 `training/text-decision-full-v1/pipeline.json`에 있다. 초기 scorer의 시스템 지시문은 원본 실험용이므로, 학습 모델을 비교할 때는 학습과 후속 평가에서 사용한 중립 지시문을 명시한다.

```python
import scorer as scorer_module

scorer_module.SYSTEM_PROMPT = (
    "Use the provided state to answer the question. "
    "Select exactly one supplied option. Reply with only its option code."
)
engine = scorer_module.ChoiceScorer(
    "/home/khkim/qwen-decision/training/text-decision-full-v1/model"
)
result = engine.score({
    "state": "I was charged twice. Please refund one charge.",
    "question": "Which team should handle this request?",
    "options": {
        "billing": "Charges and refunds",
        "technical": "App errors",
        "sales": "New purchases"
    }
})
print(result["choice"], result["probabilities"])
```

위 코드는 사용 방법 예시다. 별도 정확도를 측정한 예제로 제시한 것은 아니다. 45개 후보의 재현에는 기본 scorer가 아니라 외부 평가의 확장 프로토콜이 필요하다. 이미 완료된 평가 폴더는 보존하고 새 평가에는 새 결과 경로를 사용한다.

| 목적 | 주요 파일 |
| --- | --- |
| 선택지 점수 반환 | [scorer.py](/home/khkim/qwen-decision/scorer.py) |
| 데이터 준비 | [prepare_text_decisions.py](/home/khkim/qwen-decision/prepare_text_decisions.py) |
| 학습과 모델 선택 | [train_text_decisions.py](/home/khkim/qwen-decision/train_text_decisions.py) |
| 16개 과제 보류 비교 | [evaluate_trained_text_decisions.py](/home/khkim/qwen-decision/evaluate_trained_text_decisions.py) |
| 새 객관식 데이터 평가 | [evaluate_unseen_text.py](/home/khkim/qwen-decision/evaluate_unseen_text.py) |
| 외부 분류 프로토콜 | [external_classification_protocol.py](/home/khkim/qwen-decision/external_classification_protocol.py) |
| 외부 평가 후속 실행 | [continue_external_classification.py](/home/khkim/qwen-decision/continue_external_classification.py) |
| 외부 평가 검산 | [audit_external_classification.py](/home/khkim/qwen-decision/audit_external_classification.py) |
