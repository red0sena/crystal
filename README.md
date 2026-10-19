# Qwen3.5 선택지 점수 실험

Qwen3.5-0.8B에 이미지·텍스트·질문·선택지를 넣고 생성 없이 선택지 점수를 반환하는 실험이다.
원본 모델 실험 이후 text 판단용 v1/v2 미세조정까지 수행했다.

논문 조사와 후속 구조 제안은 [Decision Model 연구 위키](wiki/README.md)에 있다.
선택지 Attention 차단, 후보 내용 점수화, reasoning 증류와 현재 Qwen 구현을 연결해 정리했다.

**2026-10-11 정리:** 모델·원시 데이터·체크포인트·문항별 예측·가상환경은 삭제했다.
코드, lock, 입력 예제, 실험 문서와 작은 설정·결과 기록은 보존했다.
재설치·고정 원본 복구·학습·평가 절차는 [재현 안내](docs/reproduce_ko.md),
파일별 역할은 [코드 색인](docs/code_index_ko.md)에 있다.

```bash
python3 -m qwen_decision.audit.verify_reproduction
bash reproduce.sh --plan
```

위 두 명령은 GPU·가상환경 없이 기록 검증과 실행 계획만 보여 준다.

후속 미세조정과 전체 실험 결과는
[Qwen3.5 선택지 판단 모델 개발과 평가 기록](/home/khkim/qwen-decision/docs/experiment_report_ko.md)에 정리했다.

## v2: 선택지 64개, 질문 유형 3종, 상태 공유

v2는 선택지 2~64개(`A–Z`, `AA–AZ`, `BA–BL`), choice·noul·score 질문, temperature 보정, 한 상태에 대한
여러 질문의 공유 추론을 지원한다. 설계는 [v2 설계](/home/khkim/qwen-decision/docs/v2_plan_ko.md),
결과는 [v2 실험 결과](/home/khkim/qwen-decision/docs/experiment_report_v2_ko.md)에 있다.
아래 이 문서의 나머지 내용은 v1 이전의 원본 모델 scorer(`scorer.py`)에 대한 설명이다.

```bash
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.core.scorer_v2 request.json --model /home/khkim/qwen-decision/training/text-decision-v2/model
```

v2 학습과 평가는 Gated DeltaNet 고속 커널을 사용했다. 새 환경에서는 `uv sync` 뒤에 별도로 설치한다.
설치하지 않아도 같은 코드가 PyTorch 기준 구현으로 실행되지만 장문에서 수 배 느리다.

```bash
uv pip install --python .venv/bin/python flash-linear-attention==0.5.2 fla-core==0.5.2
```

## 실행

아래 명령은 재현 안내에 따라 `.venv`와 `model`을 복구한 뒤 실행한다.
코드는 `qwen_decision/<그룹>/` 패키지에 있으며([폴더 구성](docs/code_index_ko.md#폴더-구성)),
**프로젝트 루트에서** `python -m`으로 실행한다.

```bash
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.core.scorer /home/khkim/qwen-decision/examples/text.json --model /home/khkim/qwen-decision/model
```

이미지 예제:

```bash
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.core.scorer /home/khkim/qwen-decision/examples/image.json --model /home/khkim/qwen-decision/model
```

긴 문서는 `--state-file /절대/경로/document.txt`로 UTF-8 파일을 읽는다.
결과를 저장하려면 `--output /절대/경로/result.json`을 붙인다.

새 환경에서는 이 폴더에서 `uv sync --frozen`으로 의존성을 설치하고,
`python3 -m qwen_decision.sources.restore_sources --groups base`로 기록된 revision의 가중치와 원본 평가 데이터를 받는다.
`--model` 생략 시 Hugging Face ID로 로드한다. CUDA가 없으면 CPU/FP32를 쓴다.
CPU 실행을 강제하려면 `--device cpu`를 사용한다.

## 입력 형식

```json
{
  "state": "문서 내용",
  "images": ["page01.png", "page02.png"],
  "question": "문서와 이미지를 고려하면 어떤 상태인가?",
  "options": {
    "normal": "정상",
    "defect": "결함 있음",
    "unknown": "판단 근거 부족"
  }
}
```

`state`와 `images`는 생략 가능하다. 이미지는 로컬 파일만 지원하며,
상대 경로는 입력 JSON의 디렉터리를 기준으로 읽는다. 선택지는 2~26개이며,
입력 순서대로 A~Z를 배정한다. 이름은 결과를 식별하는 키이고 설명이 프롬프트에 들어간다.
`unknown`은 자동 거절 기능이 아니므로 필요하면 선택지로 직접 넣는다.

## 계산 방식

1. 공식 chat template을 `enable_thinking=False`로 적용한다.
2. 각 답변 문자 A~Z가 실제 응답 경계에서 단일 토큰인지 검사한다.
3. 이미지·텍스트를 처리하는 backbone을 **한 번 forward**한다 (`use_cache=False`).
4. 마지막 입력 위치의 hidden state에 기존 LM Head의 선택지 토큰 행만 곱한다.
5. `softmax(option_logits / temperature)`로 선택지 사이의 분포를 계산한다.

`generate()`를 호출하지 않으며 `output_tokens`는 항상 0이다.
기존 LM Head/embedding 가중치는 그대로 유지하고 계산하는 출력 행만 줄인다.
질문마다 전체 입력을 다시 처리한다. 여러 질문의 공통 상태 재사용은 아직 구현하지 않았다.

`--head-mode full`은 검증용으로 마지막 위치에서 전체 vocabulary logits을
계산한 뒤 선택지 점수를 골라낸다. 기본값은 `selected`이다.

`probabilities`는 **제공된 선택지 사이에서 정규화한 점수**이며, 실제 정답률로
보정된 확률이 아니다. 선택지 내용·순서·프롬프트·temperature에 따라 달라진다.
`--temperature`는 분포의 날카로움을 조절하며 기본값은 1.0이다.

## 입력 길이와 이미지

기본 제한은 이미지 토큰을 포함한 32,768 tokens이다. 제한을 넘으면 오류를
반환하며 입력을 자동으로 잘라내지 않는다. `--max-input-tokens`로 변경할 수
있지만 실제 모델 context 한도도 검사한다.

이미지는 종횡비를 유지하여 기본적으로 약 1,048,576 pixels 이내로 처리한다.
`--max-image-pixels`로 변경할 수 있다. 문서의 작은 글씨를 읽어야 한다면 이
값과 정확도를 함께 확인해야 한다. 입력 길이 지원은 장문 정확도를 보장하지 않는다.

## 검증과 시간 해석

```bash
/home/khkim/qwen-decision/.venv/bin/python -m tests.test_smoke
```

실제 모델로 단일 forward/생성 미사용, 텍스트 분류, 빨강→파랑 이미지 교체,
선택지 행 계산과 전체 LM Head의 일치, 8K 이상 입력 실행,
토큰 제한 초과 시 추론 전 거부를 검증한다. 결과는 `smoke-results.json`에 저장한다.
이미지 예제는 입력 경로와 시각 처리의 smoke test이며 일반 시각 성능 평가가 아니다.

`forward_ms`는 CUDA 동기화를 포함한 forward 시간이다. `total_ms`는
전처리를 포함하고 모델 로딩은 제외한다. warmup/compile을 적용하지 않은 단일
호출 시간이며 d1의 공식 최적화 벤치마크와 직접 비교하지 않는다.
초기 실험에서는 선택적 `causal_conv1d`/`flash-linear-attention` 커널을 설치하지 않아
DeltaNet 기준 PyTorch 구현을 사용했다. v2는 위의 FLA 0.5.2를 별도 설치해 사용했다.

## Python에서 모델을 한 번 로드해 재사용

```python
from scorer import ChoiceScorer

scorer = ChoiceScorer("/home/khkim/qwen-decision/model")
result = scorer.score({
    "state": "한 주문에 두 번 결제되어 환불을 요청함",
    "question": "어느 팀이 담당해야 하나?",
    "options": {"billing": "결제와 환불", "technical": "앱 오류"},
})
print(result["probabilities"])
```

## MMLU-Pro 전체 평가 5회

```bash
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.evaluate.evaluate_mmlu_pro --repeats 5 --batch-size 128 --output /home/khkim/qwen-decision/results/mmlu-pro-0shot-5runs
```

공식 `TIGER-Lab/MMLU-Pro`의 test 12,032문항을 평가한다. 원본 scorer와
같은 프롬프트를 사용하고, 정답·해설·few-shot 예시는 넣지 않는 **0-shot**이다.
생성과 샘플링 없이 선택지 softmax의 argmax를 고른다. 이는 공식 평가 코드의
5-shot CoT 생성 프로토콜과 다르므로 발표 점수와 직접 비교하지 않는다.
공식 전처리대로 정확히 `N/A`인 빈 선택지는 제거하고 정답 인덱스를 확인한다.

효율을 위해 최대 128문항을 독립적인 배치 행으로 처리한다. 문항 길이순으로
고정 배치하며, 오른쪽 padding 이전의 마지막 실제 토큰에서 점수를 읽는다.
BF16은 배치 모양에 따른 반올림 차이가 있어 접전 선택지의 순위가 단일 문항
실행과 달라질 수 있다. `batch-validation.json`에 단일 문항과의 비교를 남긴다.
5회는 같은 프롬프트·선택지 순서·배치 배치를 사용하고 모두 실제 forward를
재실행한다. 입력 토큰은 재사용하지만 이전 예측값은 재사용하지 않는다.

결과 디렉터리에는 `manifest.json`, `run-1.jsonl`부터 `run-5.jsonl`,
각 회차의 summary, 전체 `summary.json`을 저장한다. manifest에는
데이터·가중치·코드의 SHA256, 다운로드 revision과 실행 설정을 기록한다.
`progress.json`과 `/home/khkim/qwen-decision/mmlu-pro-evaluation.log`에서 진행 상황을 볼 수 있다.
완료된 결과가 있는 디렉터리는 덮어쓰지 않으므로 재실행 시 새 `--output`을 지정한다.

완료 후 검산:

```bash
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.audit.audit_mmlu_pro
```

전체 문항의 누락·중복, 정답 인덱스, softmax와 argmax, 정답 수, 과목별 집계,
5회 평균·표준편차·예측 일치 여부를 원본 데이터와 대조한다.
`audit.json`과 `accuracy-by-category.csv`를 결과 디렉터리에 저장한다.

## 원본 모델의 답 생성 평가 5회

```bash
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.evaluate.evaluate_mmlu_pro_generate --repeats 5 --output /home/khkim/qwen-decision/results/mmlu-pro-generate-5runs
```

같은 test 문항, 선택지 순서, 시스템 프롬프트와 입력 토큰을 사용한다.
`enable_thinking=False`로 reasoning을 끄고, 원본 모델의 전체 vocabulary에서
`generate()`로 greedy 답을 생성한다. 선택지 토큰 제한이나 새 학습은 없다.
답 표기는 비교를 위해 A–J를 유지한다. 앞뒤 공백을 제외하고 유효한 선택지
문자 하나인 응답만 인정하며, 설명·복수 답·범위 밖 답은 오답으로 집계한다.
출력 상한은 8토큰이고 정상적인 답의 종료는 tokenizer의 `<|im_end|>`이다.

일반적인 decoder 생성에 맞게 왼쪽 padding을 사용한다. 이전 점수 평가의
오른쪽 padding과 BF16 반올림 차이로 접전 문항의 선택이 달라질 수 있다.
5회 모두 실제 생성하며, raw 응답·생성 토큰 ID·정답 여부를 JSONL에 남긴다.
입력과 reasoning 조건은 같지만 공식 5-shot CoT 점수와는 별도 프로토콜이다.

완료 후 원본 데이터와 생성 토큰을 대조하여 검산한다:

```bash
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.audit.audit_mmlu_pro_generate
```

`audit.json`에는 문항 누락·중복, 응답 파싱, 정답 수, 5회 일치와 이전 점수
방식 대비 차이를 기록한다. `accuracy-by-category.csv`에 과목별 결과를 저장한다.

2026-10-10 전체 실행 결과는 5회 모두 **2,340/12,032 = 19.4481%**다.
형식 오류는 0건이고, 모든 문항이 답 문자 1토큰과 종료 토큰을 생성했다.
5회 원문 토큰과 JSONL SHA256이 동일하며 독립 검산을 통과했다.
이전 선택지 점수 방식은 2,337/12,032 = 19.4232%다. 228문항의 선택이
달랐고 정답 수 차이는 3개(+0.02493%p)다. 바뀐 문항은 이전 top-2 logits
차이가 모두 0.25 이내인 접전이다. 왼쪽/오른쪽 padding과 전체/선택 LM Head의
BF16 계산 차이도 포함하므로 이 작은 차이를 방법의 우열로 해석하지 않는다.

## 다른 객관식 벤치마크: 각 1회

```bash
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.evaluate.evaluate_choice_benchmarks
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.audit.audit_choice_benchmarks
```

2026-10-10, 같은 원본 Qwen·0-shot·reasoning OFF·greedy 답 생성 조건으로
세 벤치마크의 전체 공개 평가 문항을 각각 1회 실행했다. 선택지 내용과 순서를
유지하고 답 표기만 A부터 순서대로 통일한다. ARC의 숫자 정답 표기도 원본
choice label을 조회해서 매핑한다. 질문·선택지만 모델에 전달한다.

| 벤치마크 | split | 정답 / 문항 | 정확도 | 균등 무작위 기대 정확도 |
|---|---|---:|---:|---:|
| ARC-Challenge | test | 668 / 1,172 | 57.00% | 25.02% |
| CommonsenseQA | validation | 518 / 1,221 | 42.42% | 20.00% |
| PIQA | validation | 1,083 / 1,838 | 58.92% | 50.00% |

CommonsenseQA와 PIQA는 공개 정답이 있는 validation을 사용한다.
PIQA는 공식 HF 로더가 지정한 AI2 원본 zip을 다운로드하고 알려진 SHA256을
대조한 뒤 dev 질문과 정답 파일만 읽었다. 외부 데이터셋 로더 코드를 실행하지 않았다.

전 문항이 답 문자와 종료 토큰만 생성했고 형식 오류는 0건이다.
독립 검산에서 원본 정답, 선택지 순서와 매핑, 문항 누락·중복, 생성 토큰 및
점수를 대조했다. 결과는 `results/choice-benchmarks-generate-1run`에 저장했다.
`summary.json`, `audit.json`, `scores.csv`, 벤치마크별 manifest와 raw JSONL을 포함한다.
이는 같은 조건의 직접 답 생성 실험이며, completion likelihood나 few-shot
방식으로 측정한 공개 리더보드 점수와는 평가 방식이 다르다.

## 같은 세 벤치마크를 선택지 확률로 평가

```bash
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.evaluate.evaluate_choice_probabilities
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.audit.audit_choice_probabilities
```

같은 원본 모델과 4,231문항을 생성 없이 각각 1회 평가한다. 프롬프트,
선택지 순서, 왼쪽 padding, 배치 구성을 앞선 생성 실행과 동일하게 유지한다.
마지막 입력 위치에서 유효한 선택지 토큰 logits을 계산하고 temperature=1의
softmax로 정규화한 뒤 argmax를 선택한다. 생성 호출은 예외로 막고, 배치당
정확히 한 번의 backbone forward와 출력 토큰 0개를 확인한다.

| 벤치마크 | 생성 정확도 | 확률 정확도 | 확률 방식 정답 / 문항 | 차이 |
|---|---:|---:|---:|---:|
| ARC-Challenge | 57.00% | 57.42% | 673 / 1,172 | +0.4266%p |
| CommonsenseQA | 42.42% | 42.34% | 517 / 1,221 | −0.0819%p |
| PIQA | 58.92% | 59.09% | 1,086 / 1,838 | +0.1632%p |

2026-10-10 실행 결과다. 각각 20, 15, 53문항의 답이 바뀌었고, 바뀐 문항의
확률 방식 top-2 logits 차이는 모두 0.25 이내였다. 선택 LM Head의 BF16
계산과 cache 설정 차이도 있어 작은 차이를 학습 성능 개선으로 해석하지 않는다.
선택지 사이의 정규화 확률이며 정답률로 보정된 confidence는 아니다.

결과는 `results/choice-benchmarks-probabilities-1run`에 저장한다.
모든 문항의 raw logits와 확률을 JSONL에 남기고, 독립 검산에서 softmax,
argmax, 정답 수, 입력 일치와 생성 결과 대비 차이를 확인했다.
`audit.json`, `comparison.csv`, 벤치마크별 `answer-differences.json`을 포함한다.

## 선택지 위치·답 표기 대조 실험

```bash
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.evaluate.evaluate_choice_bias
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.audit.audit_choice_bias
```

세 벤치마크를 다음 조건마다 전체 1회씩 확률 방식으로 평가한다.

- `shuffle_contents`: 선택지 내용을 문항별로 섞고 화면 순서대로 A–E를 다시 붙인다. 기존 지시문을 유지한다.
- `shuffle_labels`: 내용 순서를 고정하고 문자 표기만 섞는다. A가 첫 위치에 있다는 관계를 분리한다.
- `neutral_letters`: 내용과 A–E 순서를 유지하고 답 안내를 예시 없는 일반적인 label 지시문으로 바꾼다.
- `neutral_shuffle_letters`: 같은 중립 지시문으로 내용은 유지하고 문자 표기만 섞는다.
- `neutral_shuffle_numbers`: 바로 앞 조건과 동일한 표기 배치에서 A→1, B→2 등 숫자로 바꾼다. 같은 중립 지시문을 사용한다.

`seed=42`로 각 문항 ID와 benchmark에서 내용/표기용 독립 permutation을
만든다. 정답을 섞기 위한 seed나 프롬프트에 사용하지 않는다. 원본 선택지
인덱스·화면 위치·화면 label을 모두 저장하고, 정답은 원래 내용에 맞춰 다시 매핑한다.
입력은 0-shot, reasoning OFF이며 선택지 한 토큰의 logits softmax argmax를
사용한다. 왼쪽 padding과 기존 배치 크기를 유지하고 생성은 예외로 막는다.

선택한 답의 원본 내용, 첫 위치 선택률, A 또는 1 선택률, 정확도를 기록한다.
동률일 때 첫 화면 선택지를 고르는 규칙이 위치 통계에 영향을 줄 수 있어
최고 logits 동률인 문항을 제외한 비율도 계산한다. 중립 지시문은 예시 삭제뿐
아니라 `uppercase letter`를 `option label`로 바꾸므로 그 효과를 예시 A 한 부분에만
귀속하지 않는다. 데이터는 `results/choice-bias-controls`에 저장하며,
`audit.json`과 `comparison.csv`에서 검산 및 조건별 지표를 확인할 수 있다.

내용과 표기를 독립적으로 동시에 섞는 추가 대조 실험:

```bash
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.evaluate.evaluate_choice_bias_crossed
/home/khkim/qwen-decision/.venv/bin/python -m qwen_decision.audit.audit_choice_bias --results /home/khkim/qwen-decision/results/choice-bias-crossed --crossed
```

문자와 숫자 조건은 같은 중립 지시문, 같은 내용 permutation, 같은 label
permutation을 사용한다. A→1 등 답 표기의 표현만 달라진다. 전체 7가지 조건,
29,617문항 추론을 실행하고 두 결과 묶음의 정답 매핑·softmax·argmax·통계를 검산했다.

2026-10-10, 내용과 표기를 모두 독립적으로 섞은 조건의 결과:

| 벤치마크 | 문자: 첫 위치 선택 | 문자: A 선택 | 숫자: 첫 위치 선택 | 숫자: 1 선택 | 문자 정확도 | 숫자 정확도 |
|---|---:|---:|---:|---:|---:|---:|
| ARC-Challenge | 19.88% | 29.69% | 20.48% | 36.18% | 53.84% | 51.37% |
| CommonsenseQA | 7.21% | 28.75% | 6.47% | 38.98% | 41.28% | 41.61% |
| PIQA | 69.31% | 48.64% | 47.77% | 67.03% | 64.80% | 63.17% |

A와 1은 첫 위치에 고정되지 않는다. 동률 문항을 제외해도 PIQA 문자 조건의
첫 위치 선택은 66.61%, 숫자 조건의 1 선택은 68.43%다. 따라서 단순한 동률
처리만으로 나타나는 패턴은 아니다. ARC·CommonsenseQA는 표기 선호가 더
두드러지고, PIQA는 답 표기 방식에 따라 위치/표기 선택 패턴이 달라진다.

별도로, 원래 내용·A–E 순서를 유지하고 안내만 중립 지시문으로 바꿨을 때
정확도는 ARC 62.12%, CommonsenseQA 53.40%, PIQA 64.20%였다.
기존 확률 평가의 57.42%, 42.34%, 59.09%보다 높지만, 이 지시문은 예시와
답 안내 표현을 함께 바꿨으므로 예시 삭제 한 부분의 효과로 단정하지 않는다.

데이터: [ARC](https://huggingface.co/datasets/allenai/ai2_arc),
[CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa),
[PIQA](https://huggingface.co/datasets/ybisk/piqa).

참고: [Qwen 공식 모델 카드](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[Transformers Qwen3.5 구현](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
