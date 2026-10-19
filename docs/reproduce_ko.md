# Qwen 선택지 판단 실험 재현 안내

2026-10-11 사용자의 요청으로 실행 산출물을 삭제하고 코드·문서·작은 실험 기록을 보존했다.
이 문서는 원본 모델 점수 실험, v1 학습, v2 학습·평가를 다시 실행하는 절차다.
기존 성능과 해석은 [v1 실험 기록](experiment_report_ko.md), [v2 설계](v2_plan_ko.md),
[v2 결과와 사후 분석](experiment_report_v2_ko.md)에 있다.

## 현재 남아 있는 것

- 프로젝트 Python 소스, 테스트, 입력 예제, `pyproject.toml`, `uv.lock`.
- 핵심 알고리즘을 설명하는 코드 주석. 기존 실행 코드의 AST는 변경하지 않았다.
- `docs/archive`: 다운로드 manifest, 학습 인자·개발 평가·선택 결과, 집계표, 검산 기록,
  작은 로그 및 진단 소스. 개별 예측 JSONL과 학습 가중치·optimizer 상태는 포함하지 않는다.
- `docs/archive/code-original`: 주석 추가 전 소스. 과거 manifest에 기록한 코드 SHA256과 대조할 수 있다.
- `docs/download_catalog.json`: 원본 모델과 원시 데이터 200개 파일의 다운로드 주소, 크기, SHA256.
- `docs/environment.json`, `docs/environment_freeze.txt`: 실제 실행 환경과 설치 패키지 버전.
- `docs/archive_manifest.json`: 보존 파일별 원래 경로, 보존 경로, 크기, SHA256.
- `docs/cleanup_inventory_before.json`, `docs/cleanup_record.json`: 정리 전 파일 목록과 삭제 기록.

삭제 대상은 `model`, `data`, `training`, `results`, `.venv`, `__pycache__`와 보존 완료한 루트 로그다.
기존 `docs/archive`의 모델 폴더는 설정 기록이며 실행 가능한 가중치는 없다.
원본 Qwen은 다시 내려받을 수 있지만 v1/v2 가중치는 아래 절차로 다시 학습해야 한다.

## 환경과 재현 범위

| 항목 | 기존 환경 |
| --- | --- |
| 운영체제·Python | Linux x86_64, Python 3.11.13 |
| GPU | NVIDIA RTX A6000, 48 GiB |
| PyTorch·CUDA | 2.10.0+cu128, CUDA 12.8 |
| Transformers·PyArrow | 5.18.0, 26.0.0 |
| 학습 정밀도 | FP32 master weights/Adam moments, BF16 autocast |
| v1 DeltaNet | PyTorch 기준 구현, 선택적 FLA/causal_conv1d 미설치 |
| v2 DeltaNet | flash-linear-attention 0.5.2, fla-core 0.5.2 |
| 원본 모델 revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| MMLU-Pro revision | `b189ec765aa7ed75c8acfea42df31fdae71f97be` |

데이터는 고정된 원본과 seed로 구성하므로 저장된 JSONL 해시와 정확히 대조할 수 있다.
학습은 GPU 커널·패키지·배치 모양에 따른 수치 차이가 있어 가중치의 비트 단위 일치나
동일한 개발 최고 체크포인트 선택을 보장하지 않는다. 기존 결과를 새 실행의 결과로 복사하지 않는다.
하드웨어나 커널을 바꾸면 속도 비교를 새로 측정한다.

`uv.lock`은 기본 실행 환경을 고정한다. v2 고속 커널은 별도 설치했으므로 lock 바깥의
두 패키지 버전을 아래처럼 명시한다. 전체 실제 패키지 목록은 `environment_freeze.txt`에 있다.
재학습은 여러 시간이 걸리며 체크포인트까지 다시 수십 GiB를 만들 수 있다.

## 실행 순서

프로젝트 루트에서 실행한다. `reproduce.sh`는 선택한 단계만 실행하며 기존 run을 삭제하지 않는다.

```bash
cd /home/khkim/qwen-decision
bash reproduce.sh --plan
python3 -m qwen_decision.audit.verify_reproduction
```

단계별 재현:

```bash
bash reproduce.sh env
bash reproduce.sh sources
bash reproduce.sh data
bash reproduce.sh train-v1
bash reproduce.sh train-v2
bash reproduce.sh evaluate-v2
```

전체 실행은 `bash reproduce.sh all`이다. 이번 용량 정리에서는 이 재학습 명령을 실행하지 않았다.
원본 추론만 필요하면 `env`, `sources`까지만 진행해도 된다.
데이터 준비 단계와 모델 평가·학습 단계를 혼동하지 않도록 단계별 출력을 확인한다.

### 1. 의존성 설치

```bash
uv sync --frozen
```

기본 의존성은 PyTorch CUDA 12.8 index와 lock을 사용한다. 학습에는 CUDA GPU가 필요하다.
원본 scorer는 CPU에서도 실행할 수 있지만 기존 GPU 속도와 비교할 수 없다.
v1을 먼저 실행한 다음 v2 커널을 설치하면 과거 실행 순서와 같다.
`train-v1` 단계는 `uv sync --frozen`을 다시 실행해 별도로 설치된 v2 커널을 제거한다.

### 2. 원본 모델과 데이터 복구

```bash
python3 -m qwen_decision.sources.restore_sources --list
python3 -m qwen_decision.sources.restore_sources
python3 -m qwen_decision.sources.restore_sources --verify-only
```

`restore_sources.py`는 standard library만 사용한다. 기록된 URL을 스트리밍 다운로드하고
크기와 SHA256이 모두 일치한 파일만 최종 경로로 옮긴다. 다른 파일이 이미 있으면 덮어쓰지 않는다.
현재 Hub main을 조회해 새 버전을 선택하지 않는다. 선택 다운로드는
`--groups base,v1`처럼 지정한다. 전체 복구는 약 2.15 GiB다.

| 그룹 | 복구하는 것 |
| --- | --- |
| `base` | 원본 Qwen 파일, MMLU-Pro, 초기 ARC/CSQA/PIQA 파일 |
| `v1` | 16개 text 과제의 원본 파일, BANKING77 CSV, 선택된 Natural Instructions task 파일 |
| `v2` | 추가 intent/topic/emotion/STS-B/review/BoolQ/QuALITY 원본 |
| `unseen` | HellaSwag/WinoGrande/OpenBookQA 원본 |
| `external` | SuperGPQA/KMMLU 및 조사했던 MATH/Omni-MATH 원본 |

일부 converted-parquet URL은 서버의 변환 branch를 가리킨다. 그 경우에도 기록된 SHA256을
검사한다. 원본이 삭제되거나 branch 내용이 바뀌면 복구가 실패하며 새 자료를 조용히 대신 사용하지 않는다.
`fetch_*.py`는 새로운 데이터 버전을 조사할 때 쓰는 다운로드 코드다. 과거 재현에는
최신 revision을 선택할 수 있는 fetch보다 이 복구 스크립트를 먼저 쓴다.

원시 파일과 함께 원본 manifest·source 목록 및 외부 데이터 inspection을 복구한다.
이는 고정한 출처·NI task 선택을 보존하기 위한 것이다. 준비된 train/test manifest나
완료된 평가 manifest는 복구하지 않아 새 평가가 실제 계산을 건너뛰지 않게 한다.

### 3. 학습·평가 데이터 구성

```bash
.venv/bin/python -m qwen_decision.prepare.prepare_text_decisions
.venv/bin/python -m qwen_decision.audit.audit_text_decisions
.venv/bin/python -m qwen_decision.prepare.prepare_text_decisions_v2
.venv/bin/python -m qwen_decision.audit.audit_text_decisions_v2
.venv/bin/python -m qwen_decision.prepare.prepare_unseen_text_evaluation
.venv/bin/python -m qwen_decision.prepare.prepare_external_classification_evaluation
python3 -m qwen_decision.audit.verify_reproduction --data
```

| 데이터 | train | dev | test |
| --- | ---: | ---: | ---: |
| text-decisions-v1 | 63,054 | 14,078 | 30,801 |
| text-decisions-v2 | 108,971 | 20,996 | 53,212 |

v1은 16개 과제, v2 학습에는 26개 과제가 있다. v2 test는 선택지 수별 변형을 포함한다.
v1 분할 seed는 20261010, v2는 20261011이다. 해시 분할을 사용하며 같은 문제에서
파생한 답 선택·분야 분류, 같은 지문, 같은 출처 그룹을 함께 이동시킨다.
v2는 v1 분할에 질문 유형을 추가하고 새 과제를 섞는다.

**MMLU-Pro 공식 test 일부를 v1 학습에 사용했다.** 학습 모델의 MMLU-Pro 평가는
우리가 분리한 보호 test만 써야 한다. 초기 원본 모델의 공식 test 12,032개 점수와
학습 모델의 일부 보호 test 점수를 같은 전체 공식 점수처럼 비교하면 안 된다.

AMC 난이도는 중복 제거 후 train의 IRT rating 분위수로 정한 상대적 난이도다.
KMMLU는 사람 정답률 0.8/0.5 경계를 우리가 정한 것이다. SuperGPQA native difficulty,
AMC 상대 난이도, KMMLU 응시자 정답률을 동일한 기준으로 해석하지 않는다.
외부 보호 평가 문항 수는 SuperGPQA 분야/난이도 각각 25,227개, KMMLU 분야 34,971개,
난이도 34,895개다. 미학습 QA는 HellaSwag 10,042 / WinoGrande 1,267 / OpenBookQA 500개다.
MATH와 Omni-MATH는 출처 조사만 했으며 당시 GPU 성능 평가 결과는 없다.

### 4. v1 학습

```bash
.venv/bin/python -m qwen_decision.train.train_text_decisions --preflight --output training/reproduction-v1-preflight
.venv/bin/python -m qwen_decision.train.run_text_training_pipeline --output training/text-decision-full-v1
```

pipeline은 학습 뒤 16개 과제의 고정 120개 표본 비교를 실행한다. 학습 기본 인자는
보존한 `docs/archive/training/text-decision-full-v1/manifest.json`과 같다.

| 설정 | v1 | v2 |
| --- | --- | --- |
| 시작 가중치 | 원본 Qwen | 원본 Qwen; v1을 이어 학습하지 않음 |
| 학습 범위 | language backbone+tied LM Head 전체, vision 고정 | 동일 |
| 학습 파라미터 | 752,393,024 | 동일 |
| optimizer | AdamW, LR 2e-5, warmup 5%, cosine decay, grad clip 1 | 동일 |
| micro/effective batch | 최대 4 / 32 | 최대 4 / 32 |
| padded token budget·입력 한도 | 8,192 / 8,192 | 12,288 / 12,288 |
| seed·혼합 | 41 natural, 73 sqrt_task_counts | 동일 |
| 각 run update 수 | 1,971 | 3,406 |
| 모델 선택 dev 표본 | 16개 과제×32=512 | 26개 과제×32=832 |
| 개발 평가·저장 간격 | 100 step | 200 step |

두 seed는 각각 원본에서 독립 학습한다. 최종·개발 최고·두 최종 가중치의 50:50 평균을
개발 평균 정확도, 동률이면 개발 loss로 비교한다. 과거 v1은 평균 모델이 선택됐다.

**v1 loss:** 각 입력 마지막 실제 토큰의 hidden state에 후보 코드의 기존 LM Head 행을 곱한다.
후보 logits에 `cross_entropy`를 적용하며 문제별 `-log P(정답)`을 계산한다.
샘플 가중치를 곱하고 effective batch의 가중치 합으로 정규화해 microbatch gradient를 누적한다.
출력 문장·풀이 token loss는 없다. 언어 모델 전체로 역전파하지만 vision은 업데이트하지 않는다.
내용 순서와 코드 배정을 독립적으로 섞고, 9개 이하에서 25% 숫자 코드를 사용한다.
RACE의 25%는 train에서만 가져온 무관한 지문을 섞고 관련 지문의 위치를 바꾼다.
loss에 순서 불변성이나 코드 편향을 직접 벌주는 항은 없다.

### 5. v2 학습

```bash
uv pip install --python .venv/bin/python flash-linear-attention==0.5.2 fla-core==0.5.2
.venv/bin/python -m qwen_decision.train.train_text_decisions_v2 --preflight --output training/reproduction-v2-preflight
.venv/bin/python -m qwen_decision.train.train_text_decisions_v2 --output training/text-decision-v2
```

`decision_format.py`가 학습·개발·평가·추론의 프롬프트를 통일한다. 코드 64개는 A–Z,
AA–AZ, BA–BL이다. 각 코드가 응답 경계에서 단일 토큰인지 실제 tokenizer로 검증한다.
선택지 내용 키는 출력 코드와 독립적이다. 큰 label 집합은 매 draw마다 선택지 수를 표본 추출한다.

질문 유형은 choice(하나 선택), noul(yes/no 점수), score(순서 있는 수준 분포)다.
v2 loss는 `-sum(q_i * log p_i)` soft CE이며 one-hot이면 v1과 같다.
STS-B의 연속 점수는 이웃 수준에 나눈 soft target을 사용한다. score는 별도 회귀 head가 아니다.
padding logits에 유한 큰 음수를 사용해 `0 * -inf`로 NaN이 나는 것을 피한다.

choice/noul 코드 배정은 sequential 60%, shuffled 20%, 전체 64개 중 subset 20%다.
score는 수준 순서를 유지하고 sequential 80%, subset 20%다.
escape 증강 5%, 입력 내 오답 지시 문장 증강 3%, RACE background 증강 25%다.

**보존한 결함:** 기존 escape 증강은 정답을 제거하면 k개, 유지하면 k+1개 선택지를 보여준다.
선택지 수가 고정된 과제에서는 정답 여부를 유추하는 단서가 될 수 있다. 이번 정리는 재현을
위해 동작을 고치지 않았으며, 주석과 결과 문서에 남겼다. 수정 학습은 새 데이터/run 버전으로 한다.

과거 정식 선택 모델은 seed 41의 개발 최고(3,400 step)였다. 두 seed 최종 평균 모델이
test에서 더 좋은 부분은 사후 분석이다. 과거 기록 재현에서는 선택 규칙을 그대로 둔다.
평균 모델을 새 정식 모델로 채택하려면 개발에서 새 규칙을 정당화하고 그 모델로 보정을 다시 맞춘다.

### 6. v2 프로토콜로 원본·v1·v2 비교

```bash
bash reproduce.sh evaluate-v2
```

세 모델에 같은 FLA 커널과 v2 프롬프트·배치 형식을 사용한다. 이 단계는 test, unseen,
external, permutation, robustness, calibration pass의 **새 문항별 logits**를 만든다.
온도는 모델 선택에 사용한 개발 표본을 제외한 dev logits에서 별도로 맞춘다.
`report_v2.py`는 선택된 세 모델을 비교하며 `audit_text_decisions_v2.py`가 만든 기준선도 사용한다.
`test_scorer_v2.py`는 공유 추론과 독립 추론의 선택·확률 차이와 시간을 확인한다.

temperature는 양의 scalar라 argmax를 바꾸지 않는다. 각 모델의 보정값을 별도로 사용하고
다른 가중치 모델에 선택 모델의 calibration을 그대로 붙이지 않는다.
confidence는 분포 형태의 요약이며 보정된 정답률이라는 보장이 없다.

상태 공유는 공통 prefix를 한 번 읽고 attention KV와 DeltaNet recurrent/conv cache를
질문마다 복사한다. 다른 질문을 서로 보지 않는다. 분리한 prefix+suffix의 토큰이 독립
프롬프트의 토큰과 정확히 같을 때만 공유하며, 그렇지 않으면 독립 경로를 쓴다.
출력 토큰 생성은 없지만 prefix와 각 suffix에 대한 계산은 필요하다.

## 초기 원본 모델·v1 평가를 별도로 다시 실행

아래는 v2 형식과 다른 초기 프로토콜이다. `uv sync --frozen`으로 고속 커널 없는 기본
환경을 준비한 상태에서 실행하면 초기 환경에 더 가깝다. 이미 완료한 output을 덮어쓰지 않는다.

```bash
.venv/bin/python -m tests.test_smoke
.venv/bin/python -m qwen_decision.evaluate.evaluate_mmlu_pro --repeats 5 --batch-size 128 --output results/mmlu-pro-0shot-5runs
.venv/bin/python -m qwen_decision.audit.audit_mmlu_pro
.venv/bin/python -m qwen_decision.evaluate.evaluate_mmlu_pro_generate --repeats 5 --output results/mmlu-pro-generate-5runs
.venv/bin/python -m qwen_decision.audit.audit_mmlu_pro_generate
.venv/bin/python -m qwen_decision.evaluate.evaluate_choice_benchmarks
.venv/bin/python -m qwen_decision.audit.audit_choice_benchmarks
.venv/bin/python -m qwen_decision.evaluate.evaluate_choice_probabilities
.venv/bin/python -m qwen_decision.audit.audit_choice_probabilities
.venv/bin/python -m qwen_decision.evaluate.evaluate_choice_bias
.venv/bin/python -m qwen_decision.audit.audit_choice_bias
.venv/bin/python -m qwen_decision.evaluate.evaluate_choice_bias_crossed
.venv/bin/python -m qwen_decision.audit.audit_choice_bias --results results/choice-bias-crossed --crossed
.venv/bin/python -m qwen_decision.evaluate.evaluate_text_decisions
.venv/bin/python -m qwen_decision.audit.audit_text_baseline
.venv/bin/python -m qwen_decision.evaluate.evaluate_unseen_text
.venv/bin/python -m qwen_decision.audit.audit_unseen_text_evaluation
.venv/bin/python -m qwen_decision.evaluate.evaluate_classification_examples
.venv/bin/python -m qwen_decision.evaluate.evaluate_full_classification
.venv/bin/python -m qwen_decision.audit.audit_full_classification
```

초기 MMLU-Pro는 0-shot, thinking off, 원본 선택지 순서이며 공식 5-shot CoT와 다르다.
위 초기 생성 비교 두 코드는 명시적으로 `generate()`를 사용한다. 나머지 점수 경로는 사용하지 않는다.
5회 반복은 deterministic 입력·배치의 재실행이며 독립 확률적 시행이 아니다.
초기 crossed bias와 v2 permutation은 코드 체계·프롬프트가 달라 같은 숫자로 취급하지 않는다.

외부 분야·난이도 v1의 45개 코드는 A–Z+a–s였고, v2는 64개 코드 체계를 사용한다.
초기 외부 전체 평가를 그대로 다시 실행하려면 다음 driver를 쓴다.

```bash
.venv/bin/python -m qwen_decision.evaluate.evaluate_external_classification
```

기존 driver는 scalar/batch BF16 확률 차이 0.05 cutoff 때문에 중단된 이력이 있다.
같은 원인으로 중단했고 `results/external-classification-v1/partial-summary.json`이 만들어졌다면
원본 조건을 보존한 아래 continuation을 실행한다. 다른 실패는 원인을 먼저 확인한다.

```bash
.venv/bin/python -m qwen_decision.evaluate.continue_external_classification
.venv/bin/python -m qwen_decision.audit.audit_external_classification
.venv/bin/python -m qwen_decision.analysis.analyze_external_choice_codes
```

continuation은 완료된 예측을 검증해 재사용하고 나머지만 계산하며 임의 cutoff 대신 실제
scalar/batch 차이를 기록한다. 전체 audit은 원시 라벨·softmax·argmax·배치 매칭을 검산한다.
v1 KMMLU 낮은 점수에는 학습 코드 범위와 선택지 수 문제가 섞여 있으므로 순수 분야 이해력으로
해석하지 않는다. 난이도는 3개 선택지라 이 45개 코드 문제와 따로 분석한다.

## 사후 평균 모델 분석 재현

`docs/archive/results/v2-eval/diagnostics`에 당시 진단 프로그램과 작은 출력표를 보존했다.
v1/v2 학습과 전체 v2 평가를 마친 뒤 실행한다. 이 진단은 체크포인트 선택을 바꾸지 않는다.
프로그램 안의 경로는 당시 `/home/khkim/qwen-decision`이므로 다른 위치에서 복구했다면
진단 소스의 경로를 새 위치로 바꾸고 변경을 기록한다.

```bash
mkdir -p results/v2-eval/diagnostics
.venv/bin/python docs/archive/results/v2-eval/diagnostics/checkpoints_mcq.py results/v2-eval/diagnostics
.venv/bin/python docs/archive/results/v2-eval/diagnostics/averaged_full.py results/v2-eval/diagnostics
.venv/bin/python docs/archive/results/v2-eval/diagnostics/compare_avg.py > results/v2-eval/diagnostics/averaged-vs-selected.txt
```

진단은 재생성된 `final-language.safetensors`, `best-language.safetensors`,
`averaged-language.safetensors`를 사용한다. 삭제된 예측 JSONL을 요약표만으로 복원하지 않는다.

## 코드와 결과 검증

```bash
python3 -m qwen_decision.audit.verify_reproduction
python3 -m qwen_decision.sources.restore_sources --verify-only
python3 -m qwen_decision.audit.verify_reproduction --data
```

첫 명령은 삭제 이후에도 실행되며 작은 보존 파일의 해시와 기존 코드의 AST를 검증한다.
둘째는 복구한 원시 파일 200개, 셋째는 준비된 데이터 13개 파일의 해시를 대조한다.
코드 주석을 추가했으므로 현재 코드 파일 SHA는 과거와 다를 수 있지만 실행 AST는 같다.
루트 모듈을 `qwen_decision/<그룹>/`과 `tests/`로 옮기면서 생긴 import 경로·`ROOT` 깊이·`code_path`·`-m` 실행 변경만
되돌린 뒤 `docs/archive/code-original`과 비교하므로, 그 외의 변경은 모두 실패로 잡힌다.
manifest 전체는 현재 코드 SHA와 실행 경로도 포함해 과거와 달라질 수 있다. 데이터 JSONL
자체의 해시, 라벨 매핑, 분할 개수는 별도로 비교한다.

이번 정리에서는 보존 파일 해시, 기존 원시 파일 200개의 해시, 기존 분할 파일 13개의 해시,
모든 Python 소스 구문, shell 구문과 실행 계획을 검증했다. 원본 다운로드나 재학습·전체 GPU
평가를 다시 수행하지 않았다. 문항별 예측을 다시 독립 검산하려면 실제 재평가가 필요하다.

파일별 역할과 명령행 인자는 [코드 색인](code_index_ko.md)에 있다.
