# v3 Pointer head 소규모 비교 (2026-10-19)

기존 v3 프롬프트와 noncausal backbone을 유지하고 선택지별 벡터를 읽는 것이 도움이 되는지 비교한다.
Sieve 구현을 이식한 결과나 Fork 실험은 아니다. Pointer head는 새로 학습해야 한다.

## 비교 방식

| 항목 | 코드 head 대조군 | Pointer |
|---|---|---|
| 입력 | 기존 v3 프롬프트 | 동일 |
| backbone 실행 | 질문마다 한 번, 캐시 없음 | 동일 |
| 점수 | 마지막 벡터 × checkpoint의 코드별 LM head 행 | 마지막 벡터와 선택지 내용 끝 벡터를 각각 128차원으로 투영한 내적 / √128 |
| 초기값 | 기존 LM head의 255개 행을 FP32로 복사 | 무작위 초기화한 두 투영 행렬 |
| 첫 단계 | backbone 고정, 코드 head 학습 | backbone 고정, Pointer 학습 |
| 두 번째 단계 | 첫 단계 best에서 마지막 N개 층과 head 학습 | 동일 |

같은 seed에서 같은 행·선택지 순서·코드 배정·업데이트 횟수·학습률을 쓴다. 새 head의 초기화와 파라미터 수는 다르다.
코드 head의 입력 임베딩은 고정되므로 head-only 대조군은 tied LM head의 선택 행을 독립 readout으로 복사한다.
첫 단계에서는 원래 checkpoint의 코드 점수도 step 0에 보존한다. 두 번째 단계에서 head 학습률과 backbone 학습률을 분리한다.

softmax attention은 SDPA noncausal로 유지하고 DeltaNet 설정은 checkpoint에서 읽는다.
첫 실험에서는 packing과 compile을 추가하지 않고 오른쪽 패딩을 사용한다. attention 구조·토큰·특수 마커를 바꾸지 않는다.
선택지 벡터는 fast tokenizer의 문자 offset으로 찾으므로 반복 문구, 다국어, 여러 줄, echo 레이아웃에서도 마지막 선택지 목록과 대응한다.
Pointer head 자체는 벡터 순열에 따라 같은 순열로 점수를 내지만, backbone의 프롬프트 순서 의존은 남아 있다.

## 실행

프로젝트 루트에서 가상환경·기존 파일을 복구한다. 비공개 저장소 다운로드에는 로컬 CLI 로그인이 필요하다.

```bash
uv sync --frozen
uv pip install --python .venv/bin/python flash-linear-attention==0.5.2 fla-core==0.5.2
.venv/bin/hf auth login
.venv/bin/hf download Kyunghwan-kim/kkh-decision-model \
  --revision a89ccb1b43913b985432f79cc16c052440f81c12 \
  --include 'runs/v3/g4b-gold3-pad0/best/*' --local-dir .
.venv/bin/hf download Kyunghwan-kim/kkh-decision-data --type dataset \
  --revision c290f94806ddf44abad0c88d4e8809b7ac61cca9 \
  --include 'data/mix/v3-gold3.jsonl' --include 'data/mix/v3-gold3.dev.jsonl' \
  --include 'data/mix/v3-gold3.manifest.json' --local-dir .

.venv/bin/python -m qwen_decision.v3.pointer_experiment \
  --base runs/v3/g4b-gold3-pad0/best --mix data/mix/v3-gold3.jsonl \
  --output runs/pointer/g4b-gold3-head100-tune20-s41 \
  --train-rows 2048 --dev-per-source 8 \
  --head-steps 100 --tune-steps 20 --unfreeze-last-layers 2 \
  --effective-batch 16 --micro-batch 4 --token-budget 4096 \
  --max-tokens 1536 --eval-every 50
```

`--tune-steps 0`이면 첫 번째 head-only 학습만 실행한다.
새 실행에는 새 output 디렉터리를 쓴다. 기존 실험은 덮어쓰지 않는다.

## 평가·출력

- 기존 grouped gold dev에서 출처당 고정 표본을 뽑고 출처 평균 정확도로 best를 고른다. step 0도 선택 가능하다.
- 원래 순서 정확도와 출처별 정확도, 역순 정확도·동일 선택 비율을 기록한다. 순서를 뜻에 사용하는 문항과 ordered 문항은 역순 평가에서 제외한다.
- `manifest.json`: 데이터·모델·실행 코드 SHA256, 실제 사용 ID, 길이 제한으로 빠진 ID, 실행 설정.
- `summary.json`: 두 head의 초기·선택된 결과와 차이. `code|pointer/frozen|tuned/` 아래에 학습 로그, 단계별 예측, best head를 저장한다.
- 추가 학습을 했으면 `language.safetensors`에 선택된 backbone을 저장한다. `load_run(Path('runs/pointer/<run>/pointer'))`로 해당 head와 backbone을 다시 읽는다. 원본 checkpoint도 필요하다.
- 지연시간은 warm batch의 backbone+head 시간을 질문 수로 나눈 값이다. 토큰화·전송·모델 로딩은 포함하지 않는다.
- `forward_peak_extra_gb`는 상주 메모리를 뺀 forward 추가 할당량이다. `peak_cuda_allocated_gb`는 상주 optimizer도 포함하므로 독립 추론의 메모리 사용량으로 해석하지 않는다.

**dev는 모델 선택에도 사용하므로 정확도 개선을 확정하는 독립 평가가 아니다.** 출처당 8개인 첫 실험으로 1%p 차이나 특정 출처의 우열을 판단하지 않는다.
독립 gold 평가셋이 있으면 `--test <mix-format.jsonl>`로 지정한다. train/dev와 겹치지 않는지 검사하고 선택이 끝난 모델만 평가한다.
JEVal·fresh·공개 benchmark 문항을 학습 데이터로 쓰지 않는다.

원래 mix의 corpus 전체 기준 분할을 재사용한다. 실험 표본에서는 원문 ID, item/document 키, 공백을 정규화한 전체 state 중복을 검사한다.
state가 비어 있으면 전체 question을 검사한다. 작은 표본에서 `mix.data_groups`를 다시 실행하면 corpus 빈도 기반 boilerplate 필터가 달라져
짧은 분류 입력의 공통 질문 양식까지 같은 문서로 묶으므로, 전체 corpus의 near-duplicate 감사를 대체한다고 주장하지 않는다.
gold3 dev의 MNLI에는 같은 원문 ID의 choice/noul 변형 한 쌍이 있어 내부 평가 ID에 prompt hash를 붙인다. split 검사는 원문 ID도 그대로 확인한다.
길이 제한을 넘는 문항은 자르지 않고 양쪽 모두 제외하고 기록한다. 실제 augmentation이 제한을 넘으면 실행을 중단한다.

## 검증 상태

`.venv/bin/python -m unittest tests.test_v3_pointer -v`로 다음을 확인했다.
실제 Qwen tokenizer의 선택지·gold·코드 매핑, 기존 프롬프트 보존, 가변 선택지 패딩과 유한 손실,
단일 forward, head 동결 학습 시 backbone 불변, 작은 무작위 Qwen의 GPU 추가 학습·선택·저장 경로.
이 테스트의 무작위 모델 결과는 정확도 실험 결과가 아니다.

## 실제 g4b 결과

2026-10-19, 로컬 GPU에서 위 명령으로 완료했다. 길이 제한을 적용한 학습 풀은 1,926행,
dev는 47개 출처의 361행이다. 각 방식은 head-only 100 update(1,600행 draw),
마지막 두 층 추가 학습 20 update(320행 draw)를 수행했다. 같은 행을 반복해서 뽑을 수 있으므로 draw 수는 고유 학습 문항 수와 다르다.
양쪽 head 파라미터 수도 거의 같다(코드 261,120개, Pointer 262,144개).

| 방식·단계 | dev 출처 평균 정확도 | 전체 문항 정확도 | 역순에서도 같은 선택 | 선택된 step |
|---|---:|---:|---:|---:|
| 기존 코드 head / head-only 선택 | 82.18% | 81.44% | 94.17% | 0 |
| Pointer / head-only 선택 | 77.13% | 76.18% | 83.97% | 100 |
| 코드 head + 마지막 두 층 / 선택 | 82.18% | 81.44% | 94.17% | 0 |
| Pointer + 마지막 두 층 / 선택 | 77.13% | 76.18% | 83.97% | 0 |

역순 일치율은 순서를 바꿀 수 있는 343쌍에서 측정했다.
Pointer의 초기 출처 평균은 47.27%, 50 update는 76.60%, 100 update는 77.13%였다.
추가 학습의 마지막 step만 보면 코드 74.57%, Pointer 74.20%로 낮아졌으므로 두 방식 모두 추가 학습 전 상태가 선택됐다.
코드 head-only도 50/100 update가 각각 79.10%/78.83%로 낮아져 원래 checkpoint가 선택됐다.

이 조건에서 Pointer는 기존 head보다 출처 평균 **5.05%p 낮았다**. 새 head가 선택지 벡터에서 판단을 학습할 수 있다는 것은 확인했지만,
정확도나 순서 강건성이 개선된 증거는 얻지 못했다. 기존 코드 head는 이미 학습된 상태이고 Pointer는 새 초기값이라는 차이가 있다.
이 작은 학습 예산·단일 seed·고정 학습률 결과로 충분히 학습한 Pointer 전체가 열등하다고 단정하지 않는다.

원본 결과는 `runs/pointer/g4b-gold3-head100-tune20-s41/summary.json`,
설정·체크섬·실제 사용 ID는 같은 디렉터리의 `manifest.json`에 있다. 해당 경로는 git에서 제외된다.

선택된 모델을 optimizer 없이 다시 로드한 추론 비용도 측정했다.

| 선택된 모델 | 질문당 batch 분담 forward 시간 | 최대 CUDA 할당 | forward 추가 할당 |
|---|---:|---:|---:|
| 코드 | 14.22 ms | 1.832 GB | 0.293 GB |
| Pointer | 14.09 ms | 1.833 GB | 0.293 GB |

같은 padded batch 조건의 단일 측정이며 이 정도 차이로 속도 우열을 주장하지 않는다.
두 방식 모두 warmup 1회와 평가 batch 178개에서 **batch당 backbone forward 1회**를 확인했다.
저장 전과 같은 batch 편성으로 다시 실행했을 때 확률 차이는 0, 예측은 모두 일치했다.
코드 대조군의 점수는 기존 `modeling.code_logits`와도 차이가 0이었다.
batch 편성을 바꾼 첫 재실행에서는 최대 확률 차이 0.00672가 관찰되어, 저장 재현 검증에는 원래 batch 순서를 사용했다.
세부 값은 `runtime-reloaded.json`, 검증 스크립트는 같은 run의 `verify_runtime.py`에 보존한다.
