# 코드 역할과 실행 인자

현재 코드의 주석은 재현 설정과 loss·정답 재배정·cache 분기 등을 설명한다. 기존 코드의 실행 AST는 주석 전과 같다(패키지 이동에 따른 import·경로 변경 제외).
입력 준비 → 검산 → 학습 → 개발 선택 → 평가 → 보정 → 집계의 전체 순서는 [재현 안내](reproduce_ko.md)를 따른다.

## 폴더 구성

원래 프로젝트 루트에 있던 모듈을 단계별 패키지로 옮겼다. 파일 이름은 그대로다.
모든 모듈은 **프로젝트 루트에서** `python -m qwen_decision.<그룹>.<모듈>` 형태로 실행한다
(예: `.venv/bin/python -m qwen_decision.core.scorer_v2 request.json`). 파일 경로로 직접 실행하면 패키지 import가 실패한다.
`data/`, `results/`, `training/`, `model/` 위치는 이전과 같이 프로젝트 루트 기준이다.

| 폴더 | 내용 |
| --- | --- |
| `qwen_decision/core/` | scorer, scorer_v2, decision_format |
| `qwen_decision/sources/` | 원본 다운로드(fetch_*)와 고정 원본 복구(restore_sources) |
| `qwen_decision/prepare/` | 데이터셋 구성(prepare_*)과 원본 조사(inspect_*) |
| `qwen_decision/train/` | v1/v2 학습과 v1 학습→평가 파이프라인 |
| `qwen_decision/evaluate/` | evaluate_*, 외부 분류 프로토콜·후속 실행 |
| `qwen_decision/analysis/` | 코드 분석·배치 진단, temperature 보정, v2 집계 |
| `qwen_decision/audit/` | audit_*와 verify_reproduction |
| `tests/` | test_smoke, test_scorer_v2 |
| `qwen_decision/paths.py` | `ROOT`와 `code_path`: manifest의 코드 해시 키(`scorer.py` 등)를 현재 파일 위치로 연결 |

## 파일별 역할

| 파일 | 역할 | CLI 인자 |
| --- | --- | --- |
| [analyze_external_choice_codes.py](../qwen_decision/analysis/analyze_external_choice_codes.py) | 외부 분야 분류의 코드 구간·위치별 예측 비중/정답률 분석. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [audit_choice_benchmarks.py](../qwen_decision/audit/audit_choice_benchmarks.py) | 기존 평가 결과를 원본 라벨·softmax·argmax·개수·해시와 독립 대조하는 audit. | `--results=ROOT / 'results/choice-benchmarks-generate-1run'` |
| [audit_choice_bias.py](../qwen_decision/audit/audit_choice_bias.py) | 기존 평가 결과를 원본 라벨·softmax·argmax·개수·해시와 독립 대조하는 audit. | `--results=ROOT / 'results/choice-bias-controls'`, `--crossed` |
| [audit_choice_probabilities.py](../qwen_decision/audit/audit_choice_probabilities.py) | 기존 평가 결과를 원본 라벨·softmax·argmax·개수·해시와 독립 대조하는 audit. | `--results=ROOT / 'results/choice-benchmarks-probabilities-1run'`, `--previous=ROOT / 'results/choice-benchmarks-generate-1run'` |
| [audit_external_classification.py](../qwen_decision/audit/audit_external_classification.py) | 기존 평가 결과를 원본 라벨·softmax·argmax·개수·해시와 독립 대조하는 audit. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [audit_full_classification.py](../qwen_decision/audit/audit_full_classification.py) | 기존 평가 결과를 원본 라벨·softmax·argmax·개수·해시와 독립 대조하는 audit. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [audit_mmlu_pro.py](../qwen_decision/audit/audit_mmlu_pro.py) | 기존 평가 결과를 원본 라벨·softmax·argmax·개수·해시와 독립 대조하는 audit. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [audit_mmlu_pro_generate.py](../qwen_decision/audit/audit_mmlu_pro_generate.py) | 기존 평가 결과를 원본 라벨·softmax·argmax·개수·해시와 독립 대조하는 audit. | `--results=ROOT / 'results/mmlu-pro-generate-5runs'`, `--previous=ROOT / 'results/mmlu-pro-0shot-5runs'` |
| [audit_text_baseline.py](../qwen_decision/audit/audit_text_baseline.py) | 기존 평가 결과를 원본 라벨·softmax·argmax·개수·해시와 독립 대조하는 audit. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [audit_text_decisions.py](../qwen_decision/audit/audit_text_decisions.py) | v1 분할·누출·정답 매핑·출처·샘플 가중치 검증. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [audit_text_decisions_v2.py](../qwen_decision/audit/audit_text_decisions_v2.py) | v2 text-light 기준선 생성과 새로운 test 문단 중복 검사. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [audit_unseen_text_evaluation.py](../qwen_decision/audit/audit_unseen_text_evaluation.py) | 기존 평가 결과를 원본 라벨·softmax·argmax·개수·해시와 독립 대조하는 audit. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [calibrate_v2.py](../qwen_decision/analysis/calibrate_v2.py) | 선택용 개발 표본을 제외한 dev logits로 bucket별 temperature fit. | `--predictions`, `--output`, `--min-rows=200` |
| [check_external_code_batching.py](../qwen_decision/analysis/check_external_code_batching.py) | 확장 코드의 scalar/배치 BF16 수치 차이 진단. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [continue_external_classification.py](../qwen_decision/evaluate/continue_external_classification.py) | 초기 외부 평가의 완료 결과를 검증하고 중단 이후 run을 계속 실행. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [decision_format.py](../qwen_decision/core/decision_format.py) | v2 공통 프롬프트·코드·유형·응답 경계 검증·score 요약. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [evaluate_choice_benchmarks.py](../qwen_decision/evaluate/evaluate_choice_benchmarks.py) | 초기 ARC/CSQA/PIQA 답 코드 생성 비교. generate 사용. | `--model=ROOT / 'model'`, `--output=ROOT / 'results/choice-benchmarks-generate-1run'`, `--batch-size=128`, `--batch-token-budget=32768` |
| [evaluate_choice_bias.py](../qwen_decision/evaluate/evaluate_choice_bias.py) | 초기 프롬프트·선택지 순서·표기별 편향 실험. | `--output=ROOT / 'results/choice-bias-controls'` |
| [evaluate_choice_bias_crossed.py](../qwen_decision/evaluate/evaluate_choice_bias_crossed.py) | 선택지 내용 순서와 코드 배정을 독립적으로 교차시키는 실험. | `--output=ROOT / 'results/choice-bias-crossed'` |
| [evaluate_choice_probabilities.py](../qwen_decision/evaluate/evaluate_choice_probabilities.py) | 초기 ARC/CSQA/PIQA 생성 없는 후보 점수 비교. | `--model=ROOT / 'model'`, `--output=ROOT / 'results/choice-benchmarks-probabilities-1run'`, `--previous=ROOT / 'results/choice-benchmarks-generate-1run'` |
| [evaluate_classification_examples.py](../qwen_decision/evaluate/evaluate_classification_examples.py) | v1 분야/난이도 각 native class의 고정 사례 평가. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [evaluate_external_classification.py](../qwen_decision/evaluate/evaluate_external_classification.py) | 초기 외부 전체 평가. 기존 0.05 cutoff 중단 이력에 주의. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [evaluate_full_classification.py](../qwen_decision/evaluate/evaluate_full_classification.py) | v1 분야/난이도 보호 test 전체 비교. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [evaluate_mmlu_pro.py](../qwen_decision/evaluate/evaluate_mmlu_pro.py) | 원본 전체 공식 test의 0-shot 후보 logits 평가. 학습 모델의 공식 점수로 쓰면 안 됨. | `--model=ROOT / 'model'`, `--dataset=ROOT / 'data/mmlu-pro/data/test-00000-of-00001.parquet'`, `--output=ROOT / 'results/mmlu-pro-0shot-5runs'`, `--repeats=5`, `--batch-size=32`, `--batch-token-budget=32768`, `--limit` |
| [evaluate_mmlu_pro_generate.py](../qwen_decision/evaluate/evaluate_mmlu_pro_generate.py) | 초기 비교용 reasoning OFF greedy 답 코드 생성. generate 사용. | `--model=ROOT / 'model'`, `--dataset=ROOT / 'data/mmlu-pro/data/test-00000-of-00001.parquet'`, `--output=ROOT / 'results/mmlu-pro-generate-5runs'`, `--repeats=5`, `--batch-size=128`, `--batch-token-budget=32768`, `--max-new-tokens=8`, `--limit` |
| [evaluate_text_decisions.py](../qwen_decision/evaluate/evaluate_text_decisions.py) | 원본의 v1 과제별 고정 120개 test 표본 평가. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [evaluate_trained_text_decisions.py](../qwen_decision/evaluate/evaluate_trained_text_decisions.py) | 원본/v1 같은 표본의 정답률과 선택지 내용 순서 일관성 비교. | `--model`, `--output` |
| [evaluate_unseen_text.py](../qwen_decision/evaluate/evaluate_unseen_text.py) | 원본/v1 미학습 QA 전체 비교. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [evaluate_v2.py](../qwen_decision/evaluate/evaluate_v2.py) | v2 형식의 test/unseen/external/permutation/robustness/calibration logits 저장. | `--model`, `--output`, `--passes='test,unseen,external,permutation,robustness,calibration'`, `--calibration-per-task=600`, `--limit=0` |
| [external_classification_protocol.py](../qwen_decision/evaluate/external_classification_protocol.py) | 초기 외부 분류 공통 프롬프트와 45개 코드 체계. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [fetch_routing_sources.py](../qwen_decision/sources/fetch_routing_sources.py) | BANKING77 CSV와 NI task 선택. 과거 선택 파일은 보존 manifest로 복구. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [fetch_text_sources.py](../qwen_decision/sources/fetch_text_sources.py) | v1 원본 다운로드. 현재 Hub revision을 선택하므로 과거 재현은 restore_sources 우선. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [fetch_text_sources_v2.py](../qwen_decision/sources/fetch_text_sources_v2.py) | v2 추가 원본 다운로드. 변환 parquet 출처도 기록. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [inspect_external_classification_data.py](../qwen_decision/prepare/inspect_external_classification_data.py) | SuperGPQA/MATH/Omni-MATH native 분야·난이도·출처 조사. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [inspect_kmmlu_classification_data.py](../qwen_decision/prepare/inspect_kmmlu_classification_data.py) | KMMLU 45개 분야·Human Accuracy 결측·분포 조사. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [prepare_external_classification_evaluation.py](../qwen_decision/prepare/prepare_external_classification_evaluation.py) | 분야/난이도 외부 입력 고정, 라벨 숨김, 중복 제외, KM 경계 정의. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [prepare_text_decisions.py](../qwen_decision/prepare/prepare_text_decisions.py) | v1 16개 과제 구성·그룹 분할·중복 제거·AMC train 난이도 경계. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [prepare_text_decisions_v2.py](../qwen_decision/prepare/prepare_text_decisions_v2.py) | v1 보존+v2 새 과제·큰 선택지·STS soft target·텍스트 중복 검사. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [prepare_unseen_text_evaluation.py](../qwen_decision/prepare/prepare_unseen_text_evaluation.py) | HellaSwag/WinoGrande/OpenBookQA 원본 검증·학습 중복 제외. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [report_v2.py](../qwen_decision/analysis/report_v2.py) | 세 모델 집계, paired bootstrap/McNemar, 기준선·보정·견고성 비교. 데이터가 있어야 import 가능. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [restore_sources.py](../qwen_decision/sources/restore_sources.py) | 기록된 원본 URL 스트리밍 복구·SHA 검사. stdlib만 사용. | `--groups='base,v1,v2,unseen,external'`, `--list`, `--verify-only`, `--workers=4` |
| [run_text_training_pipeline.py](../qwen_decision/train/run_text_training_pipeline.py) | v1 학습 후 보호 test 표본 평가를 순차 실행하고 상태 기록. | `--output=ROOT / 'training/text-decision-full-v1'`, `--max-steps=0` |
| [scorer.py](../qwen_decision/core/scorer.py) | 초기 text/image scorer. 한 번 forward, A–Z 후보 LM Head 행, 생성 없음. | `request`, `--model=DEFAULT_MODEL`, `--device='auto'`, `--state-file`, `--temperature=1.0`, `--head-mode='selected'`, `--max-input-tokens=32768`, `--max-image-pixels=1048576`, `--output` |
| [scorer_v2.py](../qwen_decision/core/scorer_v2.py) | v2 text scorer. choice/noul/score, 64개 후보, 보정, prefix cache 공유. | `request`, `--model=str(Path(__file__).resolve().parent / 'training/text-decision-v2/model')`, `--device='cuda'`, `--state-file`, `--no-share-state`, `--uncalibrated`, `--max-input-tokens=32768` |
| [test_scorer_v2.py](../tests/test_scorer_v2.py) | v2 공유/독립 경로의 선택·확률·시간, 유형 혼합·제한 검사. | `--model`, `--output`, `--articles=12`, `--fp32` |
| [test_smoke.py](../tests/test_smoke.py) | 원본 text/image·선택/full head·긴 입력·제한·생성 미사용 smoke. | 고정 경로로 실행; 소스 상단 ROOT/DATA/OUT 참고 |
| [train_text_decisions.py](../qwen_decision/train/train_text_decisions.py) | v1 weighted CE, 두 독립 seed, full language FT, dev 선택. | `--output=DEFAULT_OUT`, `--preflight`, `--max-steps=0`, `--micro-batch=4`, `--effective-batch=32`, `--token-budget=8192`, `--max-tokens=8192`, `--lr=2e-05`, `--long-probability=0.25`, `--eval-every=100`, `--save-every=100`, `--dev-per-task=32` |
| [train_text_decisions_v2.py](../qwen_decision/train/train_text_decisions_v2.py) | v2 soft CE, 선택지 수·코드·escape·injection 증강, dev 선택. | `--output=DEFAULT_OUT`, `--preflight`, `--max-steps=0`, `--micro-batch=4`, `--effective-batch=32`, `--token-budget=12288`, `--max-tokens=12288`, `--lr=2e-05`, `--long-probability=0.25`, `--escape-probability=0.05`, `--injection-probability=0.03`, `--eval-every=200`, `--save-every=200`, `--dev-per-task=32` |
| [verify_reproduction.py](../qwen_decision/audit/verify_reproduction.py) | 보존 hash·코드 AST·원래 분할 hash 오프라인 검증. | `--data` |

## 과거 진단 소스

평균 모델·개별 seed 비교의 일회성 프로그램은 `archive/results/v2-eval/diagnostics`에 보존했다.
고정된 절대 경로가 있으므로 다른 위치에서 복구할 때에는 경로를 확인한다. 실행 절차는 재현 안내에 있다.

## 결과 파일을 읽을 때

`docs/archive`는 과거 기록이다. 현재 결과 output으로 복사해서 완료된 평가처럼 사용하지 않는다.
완전한 문항별 검산은 실제 재평가 뒤 새 예측 JSONL을 대상으로 수행한다.
과거 source SHA는 `archive/code-original`과 비교한다. 현재 새 run의 manifest에는 현재 코드 SHA가 기록된다.
