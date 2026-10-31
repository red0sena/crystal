# 구조 변경 평가 계획

[위키 홈](../README.md) · [설계안](qwen-next.md)

이 페이지는 원래 다음 실험 계획이다. 측정 결과는 아니다.

## 구현된 평가 (2026-10-16, v3)

[작업 기록](../project/v3-log.md)에 결과가 있다. 평가는 Jev Decision Index 제출 기준으로 짰다.

| 평가 | 내용 | 코드 |
|---|---|---|
| 공개 suite 샘플 | 재구성한 0.3 suite에서 벤치마크당 40그룹(2,149행)을 뽑고, 키트의 `run`·채점기로 평가한다. 보드의 d3-lite 결과와 결정 일치율(99.36%)과 점수(39.65 대 39.56)를 확인했다 | `qwen_decision/v3/board_check.py` |
| 비공개 대리셋 | JEVal(58개 출처). 2026-10-17부터는 학습한 lex-glue·CUAD를 빼고 51개 출처를 문서 그룹 단위로 dev·restricted로 나눠 dev(출처당 60, 2,971문항)만 쓴다. same/new로 나눈 매크로 skill. 학습에 쓰지 않는다 | `qwen_decision/v3/proxy_eval.py` |
| 맞대결 | 위 두 평가를 d3-lite와 같은 문항에서 비교한다 | 같음 |
| 오염 검사 | index 전체 행(0.2에서 잘린 ToolRet 질의 포함)과 JEVal에 대한 학습 데이터 필터, 양성 대조 테스트 | `qwen_decision/v3/decontam.py` |

2026-10-17에 추가된 평가:

| 평가 | 내용 | 코드 |
|---|---|---|
| JEVal 프로토콜 v2 | 학습에 쓴 lex-glue·CUAD를 빼고 51개 출처를 문서 그룹 단위로 dev(4,931)와 restricted(4,926)로 나눴다. 설계 결정은 dev로만, restricted는 최종 후보에만 쓴다. 요약에 프로토콜 sha256이 남는다 | `proxy_eval.py --split dev\|restricted` |
| 정답 dev(학습 중 모델 선택) | 학습 셋과 같은 출처, 원본 항목·문서·근사 중복 단위로 떼어 둔 dev. 출처별 정확도 평균으로 best를 고른다 | `mix.py`, `train.py` |
| 학습 표본 진단 | 출처별 고정 학습 25행을 증강 없이 채점. 학습 표본도 틀리면 미학습, dev만 틀리면 일반화 문제 | `train.py --train-probe-per-source` |
| 순서 편향 | choice 문항을 4가지 표시 순서로 채점해 답 뒤집힘 비율, 위치별 재현율 표준편차(RStd), 첫 위치 선택률, 순서 평균 정확도를 따로 낸다 | `bias_eval.py` |
| 문항별 망각 | 같은 조건(noncausal·d3 프롬프트) 원본 대비 맞힘→틀림과 틀림→맞힘을 소스별로 센다 | `forget_report.py` |
| 가중치 비교 | 원본 대비 층별 상대 L2 이동량(d3-lite와 비교) | `weight_drift.py` |
| fresh-v1 (10-18) | 학습·JEVal·index 어디에도 없는 16개 출처 1,335문항(PIQA, QASC, QuaRTz, COPA, CB, WiC, GoEmotions val, ETHICS, SWAG, BigBench 7개). 학습 셋 빌드가 같은 상태 텍스트를 뺀다. d3-lite에게도 새 데이터인지는 보장하지 못한다 | `fresh_eval.py` |
| 선택지 형식 민감도 (10-18) | 같은 문항을 자리표시 키(opt_k: 내용)와 내용 키로 채점해 비교(차이 0.4%p로 원인 아님) | `format_probe.py` |
| dev 크기 (10-18) | 출처당 min(300, 10%), 행 비례 그룹 추출. 짝지은 비교로 출처 5점·매크로 0.7점 감지([검정력](../papers/power-analysis.md)) | `mix.hold_out` |

checkpoint 선택은 공개 점수가 아니라 대리셋과 맞대결로 한다. 아래 원래 계획 중 순서 대칭성은 `bias_eval.py`로 일부 측정했다(코드와 위치를 교차한 분리 측정은 아직 아니다). 후보 수 축은 따로 측정하지 않았다.

2026-10-23 이후 추가된 평가:

| 평가 | 내용 | 코드 |
|---|---|---|
| 장문 테스트 (10-23) | 공개 장문 5종(BABILong, JEVal 장문, QASPER, DocNLI 장문, LongBench v2)을 평가 전용으로 만들었다. 장문 점수는 앞의 4셋 평균이다 | `long_context_experiment.py`, [장문 테스트](../project/long-context-echo.md) |
| 선택지 순서 측정 (10-25) | 학습 없이 4순서 평균. 선택지 격리가 줄 몫의 대략적 추정(앙상블 효과 포함) | [순서 측정](../project/order-probe.md) |
| 원래 Qwen 비교의 판정 (10-26~) | 주 지표는 정답 NLL(selection 4,147문항의 출처 평균, 장문 4셋 평균). seed 5개 중 4개 이상이 같은 방향이고 계층 bootstrap 95% 구간이 0을 넘어야 "나음". 검정은 첫 fit 전에 `plan.json`으로 고정한다 | `training_method_experiment.py` 등, [평가 점검](../project/evaluation-validity.md) |

## 비교할 축

| 축 | 측정 | 의미 |
|---|---|---|
| 정답 품질 | accuracy·macro F1·과제별 지표 | 평균뿐 아니라 과제별 회귀 확인 |
| 순서 대칭성 | canonical ID 기준 flip·JS divergence·probability drift·worst permutation accuracy | softmax confidence만으로 대칭성 판단 금지 |
| 코드 prior | 내용 순서와 코드 매핑을 독립 교차 | 표시 위치와 출력 코드 편향 분리 |
| 후보 수 | K=2,4,16,32,64 및 지원하면 128,255 | 응답 코드 coverage와 semantic 오류 분리 |
| 장문 | 2K,8K,16K,32K 길이별 정확도·latency·VRAM | 모델 입력 지원 길이와 실효 이해를 분리 |
| candidate interaction | 가장 저렴함·상대 비교·None-of-the-above·모두 해당 | 독립 후보 scorer의 한계 확인 |
| ordinal | 등급 recall·MAE·기대 등급 오차·분포·척도 활용 | nominal 분류와 분리 |
| confidence | NLL·Brier·ECE 및 sample 수/오차 | calibration 전후; argmax 불변 검산 |
| 실제 비용 | prefix·candidate encoding·head·전체 latency, peak VRAM | cached/uncached, warm/cold 구분 |

후보 순서를 바꿀 때 정답 ID와 참조 문구도 의미에 맞게 remap한다. 정확히 동률인 argmax는 표시 순서로 tie-break할 수 있으므로 수치 오차·동률 정책을 따로 기록한다.

## 공정한 대조군

- 현재 selected-row LM Head와 단순 scalar candidate scorer, Bi-encoder, 작은 Set Attention.
- Backbone frozen head-only와 Backbone adaptation을 따로 비교.
- CE-only와 동일 teacher targets의 soft-label 또는 설명 loss를 별도 비교.
- 독립 분기·prefix 공유의 토큰화 일치, full independent 입력과의 수치 비교.
- 반복 분류는 candidate cache hit/miss를 나누고, 신규 후보 문제는 인코딩 비용 포함.

## 누출과 shortcut 방지

원본 문제 group 단위 split을 고정한다. 같은 문제의 정답 선택·분야·난이도 파생 요청을 다른 split에 넣지 않는다. 정답 후보 제거 여부나 None-of-the-above 유무가 K·template와 상관되게 만들지 않는다. 벤치마크 학습 이력을 공개하고 official test가 train에 포함된 경우 독립 test로 보고하지 않는다.

전체 벤치마크가 어려우면 먼저 고정 표본으로 비교하되 '전체 benchmark score'로 표기하지 않는다. 특정 구조를 test 결과로 고른 뒤 같은 test를 최종 검증이라고 재사용하지 않는다.

