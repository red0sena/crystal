# 논문 기반 v3 구조 검색 (2026-10-19)

목표는 같은 g4b 가중치·입력에서 기본 구조보다 나은 Decision 모델을 찾는 것이다.
alphaXiv 검색 2회 후 관련 논문 PDF 본문을 확인했다. 아래 구현은 논문의 아이디어를 Qwen3.5에 적용한 변형이며 논문 수치의 재현이라고 주장하지 않는다.

## 문헌에서 확인한 것

| 논문 | 본문 근거 | 이번 실험과의 연결 |
|---|---|---|
| [Going deeper with Image Transformers](https://www.alphaxiv.org/abs/2103.17239), §2–4 | attention·MLP residual branch의 채널별 스케일을 학습하는 LayerScale. 영상 분류에서 깊은 모델 학습을 개선했다. | 이미 학습한 Qwen에는 near-zero 초기값 대신 스케일 1에서 시작한다. 스칼라/채널별 residual 조절을 비교한다. 영상 결과가 언어 판단에 이전된다는 보장은 없다. |
| [Token- and Layer-Selective Probes for Efficient Single-Pass Classification](https://www.alphaxiv.org/abs/2601.13288), §3–4 | 토큰→층의 두 단계 집계. 작은 scoring gate와 downcast attention이 고정 pooling보다 좋은 분류 결과를 보였다. | 각 층의 마지막 토큰·전체 평균·선택지 끝 평균을 결합한다. 기존 코드 head를 유지하고 집계 경로를 residual로 추가한다. 고정 클래스 실험을 가변 후보 판단에 적용하는 변형이다. |
| [Transformer Layers as Painters](https://www.alphaxiv.org/abs/2407.09298), §3 | 중간 층 생략·재배치에 견디는 사례가 있다. 주된 결과는 성능 하락을 줄이는 것으로, 정확도 상승을 보장하지 않는다. | 24층 중 중간 층 하나 생략, 인접 층 순서 교환을 실제 가중치 갱신 없이 조사한다. |
| [Manifold-Constrained Hyper-Connections for Parameter-Efficient Finetuning](https://www.alphaxiv.org/abs/2607.18130), §3–5 | frozen backbone의 residual routing을 학습한다. identity 보존이 도움이 됐으나 downstream에서 일관되게 LoRA를 이기지는 못했다. | pretrained 동작을 보존하는 초기화의 근거로 참고했다. multi-stream mHC 전체를 구현한 것은 아니다. |
| [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://www.alphaxiv.org/abs/2303.08112), §2–3 | 층별 표현 기저 차이로 단순 logit lens가 불안정하다. affine translator를 학습한다. | 중간 층을 그대로 해석하는 것의 한계를 고려한다. 논문의 distillation 목표는 이번 gold-only 학습에 쓰지 않는다. |
| [(IA)³ / T-Few](https://www.alphaxiv.org/abs/2205.05638), §3.3 | attention key/value와 FFN 중간 활성화를 학습 벡터로 조절하고 1로 초기화한다. T0의 소수 예제 분류에서 좋은 결과를 보였다. | softmax attention의 K/V 및 모든 FFN 내부 활성화를 조절하는 추가 후보. DeltaNet recurrence·Qwen key norm은 유지한다. T-Few의 추가 loss나 adapter 사전학습은 재현하지 않는다. |
| [DoLa](https://www.alphaxiv.org/abs/2309.03883), §2 | 최종 층과 중간 층의 logit 차이를 이용하고 JS divergence로 중간 층을 선택한다. | 코드 전체를 유지한 weighted contrast 변형. 원 논문의 vocabulary filtering·생성 decoder는 사용하지 않는다. |
| [Parameter-Efficient Transfer Learning for NLP](https://www.alphaxiv.org/abs/1902.00751), §2 | identity에 가까운 bottleneck residual adapter로 frozen Transformer를 적응시킨다. | 마지막 표현 뒤에 128차원 FFN 하나를 추가하는 변형. 논문의 층마다 배치하는 전체 adapter를 재현한 것은 아니다. |
| [Set Transformer](https://www.alphaxiv.org/abs/1810.00825), §3 | 위치 인코딩 없는 set attention으로 원소 간 관계를 학습한다. | 선택지별 벡터를 작은 두 층 Transformer에서 비교하고 원래 코드 점수에 보정값을 더한다. backbone 벡터가 이미 표시 순서를 담으므로 전체 모델의 순열 불변성을 주장하지 않는다. |

[dQwen3.5](https://www.alphaxiv.org/abs/2609.20751)도 확인했으나 본문은 DeltaNet을 **causal로 유지**한다.
[Hyperloop Transformers](https://www.alphaxiv.org/abs/2604.21254)는 반복 층과 hyper-connection을 큰 규모로 사전학습한 결과다.
둘 다 기존 decision 체크포인트에 역방향 scan이나 반복을 바로 추가하면 정확도가 오른다는 근거로 쓰지 않는다.

## 구현과 비교

`qwen_decision/v3/architecture.py`에 구조, `architecture_search.py`에 단계별 실행을 구현했다.

| 후보 | 학습하는 것 | 기본 모델과 같은 초기 출력 |
|---|---|---|
| depth mix | 층 혼합 가중치·residual gate | 예 |
| token pooling | 마지막 층의 세 summary 혼합·gate | 예 |
| token-layer scoring | 토큰 summary와 층을 고르는 scoring gate | 예 |
| class attention | 마지막 벡터가 층×summary를 읽는 저차원 cross attention | 예(출력 투영 0 초기화) |
| branch scalar | 각 attention/DeltaNet·MLP residual branch의 스케일 | 예 |
| branch channel | 각 branch의 채널별 스케일 | 예 |
| inner activation / IA3 변형 | softmax K/V, FFN 중간 활성화의 채널별 스케일 | 예 |
| residual FFN | 마지막 표현 뒤의 작은 nonlinear adapter | 예 |
| layer contrast | 중간 층 창·weighted contrast 계수, mean/JS 선택 | 계수 0이면 예, 후보는 학습 없이 평가 |
| residual Pointer | 기존 코드 점수에 선택지별 query/key 점수 추가 | 예(query 0 초기화) |
| residual Set Transformer | 문맥·선택지 벡터의 집합 attention과 공유 보정 scorer | 예(보정 scorer 0 초기화) |
| layer surgery | 중간 층 생략·인접 순서 교환 | 아니오, 바로 평가하는 진단 |

branch 스케일은 `1 + 0.2*tanh(g)`이고 g는 0에서 시작한다. 기존 가중치는 동결한다.
IA3 변형은 `1+tanh(g)`로 스케일을 제한하고 g=0에서 시작한다. 1차 readout 검색의 변화가 작아, confirmation을 보기 전에 추가했다.
추가 후보들도 selection 결과만 보며 확장했다. scoring gate가 평균 표현을 빼는 방향을 학습한 것을 보고 DoLa 변형을 조사했고,
Pointer를 대체 head로 쓰던 첫 실험과 달리 원래 점수를 보존하는 residual Pointer/Set Transformer도 비교한다.
pooling·층 결합은 기존 LM 코드 basis를 유지하며, 무작위 새 분류 head에서 학습을 시작하지 않는다.
class attention의 key/value는 층별 세 summary다. 원 논문의 모든 토큰에 attention하는 것과 구분한다.
기본 코드 head를 동일 cached 데이터에서 학습하는 대조군과 원래 동결 모델도 포함한다.

## 데이터·선택 규칙

- 공통 시작점: `runs/v3/g4b-gold3-pad0/best`. 데이터: 기존 `v3-gold3`의 train/dev.
- train 원문 8,192개 중 1,536토큰 제한을 통과한 7,615개 사용. 자르지 않고 제외한다.
- gold3 dev 전체를 원문 ID·item/document 키·같은 전체 state 기준으로 묶고, group의 hash로 selection/confirmation에 배정한다.
- 길이 조건을 통과한 문항에서 출처당 selection 최대 32개, confirmation 최대 64개를 고정 추출한다.
- 실제 selection 1,474문항/47출처/1,269그룹, confirmation 2,956문항/47출처/2,371그룹.
- selection과 confirmation의 학습 노출은 없다. train 원문과의 ID·문서 키·전체 state 중복을 검사했다.
- **g4b 자체는 과거 gold3 dev 전체로 early stopping했다.** confirmation은 이번 구조 검색에 쓰지 않은 분할이며 완전히 새로운 벤치마크라고 부르지 않는다.
- 신규 teacher label·JEVal·공개 시험 문항 학습은 없다. raw mix의 기존 gold-label 방침을 따른다.
- 선택 지표는 출처 평균 정확도. 같은 구조·학습률은 두 seed(41,73)의 selection 평균으로 비교한다.
- readout은 batch 128 × 256 update, branch gate는 batch 16 × 128 update. 계열 간 학습량 차이를 숨기지 않는다.
- branch/IA3와 똑같은 128 update·batch 16·예제 순서로 기존 코드 head를 학습하는 대조군도 별도로 둔다. 최종 비교는 선택된 계열에 맞는 대조군을 사용한다.
- readout 학습률 2개씩과 두 seed를 비교한다. branch gate는 0.003 고정. surgery는 출처당 8개로 먼저 거르고 상위 2개만 전체 selection으로 평가한다.
- confirmation을 보기 전에 최종 구조와 학습률을 `frozen-selection.json`에 고정한다. 각 seed를 모두 보고 가장 잘 나온 seed만 고르지 않는다.
- 기본 모델 및 학습한 기존 코드 head와 문항을 짝지어 차이를 계산한다. 출처 내 문서 단위 paired bootstrap 구간도 기록한다.

## 실행

프로젝트 루트의 기존 `.venv`, g4b 모델 및 gold3 데이터를 사용한다.

```bash
.venv/bin/python -m qwen_decision.v3.architecture_search prepare --output runs/architecture/search-v1
.venv/bin/python -m qwen_decision.v3.architecture_search readouts --output runs/architecture/search-v1
.venv/bin/python -m qwen_decision.v3.architecture_search surgery --output runs/architecture/search-v1
.venv/bin/python -m qwen_decision.v3.architecture_search gates --output runs/architecture/search-v1
.venv/bin/python -m qwen_decision.v3.architecture_search inner --output runs/architecture/search-v1
.venv/bin/python -m qwen_decision.v3.architecture_search controls --output runs/architecture/search-v1
.venv/bin/python -m qwen_decision.v3.architecture_search contrast --output runs/architecture/search-v1
.venv/bin/python -m qwen_decision.v3.architecture_search options --output runs/architecture/search-v1
.venv/bin/python -m qwen_decision.v3.architecture_search confirm --output runs/architecture/search-v1
.venv/bin/python -m qwen_decision.v3.architecture_search external --output runs/architecture/search-v1
```

`prepare`는 새 디렉터리를 요구한다. cached 학습은 train/selection만 읽고 confirmation 점수는 마지막 단계에서만 계산한다.
새 구조의 identity 초기화·단일 실행 summary·padding·checkpoint gradient·원문 분할을
`tests/test_v3_architecture.py`에서 검증한다. 실측 결과는 아래에 있다.

Qwen3.5에서 층을 생략·교환할 때는 `layer_types`와 `num_hidden_layers`도 이동한 실제 layer에 맞춘다.
원래 순서의 mask dispatch를 두면 attention과 DeltaNet에 서로 다른 종류의 mask가 들어가므로, 이 경로도 작은 hybrid 모델로 검증했다.

`external`은 선택한 구조·seed 41을 고정한 후 JEVal v2 restricted의 출처당 최대 40개에서 한 번만 평가한다.
같은 1,536토큰 제한을 적용한 부분집합이며 전체 JEVal/리더보드 점수로 부르지 않는다. 원래 protocol과 데이터 SHA256도 보존한다.
추론은 `architecture_inference.Scorer`에서 선택된 readout·body 변경을 실제 forward에 적용한다. 기존 가중치와 선택된 adapter 파일이 필요하다.

## 결과: 이번 검색에서 확인된 개선 없음

모든 학습·평가를 완료했다. **선택용에서 보인 작은 상승은 confirmation과 JEVal에서 재현되지 않았다.**
현재 모델을 새 구조로 교체할 근거를 얻지 못했다. 이 결론은 이번 동결 가중치·소규모 추가 학습 조건에 한정한다.

설정·seed 조합 수는 99개다: cached readout 26, 같은 gate 예산의 기존 head 대조군 4,
층 생략·교환 30 + 기본 구조 1, residual gate 4, IA3 2, layer contrast 24, residual Pointer/Set Transformer 8.
99개의 서로 다른 아키텍처를 사전학습했다는 뜻이 아니다.

| 후보 | 가장 좋은 학습률의 selection 출처 평균(두 seed 평균) |
|---|---:|
| 기본 구조 | 78.82% |
| 기존 코드 head 추가 학습(readout 예산) | 78.89% |
| 토큰·층 scoring gate | **79.19%** |
| residual Set Transformer | **79.19%** |
| residual Pointer | 79.12% |
| token pooling | 79.02% |
| class attention | 78.95% |
| IA3 내부 활성화 | 78.92% |
| residual FFN | 78.92% |
| branch scalar | 78.92% |
| branch channel | 78.82% |
| 단순 depth mix | 78.82% |

학습 없는 layer contrast 중 가장 높은 값은 79.09%다. 층 생략·교환은 작은 선별 표본에서 양수인 사례가 있었지만,
통과한 두 후보의 전체 selection 결과는 swap(8,9) 77.90%, layer 8 생략 77.57%로 낮았다(인덱스는 0부터).
선별을 통과하지 않은 나머지 층 변경을 전체 selection에서 평가했다고 주장하지 않는다.

동점은 먼저 등록된 후보를 선택하는 정해진 실행 순서로 처리해 scoring gate를 선택했다.
공동 1위인 Set Transformer로 confirmation 결과를 본 뒤 후보를 바꾸지 않았다.

| 고정된 모델 | selection 1,474문항 | confirmation 2,956문항 | JEVal restricted 짧은 문맥 1,645문항 |
|---|---:|---:|---:|
| 기본 구조 | 78.82% | **78.25%** | **58.02%** |
| scoring gate, seed 41 | 79.15% | 78.20% | 57.86% |
| scoring gate, seed 73 | 79.22% | 78.10% | 평가하지 않음(seed 41 사전 고정) |

모든 값은 출처 평균 정확도다. seed 41의 confirmation 차이는 **−0.04%p**, 문서 단위 paired bootstrap 95% 구간은
**[−0.34, +0.27]%p**다(10개를 새로 맞히고 11개를 새로 틀림). seed 73은 −0.14%p, [−0.43, +0.17]%p다.
같은 readout 학습 예산의 코드 head 대조군도 confirmation에서 seed 41은 78.41%, seed 73은 78.25%로,
새 구조가 단순 head 추가 학습보다 낫다는 근거가 없었다.

JEVal의 차이는 **−0.16%p**, 95% 구간은 **[−0.68, +0.39]%p**다(10개를 새로 맞히고 13개를 새로 틀림).
44개 출처를 평가했고 길이 제한으로 343개를 제외했다. 전체 JEVal/보드 결과가 아니다.
chance-corrected macro skill도 기본 38.27 → 후보 38.04로 개선되지 않았다.
이번에 사용한 restricted ID는 이제 결과를 본 평가 문항이므로 후속 구조 선택에 쓰지 않는다.

## 재로딩·실행 검증

21개 테스트가 통과했다. 실제 선택된 모델을 저장 파일에서 다시 구성해 전체 selection 1,474문항을 재실행했고,
원래 저장한 예측과 **불일치 0개**였다. 각 방식 모두 511개 batch 호출에서 backbone forward가 정확히 511번이었다.

같은 128문항에서 모든 batch를 warmup한 뒤 3회 반복한 forward+head 시간의 중앙값이다.
토큰화·입력 전송·모델 로딩은 제외하며 optimizer는 로드하지 않았다.

| 모델 | 질문당 batch 분담 시간 | 최대 torch CUDA 할당 |
|---|---:|---:|
| 기본 | 14.47 ms | 1.817 GB |
| scoring gate | 16.37 ms | 1.819 GB |

이 측정에서도 새 구조를 채택할 이점은 없었다. 모든 중간 층을 함께 materialize하지 않고 층별 summary만 수집한다.

작은 결과 기록은 [architecture-search-results.json](architecture-search-results.json)에 보존했다.
원문 단위 예측, 학습 로그, 체크섬, 선택된 adapter와 재로딩 검증 코드는 `runs/architecture/search-v1/`에 있다.
이 디렉터리는 git에서 제외된다. 해당 실험은 기본 모델 가중치를 덮어쓰지 않았다.
