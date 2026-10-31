# 구조 검색 이후의 다른 방법 조사 (2026-10-19)

[위키 홈](../README.md) · [이전 구조 검색](architecture-search.md)

alphaXiv 검색 2회와 14편의 PDF 본문으로 입력 표현, 학습 목표, 추가 추론 계산을 조사했다.
논문에서 좋아진 조건과 우리 Qwen3.5-0.8B에서 확인한 결과를 구분한다.
새 교사 label·rationale는 사용하지 않는다.

## 입력과 추론에서 시도할 수 있는 것

| 방법·논문 | 본문에서 확인한 근거 | 우리 모델에 적용할 때의 조건 |
|---|---|---|
| [Prompt Repetition](https://www.alphaxiv.org/abs/2512.14982), p.1–2 | reasoning을 끈 일반 LLM 비교 70건 중 유의한 승리 47건, 패배 0건(p<0.1). 입력을 두 번 제시한다. | causal DeltaNet에는 두 번째 읽기가 도움이 될 가능성이 있다. full attention은 이미 양방향이다. 토큰 수와 GPU 시간이 늘 수 있으며 논문의 지연시간 결과를 그대로 가져오지 않는다. |
| [Labels Override Definitions](https://www.alphaxiv.org/abs/2610.02586), §5.5 | LAYA-TD의 synthetic PolicyBench에서 이름 제거/익명화가 유리했다. 반면 일반 classification에서 full 84.9% 대비 이름 제거 80.7%로 내려갔다. | 정의가 충분한 경우에만 이름 제거가 의미를 보존한다. 정의가 없으면 이름을 남긴다. 우리 `format.options()`도 이름과 정의를 합치므로 직접 비교할 수 있다. |
| [Output Prefilling](https://www.alphaxiv.org/abs/2505.15323), §4, Table 1 | 답변 시작에 자연어 prefix를 넣고 첫 토큰을 읽는다. Qwen2-7B의 14개 MCQA 평균 개선은 +2.4%p였다. | 이미 코드 답변으로 학습한 g4b와는 조건이 다르다. prefix와 답 코드가 BPE에서 합쳐지지 않는지 검사해야 한다. |
| [Contextual Calibration](https://www.alphaxiv.org/abs/2102.09690), §5 | GPT-2/3 few-shot에서 N/A, [MASK], 빈 입력의 평균 예측으로 class prior를 추정해 나눈다. | 기존 코드 학습이 편향을 이미 줄였을 수 있다. 같은 question/options면 prior는 재사용 가능하지만 새 schema에는 추가 forward가 필요하다. |
| [Consider, Then Choose / Runoff](https://www.alphaxiv.org/abs/2610.18233), p.1–3,6,8 | 각 후보를 yes/no로 독립 평가해 줄인 뒤 순열을 빠짐없이 비교한다. 20선택지에서 cyclic 평균과 비슷한 정확도, 3–4배 속도. 원래 single-pass보다 빠르다는 뜻은 아니다. | 순서 불변성이 주된 이점이다. 우리 순서 편향은 이미 작고 후보별 재평가 비용이 생긴다. 후보 간 상대 비교나 all/none 같은 선택지는 별도 취급이 필요하다. 이번에는 구현하지 않았다. |

질문·선택지를 지문 앞과 뒤에 두는 `echo`도 비교한다. 기존 구현에 있었으나 이전 구조 검색에는 포함하지 않았다.

## 학습 자체를 바꾸는 후보

**당시(10-19) 우선 후보는 기존 gold 쌍을 함께 학습하는 목표였다.** 현재 gold5/6은 VitaminC의 한 case와
CONDAQA의 원문·편집본을 같은 item으로 보존하지만, 일반 CE는 각 행을 따로 처리한다.
다음은 검증된 우리 모델 개선이 아니라, 추가 비교가 필요한 설계다.

| 방법·논문 | 실제 근거 | 적용 및 대조 실험 |
|---|---|---|
| [PACT](https://www.alphaxiv.org/abs/2609.35865), §IV–VI | 반사실 쌍의 log-odds 변화 margin, 코드 순열 JSD, 증거 필요성, ordinal loss. Qwen3.5-9B, frozen holdout 324개/3seed에서 기존 recipe 정확도 85.2% 대비 84.6%로 유의한 개선은 없었다. position flip 13.8→9.8%, ordinal MAE .311→.232는 개선됐다. | label이 바뀐 실제 쌍을 gold에서 찾아 CE+pair margin을 CE와 비교한다. 문항별 class 의미를 정렬해야 하며 다른 질문/choice·noul을 무조건 묶지 않는다. 증거 삭제 후 unknown이라는 인증이 없는 우리 자료에 evidence-necessity 항을 그대로 쓰면 안 된다. |
| [R-Drop](https://www.alphaxiv.org/abs/2106.14448), §2–4 | 동일 정답 문항을 dropout 두 view로 학습하고 symmetric KL을 추가한다. RoBERTa 등 NLP fine-tuning과 translation에서 개선, Wikitext-103 Transformer test perplexity 26.62→24.94. | 추론은 그대로다. 현재 Qwen attention_dropout=0.0이므로 forward 두 번만 실행하면 의도한 noise가 생기지 않는다. dropout을 실제로 켠 CE 대조군과 CE+KL을 같은 두-view 계산량으로 비교해야 한다. 새로운 teacher는 없다. |
| [Hard Samples, Bad Labels](https://www.alphaxiv.org/abs/2511.16512), Tables 1–3 | 어려운 샘플을 무조건 강조하지 않고 낮은 weight/zero gradient로 처리한다. 주된 지표는 인위적으로 오염한 MNIST/CIFAR의 label-error 탐지 F1이다. | 정확도가 낮다는 이유만으로 hard replay/focal loss를 우선할 근거로 쓰지 않는다. 진짜 어려운 금융·개체 문항과 오답 label을 먼저 구분해야 한다. 우리 언어 판단 정확도 상승은 미검증이다. |

pair 목표를 비교한다면 같은 새 gold 데이터·draw 순서·backbone 시작점에서 **CE / CE+pair margin /
CE+view consistency**를 두 seed로 비교하는 것이 다음 실험이다. 이전 연구에서 작은 readout을 바꾼
것과 달리 attention·FFN까지 학습시키는 adapter 또는 full fine-tuning을 공통으로 사용해야 한다.
추론 코드만 바꾸면 학습 목표의 효과를 검증할 수 없다.

> **10-31 갱신:** 셋 다 했다. 순열 JSD는 [조합 실험](robust-combinations.md)(10-19), 쌍 margin은 [증거 쌍 손실](evidence-pairs-results.json)(10-22), 원래 Qwen에서의 순서 일관성(consist)은 [Loss 비교](loss-screen.md)(10-31)에서 쟀다. 모두 CE를 확실히 넘지 못했다.

## 실제 구조를 더 크게 바꾸는 방향

- [Recurrent Depth / Huginn](https://www.alphaxiv.org/abs/2502.05171), Table 4: 중간 block을 latent space에서 반복한다. 같은 180B 학습 토큰의 fixed-depth baseline 대비 ARC-C 26.96→29.18%, HellaSwag 37.34→48.80%였다. 반복 수 32의 추가 계산을 사용한다. 주 모델은 3.5B/800B 토큰 사전학습으로, 기존 0.8B에 층을 복사하면 같은 효과가 난다는 증거는 아니다.
- [Universal Transformers](https://www.alphaxiv.org/abs/1807.03819): depth 방향으로 가중치를 공유하고 halting을 학습한다. algorithmic/generalization 과제와 translation에서 개선했다. 이 역시 반복 구조를 학습한 결과다.
- [Pause Tokens](https://www.alphaxiv.org/abs/2310.02226), §4–5: 1B에서 SQuAD +18 EM, CommonSenseQA +8%p를 보고했다. 핵심 성공 조건은 pause를 넣은 사전학습과 미세조정이다. 일반 pretrained 모델에 마침표만 추가한 ablation은 개선되지 않았다. 현재 checkpoint에 빈 토큰만 추가하는 방법은 우선하지 않는다.

반복 깊이는 head 교체와 달리 실제 계산 능력을 늘리는 방향이지만 지연시간과 적응 학습 예산이 필요하다.
이번에는 세 방법을 우리 가중치에서 학습·재현하지 않았다.

## 근거를 과대해석하지 않은 나머지 논문

- [Can a Cacheable Decision Model Follow Rules?](https://www.alphaxiv.org/abs/2609.37832): joint encoding이 cacheable dual-encoder보다 강한 조건이 있었다. rule-aware synthetic 학습의 큰 상승은 실제 prose로 일관되게 이전되지 않았다. real-prose 추가 학습의 contracts fit 정확도는 .870→.777이었다. 후보 캐시 전환을 정확도 개선안으로 우선하지 않는다.
- [Training Language Models To Be Coherent Decision-Makers](https://www.alphaxiv.org/abs/2610.09164): 비용/효용이 주어진 binary 결정을 학습한다. belief 보존과 missing-information routing이라는 별도 목표다. 일반 MCQA 정확도 상승으로 바꿔 읽지 않는다.
- [Verdicts Without Annotated Evidence](https://www.alphaxiv.org/abs/2610.06962), §4: ContractNLI 9B에서 label-only .896–.899, rejection-sampled rationale SFT .797. 단일 seed·행 coverage·parse-failure 차이가 있다. 우리 gold-only 정책을 유지할 이유는 되지만, 이미 gold-only인 모델에 새 개선을 주는 기법은 아니다.

## 이번 입력 실험의 사전 규칙과 실행

`qwen_decision/v3/input_experiment.py`에 training-free 실험을 추가했다. 기본 serving 입력/가중치는 변경하지 않는다.

- 동일 g4b checkpoint, 기존 architecture selection 1,474문항/47출처를 사용한다. baseline 토큰과 target이 기존 저장값과 완전히 같은지 확인한다.
- 학습 없이 정의만 사용, echo, 입력 2회 반복, 답변 prefix, contextual calibration 세 강도(.25/.5/1)를 비교한다. 대조군 외 후보는 7개다.
- 모든 후보를 **같은 문항**에서 평가한다. 길이를 자르거나 후보별로 행을 빼지 않는다. 최대 4,096토큰이며 원래 selection의 최대 길이는 1,524였다.
- calibration은 내용 없는 state 세 가지의 **확률 평균**을 prior로 삼고 `logits - alpha*log(prior)`를 계산한다. alpha=1은 논문의 classification decision rule과 같은 argmax다. temperature scaling과 달리 argmax를 바꿀 수 있다.
- selection 출처 평균 +0.5%p 이상인 후보만 고정한 뒤 confirmation한다는 기준을 실행 전에 `plan.json`에 쓴다.
- 후속 confirmation이 필요하면 이전 architecture selection/confirmation과 Pointer dev에 속한 **문서 그룹 전체**를 제외하고 gold3-dev에서 고정 추출한다. 예전 confirmation·JEVal restricted는 재선택에 쓰지 않는다.
- g4b 자체는 과거 gold3 dev 전체로 early stopping한 모델이다. 새 confirmation도 완전히 새로운 benchmark라고 부르지 않는다.

```bash
.venv/bin/python -m qwen_decision.v3.input_experiment screen --output runs/inputs/search-v2
# 사전 기준을 통과한 후보가 있을 때만:
.venv/bin/python -m qwen_decision.v3.input_experiment confirm --output runs/inputs/search-v2
.venv/bin/python -m unittest tests.test_v3_inputs -v
```

첫 `search-v1` 실행은 prefill의 `:`와 답 코드가 합쳐지는 토큰 경계 검사에서 중단됐다.
`search-v2`는 prefix 끝에 줄바꿈을 추가하고 **모든 후보의 tokenization을 inference 전에 검사**한다.
이 변경은 평가 결과가 아닌 입력 유효성 오류를 수정한 것이며 실패 run도 보존한다.

## 결과

실행 결과는 `runs/inputs/search-v2/selection-summary.json`, 예측과 prior는 같은 디렉터리의 JSONL에 남긴다.

| 방법 | selection 출처 평균 정확도 | 기본 대비 |
|---|---:|---:|
| 기본 | 78.82% | — |
| 선택지 정의만 사용 | 78.89% | +0.07%p |
| 질문·선택지 앞뒤 배치(echo) | 76.30% | −2.52%p |
| 입력 2회 반복 | **79.35%** | **+0.53%p** |
| 답변 prefix | 78.95% | +0.13%p |
| prior 보정 α=.25 | 78.43% | −0.39%p |
| prior 보정 α=.5 | 78.11% | −0.72%p |
| prior 보정 α=1 | 74.13% | −4.69%p |

`frozen-candidate.json`에서 repeat를 고정했다. 선택용 상승만으로 채택하지 않고 별도 문서에서 확인한다.
입력 token 수 합은 기본 528,693, 반복 983,686(1.86배)이었다. 고정 system/chat 부분은 반복하지 않는다.
단 한 번의 selection 처리 시간은 기본 22.01초, 반복 34.41초였다. 이는 CPU collate·GPU forward·logit 복사를
포함하며 토큰화와 로딩은 제외한다. 반복 측정한 production latency로 부르지 않는다.

token boundary·동일 정의의 별도 선택지 유지·target 정렬·content-free 입력·prior 확률 평균을 포함한 6개 테스트가 통과했다.

### 별도 문서에서 확인: 입력 반복도 개선되지 않음

이전 실험의 selection/confirmation 및 Pointer dev를 제외한 2,573문항, 45출처, 2,263문서 그룹을 사용했다.
원문 ID·전체 state·item·document key를 다시 검사해 과거 노출 문항들과 공유 key가 **0개**임을 확인했다.
이는 이번 입력 검색과의 분리다. g4b의 과거 early stopping에 gold3-dev가 쓰였다는 한계는 남는다.

| 고정한 방법 | selection 1,474문항/47출처 | confirmation 2,573문항/45출처 |
|---|---:|---:|
| 기본 | 78.82% | **79.81%** |
| 입력 2회 반복 | **79.35%** | 78.92% |

confirmation 차이는 **−0.89%p**, 출처 내 문서 단위 paired bootstrap 2,000회의 95% 구간은
**[−1.91, +0.15]%p**였다. 새로 맞힌 74개보다 새로 틀린 92개가 많았다.
구간이 0을 포함하므로 통계적으로 악화가 확정됐다고도 주장하지 않는다. **개선 근거가 없으므로 채택하지 않는다.**
선택/확인셋은 출처 구성도 다르므로 두 열 간 절대 점수 이동을 학습 효과로 해석하지 않는다.

이번 7개 후보 모두 우리 모델의 확인된 개선으로 남지 않았다. 단순 inference 변경의 실패가
같은 아이디어를 충분히 적응 학습한 모델에서도 실패한다는 뜻은 아니다. 기존 모델과 serving 기본값을 유지한다.
이 실험에서 JEVal restricted를 추가로 열어보지는 않았다.

[작은 결과 기록](other-methods-results.json)에 지표·checksum·분할 감사·검증을 보존했다.
`runs/inputs/search-v2/`에는 모든 예측, calibration prior, 후보 고정 시점과 confirmation ID가 남는다.
이번 confirmation ID도 이제 관찰한 평가 문항으로 취급해야 한다.
